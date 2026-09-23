"""Observed longitudinal surface patches, not blanket removal of planar objects."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from threadpoolctl import ThreadpoolController

from scipy.spatial import cKDTree
from . import accelerator
from .geometry import query_workers
from .segmentation import level_rotation


_THREAD_POOLS = ThreadpoolController()


class TunnelBackground:
    def __init__(self, points: np.ndarray, geometry, config: dict):
        self.config = config["background"]
        cfg = self.config
        self.rotation = level_rotation(geometry.plane)
        self.patches = []
        self.input_points = len(points)
        self.query_workers = config.get("query_workers", -1)
        self.native = accelerator.native()
        leveled = points @ self.rotation.T
        # Never learn an obstruction inside supported vehicle clearance as lining.
        # The same bed fit is reused by classification instead of refitted.
        bed, ground_uncertainty = geometry.ground(points)
        _, _, _, supported, overlap = geometry.classify(points, ground=(bed, ground_uncertainty))
        above_rail = points[:, 2] - bed - geometry.rail_head_height_m
        eligible = leveled[~(supported & overlap) & (above_rail >= cfg["min_seed_height_m"])]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(eligible)).voxel_down_sample(cfg["voxel_m"])
        sample = np.asarray(pcd.points)
        self.sample = sample
        if len(sample) < cfg["min_support"]:
            return
        self.normal_tree = cKDTree(sample)
        neighbors, _, eigenvalues, self.normals = accelerator.normal_statistics(
            self.sample, cfg["normal_radius_m"], cfg["normal_max_neighbors"],
            cfg["normal_min_neighbors"], self.native)
        self.normal_reliable = ((neighbors >= cfg["normal_min_neighbors"]) & (eigenvalues[:, 1] > cfg["normal_min_variance_m2"])
                                & (eigenvalues[:, 0] <= cfg["normal_planarity_ratio"] * eigenvalues[:, 1]))
        o3d.utility.random.seed(config["seed"])
        # Global proposals serve straight tunnels; overlapping windows let curved
        # surfaces be represented by independently supported local planar pieces.
        domains = [(float(leveled[:, 0].min()), float(leveled[:, 0].max()))] if len(leveled) else []
        if len(leveled):
            domains += [(float(x), float(x + cfg["window_m"])) for x in
                        np.arange(leveled[:, 0].min(), leveled[:, 0].max(), cfg["window_m"] / 2)]
        scratch_rows = np.empty_like(sample)
        if domains:
            # Window membership for every domain in one call; the slices are the
            # same rows the per-domain mask selected.
            window_ids, offsets = accelerator.window_indices(sample, np.asarray(domains, dtype=float), self.native)
            windowed = sample[window_ids]
        for domain, (lo, hi) in enumerate(domains):
            remainder = windowed[offsets[domain]:offsets[domain + 1]]
            for _ in range(cfg["planes_per_window"]):
                if len(remainder) < cfg["min_support"]:
                    break
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remainder))
                # Open3D's parallel adaptive stopping depends on scheduling,
                # even with a fixed seed. Serialize proposals, not all geometry.
                with _THREAD_POOLS.limit(limits=cfg.get("ransac_threads", 1), user_api="openmp"):
                    plane, indices = cloud.segment_plane(cfg["fit_distance_m"], 3,
                        cfg["ransac_iterations"], probability=cfg["ransac_probability"])
                indices = np.asarray(indices, dtype=int)
                if len(indices) < cfg["min_support"]:
                    break
                support = remainder[indices]
                remainder = accelerator.remove_rows(remainder, indices, scratch_rows, self.native)
                plane = np.asarray(plane)
                normal = plane[:3]
                if abs(normal[0]) > cfg["max_longitudinal_normal"]:
                    continue  # Cross-track panels and end walls are not lining.
                if abs(normal[1]) >= cfg["min_cross_normal"]:
                    kind, transverse = "wall", 2
                elif abs(normal[2]) >= cfg["min_cross_normal"]:
                    kind, transverse = "ceiling", 1
                else:
                    continue
                if np.ptp(support[:, 0]) < cfg["min_longitudinal_span_m"] or np.ptp(support[:, transverse]) < cfg["min_transverse_span_m"]:
                    continue
                # Require observed support in each longitudinal strip. A fitted
                # plane cannot erase geometry across an unobserved gap or range.
                strips = accelerator.support_strips(support, transverse, cfg["strip_m"],
                                                    cfg["min_strip_support"], cfg["min_strip_span_m"],
                                                    self.native)
                if len(strips):
                    self.patches.append((plane, transverse, strips, kind))

    def mask(self, points: np.ndarray, protected: np.ndarray) -> np.ndarray:
        """Points claimed by observed longitudinal surfaces.

        Candidate windowing, the distance band and the per-patch bookkeeping are
        computed once by the native kernel; the single neighbour query supplies
        its normal evidence.
        """
        cfg = self.config
        leveled = points @ self.rotation.T
        background = np.zeros(len(points), dtype=bool)
        if not self.patches:
            return background
        planes = np.asarray([patch[0] for patch in self.patches], dtype=float)
        transverse = np.asarray([patch[1] for patch in self.patches], dtype=np.int64)
        bounds = np.asarray([[min(strip[0] for strip in patch[2]) - cfg["support_margin_m"],
                              max(strip[1] for strip in patch[2]) + cfg["support_margin_m"]]
                             for patch in self.patches], dtype=float)
        strip_boxes, strip_offsets = [], np.zeros(len(self.patches) + 1, dtype=np.int64)
        for index, patch in enumerate(self.patches):
            strip_boxes.extend(patch[2])
            strip_offsets[index + 1] = len(strip_boxes)
        strip_boxes = np.asarray(strip_boxes, dtype=float).reshape(-1, 4)
        union, candidates, candidate_offsets = accelerator.mask_candidates(
            leveled, planes, bounds, cfg["remove_distance_m"], self.native)
        distances, nearest = self.normal_tree.query(
            leveled[union], workers=query_workers(len(union), self.query_workers))
        reliable = (distances <= cfg["normal_radius_m"]) & self.normal_reliable[nearest]
        aligned = self.normals[nearest]
        slot = np.full(len(points), -1, dtype=np.int64)
        slot[union] = np.arange(len(union))
        accelerator.mask_apply(leveled, protected, background, planes, transverse, strip_boxes,
                               strip_offsets, candidates, candidate_offsets,
                               np.ascontiguousarray(reliable[slot[candidates]]),
                               np.ascontiguousarray(aligned[slot[candidates]]),
                               self.sample, self.normals, self.normal_reliable,
                               cfg["support_margin_m"], cfg["protrusion_depth_m"],
                               cfg["protection_radius_m"], cfg["normal_alignment_cos"], self.native)
        return background

    def describe(self):
        return {"method": "open3d_supported_longitudinal_planes", "patches": len(self.patches),
                "wall_patches": sum(p[3] == "wall" for p in self.patches),
                "ceiling_patches": sum(p[3] == "ceiling" for p in self.patches),
                "fit_input_points": self.input_points}
