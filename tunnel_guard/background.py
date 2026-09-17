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
        self.native = accelerator.native(config)
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
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=cfg["normal_radius_m"], max_nn=cfg["normal_max_neighbors"])
        pcd.estimate_normals(search)
        pcd.estimate_covariances(search)
        self.normal_tree = cKDTree(sample)
        self.normals = np.asarray(pcd.normals)
        eigenvalues = np.linalg.eigvalsh(np.asarray(pcd.covariances))
        neighbors = self.normal_tree.query_ball_point(
            sample, cfg["normal_radius_m"], return_length=True,
            workers=query_workers(len(sample), self.query_workers))
        self.normal_reliable = ((neighbors >= cfg["normal_min_neighbors"]) & (eigenvalues[:, 1] > cfg["normal_min_variance_m2"])
                                & (eigenvalues[:, 0] <= cfg["normal_planarity_ratio"] * eigenvalues[:, 1]))
        o3d.utility.random.seed(config["seed"])
        # Global proposals serve straight tunnels; overlapping windows let curved
        # surfaces be represented by independently supported local planar pieces.
        domains = [(float(leveled[:, 0].min()), float(leveled[:, 0].max()))] if len(leveled) else []
        if len(leveled):
            domains += [(float(x), float(x + cfg["window_m"])) for x in
                        np.arange(leveled[:, 0].min(), leveled[:, 0].max(), cfg["window_m"] / 2)]
        for lo, hi in domains:
            remainder = sample[(sample[:, 0] >= lo) & (sample[:, 0] <= hi)]
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
                retained = np.ones(len(remainder), dtype=bool)
                retained[indices] = False
                remainder = remainder[retained]
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
                strips = []
                strip_ids = np.floor(support[:, 0] / cfg["strip_m"]).astype(int)
                _, inverse, counts = np.unique(strip_ids, return_inverse=True, return_counts=True)
                order = np.argsort(inverse, kind="stable")
                bounds = np.concatenate(([0], np.cumsum(counts)))
                strip_x = support[order, 0]
                strip_t = support[order, transverse]
                for index in range(len(counts)):
                    if counts[index] < cfg["min_strip_support"]:
                        continue
                    start, stop = int(bounds[index]), int(bounds[index + 1])
                    span = float(strip_t[start:stop].max() - strip_t[start:stop].min())
                    if span < cfg["min_strip_span_m"]:
                        continue
                    strips.append((float(strip_x[start:stop].min()), float(strip_x[start:stop].max()),
                                   float(strip_t[start:stop].min()), float(strip_t[start:stop].max())))
                if strips:
                    self.patches.append((plane, transverse, strips, kind))

    def mask(self, points: np.ndarray, protected: np.ndarray) -> np.ndarray:
        cfg = self.config
        leveled = points @ self.rotation.T
        background = np.zeros(len(points), dtype=bool)
        if not self.patches:
            return background
        # A patch can only claim points inside its observed longitudinal strips,
        # and only points within remove_distance of its plane can be candidates.
        # Neither test depends on the running mask, so every patch's candidates
        # are collected first and answered by one neighbour query.
        prepared = []
        for plane, transverse, strips, _ in self.patches:
            lo = min(strip[0] for strip in strips) - cfg["support_margin_m"]
            hi = max(strip[1] for strip in strips) + cfg["support_margin_m"]
            candidates = accelerator.patch_candidates(leveled, plane, lo, hi,
                                                      cfg["remove_distance_m"], self.native)
            if len(candidates):
                prepared.append((plane, transverse, strips, candidates))
        if not prepared:
            return background
        union = np.unique(np.concatenate([candidates for _, _, _, candidates in prepared]))
        distances, nearest = self.normal_tree.query(
            leveled[union], workers=query_workers(len(union), self.query_workers))
        reliable = (distances <= cfg["normal_radius_m"]) & self.normal_reliable[nearest]
        aligned = self.normals[nearest]
        slot = np.full(len(points), -1, dtype=np.int64)
        slot[union] = np.arange(len(union))
        for plane, transverse, strips, candidates in prepared:
            index = slot[candidates]
            # A panel face can approach the wall without becoming part of it.
            # Preserve locally well-supported normals that disagree with lining.
            near = (~protected[candidates] & ~background[candidates]
                    & ~(reliable[index] & (np.abs(aligned[index] @ plane[:3]) < cfg["normal_alignment_cos"])))
            near_ids = candidates[near]
            if len(near_ids):
                protrusion = accelerator.protrusion_ids(
                    self.sample, self.normals, self.normal_reliable, plane,
                    cfg["protrusion_depth_m"], cfg["protection_radius_m"], cfg["normal_alignment_cos"],
                    self.native)
                if len(protrusion):
                    # Keep attachment edges near a supported protruding face; otherwise
                    # its near-wall column is amputated by the surface-distance band.
                    keep = accelerator.keep_outside_radius(
                        leveled[near_ids], self.sample[protrusion], cfg["protection_radius_m"], self.native)
                    near_ids = near_ids[keep]
            if not len(near_ids):
                continue
            background[accelerator.strip_inside(leveled, near_ids, strips, cfg["support_margin_m"],
                                                transverse, self.native)] = True
        return background

    def describe(self):
        return {"method": "open3d_supported_longitudinal_planes", "patches": len(self.patches),
                "wall_patches": sum(p[3] == "wall" for p in self.patches),
                "ceiling_patches": sum(p[3] == "ceiling" for p in self.patches),
                "fit_input_points": self.input_points}
