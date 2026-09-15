"""Observed longitudinal surface patches, not blanket removal of planar objects."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from scipy.spatial import cKDTree
from .segmentation import level_rotation


class TunnelBackground:
    def __init__(self, points: np.ndarray, geometry, config: dict):
        self.config = config["background"]
        cfg = self.config
        self.rotation = level_rotation(geometry.plane)
        self.patches = []
        self.input_points = len(points)
        leveled = points @ self.rotation.T
        _, _, _, supported, overlap = geometry.classify(points)
        # Never learn an obstruction inside supported vehicle clearance as lining.
        bed, _ = geometry.ground(points)
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
        neighbors = self.normal_tree.query_ball_point(sample, cfg["normal_radius_m"], return_length=True)
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
                plane, indices = cloud.segment_plane(cfg["fit_distance_m"], 3, cfg["ransac_iterations"], probability=cfg["ransac_probability"])
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
                for strip in np.unique(strip_ids):
                    q = support[strip_ids == strip]
                    if len(q) < cfg["min_strip_support"] or np.ptp(q[:, transverse]) < cfg["min_strip_span_m"]:
                        continue
                    strips.append((float(q[:, 0].min()), float(q[:, 0].max()),
                                   float(q[:, transverse].min()), float(q[:, transverse].max())))
                if strips:
                    self.patches.append((plane, transverse, strips, kind))

    def mask(self, points: np.ndarray, protected: np.ndarray) -> np.ndarray:
        cfg = self.config
        leveled = points @ self.rotation.T
        background = np.zeros(len(points), dtype=bool)
        if not self.patches:
            return background
        distances, nearest = self.normal_tree.query(leveled)
        reliable = (distances <= cfg["normal_radius_m"]) & self.normal_reliable[nearest]
        for plane, transverse, strips, _ in self.patches:
            near = (np.abs(leveled @ plane[:3] + plane[3]) <= cfg["remove_distance_m"]) & ~protected & ~background
            # A panel face can approach the wall without becoming part of it.
            # Preserve locally well-supported normals that disagree with lining.
            near &= ~(reliable & (np.abs(self.normals[nearest] @ plane[:3]) < cfg["normal_alignment_cos"]))
            sample_distance = np.abs(self.sample @ plane[:3] + plane[3])
            protrusion = (self.normal_reliable & (sample_distance >= cfg["protrusion_depth_m"])
                          & (sample_distance <= cfg["protection_radius_m"])
                          & (np.abs(self.normals @ plane[:3]) < cfg["normal_alignment_cos"]))
            near_ids = np.flatnonzero(near)
            if protrusion.any() and len(near_ids):
                # Keep attachment edges near a supported protruding face; otherwise
                # its near-wall column is amputated by the surface-distance band.
                distance, _ = cKDTree(self.sample[protrusion]).query(leveled[near_ids])
                near[near_ids[distance <= cfg["protection_radius_m"]]] = False
            ids = np.flatnonzero(near)
            if not len(ids):
                continue
            q = leveled[ids]
            inside = np.zeros(len(ids), dtype=bool)
            for lo, hi, bottom, top in strips:
                inside |= ((q[:, 0] >= lo - cfg["support_margin_m"]) & (q[:, 0] <= hi + cfg["support_margin_m"])
                           & (q[:, transverse] >= bottom - cfg["support_margin_m"])
                           & (q[:, transverse] <= top + cfg["support_margin_m"]))
            background[ids[inside]] = True
        return background

    def describe(self):
        return {"method": "open3d_supported_longitudinal_planes", "patches": len(self.patches),
                "wall_patches": sum(p[3] == "wall" for p in self.patches),
                "ceiling_patches": sum(p[3] == "ceiling" for p in self.patches),
                "fit_input_points": self.input_points}
