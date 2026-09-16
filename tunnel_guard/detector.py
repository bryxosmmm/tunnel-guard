"""Class-agnostic occupancy detection with published KISS-ICP motion compensation."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import TrackGeometry, voxel_representatives
from .segmentation import density_labels, published_labels


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    rotation = np.asarray(config["sensor_rotation"], dtype=float)
    translation = np.asarray(config["sensor_translation"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0)):
        raise ValueError("sensor_rotation must be a proper orthonormal 3x3 rotation")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("sensor_translation must contain three finite metres")
    envelope = np.asarray(config["envelope_segments_m"])
    if (envelope.ndim != 2 or envelope.shape[1] != 4 or len(envelope) < 2
            or not np.isfinite(envelope).all() or np.any(envelope[:, 2:] <= 0)
            or np.any(envelope[:, 1] <= envelope[:, 0])
            or not np.allclose(envelope[1:, 0], envelope[:-1, 1])):
        raise ValueError("envelope segments must be contiguous increasing height intervals with positive widths")
    if not 1 <= config["confirmation_hits"] <= config["confirmation_window"]:
        raise ValueError("confirmation_hits must fit confirmation_window")
    for name in ("geometry_voxel_m", "cluster_voxel_m", "ground_segment_m", "rail_bin_m",
                 "density_radius_m", "density_vertical_scale", "density_reference_range_m",
                 "odometry_voxel_m", "frame_max_gap_s", "track_max_gap_s"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("ground_extension_max_offset_m",
                 "ground_extension_window_growth", "ground_extension_max_window_m",
                 "rail_extension_max_offset_m", "rail_extension_window_growth",
                 "rail_extension_max_window_m", "path_sigma_floor_m"):
        if name in config and (not np.isfinite(config[name]) or config[name] <= 0):
            raise ValueError(f"{name} must be positive and finite")
    if "rail_head_width_m" in config and (not np.isfinite(config["rail_head_width_m"])
                                          or not 0 < config["rail_head_width_m"] < 0.5 * config["rail_gauge_m"]):
        raise ValueError("rail_head_width_m must be positive and less than half the standard gauge")
    for name in ("ground_extension_model_anchors", "ground_extension_misses_max",
                 "ground_extension_min_points", "rail_model_anchors", "rail_extension_min_points",
                 "rail_extension_misses_max"):
        if name in config and (isinstance(config[name], bool) or not isinstance(config[name], int)
                               or config[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    band = config.get("rail_head_band_m")
    if band is not None and not (isinstance(band, list) and len(band) == 2 and band[0] < band[1]):
        raise ValueError("rail_head_band_m must be [low, high] with low < high")
    bounds = config.get("path_growth_bounds_m_per_m")
    if bounds is not None and not (isinstance(bounds, list) and len(bounds) == 2 and 0 < bounds[0] <= bounds[1]):
        raise ValueError("path_growth_bounds_m_per_m must be [min, max] with 0 < min <= max")
    if "rail_support_symmetry_min" in config and not 0 < config["rail_support_symmetry_min"] <= 1:
        raise ValueError("rail_support_symmetry_min must be in (0, 1]")
    if not 0 <= config["min_range_m"] < config["max_range_m"]:
        raise ValueError("Invalid sensor range bounds")
    if not isinstance(config.get("deskew_enabled", False), bool):
        raise ValueError("deskew_enabled must be boolean")
    return config


def cluster_candidates(points: np.ndarray, geometry: TrackGeometry, config: dict) -> list[dict]:
    _, context, _, _, _ = geometry.classify(points)
    cloud = voxel_representatives(points[context], config["cluster_voxel_m"])
    if not len(cloud):
        return []
    core, _, heights, observed, nominal_overlap = geometry.classify(cloud, remove_background=False)
    method = config["segmentation_method"]
    if method == "density":
        labels, density_core = density_labels(cloud, config)
    else:
        labels = published_labels(cloud, geometry.plane, config, method)
        density_core = labels >= 0
    objects = []
    for label in np.unique(labels):
        if label < 0:
            continue
        indices = np.flatnonzero(labels == label)
        inside = indices[core[indices]]
        if len(indices) < config["weak_min_voxels"]:
            continue
        q = cloud[indices]
        minimum, maximum = q.min(axis=0), q.max(axis=0)
        extent = maximum - minimum
        if extent.max() < config["cluster_min_extent_m"]:
            continue
        intersects = len(inside) >= config["weak_min_voxels"]
        unresolved = np.count_nonzero(~observed[indices] & nominal_overlap[indices]) >= config["weak_min_voxels"]
        dense_count = int(np.count_nonzero(density_core[indices]))
        if not intersects and not unresolved and dense_count == 0:
            continue
        instant = (dense_count >= config["immediate_min_voxels"]
                   and extent[2] >= config["immediate_min_height_m"])
        objects.append({"bbox_min": minimum.tolist(), "bbox_max": maximum.tolist(),
                        "center": ((minimum + maximum) / 2).tolist(), "extent_m": extent.tolist(),
                        "distance_m": float(np.min(q[:, 0])),
                        "path_relation": "intersecting" if intersects else ("unresolved" if unresolved else "adjacent"),
                        "support_voxels": len(indices), "density_core_voxels": dense_count,
                        "in_envelope_voxels": len(inside), "_support_points": q,
                        "height_above_bed_m": [float(heights[indices].min()), float(heights[indices].max())],
                        "immediate": bool(instant)})
    return sorted(objects, key=lambda o: o["distance_m"])


class Detector:
    def __init__(self, config: dict):
        self.config = config
        self.odometry = self._new_odometry()
        self.previous_source = None
        self.previous_pose = np.eye(4)
        self.last_timestamp = None
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.frame_number = 0
        self.display_points = np.empty((0, 3))
        self.display_support = {}
        self.motion_translation_covariance = np.eye(3) * config["tracking_pose_sigma_m"]**2

    def _new_odometry(self):
        cfg = KISSConfig()
        cfg.data.min_range = self.config["min_range_m"]
        cfg.data.max_range = self.config["max_range_m"]
        cfg.data.deskew = self.config.get("deskew_enabled", False)
        cfg.mapping.voxel_size = self.config["odometry_voxel_m"]
        cfg.registration.max_num_iterations = self.config["odometry_max_iterations"]
        cfg.registration.max_num_threads = self.config["odometry_threads"]
        return KissICP(cfg)

    def _motion(self, points: np.ndarray, point_times: np.ndarray):
        frame, source = self.odometry.register_frame(points, point_times)
        pose = self.odometry.last_pose.copy()
        quality = {"valid": False, "reason": "first_frame",
                   "deskew_timestamps": bool(len(point_times)) and self.config.get("deskew_enabled", False),
                   "overlap": None, "median_residual_m": None}
        self.motion_translation_covariance = np.eye(3) * self.config["tracking_pose_sigma_m"]**2
        if len(source) >= 12:
            sample = source[::max(1, len(source) // 1024)][:1024]
            _, neighbors = cKDTree(source).query(sample, k=12)
            neighborhoods = source[neighbors]
            centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
            eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered) / 12)
            planar = eigenvalues[:, 0] < 0.2 * np.maximum(eigenvalues[:, 1], 1e-9)
            if np.count_nonzero(planar) >= 20:
                normals = eigenvectors[planar, :, 0]
                information = normals.T @ normals / len(normals)
                strengths, directions = np.linalg.eigh(information)
                weak = strengths < self.config["odometry_min_normal_eigenvalue"]
                inflation = self.config["tracking_pose_sigma_m"]**2 / np.maximum(strengths, .01)
                local_covariance = (directions * inflation) @ directions.T
                self.motion_translation_covariance = pose[:3, :3] @ local_covariance @ pose[:3, :3].T
                quality.update(translation_observability_eigenvalues=strengths.tolist(),
                               weak_translation_axes=int(weak.sum()),
                               translation_covariance_heuristic_m2=self.motion_translation_covariance.tolist())
            else:
                quality.update(translation_observability_eigenvalues=None, weak_translation_axes=3)
        if self.previous_source is not None and len(source) and len(self.previous_source):
            current_to_previous = np.linalg.inv(self.previous_pose) @ pose
            aligned = source @ current_to_previous[:3, :3].T + current_to_previous[:3, 3]
            distances, _ = cKDTree(self.previous_source).query(aligned, workers=1)
            median = float(np.median(distances))
            overlap = float(np.mean(distances < self.config["odometry_max_residual_m"] * 2))
            valid = (np.isfinite(pose).all() and median <= self.config["odometry_max_residual_m"]
                     and overlap >= self.config["odometry_min_overlap"])
            quality |= {"valid": bool(valid), "reason": "registered" if valid else "registration_rejected",
                        "overlap": overlap, "median_residual_m": median}
        self.previous_source, self.previous_pose = source.copy(), pose
        return frame, pose, quality

    def _associate(self, objects: list[dict], pose: np.ndarray, stamp: float, motion_valid: bool):
        cfg = self.config
        if not motion_valid:
            self.tracks.clear()
        self.tracks = {key: t for key, t in self.tracks.items() if stamp - t["stamp"] <= cfg["track_max_gap_s"]}
        ids = list(self.tracks)
        world = np.array([o["center"] for o in objects], dtype=float).reshape(-1, 3)
        world = world @ pose[:3, :3].T + pose[:3, 3]
        measurement_cov = np.eye(3) * cfg["tracking_position_sigma_m"]**2 + self.motion_translation_covariance
        predicted = {}
        for key in ids:
            t = self.tracks[key]
            dt = stamp - t["stamp"]
            transition = np.eye(6)
            transition[:3, 3:] = np.eye(3) * dt
            noise_map = np.vstack((np.eye(3) * dt**2 / 2, np.eye(3) * dt))
            covariance = transition @ t["covariance"] @ transition.T + noise_map @ noise_map.T * cfg["tracking_acceleration_sigma_mps2"]**2
            predicted[key] = (transition @ t["state"], covariance)
        matched = {}
        if ids and len(world):
            cost = np.full((len(ids), len(world)), 1e6)
            extents = np.asarray([o["extent_m"] for o in objects]) + .1
            for row, key in enumerate(ids):
                state, covariance = predicted[key]
                inverse = np.linalg.inv(covariance[:3, :3] + measurement_cov)
                residual = world - state[:3]
                mahalanobis = np.einsum("ni,ij,nj->n", residual, inverse, residual)
                shape = np.linalg.norm(np.log(extents / (self.tracks[key]["extent"] + .1)), axis=1)
                allowed = (mahalanobis <= cfg["tracking_mahalanobis_gate"]) & (shape <= cfg["tracking_extent_log_gate"])
                cost[row, allowed] = mahalanobis[allowed] + shape[allowed]
            # Add unmatched assignments: an impossible pair cannot steal a valid match.
            padded = np.column_stack((cost, np.full((len(ids), len(ids)), cfg["tracking_mahalanobis_gate"] + cfg["tracking_extent_log_gate"] + 1)))
            rows, columns = linear_sum_assignment(padded)
            for row, column in zip(rows, columns):
                if column < len(world) and cost[row, column] < 1e6:
                    matched[int(column)] = ids[row]
        for index, obj in enumerate(objects):
            key = matched.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
                state = np.concatenate((world[index], np.zeros(3)))
                covariance = np.diag([cfg["tracking_position_sigma_m"]**2] * 3 + [1.] * 3)
                self.tracks[key] = {"history": deque(maxlen=cfg["confirmation_window"]), "first_stamp": stamp,
                                    "evidence": deque(), "state": state, "covariance": covariance}
            else:
                state, covariance = predicted[key]
                gain = covariance[:, :3] @ np.linalg.inv(covariance[:3, :3] + measurement_cov)
                state += gain @ (world[index] - state[:3])
                observation = np.zeros((3, 6))
                observation[:, :3] = np.eye(3)
                factor = np.eye(6) - gain @ observation
                covariance = factor @ covariance @ factor.T + gain @ measurement_cov @ gain.T
            track = self.tracks[key]
            support = obj.pop("_support_points")
            self.display_support[key] = support
            support_world = support @ pose[:3, :3].T + pose[:3, 3]
            # Track-local spatial evidence compensates estimated object translation.
            # Bounds remain from this frame; past points never fabricate present shape.
            track["evidence"].append((stamp, support_world - world[index]))
            while track["evidence"] and stamp - track["evidence"][0][0] > cfg["evidence_window_s"]:
                track["evidence"].popleft()
            evidence = voxel_representatives(np.vstack([e[1] for e in track["evidence"]]), cfg["cluster_voxel_m"])
            if not track["history"] or track["history"][-1] != self.frame_number:
                track["history"].append(self.frame_number)
            hits = sum(f > self.frame_number - cfg["confirmation_window"] for f in track["history"])
            confirmed = obj["immediate"] or (hits >= cfg["confirmation_hits"] and len(evidence) >= cfg["evidence_min_points"])
            track.update(state=state, covariance=covariance, stamp=stamp, extent=np.asarray(obj["extent_m"]))
            obj.update(track_id=key, hits=hits, confirmed=bool(confirmed), accumulated_support_voxels=len(evidence),
                       last_observed_s=stamp,
                       evidence_timestamps_s=[e[0] for e in track["evidence"]],
                       covariance_kind="heuristic_not_calibrated",
                       velocity_world_mps=state[3:].tolist(), position_covariance_m2=covariance[:3, :3].tolist(),
                       confirmation="immediate_geometry" if obj["immediate"] else ("temporal_evidence" if confirmed else "pending"),
                       track_age_s=stamp - track["first_stamp"])

    def process(self, points: np.ndarray, timestamp_s: float, point_times: np.ndarray | None = None) -> dict:
        started = time.perf_counter()
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(timestamp_s):
            raise ValueError("Expected Nx3 points and a finite timestamp")
        if self.last_timestamp is not None and timestamp_s <= self.last_timestamp:
            raise ValueError("Scan timestamps must be strictly increasing; reset detector between bags")
        dt = None if self.last_timestamp is None else timestamp_s - self.last_timestamp
        reset = dt is not None and dt > self.config["frame_max_gap_s"]
        point_times = np.empty(0) if point_times is None else np.asarray(point_times, dtype=float)
        if point_times.ndim != 1 or len(point_times) not in (0, len(points)):
            raise ValueError("point_times must be empty or match the number of points")
        if len(point_times) and (not np.isfinite(point_times).all() or point_times.min() < 0 or point_times.max() > 1):
            raise ValueError("point_times must be finite and normalized to [0, 1]")
        if reset:
            self.odometry = self._new_odometry()
            self.previous_source = None
            self.previous_pose = np.eye(4)
            self.tracks.clear()
        self.last_timestamp = timestamp_s
        self.frame_number += 1
        self.display_support = {}
        finite = np.isfinite(points).all(axis=1)
        radii = np.linalg.norm(points, axis=1)
        keep = finite & (radii >= self.config["min_range_m"]) & (radii <= self.config["max_range_m"])
        points = points[keep]
        self.display_points = points
        if len(point_times):
            point_times = point_times[keep]
            if len(point_times) and np.ptp(point_times) == 0:
                point_times = np.empty(0)
        result = {"timestamp_s": timestamp_s, "status": "unknown", "objects": [], "nearest_obstacle_m": None,
                  "input_valid_points": len(points), "gap_reset": reset,
                  "coordinate_frame": "tunnel_guard_local",
                  "distance_method": "minimum_forward_x_of_current_observed_cluster",
                  "distance_origin": "configured_processing_frame_origin",
                  "distance_along_path_m": None,
                  "health": "unavailable", "health_reasons": ["insufficient_returns"],
                  "envelope_calibration": self.config["envelope_calibration"]}
        if len(points) < self.config["ground_min_support"]:
            self.tracks.clear()
            self.previous_source = None
            self.odometry = self._new_odometry()
            return result | {"reason": "insufficient_returns", "processing_s": time.perf_counter() - started}
        motion_started = time.perf_counter()
        frame, pose, motion = self._motion(points, point_times)
        self.display_points = frame
        motion_s = time.perf_counter() - motion_started
        crop = ((frame[:, 0] >= self.config["min_forward_m"])
                & (np.abs(frame[:, 1]) < self.config["context_half_width_m"]))
        reduced = voxel_representatives(frame[crop], self.config["geometry_voxel_m"])
        geometry = TrackGeometry(reduced, self.config)
        objects = cluster_candidates(reduced, geometry, self.config) if geometry.valid else []
        self._associate(objects, pose, timestamp_s, motion["valid"])
        hazards = [o for o in objects if o["path_relation"] in ("intersecting", "unresolved")]
        confirmed = [o for o in hazards if o["confirmed"]]
        certain = [o for o in confirmed if o["path_relation"] == "intersecting"]
        status = ("obstacle" if certain else ("unresolved_obstacle" if confirmed else
                  ("candidate" if hazards else ("no_obstacle_observed" if geometry.valid else "unknown"))))
        bins = []
        _, _, _, observed, _ = geometry.classify(reduced, remove_background=False)
        for lo, hi in zip(self.config["range_bins_m"][:-1], self.config["range_bins_m"][1:]):
            mask = (reduced[:, 0] >= lo) & (reduced[:, 0] < hi)
            raw_mask = (frame[:, 0] >= lo) & (frame[:, 0] < hi) & crop
            bins.append({"range_m": [lo, hi], "returns": int(mask.sum()),
                         "returns_before_geometry_voxel": int(raw_mask.sum()),
                         "geometry_supported_returns": int(np.count_nonzero(mask & observed))})
        health_reasons = []
        if not geometry.valid:
            health_reasons.append(geometry.reason)
        if not motion["valid"]:
            health_reasons.append(motion["reason"])
        if motion.get("weak_translation_axes", 3):
            health_reasons.append("weak_or_unmeasured_translation_observability")
        if not self.config.get("sensor_profile_verified", False):
            health_reasons.append("unverified_sensor_profile_and_extrinsics")
        if not self.config.get("deskew_enabled", False):
            health_reasons.append("deskew_disabled_unverified_timing")
        elif not len(point_times):
            health_reasons.append("deskew_timestamps_unavailable")
        return result | {"status": status, "reason": geometry.reason, "objects": objects,
                         "health": "unavailable" if not geometry.valid else ("degraded" if health_reasons else "normal"),
                         "health_reasons": health_reasons,
                         "nearest_obstacle_m": min((o["distance_m"] for o in confirmed), default=None),
                         "geometry": geometry.describe(), "motion": motion, "pose": pose.tolist(),
                         "range_observability": bins, "geometry_points": len(reduced),
                         "processing_s": time.perf_counter() - started, "motion_s": motion_s}
