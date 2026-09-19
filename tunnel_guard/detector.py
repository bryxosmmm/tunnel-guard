"""Class-agnostic occupancy detection with published KISS-ICP motion compensation."""
from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import time

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from . import accelerator
from .geometry import TrackGeometry, voxel_representatives
from .mounting import observe_mounting, validate_mounting_config
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
    if (envelope.ndim != 2 or envelope.shape[1] != 4 or len(envelope) < 1
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
    if not 0 <= config["min_range_m"] < config["max_range_m"]:
        raise ValueError("Invalid sensor range bounds")
    if not isinstance(config.get("deskew_enabled", False), bool):
        raise ValueError("deskew_enabled must be boolean")
    if config.get("voxel_backend", "numpy") not in ("numpy", "cpp"):
        raise ValueError("voxel_backend must be numpy or cpp")
    if config.get("voxel_backend") == "cpp":
        from . import _native  # Fail explicitly when the selected accelerator is unavailable.
    ransac_threads = config.get("background", {}).get("ransac_threads", 1)
    if type(ransac_threads) is not int or ransac_threads < 1:
        raise ValueError("background.ransac_threads must be a positive integer")
    query_workers = config.get("query_workers", -1)
    if type(query_workers) is not int or (query_workers < 1 and query_workers != -1):
        raise ValueError("query_workers must be -1 (all cores) or a positive integer")
    if not isinstance(config.get("native_kernels", False), bool):
        raise ValueError("native_kernels must be boolean")
    if config.get("native_kernels", False):
        from . import _native  # Fail explicitly when the selected kernels are unavailable.
    if config.get("obstacle_distance_mode", "cluster_min_x") not in ("cluster_min_x", "envelope_support_min_x"):
        raise ValueError("Unknown obstacle_distance_mode")
    if config.get("rail_center_estimator", "histogram") not in ("histogram", "paired_line"):
        raise ValueError("Unknown rail_center_estimator")
    if config.get("rail_initial_heading", "zero") not in ("zero", "fitted"):
        raise ValueError("Unknown rail_initial_heading")
    if config.get("rail_pair_continuity", "window") not in ("window", "relocated"):
        raise ValueError("Unknown rail_pair_continuity")
    if config.get("rail_frame_mode", "bed") not in ("bed", "local_3d"):
        raise ValueError("Unknown rail_frame_mode")
    if config.get("rail_anchor_support", "window") not in ("window", "bracketed", "measured"):
        raise ValueError("Unknown rail_anchor_support")
    validate_mounting_config(config)
    return config


def voxel_unique_at(config: dict, reductive_size: float | None) -> bool:
    """Skip reduction only on the identical grid, preserving key order.

    A coarser grid may imply uniqueness, but its lexicographic order need not
    match the finer grid. That order affects cluster labels and track IDs.
    """
    if reductive_size is None:
        return False
    size = config["cluster_voxel_m"]
    if not (size > 0 and reductive_size > 0):
        return False
    return size == reductive_size


def cluster_candidates(points: np.ndarray, geometry: TrackGeometry, config: dict,
                       diagnostics: dict | None = None, arrays: dict | None = None,
                       *, reduced_on_grid_m: float | None = None,
                       classification_out: list | None = None) -> list[dict]:
    classification = geometry.classify(points)
    if classification_out is not None:
        # The caller needs the same classification for its range bins; the two
        # are identical because the inputs are, and background removal only
        # touches `context`.
        classification_out.append(classification)
    _, context, _, _, _ = classification
    if voxel_unique_at(config, reduced_on_grid_m):
        # The context cloud is already one point per cluster voxel in key order,
        # so the reduction below would only reproduce the same rows in the same
        # order; filtering keeps it identical without the sort.
        cloud = points[context]
    else:
        cloud = voxel_representatives(points[context], config["cluster_voxel_m"], config.get("voxel_backend", "numpy"))
    if diagnostics is not None:
        diagnostics.update(state="ran", context_points=int(context.sum()), cluster_points=len(cloud), rejected={})
    if arrays is not None:
        before = geometry.classify(points, remove_background=False)[1]
        arrays.update(context_before_background=points[before], context_after_background=points[context],
                      cluster_points=cloud)
    if not len(cloud):
        return []
    core, _, heights, observed, nominal_overlap, boundary = geometry.classify(
        cloud, remove_background=False, include_boundary=True)
    # Mixed evidence (one interior + one boundary return) must not disappear
    # merely because neither subset separately reaches weak_min_voxels.
    uncertain_support = core | (~observed & nominal_overlap) | boundary
    method = config["segmentation_method"]
    if method == "density":
        labels, density_core = density_labels(cloud, config)
    else:
        labels = published_labels(cloud, geometry.plane, config, method)
        density_core = labels >= 0
    objects = []
    rejected = Counter()
    components = []
    if arrays is not None:
        arrays.update(cluster_labels=labels, cluster_core=core, cluster_observed=observed,
                      cluster_nominal_overlap=nominal_overlap, cluster_boundary_uncertain=boundary)
    native = accelerator.native(config)
    if native is not None and len(cloud):
        # One native call groups the cloud by component and computes every
        # statistic; the dictionaries are assembled here, where they cost under a
        # millisecond, in the original field order.
        objects, rejected_counts, rows = accelerator.cluster_objects(
            cloud, labels, core, boundary, density_core, heights, uncertain_support, config, native)
        rejected.update(rejected_counts)
        if arrays is not None:
            components = rows
            diagnostics["components"] = components
        if diagnostics is not None:
            diagnostics.update(rejected=dict(rejected), accepted=len(objects),
                               noise_points=int(np.count_nonzero(labels < 0)))
        return sorted(objects, key=lambda o: o["cluster_nearest_x_m"])
    else:
        order = np.argsort(labels, kind="stable")
        # One stable grouping replaces a full labels==label scan per component.
        # Stable order preserves original point order within every cluster, hence
        # witness selection, box ties and temporal evidence ordering stay unchanged.
        sorted_labels = labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
        stops = np.r_[starts[1:], len(order)]
        for start, stop in zip(starts, stops):
            label = sorted_labels[start]
            if label < 0:
                continue
            indices = order[start:stop]
            inside = indices[core[indices]]
            if len(indices) < config["weak_min_voxels"]:
                rejected["below_weak_min_voxels"] += 1
                if arrays is not None:
                    components.append({"component_id": int(label), "reason": "below_weak_min_voxels", "points": len(indices)})
                continue
            q = cloud[indices]
            minimum, maximum = q.min(axis=0), q.max(axis=0)
            extent = maximum - minimum
            if extent.max() < config["cluster_min_extent_m"]:
                rejected["below_min_extent"] += 1
                if arrays is not None:
                    components.append({"component_id": int(label), "reason": "below_min_extent", "points": len(indices)})
                continue
            intersects = len(inside) >= config["weak_min_voxels"]
            unresolved = np.count_nonzero(uncertain_support[indices]) >= config["weak_min_voxels"]
            dense_count = int(np.count_nonzero(density_core[indices]))
            if not intersects and not unresolved and dense_count == 0:
                rejected["weak_without_envelope_support"] += 1
                if arrays is not None:
                    components.append({"component_id": int(label), "reason": "weak_without_envelope_support", "points": len(indices)})
                continue
            instant = (dense_count >= config["immediate_min_voxels"]
                       and extent[2] >= config["immediate_min_height_m"])
            interior_dense_count = int(np.count_nonzero(density_core[inside]))
            interior_height = float(np.ptp(cloud[inside, 2])) if len(inside) else 0.
            intersection_immediate = (interior_dense_count >= config["immediate_min_voxels"]
                                      and interior_height >= config["immediate_min_height_m"])
            distance_points = q
            distance_method = "cluster_min_x"
            if config.get("obstacle_distance_mode", "cluster_min_x") == "envelope_support_min_x":
                if intersects:
                    distance_points = cloud[inside]
                    distance_method = "supported_envelope_min_x"
                elif unresolved:
                    distance_points = q[uncertain_support[indices]]
                    distance_method = "unresolved_envelope_evidence_min_x"
            witness = distance_points[np.argmin(distance_points[:, 0])]
            objects.append({"bbox_min": minimum.tolist(), "bbox_max": maximum.tolist(),
                            "component_id": int(label),
                            "cluster_nearest_x_m": float(np.min(q[:, 0])),
                            "supported_envelope_nearest_x_m": float(np.min(cloud[inside, 0])) if len(inside) else None,
                            "unresolved_envelope_nearest_x_m": float(np.min(q[uncertain_support[indices], 0])) if unresolved else None,
                            "center": ((minimum + maximum) / 2).tolist(), "extent_m": extent.tolist(),
                            "distance_m": float(witness[0]), "distance_method": distance_method,
                            "distance_support_point": witness.tolist(), "distance_support_points": len(distance_points),
                            "path_relation": "intersecting" if intersects else ("unresolved" if unresolved else "adjacent"),
                            "support_voxels": len(indices), "density_core_voxels": dense_count,
                            "in_envelope_voxels": len(inside), "_support_points": q,
                            "boundary_uncertain_voxels": int(np.count_nonzero(boundary[indices])),
                            "path_relation_reason": ("inside_heuristic_path_and_ground_interval" if intersects else
                                ("envelope_boundary_uncertainty" if unresolved and np.any(boundary[indices]) else
                                 ("unsupported_nominal_envelope" if unresolved else "outside_envelope_evidence"))),
                            "height_above_bed_m": [float(heights[indices].min()), float(heights[indices].max())],
                            "immediate": bool(instant),
                            "interior_density_core_voxels": interior_dense_count,
                            "interior_height_span_m": interior_height,
                            "intersection_immediate": bool(intersection_immediate)})
            if arrays is not None:
                components.append({"component_id": int(label), "reason": "accepted", "points": len(indices)})
    if diagnostics is not None:
        diagnostics.update(rejected=dict(rejected), accepted=len(objects),
                           noise_points=int(np.count_nonzero(labels < 0)))
        if arrays is not None:
            diagnostics["components"] = components
    # Keep association order independent of the selected distance definition.
    return sorted(objects, key=lambda o: o["cluster_nearest_x_m"])

class Detector:
    def __init__(self, config: dict):
        self.native_kernels = accelerator.native(config)
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
        self.diagnostic_arrays = {}
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
        # Prediction and the association cost are the same per-track arithmetic as
        # before, batched over tracks: the Python loop around these small matrices
        # cost more than the matrices themselves. Each batched operation still
        # dispatches one BLAS/LAPACK call per matrix, so the values are those of the
        # unbatched form.
        predicted = {}
        if ids:
            states = np.stack([self.tracks[key]["state"] for key in ids])
            covariances = np.stack([self.tracks[key]["covariance"] for key in ids])
            gaps = np.asarray([stamp - self.tracks[key]["stamp"] for key in ids])
            identity = np.eye(6)
            shift = np.zeros((6, 6))
            shift[:3, 3:] = np.eye(3)
            transitions = identity[None] + shift[None] * gaps[:, None, None]
            # Built as the original expression and multiplied per matrix, so the
            # batched call dispatches the same BLAS routine on the same operands.
            noise_maps = np.zeros((len(ids), 6, 3))
            noise_maps[:, :3, :3] = np.eye(3)[None] * (gaps**2 / 2.0)[:, None, None]
            noise_maps[:, 3:, :3] = np.eye(3)[None] * gaps[:, None, None]
            noise = (noise_maps @ np.transpose(noise_maps, (0, 2, 1))) * cfg["tracking_acceleration_sigma_mps2"]**2
            advanced = transitions @ covariances @ np.transpose(transitions, (0, 2, 1)) + noise
            for row, key in enumerate(ids):
                # Kept per track: a matrix-vector product is a different BLAS
                # routine than the batched matrix-matrix form.
                predicted[key] = (transitions[row] @ states[row], advanced[row])
        matched = {}
        if ids and len(world):
            cost = np.full((len(ids), len(world)), 1e6)
            extents = np.asarray([o["extent_m"] for o in objects]) + .1
            track_extents = np.asarray([self.tracks[key]["extent"] for key in ids]) + .1
            positions = np.stack([predicted[key][0][:3] for key in ids])
            covariances = np.stack([predicted[key][1][:3, :3] for key in ids])
            inverse = np.linalg.inv(covariances + measurement_cov)
            residual = world[None, :, :] - positions[:, None, :]
            mahalanobis = np.einsum("tni,tij,tnj->tn", residual, inverse, residual)
            shape = np.linalg.norm(np.log(extents[None, :, :] / track_extents[:, None, :]), axis=2)
            allowed = (mahalanobis <= cfg["tracking_mahalanobis_gate"]) & (shape <= cfg["tracking_extent_log_gate"])
            cost[allowed] = mahalanobis[allowed] + shape[allowed]
            # Add unmatched assignments: an impossible pair cannot steal a valid match.
            padded = np.column_stack((cost, np.full((len(ids), len(ids)), cfg["tracking_mahalanobis_gate"] + cfg["tracking_extent_log_gate"] + 1)))
            rows, columns = linear_sum_assignment(padded)
            for row, column in zip(rows, columns):
                if column < len(world) and cost[row, column] < 1e6:
                    matched[int(column)] = ids[row]
        pending: list[dict] = []
        for index, obj in enumerate(objects):
            key = matched.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
                state = np.concatenate((world[index], np.zeros(3)))
                covariance = np.diag([cfg["tracking_position_sigma_m"]**2] * 3 + [1.] * 3)
                self.tracks[key] = {"history": deque(maxlen=cfg["confirmation_window"]), "first_stamp": stamp,
                                    "intersection_history": deque(maxlen=cfg["confirmation_window"]),
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
            if not track["history"] or track["history"][-1] != self.frame_number:
                track["history"].append(self.frame_number)
            hits = sum(f > self.frame_number - cfg["confirmation_window"] for f in track["history"])
            # Object persistence cannot confirm a new path intrusion. Count only
            # current-scan interior evidence, once per strictly increasing scan.
            interior_history = track["intersection_history"]
            if obj["path_relation"] == "intersecting" and (not interior_history or interior_history[-1][0] != self.frame_number):
                interior_history.append((self.frame_number, stamp))
            recent_interior = [(f, s) for f, s in interior_history
                               if f > self.frame_number - cfg["confirmation_window"]]
            track.update(state=state, covariance=covariance, stamp=stamp, extent=np.asarray(obj["extent_m"]))
            # Accumulated support is the one quantity that needs its own call per
            # track; the stacks are counted together after this loop, so the field
            # writes below keep their original order.
            pending.append({"obj": obj, "key": key, "hits": hits, "recent_interior": recent_interior,
                            "state": state, "covariance": covariance, "track": track,
                            "evidence": np.vstack([e[1] for e in track["evidence"]])})
        # One native call counts every track's accumulated evidence, instead of
        # one call per track: the same distinct-voxel count, same per-track value.
        counts = accelerator.voxel_counts([record["evidence"] for record in pending],
                                          cfg["cluster_voxel_m"], self.native_kernels)
        for record, count in zip(pending, counts):
            record["count"] = int(count)
        for record in pending:
            obj, key, hits = record["obj"], record["key"], record["hits"]
            recent_interior = record["recent_interior"]
            state, covariance, track = record["state"], record["covariance"], record["track"]
            confirmed = obj["immediate"] or (hits >= cfg["confirmation_hits"]
                                             and int(record["count"]) >= cfg["evidence_min_points"])
            intersection_confirmed = (confirmed and obj["path_relation"] == "intersecting"
                                      and (obj["intersection_immediate"] or len(recent_interior) >= cfg["confirmation_hits"]))
            obj.update(track_id=key, hits=hits, confirmed=bool(confirmed),
                       accumulated_support_voxels=int(record["count"]),
                       intersection_confirmed=bool(intersection_confirmed),
                       intersection_hits=len(recent_interior),
                       intersection_evidence_timestamps_s=[s for _, s in recent_interior],
                       intersection_confirmation=("immediate_interior_geometry" if intersection_confirmed and obj["intersection_immediate"]
                                                  else "temporal_interior_evidence" if intersection_confirmed else "pending"),
                       last_observed_s=stamp,
                       evidence_timestamps_s=[e[0] for e in track["evidence"]],
                       covariance_kind="heuristic_not_calibrated",
                       velocity_world_mps=state[3:].tolist(), position_covariance_m2=covariance[:3, :3].tolist(),
                       confirmation="immediate_geometry" if obj["immediate"] else ("temporal_evidence" if confirmed else "pending"),
                       track_age_s=stamp - track["first_stamp"])

    def process(self, points: np.ndarray, timestamp_s: float, point_times: np.ndarray | None = None,
                *, capture_diagnostics: bool = False) -> dict:
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
        self.diagnostic_arrays = {"decoded_points": points} if capture_diagnostics else {}
        pipeline = {"geometry": {"state": "not_run"}, "segmentation": {"state": "not_run"},
                    "association": {"state": "not_run"}}
        # A non-finite coordinate always yields a non-finite radius, which fails
        # one of the range bounds, so an explicit finite test would drop exactly
        # the same rows. Measurements are decoded as float64 triples.
        keep = accelerator.range_indices(points, self.config["min_range_m"], self.config["max_range_m"],
                                         self.native_kernels)
        points = points[keep]
        if capture_diagnostics:
            self.diagnostic_arrays["range_points"] = points
        self.display_points = points
        if len(point_times):
            point_times = point_times[keep]
            if len(point_times) and np.ptp(point_times) == 0:
                point_times = np.empty(0)
        result = {"timestamp_s": timestamp_s, "status": "unknown", "objects": [], "nearest_obstacle_m": None,
                  "input_valid_points": len(points), "gap_reset": reset,
                  "coordinate_frame": "tunnel_guard_local",
                  "distance_method": ("minimum_forward_x_of_envelope_evidence_for_hazards"
                      if self.config.get("obstacle_distance_mode", "cluster_min_x") == "envelope_support_min_x"
                      else "minimum_forward_x_of_current_observed_cluster"),
                  "distance_origin": "configured_processing_frame_origin",
                  "distance_along_path_m": None,
                  "health": "unavailable", "health_reasons": ["insufficient_returns"],
                  "pipeline": pipeline,
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
        # Crop and reduce in one pass: the cropped copy is never materialised, and
        # the native kernel partitions by the leading key axis so each slice is
        # deduplicated and sorted in cache.
        accelerator_module = self.native_kernels if self.config.get("voxel_backend", "numpy") == "cpp" else None
        reduced = frame[accelerator.crop_voxels(frame, self.config["min_forward_m"],
                                                self.config["context_half_width_m"],
                                                self.config["geometry_voxel_m"], accelerator_module)]
        if capture_diagnostics:
            self.diagnostic_arrays.update(registered_points=frame, cropped_points=frame[crop], geometry_voxel_points=reduced)
        geometry = TrackGeometry(reduced, self.config)
        mounting_started = time.perf_counter()
        mounting = observe_mounting(reduced, geometry, self.config)
        mounting_s = time.perf_counter() - mounting_started
        if mounting is not None:
            result.update(mounting=mounting, mounting_observation_s=mounting_s)
        pipeline["geometry"] = {"state": "ran", "valid": geometry.valid, "reason": geometry.reason}
        carried: list = []
        objects = cluster_candidates(reduced, geometry, self.config, pipeline["segmentation"],
                                     self.diagnostic_arrays if capture_diagnostics else None,
                                     reduced_on_grid_m=self.config["geometry_voxel_m"],
                                     classification_out=carried) if geometry.valid else []
        if not geometry.valid:
            pipeline["segmentation"]["reason"] = geometry.reason
        self._associate(objects, pose, timestamp_s, motion["valid"])
        pipeline["association"] = {"state": "ran", "candidates": len(objects),
                                   "confirmed": sum(o["confirmed"] for o in objects),
                                   "history_cleared_for_motion": not motion["valid"]}
        hazards = [o for o in objects if o["path_relation"] in ("intersecting", "unresolved")]
        confirmed = [o for o in hazards if o["confirmed"]]
        certain = [o for o in confirmed if o["intersection_confirmed"]]
        status = ("obstacle" if certain else ("unresolved_obstacle" if confirmed else
                  ("candidate" if hazards else ("no_obstacle_observed" if geometry.valid else "unknown"))))
        bins = []
        # Reuse the classification the cluster stage already computed for exactly
        # this array instead of classifying it a second time.
        classification = carried[0] if carried else geometry.classify(reduced, remove_background=False)
        _, _, _, observed, _ = classification
        edges = self.config["range_bins_m"]
        if self.native_kernels is not None:
            counts = accelerator.range_summary(reduced, frame, crop, observed,
                                               np.column_stack((edges[:-1], edges[1:])), self.native_kernels)
            for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                bins.append({"range_m": [lo, hi], "returns": int(counts[index, 0]),
                             "returns_before_geometry_voxel": int(counts[index, 1]),
                             "geometry_supported_returns": int(counts[index, 2])})
        else:
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (reduced[:, 0] >= lo) & (reduced[:, 0] < hi)
                raw_mask = (frame[:, 0] >= lo) & (frame[:, 0] < hi) & crop
                bins.append({"range_m": [lo, hi], "returns": int(mask.sum()),
                             "returns_before_geometry_voxel": int(raw_mask.sum()),
                             "geometry_supported_returns": int(np.count_nonzero(mask & observed))})
        health_reasons = []
        if self.config.get("rail_frame_mode", "bed") == "local_3d":
            health_reasons.append("experimental_local_rail_frames")
            if not geometry.frame_segments():
                health_reasons.append("local_rail_frames_unavailable")
                if status == "no_obstacle_observed":
                    status = "unknown"
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
