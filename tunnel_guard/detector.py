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
from .geometry import CrossSection, TrackGeometry, voxel_representative_indices
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
    ransac_threads = config.get("background", {}).get("ransac_threads", 1)
    if type(ransac_threads) is not int or ransac_threads < 1:
        raise ValueError("background.ransac_threads must be a positive integer")
    if not isinstance(config.get("odometry_native_preprocess", True), bool):
        raise ValueError("odometry_native_preprocess must be boolean")
    plane_backend = config.get("background", {}).get("plane_backend", "open3d")
    if plane_backend not in ("open3d", "native"):
        raise ValueError("background.plane_backend must be 'open3d' or 'native'")
    if plane_backend == "native" and config.get("native_backend", "cpu") != "cuda":
        raise ValueError("background.plane_backend 'native' needs native_backend 'cuda': the exact port "
                         "lives in the device backend")
    if config.get("native_backend", "cpu") not in ("cpu", "cuda"):
        raise ValueError("Unknown native_backend; expected 'cpu' or 'cuda'")
    query_workers = config.get("query_workers", -1)
    if type(query_workers) is not int or (query_workers < 1 and query_workers != -1):
        raise ValueError("query_workers must be -1 (all cores) or a positive integer")
    accelerator.native(config)  # The detector has one required decision backend.
    if config.get("obstacle_distance_mode", "cluster_min_x") not in ("cluster_min_x", "envelope_support_min_x"):
        raise ValueError("Unknown obstacle_distance_mode")
    if config.get("rail_center_estimator", "histogram") not in ("histogram", "paired_line"):
        raise ValueError("Unknown rail_center_estimator")
    if config.get("rail_initial_heading", "zero") not in ("zero", "fitted"):
        raise ValueError("Unknown rail_initial_heading")
    if config.get("rail_pair_continuity", "window") not in ("window", "relocated"):
        raise ValueError("Unknown rail_pair_continuity")
    if config.get("rail_frame_mode", "bed") != "bed":
        raise ValueError("rail_frame_mode must be bed; local_3d has no C++ implementation")
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


def _far_field_bounds(distances_m: np.ndarray, geometry: TrackGeometry) -> list:
    """The path uncertainty the classifier itself used at each distance.

    Same value as one `_far_field_bound` call per distance: `TrackGeometry.path` applies only
    element-wise operations, so evaluating the distances together returns the identical
    numbers for each of them and costs one NumPy dispatch instead of one per object.
    """
    if len(geometry.rail_anchors) < 2:
        return [None] * len(distances_m)
    if not len(distances_m):
        return []
    uncertainty = geometry.path(np.asarray(distances_m, dtype=float))[2]
    return [float(value) if np.isfinite(value) else None for value in uncertainty]


class _NativePreprocessor:
    """kiss-ICP's frame preprocessing, replaced by the selection it performs.

    With deskew disabled the library keeps the points whose range falls strictly inside the
    configured band, in input order and unchanged - a selection, not arithmetic. Doing it
    here is therefore the same frame, and it is measured at a fraction of the cost. If the
    deskew is ever enabled this refuses to stand in, because then the preprocessing includes
    motion compensation.
    """

    def __init__(self, config: dict):
        self.minimum = float(config["min_range_m"])
        self.maximum = float(config["max_range_m"])
        self.native = accelerator.native(config)

    def preprocess(self, frame: np.ndarray, timestamps: np.ndarray, relative_motion: np.ndarray):
        # Timestamps may be present while the deskew is disabled: the library ignores them in
        # exactly that case, and the deskew flag is checked where this object is installed.
        rows = accelerator.range_indices_open(frame, self.minimum, self.maximum, self.native)
        return frame[rows]


def relation_reason(obj: dict) -> str:
    """Why this object has the relation to the corridor that it has.

    One place, the native path, derived from the object's own evidence counts, so the reason
    a consumer reads is the reason the decision was taken in.

    * `inside_heuristic_path_and_ground_interval` - certified interior support.
    * `certified_interior_shared_with_structure` - the same, but every certified voxel sits in a
      cell that the tunnel's own cross-section occupies elsewhere along the scan, so the object
      and the tunnel are not separable by returns alone.
    * `envelope_boundary_uncertainty` - measured interior support whose uncertainty interval
      crosses the contour edge, or edge uncertainty on a surface that is not the tunnel's.
    * `structure_crossing_envelope` - interior support that belongs to the tunnel's own
      cross-section; the contour clips it at its edge. Reported, never a hazard.
    * `unmeasured_corridor` - interior support only where the bed or the centre-line is
      beyond its uncertainty budget, so the lateral relation is our extrapolation.
    * `outside_envelope_evidence` - no interior support at all.
    """
    if obj["path_relation"] == "unknown":
        return "geometry_unavailable"
    if obj["path_relation"] == "intersecting":
        return ("inside_heuristic_path_and_ground_interval" if obj["certified_unexplained_voxels"]
                else "certified_interior_shared_with_structure")
    if obj["path_relation"] == "unresolved":
        return "envelope_boundary_uncertainty"
    if obj["interior_structural_voxels"]:
        return "structure_crossing_envelope"
    if obj["interior_unmeasured_voxels"]:
        return "unmeasured_corridor"
    if obj["boundary_unexplained_voxels"]:
        return "envelope_boundary_uncertainty"
    return "outside_envelope_evidence"


def cluster_candidates(points: np.ndarray, geometry: TrackGeometry, config: dict,
                       diagnostics: dict | None = None, arrays: dict | None = None,
                       *, reduced_on_grid_m: float | None = None,
                       classification_out: list | None = None,
                       cluster_rows_out: list | None = None) -> list[dict]:
    """Candidate objects of this scan.

    `cluster_rows_out`, when given, receives the single row index array (into `points`) of the
    reduced cluster cloud the objects were built from. It is the exact selection this function
    already made - no nearest-point matching is repeated - and only the diagnostic caller asks
    for it, so the detection path pays nothing.
    """
    classification, section = geometry.classify_with_section(points)
    if classification_out is not None:
        # The caller needs the same classification for its range bins; the two
        # are identical because the inputs are, and background removal only
        # touches `context`.
        classification_out.append(classification)
    _, context, _, _, _ = classification
    # The tunnel's own cross-section is measured on EVERY return of this scan, not on the
    # object pool: the pool has already had the fitted tunnel surfaces removed from it, so a
    # structural test run on the pool would be asked to recognise the tunnel from what is
    # left of it. The verdict is then carried onto the pool rows.
    structural_full = (geometry.structural_mask(points, section) if geometry.valid
                       else np.zeros(len(points), dtype=bool))
    if voxel_unique_at(config, reduced_on_grid_m):
        # The context cloud is already one point per cluster voxel in key order,
        # so the reduction below would only reproduce the same rows in the same
        # order; filtering keeps it identical without the sort.
        cloud = points[context]
        cluster_rows = np.flatnonzero(context) if cluster_rows_out is not None else None
        cloud_section = CrossSection(section.lateral[context], section.running_height[context],
                                     section.gauge[context])
        structural = structural_full[context]
    else:
        chosen = voxel_representative_indices(points[context], config["cluster_voxel_m"])
        cluster_rows = np.flatnonzero(context)[chosen]
        cloud = points[cluster_rows]
        cloud_section = CrossSection(section.lateral[cluster_rows], section.running_height[cluster_rows],
                                     section.gauge[cluster_rows])
        structural = structural_full[cluster_rows]
    if cluster_rows_out is not None:
        # Exact source rows selected by the native voxel reducer.
        cluster_rows_out.append(cluster_rows)
    if diagnostics is not None:
        diagnostics.update(state="ran", context_points=int(context.sum()), cluster_points=len(cloud), rejected={})
    if arrays is not None:
        before = geometry.classify(points, remove_background=False)[1]
        arrays.update(context_before_background=points[before], context_after_background=points[context],
                      cluster_points=cloud)
    if not len(cloud):
        return []
    masks, _ = geometry.classify_with_section(
        cloud, remove_background=False, include_boundary=True)
    core, _, heights, observed, nominal_overlap, boundary = masks
    # The rail-relative coordinates and the tunnel's own cross-section, both computed from
    # this scan alone. `interior` is the point estimate: the return's nominal position is
    # inside the reference contour. A claim on the corridor needs that AND a measured
    # coordinate AND evidence that is not the tunnel's own surface:
    #
    # * `observed` - where the bed or the centre-line is beyond its uncertainty budget the
    #   lateral coordinate is our own extrapolation. A nominal-intrusion claim built on it
    #   is a statement about our model, not about the corridor: measured, 94-98 per cent of
    #   the nominal-interior returns on real frames sit beyond that horizon, and they are
    #   what made the tunnel's own arch read as an intrusion at 60-200 m.
    # * `~structural` - a surface of the tunnel itself (wall, arch, ceiling, floor, walkway
    #   edge, cable tray) that the contour merely clips at its edge is a reference-contour
    #   interaction, not an object. The returns stay in the output with this reason.
    #
    # `boundary` (the uncertainty interval crosses the edge) is still computed and reported
    # per object, but it is evidence of doubt, not a claim of intrusion: it is the reason a
    # frame is `candidate` rather than `obstacle`, and it never by itself makes a hazard.
    lateral, running_height = cloud_section.lateral, cloud_section.running_height
    claim = nominal_overlap & observed & ~structural
    unmeasured = nominal_overlap & ~observed
    # CERTIFIED interior support is deliberately NOT gated by `~structural`, and the asymmetry
    # is measured rather than aesthetic. `unresolved` rests on the uncertainty interval, so it
    # needs the strongest exclusion - without it, the tunnel's own edge raised the alarm on
    # 98 % of frames. `intersecting` rests on a whole interval lying inside the contour, and
    # cross-section cell sharing cannot prove that a compact measured group is
    # infrastructure: an object may share cells with the bed or walkway elsewhere.
    # Suppressing certified interior evidence on that basis would conflate cell
    # occupancy with physical identity. Preserve the evidence and expose its split
    # through `certified_unexplained_voxels`, without claiming a verified object class.
    # Mixed evidence (one interior + one boundary return) must not disappear
    # merely because neither subset separately reaches weak_min_voxels.
    uncertain_support = core | claim
    method = config["segmentation_method"]
    split_groups = {}
    if method == "density":
        if geometry.valid:
            _, ground_uncertainty = geometry.ground(cloud)
            surface_contact = ((ground_uncertainty <= config["ground_max_uncertainty_m"])
                               & (heights - geometry.rail_head_height_m <= ground_uncertainty))
        else:
            surface_contact = np.zeros(len(cloud), dtype=bool)
        labels, density_core, split_groups = density_labels(
            cloud, config, surface_contact=surface_contact, surface_plane=geometry.plane)
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
    if len(cloud):
        objects, rejected_counts, rows = accelerator.cluster_objects(
            cloud, labels, core, boundary, density_core, heights, uncertain_support, config, native,
            geometry_valid=geometry.valid)
        rejected.update(rejected_counts)
        if arrays is not None:
            components = rows
    if diagnostics is not None:
        diagnostics.update(rejected=dict(rejected), accepted=len(objects),
                           noise_points=int(np.count_nonzero(labels < 0)))
        if arrays is not None:
            diagnostics["components"] = components
    # Every object carries the rail-relative coordinates of its own support and the split of
    # its evidence into the kinds the decision distinguishes. Computed here, once, for both
    # backends, so a reported reason and the decision it explains can never come from
    # different code. `_support_indices` indexes this cluster cloud, the same array the masks
    # above index.
    for obj in objects:
        group = split_groups.get(obj["component_id"])
        secondary = group is not None and group["parent"] != obj["component_id"]
        obj["_split_secondary"] = secondary
        obj["_association_reference"] = group if group is not None and not secondary else None
        obj["tracking_reference"] = ("split_component" if secondary else
                                     "unsplit_density_component" if group is not None else
                                     "observed_component")
        if group is not None and not secondary:
            obj["tracking_reference_bounds"] = {
                "bbox_min": group["bbox_min"], "bbox_max": group["bbox_max"]}
        rows = obj.pop("_support_indices")
        obj["lateral_m"] = ([float(lateral[rows].min()), float(lateral[rows].max())]
                            if geometry.valid else [None, None])
        obj["height_above_railhead_m"] = ([float(running_height[rows].min()),
                                          float(running_height[rows].max())]
                                         if geometry.valid else [None, None])
        obj["interior_voxels"] = int(np.count_nonzero(nominal_overlap[rows]))
        obj["certified_unexplained_voxels"] = int(np.count_nonzero(core[rows] & ~structural[rows]))
        obj["interior_structural_voxels"] = int(np.count_nonzero(nominal_overlap[rows] & structural[rows]))
        obj["interior_unmeasured_voxels"] = int(np.count_nonzero(unmeasured[rows] & ~structural[rows]))
        obj["claim_voxels"] = int(np.count_nonzero(claim[rows]))
        obj["boundary_unexplained_voxels"] = int(np.count_nonzero(boundary[rows] & ~structural[rows]))
        obj["path_relation_reason"] = relation_reason(obj)
        obj["_unexplained"] = rows[boundary[rows] & ~structural[rows]]
    # How far edge-uncertain evidence sits from the tunnel's own cross-section. The cell test
    # cannot hold a surface whose rail-relative position moves more than a cell along the
    # scan - measured, the tunnel's own wall base and platform edge do - so a few of their
    # returns come out unexplained. Those returns are attached to the surface they belong to;
    # a foreign object standing in the corridor is not. The claim channel does not use this
    # (an object's own nominal intrusion is the safety question); the weaker doubt channel
    # does, so a fragment of the tunnel can never raise the frame's state.
    isolation = float(config.get("structure_isolation_m", 0.3))
    tunnel = points[structural_full]
    for obj in objects:
        if not len(obj["_unexplained"]):
            continue
        probe = cloud[obj["_unexplained"]]
        # Exact nearest distance, restricted to the structural returns inside the probe's own
        # bounding box grown by the isolation distance: a tree over fifty thousand points per
        # frame cost more than the whole classifier, and the answer only needs to be known to
        # the tolerance it is compared against.
        low, high = probe.min(axis=0) - isolation, probe.max(axis=0) + isolation
        near = tunnel[np.all((tunnel >= low) & (tunnel <= high), axis=1)]
        if not len(near):
            obj["structure_distance_m"] = None
            continue
        delta = near[None, :, :] - probe[:, None, :]
        obj["structure_distance_m"] = float(np.sqrt(np.min(np.einsum("ijk,ijk->ij", delta, delta))))
    for obj in objects:
        obj.pop("_unexplained", None)
        obj.setdefault("structure_distance_m", None)
    # Keep association order independent of the selected distance definition.
    return sorted(objects, key=lambda o: o["cluster_nearest_x_m"])

class Detector:
    def __init__(self, config: dict):
        self.native_kernels = accelerator.native(config)
        self.config = config
        self.odometry = self._new_odometry()
        self.previous_source = None
        self.previous_pose = np.eye(4)
        self.cached_background = None
        self.background_frame_count = 0
        self.background_fit_pose = np.zeros(3)
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
        odometry = KissICP(cfg)
        if self.config.get("odometry_native_preprocess", True):
            if cfg.data.deskew:
                raise ValueError("odometry_native_preprocess requires deskew to be disabled: the "
                                 "library's preprocessing would include motion compensation")
            odometry.preprocessor = _NativePreprocessor(self.config)
        return odometry

    def _motion(self, points: np.ndarray, point_times: np.ndarray):
        if self.diagnostic_arrays:
            self.diagnostic_arrays.update(
                motion_initial_guess=self.odometry.last_pose @ self.odometry.last_delta,
                motion_map=self.odometry.local_map.point_cloud(),
                motion_sigma=np.array(self.odometry.adaptive_threshold.get_threshold()))
        frame, source = self.odometry.register_frame(points, point_times)
        pose = self.odometry.last_pose.copy()
        if self.diagnostic_arrays:
            self.diagnostic_arrays.update(motion_source=source, motion_pose=pose)
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
            # The residual is what the alignment is actually verified to, so it is reported as
            # the position uncertainty of this frame and used as one, instead of being compared
            # to a threshold and then discarded. It is a median nearest-neighbour distance over
            # a voxel-downsampled pair, so it bounds the pose error rather than equalling it.
            quality |= {"valid": bool(valid), "reason": "registered" if valid else "registration_rejected",
                        "overlap": overlap, "median_residual_m": median,
                        "position_sigma_m": max(float(median), self.config["tracking_pose_sigma_m"])}
        self.previous_source, self.previous_pose = source.copy(), pose
        return frame, pose, quality


    def _associate(self, objects: list[dict], pose: np.ndarray, stamp: float, motion: dict):
        cfg = self.config
        motion_valid = bool(motion["valid"])
        local_config = cfg.get("local_confirmation")
        # A rejected registration used to clear every track. Measured, the rejection is the tail
        # of one continuous residual distribution - accepted frames sit at a median residual of
        # 0.21 m against a 0.30 m gate, rejected ones at 0.36 m, and the overlap gate never
        # fires - so clearing destroyed the history for a marginal overshoot, and it cost a
        # 91-frame stretch of doubleT_platform in which the train travelled 25 m. Tracks now
        # persist; the frame is marked positionally uncertain and its hits do not count towards
        # confirmation, because a hit measured through an unverified transform is not
        # independent evidence. `track_max_gap_s` still ages out anything longer.
        self.tracks = {key: t for key, t in self.tracks.items() if stamp - t["stamp"] <= cfg["track_max_gap_s"]}
        ids = list(self.tracks)
        secondary = np.fromiter((obj.pop("_split_secondary") for obj in objects),
                                dtype=bool, count=len(objects))
        references = [obj.pop("_association_reference") for obj in objects]
        centers = np.array([o["center"] for o in objects], dtype=float).reshape(-1, 3)
        association_centers = (np.asarray([
            reference["center"] if reference is not None else obj["center"]
            for reference, obj in zip(references, objects)], dtype=float).reshape(-1, 3)
            if any(reference is not None for reference in references) else centers)
        association_extents = [reference["extent_m"] if reference is not None else obj["extent_m"]
                               for reference, obj in zip(references, objects)]
        world = association_centers @ pose[:3, :3].T + pose[:3, 3]
        if self.diagnostic_arrays:
            self.diagnostic_arrays["association_centers_world"] = world.copy()
        # The measured registration residual enters the association covariance, so a frame whose
        # pose is only verified to 0.3 m cannot claim a 0.1 m position uncertainty.
        pose_sigma = float(motion.get("position_sigma_m", cfg["tracking_pose_sigma_m"]))
        measurement_cov = (np.eye(3) * (cfg["tracking_position_sigma_m"]**2 + pose_sigma**2)
                           + self.motion_translation_covariance)
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
        # Added fragments must not change assignments among the original density
        # groups. Solve the two pools independently, including unmatched columns,
        # so even assignment tie-breaking cannot couple the pools.
        for fragment_pool in (False, True):
            pool_ids = [key for key in ids if self.tracks[key]["split_secondary"] == fragment_pool]
            pool_columns = np.flatnonzero(secondary == fragment_pool)
            if not pool_ids or not len(pool_columns):
                continue
            cost = np.full((len(pool_ids), len(pool_columns)), 1e6)
            extents = np.asarray([association_extents[index] for index in pool_columns]) + .1
            track_extents = np.asarray([self.tracks[key]["extent"] for key in pool_ids]) + .1
            positions = np.stack([predicted[key][0][:3] for key in pool_ids])
            covariances = np.stack([predicted[key][1][:3, :3] for key in pool_ids])
            inverse = np.linalg.inv(covariances + measurement_cov)
            residual = world[pool_columns][None, :, :] - positions[:, None, :]
            mahalanobis = np.einsum("tni,tij,tnj->tn", residual, inverse, residual)
            shape = np.linalg.norm(np.log(extents[None, :, :] / track_extents[:, None, :]), axis=2)
            allowed = (mahalanobis <= cfg["tracking_mahalanobis_gate"]) & (shape <= cfg["tracking_extent_log_gate"])
            cost[allowed] = mahalanobis[allowed] + shape[allowed]
            padded = np.column_stack((cost, np.full((len(pool_ids), len(pool_ids)),
                cfg["tracking_mahalanobis_gate"] + cfg["tracking_extent_log_gate"] + 1)))
            rows, columns = linear_sum_assignment(padded)
            for row, column in zip(rows, columns):
                if column < len(pool_columns) and cost[row, column] < 1e6:
                    matched[int(pool_columns[column])] = pool_ids[row]
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
            local_hits = 0
            local_overlap = None
            if local_config is not None:
                # Association proposes identity; only repeated measured shapes provide
                # this evidence. No rejected pose enters the comparison or support count.
                eligible = (obj["path_relation"] == "intersecting"
                            and obj["certified_unexplained_voxels"] >= cfg["claim_min_support_voxels"]
                            and len(support) >= cfg["evidence_min_points"])
                if eligible:
                    local_shape = support - centers[index]
                    previous_shape = track.get("local_shape")
                    consecutive = track.get("local_frame") == self.frame_number - 1
                    if consecutive and previous_shape is not None:
                        radius = float(local_config["shape_radius_m"])
                        forward = accelerator.keep_outside_radius(
                            local_shape, previous_shape, radius, self.native_kernels)
                        backward = accelerator.keep_outside_radius(
                            previous_shape, local_shape, radius, self.native_kernels)
                        local_overlap = min(1.0 - float(forward.mean()),
                                            1.0 - float(backward.mean()))
                    local_hits = (min(track.get("local_hits", 0) + 1, cfg["confirmation_window"])
                                  if local_overlap is not None
                                  and local_overlap >= local_config["minimum_symmetric_overlap"] else 1)
                    track.update(local_shape=local_shape, local_frame=self.frame_number,
                                 local_hits=local_hits)
                else:
                    track.pop("local_shape", None)
                    track.pop("local_frame", None)
                    track["local_hits"] = 0
                obj.update(sensor_frame_hits=local_hits, sensor_frame_shape_overlap=local_overlap)
            local_confirmed = (not motion_valid and local_config is not None
                               and local_hits >= cfg["confirmation_hits"])
            if self.diagnostic_arrays:
                self.diagnostic_arrays[f"support_{key}"] = support
                self.diagnostic_arrays[f"support_world_{key}"] = support @ pose[:3, :3].T + pose[:3, 3]
                self.diagnostic_arrays[f"support_relative_{key}"] = support - centers[index]
            # Track-local spatial evidence compensates estimated object translation.
            # Bounds remain from this frame; past points never fabricate present shape.
            # Through an unverified transform nothing is accumulated: a point placed by a pose
            # that was not verified would manufacture a stable object out of a bad registration.
            if motion_valid:
                # R(p-c), not (Rp+t)-(Rc+t): a point at its own center must
                # stay exactly zero, rather than creating cells across floor(0).
                track["evidence"].append((stamp, (support - centers[index]) @ pose[:3, :3].T))
                while track["evidence"] and stamp - track["evidence"][0][0] > cfg["evidence_window_s"]:
                    track["evidence"].popleft()
                if not track["history"] or track["history"][-1] != self.frame_number:
                    track["history"].append(self.frame_number)
            hits = sum(f > self.frame_number - cfg["confirmation_window"] for f in track["history"])
            # Object persistence cannot confirm a new path intrusion. Count only
            # current-scan interior evidence, once per strictly increasing scan.
            interior_history = track["intersection_history"]
            if motion_valid and obj["path_relation"] == "intersecting" and (
                    not interior_history or interior_history[-1][0] != self.frame_number):
                interior_history.append((self.frame_number, stamp))
            recent_interior = [(f, s) for f, s in interior_history
                               if f > self.frame_number - cfg["confirmation_window"]]
            track.update(state=state, covariance=covariance, stamp=stamp,
                         extent=np.asarray(association_extents[index]), split_secondary=bool(secondary[index]))
            # Accumulated support is the one quantity that needs its own call per
            # track; the stacks are counted together after this loop, so the field
            # writes below keep their original order.
            pending.append({"obj": obj, "key": key, "hits": hits, "recent_interior": recent_interior,
                            "state": state, "covariance": covariance, "track": track,
                            "local_confirmed": local_confirmed,
                            "evidence": (np.vstack([e[1] for e in track["evidence"]])
                                         if track["evidence"] else np.empty((0, 3)))})
        # One native call counts every track's accumulated evidence, instead of
        # one call per track: the same distinct-voxel count, same per-track value.
        counts = accelerator.voxel_counts([record["evidence"] for record in pending],
                                          cfg["cluster_voxel_m"], self.native_kernels)
        for record, count in zip(pending, counts):
            record["count"] = int(count)
            if self.diagnostic_arrays:
                self.diagnostic_arrays[f"evidence_{record['key']}"] = record["evidence"]
                self.diagnostic_arrays[f"evidence_stamps_{record['key']}"] = np.array(
                    [entry[0] for entry in record["track"]["evidence"]])
        for record in pending:
            obj, key, hits = record["obj"], record["key"], record["hits"]
            recent_interior = record["recent_interior"]
            state, covariance, track = record["state"], record["covariance"], record["track"]
            local_confirmed = record["local_confirmed"]
            # Presence is evidence for a measured component, not a corridor claim.
            # Dense adjacent objects may be present without becoming hazards.
            presence_confirmed = (local_confirmed or obj["immediate"]
                                  or (hits >= cfg["confirmation_hits"]
                                      and int(record["count"]) >= cfg["evidence_min_points"]))
            # ADMISSION vs CERTIFICATION: `weak_min_voxels` decides what a component needs to be
            # REPORTED at all; `claim_min_support_voxels` is the count of interior-support voxels a
            # candidate needs to CLAIM a hazard. The support count, not the component's size, is what
            # matters: a 4-voxel component whose only supporting voxel is one of them offers exactly
            # one voxel of evidence, and certification on that is what the singletons surfaced.
            #
            # Instant confirmation is INTERIOR geometry: dense, tall support that is inside the
            # contour, which is a dense object right in front of the train. It used to read the whole
            # component's density and extent instead, so any dense wall or arch fragment confirmed as
            # a hazard on a single frame whatever its relation to the corridor - measured, that is
            # what put a 37-voxel component at 52 m into `nearest_obstacle_m` with one interior voxel.
            # A candidate whose interior evidence is thin waits for temporal evidence instead.
            confirmed = (local_confirmed or obj["intersection_immediate"]
                         or (int(obj["uncertain_voxels"]) >= cfg.get("claim_min_support_voxels",
                                                                     cfg["weak_min_voxels"])
                             and hits >= cfg["confirmation_hits"]
                             and int(record["count"]) >= cfg["evidence_min_points"]))
            intersection_confirmed = (confirmed and obj["path_relation"] == "intersecting"
                                      and (local_confirmed or obj["intersection_immediate"]
                                           or len(recent_interior) >= cfg["confirmation_hits"]))
            obj.update(track_id=key, hits=hits, confirmed=bool(confirmed),
                       presence_confirmed=bool(presence_confirmed),
                       presence_confirmation=("sensor_frame_shape_recurrence" if local_confirmed
                                              else "immediate_geometry" if obj["immediate"]
                                              else "temporal_evidence" if presence_confirmed else "pending"),
                       accumulated_support_voxels=int(record["count"]),
                       intersection_confirmed=bool(intersection_confirmed),
                       intersection_hits=len(recent_interior),
                       intersection_evidence_timestamps_s=[s for _, s in recent_interior],
                       intersection_confirmation=("sensor_frame_shape_recurrence" if local_confirmed
                                                  else "immediate_interior_geometry" if intersection_confirmed and obj["intersection_immediate"]
                                                  else "temporal_interior_evidence" if intersection_confirmed else "pending"),
                       last_observed_s=stamp,
                       evidence_timestamps_s=[e[0] for e in track["evidence"]],
                       covariance_kind="heuristic_not_calibrated",
                       velocity_world_mps=state[3:].tolist(), position_covariance_m2=covariance[:3, :3].tolist(),
                       confirmation=("sensor_frame_shape_recurrence" if local_confirmed
                                     else "immediate_interior_geometry" if obj["intersection_immediate"]
                                     else "temporal_evidence" if confirmed else "pending"),
                       track_age_s=stamp - track["first_stamp"])

    def process(self, points: np.ndarray, timestamp_s: float, point_times: np.ndarray | None = None,
                *, capture_diagnostics: bool = False, point_attributes=None) -> dict:
        """Detect on one scan.

        `point_attributes` is optional sensor metadata decoded for exactly these `points` rows
        (see `io.PointAttributes`). It is never an input to a decision: the detection, support and
        temporal confirmation below read geometry alone, and the field numbers do not reach them.
        It is reported and, when `capture_diagnostics` is set, saved alongside the point arrays
        with a provenance index into the original cloud. Legacy and synthetic callers pass nothing.
        """
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
        if point_attributes is not None:
            # Sensor fields are only provenance when each one names exactly these decoded rows.
            # Checked here, before any detector state below moves, so a caller cannot attach
            # mismatched metadata to a scan or advance the stream with it accepted. Index/shape
            # checks only: the correspondence is positional, never re-derived from the cloud.
            source_indices = np.asarray(point_attributes.source_indices)
            if source_indices.ndim != 1 or len(source_indices) != len(points):
                raise ValueError("point_attributes.source_indices must hold one source slot per decoded "
                                 f"point ({len(points)} rows), got shape {source_indices.shape}")
            for key, values in point_attributes.values.items():
                shape = np.asarray(values).shape
                if shape != (len(points),):
                    raise ValueError(f"point_attributes.values[{key!r}] must align one value per decoded "
                                     f"point ({len(points)} rows), got shape {shape}")
        if reset:
            self.odometry = self._new_odometry()
            self.previous_source = None
            self.previous_pose = np.eye(4)
            self.tracks.clear()
        self.last_timestamp = timestamp_s
        self.frame_number += 1
        self.display_support = {}
        self.diagnostic_arrays = {"decoded_points": points} if capture_diagnostics else {}
        if capture_diagnostics and point_attributes is not None:
            # Provenance for the decoded stage: every source field aligned to the decoded rows,
            # plus the full original-slot validity mask, saved once here.
            self.diagnostic_arrays["decoded_xyz_valid_mask"] = point_attributes.xyz_valid_mask
            self.diagnostic_arrays.update(
                {f"decoded_{key}": value for key, value in point_attributes.arrays().items()})
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
            if point_attributes is not None:
                # `keep` are the exact rows this filter already selected, so the same selection
                # carries the metadata without a second lookup.
                self.diagnostic_arrays.update(
                    {f"range_{key}": value for key, value in point_attributes.arrays(keep).items()})
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
        if point_attributes is not None:
            # What the sensor fields are, what they cover, and what is NOT verified about them.
            # Present whenever a caller supplies decoded attributes; JSON-safe.
            result["sensor_attributes"] = point_attributes.summary()
        if len(points) < self.config["ground_min_support"]:
            self.tracks.clear()
            self.previous_source = None
            self.odometry = self._new_odometry()
            return result | {"reason": "insufficient_returns", "processing_s": time.perf_counter() - started}
        motion_started = time.perf_counter()
        frame, pose, motion = self._motion(points, point_times)
        self.display_points = frame
        motion_s = time.perf_counter() - motion_started
        # Provenance through the motion stage: `_motion` transforms one point at a time, so the KISS
        # row-preserving contract is that `frame` keeps the row order and count of `points`, and
        # `keep` still names these decoded rows. A future deskew that resamples must preserve the
        # count or fail here: dropping the trace on a mismatch would silently lose provenance, and a
        # count alone cannot prove order, so this asserts the contract instead of guarding it.
        trace_rows = keep if capture_diagnostics and point_attributes is not None else None
        if trace_rows is not None and len(frame) != len(points):
            raise RuntimeError(
                f"motion stage changed the point count ({len(points)} -> {len(frame)}); "
                "sensor provenance cannot be carried through this scan")
        # Fixed band around the sensor axis: a corridor-following window (the previous frame's
        # extrapolated centre) added ambiguous hazards without changing any frame decision, so the
        # base window is used unconditionally.
        base_half = float(self.config["context_half_width_m"])
        crop = (frame[:, 0] >= self.config["min_forward_m"]) & (np.abs(frame[:, 1]) < base_half)
        # Crop and reduce in one pass: the cropped copy is never materialised, and
        # the native kernel partitions by the leading key axis so each slice is
        # deduplicated and sorted in cache. The kernel is handed the masked cloud with a
        # half-width that covers it, so its partition stays exact.
        accelerator_module = self.native_kernels
        # The kernel returns indices into the array it is given and re-clips laterally, so it is handed
        # the already-masked subset with a half-width that cannot clip it again.
        subset = frame[crop]
        # The kernel applies its own symmetric lateral limit, so the limit is read off the masked
        # subset itself plus one voxel, which clears the strict inequality and keeps the grid as
        # tight as the data allows without dropping a point the mask kept.
        reach = (max(abs(float(subset[:, 1].min())), abs(float(subset[:, 1].max())))
                 + self.config["geometry_voxel_m"]) if len(subset) else base_half
        voxel_rows = accelerator.crop_voxels(subset, self.config["min_forward_m"], reach,
                                             self.config["geometry_voxel_m"], accelerator_module)
        reduced = subset[voxel_rows]
        geometry_rows = None
        if capture_diagnostics:
            self.diagnostic_arrays.update(registered_points=frame, cropped_points=frame[crop], geometry_voxel_points=reduced)
            if trace_rows is not None:
                # `crop` then `voxel_rows` are the exact selections above, composed: the rows
                # into `frame` are `flatnonzero(crop)` and the kept rows into that subset are
                # `voxel_rows`. `reduced` is not re-searched for.
                crop_rows = np.flatnonzero(crop)
                geometry_rows = trace_rows[crop_rows[voxel_rows]]
                self.diagnostic_arrays.update(
                    {f"geometry_voxel_{key}": value
                     for key, value in point_attributes.arrays(geometry_rows).items()})
        # The criterion is DISTANCE TRAVELLED, not frames: the model's validity is a spatial property, so a
        # stopped train needs no refit and a fast one needs refits sooner. Travel is measured on the pose
        # the odometry already produced, so it costs nothing to evaluate.
        travel_limit = float(self.config.get("background_refit_travel_m", 0.0))
        cadence = int(self.config.get("background_refit_every_frames", 1))
        if travel_limit > 0.0:
            here = np.asarray(pose, dtype=float)[:3, 3]
            refit_due = (self.cached_background is None
                         or float(np.linalg.norm(here - self.background_fit_pose)) >= travel_limit)
            if refit_due:
                self.background_fit_pose = here
        elif cadence > 1:
            self.background_frame_count += 1
            refit_due = (self.background_frame_count % cadence) == 1
        else:
            refit_due = True
        reuse = (not refit_due) and self.cached_background is not None
        # Skipping the CONSTRUCTION is what saves the time; disabling it and attaching the cached model
        # afterwards would still pay the 40 ms and throw the result away.
        geometry_config = (dict(self.config, background=dict(self.config["background"], enabled=False))
                           if reuse else self.config)
        geometry = TrackGeometry(reduced, geometry_config)
        if reuse and geometry.valid:
            geometry.background = self.cached_background
        elif refit_due:
            self.cached_background = geometry.background
        mounting_started = time.perf_counter()
        mounting = observe_mounting(reduced, geometry, self.config)
        mounting_s = time.perf_counter() - mounting_started
        if mounting is not None:
            result.update(mounting=mounting, mounting_observation_s=mounting_s)
        pipeline["geometry"] = {"state": "ran", "valid": geometry.valid, "reason": geometry.reason}
        carried: list = []
        cluster_rows_out: list = []
        objects = cluster_candidates(reduced, geometry, self.config, pipeline["segmentation"],
                                     self.diagnostic_arrays if capture_diagnostics else None,
                                     reduced_on_grid_m=self.config["geometry_voxel_m"],
                                     classification_out=carried,
                                     cluster_rows_out=(cluster_rows_out if trace_rows is not None else None))
        if trace_rows is not None:
            # The representative cloud rows the cluster stage already selected, mapped back
            # through the geometry-voxel rows to the original slots of this scan.
            obj_rows = cluster_rows_out[0] if cluster_rows_out else np.empty(0, dtype=np.int64)
            self.diagnostic_arrays.update(
                {f"cluster_{key}": value
                 for key, value in point_attributes.arrays(geometry_rows[obj_rows]).items()})
        self._associate(objects, pose, timestamp_s, motion)
        pipeline["association"] = {"state": "ran", "candidates": len(objects),
                                   "confirmed": sum(o["confirmed"] for o in objects),
                                   "history_retained_for_motion": motion["valid"]}
        # A hazard needs interior evidence this frame actually measured, on something that is
        # not the tunnel's own cross-section. Both halves are decided per point upstream
        # (`claim`), so this is a partition of the object list, not a second policy.
        def is_hazard(obj: dict) -> bool:
            return obj["path_relation"] in ("intersecting", "unresolved")

        def is_doubt(obj: dict) -> bool:
            # Something is present, in a coordinate this frame actually measured, whose
            # relation to the corridor it cannot resolve: edge uncertainty that is neither the
            # tunnel's own cross-section nor attached to it. `structure_isolation_m` is what
            # separates an object standing in the corridor from a few returns of the tunnel's
            # own surface that the cell test could not hold. Doubt is reported and never
            # counted as an intrusion.
            # `None` means no structural return inside the isolation distance at all, which is
            # the most isolated case there is, not a missing measurement.
            distance = obj["structure_distance_m"]
            return (obj["boundary_unexplained_voxels"] >= int(self.config.get("claim_min_support_voxels", 2))
                    and (distance is None or distance > isolation))

        def is_range_unknown(obj: dict) -> bool:
            # Interior evidence only where the bed or the centre-line is beyond its own
            # uncertainty budget. The lateral relation there is our extrapolation, so this is
            # not doubt about an intrusion - it is the statement that the corridor's lateral
            # reference does not reach that far. Counted and reported separately, with
            # `supported_range_m` and each object's own `far_field_lateral_bound_m`.
            return bool(obj["interior_unmeasured_voxels"])

        isolation = float(self.config.get("structure_isolation_m", 0.3))
        hazards = [o for o in objects if is_hazard(o)]
        confirmed = [o for o in hazards if o["confirmed"]]
        certain = [o for o in confirmed if o["intersection_confirmed"]]
        doubt = [o for o in objects if not is_hazard(o) and is_doubt(o)]
        range_unknown = [o for o in objects if not is_hazard(o) and is_range_unknown(o)]
        status = ("obstacle" if certain else ("unresolved_obstacle" if confirmed else
                  ("candidate" if (hazards or doubt) else
                   ("no_obstacle_observed" if geometry.valid else "unknown"))))
        confirmed_ids = {id(o) for o in confirmed}
        uncertified = [o for o in hazards + doubt if id(o) not in confirmed_ids]
        bins = []
        # Reuse the classification the cluster stage already computed for exactly
        # this array instead of classifying it a second time.
        classification = carried[0] if carried else geometry.classify(reduced, remove_background=False)
        _, _, _, observed, _ = classification
        edges = self.config["range_bins_m"]
        counts = accelerator.range_summary(reduced, frame, crop, observed,
                                           np.column_stack((edges[:-1], edges[1:])), self.native_kernels)
        for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            bins.append({"range_m": [lo, hi], "returns": int(counts[index, 0]),
                         "returns_before_geometry_voxel": int(counts[index, 1]),
                         "geometry_supported_returns": int(counts[index, 2])})
        health_reasons = []
        calibration_caveats = []
        if not geometry.valid:
            health_reasons.append(geometry.reason)
        if not motion["valid"]:
            health_reasons.append(motion["reason"])
        if motion.get("weak_translation_axes", 3):
            health_reasons.append("weak_or_unmeasured_translation_observability")
        # Calibration caveats are permanent until an external input arrives, so they are kept
        # out of `health`: mixing "this frame went wrong" with "this project has no verified
        # extrinsics" made the field constant and therefore uninformative (measured: `normal`
        # on 0 of 2488 frames, entirely because of these two).
        if not self.config.get("sensor_profile_verified", False):
            calibration_caveats.append("unverified_sensor_profile_and_extrinsics")
        if not self.config.get("deskew_enabled", False):
            calibration_caveats.append("deskew_disabled_unverified_timing")
        elif not len(point_times):
            calibration_caveats.append("deskew_timestamps_unavailable")
        # One place for the native path: the native component path builds its own records, so the
        # far-field lateral bound is attached here rather than inside either builder.
        if objects:
            bounds = _far_field_bounds([obj["distance_m"] for obj in objects], geometry)
            for obj, bound in zip(objects, bounds):
                obj["far_field_lateral_bound_m"] = bound
        return result | {"status": status, "reason": geometry.reason,
                         "supported_range_m": geometry.supported_range_m(), "objects": objects,
                         "health": "unavailable" if not geometry.valid else ("degraded" if health_reasons else "normal"),
                         "health_reasons": health_reasons + calibration_caveats,
                         "health_degradations": health_reasons,
                         "calibration_caveats": calibration_caveats,
                         "position_uncertain": not motion["valid"],
                         "nearest_obstacle_m": min((o["distance_m"] for o in confirmed), default=None),
                         "nearest_candidate_m": min((o["distance_m"] for o in uncertified), default=None),
                         "unresolved_range_objects": len(range_unknown),
                         "nearest_unresolved_range_m": min((o["distance_m"] for o in range_unknown), default=None),
                         "geometry": geometry.describe(), "motion": motion, "pose": pose.tolist(),
                         "range_observability": bins, "geometry_points": len(reduced),
                         "processing_s": time.perf_counter() - started, "motion_s": motion_s}
