"""Locally supported track-bed and paired-rail geometry, never an empty-scan map.

Region-wise ground modelling follows the motivation of Himmelsbach et al. (IV
2010); this is a new rail-specific implementation, not a paper reproduction.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from . import accelerator


def voxel_representatives(points: np.ndarray, size: float, backend: str = "numpy") -> np.ndarray:
    if not len(points):
        return points
    if backend == "cpp":
        from . import _native
        if points.dtype != np.float64:
            raise ValueError("cpp voxel backend requires float64 measurements")
        indices = np.frombuffer(_native.voxel_indices(np.ascontiguousarray(points), size), dtype=np.int64)
    elif backend == "numpy":
        _, indices = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    else:
        raise ValueError("Unknown voxel backend")
    return points[indices]


# scipy prepares worker threads per call, so parallelism only pays for large
# query batches: measured on this machine, 1024 points are still slower with
# workers and 4096 points break even. The threshold selects scheduling only;
# each query is independent, so results are identical either way.
PARALLEL_QUERY_MIN_POINTS = 4096


def query_workers(count: int, configured: int) -> int:
    """Worker count for one independent nearest-neighbour query batch."""
    if configured != -1:
        return configured
    return -1 if count >= PARALLEL_QUERY_MIN_POINTS else 1


def nearest_anchor_distance(x: np.ndarray, anchor_x: np.ndarray) -> np.ndarray:
    """Exact nearest |x - anchor_x| without materialising an N x M difference.

    Anchors are strictly increasing, so the nearest anchor is one of the two
    bracketing a value; the compared differences are the same subtractions the
    dense form would take, and NaN propagates as before.
    """
    position = np.clip(np.searchsorted(anchor_x, x), 1, len(anchor_x) - 1)
    return np.minimum(np.abs(x - anchor_x[position - 1]), np.abs(anchor_x[position] - x))


def robust_plane(points: np.ndarray, config: dict) -> tuple[np.ndarray | None, dict]:
    lo, hi = config["ground_fit_range_m"]
    min_height, max_height = config["ground_sensor_height_bounds_m"]
    mask = ((points[:, 0] >= lo) & (points[:, 0] <= hi)
            & (np.abs(points[:, 1]) < config["ground_fit_half_width_m"])
            & (points[:, 2] > -max_height) & (points[:, 2] < -min_height))
    sample = voxel_representatives(points[mask], max(config["geometry_voxel_m"], 0.12), config.get("voxel_backend", "numpy"))
    rng = np.random.default_rng(config["seed"])
    if len(sample) > 6000:
        sample = sample[rng.choice(len(sample), 6000, replace=False)]
    diagnostics = {"sample_points": len(sample), "valid": False}
    if len(sample) < config["ground_min_support"]:
        return None, diagnostics | {"reason": "insufficient_track_bed_points"}
    design = np.column_stack((sample[:, :2], np.ones(len(sample))))
    best, best_count = None, 0
    tolerance = config["ground_inlier_m"]
    for _ in range(config["ground_ransac_trials"]):
        ids = rng.choice(len(sample), 3, replace=False)
        try:
            plane = np.linalg.solve(design[ids], sample[ids, 2])
        except np.linalg.LinAlgError:
            continue
        if (np.any(np.abs(plane[:2]) > config["ground_max_slopes"])
                or not -max_height < plane[2] < -min_height):
            continue
        count = int(np.count_nonzero(np.abs(sample[:, 2] - design @ plane) < tolerance))
        if count > best_count:
            best, best_count = plane, count
    if best is None:
        return None, diagnostics | {"reason": "no_upright_track_bed_plane"}
    for _ in range(3):
        good = np.abs(sample[:, 2] - design @ best) < tolerance
        best = np.linalg.lstsq(design[good], sample[good, 2], rcond=None)[0]
    good = np.abs(sample[:, 2] - design @ best) < tolerance
    support = int(good.sum())
    fraction = support / len(sample)
    spread = np.ptp(sample[good, :2], axis=0) if support else np.zeros(2)
    valid = (support >= config["ground_min_support"] and fraction >= config["ground_min_fraction"]
             and spread[0] >= (hi - lo) * 0.4 and spread[1] >= 0.6
             and np.all(np.abs(best[:2]) <= config["ground_max_slopes"]))
    residual = float(np.median(np.abs(sample[good, 2] - design[good] @ best))) if support else tolerance
    diagnostics |= {"valid": bool(valid), "support": support, "fraction": fraction,
                    "median_residual_m": residual, "span_m": spread.tolist(),
                    "reason": "supported_track_bed" if valid else "weak_track_bed_plane"}
    return (best if valid else None), diagnostics


class TrackGeometry:
    def __init__(self, points: np.ndarray, config: dict):
        self.config = config
        self.background = None
        self.plane, self.ground_quality = robust_plane(points, config)
        self.ground_anchors = np.empty((0, 3))
        self.rail_anchors = np.empty((0, 4))
        self.rail_head_height_m = None
        self.reason = self.ground_quality["reason"]
        if self.plane is None:
            return
        self._ground_profile(points)
        self._rail_profile(points)
        if len(self.rail_anchors) < 2:
            self.reason = "insufficient_paired_rail_support"
        else:
            self.reason = "supported_geometry"
        if self.valid and config["background"]["enabled"]:
            from .background import TunnelBackground
            self.background = TunnelBackground(points, self, config)

    @property
    def valid(self) -> bool:
        return self.plane is not None and len(self.rail_anchors) >= 2

    def _ground_profile(self, points: np.ndarray):
        cfg = self.config
        residual = points[:, 2] - (points[:, :2] @ self.plane[:2] + self.plane[2])
        lateral_ok = np.abs(points[:, 1]) < cfg["ground_fit_half_width_m"]
        anchors = []
        previous = 0.0
        half = cfg["ground_local_window_m"] / 2
        limit = cfg["ground_inlier_m"] * 2
        # Sorting once makes each longitudinal window a contiguous slice of the
        # same measurements; the original inequalities are then applied to the
        # slice, so the selected set, its median and its spread are unchanged.
        order = np.argsort(points[:, 0], kind="stable")
        sorted_x = points[order, 0]
        for x in np.arange(cfg["ground_segment_m"], cfg["max_range_m"], cfg["ground_segment_m"]):
            start = np.searchsorted(sorted_x, x - half, side="left")
            stop = np.searchsorted(sorted_x, x + half, side="right")
            ids = order[start:stop]
            ids = ids[(np.abs(sorted_x[start:stop] - x) < half) & lateral_ok[ids]
                      & (np.abs(residual[ids] - previous) < limit)]
            values = residual[ids]
            if len(values) < max(12, cfg["ground_min_support"] // 3):
                continue
            support_points = points[ids]
            if np.ptp(support_points[:, 0]) < 1.5 or np.ptp(support_points[:, 1]) < 0.4:
                continue
            shift = float(np.median(values))
            mad = float(np.median(np.abs(values - shift)))
            if anchors and abs(shift - previous) / (x - anchors[-1][0]) > cfg["ground_max_slopes"][0]:
                continue
            anchors.append((float(x), shift, max(mad * 1.4826, 0.015)))
            previous = shift
        self.ground_anchors = np.asarray(anchors, dtype=float).reshape(-1, 3)

    def ground(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.plane is None or not len(self.ground_anchors):
            return np.full(len(points), np.nan), np.full(len(points), np.inf)
        native = accelerator.native(self.config)
        if native is not None:
            return accelerator.ground_values(points, self.plane, self.ground_anchors,
                                             self.config["ground_max_extrapolation_m"], native)
        x = points[:, 0]
        anchors = self.ground_anchors
        shift = np.interp(x, anchors[:, 0], anchors[:, 1])
        nearest = nearest_anchor_distance(x, anchors[:, 0])
        uncertainty = np.interp(x, anchors[:, 0], anchors[:, 2]) + nearest * 0.008
        uncertainty[nearest > self.config["ground_max_extrapolation_m"]] = np.inf
        z = points[:, :2] @ self.plane[:2] + self.plane[2] + shift
        return z, uncertainty

    def _rail_profile(self, points: np.ndarray):
        cfg = self.config
        z, uncertainty = self.ground(points)
        h = points[:, 2] - z
        lo, hi = cfg["rail_height_bounds_m"]
        eligible = ((h > lo) & (h < hi) & (np.abs(points[:, 1]) < 4)
                    & (uncertainty < cfg["ground_max_uncertainty_m"]))
        rail = points[eligible]
        anchors, head_heights = [], []
        bin_size = cfg["rail_bin_m"]
        offset = int(np.ceil(4 / bin_size)) + 2
        # Same treatment as the bed profile: one sort, then the identical
        # inequality is applied to each contiguous longitudinal slice.
        order = np.argsort(rail[:, 0], kind="stable")
        sorted_x = rail[order, 0]
        for x in np.arange(5.0, cfg["max_range_m"], cfg["ground_segment_m"]):
            window = min(cfg["rail_max_window_m"], cfg["rail_window_m"] + x * cfg["rail_window_growth"])
            half = window / 2
            start = np.searchsorted(sorted_x, x - half, side="left")
            stop = np.searchsorted(sorted_x, x + half, side="right")
            ids = order[start:stop]
            q = rail[ids[np.abs(sorted_x[start:stop] - x) < half]]
            if not len(q):
                continue
            slope = 0.0
            if len(anchors) > 1:
                slope = float(np.clip((anchors[-1][1] - anchors[-2][1]) / (anchors[-1][0] - anchors[-2][0]),
                                      -cfg["rail_max_heading"], cfg["rail_max_heading"]))
            # Compensate heading before pooling a long-range window; otherwise a
            # real curved/oblique rail smears out of a narrow lateral histogram.
            lateral = q[:, 1] - slope * (q[:, 0] - x)
            valid = np.abs(lateral) < 4
            q, lateral = q[valid], lateral[valid]
            keys = np.unique(np.floor(np.column_stack((q[:, 0] / 0.30, lateral / bin_size))).astype(np.int64), axis=0)
            counts = np.bincount(keys[:, 1] + offset, minlength=2 * offset + 1)
            peaks, _ = find_peaks(counts, distance=max(1, int(0.30 / bin_size)), prominence=2)
            peaks = peaks[counts[peaks] >= cfg["rail_min_longitudinal_bins"]]
            expected = anchors[-1][1] + slope * (x - anchors[-1][0]) if anchors else 0.0
            allowed = (cfg["rail_max_heading"] * (x - anchors[-1][0]) + 0.15
                       if anchors else cfg["path_initial_center_bound_m"])
            pairs = []
            for i, left in enumerate(peaks):
                for right in peaks[i + 1:]:
                    gauge = (right - left) * bin_size
                    if abs(gauge - cfg["rail_gauge_m"]) > cfg["rail_gauge_tolerance_m"]:
                        continue
                    center = ((left + right) / 2 - offset + 0.5) * bin_size
                    if abs(center - expected) > allowed:
                        continue
                    left_points = np.abs(lateral - (center - gauge / 2)) < cfg["rail_half_width_m"] / 2
                    right_points = np.abs(lateral - (center + gauge / 2)) < cfg["rail_half_width_m"] / 2
                    if any(np.count_nonzero(mask) < 2 or np.ptp(q[mask, 0]) < cfg["rail_min_span_m"]
                           for mask in (left_points, right_points)):
                        continue
                    support = int(min(counts[left], counts[right]))
                    score = support / (1 + 4 * abs(center - expected) + 8 * abs(gauge - cfg["rail_gauge_m"]))
                    pairs.append((score, center, gauge, support))
            if not pairs:
                continue
            pairs.sort(reverse=True)
            if (len(pairs) > 1 and pairs[1][0] > 0.8 * pairs[0][0]
                    and abs(pairs[0][1] - pairs[1][1]) > 0.3):
                break
            _, center, gauge, support = pairs[0]
            rail_mask = np.abs(np.abs(lateral - center) - gauge / 2) < cfg["rail_half_width_m"] / 2
            bed, _ = self.ground(q[rail_mask])
            head_heights.append(float(np.quantile(q[rail_mask, 2] - bed, 0.8)))
            anchors.append((float(x), center, gauge, support))
        self.rail_anchors = np.asarray(anchors, dtype=float).reshape(-1, 4)
        if head_heights:
            self.rail_head_height_m = float(np.median(head_heights))

    def path(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.rail_anchors) < 2:
            return np.zeros(len(x)), np.full(len(x), self.config["rail_gauge_m"]), np.full(len(x), np.inf)
        a = self.rail_anchors
        center = np.interp(x, a[:, 0], a[:, 1])
        # Endpoint tangent extrapolation; uncertainty grows with unsupported distance.
        for edge, other, mask in [(0, 1, x < a[0, 0]), (-1, -2, x > a[-1, 0])]:
            slope = np.clip((a[edge, 1] - a[other, 1]) / (a[edge, 0] - a[other, 0]),
                            -self.config["rail_max_heading"], self.config["rail_max_heading"])
            center[mask] = a[edge, 1] + slope * (x[mask] - a[edge, 0])
        nearest = nearest_anchor_distance(x, a[:, 0])
        uncertainty = 0.06 + 0.008 * nearest + 0.0003 * nearest**2
        uncertainty[nearest > self.config["path_max_extrapolation_m"]] = np.inf
        return center, np.interp(x, a[:, 0], a[:, 2]), uncertainty

    def classify(self, points: np.ndarray, *, remove_background: bool = True,
                 include_boundary: bool = False, ground: tuple | None = None):
        """Classify support; optionally expose uncertain lateral envelope intersections.

        The interval uses the existing heuristic path uncertainty, not calibrated
        probability or a guarantee about the physical vehicle envelope. A caller
        that already fitted the bed for the same array may pass it in; the
        returned values are those of the identical fit.
        """
        cfg = self.config
        native = accelerator.classify_geometry(points, self, accelerator.native(cfg))
        if native is not None:
            core, context, height, observed, nominal_overlap, boundary = native
            if remove_background and self.background is not None:
                context &= ~self.background.mask(points, (observed & nominal_overlap) | boundary)
            return (core, context, height, observed, nominal_overlap, boundary) if include_boundary \
                else (core, context, height, observed, nominal_overlap)
        z, ground_uncertainty = self.ground(points) if ground is None else ground
        center, gauge, path_uncertainty = self.path(points[:, 0])
        height = points[:, 2] - z
        rail_head = self.rail_head_height_m if self.rail_head_height_m is not None else np.nan
        # Project into a perpendicular track-bed cross-section. A z-only shear
        # incorrectly keeps lateral position unchanged when the sensor is rolled.
        slope = self.plane[1] if self.plane is not None else 0.0
        normal_scale = np.sqrt(1 + np.sum(self.plane[:2] ** 2)) if self.plane is not None else 1.0
        running_height = (height - rail_head) / normal_scale
        dy = points[:, 1] - center
        lateral = (dy + slope * (height - rail_head + slope * dy)) / np.sqrt(1 + slope**2)
        envelope = np.asarray(cfg["envelope_segments_m"])
        segment = envelope[np.clip(np.searchsorted(envelope[:, 0], running_height, side="right") - 1, 0, len(envelope) - 1)]
        fraction = np.clip((running_height - segment[:, 0]) / (segment[:, 1] - segment[:, 0]), 0, 1)
        width = segment[:, 2] + fraction * (segment[:, 3] - segment[:, 2]) + cfg["envelope_margin_m"]
        observed = ((ground_uncertainty <= cfg["ground_max_uncertainty_m"])
                    & (path_uncertainty <= cfg["path_max_uncertainty_m"]))
        vertical = ((running_height >= envelope[0, 0] + ground_uncertainty)
                    & (running_height <= envelope[-1, 1]))
        # Remove only the measured rail-head band, not all points near a rail.
        on_rail = (observed & (np.abs(np.abs(lateral) - gauge / 2) < cfg["rail_half_width_m"] + path_uncertainty)
                   & (running_height <= cfg["rail_vertical_margin_m"]))
        ground_supported = (ground_uncertainty <= cfg["ground_max_uncertainty_m"]) & vertical & ~on_rail
        # A supported path estimate still has nonzero uncertainty. Project its
        # lateral offset through the same roll transform as the measured points.
        # A nominal crossing by a millimetre is not a definite intrusion when
        # the path centre is uncertain by centimetres. Preserve both sides of
        # that boundary as unresolved evidence instead of discarding the points.
        lateral_uncertainty = path_uncertainty * np.sqrt(1 + slope**2)
        nominal_core = ground_supported & observed & (np.abs(lateral) <= width)
        core = nominal_core & (np.abs(lateral) + lateral_uncertainty <= width)
        boundary = (ground_supported & observed & ~core
                    & (np.abs(lateral) - lateral_uncertainty <= width))
        # Segmentation precedes the collision gate. Do not amputate the feet or
        # head of an object just because only part intersects the envelope.
        segmentation_height = (running_height >= cfg["min_running_height_m"]) & (running_height <= envelope[-1, 1] + cfg["cluster_context_margin_m"])
        context = segmentation_height & ~on_rail & (np.abs(lateral) <= cfg["segmentation_context_half_width_m"])
        nominal_overlap = ((running_height >= envelope[0, 0]) & (running_height <= envelope[-1, 1])
                           & ~on_rail & (np.abs(lateral) <= width))
        if remove_background and self.background is not None:
            background = self.background.mask(points, (observed & nominal_overlap) | boundary)
            context &= ~background
        result = (core, context, height, observed, nominal_overlap)
        return result + (boundary,) if include_boundary else result

    def describe(self) -> dict:
        return {"valid": self.valid, "reason": self.reason, "ground_quality": self.ground_quality,
                "lateral_boundary_policy": "heuristic_path_interval",
                "ground_plane": None if self.plane is None else self.plane.tolist(),
                "rail_head_height_m": self.rail_head_height_m,
                "ground_anchors": self.ground_anchors.tolist(), "rail_anchors": self.rail_anchors.tolist(),
                "background": None if self.background is None else self.background.describe()}
