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


def envelope_width_bounds(low: np.ndarray, high: np.ndarray, segments: np.ndarray,
                          margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Extrema of a piecewise-linear half-width over each height interval.

    Visit both sides of segment discontinuities. Closed segment endpoints make
    an exactly coincident step conservative, without sampling away narrow steps.
    Intervals outside the vertical contour have no overlap (inf, -inf).
    """
    minimum, maximum = np.full(len(low), np.inf), np.full(len(low), -np.inf)
    for bottom, top, start, end in segments:
        a, b = np.maximum(low, bottom), np.minimum(high, top)
        overlap = a <= b
        wa = start + (end - start) * (a[overlap] - bottom) / (top - bottom) + margin
        wb = start + (end - start) * (b[overlap] - bottom) / (top - bottom) + margin
        minimum[overlap] = np.minimum(minimum[overlap], np.minimum(wa, wb))
        maximum[overlap] = np.maximum(maximum[overlap], np.maximum(wa, wb))
    return minimum, maximum


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
    # The proposals are drawn and solved one at a time, exactly as before, so the
    # random stream and the singular-trial skips are unchanged. Only the inlier
    # counting is batched over proposals afterwards.
    proposals = []
    for _ in range(config["ground_ransac_trials"]):
        ids = rng.choice(len(sample), 3, replace=False)
        try:
            plane = np.linalg.solve(design[ids], sample[ids, 2])
        except np.linalg.LinAlgError:
            continue
        if (np.any(np.abs(plane[:2]) > config["ground_max_slopes"])
                or not -max_height < plane[2] < -min_height):
            continue
        proposals.append(plane)
    if proposals:
        candidates = np.stack(proposals)
        projected = design @ candidates.T
        counts = np.count_nonzero(np.abs(sample[:, 2, None] - projected) < tolerance, axis=0)
        winner = int(np.argmax(counts))
        best, best_count = candidates[winner], int(counts[winner])
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


def refine_rail_pair(q, x, center, gauge, slope, cfg):
    """Fit both supported heads at x, balancing longitudinal bins and sides.

    Histogram peaks propose a pair; they are not point estimates at the window
    center when support is asymmetric. Retain the configured strip width; never
    force installation height or use another acquisition.
    """
    for _ in range(2):
        representatives, sides = [], []
        for side in (-1, 1):
            residual = q[:, 1] - (center + slope * (q[:, 0] - x) + side * gauge / 2)
            head = q[np.abs(residual) < cfg["rail_half_width_m"] / 2]
            if len(head) < 2 or np.ptp(head[:, 0]) < cfg["rail_min_span_m"]:
                return None
            keys = np.floor(head[:, 0] / .30).astype(np.int64)
            for key in np.unique(keys):
                representatives.append(np.median(head[keys == key, :2], axis=0))
                sides.append(side)
        a = np.asarray(representatives)
        design = np.column_stack((a[:, 0] - x, np.ones(len(a)), np.asarray(sides) / 2))
        fit, _, rank, _ = np.linalg.lstsq(design, a[:, 1], rcond=None)
        if rank < 3:
            return None
        slope, center, gauge = map(float, fit)
        if (abs(slope) > cfg["rail_max_heading"]
                or abs(gauge - cfg["rail_gauge_m"]) > cfg["rail_gauge_tolerance_m"]
                or np.max(np.abs(design @ fit - a[:, 1])) > cfg["rail_half_width_m"] / 2):
            return None
    return center, gauge, slope


class TrackGeometry:
    def __init__(self, points: np.ndarray, config: dict):
        self.config = config
        self.background = None
        self.plane, self.ground_quality = robust_plane(points, config)
        self.ground_anchors = np.empty((0, 3))
        self.rail_anchors = np.empty((0, 4))
        self.rail_head_height_m = None
        self.rail_support_diagnostics = []
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
        native = accelerator.native(cfg)
        if native is not None:
            anchors = accelerator.ground_profile(points, self.plane, cfg["ground_segment_m"], cfg["max_range_m"],
                                                 cfg["ground_local_window_m"], cfg["ground_fit_half_width_m"],
                                                 cfg["ground_inlier_m"], cfg["ground_min_support"],
                                                 cfg["ground_max_slopes"][0], native)
            self.ground_anchors = anchors.reshape(-1, 3)
            return
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
            if len(anchors) == 1 and cfg.get("rail_initial_heading", "zero") == "fitted":
                # The first fitted pair already measures a local tangent. Keep
                # it until two anchors define a secant; resetting to zero here
                # makes the next window search across an oblique rail pair.
                slope = self.rail_support_diagnostics[-1]["heading_slope"]
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
                    # A long window can see both rails entirely before/after x.
                    # Such support cannot reset extrapolation uncertainty at x.
                    if cfg.get("rail_anchor_support", "window") == "bracketed" and any(
                            not np.min(q[mask, 0]) <= x <= np.max(q[mask, 0])
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
            if cfg.get("rail_center_estimator", "histogram") == "paired_line":
                refined = refine_rail_pair(q, x, center, gauge, slope, cfg)
                if refined is not None and abs(refined[0] - expected) <= allowed:
                    center, gauge, slope = refined
                    lateral = q[:, 1] - slope * (q[:, 0] - x)
            rail_mask = np.abs(np.abs(lateral - center) - gauge / 2) < cfg["rail_half_width_m"] / 2
            anchor_x = float(x)
            if cfg.get("rail_anchor_support", "window") == "measured":
                heads = [q[np.abs(lateral - center - side * gauge / 2) < cfg["rail_half_width_m"] / 2, 0]
                         for side in (-1, 1)]
                if any(len(head) < 2 for head in heads):
                    continue
                support_low = max(head.min() for head in heads)
                support_high = min(head.max() for head in heads)
                if support_low > support_high:
                    continue
                # Locate the anchor within BOTH measured longitudinal hulls.
                # Keep the same fitted lines; do not throw away an oblique pair
                # merely because the arbitrary window center has no support.
                anchor_x = float(np.clip(x, support_low, support_high))
                if anchors and anchor_x <= anchors[-1][0]:
                    continue
                center += slope * (anchor_x - x)
                lateral = q[:, 1] - slope * (q[:, 0] - anchor_x)
                rail_mask = np.abs(np.abs(lateral - center) - gauge / 2) < cfg["rail_half_width_m"] / 2
            if cfg.get("rail_anchor_support", "window") == "bracketed":
                heads = [q[np.abs(lateral - center - side * gauge / 2) < cfg["rail_half_width_m"] / 2, 0]
                         for side in (-1, 1)]
                if any(len(head) < 2 or not head.min() <= x <= head.max() for head in heads):
                    continue
            bed, _ = self.ground(q[rail_mask])
            head_heights.append(float(np.quantile(q[rail_mask, 2] - bed, 0.8)))
            heads = [q[np.abs(lateral - center - side * gauge / 2) < cfg["rail_half_width_m"] / 2, 0]
                     for side in (-1, 1)]
            self.rail_support_diagnostics.append({"x_m": anchor_x, "window_center_m": float(x), "heading_slope": float(slope),
                "side_longitudinal_ranges_m": [[float(head.min()), float(head.max())] if len(head) else None
                                                for head in heads],
                "bracketed": bool(all(len(head) and head.min() <= anchor_x <= head.max() for head in heads))})
            anchors.append((anchor_x, center, gauge, support))
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
                eligible = np.flatnonzero(context & ~((observed & nominal_overlap) | boundary))
                if len(eligible):
                    context[eligible] &= ~self.background.mask(points[eligible], np.zeros(len(eligible), dtype=bool))
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
        # Remove only the measured rail-head band, not all points near a rail.
        on_rail = (observed & (np.abs(np.abs(lateral) - gauge / 2) < cfg["rail_half_width_m"] + path_uncertainty)
                   & (running_height <= cfg["rail_vertical_margin_m"]))
        # Propagate existing bed-height error through BOTH coordinates. A height
        # error can also cross a step in the reference contour's half-width.
        # Marginal bounds discard correlation and are conservative: they can
        # retain extra unresolved evidence, never certify a marginal intrusion.
        bed_error = np.where(observed, ground_uncertainty, 0.)
        height_error = bed_error / normal_scale
        low, high = running_height - height_error, running_height + height_error
        min_width, max_width = envelope_width_bounds(low, high, envelope, cfg["envelope_margin_m"])
        lateral_uncertainty = (np.where(observed, path_uncertainty, 0.) * np.sqrt(1 + slope**2)
                               + abs(slope) * bed_error / np.sqrt(1 + slope**2))
        vertical_inside = (low >= envelope[0, 0]) & (high <= envelope[-1, 1])
        core = observed & ~on_rail & vertical_inside & (np.abs(lateral) + lateral_uncertainty <= min_width)
        possible = ((high >= envelope[0, 0]) & (low <= envelope[-1, 1])
                    & (np.abs(lateral) - lateral_uncertainty <= max_width))
        boundary = observed & ~on_rail & ~core & possible
        # Segmentation precedes the collision gate. Do not amputate the feet or
        # head of an object just because only part intersects the envelope.
        segmentation_height = (running_height >= cfg["min_running_height_m"]) & (running_height <= envelope[-1, 1] + cfg["cluster_context_margin_m"])
        context = segmentation_height & ~on_rail & (np.abs(lateral) <= cfg["segmentation_context_half_width_m"])
        nominal_overlap = ((running_height >= envelope[0, 0]) & (running_height <= envelope[-1, 1])
                           & ~on_rail & (np.abs(lateral) <= width))
        if remove_background and self.background is not None:
            # Background decisions are consumed only for segmentation context;
            # protected points cannot be removed. Each mask decision depends on
            # the frozen surface model, not on other query points, so avoid the
            # expensive nearest-normal search for all unused/protected returns.
            eligible = np.flatnonzero(context & ~((observed & nominal_overlap) | boundary))
            if len(eligible):
                context[eligible] &= ~self.background.mask(points[eligible], np.zeros(len(eligible), dtype=bool))
        result = (core, context, height, observed, nominal_overlap)
        return result + (boundary,) if include_boundary else result

    def describe(self) -> dict:
        return {"valid": self.valid, "reason": self.reason, "ground_quality": self.ground_quality,
                "lateral_boundary_policy": "heuristic_path_and_ground_interval",
                "boundary_uncertainty_scope": "path_center_and_bed_height_only_not_full_extrinsics",
                "ground_plane": None if self.plane is None else self.plane.tolist(),
                "rail_center_estimator": self.config.get("rail_center_estimator", "histogram"),
                "rail_anchor_support": self.config.get("rail_anchor_support", "window"),
                "rail_support_diagnostics": self.rail_support_diagnostics,
                "rail_head_height_m": self.rail_head_height_m,
                "ground_anchors": self.ground_anchors.tolist(), "rail_anchors": self.rail_anchors.tolist(),
                "background": None if self.background is None else self.background.describe()}
