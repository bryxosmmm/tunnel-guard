"""Locally supported track-bed and paired-rail geometry, never an empty-scan map.

Region-wise ground modelling follows the motivation of Himmelsbach et al. (IV
2010); this is a new rail-specific implementation, not a paper reproduction.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks

from . import accelerator


class CrossSection(NamedTuple):
    """Rail-relative coordinates of a set of returns, in the decision frame.

    `lateral` is signed and measured from the estimated centre-line; `running_height` is
    above the measured running surface (the rail heads). These are the two quantities the
    reference-contour test consumes, so they are produced in one place, reported as-is, and
    never recomputed for a decision that depends on them.
    """
    lateral: np.ndarray
    running_height: np.ndarray
    gauge: np.ndarray


def voxel_representatives(points: np.ndarray, size: float) -> np.ndarray:
    return points[voxel_representative_indices(points, size)]



def voxel_representative_indices(points: np.ndarray, size: float) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=np.int64)
    native = accelerator.native({})
    if points.dtype != np.float64:
        raise ValueError("C++ voxel kernel requires float64 measurements")
    return np.frombuffer(native.voxel_indices(np.ascontiguousarray(points), size), dtype=np.int64)



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


def robust_plane(points: np.ndarray, config: dict) -> tuple[np.ndarray | None, dict]:
    lo, hi = config["ground_fit_range_m"]
    min_height, max_height = config["ground_sensor_height_bounds_m"]
    mask = ((points[:, 0] >= lo) & (points[:, 0] <= hi)
            & (np.abs(points[:, 1]) < config["ground_fit_half_width_m"])
            & (points[:, 2] > -max_height) & (points[:, 2] < -min_height))
    sample = voxel_representatives(points[mask], max(config["geometry_voxel_m"], 0.12))
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
        self.rail_rejections = []
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
        if self.plane is None:
            self.ground_anchors = np.empty((0, 3), dtype=float)
            return
        cfg = self.config
        native = accelerator.native(cfg)
        self.ground_anchors = accelerator.ground_profile(
            points, self.plane, cfg["ground_segment_m"], cfg["max_range_m"],
            cfg["ground_local_window_m"], cfg["ground_fit_half_width_m"], cfg["ground_inlier_m"],
            cfg["ground_min_support"], cfg["ground_max_slopes"][0], native).reshape(-1, 3)


    def ground(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.plane is None or not len(self.ground_anchors):
            return np.full(len(points), np.nan), np.full(len(points), np.inf)
        return accelerator.ground_values(points, self.plane, self.ground_anchors,
                                         self.config["ground_max_extrapolation_m"],
                                         accelerator.native(self.config))


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
            if anchors and cfg.get("rail_pair_continuity", "window") == "relocated":
                dx = anchor_x - anchors[-1][0]
                # Recheck at the actual measurement location, not the window
                # centre. A pair behind the window must not inherit its larger
                # lateral search allowance. Use the existing heading bound.
                if dx <= 0 or abs(center - anchors[-1][1]) > cfg["rail_max_heading"] * dx:
                    self.rail_rejections.append({"window_x_m": float(x), "anchor_x_m": anchor_x,
                                                "reason": "relocated_heading_exceeds_bound"})
                    continue
            if anchors and cfg.get("rail_anchor_continuity"):
                # The heading bound above is a bound on RATE, not on the sequence: at the far edge of the rail
                # support a pair can satisfy it and still sit a metre off the trend the previous anchors
                # describe, because a metre of lateral error spread over ten metres is under 6.9 degrees. Such
                # an anchor is the edge the continuation is fitted from, and one of them sets the SIGN of the
                # fitted curvature: measured on roundT_squareT_pressureGate_squareT frame 350, the pair accepted
                # at x = 50.04 m had gauge 1.428 m (9.2 cm off the frame's own anchors) and its centre sat
                # 1.39 m below the trend of the three before it, and the corridor then left the tunnel by
                # -26 m at 120 m while the tunnel curved the other way. The guard compares the anchor with its
                # OWN recent history rather than with the nominal gauge, because the fitted gauge of a whole
                # recording is biased (1.56-1.65 m across these recordings), so an absolute tolerance throws
                # away good anchors with the bad one - measured: 0.05 m tolerance cut a frame from 9 anchors
                # to 4 and its supported range from 62.5 m to 47.5 m.
                plan = cfg["rail_anchor_continuity"]
                recent = anchors[-int(plan.get("history", 4)):]
                recent_x = np.asarray([entry[0] for entry in recent])
                recent_y = np.asarray([entry[1] for entry in recent])
                gauge_median = float(np.median([entry[2] for entry in recent]))
                slope_fit = float(np.polyfit(recent_x, recent_y, 1)[0]) if len(recent) > 1 and np.ptp(recent_x) > 0 else 0.0
                ahead = anchor_x - recent_x[-1]
                trend_centre = float(recent_y[-1] + slope_fit * ahead)
                gauge_deviation = abs(gauge - gauge_median)
                centre_deviation = abs(center - trend_centre)
                if (gauge_deviation > float(plan["gauge_m"])
                        or centre_deviation > float(plan["centre_m"]) + float(plan["centre_slope"]) * ahead):
                    self.rail_rejections.append({"window_x_m": float(x), "anchor_x_m": anchor_x,
                        "reason": "anchor_inconsistent_with_recent_anchors",
                        "gauge_m": float(gauge), "recent_gauge_median_m": gauge_median,
                        "gauge_deviation_m": float(gauge_deviation), "centre_m": float(center),
                        "trend_centre_m": trend_centre, "centre_deviation_m": float(centre_deviation)})
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
        self._drop_inconsistent_trailing_anchors(anchors)
        self.rail_anchors = np.asarray(anchors, dtype=float).reshape(-1, 4)
        if head_heights:
            self.rail_head_height_m = float(np.median(head_heights))

    def _drop_inconsistent_trailing_anchors(self, anchors: list) -> None:
        """Remove trailing anchors that contradict the prefix they are fitted from.

        The continuation past the last anchor is fitted in that anchor's own frame, so a trailing anchor
        that disagrees with the anchors before it decides the SIGN of the extrapolated curvature. Only
        trailing anchors are examined here, on purpose: an anchor in the middle of the span also moves
        the interpolated corridor, and rejecting those changes decisions inside measured support as well
        as beyond it. Measured on roundT_squareT_pressureGate_squareT frame 350, the trailing pair had
        support 4 (the admissible floor), gauge 1.428 m against the frame's own 1.59 m, and its centre
        stood 1.39 m below the trend of the three anchors before it; the corridor then left the tunnel by
        -26 m at 120 m while the tunnel curved towards +y.
        """
        plan = self.config.get("rail_anchor_edge_continuity")
        if not plan:
            return
        while len(anchors) > 2:
            history = anchors[-1 - int(plan.get("history", 4)):-1]
            if len(history) < 2:
                return
            xs = np.asarray([entry[0] for entry in history], dtype=float)
            ys = np.asarray([entry[1] for entry in history], dtype=float)
            if np.ptp(xs) <= 0:
                return
            ahead = float(anchors[-1][0] - xs[-1])
            trend = float(ys[-1] + np.polyfit(xs, ys, 1)[0] * ahead)
            deviation = abs(float(anchors[-1][1]) - trend)
            if deviation <= float(plan["centre_m"]) + float(plan["centre_slope"]) * ahead:
                return
            rejected = anchors.pop()
            self.rail_rejections.append({"anchor_x_m": float(rejected[0]),
                "reason": "trailing_anchor_inconsistent_with_its_prefix",
                "centre_m": float(rejected[1]), "trend_centre_m": trend,
                "centre_deviation_m": deviation, "gauge_m": float(rejected[2]),
                "support": int(rejected[3])})




    def _continuation(self, edge: int) -> tuple[float, float, float, float, float, float, float]:
        """Local quadratic continuation of the measured centre-line past an anchor edge.

        Returns the edge point, the continuation slope and curvature, and the fit's own
        uncertainty: the two standard errors and their covariance. The covariance is not
        decoration: the fit is one-sided, so slope and curvature are strongly correlated
        there, and the prediction variance d^2 Var(a) + d^4 Var(b) + 2 d^3 Cov(a,b) loses
        exactly the term that grows fastest with distance if the cross term is dropped.

        The rails are measured over roughly 40 m, which is too short to resolve the alignment by
        a straight tangent: on a curve the true centre-line leaves it quadratically, and at 60 m
        that departure (2-4 m on R = 500-1000 m) dwarfs the 1.5 m corridor half-width. The
        continuation is therefore fitted, not assumed:

        * the fit is expressed in the edge anchor's own frame (t = (x - x_edge)/window,
          v = y - y_edge), so the curve passes through the measured edge point and ``center(x)``
          stays continuous where it replaces the interpolation;
        * the curvature is shrunk to zero unless the window supports it (|b| beyond its own
          standard error), because a short lever cannot distinguish a gentle curve from noise and
          inventing one would bend a straight tunnel;
        * the fit's own covariance is returned so the caller can propagate the extrapolation
          error instead of inventing an uncertainty.

        Returns ``(x_edge, y_edge, slope, curvature, slope_sigma, curvature_sigma, covariance)``.
        """
        cfg = self.config
        a = self.rail_anchors
        window = float(cfg.get("path_curve_window_m", 30.0))
        if edge == 0:
            selected = a[a[:, 0] <= a[0, 0] + window]
            x_edge, y_edge = float(selected[0, 0]), float(selected[0, 1])
        else:
            selected = a[a[:, 0] >= a[-1, 0] - window]
            x_edge, y_edge = float(selected[-1, 0]), float(selected[-1, 1])
        if len(selected) < 3 or window <= 0:
            if len(a) >= 2:
                other = 1 if edge == 0 else -2
                raw = (a[edge, 1] - a[other, 1]) / (a[edge, 0] - a[other, 0])
            else:
                raw = 0.0
            return x_edge, y_edge, float(np.clip(raw, -cfg["rail_max_heading"], cfg["rail_max_heading"])), 0.0, 0.0, 0.0, 0.0
        t = (selected[:, 0] - x_edge) / window
        v = selected[:, 1] - y_edge
        s11 = s12 = s22 = b1 = b2 = 0.0
        for t_i, v_i in zip(t, v):                        # same summation order as the C++ kernel
            tt, ttt, tttt = t_i * t_i, t_i * t_i * t_i, t_i * t_i * t_i * t_i
            s11 += tt
            s12 += ttt
            s22 += tttt
            b1 += t_i * v_i
            b2 += tt * v_i
        determinant = s11 * s22 - s12 * s12
        if determinant <= 0.0:
            return x_edge, y_edge, 0.0, 0.0, 0.0, 0.0, 0.0
        slope_scaled = (b1 * s22 - b2 * s12) / determinant
        curvature_scaled = (s11 * b2 - s12 * b1) / determinant
        residual_square = 0.0
        for t_i, v_i in zip(t, v):
            residual = v_i - (slope_scaled * t_i + curvature_scaled * t_i * t_i)
            residual_square += residual * residual
        dof = len(selected) - 2
        variance = residual_square / dof if dof > 0 else 0.0
        slope_sigma = np.sqrt(max(variance * s22 / determinant, 0.0)) / window
        curvature_sigma = np.sqrt(max(variance * s11 / determinant, 0.0)) / (window * window)
        # Off-diagonal of the same inverse: Cov(a, b) = -s^2 s12 / (det * window^3) in physical units.
        covariance = -variance * s12 / (determinant * window ** 3)
        if abs(curvature_scaled) < cfg.get("path_curvature_significance", 4.0) * curvature_sigma * window * window:
            curvature_scaled = 0.0
        slope = float(np.clip(slope_scaled / window, -cfg["rail_max_heading"], cfg["rail_max_heading"]))
        return (x_edge, y_edge, slope, float(curvature_scaled / (window * window)),
                float(slope_sigma), float(curvature_sigma), float(covariance))

    def path(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.rail_anchors) < 2:
            return np.zeros(len(x)), np.full(len(x), self.config["rail_gauge_m"]), np.full(len(x), np.inf)
        cfg = self.config
        a = self.rail_anchors
        center = np.interp(x, a[:, 0], a[:, 1])
        nearest = nearest_anchor_distance(x, a[:, 0])
        uncertainty = 0.06 + 0.008 * nearest + 0.0003 * nearest**2
        # Beyond the measured anchors, continue along the fitted local curvature instead of a
        # straight tangent, and propagate that fit's error into the path uncertainty. Where the
        # anchors do not support a curvature the fit returns zero and this reduces to the
        # previous straight continuation.
        for edge, mask in [(0, x < a[0, 0]), (-1, x > a[-1, 0])]:
            if not mask.any():
                continue
            (x_edge, y_edge, slope, curvature, slope_sigma, curvature_sigma,
             covariance) = self._continuation(edge)
            distance = x[mask] - x_edge
            center[mask] = y_edge + slope * distance + curvature * distance * distance
            # Full prediction variance: d^2 Var(a) + d^4 Var(b) + 2 d^3 Cov(a, b). The fit is
            # one-sided, so the cross term is positive and grows faster than either diagonal
            # term; dropping it understated the continuation error where it decides the
            # corridor. Clamped because round-off can make a PSD form marginally negative.
            variance = (distance * slope_sigma) ** 2 + (distance ** 2 * curvature_sigma) ** 2 \
                + 2.0 * distance ** 3 * covariance
            extension = np.sqrt(np.maximum(variance, 0.0))
            uncertainty[mask] = np.sqrt(uncertainty[mask] ** 2 + extension**2)
        uncertainty[nearest > cfg["path_max_extrapolation_m"]] = np.inf
        return center, np.interp(x, a[:, 0], a[:, 2]), uncertainty



    def reference_half_width(self, running_height: np.ndarray) -> np.ndarray:
        """Half-width of the reference contour at a height above the running surface.

        Heights outside the contour's vertical span take the nearest edge's half-width, so
        a caller comparing a return to the contour's lateral reach always has a value.
        """
        envelope = np.asarray(self.config["envelope_segments_m"], dtype=float)
        row = np.clip(np.searchsorted(envelope[:, 0], running_height, side="right") - 1, 0, len(envelope) - 1)
        segment = envelope[row]
        fraction = np.clip((running_height - segment[:, 0]) / (segment[:, 1] - segment[:, 0]), 0, 1)
        return segment[:, 2] + fraction * (segment[:, 3] - segment[:, 2])

    def structural_mask(self, points: np.ndarray, section: "CrossSection | None" = None) -> np.ndarray:
        """Returns that belong to the tunnel's own cross-section rather than to an object.

        Two necessary conditions, both measured from this scan alone, asked at the resolution
        each one needs:

        * longitudinal support, on a coarse cell (`structure_support_cell_m`): the occupied
          stations of that cell span at least `structure_min_length_m` and number at least
          `structure_min_stations`. A SPAN rather than a run of consecutive stations: a wall
          seen at grazing incidence at 55 m lands in the same cell at eight stations spread
          over fifty metres, and a run would call that sparse while calling a 3 m object
          continuous. Coarse, because a surface only holds a cell for metres if the cell is
          at least as wide as the error in its own coordinates: measured, a fixed platform
          edge moves 0.1-0.3 m in lateral across one station as the estimated centre drifts.
          Two grids offset by half a cell are asked and either may answer, so a surface
          sitting on a cell boundary is not lost.
        * outward reach, on a fine cell (`structure_cell_m`): at that height the occupied
          region runs outward from the cell, past the reference contour's half-width, with no
          break longer than `structure_outward_gap_m`. Fine, because this is what separates a
          20 cm object standing inside the corridor from the walkway edge beside it - the gap
          between them is one cell - and thickened by one row, because a vertical face and
          the top surface it carries are adjacent rows of a 10 cm grid.

        This gates the envelope CLAIM only. A return is never removed from the object pool,
        so an object whose interior evidence is structural is still reported, with its
        evidence and this reason, and never silently disappears.
        """
        cfg = self.config
        if section is None:
            raise ValueError("structural_mask requires the C++ classifier cross-section")
        if not len(points):
            return np.zeros(0, dtype=bool)
        fine = float(cfg.get("structure_cell_m", 0.1))
        coarse = float(cfg.get("structure_support_cell_m", 0.4))
        station = float(cfg.get("structure_station_m", 1.0))
        x = points[:, 0]
        station_index = np.floor(x / station).astype(np.int64)
        support = np.zeros(len(points), dtype=bool)
        for offset in (0.0, coarse / 2.0):
            column = np.floor((section.lateral + offset) / coarse).astype(np.int64)
            row = np.floor((section.running_height + offset) / coarse).astype(np.int64)
            # One composite key sorts by cell and then by station in a single pass.
            order = np.argsort((column * 4096 + row) * 1024 + station_index, kind="stable")
            keys = ((column * 4096 + row) * 1024)[order]
            stations = station_index[order]
            breaks = np.r_[True, np.diff(keys) != 0]
            starts = np.flatnonzero(breaks)
            stops = np.r_[starts[1:], len(order)]
            cell_of = np.repeat(np.arange(len(starts)), stops - starts)
            # A cell is longitudinally supported when its occupied stations SPAN at least
            # `structure_min_length_m` and number at least `structure_min_stations`. Span, not
            # a run: a wall seen at grazing incidence at 55 m lands in the same cell at eight
            # stations spread over fifty metres, and a run of consecutive stations would call
            # that sparse and a 3 m object continuous, which is backwards.
            span = (stations[stops - 1][cell_of] - stations[starts][cell_of] + 1) * station
            count = (stops - starts)[cell_of]
            per_cell = np.zeros(len(starts), dtype=bool)
            np.logical_or.at(per_cell, cell_of,
                             (span >= float(cfg.get("structure_min_length_m", 5.0)))
                             & (count >= int(cfg.get("structure_min_stations", 3))))
            support[order] = per_cell[cell_of]
        # Outward reach, on a cross-section thickened by one row and by the reach gap: a
        # vertical face is joined to the top surface it carries, and a surface whose inner
        # edge stands a few centimetres off the rest of its own structure at that height is
        # still one surface. Widening the reach cannot merge an object with the structure
        # beside it - that is what the longitudinal support above is for, and it is asked on
        # the raw cells, not on this grid.
        column = np.floor(section.lateral / fine).astype(np.int64)
        row = np.floor(section.running_height / fine).astype(np.int64)
        column_min, column_max = int(column.min()), int(column.max())
        row_min, row_max = int(row.min()), int(row.max())
        width, height = column_max - column_min + 1, row_max - row_min + 2
        thick = np.zeros((width, height), dtype=bool)
        thick[column - column_min, row - row_min + 1] = True
        thick[:, 1:] |= thick[:, :-1].copy()
        thick[:, :-1] |= thick[:, 1:].copy()
        gap = int(np.ceil(float(cfg.get("structure_outward_gap_m", 0.3)) / fine))
        for _ in range(gap):
            thick[1:, :] |= thick[:-1, :].copy()
            thick[:-1, :] |= thick[1:, :].copy()
        limit = self.reference_half_width((np.arange(row_min - 1, row_max + 1) + 0.5) * fine) \
            + float(cfg.get("structure_outward_margin_m", 0.2))
        positions = np.arange(width)
        outward = np.zeros((width, height), dtype=bool)
        for index in range(height):
            here = thick[:, index]
            empty = np.flatnonzero(~here)
            if not len(empty):
                outward[:, index] = True
                continue
            # A run that reaches an edge of the grid has no empty cell on that side, and the
            # edge of the grid is where it ends. Clamping the search to the nearest empty cell
            # instead returned an empty cell on the WRONG side, which reported a short run for
            # the outermost surface of every frame.
            after = np.searchsorted(empty, positions, side="left")
            next_empty = np.where(after < len(empty), empty[np.minimum(after, len(empty) - 1)], width)
            before = np.searchsorted(empty, positions, side="right") - 1
            previous_empty = np.where(before >= 0, empty[np.maximum(before, 0)], -1)
            right = (np.where(here, next_empty - 1, -1) + column_min + 1) * fine
            left = (np.where(here, previous_empty + 1, 0) + column_min) * fine
            absolute = positions + column_min
            outward[:, index] = here & (np.where(absolute >= 0, right, -left) >= limit[index])
        return support & outward[column - column_min, row - row_min + 1]

    def classify(self, points: np.ndarray, *, remove_background: bool = True,
                 include_boundary: bool = False, ground: tuple | None = None):
        """Classify support; optionally expose uncertain envelope intersections.

        The interval uses the existing heuristic path uncertainty, not calibrated
        probability or a guarantee about the physical vehicle envelope. A caller
        that already fitted the bed for the same array may pass it in; the
        returned values are those of the identical fit.
        """
        masks, _ = self.classify_with_section(points, remove_background=remove_background,
                                             include_boundary=include_boundary, ground=ground)
        return masks

    def classify_with_section(self, points: np.ndarray, *, remove_background: bool = True,
                              include_boundary: bool = False, ground: tuple | None = None):
        if self.config.get("rail_frame_mode", "bed") != "bed":
            raise ValueError("rail_frame_mode must be bed; only the C++ bed classifier is supported")
        native = accelerator.classify_geometry(points, self, accelerator.native(self.config))
        core, context, height, observed, nominal_overlap, boundary, lateral, running, gauge = native
        if remove_background and self.background is not None:
            eligible = np.flatnonzero(context & ~((observed & nominal_overlap) | boundary))
            if len(eligible):
                context[eligible] &= ~self.background.mask(points[eligible], np.zeros(len(eligible), dtype=bool))
        masks = (core, context, height, observed, nominal_overlap, boundary) if include_boundary else (
            core, context, height, observed, nominal_overlap)
        return masks, CrossSection(lateral, running, gauge)


    def supported_range_m(self) -> float | None:
        """How far ahead this frame's own evidence supports the corridor, in metres.

        The value describes the evidence, not a sensor specification, and it does not certify that
        the corridor is clear: it states how far the reported centre-line and bed can be read from
        what was measured here. A station counts as supported only when the path uncertainty is
        within `path_max_uncertainty_m`, the bed uncertainty is within `ground_max_uncertainty_m`,
        The two uncertainties grow with distance from their
        nearest measured anchor, so support ends where either policy stops holding.

        The reported value is the far end of the CONTIGUOUS run that starts at the first supported
        station and ends at the last supported station before the first unsupported one: an island
        of support beyond a gap does not extend it, since a consumer reading that island would
        still have to cross the gap. Evaluated on a 0.5 m grid from `min_forward_m`, reporting only.
        """
        if not self.valid:
            return None
        grid = np.arange(float(self.config["min_forward_m"]), float(self.config["max_range_m"]), 0.5)
        if not len(grid):
            return None
        stations = np.column_stack((grid, np.zeros(len(grid)), np.zeros(len(grid))))
        _, bed_uncertainty = self.ground(stations)
        _, _, path_uncertainty = self.path(grid)
        supported = (np.isfinite(path_uncertainty) & (path_uncertainty <= self.config["path_max_uncertainty_m"])
                     & np.isfinite(bed_uncertainty) & (bed_uncertainty <= self.config["ground_max_uncertainty_m"]))
        run = np.flatnonzero(supported)
        if not len(run):
            return 0.0
        first = int(run[0])
        gap = np.flatnonzero(~supported[first:])
        last = first + int(gap[0]) - 1 if len(gap) else len(grid) - 1
        return float(grid[last])

    def describe(self) -> dict:
        return {"valid": self.valid, "reason": self.reason, "ground_quality": self.ground_quality,
                "lateral_boundary_policy": "heuristic_path_and_ground_interval",
                "boundary_uncertainty_scope": "path_center_and_bed_height_only_not_full_extrinsics",
                "ground_plane": None if self.plane is None else self.plane.tolist(),
                "rail_center_estimator": self.config.get("rail_center_estimator", "histogram"),
                "rail_anchor_support": self.config.get("rail_anchor_support", "window"),
                "rail_support_diagnostics": self.rail_support_diagnostics,
                "rail_frame_mode": self.config.get("rail_frame_mode", "bed"),
                "rail_rejections": self.rail_rejections,
                "rail_head_height_m": self.rail_head_height_m,
                "ground_anchors": self.ground_anchors.tolist(), "rail_anchors": self.rail_anchors.tolist(),
                "background": None if self.background is None else self.background.describe()}
