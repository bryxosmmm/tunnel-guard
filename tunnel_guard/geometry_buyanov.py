"""Locally supported track-bed and paired-rail geometry, never an empty-scan map.

Region-wise ground modelling follows the motivation of Himmelsbach et al. (IV
2010); this is a new rail-specific implementation, not a paper reproduction.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from .background import TunnelBackground

# Geometry knobs introduced 2026-09-16 with the estimator repairs. Shipped configs
# state them explicitly; historical snapshots predate them and fall back here so an
# old recipe still replays under the current estimator.
GEOMETRY_DEFAULTS = {
    # Off: model-verified longitudinal extension extended the certified horizon
    # (synthetic 83->120 m, real 63->70-80 m) but cost detection of thin objects
    # at range on the synthetic panels (dev event recall 0.9028 -> 0.8889; a 100 m
    # mini-panel 4/6 -> 2/6). A thin far cluster can no longer be reported: with
    # geometry observed it is neither 'intersecting' (too few measured points above
    # the lower contour bound) nor 'unresolved' (that class requires ~observed).
    # Enable only once that acceptance gap is fixed and the panels hold.
    "longitudinal_extension_enabled": False,
    "ground_extension_model_anchors": 5,
    "ground_extension_max_offset_m": 0.06,
    "ground_extension_misses_max": 2,
    # Beyond ~70 m a 12 m longitudinal window holds too few beam rows to verify a
    # bed patch, so the extension window grows with range instead of lowering the
    # evidence floor.
    "ground_extension_window_growth": 0.5,
    "ground_extension_max_window_m": 60.0,
    "ground_extension_min_points": 8,
    "rail_extension_min_points": 4,
    "rail_extension_max_offset_m": 0.12,
    "rail_extension_misses_max": 3,
    "rail_extension_window_growth": 0.6,
    "rail_extension_max_window_m": 60.0,
    "rail_model_anchors": 8,
    "rail_head_band_m": [-0.10, 0.06],
    # Declared rail profile width. The head's outer face is never returned (the
    # beam only sees the inner face and the top), so rail centres are placed from
    # the measured inner faces plus this assumed width; gauge itself stays measured.
    "rail_head_width_m": 0.07,
    # Isolation band for measuring one rail head. rail_half_width_m is the wider
    # removal band; using it here swallows check rails and fastenings at switches.
    "rail_band_margin_m": 0.02,
    "rail_support_symmetry_min": 0.4,
    "path_sigma_floor_m": 0.06,
    "path_growth_bounds_m_per_m": [0.004, 0.02],
}


def setting(config: dict, name: str):
    return config[name] if name in config else GEOMETRY_DEFAULTS[name]


def voxel_representatives(points: np.ndarray, size: float) -> np.ndarray:
    """First point of each occupied voxel, in np.unique(axis=0) row order."""
    if not len(points):
        return points
    keys = np.floor(points / size).astype(np.int64)
    # One stable lexsort replaces unique's per-column passes; ties keep the
    # earliest input point, which is the representative np.unique would return.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    np.any(ordered[1:] != ordered[:-1], axis=1, out=first[1:])
    return points[order[first]]


def nearest_anchor_distance(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Exact distance to the closest anchor for an increasing positions array."""
    index = np.searchsorted(positions, x)
    lower = positions[np.clip(index - 1, 0, len(positions) - 1)]
    upper = positions[np.clip(index, 0, len(positions) - 1)]
    return np.minimum(np.abs(x - lower), np.abs(x - upper))


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
        self.rail_head_anchors = np.empty((0, 3))
        self.rail_fit_diagnostics = []
        self.rail_sigma = np.empty(0)
        self.rail_gauge_inner = np.empty(0)
        self.gauge_convention_conflict = False
        self.path_growth_m_per_m = None
        self.path_horizon_m = None
        self.rail_head_height_m = None
        self.reason = self.ground_quality["reason"]
        if self.plane is None:
            return
        self._ground_profile(points)
        self._rail_profile(points)
        self.path_horizon_m = self._path_horizon()
        if len(self.rail_anchors) < 2:
            self.reason = "insufficient_paired_rail_support"
        elif not self._head_plausible():
            # A bed plane that locked onto a lower surface (walkway, duct floor)
            # still yields paired rails, just at an impossible head height.
            self.reason = "implausible_rail_head_above_bed"
        else:
            self.reason = "supported_geometry"
        if self.valid and config["background"]["enabled"]:
            self.background = TunnelBackground(points, self, config)

    @property
    def valid(self) -> bool:
        return (self.plane is not None and len(self.rail_anchors) >= 2
                and self._head_plausible())

    def _head_plausible(self) -> bool:
        if self.rail_head_height_m is None:
            return False
        lo, hi = self.config["rail_height_bounds_m"]
        return bool(lo <= self.rail_head_height_m <= hi)

    def _path_horizon(self) -> float | None:
        """End of the first interval passing the heuristic lateral-path bound.

        This excludes ground uncertainty, visibility and swept-envelope validation;
        it is neither a certified clearance horizon nor a detection-range claim.
        """
        if len(self.rail_anchors) < 2:
            return None
        x = np.arange(self.config["min_forward_m"], self.config["max_range_m"], 0.5)
        _, _, uncertainty = self.path(x)
        inside = np.isfinite(uncertainty) & (uncertainty <= self.config["path_max_uncertainty_m"])
        supported = np.flatnonzero(inside)
        if not len(supported):
            return None
        breaks = np.flatnonzero(~inside)
        after = breaks[breaks > supported[0]]
        end = int(after[0]) - 1 if len(after) else len(x) - 1
        return float(x[end])

    def _ground_profile(self, points: np.ndarray):
        cfg = self.config
        half_width = cfg["ground_fit_half_width_m"]
        residual = points[:, 2] - (points[:, :2] @ self.plane[:2] + self.plane[2])
        anchors = []
        previous = 0.0
        for x in np.arange(cfg["ground_segment_m"], cfg["max_range_m"], cfg["ground_segment_m"]):
            mask = ((np.abs(points[:, 0] - x) < cfg["ground_local_window_m"] / 2)
                    & (np.abs(points[:, 1]) < half_width)
                    & (np.abs(residual - previous) < cfg["ground_inlier_m"] * 2))
            values = residual[mask]
            if len(values) < max(12, cfg["ground_min_support"] // 3):
                continue
            support_points = points[mask]
            if np.ptp(support_points[:, 0]) < 1.5 or np.ptp(support_points[:, 1]) < 0.4:
                continue
            shift = float(np.median(values))
            mad = float(np.median(np.abs(values - shift)))
            if anchors and abs(shift - previous) / (x - anchors[-1][0]) > cfg["ground_max_slopes"][0]:
                continue
            anchors.append((float(x), shift, max(mad * 1.4826, 0.015)))
            previous = shift
        if setting(cfg, "longitudinal_extension_enabled"):
            anchors.extend(self._ground_extension(points, residual, anchors, half_width))
        self.ground_anchors = np.asarray(anchors, dtype=float).reshape(-1, 3)

    def _ground_extension(self, points: np.ndarray, residual: np.ndarray, anchors: list,
                          half_width: float):
        """Continue the bed profile beyond the last detection by predicting each
        window from the accepted trend and accepting only what verifies it. The
        rail band search needs a bed datum at range, so the bed must lead."""
        cfg = self.config
        if len(anchors) < 3:
            return []
        model_anchors = setting(cfg, "ground_extension_model_anchors")
        max_offset = setting(cfg, "ground_extension_max_offset_m")
        window_growth = setting(cfg, "ground_extension_window_growth")
        window_max = setting(cfg, "ground_extension_max_window_m")
        min_points = setting(cfg, "ground_extension_min_points")
        accepted, misses = [], 0
        xs = [a[0] for a in anchors]
        shifts = [a[1] for a in anchors]
        for x in np.arange(anchors[-1][0] + cfg["ground_segment_m"], cfg["max_range_m"], cfg["ground_segment_m"]):
            recent = slice(max(0, len(xs) - model_anchors), None)
            slope, intercept = np.polyfit(np.asarray(xs)[recent], np.asarray(shifts)[recent], 1)
            predicted = float(slope * x + intercept)
            window = min(window_max, max(cfg["ground_local_window_m"], x * window_growth))
            mask = ((np.abs(points[:, 0] - x) < window / 2)
                    & (np.abs(points[:, 1]) < half_width)
                    & (np.abs(residual - predicted) < cfg["ground_inlier_m"] * 2))
            values = residual[mask]
            support_points = points[mask]
            verified = (len(values) >= min_points
                        and np.ptp(support_points[:, 0]) >= 1.5 and np.ptp(support_points[:, 1]) >= 0.4)
            shift = float(np.median(values)) if len(values) else np.nan
            verified = verified and abs(shift - predicted) <= max_offset
            if not verified:
                misses += 1
                if misses >= setting(cfg, "ground_extension_misses_max"):
                    break
                continue
            misses = 0
            mad = float(np.median(np.abs(values - shift)))
            accepted.append((float(x), shift, max(mad * 1.4826, 0.015)))
            xs.append(float(x))
            shifts.append(shift)
        return accepted

    def ground(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.plane is None or not len(self.ground_anchors):
            return np.full(len(points), np.nan), np.full(len(points), np.inf)
        x = points[:, 0]
        anchors = self.ground_anchors
        shift = np.interp(x, anchors[:, 0], anchors[:, 1])
        nearest = nearest_anchor_distance(x, anchors[:, 0])
        uncertainty = np.interp(x, anchors[:, 0], anchors[:, 2]) + nearest * 0.008
        uncertainty[nearest > self.config["ground_max_extrapolation_m"]] = np.inf
        z = points[:, :2] @ self.plane[:2] + self.plane[2] + shift
        return z, uncertainty

    def _rail_band_center(self, lateral: np.ndarray, seed: float, half_width: float):
        """Centre and half-span of one rail's return band.

        The lateral density mode sits on the rail side facing the sensor, so the
        mode only seeds a band; the reported position is the midpoint of the
        band's lateral extent, which is unbiased once the band covers the head.
        """
        band = np.abs(lateral - seed) <= half_width
        for _ in range(3):
            values = lateral[band]
            if len(values) < 2:
                return None, 0.0, 0
            low, high = np.percentile(values, [0.5, 99.5])
            band = np.abs(lateral - 0.5 * (low + high)) <= half_width
        values = lateral[band]
        if len(values) < 2:
            return None, 0.0, 0
        low, high = np.percentile(values, [0.5, 99.5])
        return float(0.5 * (low + high)), float(0.5 * (high - low)), int(len(values))

    def _rail_profile(self, points: np.ndarray):
        cfg = self.config
        z, uncertainty = self.ground(points)
        h = points[:, 2] - z
        lo, hi = cfg["rail_height_bounds_m"]
        eligible = ((h > lo) & (h < hi) & (np.abs(points[:, 1]) < 4)
                    & (uncertainty < cfg["ground_max_uncertainty_m"]))
        rail = points[eligible]
        rail_h = h[eligible]
        self.path_growth_m_per_m = None
        anchors, heads, sigmas, inner_gauges, gauges = [], [], [], [], []
        band_half = 0.5 * setting(cfg, "rail_head_width_m") + setting(cfg, "rail_band_margin_m")
        bin_size = cfg["rail_bin_m"]
        offset = int(np.ceil(4 / bin_size)) + 2
        for x in np.arange(5.0, cfg["max_range_m"], cfg["ground_segment_m"]):
            window = min(cfg["rail_max_window_m"], cfg["rail_window_m"] + x * cfg["rail_window_growth"])
            q = rail[np.abs(rail[:, 0] - x) < window / 2]
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
                    seed_gauge = (right - left) * bin_size
                    if abs(seed_gauge - cfg["rail_gauge_m"]) > (cfg["rail_gauge_tolerance_m"]
                                                                + 2 * cfg["rail_half_width_m"]):
                        continue
                    left_center, left_half, left_n = self._rail_band_center(
                        lateral, (left - offset + 0.5) * bin_size, band_half)
                    right_center, right_half, right_n = self._rail_band_center(
                        lateral, (right - offset + 0.5) * bin_size, band_half)
                    if left_center is None or right_center is None or min(left_n, right_n) < 2:
                        continue
                    # rail_gauge_m is the standard gauge: the distance between the
                    # inner faces, which is what the pairing test must check.
                    left_inner = left_center + left_half
                    right_inner = right_center - right_half
                    gauge_inner = right_inner - left_inner
                    head_half = 0.5 * setting(cfg, "rail_head_width_m")
                    left_rail, right_rail = left_inner - head_half, right_inner + head_half
                    center = 0.5 * (left_rail + right_rail)
                    # rail_gauge_m is a declared standard (inner faces). Whether the
                    # observed separation is consistent with it depends on which
                    # convention the data follows, so a pair is accepted under
                    # either reading and the conflict is reported rather than hidden.
                    gauge_center = right_rail - left_rail
                    disagreement = min(abs(gauge_inner - cfg["rail_gauge_m"]),
                                       abs(gauge_center - (cfg["rail_gauge_m"] + setting(cfg, "rail_head_width_m"))))
                    if disagreement > cfg["rail_gauge_tolerance_m"]:
                        continue
                    gauges.append((gauge_inner, gauge_center))
                    if abs(center - expected) > allowed:
                        continue
                    rails = (np.abs(lateral - left_center) <= cfg["rail_half_width_m"],
                             np.abs(lateral - right_center) <= cfg["rail_half_width_m"])
                    if any(np.count_nonzero(mask) < 2 or np.ptp(q[mask, 0]) < cfg["rail_min_span_m"]
                           for mask in rails):
                        continue
                    support = int(min(left_n, right_n))
                    score = support / (1 + 4 * abs(center - expected) + 8 * abs(gauge_inner - cfg["rail_gauge_m"]))
                    pairs.append((score, center, right_rail - left_rail, support, gauge_inner,
                                  left_rail, right_rail, left_half, right_half, disagreement))
            if not pairs:
                continue
            pairs.sort(reverse=True)
            if (len(pairs) > 1 and pairs[1][0] > 0.8 * pairs[0][0]
                    and abs(pairs[0][1] - pairs[1][1]) > 0.3):
                break
            (_, center, gauge_center, support, gauge_inner, left_center, right_center,
             left_half, right_half, disagreement) = pairs[0]
            rail_mask = np.abs(np.abs(lateral - center) - gauge_center / 2) < cfg["rail_half_width_m"]
            bed, _ = self.ground(q[rail_mask])
            # Diagnostic samples only; these do not feed fitting or acceptance.
            head_samples = {"x_m": float(x)}
            for side, position in (("left", left_center), ("right", right_center)):
                selected = np.abs(lateral - position) <= band_half
                side_points = q[selected]
                side_bed, _ = self.ground(side_points)
                head_samples[side] = {"points": len(side_points),
                    "height_above_bed_p80_m": float(np.quantile(side_points[:, 2] - side_bed, .8)) if len(side_points) else None,
                    "lateral_median_abs_residual_m": float(np.median(np.abs(lateral[selected] - position))) if len(side_points) else None}
            self.rail_fit_diagnostics.append(head_samples)
            anchors.append((float(x), center, gauge_center, support))
            heads.append(float(np.quantile(q[rail_mask, 2] - bed, 0.8)))
            sigmas.append(max(0.01, 0.25 * (left_half + right_half) / np.sqrt(max(1, support))))
            inner_gauges.append(gauge_inner)
        if len(anchors) >= 3 and setting(cfg, "longitudinal_extension_enabled"):
            self._rail_extension(rail, rail_h, anchors, heads, sigmas, inner_gauges)
        self.rail_anchors = np.asarray(anchors, dtype=float).reshape(-1, 4)
        self.rail_head_anchors = np.asarray([[a[0], head, sigma]
                                             for a, head, sigma in zip(anchors, heads, sigmas)],
                                            dtype=float).reshape(-1, 3)
        self.rail_sigma = np.asarray(sigmas, dtype=float)
        self.rail_gauge_inner = np.asarray(inner_gauges, dtype=float)
        # Flagged when the observed inner-face separation cannot be reconciled
        # with the declared standard gauge, so the pair was accepted on the
        # centre-to-centre reading instead. That means the declared gauge and the
        # observed rails disagree and a survey, not a threshold, must settle it.
        self.gauge_convention_conflict = bool(
            gauges and abs(float(np.median([g[0] for g in gauges])) - cfg["rail_gauge_m"])
            > cfg["rail_gauge_tolerance_m"])
        if heads:
            self.rail_head_height_m = float(np.median(heads))

    def _rail_extension(self, rail: np.ndarray, rail_h: np.ndarray, anchors: list, heads: list,
                        sigmas: list, inner_gauges: list):
        """Continue the alignment past the last detected anchor by predicting the
        rail pair from the accepted trend and accepting a window only when its
        returns verify that prediction. Sparse far-field returns can constrain a
        model even when no window alone supports independent peak detection."""
        cfg = self.config
        model_anchors = setting(cfg, "rail_model_anchors")
        low_band, high_band = setting(cfg, "rail_head_band_m")
        min_points = setting(cfg, "rail_extension_min_points")
        max_offset = setting(cfg, "rail_extension_max_offset_m")
        symmetry_min = setting(cfg, "rail_support_symmetry_min")
        window_growth = setting(cfg, "rail_extension_window_growth")
        window_max = setting(cfg, "rail_extension_max_window_m")
        low_growth, high_growth = setting(cfg, "path_growth_bounds_m_per_m")
        residuals, misses = [], 0
        for x in np.arange(anchors[-1][0] + cfg["ground_segment_m"], cfg["max_range_m"], cfg["ground_segment_m"]):
            recent = slice(max(0, len(anchors) - model_anchors), None)
            known_x = np.asarray([a[0] for a in anchors])[recent]
            known_c = np.asarray([a[1] for a in anchors])[recent]
            coefficients = np.polyfit(known_x, known_c, 2 if len(known_x) >= 4 else 1)
            predicted = float(np.polyval(coefficients, x))
            heading = float(np.clip(np.polyval(np.polyder(coefficients), x),
                                    -cfg["rail_max_heading"], cfg["rail_max_heading"]))
            gauge_inner = float(np.median(inner_gauges[recent]))
            gauge_center = gauge_inner + setting(cfg, "rail_head_width_m")
            band_half = 0.5 * setting(cfg, "rail_head_width_m") + setting(cfg, "rail_band_margin_m")
            head = float(np.median(heads[recent]))
            window = min(window_max, max(cfg["rail_window_m"], x * window_growth))
            band = np.abs(rail[:, 0] - x) < window / 2
            band_points, lateral, height = rail[band], rail[band, 1] - heading * (rail[band, 0] - x), rail_h[band]
            measured, verified = [], True
            for side in (-1.0, 1.0):
                target = predicted + side * gauge_center / 2
                near = ((np.abs(lateral - target) <= band_half * 2)
                        & (height >= head + low_band) & (height <= head + high_band))
                center, half, count = self._rail_band_center(lateral[near], target, band_half)
                if center is None:
                    verified = False
                    break
                span = float(np.ptp(band_points[near, 0])) if np.count_nonzero(near) else 0.0
                measured.append((center, half, count, span))
            if verified:
                left_center, left_half, left_n, left_span = measured[0]
                right_center, right_half, right_n, right_span = measured[1]
                head_half = 0.5 * setting(cfg, "rail_head_width_m")
                left_rail, right_rail = (left_center + left_half) - head_half, (right_center - right_half) + head_half
                center = 0.5 * (left_rail + right_rail)
                residual = center - predicted
                symmetry = min(left_n, right_n) / max(left_n, right_n)
                measured_inner = (right_center - right_half) - (left_center + left_half)
                verified = (min(left_n, right_n) >= min_points and min(left_span, right_span) >= cfg["rail_min_span_m"]
                            and abs(residual) <= max_offset and symmetry >= symmetry_min
                            and abs(measured_inner - cfg["rail_gauge_m"]) <= cfg["rail_gauge_tolerance_m"])
            if not verified:
                misses += 1
                if misses >= setting(cfg, "rail_extension_misses_max"):
                    break
                continue
            misses = 0
            near_head = (height >= head + low_band) & (height <= head + high_band)
            anchors.append((float(x), center, right_rail - left_rail, int(min(left_n, right_n))))
            heads.append(float(np.quantile(height[near_head], 0.8)))
            sigmas.append(max(0.01, 0.25 * (left_half + right_half) / np.sqrt(max(1, min(left_n, right_n)))))
            inner_gauges.append(measured_inner)
            residuals.append(abs(residual) / cfg["ground_segment_m"])
        if residuals:
            # Extrapolation growth comes from the model residuals actually observed.
            self.path_growth_m_per_m = float(np.clip(np.median(residuals), low_growth, high_growth))

    def rail_head_profile(self, x: np.ndarray) -> np.ndarray:
        """Rail-head height above the local bed along the path: the datum the
        reference contour is measured from. Held constant past the last anchor."""
        if not len(self.rail_head_anchors):
            value = self.rail_head_height_m if self.rail_head_height_m is not None else np.nan
            return np.full(len(x), value)
        anchors = self.rail_head_anchors
        return np.interp(x, anchors[:, 0], anchors[:, 1])

    def path(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        if len(self.rail_anchors) < 2:
            return np.zeros(len(x)), np.full(len(x), cfg["rail_gauge_m"]), np.full(len(x), np.inf)
        a = self.rail_anchors
        center = np.interp(x, a[:, 0], a[:, 1])
        # Endpoint tangent extrapolation; uncertainty grows with unsupported distance.
        for edge, other, mask in [(0, 1, x < a[0, 0]), (-1, -2, x > a[-1, 0])]:
            slope = np.clip((a[edge, 1] - a[other, 1]) / (a[edge, 0] - a[other, 0]),
                            -cfg["rail_max_heading"], cfg["rail_max_heading"])
            center[mask] = a[edge, 1] + slope * (x[mask] - a[edge, 0])
        nearest = nearest_anchor_distance(x, a[:, 0])
        floor = setting(cfg, "path_sigma_floor_m")
        sigma = (np.interp(x, a[:, 0], self.rail_sigma)
                 if len(self.rail_sigma) == len(a) else np.full(len(x), floor))
        growth = (self.path_growth_m_per_m if self.path_growth_m_per_m is not None
                  else setting(cfg, "path_growth_bounds_m_per_m")[1])
        uncertainty = np.maximum(sigma, floor) + growth * nearest
        uncertainty[nearest > cfg["path_max_extrapolation_m"]] = np.inf
        return center, np.interp(x, a[:, 0], a[:, 2]), uncertainty

    def classify(self, points: np.ndarray, *, remove_background: bool = True):
        cfg = self.config
        z, ground_uncertainty = self.ground(points)
        center, gauge, path_uncertainty = self.path(points[:, 0])
        height = points[:, 2] - z
        rail_head = self.rail_head_profile(points[:, 0])
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
        core = ground_supported & observed & (np.abs(lateral) <= width)
        # Segmentation precedes the collision gate. Do not amputate the feet or
        # head of an object just because only part intersects the envelope.
        segmentation_height = (running_height >= cfg["min_running_height_m"]) & (running_height <= envelope[-1, 1] + cfg["cluster_context_margin_m"])
        context = segmentation_height & ~on_rail & (np.abs(lateral) <= cfg["segmentation_context_half_width_m"])
        nominal_overlap = ((running_height >= envelope[0, 0]) & (running_height <= envelope[-1, 1])
                           & ~on_rail & (np.abs(lateral) <= width))
        if remove_background and self.background is not None:
            background = self.background.mask(points, observed & nominal_overlap)
            context &= ~background
        return core, context, height, observed, nominal_overlap

    def describe(self) -> dict:
        samples = np.asarray(self.config["range_bins_m"], dtype=float)
        centers, _, path_sigma = self.path(samples)
        _, ground_sigma = self.ground(np.column_stack((samples, centers, np.zeros(len(samples)))))
        uncertainty = [{"x_m": float(x), "path_sigma_m": float(p) if np.isfinite(p) else None,
                        "ground_sigma_m": float(g) if np.isfinite(g) else None,
                        "geometry_supported": bool(self.valid and p <= self.config["path_max_uncertainty_m"]
                                                   and g <= self.config["ground_max_uncertainty_m"])}
                       for x, p, g in zip(samples, path_sigma, ground_sigma)]
        return {"valid": self.valid, "reason": self.reason, "ground_quality": self.ground_quality,
                "ground_plane": None if self.plane is None else self.plane.tolist(),
                "rail_head_height_m": self.rail_head_height_m,
                "rail_head_anchors": self.rail_head_anchors.tolist(),
                "rail_fit_diagnostics": self.rail_fit_diagnostics,
                "uncertainty_by_range": uncertainty,
                "evidence_note": "Rail heights/residuals sample returns near inferred rail centres above fitted bed; gauge uses visible bands plus assumed head width. Uncertainty is heuristic, null means unsupported. path_horizon_m covers lateral support only; ground support is separate. No surveyed gauge, cant, extrinsics or swept envelope validation.",
                "rail_sigmas_m": self.rail_sigma.tolist(),
                "gauge_inner_median_m": (float(np.median(self.rail_gauge_inner))
                                         if len(self.rail_gauge_inner) else None),
                "gauge_convention_conflict": self.gauge_convention_conflict,
                "path_growth_m_per_m": self.path_growth_m_per_m,
                "path_horizon_m": self.path_horizon_m,
                "ground_anchors": self.ground_anchors.tolist(), "rail_anchors": self.rail_anchors.tolist(),
                "background": None if self.background is None else self.background.describe()}
