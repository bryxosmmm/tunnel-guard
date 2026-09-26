"""Track alignment from the long lever: curvature shared by every longitudinal surface.

The rails give the absolute track centre but die near 40 m, and a quadratic fitted to them predicts
the centre-line only to 2.2 m at 100 m and 7.4 m at 150 m (measured), against a 1.535 m corridor
half-width. The tunnel lining is the longer lever: it is an extrusion of one profile along the
alignment, so *every* longitudinal surface - either wall, the ceiling, a cable tray, a platform edge
- has the alignment's curvature and only its own offset. Offsets therefore must never be combined
(that is what wrecked per-bin medians: neighbouring returns belong to different surfaces), while the
curvature may be pooled, which is what buys precision.

Model, fitted by partial regression so the shared curvature is separated from per-surface nuisance:

    y_s(x) = a_s + b_s * x + c * x^2      (c shared, (a_s, b_s) free per surface)

Rails enter as one more surface with a tight weight, so the near field stays anchored to the
measurement while the long surfaces decide the shape. The returned sigma_c is the standard error of
the shared curvature, and it is what sets how far the corridor may be modelled: the lateral error
at range x is sigma_c * x^2 / 2, so a corridor budget in metres converts directly into a horizon.

Nothing here assumes a tunnel profile, a wall pair or a constant offset; sections where the profile
changes appear as surface ends, not as curvature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BIN_M = 5.0
GAP_M = 0.35


@dataclass
class Trace:
    """One longitudinal surface as (x, y) samples with their return counts."""
    x: np.ndarray
    y: np.ndarray
    counts: np.ndarray
    weight: float = 1.0
    label: str = "surface"
    sample_weights: np.ndarray | None = None      # per-sample weights, overriding `weight`

    @property
    def span(self) -> float:
        return float(self.x.max() - self.x.min())


@dataclass
class Alignment:
    """Fitted shared curvature with the evidence that produced it."""
    valid: bool
    curvature: float = 0.0
    curvature_sigma: float = float("inf")
    surfaces: int = 0
    longest_span_m: float = 0.0
    residual_m: float = float("inf")
    spans_m: list[float] = field(default_factory=list)
    reason: str = "not_fitted"

    def horizon_m(self, budget_m: float) -> float:
        """Range at which the curvature error alone would fill the corridor budget."""
        if not self.valid or self.curvature_sigma <= 0:
            return 0.0
        return float(np.sqrt(2 * budget_m / self.curvature_sigma))

    def offset_at(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.curvature * np.asarray(x, dtype=float) ** 2 / 2


def surface_traces(points: np.ndarray, bed_z: float, config: dict) -> list[Trace]:
    """Chain per-bin lateral clusters into longitudinal surfaces (no offset is ever combined)."""
    band = config.get("alignment_band_above_bed_m", [0.2, 4.0])
    lateral_limit = config.get("alignment_max_lateral_m", 8.0)
    min_points = int(config.get("alignment_min_points_per_bin", 12))
    min_span = float(config.get("alignment_min_span_m", 40.0))
    max_range = float(config.get("alignment_max_range_m", 260.0))
    height = points[:, 2] - bed_z
    lateral = points[:, 1]
    keep = ((height >= band[0]) & (height <= band[1]) & (np.abs(lateral) <= lateral_limit)
            & (points[:, 0] > 0) & (points[:, 0] <= max_range))
    x, y = points[keep, 0], lateral[keep]

    open_traces: list[dict] = []
    closed: list[dict] = []
    edges = np.arange(0.0, max_range + BIN_M, BIN_M)
    for lo, hi in zip(edges[:-1], edges[1:]):
        centre = 0.5 * (lo + hi)
        selected = (x >= lo) & (x < hi)
        groups: list[tuple[float, int]] = []
        if selected.sum() >= min_points:
            values = np.sort(y[selected])
            for cluster in np.split(values, np.flatnonzero(np.diff(values) > 0.25) + 1):
                if len(cluster) >= 4:
                    groups.append((float(np.median(cluster)), len(cluster)))
        used = set()
        for trace in open_traces:
            best, best_gap = None, GAP_M
            for index, (value, _) in enumerate(groups):
                if index in used:
                    continue
                gap = abs(value - trace["y"][-1])
                if gap < best_gap:
                    best, best_gap = index, gap
            if best is not None:
                value, count = groups[best]
                used.add(best)
                trace["x"].append(centre)
                trace["y"].append(value)
                trace["counts"].append(count)
        for trace in [t for t in open_traces if centre - t["x"][-1] > BIN_M + 1e-6]:
            open_traces.remove(trace)
            closed.append(trace)
        for index, (value, count) in enumerate(groups):
            if index not in used:
                open_traces.append({"x": [centre], "y": [value], "counts": [count]})

    traces: list[Trace] = []
    for trace in closed + open_traces:
        if len(trace["x"]) < 6:
            continue
        candidate = Trace(np.array(trace["x"]), np.array(trace["y"]), np.array(trace["counts"]))
        if candidate.span >= min_span:
            traces.append(candidate)
    return traces


def fit_alignment(traces: list[Trace], rail_xy: np.ndarray | None, config: dict) -> Alignment:
    """Shared curvature by partial regression; rails included as an anchor surface."""
    pool = [t for t in traces if t.span >= float(config.get("alignment_min_span_m", 40.0))]
    if rail_xy is not None and len(rail_xy) >= 4:
        pool.append(Trace(rail_xy[:, 0], rail_xy[:, 1], np.full(len(rail_xy), 100),
                          weight=float(config.get("alignment_rail_weight", 4.0)), label="rails"))
    if len(pool) < int(config.get("alignment_min_surfaces", 2)):
        return Alignment(False, reason=f"only {len(pool)} usable traces")

    # Project the curvature regressor out of each surface's own affine part: the leftover is what
    # actually informs the shared c, independently of that surface's offset and slope.
    weighted_xy = 0.0
    total_information = 0.0
    for trace in pool:
        x, y = trace.x, trace.y
        if len(x) < 6 or trace.span <= 0:
            continue
        design = np.column_stack((np.ones(len(x)), x))
        orthogonal = x**2 - design @ np.linalg.lstsq(design, x**2, rcond=None)[0]
        weight = trace.weight
        weighted_xy += weight * float(orthogonal @ y)
        total_information += weight * float(orthogonal @ orthogonal)
    if total_information <= 0:
        return Alignment(False, reason="no curvature information")
    curvature = weighted_xy / total_information

    # Residual after fitting shared c plus each surface's own affine part.
    scatter = 0.0
    dof = 0
    spans = []
    for trace in pool:
        x, y = trace.x, trace.y
        design = np.column_stack((np.ones(len(x)), x))
        target = y - curvature * x**2
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coef
        scatter += trace.weight * float(residual @ residual)
        dof += max(1, len(x) - 3)
        spans.append(trace.span)
    variance = scatter / max(dof, 1)
    sigma = float(np.sqrt(variance / total_information))
    return Alignment(True, curvature=float(curvature), curvature_sigma=sigma, surfaces=len(pool),
                     longest_span_m=float(max(spans)), residual_m=float(np.sqrt(variance)),
                     spans_m=[float(s) for s in sorted(spans, reverse=True)],
                     reason="fitted")


def axis_trace(axis: "CrossSectionAxis", weight_scale: float = 1.0) -> Trace | None:
    """The measured cross-section centres as one weighted trace for fit_spline (1/sigma^2 weighting)."""
    if not axis.valid or len(axis.samples) < 4:
        return None
    stations = np.array([sample.station_m for sample in axis.samples])
    laterals = np.array([sample.lateral_m for sample in axis.samples])
    sigmas = np.array([max(sample.sigma_m, 1e-3) for sample in axis.samples])
    return Trace(stations, laterals, np.array([sample.points for sample in axis.samples]),
                 sample_weights=weight_scale / sigmas**2, label="axis")


def _hat_basis(knots: np.ndarray, x) -> np.ndarray:
    """Linear hat basis over the knots; outside the knot range the end values extend flat."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    index = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
    span = knots[index + 1] - knots[index]
    weight = np.clip((x - knots[index]) / span, 0.0, 1.0)
    basis = np.zeros((len(x), len(knots)))
    basis[np.arange(len(x)), index] = 1.0 - weight
    basis[np.arange(len(x)), index + 1] = weight
    return basis

@dataclass
class SplineAlignment:
    """Alignment as a function, fitted from every surface at once with a smoothness prior."""
    valid: bool
    knots: np.ndarray = field(default_factory=lambda: np.empty(0))
    values: np.ndarray = field(default_factory=lambda: np.empty(0))
    covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lam: float = 0.0
    surfaces: int = 0
    residual_m: float = float("inf")
    spans_m: list[float] = field(default_factory=list)
    reason: str = "not_fitted"

    def _basis(self, x):
        return _hat_basis(self.knots, x)

    def predict(self, x) -> np.ndarray:
        return self._basis(x) @ self.values
        index = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
        span = knots[index + 1] - knots[index]
        weight = np.clip((x - knots[index]) / span, 0.0, 1.0)
        basis = np.zeros((len(x), len(knots)))
        basis[np.arange(len(x)), index] = 1.0 - weight
        basis[np.arange(len(x)), index + 1] = weight
        return basis

    def sigma(self, x) -> np.ndarray:
        basis = self._basis(x)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", basis, self.covariance, basis), 0.0))

    def horizon_m(self, budget_m: float) -> float:
        """Furthest knot whose posterior uncertainty is still inside the corridor budget."""
        if not self.valid:
            return 0.0
        inside = np.flatnonzero(self.sigma(self.knots) <= budget_m)
        return float(self.knots[inside.max()]) if len(inside) else 0.0


def fit_spline(traces: list[Trace], rail_xy: np.ndarray | None, config: dict,
               knot_m: float | None = None, budget_m: float = 0.5,
               lambdas: np.ndarray | None = None) -> SplineAlignment:
    """Joint penalized-spline alignment: y_s(x) = a_s + b_s x + f(x), f smooth, f(0)=0.

    Every extruded surface shares f and keeps its own (a_s, b_s); the rails enter as a tightly
    weighted surface so the near field stays anchored, and f is pinned at the origin with the
    measured heading. The curvature is *not* assumed constant - the second-difference penalty lets
    f follow a changing alignment, which a single quadratic provably cannot (measured 20-38 m at
    200-250 m). The posterior covariance yields the range over which the corridor may be modelled.
    """
    pool = [t for t in traces if t.span >= float(config.get("alignment_min_span_m", 40.0))]
    if rail_xy is not None and len(rail_xy) >= 4:
        pool.append(Trace(rail_xy[:, 0], rail_xy[:, 1], np.full(len(rail_xy), 100),
                          weight=float(config.get("alignment_rail_weight", 4.0)), label="rails"))
    if len(pool) < 2:
        return SplineAlignment(False, reason=f"only {len(pool)} usable traces")
    knot_m = float(knot_m or config.get("alignment_knot_m", 10.0))
    end = max(float(t.x.max()) for t in pool)
    knots = np.arange(0.0, end + knot_m, knot_m)
    # heading anchor from the rails (or the longest trace) fixes f'(0); f(0) = 0 fixes the level.
    anchor = next((t for t in pool if t.label == "rails"), max(pool, key=lambda t: t.span))
    slope0 = float(np.polyfit(anchor.x, anchor.y, 1)[0])

    blocks, targets, weights = [], [], []
    for trace in pool:
        basis = _hat_basis(knots, trace.x)
        design = np.column_stack((np.ones(len(trace.x)), trace.x, basis))   # a_s, b_s, f
        blocks.append(design)
        targets.append(trace.y)
        weights.append(trace.sample_weights if trace.sample_weights is not None
                       else np.full(len(trace.x), trace.weight))
    # soft constraints: f at x=0 is zero, and its slope at the first interval matches the anchor
    constraint_design = np.zeros((2, 2 + len(knots)))
    constraint_design[0, 2] = 1.0
    constraint_design[1, 2] = -1.0 / knot_m
    constraint_design[1, 3] = 1.0 / knot_m
    constraint_targets = np.array([0.0, slope0])
    constraint_weight = float(config.get("alignment_anchor_weight", 50.0))
    blocks.append(constraint_design)
    targets.append(constraint_targets)
    weights.append(np.full(len(constraint_targets), constraint_weight))
    design = np.vstack(blocks)
    target = np.concatenate(targets)
    weight = np.concatenate(weights)

    penalty = np.zeros((design.shape[1], design.shape[1]))
    for k in range(len(knots) - 2):
        row = np.zeros(design.shape[1])
        row[2 + k] = 1.0
        row[2 + k + 1] = -2.0
        row[2 + k + 2] = 1.0
        penalty += np.outer(row, row)

    weighted = design * weight[:, None]
    normal = design.T @ weighted
    rhs = design.T @ (weight * target)
    if lambdas is None:
        lambdas = np.logspace(-6, 8, 29)
    # Smooth as much as the data allows: take the largest penalty whose weighted residual still
    # reaches the measured noise floor. Adaptive GCV mis-weights the per-surface nuisance blocks and
    # was measured to over-smooth (residual 1.7-4.2 m, i.e. an almost straight alignment).
    floor = float(config.get("alignment_noise_floor_m", 0.10))
    total_weight = float(weight.sum())
    best = None
    for lam in sorted(lambdas, reverse=True):
        matrix = normal + lam * penalty
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            continue
        residual = target - design @ solution
        scatter = float((weight * residual**2).sum())
        rms = float(np.sqrt(scatter / total_weight))
        if rms <= floor:
            best = (rms, lam, solution, matrix, scatter)
            break
        best = (rms, lam, solution, matrix, scatter)      # keep the last as the fallback
    if best is None:
        return SplineAlignment(False, reason="no solvable penalty")
    _, lam, solution, matrix, scatter = best
    dof = max(1, design.shape[0] - len(knots))
    variance = scatter / dof
    covariance_full = np.linalg.inv(matrix) * variance
    values = solution[2:]
    return SplineAlignment(True, knots=knots, values=values,
                           covariance=covariance_full[2:, 2:], lam=float(lam), surfaces=len(pool),
                           residual_m=float(np.sqrt(variance)),
                           spans_m=[float(t.span) for t in sorted(pool, key=lambda t: -t.span)],
                           reason="fitted")


@dataclass
class AxisSample:
    station_m: float
    lateral_m: float
    height_m: float
    sigma_m: float
    points: int
    radius_m: float


@dataclass
class CrossSectionAxis:
    """Alignment as a sequence of measured cross-section centres (no functional form)."""
    valid: bool
    samples: list[AxisSample] = field(default_factory=list)
    breaks: list[float] = field(default_factory=list)
    rail_offset_median_m: float = float("nan")
    rail_offset_spread_m: float = float("inf")
    reason: str = "not_fitted"

    @property
    def furthest_m(self) -> float:
        return self.samples[-1].station_m if self.samples else 0.0

    def run_from(self, station_m: float) -> list[AxisSample]:
        """Samples in the continuous run containing this station (section changes break runs)."""
        run: list[AxisSample] = []
        for sample in self.samples:
            if sample.station_m < station_m:
                continue
            if run and any(abs(sample.station_m - b) < 1e-6 for b in self.breaks):
                return run
            run.append(sample)
        return run


def cross_section_axis(points: np.ndarray, rail_anchors: np.ndarray, config: dict) -> CrossSectionAxis:
    """Alignment from robust circle fits to perpendicular slabs of the scan.

    Bootstrap frame: a line through the near-field rail anchors (the only absolute centre available),
    then a slab every `alignment_station_m` along it. Each slab's lining points are fitted with a
    circle under a Huber loss - the bore's cross-section - and its centre is an axis sample. The
    primitive matters: a slab averages over azimuth and height, so an attachment is a minority of the
    samples and is down-weighted, whereas the earlier per-bin Cartesian trace followed whichever
    cluster was nearest and could not be fitted by any shared function.

    Only the axis *shape* is taken from the bore; the absolute offset stays with the rails, because a
    constant bore-versus-track offset cancels in the shape and would otherwise be invented as signal.
    """
    if len(rail_anchors) < 3:
        return CrossSectionAxis(False, reason="fewer than three rail anchors")
    cfg = config
    near = rail_anchors[rail_anchors[:, 0] <= float(cfg.get("alignment_bootstrap_range_m", 20.0))]
    if len(near) < 3:
        near = rail_anchors[:3]
    heading = float(np.polyfit(near[:, 0], near[:, 1], 1)[0])
    along = np.array([1.0, heading]) / np.hypot(1.0, heading)
    lateral_dir = np.array([-along[1], along[0]])
    bed_z = float(cfg.get("alignment_bed_z_m", -1.32))
    station_step = float(cfg.get("alignment_station_m", 5.0))
    half_width = float(cfg.get("alignment_slab_half_m", 0.6))
    max_station = float(cfg.get("alignment_max_range_m", 260.0))
    min_points = int(cfg.get("alignment_min_slab_points", 15))
    max_sigma = float(cfg.get("alignment_max_slab_sigma_m", 0.15))

    xy = points[:, :2]
    along_coord = xy @ along
    lateral = xy @ lateral_dir
    height = points[:, 2]

    samples: list[AxisSample] = []
    station = station_step
    while station <= max_station:
        slab = (np.abs(along_coord - station) <= half_width) & (np.abs(lateral) <= 8.0) \
            & (height >= bed_z - 0.4) & (height <= bed_z + 6.0)
        if slab.sum() >= min_points:
            centre, radius, sigma = _robust_circle(lateral[slab], height[slab])
            if centre is not None and sigma <= max_sigma and 1.5 <= radius <= 12.0:
                samples.append(AxisSample(station, float(centre[0]), float(centre[1]),
                                          float(sigma), int(slab.sum()), float(radius)))
        station += station_step
    if len(samples) < 4:
        return CrossSectionAxis(False, reason=f"only {len(samples)} usable slabs")

    # A section change is a step that departs from the local trend, not the along-track drift a
    # curve produces (at 5 m stations on R = 1000 m the centre legitimately moves ~0.75 m), so the
    # second difference is what is tested: a straight or curved run has a smooth one.
    breaks = []
    lateral_values = np.array([s.lateral_m for s in samples])
    sigmas = np.array([s.sigma_m for s in samples])
    for index in range(1, len(samples) - 1):
        second = lateral_values[index - 1] - 2 * lateral_values[index] + lateral_values[index + 1]
        allowed = max(0.20, 6.0 * max(sigmas[index - 1], sigmas[index], sigmas[index + 1]))
        if abs(second) > allowed:
            breaks.append(samples[index].station_m)
    rail_interp = np.interp([s.station_m for s in samples],
                            rail_anchors[:, 0] * np.hypot(1.0, heading) + rail_anchors[:, 1] * heading / np.hypot(1.0, heading),
                            rail_anchors[:, 1])
    offsets = rail_interp - np.array([s.lateral_m for s in samples])
    return CrossSectionAxis(True, samples=samples, breaks=breaks,
                            rail_offset_median_m=float(np.median(offsets)),
                            rail_offset_spread_m=float(np.percentile(np.abs(offsets - np.median(offsets)), 90)),
                            reason="fitted")


def _robust_circle(lateral: np.ndarray, height: np.ndarray, iterations: int = 6):
    """Kasa algebraic circle fit with Huber reweighting.

    Returns (centre, radius, sigma_centre) where sigma_centre is the *linearised* standard error of
    the fitted centre, not the residual scatter divided by sqrt(n). At long range a slab sees a
    nearly straight arc, which a circle can fit with small residuals while its centre is barely
    determined; the covariance of the normal equations sees that conditioning, the residual does not.
    """
    x, y = np.asarray(lateral, float), np.asarray(height, float)
    if len(x) < 6:
        return None, 0.0, float("inf")
    weight = np.ones(len(x))
    centre = np.array([float(np.median(x)), float(np.median(y))])
    radius = float(np.median(np.hypot(x - centre[0], y - centre[1])))
    sigma_centre = float("inf")
    for _ in range(iterations):
        design = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
        target = x**2 + y**2
        weighted = design * weight[:, None]
        normal = design.T @ weighted
        solution, *_ = np.linalg.lstsq(weighted, weight * target, rcond=None)
        residual = weight * (target - design @ solution)
        dof = max(1, int(weight.sum() * 3 / 3) - 3)
        scatter = float((residual**2).sum() / max(weight.sum(), 1e-9))
        try:
            covariance = np.linalg.inv(normal) * scatter / max(dof, 1)
            sigma_centre = float(np.sqrt(max(np.linalg.eigvalsh(covariance[:2, :2]).max(), 0.0)))
        except np.linalg.LinAlgError:
            sigma_centre = float("inf")
        centre = solution[:2]
        radius = float(np.sqrt(max(solution[2] + centre @ centre, 0.0)))
        geometric = np.hypot(x - centre[0], y - centre[1]) - radius
        scale = 1.4826 * np.median(np.abs(geometric - np.median(geometric))) + 1e-6
        weight = np.clip(1.5 * scale / np.maximum(np.abs(geometric), 1e-9), 0.0, 1.0)
    return centre, radius, sigma_centre
