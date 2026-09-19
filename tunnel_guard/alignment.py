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
