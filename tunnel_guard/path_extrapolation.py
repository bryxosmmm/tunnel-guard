"""Measure the lateral error of the straight-line track-centre extrapolation.

``TrackGeometry.path`` continues the track centre along the endpoint tangent past the last
rail anchor, so on a curve of radius R the prediction should fall to the outside of the
curve by roughly d^2/(2R) after d metres. This command measures that error inside a single
scan: the anchors past a cut are hidden, the real ``path`` is called on what remains, and
its prediction is compared against the anchors that were hidden. No odometry, no
annotations and no cross-frame registration are involved -- retained and held-out anchors
are observed in the same scan, so the measurement carries no registration error of its own.

The held-out anchors are the rail-pair detector's own output, not independent ground truth,
and its per-step search is bounded to ``rail_max_heading`` around the previous anchor. That
bound biases the anchors toward the straight continuation, so a measured error here is a
lower bound on the true one, never an inflated one.

Two alternatives are scored on the same held-out anchors: a local arc, which tests whether a
curvature model predicts better, and a straight line whose slope is least-squares fitted over
a window of anchors instead of the last two, which separates curvature bias from heading
noise. Nothing in ``geometry.py`` is modified or re-implemented.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag

# Anchors sit on a fixed ground_segment_m grid, so the extrapolation distance is discrete and
# is reported per exact value rather than binned. Curvature strata are named by radius so the
# hypothesis is tested against measured geometry instead of a recording's name. On the
# sourcecraft subset the two sharpest strata turned out to select frames whose anchors are
# unstable, not curved track: a per-frame fit resolves curvature only down to about R = 1500 m,
# so a stratum name is a bucket label, never evidence that the track has that radius.
CURVATURE_STRATA = (("straight_R_over_2000m", 0.0, 1 / 2000),
                    ("gentle_R_800_2000m", 1 / 2000, 1 / 800),
                    ("moderate_R_400_800m", 1 / 800, 1 / 400),
                    ("sharp_R_under_400m", 1 / 400, np.inf))


def scan_geometry(points: np.ndarray, detector: dict) -> TrackGeometry:
    """The crop, voxel reduction and geometry that ``on_track.frame_candidates`` performs."""
    forward = ((points[:, 0] >= detector["min_forward_m"]) & (points[:, 0] <= detector["max_range_m"])
               & (np.abs(points[:, 1]) <= detector["context_half_width_m"]))
    reduced = voxel_representatives(points[forward], detector["geometry_voxel_m"])
    return TrackGeometry(reduced, detector)


def arc_extrapolation(retained: np.ndarray, x: np.ndarray, window_m: float):
    """Local quadratic centre y(x)=a+bx+cx^2, evaluated at ``x``, with its signed curvature.

    Over a metro curve radius a shallow arc and a quadratic in x agree far inside the anchor
    noise, and the quadratic stays conditioned where a circle fit degenerates on a short arc.
    """
    local = retained[retained[:, 0] >= retained[-1, 0] - window_m]
    if len(local) < 3:
        return None, None
    c, b, _ = coefficients = np.polyfit(local[:, 0], local[:, 1], 2)
    heading = 2 * c * retained[-1, 0] + b
    curvature = float(2 * c / (1 + heading**2) ** 1.5)
    return np.polyval(coefficients, x), curvature


def fitted_slope_extrapolation(retained: np.ndarray, x: np.ndarray, window_m: float,
                               max_heading: float):
    """Straight continuation whose slope is least-squares fitted over a window of anchors.

    ``path`` takes its slope from the last two anchors alone, a 5 m baseline at
    ``ground_segment_m`` spacing, so anchor noise enters the heading almost undamped and is
    then multiplied by the extrapolation distance. Same straight-line model, longer baseline.
    """
    local = retained[retained[:, 0] >= retained[-1, 0] - window_m]
    if len(local) < 3:
        return None
    slope = float(np.clip(np.polyfit(local[:, 0], local[:, 1], 1)[0], -max_heading, max_heading))
    return retained[-1, 1] + slope * (x - retained[-1, 0])


def frame_rows(geometry: TrackGeometry, arc_window_m: float, slope_window_m: float) -> list[dict]:
    """Score every cut that leaves at least two retained and one held-out anchor."""
    anchors = geometry.rail_anchors
    max_heading = geometry.config["rail_max_heading"]
    rows = []
    for cut in range(1, len(anchors) - 1):
        retained, held = anchors[: cut + 1], anchors[cut + 1 :]
        # A shallow copy measures the real path() on a shortened anchor set without disturbing
        # the source geometry. rail_sigma is parallel to the anchors and path() falls back to a
        # flat floor when the lengths disagree, so it has to be cut to the same length.
        # path_growth_m_per_m is a per-frame scalar fitted on the full anchor set and is left as
        # it is, which if anything flatters the stated uncertainty rather than inflating it.
        truncated = copy.copy(geometry)
        truncated.rail_anchors = retained
        if len(geometry.rail_sigma) == len(anchors):
            truncated.rail_sigma = geometry.rail_sigma[: cut + 1]
        line, _, stated = truncated.path(held[:, 0])
        arc, curvature = arc_extrapolation(retained, held[:, 0], arc_window_m)
        fitted = fitted_slope_extrapolation(retained, held[:, 0], slope_window_m, max_heading)
        for i in range(len(held)):
            rows.append({
                "cut_x_m": float(retained[-1, 0]),
                "retained_anchors": len(retained),
                "extrapolation_m": float(held[i, 0] - retained[-1, 0]),
                "observed_center_m": float(held[i, 1]),
                "line_center_m": float(line[i]),
                "line_error_m": float(line[i] - held[i, 1]),
                "stated_uncertainty_m": float(stated[i]),
                "arc_error_m": None if arc is None else float(arc[i] - held[i, 1]),
                "fitted_slope_error_m": None if fitted is None else float(fitted[i] - held[i, 1]),
                "retained_curvature_1_per_m": curvature,
            })
    return rows


def quadratic_coefficient(distance: np.ndarray, error: np.ndarray) -> dict:
    """Least-squares fit of |error| = c d^2 through the origin; R = 1/(2c) if the model holds."""
    denominator = float(np.sum(distance**4))
    if denominator <= 0:
        return {"coefficient_1_per_m": None, "implied_radius_m": None}
    c = float(np.sum(error * distance**2) / denominator)
    return {"coefficient_1_per_m": round(c, 8),
            "implied_radius_m": round(1 / (2 * c), 1) if c > 0 else None}


def distance_rows(distance, line, stated, arc, curvature, mask) -> list[dict]:
    """One report row per exact extrapolation distance inside a curvature stratum."""
    out = []
    for value in np.unique(distance[mask]):
        at = mask & (distance == value)
        if at.sum() < 5:
            continue
        believable = at & np.isfinite(stated)
        scored = at & np.isfinite(arc)
        out.append({
            "extrapolation_m": float(value),
            "samples": int(at.sum()),
            "median_abs_line_error_m": round(float(np.median(line[at])), 3),
            "p90_abs_line_error_m": round(float(np.quantile(line[at], 0.9)), 3),
            "max_abs_line_error_m": round(float(line[at].max()), 3),
            # The hypothesis under test: a tangent leaves an arc of radius R by d^2/(2R).
            "predicted_abs_error_m": round(float(np.median(np.abs(curvature[at])) * value**2 / 2), 3),
            "median_stated_uncertainty_m": (round(float(np.median(stated[believable])), 3)
                                            if believable.any() else None),
            "share_line_error_above_stated": (round(float(np.mean(line[believable] > stated[believable])), 3)
                                              if believable.any() else None),
            "median_abs_arc_error_m": round(float(np.median(arc[scored])), 3) if scored.any() else None,
        })
    return out


def summarize(rows: list[dict]) -> dict:
    distance = np.array([r["extrapolation_m"] for r in rows])
    line = np.abs(np.array([r["line_error_m"] for r in rows]))
    stated = np.array([r["stated_uncertainty_m"] for r in rows])
    arc = np.abs(np.array([np.nan if r["arc_error_m"] is None else r["arc_error_m"] for r in rows]))
    curvature = np.array([np.nan if r["retained_curvature_1_per_m"] is None
                          else r["retained_curvature_1_per_m"] for r in rows])
    strata = []
    for name, lo, hi in CURVATURE_STRATA:
        mask = (np.abs(curvature) >= lo) & (np.abs(curvature) < hi)
        if mask.sum() < 5:
            continue
        scored = mask & np.isfinite(arc)
        strata.append({
            "stratum": name,
            "samples": int(mask.sum()),
            "median_radius_m": (round(float(1 / np.median(np.abs(curvature[mask]))), 1)
                                if np.median(np.abs(curvature[mask])) > 0 else None),
            "line_error_vs_distance": quadratic_coefficient(distance[mask], line[mask]),
            "arc_error_vs_distance": (quadratic_coefficient(distance[scored], arc[scored])
                                      if scored.any() else None),
            "by_distance": distance_rows(distance, line, stated, arc, curvature, mask),
        })
    measurable = np.isfinite(curvature)
    return {
        "samples": len(rows),
        "samples_with_curvature": int(measurable.sum()),
        "median_abs_line_error_m": round(float(np.median(line)), 3),
        "line_error_vs_distance": quadratic_coefficient(distance, line),
        "median_radius_m": (round(float(1 / np.median(np.abs(curvature[measurable]))), 1)
                            if measurable.any() and np.median(np.abs(curvature[measurable])) > 0 else None),
        "sharpest_radius_m": (round(float(1 / np.max(np.abs(curvature[measurable]))), 1)
                              if measurable.any() and np.max(np.abs(curvature[measurable])) > 0 else None),
        "curvature_strata": strata,
    }


def center_profile(observed: dict[float, list[float]]) -> list[dict]:
    """How far the observed track centre sits off the sensor axis at each anchored range.

    ``on_track.report_candidates`` rejects an event on raw |y| against a GOST half-width, so
    this is the quantity that decides whether such an event was off the track or on a track
    that has curved away from the axis.
    """
    return [{"anchor_x_m": x, "frames": len(values),
             "median_abs_center_m": round(float(np.median(np.abs(values))), 3),
             "p90_abs_center_m": round(float(np.quantile(np.abs(values), 0.9)), 3),
             "max_abs_center_m": round(float(np.max(np.abs(values))), 3)}
            for x, values in sorted(observed.items()) if len(values) >= 5]


def measure(bag: Path, detector: dict, every: int, max_frames: int | None, arc_window_m: float,
            slope_window_m: float) -> dict:
    rows, coverage, frames, unusable = [], [], 0, 0
    observed: dict[float, list[float]] = {}
    for scan in iter_bag(bag, detector, every=every, max_frames=max_frames):
        geometry = scan_geometry(scan.points, detector)
        if not geometry.valid:
            unusable += 1
            continue
        frames += 1
        coverage.append(float(geometry.rail_anchors[-1, 0]))
        for x, center in geometry.rail_anchors[:, :2]:
            observed.setdefault(float(x), []).append(float(center))
        rows.extend(frame_rows(geometry, arc_window_m, slope_window_m))
    anchored = np.asarray(coverage) if coverage else np.zeros(0)
    return {
        "bag": str(bag),
        "frames_with_geometry": frames,
        "frames_without_geometry": unusable,
        "median_last_anchor_m": round(float(np.median(anchored)), 2) if len(anchored) else None,
        "p10_last_anchor_m": round(float(np.quantile(anchored, 0.1)), 2) if len(anchored) else None,
        "max_last_anchor_m": round(float(anchored.max()), 2) if len(anchored) else None,
        "observed_center_profile": center_profile(observed),
        "summary": summarize(rows) if rows else None,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True,
                        help="repeat to measure several recordings in one run")
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--arc-window-m", type=float, default=20.0)
    parser.add_argument("--slope-window-m", type=float, default=20.0)
    args = parser.parse_args()

    detector = load_config(args.config)
    # Anchors are complete before TunnelBackground is built and do not depend on it; skipping
    # it removes plane RANSAC from every frame and changes nothing this command measures.
    detector = {**detector, "background": {**detector["background"], "enabled": False}}

    results = [measure(bag, detector, args.every, args.max_frames, args.arc_window_m,
                       args.slope_window_m) for bag in args.bag]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": str(args.config), "every": args.every, "max_frames": args.max_frames,
        "arc_window_m": args.arc_window_m, "slope_window_m": args.slope_window_m,
        "seed": detector["seed"], "recordings": results,
    }, indent=1) + "\n")

    for result in results:
        summary = result["summary"]
        print(f"\n{Path(result['bag']).name}: {result['frames_with_geometry']} frames with geometry "
              f"({result['frames_without_geometry']} without), last anchor median "
              f"{result['median_last_anchor_m']} m")
        if summary is None:
            continue
        print(f"  {summary['samples']} held-out anchors; median fitted radius "
              f"{summary['median_radius_m']} m, sharpest {summary['sharpest_radius_m']} m")
        profile = [p for p in result["observed_center_profile"] if p["anchor_x_m"] % 10 == 0]
        print("  observed |track centre| off the sensor axis: "
              + ", ".join(f"{p['anchor_x_m']:.0f} m: {p['median_abs_center_m']:.2f}"
                          f"/{p['max_abs_center_m']:.2f}" for p in profile))
        for stratum in summary["curvature_strata"]:
            print(f"  [{stratum['stratum']}] n={stratum['samples']}, median R="
                  f"{stratum['median_radius_m']} m, |e|=c*d^2 implies R="
                  f"{stratum['line_error_vs_distance']['implied_radius_m']} m")
            print(f"    {'d':>6} {'n':>6} {'med|e|':>8} {'p90|e|':>8} {'d^2/2R':>8} "
                  f"{'stated':>8} {'e>stated':>9} {'med|arc|':>9}")
            for row in stratum["by_distance"]:
                nan = float("nan")
                print(f"    {row['extrapolation_m']:>6.0f} {row['samples']:>6} "
                      f"{row['median_abs_line_error_m']:>8.3f} {row['p90_abs_line_error_m']:>8.3f} "
                      f"{row['predicted_abs_error_m']:>8.3f} "
                      f"{(nan if row['median_stated_uncertainty_m'] is None else row['median_stated_uncertainty_m']):>8.3f} "
                      f"{(nan if row['share_line_error_above_stated'] is None else row['share_line_error_above_stated']):>9.2f} "
                      f"{(nan if row['median_abs_arc_error_m'] is None else row['median_abs_arc_error_m']):>9.3f}")
    print(f"\nEvidence: {args.output}")


if __name__ == "__main__":
    main()
