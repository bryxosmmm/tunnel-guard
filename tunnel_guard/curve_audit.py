"""Audit the corridor's continuation on recorded track: how curved, and which model is closer.

The six sourcecraft recordings have almost no measurable curvature, so the fitted quadratic continuation
was validated where the track is nearly straight. This reads a run over the extended recording instead,
keeps the frames whose own anchor chain is genuinely curved, and scores two models against a label the
sensor itself supplies later:

    for each curved frame k, take a look-ahead point on the model's centre-line, carry it into a LATER
    frame l that has driven over the same ground, and compare it with the rails l actually measures there.

Nothing is hand-annotated and the reference is the sensor's own later measurement, so this scores the
model rather than a labelling convention. A pair is scored only when frame l's anchors span the
look-ahead point, which is what makes the comparison meaningful, and the qualifying count is reported.

    python -m tunnel_guard.curve_audit --run build/extended-curved-scan --lookahead 20 40 60

Curvature is measured from the frame's own anchors (quadratic fit, radius = 1 / (2|a2|)) - the same
quantity the continuation shrinks when its window cannot support it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag


def fitted_radius(anchors: np.ndarray) -> tuple[float, float]:
    """Radius and quadratic residual of a frame's own anchor chain."""
    if len(anchors) < 5 or np.ptp(anchors[:, 0]) < 1.0:
        return np.inf, np.inf
    coefficients = np.polyfit(anchors[:, 0], anchors[:, 1], 2)
    residual = float(np.abs(anchors[:, 1] - np.polyval(coefficients, anchors[:, 0])).max())
    radius = float(1 / (2 * abs(coefficients[0]))) if coefficients[0] else np.inf
    return radius, residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory with per-recording jsonl")
    parser.add_argument("--bags", type=Path, default=Path("data/extended_subset"))
    parser.add_argument("--config", type=Path, default=Path("configs/detector-native.json"))
    parser.add_argument("--lookahead", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    parser.add_argument("--radius-max", type=float, default=800.0, help="curved frames below this radius")
    parser.add_argument("--min-travel-m", type=float, default=10.0,
                        help="how far the vehicle must have advanced before its measurement can label a "
                             "look-ahead point past the source frame's anchor end")
    parser.add_argument("--stride", type=int, default=4, help="later frames tried, every Nth")
    args = parser.parse_args()

    config = load_config(args.config)
    # The previous model: no fitted curvature, the tangent with its clipped slope.
    straight = dict(config, path_curve_window_m=0.0)
    scored: dict[float, list[tuple[float, float]]] = {look: [] for look in args.lookahead}
    seen = curved = 0
    radii: list[float] = []
    travels: list[float] = []
    recordings = 0
    for path in sorted(args.run.glob("*.jsonl")):
        if path.name.endswith("-timing.jsonl"):
            continue
        rows = [json.loads(line) for line in path.open()]
        poses = {r["frame"]: np.asarray(r["pose"], float) for r in rows
                 if r.get("pose") and (r.get("geometry") or {}).get("rail_anchors")}
        if not poses:
            continue
        bag = args.bags / path.stem
        if not bag.exists():
            print(f"  {path.stem}: no bag at {bag}, skipped")
            continue
        recordings += 1
        # The detector builds its geometry from the voxel representatives of the forward, cropped cloud,
        # not from the raw scan. Rebuilding it any other way would score a different corridor than the
        # one that ships, and would cost several times more per frame.
        points_by_frame = {}
        for scan in iter_bag(bag, config):
            if scan.index not in poses:
                continue
            points = scan.points
            forward = ((points[:, 0] >= config["min_forward_m"])
                       & (points[:, 0] <= config["max_range_m"])
                       & (np.abs(points[:, 1]) <= config["context_half_width_m"]))
            points_by_frame[scan.index] = voxel_representatives(points[forward],
                                                                config["geometry_voxel_m"])
        # One geometry per frame per model: the fit is the expensive part, not the comparison.
        new = {frame: TrackGeometry(points, config) for frame, points in points_by_frame.items()}
        old = {frame: TrackGeometry(points, straight) for frame, points in points_by_frame.items()}
        continuations = {}
        for frame, geometry in new.items():
            if len(geometry.rail_anchors) >= 3:
                continuations[frame] = (geometry._continuation(-1), old[frame]._continuation(-1)
                                        if len(old[frame].rail_anchors) >= 3 else None)
        order = sorted(continuations)
        steps = [np.linalg.norm(poses[b][:3, 3] - poses[a][:3, 3]) for a, b in zip(order, order[1:])]
        # Total travel is what bounds the look-ahead this recording can label; the per-frame step is
        # what the crawl looks like. Reporting one as the other would misread the whole measurement.
        travel = float(np.linalg.norm(poses[order[-1]][:3, 3] - poses[order[0]][:3, 3])) if order else 0.0
        travels.append(travel)
        print(f"  {path.stem}: {len(points_by_frame)} frames, {len(continuations)} with an anchor chain, "
              f"step median {np.median(steps):.2f} m, total travel {travel:.1f} m", flush=True)
        for frame, (edge_new, edge_old) in continuations.items():
            seen += 1
            radius, _ = fitted_radius(new[frame].rail_anchors)
            radii.append(radius)
            if radius > args.radius_max or edge_old is None:
                continue
            curved += 1
            # A later frame can only label a point past this frame's anchor end if the vehicle has
            # advanced far enough for that ground to fall inside its own measured span. Frames are 0.1 s
            # apart and this recording crawls, so adjacency is useless here: candidates are chosen by
            # travel, not by index.
            later = [f for f in sorted(continuations)
                     if f > frame
                     and np.linalg.norm(poses[f][:3, 3] - poses[frame][:3, 3]) >= args.min_travel_m][:: args.stride]
            for other in later:
                label = new[other].rail_anchors
                if len(label) < 2:
                    continue
                for look in args.lookahead:
                    errors = []
                    # Both models are evaluated at the same physical place: the end of the measured
                    # anchors plus the look-ahead. Only their slope and curvature differ.
                    x_look = float(new[frame].rail_anchors[-1, 0]) + look
                    for (x_edge, y_edge, slope, curvature, *_) in (edge_new, edge_old):
                        y_look = y_edge + slope * (x_look - x_edge) + curvature * (x_look - x_edge) ** 2
                        world = poses[frame][:3, :3] @ np.array([x_look, y_look, 0.0]) + poses[frame][:3, 3]
                        local = (world - poses[other][:3, 3]) @ poses[other][:3, :3]
                        if not (label[0, 0] <= local[0] <= label[-1, 0]):
                            errors = None
                            break
                        errors.append(abs(float(local[1] - np.interp(local[0], label[:, 0], label[:, 1]))))
                    if errors:
                        scored[look].append((errors[0], errors[1]))
    finite = [r for r in radii if np.isfinite(r)]
    print(f"  recordings {recordings}, frames {seen}, curved (<= {args.radius_max:g} m) {curved}")
    if travels:
        print(f"  total travel per recording: median {np.median(travels):.1f} m, minimum {min(travels):.1f} m, "
              f"maximum {max(travels):.1f} m; pair candidates need {args.min_travel_m:g} m between frames")
    if finite:
        print(f"  radius quartiles: p25 {np.percentile(finite, 25):.0f} m, median {np.median(finite):.0f} m, "
              f"p75 {np.percentile(finite, 75):.0f} m, min {min(finite):.0f} m")
    print("  look-ahead   pairs   curvature model (m)   tangent model (m)")
    for look in args.lookahead:
        pairs = scored[look]
        if not pairs:
            print(f"  {look:8.0f} m   {len(pairs):5d}   (no qualifying frame pair)")
            continue
        print(f"  {look:8.0f} m   {len(pairs):5d}   |median| {np.median([p[0] for p in pairs]):7.3f}"
              f"          |median| {np.median([p[1] for p in pairs]):7.3f}")


if __name__ == "__main__":
    main()
