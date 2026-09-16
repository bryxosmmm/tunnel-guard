"""Measure whether the estimated track centre stays locked to one track.

``TrackGeometry`` finds rail pairs by walking outwards in ``ground_segment_m`` steps, each step
allowed to move laterally by ``rail_max_heading * dx + 0.15`` around the previous anchor. Over
the anchored range that budget adds up to several metres, so nothing in the estimator forbids it
from walking off onto a neighbouring track, a platform edge or a diverging leg of a switch, one
legal step at a time.

This command tests the lock without annotations, using two facts the recording supplies by
itself. Inside a frame, the anchor-to-anchor step says whether the centre was dragged at the
per-step bound rather than supported by evidence. Between consecutive frames 0.1 s apart, the
centre at a *fixed range ahead* cannot move by metres, so a jump there is an estimator failure
and not track geometry. Gauge, support and rail-head height at the same anchors say whether
whatever it locked onto still looks like a rail pair.

Only anchored ranges are probed: extrapolation past the last anchor is a separate question,
measured in ``path_extrapolation``. Nothing in ``geometry.py`` is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag

PROBE_RANGES_M = (20.0, 30.0, 40.0)
EXCURSION_M = 1.0
JUMP_M = 0.5


def frame_geometry(points: np.ndarray, detector: dict) -> TrackGeometry:
    forward = ((points[:, 0] >= detector["min_forward_m"]) & (points[:, 0] <= detector["max_range_m"])
               & (np.abs(points[:, 1]) <= detector["context_half_width_m"]))
    reduced = voxel_representatives(points[forward], detector["geometry_voxel_m"])
    return TrackGeometry(reduced, detector)


def frame_row(geometry: TrackGeometry, detector: dict) -> dict:
    """Probe the anchored centre and describe how hard the estimator worked to get there."""
    anchors = geometry.rail_anchors
    step = np.diff(anchors[:, 1]) / np.diff(anchors[:, 0])
    # A step riding the clip is the tracker being dragged by whatever it last matched, not
    # evidence that the track turns; the bound is rail_max_heading with a fixed 0.15 m slack.
    budget = detector["rail_max_heading"] + 0.15 / np.diff(anchors[:, 0])
    row = {"first_anchor_m": float(anchors[0, 0]), "last_anchor_m": float(anchors[-1, 0]),
           "anchors": len(anchors),
           "rail_head_height_m": (None if geometry.rail_head_height_m is None
                                  else round(float(geometry.rail_head_height_m), 3)),
           "max_abs_anchor_slope": round(float(np.abs(step).max()), 4) if len(step) else None,
           "steps_at_bound": int(np.count_nonzero(np.abs(step) >= 0.95 * budget)) if len(step) else 0,
           "steps": len(step), "probes": {}}
    for probe in PROBE_RANGES_M:
        if not anchors[0, 0] <= probe <= anchors[-1, 0]:
            continue
        nearest = int(np.argmin(np.abs(anchors[:, 0] - probe)))
        row["probes"][str(probe)] = {
            "center_m": round(float(np.interp(probe, anchors[:, 0], anchors[:, 1])), 3),
            "gauge_m": round(float(anchors[nearest, 2]), 3),
            "support": int(anchors[nearest, 3])}
    return row


def episodes(rows: list[dict], probe: str) -> list[dict]:
    """Consecutive runs of frames whose probed centre sits beyond the excursion threshold."""
    out, current = [], None
    for row in rows:
        probed = row["probes"].get(probe)
        far = probed is not None and abs(probed["center_m"]) > EXCURSION_M
        if far and current is not None and row["frame"] - current["last_frame"] <= 2:
            current["last_frame"] = row["frame"]
            current["centers"].append(probed["center_m"])
            current["gauges"].append(probed["gauge_m"])
            current["supports"].append(probed["support"])
        elif far:
            if current:
                out.append(current)
            current = {"first_frame": row["frame"], "last_frame": row["frame"],
                       "centers": [probed["center_m"]], "gauges": [probed["gauge_m"]],
                       "supports": [probed["support"]]}
        elif current is not None:
            out.append(current)
            current = None
    if current:
        out.append(current)
    return [{"first_frame": e["first_frame"], "last_frame": e["last_frame"],
             "frames": len(e["centers"]),
             "max_abs_center_m": round(float(np.max(np.abs(e["centers"]))), 2),
             "median_center_m": round(float(np.median(e["centers"])), 2),
             "side": "left" if np.median(e["centers"]) > 0 else "right",
             "median_gauge_m": round(float(np.median(e["gauges"])), 3),
             "median_support": int(np.median(e["supports"]))} for e in out]


def summarize(rows: list[dict], detector: dict) -> dict:
    out = {"frames": len(rows), "by_probe": {}}
    dragged = [r["steps_at_bound"] / r["steps"] for r in rows if r["steps"]]
    out["share_of_steps_at_bound"] = round(float(np.mean(dragged)), 3) if dragged else None
    out["frames_with_any_step_at_bound"] = int(sum(1 for r in rows if r["steps_at_bound"]))
    heights = [r["rail_head_height_m"] for r in rows if r["rail_head_height_m"] is not None]
    out["rail_head_height_m"] = {"p50": round(float(np.median(heights)), 3),
                                 "p10": round(float(np.quantile(heights, 0.1)), 3),
                                 "p90": round(float(np.quantile(heights, 0.9)), 3)} if heights else None
    for probe in PROBE_RANGES_M:
        key = str(probe)
        probed = [(r["frame"], r["probes"][key]) for r in rows if key in r["probes"]]
        if len(probed) < 5:
            continue
        centers = np.array([p["center_m"] for _, p in probed])
        frames = np.array([f for f, _ in probed])
        adjacent = np.diff(frames) == 1
        jumps = np.abs(np.diff(centers))[adjacent]
        far = np.abs(centers) > EXCURSION_M
        near_gauge = np.array([p["gauge_m"] for _, p in probed])
        near_support = np.array([p["support"] for _, p in probed])
        out["by_probe"][key] = {
            "frames_probed": len(probed),
            "median_abs_center_m": round(float(np.median(np.abs(centers))), 3),
            "p90_abs_center_m": round(float(np.quantile(np.abs(centers), 0.9)), 3),
            "max_abs_center_m": round(float(np.abs(centers).max()), 3),
            "share_beyond_excursion": round(float(far.mean()), 3),
            "consecutive_pairs": int(adjacent.sum()),
            "median_frame_to_frame_step_m": round(float(np.median(jumps)), 3) if len(jumps) else None,
            "p99_frame_to_frame_step_m": round(float(np.quantile(jumps, 0.99)), 3) if len(jumps) else None,
            "max_frame_to_frame_step_m": round(float(jumps.max()), 3) if len(jumps) else None,
            "share_jumps_over_threshold": round(float((jumps > JUMP_M).mean()), 4) if len(jumps) else None,
            "gauge_m": {"on_track": round(float(np.median(near_gauge[~far])), 3) if (~far).any() else None,
                        "excursion": round(float(np.median(near_gauge[far])), 3) if far.any() else None},
            "support": {"on_track": int(np.median(near_support[~far])) if (~far).any() else None,
                        "excursion": int(np.median(near_support[far])) if far.any() else None},
            "episodes": episodes(rows, key)}
    return out


def measure(bag: Path, detector: dict, every: int, max_frames: int | None) -> dict:
    rows, invalid = [], 0
    for scan in iter_bag(bag, detector, every=every, max_frames=max_frames):
        geometry = frame_geometry(scan.points, detector)
        if not geometry.valid:
            invalid += 1
            continue
        row = frame_row(geometry, detector)
        row["frame"] = scan.index
        row["timestamp_s"] = scan.timestamp_s
        rows.append(row)
    return {"bag": str(bag), "frames_without_geometry": invalid,
            "summary": summarize(rows, detector) if rows else None, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = load_config(args.config)
    detector = {**detector, "background": {**detector["background"], "enabled": False}}
    results = [measure(bag, detector, args.every, args.max_frames) for bag in args.bag]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": str(args.config), "every": args.every,
        "probe_ranges_m": list(PROBE_RANGES_M), "excursion_m": EXCURSION_M, "jump_m": JUMP_M,
        "seed": detector["seed"], "recordings": results}, indent=1) + "\n")

    for result in results:
        summary = result["summary"]
        print(f"\n{Path(result['bag']).name}: {summary['frames']} frames, "
              f"steps at the per-step bound in {summary['frames_with_any_step_at_bound']} of them "
              f"({summary['share_of_steps_at_bound']} of all steps)")
        print(f"  {'probe':>6} {'n':>5} {'med|c|':>7} {'p90|c|':>7} {'max|c|':>7} {'>1m':>6} "
              f"{'medjump':>8} {'p99jump':>8} {'maxjump':>8} {'gauge on/off':>14} {'supp on/off':>13} {'eps':>4}")
        for probe, row in summary["by_probe"].items():
            gauge = f"{row['gauge_m']['on_track']}/{row['gauge_m']['excursion']}"
            support = f"{row['support']['on_track']}/{row['support']['excursion']}"
            print(f"  {float(probe):>6.0f} {row['frames_probed']:>5} {row['median_abs_center_m']:>7.3f} "
                  f"{row['p90_abs_center_m']:>7.3f} {row['max_abs_center_m']:>7.3f} "
                  f"{row['share_beyond_excursion']:>6.2f} "
                  f"{row['median_frame_to_frame_step_m']:>8.3f} {row['p99_frame_to_frame_step_m']:>8.3f} "
                  f"{row['max_frame_to_frame_step_m']:>8.3f} {gauge:>14} {support:>13} "
                  f"{len(row['episodes']):>4}")
    print(f"\nEvidence: {args.output}")


if __name__ == "__main__":
    main()
