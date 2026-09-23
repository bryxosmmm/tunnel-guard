"""Causal test: predict the centre-line ahead from accumulated rail history.

A single scan measures the rails only ~40 m ahead. But the train drives over the route: every frame
adds its own measured rail centre-line, so after a few hundred metres the alignment is measured over a
long stretch, and what a quadratic cannot see from one scan - how the curvature *changes* along arc
length - is present in that history. This asks the only question that matters for the corridor: at
frame k, using frames up to k, how well does a curve fitted to the accumulated rail samples predict
the centre-line at 60-250 m ahead, compared with the current scan's own anchor-window fit?

Poses come from the recorded run, so the history is in the map frame and the test is strictly causal.
Rails measured by *later* frames are the reference - they see the predicted ground at short range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import load_config
from .geometry import TrackGeometry
from .io import iter_bag

STEP_M = 1.72


def history_samples(frame_index: int, anchors: dict, poses: dict, pose_now: np.ndarray,
                    past_frames: int) -> np.ndarray:
    """Rail centre-line samples from frames up to now, expressed in the current sensor frame."""
    rows = []
    for earlier in range(max(0, frame_index - past_frames), frame_index + 1):
        if earlier not in anchors or earlier not in poses or len(anchors[earlier]) == 0:
            continue
        a = anchors[earlier]
        world = (poses[earlier][:3, :3] @ np.column_stack((a[:, 0], a[:, 1], np.zeros(len(a)))).T).T \
            + poses[earlier][:3, 3]
        local = (world - pose_now[:3, 3]) @ pose_now[:3, :3]
        rows.append(local[:, :2])
    return np.vstack(rows) if rows else np.empty((0, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--lookahead", type=float, nargs="+", default=[60.0, 100.0, 150.0, 200.0, 250.0])
    parser.add_argument("--past-frames", type=int, nargs="+", default=[60, 120, 200])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    with open(args.poses) as stream:
        poses = {json.loads(line)["frame"]: np.asarray(json.loads(line)["pose"], float) for line in stream}

    geometry, anchors = {}, {}
    sources = set(args.frames)
    for scan in iter_bag(args.bag, cfg, max_frames=max(args.frames) + 1):
        g = TrackGeometry(scan.points, cfg)
        anchors[scan.index] = g.rail_anchors
        if scan.index in sources:
            geometry[scan.index] = g

    results = {}
    print("look-ahead   n    window(shipped)   history60   history120   history200   (|median| error, m)")
    for look in args.lookahead:
        per_model: dict[str, list[float]] = {}
        for k in sorted(geometry):
            g = geometry[k]
            if len(g.rail_anchors) < 3 or look <= g.rail_anchors[-1, 0]:
                continue
            pose_now = poses[k]
            for n in range(4, 250):
                later = k + n
                if later not in anchors or later not in poses:
                    continue
                travelled = n * STEP_M
                if look - travelled < 2.0 or look - travelled > anchors[later][-1, 0]:
                    continue
                bed = float(np.interp(look, g.ground_anchors[:, 0], g.ground_anchors[:, 1])) \
                    if len(g.ground_anchors) else 0.0
                predicted = {"window(shipped)": float(g.path(np.array([look]))[0][0])}
                for extra in args.past_frames:
                    samples = history_samples(k, anchors, poses, pose_now, extra)
                    if len(samples) < 20 or samples[:, 0].max() < 10.0:
                        continue
                    # cubic in the sensor frame, weighted to the recent stretch by construction
                    design = np.column_stack((samples[:, 0], samples[:, 0] ** 2, samples[:, 0] ** 3, np.ones(len(samples))))
                    coef, *_ = np.linalg.lstsq(design, samples[:, 1], rcond=None)
                    predicted[f"history{extra}"] = float(np.array([look, look**2, look**3, 1.0]) @ coef)
                compared = False
                for name, lateral in predicted.items():
                    world = pose_now[:3, :3] @ np.array([look, lateral, bed]) + pose_now[:3, 3]
                    local = (world - poses[later][:3, 3]) @ poses[later][:3, :3]
                    if not (2.0 <= local[0] <= anchors[later][-1, 0]):
                        continue
                    measured = float(np.interp(local[0], anchors[later][:, 0], anchors[later][:, 1]))
                    per_model.setdefault(name, []).append(abs(local[1] - measured))
                    compared = True
                if compared:
                    break
        row = {name: (float(np.median(values)) if values else None)
               for name, values in per_model.items()}
        results[str(look)] = {"n": max((len(v) for v in per_model.values()), default=0), **row}
        cells = "   ".join(f"{row.get(name):8.3f}" if row.get(name) is not None else "     n/a"
                           for name in ("window(shipped)", "history60", "history120", "history200"))
        print(f"  {look:5.0f} m {results[str(look)]['n']:3d}   {cells}")
    if args.json_out:
        args.json_out.write_text(json.dumps({"bag": str(args.bag), "config": str(args.config),
                                            "results": results}, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
