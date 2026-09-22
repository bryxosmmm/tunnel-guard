"""Anatomy of a recorded hazard load: what the nearest alarm on each frame is made of.

Reads one or more run directories of detector JSONL and reports, for the object that sets each
frame's `nearest_obstacle_m`, its size, support, relation reason, distance and the longitudinal
extent of its component. Streaming, scalars only, so a full 2488-frame corpus costs seconds.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

HAZARDS = ("intersecting", "unresolved")


def rows(paths):
    for path in paths:
        with open(path) as stream:
            for line in stream:
                yield json.loads(line)


def summarize(values, quantiles=(10, 50, 90)) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0}
    return {"n": len(values)} | {f"p{q}": round(float(np.percentile(values, q)), 3) for q in quantiles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    paths = sorted(p for p in glob.glob(str(args.run / "*.jsonl")) if "timing" not in p)
    nearest, per_frame, statuses, reasons = [], [], Counter(), Counter()
    for row in rows(paths):
        statuses[row["status"]] += 1
        hazards = [o for o in row["objects"] if o["confirmed"] and o["path_relation"] in HAZARDS]
        per_frame.append((len(row["objects"]), sum(1 for o in row["objects"] if o["path_relation"] == "adjacent"),
                          len(hazards), row["nearest_obstacle_m"]))
        for obj in row["objects"]:
            reasons[obj["path_relation_reason"]] += 1
        if hazards:
            obj = min(hazards, key=lambda o: o["distance_m"])
            nearest.append({"distance": obj["distance_m"], "support": obj["support_voxels"],
                            "extent": max(obj["extent_m"]), "span_x": obj["bbox_max"][0] - obj["bbox_min"][0],
                            "x0": obj["bbox_min"][0], "uncertain": obj["uncertain_voxels"],
                            "inside": obj["in_envelope_voxels"], "boundary": obj["boundary_uncertain_voxels"],
                            "immediate": float(obj["immediate"]), "hits": obj["hits"],
                            "top_height": obj["height_above_bed_m"][1], "reason": obj["path_relation_reason"]})
    print(f"{args.label or args.run}: {len(per_frame)} frames")
    print("  statuses:", dict(statuses))
    print("  object reasons:", dict(reasons))
    print("  objects/frame p50:", summarize([r[0] for r in per_frame])["p50"],
          " adjacent/frame p50:", summarize([r[1] for r in per_frame])["p50"],
          " hazards/frame p50:", summarize([r[2] for r in per_frame])["p50"])
    if not nearest:
        print("  no confirmed hazard")
        return
    for key in ("distance", "x0", "span_x", "support", "extent", "uncertain", "inside", "boundary",
                "top_height", "hits"):
        print(f"  nearest hazard {key:12s}", summarize([n[key] for n in nearest]))
    print("  nearest hazard immediate share:", round(float(np.mean([n["immediate"] for n in nearest])), 3))
    print("  nearest hazard reasons:", dict(Counter(n["reason"] for n in nearest)))


if __name__ == "__main__":
    main()
