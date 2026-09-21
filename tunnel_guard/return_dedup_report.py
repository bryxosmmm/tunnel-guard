"""Evaluate retained real replay outputs and diagnose KISS sampling on their first scan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import Detector, load_config
from .io import iter_bag
from .panel_report import compare, difference
from .run import digest, write_json


def ordered_points(points):
    return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    reports = []
    for entry in panel["comparisons"]:
        result = compare(entry)
        rows = [[json.loads(line) for line in Path(entry[v]).read_text().splitlines()]
                for v in ("before", "after")]
        result["latency_ms"] = {
            v: {k: dict(zip(("p50", "p95"),
                            np.quantile([r[k] * 1000 for r in data], [.5, .95]).tolist()))
                for k in ("decode_s", "processing_s", "motion_s", "read_and_process_s")}
            for v, data in zip(("before", "after"), rows)}
        result["removed_returns"] = {
            "total": sum(r["duplicate_return_points"] for r in rows[1]),
            "median_per_frame": float(np.median([r["duplicate_return_points"] for r in rows[1]])),
        }
        paired = list(zip(*rows))
        result["max_pose_translation_difference_m"] = max(
            float(np.linalg.norm(np.asarray(a["pose"])[:3, 3] - np.asarray(b["pose"])[:3, 3]))
            for a, b in paired)
        result["changed_frames_additional"] = {
            field: sum(difference(a.get(field), b.get(field), [0.]) for a, b in paired)
            for field in ("supported_range_m", "geometry_points")}
        # Objects are sorted by current measured cluster distance in this detector.
        # Record array differences rather than silently aligning on changed track IDs.
        result["changed_frames_object_fields"] = {
            field: sum(difference([o[field] for o in a["objects"]],
                                  [o[field] for o in b["objects"]], [0.]) for a, b in paired)
            for field in ("track_id", "confirmed", "intersection_confirmed", "support_voxels")}
        result["manifest_sha256"] = {
            v: digest(Path(entry[v]).parent / "manifest.json") for v in ("before", "after")}

        # Actual KISS-ICP voxelization, using the identical real acquisition with
        # either decoder setting. No replacement/approximation of its algorithm.
        config = load_config(panel["detector_config"])
        stages = []
        for enabled in (False, True):
            cfg = config | {"deduplicate_returns": enabled}
            scan = next(iter_bag(Path(entry["bag_path"]), cfg, max_frames=1))
            odometry = Detector(cfg).odometry
            frame = odometry.preprocessor.preprocess(scan.points, scan.point_times, np.eye(4))
            source, downsample = odometry.voxelize(frame)
            stages.append((source, downsample))
        result["first_scan_kiss_voxelization"] = {
            name: {
                "before_points": len(a), "after_points": len(b),
                "same_order": bool(np.array_equal(a, b)),
                "same_point_multiset": bool(np.array_equal(ordered_points(a), ordered_points(b))),
            }
            for name, a, b in (("map", stages[0][1], stages[1][1]),
                               ("registration", stages[0][0], stages[1][0]))}
        reports.append(result)
    write_json(args.output, {
        "decision": "rejected_for_default_use",
        "hypothesis": "Exact return removal accelerates processing without changing detector decisions.",
        "acceptance": "Not met: odometry, track identity and confirmations change.",
        "panel_sha256": digest(args.panel), "comparisons": reports,
        "limitations": [
            "Development prefixes; no ground-truth trajectory or exhaustive object labels.",
            "Pose differences establish non-equivalence, not which trajectory is more accurate.",
            "Single sequential before/after runs; timings are exploratory, not a repeatability claim.",
            "No automated tests; actual detector replay and original KISS voxelization only.",
            "Timestamp quantization and emission identity are not verified for other sensors.",
            "ROS integration was not run on this Mac.",
        ],
    })


if __name__ == "__main__":
    main()
