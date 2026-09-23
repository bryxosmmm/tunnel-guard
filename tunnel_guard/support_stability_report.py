"""Compare real support counts exactly and locate captured numerical divergence."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from .object_evidence_report import rows
from .run import digest, write_json


def compare(before, after, bags):
    totals = Counter()
    changes = []
    geometry = ("track_id", "component_id", "bbox_min", "bbox_max", "support_voxels", "path_relation")
    fields = ("accumulated_support_voxels", "presence_confirmed", "presence_confirmation", "confirmed", "confirmation",
              "intersection_confirmed", "intersection_confirmation", "hits", "intersection_hits", "immediate",
              "intersection_immediate", "evidence_timestamps_s", "intersection_evidence_timestamps_s")
    for bag in bags:
        for old, new in zip_longest(rows(before, bag), rows(after, bag)):
            if old is None or new is None:
                raise ValueError(f"Incomplete replay: {bag}")
            if any(old[key] != new[key] for key in ("frame", "source_scan_id", "measurement_timestamp_ns")):
                raise ValueError(f"Source scan mismatch: {bag}")
            totals["frames"] += 1
            totals["pose_bit_patterns_differ"] += np.asarray(old["pose"], dtype=np.float64).tobytes() != np.asarray(new["pose"], dtype=np.float64).tobytes()
            if any(old[key] != new[key] for key in ("status", "nearest_obstacle_m", "nearest_candidate_m")):
                totals["alarm_frames_changed"] += 1
                changes.append({"bag": bag, "frame": new["frame"], "alarm_before": old["status"], "alarm_after": new["status"]})
            if len(old["objects"]) != len(new["objects"]):
                totals["object_cardinality_changed"] += 1
            for a, b in zip(old["objects"], new["objects"]):
                totals["objects"] += 1
                changed = {key: [a[key], b[key]] for key in geometry + fields if a[key] != b[key]}
                if changed:
                    changes.append({"bag": bag, "frame": new["frame"], "track_id": b["track_id"],
                                    "current_support_voxels": b["support_voxels"], "changes": changed})
                    for key in changed:
                        totals[key + "_changed"] += 1
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (before, after)]
    source_hashes = [{Path(key).name: value for key, value in manifest["source_sha256"].items()} for manifest in manifests]
    return {"before": str(before), "after": str(after), "totals": dict(totals), "changes": changes,
            "identical_source_hashes": source_hashes[0] == source_hashes[1],
            "identical_native": manifests[0]["native_accelerator"] == manifests[1]["native_accelerator"],
            "identical_config": json.loads((before / "detector.json").read_text()) == json.loads((after / "detector.json").read_text()),
            "identical_environment": all(manifests[0][key] == manifests[1][key] for key in ("python", "platform", "machine", "packages")),
            "manifests": manifests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "reporter.py").write_bytes(Path(__file__).read_bytes())
    (output / "experiment.json").write_bytes(args.experiment.read_bytes())
    comparisons = {}
    for name, case in plan["comparisons"].items():
        for root in (case["before"], case["after"]):
            if "finished_unix_s" not in json.loads((Path(root) / "manifest.json").read_text()):
                raise ValueError(f"Replay is not complete: {root}")
        comparisons[name] = compare(Path(case["before"]), Path(case["after"]), case["bags"])
        if comparisons[name]["totals"]["frames"] != case["expected_frames"]:
            raise ValueError("Incomplete fixed panel")
    stages = []
    for frame in plan["trace_frames"]:
        name = f"doubleT_platform_{frame:06d}.npz"
        roots = [Path(plan["comparisons"]["control_repeat"][arm]) for arm in ("before", "after")]
        with np.load(roots[0] / "diagnostics" / name) as a, np.load(roots[1] / "diagnostics" / name) as b:
            evidence = {}
            for key in ("decoded_points", "range_points", "motion_source", "motion_initial_guess", "motion_map", "motion_sigma", "motion_pose", "support_3303", "support_world_3303", "evidence_3303"):
                if key in a and key in b:
                    x, y = a[key], b[key]
                    evidence[key] = {"same_bytes": x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes(),
                                     "max_abs_difference": float(np.max(np.abs(x - y))) if x.size and x.shape == y.shape else None}
                    if key == "evidence_3303":
                        evidence[key].update(before=x.tolist(), after=y.tolist(),
                            before_keys=np.floor(x / plan["voxel_m"]).astype(np.int64).tolist(),
                            after_keys=np.floor(y / plan["voxel_m"]).astype(np.int64).tolist())
            stages.append({"frame": frame, "stages": evidence})
    target = {}
    for arm in ("before", "after"):
        root = Path(plan["comparisons"]["fix_repeat"][arm])
        with np.load(root / "diagnostics/doubleT_platform_000291.npz") as arrays:
            stack = arrays["evidence_3303"]
            target[arm] = {"support": arrays["support_3303"].tolist(), "centered_support": arrays["support_relative_3303"].tolist(),
                           "evidence": stack.tolist(), "exactly_zero": bool(np.all(stack == 0)),
                           "floor_count": len(np.unique(np.floor(stack / plan["voxel_m"]).astype(np.int64), axis=0))}
    repeat = comparisons["fix_repeat"]
    allowed = {"frames", "objects", "pose_bit_patterns_differ"}
    repeat_ok = not any(value for key, value in repeat["totals"].items() if key not in allowed)
    regression = comparisons["before_fix"]["totals"]
    invariant_fields = ("alarm_frames_changed", "object_cardinality_changed", "track_id_changed", "component_id_changed",
                        "bbox_min_changed", "bbox_max_changed", "support_voxels_changed", "path_relation_changed",
                        "confirmed_changed", "intersection_confirmed_changed")
    criteria = {"fixed_counts_and_confirmations_repeat_exactly": repeat_ok,
                "geometry_and_alarms_preserved": not any(regression.get(key, 0) for key in invariant_fields),
                "singleton_center_has_one_real_cell": all(t["exactly_zero"] and t["floor_count"] == 1 for t in target.values()),
                "identical_repeat_code_config_binary_environment": all(repeat[key] for key in
                    ("identical_source_hashes", "identical_native", "identical_config", "identical_environment"))}
    if plan.get("require_identical_poses", False):
        criteria["serial_poses_repeat_bitwise"] = repeat["totals"].get("pose_bit_patterns_differ", 0) == 0
    report = {"plan": plan, "criteria": criteria, "comparisons": comparisons, "first_divergence_trace": stages,
              "fixed_target": target, "registration_repeats": {name: json.loads(Path(path).read_text()) for name, path in plan["registration_reports"].items()},
              "recipe_sha256": digest(output / "experiment.json"), "reporter_sha256": digest(output / "reporter.py"),
              "limits": "Exact integer comparison, no epsilon or tolerance. Floating ICP poses may still differ under parallel reductions. Repeatability on this panel is not geometric uncertainty calibration, field recall, or general cross-platform bitwise determinism."}
    write_json(output / "summary.json", report)
    print(json.dumps({"criteria": criteria, "comparisons": {name: r["totals"] for name, r in comparisons.items()}}, indent=2))
    if not all(criteria.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
