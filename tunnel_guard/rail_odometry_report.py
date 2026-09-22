"""Compare a complete real rail-odometry experiment with its frozen upstream replay."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate_frames
from .run import digest, write_json


def rows(root, bag):
    path = root / f"{bag}.jsonl"
    if not path.exists():
        path = root / f"{bag}.jsonl.gz"
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def quantiles(values):
    return dict(zip(("p10", "p50", "p90", "p95"), map(float, np.quantile(values, [.1, .5, .9, .95])))) if values else None


def summarize(records):
    poses = [np.asarray(r["pose"]) for r in records if r.get("pose") is not None]
    travel = sum(float(np.linalg.norm(b[:3, 3] - a[:3, 3])) for a, b in zip(poses, poses[1:]))
    endpoint = float(np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3])) if poses else None
    valid = [r["motion"] for r in records if r["motion"].get("valid")]
    rejected = [r["motion"] for r in records if not r["motion"].get("valid") and r["motion"].get("median_residual_m") is not None]
    strengths = [r["motion"]["translation_observability_eigenvalues"][0] for r in records
                 if r["motion"].get("translation_observability_eigenvalues") is not None]
    return {"frames": len(records), "status_frames": dict(Counter(r["status"] for r in records)),
            "hazard_frames": sum(r["status"] in ("obstacle", "unresolved_obstacle") for r in records),
            "pose_path_sum_m": travel, "endpoint_separation_m": endpoint,
            "gap_resets": sum(r["gap_reset"] for r in records),
            "pose_path_validity": "Raw reported-pose sum; reset jumps are not physical travel; endpoint separation is not ground-truth drift",
            "accepted_registrations": len(valid), "rejected_registrations_with_residual": len(rejected),
            "accepted_residual_m": quantiles([m["median_residual_m"] for m in valid]),
            "rejected_residual_m": quantiles([m["median_residual_m"] for m in rejected]),
            "accepted_overlap": quantiles([m["overlap"] for m in valid]),
            "measured_weak_axis_frames": sum(bool(r["motion"].get("weak_translation_axes", 0)) for r in records
                                             if r["motion"].get("translation_observability_eigenvalues") is not None),
            "unmeasured_observability_frames": sum(r["motion"].get("translation_observability_eigenvalues") is None for r in records),
            "minimum_normal_eigenvalue": quantiles(strengths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    before, after, output = (Path(recipe[k]) for k in ("before", "after", "output"))
    manifest = json.loads((after / "manifest.json").read_text())
    original_manifest = json.loads((before / "manifest.json").read_text())
    if "finished_unix_s" not in manifest:
        raise ValueError("Full candidate replay has not completed")
    original = json.loads((before / "detector.json").read_text())
    candidate = json.loads((after / "detector.json").read_text())
    plan = candidate.pop("rail_motion_correction")
    if original != candidate or recipe["seed"] != original["seed"]:
        raise ValueError("The only allowed detector configuration change is rail_motion_correction")
    annotations = {name: json.loads(Path(path).read_text()) for name, path in recipe["annotations"].items()}
    annotation_keys = {(f["bag"], f["frame"]) for ann in annotations.values() for f in ann["frames"]}
    predictions = [{}, {}]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", recipe)
    reports = []
    tolerance = recipe["continuous_tolerance"]
    with gzip.GzipFile(filename=str(output / "frames.jsonl.gz"), mode="wb", mtime=0) as stream:
        for bag in recipe["bags"]:
            arms = [[], []]
            changes = []
            reasons = Counter()
            accepted_corrections = 0
            rail_strengths, rail_weak_axes, correction_times = [], [], []
            raw_pose_max_difference = 0.0
            for old, new in zip_longest(rows(before, bag), rows(after, bag)):
                if old is None or new is None:
                    raise ValueError(f"Frame count mismatch in {bag}")
                if (old["frame"], old["measurement_timestamp_ns"], old["source_scan_id"]) != (new["frame"], new["measurement_timestamp_ns"], new["source_scan_id"]):
                    raise ValueError(f"Different source scans in {bag}")
                for index, row in enumerate((old, new)):
                    arms[index].append({k: row.get(k, {} if k == "motion" else None)
                                        for k in ("frame", "timestamp_s", "status", "pose", "motion", "gap_reset")})
                    if (bag, row["frame"]) in annotation_keys:
                        predictions[index][(bag, row["frame"])] = {"status": row["status"], "objects": [
                            {key: obj[key] for key in ("bbox_min", "bbox_max", "confirmed", "path_relation", "distance_m")}
                            for obj in row["objects"]]}
                a, b = old.get("motion", {}), new.get("motion", {})
                correction = b.get("rail_correction", {})
                reasons[correction.get("reason", "not_run")] += 1
                accepted_corrections += bool(correction.get("accepted"))
                if "rail_translation_eigenvalues" in correction:
                    rail_strengths.append(correction["rail_translation_eigenvalues"][0])
                    rail_weak_axes.append(np.abs(correction["rail_weak_direction"]))
                if "rail_odometry_s" in new:
                    correction_times.append(new["rail_odometry_s"])
                if "raw_ICP_pose" in b:
                    raw_pose_max_difference = max(raw_pose_max_difference,
                        float(np.max(np.abs(np.asarray(b["raw_ICP_pose"]) - np.asarray(old["pose"])))))
                record = {"bag": bag, "frame": new["frame"], "measurement_timestamp_ns": new["measurement_timestamp_ns"],
                          "before_status": old["status"], "after_status": new["status"],
                          "before_pose": old.get("pose"), "after_pose": new.get("pose"),
                          "before_motion": a, "after_motion": b, "gap_reset": new["gap_reset"]}
                record["accepted_registration_lost"] = bool(a.get("valid") and not b.get("valid"))
                record["accepted_residual_worsened"] = bool(a.get("valid") and b.get("median_residual_m") is not None
                    and b["median_residual_m"] > a["median_residual_m"] + tolerance)
                record["accepted_overlap_worsened"] = bool(a.get("valid") and b.get("overlap") is not None
                    and b["overlap"] + tolerance < a["overlap"])
                record["hazard_frame_lost"] = old["status"] in ("obstacle", "unresolved_obstacle") and new["status"] not in ("obstacle", "unresolved_obstacle")
                record["status_changed"] = old["status"] != new["status"]
                record["weak_flag_changed"] = a.get("weak_translation_axes") != b.get("weak_translation_axes")
                record["forward_component_changed"] = bool(correction.get("raw_relative") is not None
                    and correction["raw_relative"][0][3] != correction["used_relative"][0][3])
                flags = ("accepted_registration_lost", "accepted_residual_worsened", "accepted_overlap_worsened",
                         "hazard_frame_lost", "status_changed", "weak_flag_changed", "forward_component_changed")
                if any(record[k] for k in flags):
                    changes.append({k: record[k] for k in ("frame", "before_status", "after_status") + flags})
                stream.write((json.dumps(record, allow_nan=False) + "\n").encode())
            raw_poses = [np.asarray(r["motion"]["raw_ICP_pose"]) for r in arms[1]
                         if "raw_ICP_pose" in r["motion"]]
            raw_pose_metrics = {
                "frames": len(raw_poses),
                "pose_path_sum_m": sum(float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
                                       for a, b in zip(raw_poses, raw_poses[1:])),
                "endpoint_separation_m": float(np.linalg.norm(raw_poses[-1][:3, 3] - raw_poses[0][:3, 3])) if raw_poses else None,
                "scope": "Unmodified KISS poses from the same candidate execution, including any epoch reset jumps"}
            reports.append({"bag": bag, "before": summarize(arms[0]), "after": summarize(arms[1]),
                "same_run_raw_ICP": raw_pose_metrics,
                "correction_reasons": dict(reasons), "accepted_corrections": accepted_corrections,
                "rail_minimum_translation_eigenvalue": quantiles(rail_strengths),
                "rail_weak_axis_mean_abs_xyz": np.mean(rail_weak_axes, axis=0).tolist() if rail_weak_axes else None,
                "rail_correction_s": quantiles(correction_times), "raw_ICP_pose_max_abs_difference": raw_pose_max_difference,
                "changes": changes})
    if sum(r["after"]["frames"] for r in reports) != recipe["expected_frames"]:
        raise ValueError("The required full panel is incomplete")
    by_bag = {r["bag"]: r for r in reports}
    stationary = by_bag[recipe["stationary_bag"]]
    accepted_stationary = stationary["after"]["pose_path_sum_m"] <= stationary["before"]["pose_path_sum_m"] + tolerance
    returns = {bag: {"smaller_endpoint_separation": by_bag[bag]["after"]["endpoint_separation_m"] < by_bag[bag]["before"]["endpoint_separation_m"] - tolerance,
                     "uninterrupted_pose_epoch": not (by_bag[bag]["before"]["gap_resets"] or by_bag[bag]["after"]["gap_resets"])} for bag in recipe["return_bags"]}
    quality_ok = not any(c[k] for r in reports for c in r["changes"] for k in ("accepted_registration_lost", "accepted_residual_worsened", "accepted_overlap_worsened"))
    alarm_ok = not any(c["hazard_frame_lost"] for c in stationary["changes"])
    label_reports = {name: {"before": evaluate_frames(predictions[0], ann), "after": evaluate_frames(predictions[1], ann),
                            "annotation_sha256": digest(Path(recipe["annotations"][name]))} for name, ann in annotations.items()}
    label_ok = all(not entry[arm]["missing_frames"] for entry in label_reports.values() for arm in ("before", "after"))
    for entry in label_reports.values():
        after_frames = {(r["bag"], r["frame"]): r for r in entry["after"]["frames"]}
        label_ok &= all(after_frames[(r["bag"], r["frame"])]["tp"] >= r["tp"] for r in entry["before"]["frames"])
        entry["nonvacuous_baseline_positive_matches"] = entry["before"]["tp"] > 0
    weak_ok = not any(c["weak_flag_changed"] or c["forward_component_changed"] for r in reports for c in r["changes"])
    report = {"bags": reports, "annotations": label_reports,
              "criteria": {"stationary_not_worse": accepted_stationary, "return_recordings": returns,
                           "accepted_registrations_not_worse": quality_ok, "stationary_hazard_frames_preserved": alarm_ok,
                           "existing_label_matches_preserved": bool(label_ok), "weak_flag_and_forward_component_preserved": weak_ok,
                           "longitudinal_accuracy": "not_measured_no_independent_reference",
                           "weak_axis_policy": "Never clear normal-based flag using the rail fit; forward ICP component is retained, not validated"},
              "promote": bool(accepted_stationary and quality_ok and alarm_ok and label_ok and weak_ok
                              and all(v["smaller_endpoint_separation"] and v["uninterrupted_pose_epoch"] for v in returns.values())),
              "promotion_scope": "Even passed proxy gates do not establish absolute longitudinal accuracy or field precision",
              "recipe_sha256": digest(args.experiment), "before_manifest_sha256": digest(before / "manifest.json"),
              "reporter_sha256": digest(Path(__file__)),
              "after_manifest_sha256": digest(after / "manifest.json"), "method_plan": plan,
              "native_provenance": {
                  "sources_identical": original_manifest["native_accelerator"]["sources_sha256"] == manifest["native_accelerator"]["sources_sha256"],
                  "binary_identical": original_manifest["native_accelerator"]["binary_sha256"] == manifest["native_accelerator"]["binary_sha256"],
                  "before_binary_sha256": original_manifest["native_accelerator"]["binary_sha256"],
                  "after_binary_sha256": manifest["native_accelerator"]["binary_sha256"],
                  "limitation": "A historical baseline is not an identical-binary control; same-run raw ICP isolates the pose correction, not a complete uncorrected detector replay"},
              "frame_records_sha256": digest(output / "frames.jsonl.gz")}
    write_json(output / "summary.json", report)
    print(json.dumps({"frames": sum(r["after"]["frames"] for r in reports), "criteria": report["criteria"], "promote": report["promote"]}, indent=2))
    if not report["promote"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
