"""Summarize paired measured-background / HYBRID SYNTHETIC detector replays."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .run import write_json


def make_report(output: Path):
    recipe = json.loads((output / "experiment.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    cases = json.loads((output / "cases.json").read_text())
    evaluation = [c for c in cases if c["split"] == "evaluation"]
    visible = [c for c in evaluation if max(c["visible_returns_per_frame"]) > 0]
    inside = [c for c in evaluation if c["position"] == "inside"]
    adjacent = [c for c in evaluation if c["position"] == "adjacent"]
    background = {}
    for split in recipe["splits"]:
        rows = [json.loads(line) for line in (output / f"baseline-{split['name']}.jsonl").read_text().splitlines()]
        background[split["name"]] = {"frames": len(rows),
                                     "status_counts": {state: sum(r["status"] == state for r in rows)
                                                       for state in sorted({r["status"] for r in rows})},
                                     "alarm_frames": sum(r["status"] in ("obstacle", "unresolved_obstacle") for r in rows),
                                     "unknown_frames": sum(r["status"] == "unknown" for r in rows),
                                     "duration_s": rows[-1]["timestamp_s"] - rows[0]["timestamp_s"]}
    def rate(group: list[dict], key: str):
        return sum(bool(c[key]) for c in group) / len(group) if group else None
    summary = {"label": "HYBRID SYNTHETIC", "hypothesis": recipe["hypothesis"],
               "criterion": recipe["promotion_criterion"], "criterion_result": None,
               "evaluation_cases": len(evaluation), "visible_cases": len(visible),
               "zero_visibility_cases": len(evaluation) - len(visible),
               "object_detection_all_cases": rate(evaluation, "object_detected"),
               "hazard_confirmation_all_cases": rate(evaluation, "hazard_confirmed"),
               "object_detection_given_support": rate(visible, "object_detected"),
               "hazard_confirmation_given_support": rate(visible, "hazard_confirmed"),
               "inside_hazard_confirmation_all_cases": rate(inside, "hazard_confirmed"),
               "adjacent_hazard_confirmation_cases": rate(adjacent, "hazard_confirmed"),
               "background": background,
               "validity": "Measured empty-development backgrounds with modeled box returns; no field recall, hardware range, private recall, or calibrated probability."}
    criterion = recipe["promotion_criterion"]
    summary["criterion_result"] = (summary["object_detection_all_cases"] is not None
                                   and summary["object_detection_all_cases"] >= criterion["minimum_object_detection_all_cases"]
                                   and summary["inside_hazard_confirmation_all_cases"] >= criterion["minimum_inside_hazard_confirmation_all_cases"]
                                   and summary["adjacent_hazard_confirmation_cases"] <= criterion["maximum_adjacent_hazard_confirmation_cases"]
                                   and background["evaluation"]["alarm_frames"] <= criterion["maximum_evaluation_background_alarm_frames"])
    columns = ["split", "case", "range_m", "shape", "dimensions_m", "position", "lateral_m", "motion",
               "visibility_mode", "visible_returns_per_frame", "intersecting_rays_per_frame",
               "foreground_occluded_rays_per_frame", "removed_returns_per_frame",
               "object_detected", "hazard_confirmed", "first_visible_frame", "first_detection_frame",
               "first_hazard_frame", "detection_latency_s", "hazard_latency_s", "matched_path_relations",
               "matched_distances_m", "statuses", "baseline_statuses", "background_alarm_frames", "unmatched_alarm_frames"]
    with (output / "matrix.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            writer.writerow({k: json.dumps(case[k], ensure_ascii=False) if isinstance(case[k], (list, dict)) else case[k]
                             for k in columns})
    summary["failures"] = [{"case": c["case"], "range_m": c["range_m"], "shape": c["shape"],
                            "position": c["position"], "visible_returns_per_frame": c["visible_returns_per_frame"],
                            "object_detected": c["object_detected"], "hazard_confirmed": c["hazard_confirmed"]}
                           for c in evaluation if not c["hazard_confirmed"]]
    summary["manifest_sources"] = [{"split": s["split"], "bag": s["bag"], "source_frame_indices": s["source_frame_indices"],
                                    "baseline_pose_valid": s["baseline_pose_valid"]} for s in manifest["sources"]]
    write_json(output / "report.json", summary)
    (output / "report.md").write_text(
        "# HYBRID SYNTHETIC — measured empty-tunnel backgrounds\n\n"
        f"Evaluation: {len(evaluation)} scenarios; {len(visible)} with at least one inserted visible return; "
        f"{len(evaluation)-len(visible)} with zero visibility.\n\n"
        f"Object detection: {summary['object_detection_all_cases']} all / {summary['object_detection_given_support']} given support.\n\n"
        f"Confirmed intersecting hazard: {summary['hazard_confirmation_all_cases']} all / {summary['hazard_confirmation_given_support']} given support; "
        f"{summary['inside_hazard_confirmation_all_cases']} for nominal inside and {summary['adjacent_hazard_confirmation_cases']} for nominal adjacent.\n\n"
        f"Paired evaluation background alarm frames: {background['evaluation']['alarm_frames']} / {background['evaluation']['frames']}.\n\n"
        f"Predeclared promotion criterion passed: {summary['criterion_result']}.\n\n"
        "The scene labels are nominal sensor-local positions; actual path occupancy requires supported rail geometry. "
        "The full modeled shape is in provenance NPZ, observed support in predictions JSONL. "
        "No claim of field recall, hardware detection range, private-set recall, or calibrated Pandar128 performance follows.\n")
    print(json.dumps({k: summary[k] for k in ("evaluation_cases", "visible_cases", "zero_visibility_cases",
                                                 "object_detection_all_cases", "hazard_confirmation_all_cases",
                                                 "object_detection_given_support", "hazard_confirmation_given_support",
                                                 "criterion_result")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    make_report(args.run)


if __name__ == "__main__":
    main()
