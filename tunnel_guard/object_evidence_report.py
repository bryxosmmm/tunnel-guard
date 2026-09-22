"""Trace real annotated support and separate presence scoring from hazard decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from .evaluate import box_iou, evaluate_frames
from .run import digest, write_json
from .trace_analysis import STAGES, inside


def rows(root, bag):
    path = root / f"{bag}.jsonl"
    if not path.exists():
        path = path.with_suffix(".jsonl.gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        yield from map(json.loads, stream)


def localized_evidence(root, row, box, threshold):
    best = max(row["objects"], key=lambda obj: box_iou(obj, box), default=None)
    evidence = {"bag": row["bag"], "frame": row["frame"],
                "annotation_origin": box.get("annotation_origin", "unspecified"),
                "best_candidate": best, "iou": box_iou(best, box) if best else 0.0,
                "stage_box_points": None, "component_points_inside_box": None}
    evidence["candidate_matches_geometry"] = evidence["iou"] >= threshold
    if best is not None:
        predicted_volume = float(np.prod(np.subtract(best["bbox_max"], best["bbox_min"])))
        label_volume = float(np.prod(np.subtract(box["bbox_max"], box["bbox_min"])))
        evidence["predicted_to_label_volume"] = predicted_volume / label_volume
        evidence["volume_only_iou_upper_bound"] = min(predicted_volume, label_volume) / max(predicted_volume, label_volume)
        evidence["size_alone_precludes_match"] = evidence["volume_only_iou_upper_bound"] < threshold
    if row.get("diagnostic_points"):
        path = root / row["diagnostic_points"]
        evidence["diagnostics_sha256"] = digest(path)
        with np.load(path, allow_pickle=False) as arrays:
            evidence["stage_box_points"] = {name: int(inside(arrays[name], box).sum())
                                           for name in STAGES if name in arrays}
            if best is not None and "cluster_labels" in arrays:
                mask = inside(arrays["cluster_points"], box)
                evidence["component_points_inside_box"] = int(np.count_nonzero(
                    mask & (arrays["cluster_labels"] == best["component_id"])))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    before, after, output = (Path(plan[key]) for key in ("before", "after", "output"))
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (before, after)]
    if any("finished_unix_s" not in manifest for manifest in manifests):
        raise ValueError("Both real replays must finish before evaluation")
    configs = [json.loads((root / "detector.json").read_text()) for root in (before, after)]
    if configs[0] != configs[1] or configs[0]["seed"] != plan["seed"]:
        raise ValueError("Configuration or seed changed")
    if manifests[0]["native_accelerator"] != manifests[1]["native_accelerator"]:
        raise ValueError("Controls require the same native binary and sources")
    # Historical outputs must be scored by their actual frozen contract, not by
    # inventing presence flags absent from the original output.
    historical_path = before / "source/tunnel_guard/evaluate.py"
    spec = importlib.util.spec_from_file_location("tunnel_guard._historical_evaluate", historical_path)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    annotations = {name: json.loads(Path(path).read_text()) for name, path in plan["annotations"].items()}
    needed = {(frame["bag"], frame["frame"]) for annotation in annotations.values() for frame in annotation["frames"]}
    predictions = [{}, {}]
    reports = []
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", plan)
    signature = ("track_id", "component_id", "bbox_min", "bbox_max", "support_voxels", "confirmed",
                 "intersection_confirmed", "path_relation", "distance_m")
    with gzip.GzipFile(filename=str(output / "frames.jsonl.gz"), mode="wb", mtime=0) as archive:
        for bag in plan["bags"]:
            counts = Counter()
            statuses = [Counter(), Counter()]
            changes = []
            for old, new in zip_longest(rows(before, bag), rows(after, bag)):
                if old is None or new is None:
                    raise ValueError(f"Frame count mismatch: {bag}")
                identity = ("frame", "measurement_timestamp_ns", "source_scan_id")
                if any(old[key] != new[key] for key in identity):
                    raise ValueError(f"Source scan mismatch: {bag}")
                counts["frames"] += 1
                for index, row in enumerate((old, new)):
                    statuses[index][row["status"]] += 1
                    if (bag, row["frame"]) in needed:
                        predictions[index][(bag, row["frame"])] = row
                same_objects = ([{key: obj[key] for key in signature} for obj in old["objects"]]
                                == [{key: obj[key] for key in signature} for obj in new["objects"]])
                same_alarm = all(old[key] == new[key] for key in
                                 ("status", "nearest_obstacle_m", "nearest_candidate_m", "nearest_unresolved_range_m"))
                accumulated_changes = sum(a["accumulated_support_voxels"] != b["accumulated_support_voxels"]
                                          for a, b in zip(old["objects"], new["objects"]))
                counts["accumulated_count_changes"] += accumulated_changes
                counts["presence_confirmed_observations"] += sum(obj["presence_confirmed"] for obj in new["objects"])
                counts["confirmed_hazard_observations"] += sum(obj["confirmed"] and obj["path_relation"] in
                                                               ("intersecting", "unresolved") for obj in new["objects"])
                record = {"bag": bag, "frame": new["frame"], "source_scan_id": new["source_scan_id"],
                          "status": new["status"], "objects_and_hazards_identical": same_objects,
                          "alarm_outputs_identical": same_alarm, "accumulated_count_changes": accumulated_changes}
                if not same_objects or not same_alarm:
                    changes.append(record)
                archive.write((json.dumps(record, allow_nan=False) + "\n").encode())
            reports.append({"bag": bag, **dict(counts), "before_statuses": dict(statuses[0]),
                            "after_statuses": dict(statuses[1]), "changes": changes})
    if sum(report["frames"] for report in reports) != plan["expected_frames"]:
        raise ValueError("Required six-recording panel is incomplete")
    label_reports = {}
    for name, annotation in annotations.items():
        evidence = []
        for frame in annotation["frames"]:
            key = (frame["bag"], frame["frame"])
            for box in frame["objects"]:
                evidence.append({"before": localized_evidence(before, predictions[0][key], box, annotation["minimum_iou"]),
                                 "after": localized_evidence(after, predictions[1][key], box, annotation["minimum_iou"])})
        label_reports[name] = {"before": historical.evaluate_frames(predictions[0], annotation),
                               "after": evaluate_frames(predictions[1], annotation),
                               "annotation_sha256": digest(Path(plan["annotations"][name])), "evidence": evidence,
                               "candidate_geometry_matches_before": sum(e["before"]["candidate_matches_geometry"] for e in evidence),
                               "candidate_geometry_matches_after": sum(e["after"]["candidate_matches_geometry"] for e in evidence)}
    anchor = plan["anchor"]
    anchor_records = [e for report in label_reports.values() for e in report["evidence"]
                      if (e["after"]["bag"], e["after"]["frame"]) == (anchor["bag"], anchor["frame"])
                      and e["after"]["annotation_origin"] == "manual_anchor"]
    anchor_ok = bool(anchor_records) and all(e["after"]["candidate_matches_geometry"] and
        e["after"]["best_candidate"]["presence_confirmed"] for e in anchor_records)
    criteria = {"manual_anchor_presence_match": anchor_ok,
                "alarm_and_observed_geometry_unchanged": not any(r["changes"] for r in reports),
                "same_detector_config_and_native_binary": True,
                "precision_established": False, "independent_holdout_recall_established": False}
    result = {"criteria": criteria, "accepted_contract_repair": anchor_ok and criteria["alarm_and_observed_geometry_unchanged"],
              "bags": reports, "annotations": label_reports,
              "interpretation": "Presence/hazard contract repair, not new geometric detections. Box-contained returns are not semantic point labels; raw slots are not independent beams. Repeated frames and propagated labels are not independent events. No exhaustive negatives establish precision.",
              "provenance": {"recipe_sha256": digest(args.experiment), "reporter_sha256": digest(Path(__file__)),
                             "historical_evaluator_sha256": digest(historical_path),
                             "current_evaluator_sha256": digest(Path(__file__).with_name("evaluate.py")),
                             "before_manifest": manifests[0], "after_manifest": manifests[1],
                             "frames_sha256": digest(output / "frames.jsonl.gz")}}
    write_json(output / "summary.json", result)
    print(json.dumps({"criteria": criteria, "labels": {name: {"before_tp": r["before"]["tp"],
          "after_tp": r["after"]["tp"], "geometric_matches": r["candidate_geometry_matches_after"]}
          for name, r in label_reports.items()}}, indent=2))
    if not result["accepted_contract_repair"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
