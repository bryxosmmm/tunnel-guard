"""Evaluate localized objects without treating unlabelled recordings as negatives."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .run import digest, write_json


def box_iou(a: dict, b: dict) -> float:
    amin, amax = np.asarray(a["bbox_min"]), np.asarray(a["bbox_max"])
    bmin, bmax = np.asarray(b["bbox_min"]), np.asarray(b["bbox_max"])
    intersection = float(np.prod(np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))))
    union = float(np.prod(amax - amin) + np.prod(bmax - bmin) - intersection)
    return intersection / union if union > 0 else 0.0


def match_objects(predictions: list[dict], truth: list[dict], minimum_iou: float) -> list[tuple[int, int, float]]:
    if not predictions or not truth:
        return []
    iou = np.array([[box_iou(p, g) for g in truth] for p in predictions])
    # Maximize the number of valid matches before their overlap. Invalid pairs
    # must not steal a match through a lower aggregate Hungarian assignment cost.
    utility = (iou >= minimum_iou).astype(float) * (min(iou.shape) + 1) + iou
    rows, cols = linear_sum_assignment(-utility)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if iou[i, j] >= minimum_iou]


def validate_annotations(annotations: dict):
    if annotations["label_status"] not in ("human_verified", "provisional_geometry", "synthetic_exact"):
        raise ValueError("Unknown label status")
    if not 0 < annotations["minimum_iou"] <= 1:
        raise ValueError("minimum_iou must be in (0,1]")
    if annotations["prediction_scope"] not in ("collision_hazards", "near_track_objects"):
        raise ValueError("prediction_scope must explicitly select hazards or all near-track objects")
    seen = set()
    for frame in annotations["frames"]:
        key = (frame["bag"], frame["frame"])
        if key in seen:
            raise ValueError(f"Duplicate annotation: {key}")
        seen.add(key)
        if not isinstance(frame["exhaustive"], bool):
            raise ValueError("Every annotated frame must declare exhaustiveness")
        ids = set()
        for obj in frame["objects"]:
            if obj["event_id"] in ids:
                raise ValueError("Duplicate event_id within a frame")
            ids.add(obj["event_id"])
            lo, hi = np.asarray(obj["bbox_min"]), np.asarray(obj["bbox_max"])
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
                raise ValueError("Annotations require finite, positive-volume 3D boxes")


def evaluate_frames(predictions: dict, annotations: dict) -> dict:
    validate_annotations(annotations)
    records, errors, overlaps = [], [], []
    tp = fn = fp = exhaustive_tp = exhaustive_fn = 0
    events = defaultdict(lambda: {"annotated_frames": 0, "matched_frames": 0, "first_detection_distance_m": None})
    missing = []
    exhaustive_count = 0
    for frame in sorted(annotations["frames"], key=lambda f: (f["bag"], f["frame"])):
        key = (frame["bag"], frame["frame"])
        if key not in predictions:
            missing.append(list(key))
            continue
        prediction = predictions[key]
        objects = [p for p in prediction["objects"] if p["confirmed"]]
        if annotations["prediction_scope"] == "collision_hazards":
            objects = [p for p in objects if p["path_relation"] in ("intersecting", "unresolved")]
        truth = frame["objects"]
        matches = match_objects(objects, truth, annotations["minimum_iou"])
        found = {j: i for i, j, _ in matches}
        tp += len(matches)
        fn += len(truth) - len(matches)
        frame_fp = None
        if frame["exhaustive"]:
            exhaustive_count += 1
            frame_fp = len(objects) - len(matches)
            fp += frame_fp
            exhaustive_tp += len(matches)
            exhaustive_fn += len(truth) - len(matches)
        for j, obj in enumerate(truth):
            event = events[f"{frame['bag']}:{obj['event_id']}"]
            event["annotated_frames"] += 1
            if j in found:
                p = objects[found[j]]
                event["matched_frames"] += 1
                distance = float(obj["bbox_min"][0])
                if event["first_detection_distance_m"] is None:
                    event["first_detection_distance_m"] = distance
                errors.append(abs(p["distance_m"] - distance))
        overlaps.extend(v for _, _, v in matches)
        records.append({"bag": frame["bag"], "frame": frame["frame"], "tp": len(matches),
                        "fn": len(truth) - len(matches), "fp": frame_fp,
                        "status": prediction["status"], "matches": matches})
    precision = exhaustive_tp / (exhaustive_tp + fp) if exhaustive_tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    # F1 only combines metrics from the same exhaustive subset.
    exhaustive_recall = exhaustive_tp / (exhaustive_tp + exhaustive_fn) if exhaustive_tp + exhaustive_fn else None
    denominator = 2 * exhaustive_tp + fp + exhaustive_fn
    f1 = 2 * exhaustive_tp / denominator if denominator else None
    validity = []
    if annotations["label_status"] == "provisional_geometry":
        validity.append("Provisional geometric annotations are not independently human-verified ground truth")
    if exhaustive_count < len(records):
        validity.append("Unmatched predictions on non-exhaustive frames are unscored; precision is not a whole-panel metric")
    if missing:
        validity.append("Missing prediction frames: panel incomplete")
    if not events:
        validity.append("No annotated positive events")
    origins = Counter(obj.get("annotation_origin", "unspecified")
                      for frame in annotations["frames"] for obj in frame["objects"])
    if origins["detector_track_propagation"]:
        validity.append("Labels include detector-track propagation; retain author review provenance and do not treat this panel as an independent holdout")
    if annotations.get("box_semantics", "unspecified") == "unspecified":
        validity.append("Annotation box semantics are unspecified: observed support and full object volume may not be comparable")
    return {"label_status": annotations["label_status"], "prediction_scope": annotations["prediction_scope"],
            "annotation_origins": dict(origins),
            "box_semantics": annotations.get("box_semantics", "unspecified"),
            "minimum_3d_iou": annotations["minimum_iou"],
            "evaluated_frames": len(records), "missing_frames": missing, "exhaustive_frames": exhaustive_count,
            "tp": tp, "fn": fn, "fp_exhaustive_only": fp, "annotated_object_recall": recall,
            "precision_exhaustive_only": precision, "recall_exhaustive_only": exhaustive_recall,
            "f1_exhaustive_only": f1,
            "event_recall": sum(v["matched_frames"] > 0 for v in events.values()) / len(events) if events else None,
            "distance_mae_m": float(np.mean(errors)) if errors else None,
            "matched_mean_3d_iou": float(np.mean(overlaps)) if overlaps else None,
            "validity_warnings": validity, "events": dict(events), "frames": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text())
    needed = {(f["bag"], f["frame"]) for f in annotations["frames"]}
    predictions = {}
    for path in sorted(args.run.glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["bag"], row["frame"])
                if key in needed:
                    predictions[key] = row
    report = evaluate_frames(predictions, annotations)
    report.update(annotations_sha256=digest(args.annotations), run=str(args.run))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("frames", "events")}, indent=2))
    if report["missing_frames"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
