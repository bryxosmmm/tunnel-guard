"""Attribute the misses and the nuisance detections of a real-base synthetic panel.

`realistic_stress` inserts an object into a recorded frame and labels its measured support. This
reports what the detector then did with it, separating failure modes that need different fixes:

- the object produced no returns in this frame, a visibility fact of the real geometry and the
  measured shot placement rather than a detector failure, reported as a validity outcome;
- nothing the detector reported overlaps the object at all: a locality or segmentation failure;
- something overlaps but was never confirmed: a confirmation failure;
- something confirmed overlaps but is not a hazard relation: a relation failure;
- a confirmed hazard overlaps: a detection.

It also counts the detections the injection cannot explain, under both conventions a scorer could
use: frames whose status claims an obstacle, and individual objects reported. On a real recording
those are nuisance detections by construction, but they are not proof of an empty tunnel: the
recording was not exhaustively labelled, so a real object could be among them. That limit is
reported with the numbers instead of buried.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .evaluate import box_iou, match_objects

HAZARD_RELATIONS = ("intersecting", "unresolved")
ALARM_STATUSES = ("obstacle", "unresolved_obstacle")
KINDS = ("no_returns_from_object", "nothing_overlaps", "covered_by_a_larger_object",
         "overlaps_but_unconfirmed", "confirmed_not_hazard", "confirmed_hazard")


def load_lines(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def coverage_of(truth: dict, candidates: list[dict], grid: int = 6) -> float:
    """Fraction of a label box sampled on a grid that falls inside any candidate box.

    One-to-one IoU cannot express "reported, but boxed differently" when the label is a few returns
    and the prediction is a whole cluster. This asks only whether the reported boxes cover the
    labelled region at all.
    """
    lo, hi = np.asarray(truth["bbox_min"], dtype=float), np.asarray(truth["bbox_max"], dtype=float)
    axes = [np.linspace(lo[i], hi[i], grid) for i in range(3)]
    sample = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if not candidates:
        return 0.0
    inside = np.zeros(len(sample), dtype=bool)
    for obj in candidates:
        low, high = np.asarray(obj["bbox_min"], dtype=float), np.asarray(obj["bbox_max"], dtype=float)
        inside |= np.all((sample >= low) & (sample <= high), axis=1)
    return float(inside.mean())


def best_overlap(label: dict, pool: list[dict]) -> tuple[float, dict | None]:
    if not pool:
        return 0.0, None
    return max(((float(box_iou(obj, label)), obj) for obj in pool), key=lambda pair: pair[0])


def point_coverage(object_points: np.ndarray, objects: list[dict], extent_factor: float,
                   nominal_extent: list[float] | np.ndarray | None = None) -> dict:
    """Fraction of the object's own returns that a comparably sized box contains.

    The bounding box of an object's returns spans volume the sensor never observed, so one-to-one
    IoU cannot separate "reported, boxed differently" from "not reported". This asks the physical
    question instead: were the returns actually measured from this object reported? A box much
    larger than the object would cover them trivially, so only boxes whose extent stays within
    `extent_factor` of the object's own extent are allowed to count, and the count of objects that
    were rejected for that reason is reported rather than hidden.

    `nominal_extent` is the object's injected dimensions, when the case records them.
    """
    extent = object_points.max(axis=0) - object_points.min(axis=0)
    # At range the sensor sees essentially one face of the object, so its returns form a thin slice
    # and the limit derived from them alone can be smaller than the object itself, which makes the
    # rule reject boxes that are merely narrower than the object they contain. The injected
    # dimensions are known to the instrument, so the limit is taken over the larger of the observed
    # and nominal extent. This can only ever lift the limit, so it removes thin-slice rejections and
    # never adds any; containment of the returns is still required below.
    nominal = np.zeros(3) if nominal_extent is None else np.asarray(nominal_extent, dtype=float)
    limit = np.maximum(np.maximum(extent, nominal) * extent_factor, 0.05)
    eligible, oversized = [], 0
    for obj in objects:
        if np.all(np.asarray(obj["extent_m"], dtype=float) <= limit):
            eligible.append(obj)
        else:
            oversized += 1
    if not eligible:
        return {"coverage": 0.0, "covering": None, "eligible_objects": 0, "boxes_rejected_as_oversized": oversized}
    inside = np.zeros(len(object_points), dtype=bool)
    for obj in eligible:
        low, high = np.asarray(obj["bbox_min"], dtype=float), np.asarray(obj["bbox_max"], dtype=float)
        inside |= np.all((object_points >= low) & (object_points <= high), axis=1)
    covering = max(eligible, key=lambda obj: float(np.count_nonzero(
        np.all((object_points >= np.asarray(obj["bbox_min"], dtype=float))
               & (object_points <= np.asarray(obj["bbox_max"], dtype=float)), axis=1))))
    return {"coverage": float(inside.mean()), "eligible_objects": len(eligible),
            "boxes_rejected_as_oversized": oversized,
            "covering": {"path_relation": covering["path_relation"],
                         "reason": covering["path_relation_reason"], "confirmed": bool(covering["confirmed"]),
                         "distance_m": covering["distance_m"], "extent_m": covering["extent_m"]}}


def classify(label: dict, objects: list[dict], returns_from_object: int, minimum_iou: float,
             object_points: np.ndarray | None = None, extent_factor: float = 3.0,
             point_threshold: float = 0.6, nominal_extent: list[float] | None = None) -> dict:
    """One outcome per injected object, with the evidence behind it."""
    confirmed = [obj for obj in objects if obj["confirmed"]]
    hazards = [obj for obj in confirmed if obj["path_relation"] in HAZARD_RELATIONS]
    iou_all, near_all = best_overlap(label, objects)
    iou_confirmed, _ = best_overlap(label, confirmed)
    iou_hazard, near_hazard = best_overlap(label, hazards)
    in_scope = [obj for obj in hazards if abs(obj["distance_m"] - float(label["bbox_min"][0])) < 3.0]
    # Coverage over every reported box, not just hazard-class ones. A cluster that swallowed the
    # object hands it a box far larger than the object, so one-to-one IoU collapses while the
    # labelled region is fully inside something: that is a segmentation outcome, not an absence.
    coverage_any = coverage_of(label, objects)
    points = None if object_points is None or not len(object_points) else \
        point_coverage(np.asarray(object_points, dtype=float), objects, extent_factor, nominal_extent)

    if returns_from_object == 0:
        kind = "no_returns_from_object"
    elif iou_hazard >= minimum_iou:
        kind = "confirmed_hazard"
    elif iou_confirmed >= minimum_iou:
        kind = "confirmed_not_hazard"
    elif iou_all >= minimum_iou:
        kind = "overlaps_but_unconfirmed"
    elif coverage_any >= 0.5:
        kind = "covered_by_a_larger_object"
    else:
        kind = "nothing_overlaps"
    return {"kind": kind, "coverage_any": coverage_any, "coverage_hazard": coverage_of(label, in_scope),
            "iou_all": iou_all, "iou_confirmed": iou_confirmed, "iou_hazard": iou_hazard,
            "point_coverage": None if points is None else points["coverage"],
            "point_detected": None if points is None else bool(points["coverage"] >= point_threshold),
            "point_covering": None if points is None else points["covering"],
            "nearest_any": None if near_all is None else
            {"confirmed": near_all["confirmed"], "path_relation": near_all["path_relation"],
             "extent_m": near_all["extent_m"], "distance_m": near_all["distance_m"]},
            "nearest_hazard": None if near_hazard is None else
            {"reason": near_hazard["path_relation_reason"], "distance_m": near_hazard["distance_m"],
             "support_voxels": near_hazard["support_voxels"]}}


def bucket_of(case: dict) -> dict:
    dims = case["dimensions_m"]
    return {"range_m": case["range_m"], "lateral_m": case["lateral_m"],
            "lateral_mode": case.get("lateral_mode"),
            "size": None if dims is None else "x".join(f"{value:g}" for value in dims)}


def summarise(table: dict) -> dict:
    out = {}
    for name, counts in table.items():
        total = counts["total"]
        out[str(name)] = {"objects": total, "detected": counts["confirmed_hazard"],
                          "recall": counts["confirmed_hazard"] / total if total else None,
                          "point_detected": counts["point_detected"],
                          "point_recall": round(counts["point_detected"] / total, 4) if total else None,
                          "mean_point_coverage": round(counts["point_coverage_sum"] / counts["point_detected_total"], 4)
                          if counts["point_detected_total"] else None,
                          "mean_coverage_any": round(counts["coverage_any_sum"] / total, 4) if total else None,
                          "kinds": {kind: counts[kind] for kind in KINDS if counts[kind]}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a realistic_stress output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-threshold", type=float, default=0.6,
                        help="fraction of the object's own returns a comparable box must contain")
    parser.add_argument("--extent-factor", type=float, default=3.0,
                        help="how much larger than the object a box may be and still count as covering it")
    args = parser.parse_args()
    minimum_iou = float(json.loads((args.run / "experiment.json").read_text())["minimum_iou"])

    annotations = json.loads((args.run / "annotations.json").read_text())
    case_rows = json.loads((args.run / "cases.json").read_text())
    cases = {(row["case"], row["frame"]): row for row in case_rows}
    predictions = {(row["bag"], row["frame"]): row for row in load_lines(args.run / "predictions.jsonl")}
    inserted = {(row["case"], row["frame"]): np.asarray(row["points"], dtype=float)
                for row in load_lines(args.run / "inserted.jsonl")} \
        if (args.run / "inserted.jsonl").is_file() else {}

    records = []
    by_kind, by_range, by_size, by_lateral, by_lateral_mode = Counter(), defaultdict(Counter), \
        defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    negative_status, negative_relations = Counter(), Counter()
    negative_frames = negative_hazard_frames = 0
    unexplained_by_relation = Counter()
    synthetic = Counter()
    adjacent_hazard_frames = 0

    for frame in sorted(annotations["frames"], key=lambda f: (f["bag"], f["frame"])):
        key = (frame["bag"], frame["frame"])
        row = predictions.get(key)
        case = cases.get((int(frame["bag"].split("_")[1]), frame["frame"]))
        if row is None or case is None:
            continue
        objects, labels = row["objects"], frame["objects"]
        hazard_pairs = [(index, obj) for index, obj in enumerate(objects)
                        if obj["confirmed"] and obj["path_relation"] in HAZARD_RELATIONS]
        matched = match_objects([obj for _, obj in hazard_pairs], labels, minimum_iou)
        matched_indices = {hazard_pairs[index][0] for index, _, _ in matched}
        unexplained = [(index, obj) for index, obj in enumerate(objects) if index not in matched_indices]

        injected = bool(case.get("object_present", case["range_m"] is not None))
        # Historical outputs have only the old gauge-based label; retain their
        # interpretation when inspecting them, while every newly generated row
        # carries the explicit contour fields.
        full_intersection = bool(case.get("full_shape_reference_contour_intersects", bool(labels)))
        observed_intersection = bool(case.get("observed_support_reference_contour_intersects", bool(labels)))
        if not labels:
            if injected:
                synthetic["injected_frames"] += 1
                synthetic["full_shape_reference_contour_intersections"] += int(full_intersection)
                synthetic["observed_support_reference_contour_intersections"] += int(observed_intersection)
                if full_intersection and not int(case["returns_from_object"]):
                    synthetic["full_shape_intersection_without_returns"] += 1
                elif full_intersection and not observed_intersection:
                    synthetic["full_shape_intersection_support_outside_contour"] += 1
                elif not full_intersection:
                    synthetic["adjacent_frames"] += 1
                    adjacent_hazard_frames += int(row["status"] in ALARM_STATUSES)
                continue
            negative_frames += 1
            negative_status[row["status"]] += 1
            if row["status"] in ALARM_STATUSES:
                negative_hazard_frames += 1
            for _, obj in unexplained:
                unexplained_by_relation[obj["path_relation"]] += 1
                if obj["confirmed"] and obj["path_relation"] in HAZARD_RELATIONS:
                    unexplained_by_relation["confirmed_hazard"] += 1
                negative_relations[obj["path_relation"]] += 1
            continue

        synthetic["injected_frames"] += 1
        synthetic["full_shape_reference_contour_intersections"] += int(full_intersection)
        synthetic["observed_support_reference_contour_intersections"] += int(observed_intersection)

        decision = classify(labels[0], objects, int(case["returns_from_object"]), minimum_iou,
                            inserted.get((case["case"], case["frame"])), args.extent_factor,
                            args.point_threshold, case.get("dimensions_m"))
        decision.update({"case": case["case"], "frame": case["frame"], "bag": key[0],
                         "status": row["status"], "detections": len(objects),
                         "returns_from_object": int(case["returns_from_object"]),
                         "rays_hitting_object": int(case["rays_hitting_object"]),
                         "strict_matched": bool(matched), "unexplained_hazards": sum(
                             1 for _, obj in unexplained
                             if obj["confirmed"] and obj["path_relation"] in HAZARD_RELATIONS),
                         "nearest_obstacle_m": row["nearest_obstacle_m"], **bucket_of(case)})
        records.append(decision)
        by_kind[decision["kind"]] += 1
        for table, name in ((by_range, decision["range_m"]), (by_size, decision["size"]),
                            (by_lateral, decision["lateral_m"]),
                            (by_lateral_mode, decision["lateral_mode"])):
            table[name]["total"] += 1
            table[name][decision["kind"]] += 1
            table[name]["coverage_any_sum"] += decision["coverage_any"]
            if decision["point_coverage"] is not None:
                table[name]["point_detected_total"] += 1
                table[name]["point_coverage_sum"] += decision["point_coverage"]
                table[name]["point_detected"] += int(decision["point_detected"])
        for _, obj in unexplained:
            unexplained_by_relation[obj["path_relation"]] += 1
            if obj["confirmed"] and obj["path_relation"] in HAZARD_RELATIONS:
                unexplained_by_relation["confirmed_hazard"] += 1

    visible = [record for record in records if record["returns_from_object"] > 0]
    detected = [record for record in visible if record["kind"] == "confirmed_hazard"]
    with_points = [record for record in records if record["point_coverage"] is not None]
    point_hits = [record for record in with_points if record["point_detected"]]
    report = {
        "label": "HYBRID SYNTHETIC",
        "run": str(args.run),
        "minimum_iou": minimum_iou,
        "point_criterion": {"point_threshold": args.point_threshold, "extent_factor": args.extent_factor,
                            "definition": ("fraction of the object's own returns contained in a reported box whose "
                                           "extent stays within extent_factor of the object's own extent")},
        "frames": {"scored_observed_support_intersections": len(records),
                   "injection_free": negative_frames},
        "objects": {"scored_observed_support_intersections": len(records),
                    "with_returns": len(visible), "no_returns_visibility_outcome": 0},
        "synthetic_provenance": {
            "full_shape_and_observed_support_are_separate": True,
            "reference_relation": ("Intersection is with the configured reference contour, not a validated "
                                   "vehicle swept envelope or a field collision label."),
            "counts": dict(synthetic),
            "adjacent_frames_claiming_hazard": adjacent_hazard_frames,
            "conditional_scoring": ("Recall is conditioned on an injected full shape and observed support that "
                                    "both intersect the selected reference contour."),
        },
        "outcome_kinds": {kind: by_kind[kind] for kind in KINDS},
        "recall": {
            "strict_matched_of_all_injected": round(sum(r["strict_matched"] for r in records) / len(records), 4),
            "detected_of_all_injected": round(len(detected) / len(records), 4),
            "detected_of_visible": round(len(detected) / len(visible), 4) if visible else None,
            "any_object_overlaps_of_visible": round(
                sum(r["iou_all"] >= minimum_iou for r in visible) / len(visible), 4) if visible else None,
            "point_detected_of_frames_with_returns": round(len(point_hits) / len(with_points), 4) if with_points else None,
            "point_detected_of_all_injected": round(len(point_hits) / len(records), 4),
            "counts": {"injected": len(records), "with_returns": len(with_points),
                       "point_detected": len(point_hits), "strict_detected": len(detected)},
        },
        "by_range_m": summarise(by_range),
        "by_size_m": summarise(by_size),
        "by_lateral_m": summarise(by_lateral),
        "by_lateral_mode": summarise(by_lateral_mode),
        "nuisance": {
            "injection_free_frames": negative_frames,
            "frames_claiming_an_obstacle": negative_hazard_frames,
            "unexplained_hazard_objects_on_injected_frames": unexplained_by_relation["confirmed_hazard"],
            "frame_status": dict(negative_status),
            "objects_by_relation": dict(negative_relations),
            "unexplained_objects_by_relation_on_injected_frames": dict(unexplained_by_relation),
            "limit": ("The recording was not exhaustively labelled, so an unexplained detection is a nuisance "
                      "detection by construction but is not proof that the tunnel was empty: a real object could "
                      "be among them. These are the injection-free frames of this panel, not certified-empty scenes."),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output.with_suffix("").with_name(args.output.stem + ".frames.jsonl")).open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_range_m",)}, indent=2))
    print(f"Evidence: {args.output}")


if __name__ == "__main__":
    main()
