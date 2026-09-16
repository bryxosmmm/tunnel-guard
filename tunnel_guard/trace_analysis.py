"""Offline annotation-box support accounting; never used as detector input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np

from .evaluate import box_iou, validate_annotations
from .run import digest, write_json


STAGES = ("decoded_points", "range_points", "registered_points", "cropped_points", "geometry_voxel_points",
          "context_before_background", "context_after_background", "cluster_points")


def inside(points, box):
    return np.all((points >= box["bbox_min"]) & (points <= box["bbox_max"]), axis=1)


def analyze(run: Path, annotations: dict):
    rows = {}
    for file in run.glob("*.jsonl"):
        with file.open() as stream:
            for line in stream:
                row = json.loads(line)
                rows[(row["bag"], row["frame"])] = row
    observations = []
    for frame in annotations["frames"]:
        row = rows.get((frame["bag"], frame["frame"]))
        for box in frame["objects"]:
            record = {"bag": frame["bag"], "frame": frame["frame"], "event_id": box["event_id"]}
            if row is None:
                observations.append(record | {"state": "missing_prediction"})
                continue
            record.update(state="observed", measurement_timestamp_ns=row["measurement_timestamp_ns"],
                          coordinate_frame=row["coordinate_frame"], health=row["health"],
                          geometry=row["pipeline"]["geometry"], association=row["pipeline"]["association"],
                          segmentation_state=row["pipeline"]["segmentation"]["state"])
            if "diagnostic_points" in row:
                with np.load(run / row["diagnostic_points"]) as arrays:
                    record["stage_box_points"] = {name: {"state": "captured", "points": int(inside(arrays[name], box).sum())}
                        if name in arrays else {"state": "not_captured", "points": None} for name in STAGES}
                    members = []
                    if "cluster_labels" in arrays:
                        labels = arrays["cluster_labels"][inside(arrays["cluster_points"], box)]
                        reasons = {x["component_id"]: x for x in row["pipeline"]["segmentation"].get("components", [])}
                        for label in np.unique(labels):
                            objects = [o for o in row["objects"] if o["component_id"] == label]
                            members.append({"component_id": int(label), "points_in_box": int(np.count_nonzero(labels == label)),
                                "decision": reasons.get(int(label), {"reason": "noise_or_unrecorded"}),
                                "track_id": objects[0]["track_id"] if objects else None,
                                "confirmed": objects[0]["confirmed"] if objects else None})
                    record["components_with_box_support"] = members
            else:
                record["stage_box_points"] = None
            best = max(row["objects"], key=lambda o: box_iou(o, box), default=None)
            if best is not None:
                record["best_candidate"] = {k: best[k] for k in ("component_id", "track_id", "confirmed", "hits", "confirmation",
                    "path_relation", "distance_m", "cluster_nearest_x_m", "supported_envelope_nearest_x_m", "bbox_min", "bbox_max")}
                record["best_candidate"].update(iou=box_iou(best, box),
                    matches_provisional_box=bool(best["confirmed"] and box_iou(best, box) >= annotations["minimum_iou"]))
            else:
                record["best_candidate"] = None
            observations.append(record)
    distance_cases = []
    for (bag, frame), row in rows.items():
        for obj in row["objects"]:
            if obj["confirmed"] and obj["path_relation"] == "intersecting" and obj["supported_envelope_nearest_x_m"] is not None:
                gap = obj["supported_envelope_nearest_x_m"] - obj["cluster_nearest_x_m"]
                if gap > 1e-6:
                    distance_cases.append({"bag": bag, "frame": frame, "track_id": obj["track_id"],
                        "component_id": obj["component_id"], "gap_m": gap,
                        "cluster_min_x_m": obj["cluster_nearest_x_m"],
                        "supported_min_x_m": obj["supported_envelope_nearest_x_m"],
                        "trace_captured": "diagnostic_points" in row})
    distance_cases.sort(key=lambda c: (-c["gap_m"], c["bag"], c["frame"], c["track_id"]))
    return {"scope": "Diagnostic box support, not semantic point labels or field safety accuracy",
        "label_status": annotations["label_status"], "annotation_provenance": annotations.get("provenance"),
        "source_events": len({(f['bag'], o['event_id']) for f in annotations['frames'] for o in f['objects']}),
        "observations": observations, "processed_frames": len(rows),
        "status_counts": dict(Counter(r['status'] for r in rows.values())),
        "candidate_instances": sum(len(r['objects']) for r in rows.values()),
        "confirmed_instances": sum(o['confirmed'] for r in rows.values() for o in r['objects']),
        "confirmed_hazard_instances": sum(o['confirmed'] and o['path_relation'] != 'adjacent' for r in rows.values() for o in r['objects']),
        "rejected_components": dict(sum((Counter(r['pipeline']['segmentation'].get('rejected', {})) for r in rows.values()), Counter())),
        "confirmed_intersections_with_distance_gap": len(distance_cases), "largest_distance_gaps": distance_cases[:20],
        "captured_distance_cases": [c for c in distance_cases if c['trace_captured']][:20],
        "false_positive_rate": None, "false_positive_validity": "No independently verified negative intervals"}


def compare_runs(before: Path, after: Path):
    def load(run):
        result = {}
        for path in run.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                result[(row['bag'], row['frame'])] = row
        return result
    old, new = load(before), load(after)
    common = sorted(old.keys() & new.keys())
    changed_decisions, distance_changes, nearest_changes = [], [], []
    witness_objects, witness_errors, membership_errors = 0, [], 0
    fields = ('component_id', 'track_id', 'bbox_min', 'bbox_max', 'confirmed', 'confirmation', 'hits', 'path_relation')
    for key in common:
        a, b = old[key], new[key]
        if (a['status'] != b['status'] or
                [[o[k] for k in fields] for o in a['objects']] != [[o[k] for k in fields] for o in b['objects']]):
            changed_decisions.append(list(key))
        aa = {o['component_id']: o for o in a['objects']}
        for obj in b['objects']:
            previous = aa.get(obj['component_id'])
            if previous is not None and obj['distance_m'] != previous['distance_m']:
                distance_changes.append({'bag': key[0], 'frame': key[1], 'component_id': obj['component_id'],
                    'before_m': previous['distance_m'], 'after_m': obj['distance_m'],
                    'definition_after': obj['distance_method'], 'confirmed': obj['confirmed'], 'relation': obj['path_relation']})
        if a['nearest_obstacle_m'] != b['nearest_obstacle_m']:
            nearest_changes.append({'bag': key[0], 'frame': key[1], 'before_m': a['nearest_obstacle_m'], 'after_m': b['nearest_obstacle_m']})
        if 'diagnostic_points' in b:
            with np.load(after / b['diagnostic_points']) as arrays:
                if 'cluster_labels' not in arrays:
                    continue
                for obj in b['objects']:
                    mask = arrays['cluster_labels'] == obj['component_id']
                    if obj['distance_method'] == 'supported_envelope_min_x':
                        mask &= arrays['cluster_core']
                    elif obj['distance_method'] == 'unresolved_nominal_envelope_min_x':
                        mask &= ~arrays['cluster_observed'] & arrays['cluster_nominal_overlap']
                    points = arrays['cluster_points'][mask]
                    witness_objects += 1
                    if not len(points):
                        membership_errors += 1
                        continue
                    witness_errors.append(abs(float(points[:, 0].min()) - obj['distance_m']))
                    membership_errors += int(not np.any(np.all(points == obj['distance_support_point'], axis=1)))
    return {'before': str(before), 'after': str(after), 'compared_frames': len(common),
        'missing_before': [list(x) for x in sorted(new.keys() - old.keys())],
        'missing_after': [list(x) for x in sorted(old.keys() - new.keys())],
        'changed_detection_frames': changed_decisions,
        'distance_changes': distance_changes, 'nearest_changes': nearest_changes,
        'witness_objects_on_captured_frames': witness_objects, 'witness_membership_errors': membership_errors,
        'max_witness_x_error_m': max(witness_errors, default=None),
        'interpretation': 'Different distance definitions, not detection range improvement. No false-positive rate without verified negatives.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare-to", type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output already exists; preserve prior evidence")
    annotations = json.loads(a.annotations.read_text())
    validate_annotations(annotations)
    report = analyze(a.run, annotations) | {"run": str(a.run), "annotations_sha256": digest(a.annotations)}
    if a.compare_to is not None:
        report['comparison'] = compare_runs(a.compare_to, a.run)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, report)
    print(json.dumps({k: report[k] for k in ('processed_frames', 'source_events', 'status_counts', 'candidate_instances',
        'confirmed_instances', 'confirmed_intersections_with_distance_gap', 'largest_distance_gaps')}, indent=2))


if __name__ == '__main__':
    main()
