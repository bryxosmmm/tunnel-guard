"""Stream comparisons of real recorded runs; alarm counts are not accuracy."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np

from .run import digest, write_json


class Statistics:
    def __init__(self):
        self.frames = 0
        self.status = Counter()
        self.health = Counter()
        self.geometry_valid = 0
        self.motion_valid = 0
        self.candidates = 0
        self.support = 0
        self.intersections = 0
        self.future_evidence = 0
        self.duplicate_evidence = 0
        self.far_candidates = 0
        self.processing = []

    def add(self, row):
        self.frames += 1
        self.status[row["status"]] += 1
        self.health[row["health"]] += 1
        self.geometry_valid += bool(row.get("geometry", {}).get("valid", False))
        self.motion_valid += bool(row.get("motion", {}).get("valid", False))
        self.processing.append(row["processing_s"])
        for obj in row["objects"]:
            self.candidates += 1
            self.support += obj["support_voxels"]
            self.intersections += obj.get("intersection_confirmed", False)
            self.far_candidates += obj["cluster_nearest_x_m"] >= 60
            for key in ("evidence_timestamps_s", "intersection_evidence_timestamps_s"):
                times = obj.get(key, [])
                self.future_evidence += any(t > row["timestamp_s"] for t in times)
                self.duplicate_evidence += len(times) != len(set(times))

    def result(self):
        return {
            "frames": self.frames,
            "status_frames": dict(self.status),
            "health_frames": dict(self.health),
            "geometry_valid_frames": self.geometry_valid,
            "motion_valid_frames": self.motion_valid,
            "candidate_observations": self.candidates,
            "support_voxel_observations": self.support,
            "confirmed_intersection_observations": self.intersections,
            "candidate_observations_at_least_60m": self.far_candidates,
            "histories_with_future_timestamps": self.future_evidence,
            "histories_with_duplicate_timestamps": self.duplicate_evidence,
            "processing_ms": {
                name: float(np.quantile(self.processing, quantile) * 1000)
                for name, quantile in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]
            }
            if self.processing
            else None,
        }


def difference(a, b, numeric):
    """Measure floating differences; retain exact comparison for IDs and decisions."""
    if type(a) is not type(b):
        return True
    if isinstance(a, dict):
        return a.keys() != b.keys() or any(difference(a[k], b[k], numeric) for k in a)
    if isinstance(a, list):
        return len(a) != len(b) or any(difference(x, y, numeric) for x, y in zip(a, b))
    if isinstance(a, float):
        numeric[0] = max(numeric[0], abs(a - b))
        return not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a != b


def compare(entry):
    before, after = Path(entry["before"]), Path(entry["after"])
    old_stats, new_stats = Statistics(), Statistics()
    fields = [
        "status",
        "nearest_obstacle_m",
        "objects",
        "geometry",
        "motion",
        "pose",
        "range_observability",
        "health",
        "health_reasons",
        "gap_reset",
    ]
    changed = Counter()
    examples = {key: [] for key in fields}
    numeric = [0.0]
    missing, extra, identity_errors = [], [], []
    with before.open() as old, after.open() as new:
        for a, b in zip_longest(old, new):
            x, y = json.loads(a) if a else None, json.loads(b) if b else None
            if x is not None:
                old_stats.add(x)
            if y is not None:
                new_stats.add(y)
            if x is None:
                extra.append(y["frame"])
                continue
            if y is None:
                missing.append(x["frame"])
                continue
            if any(
                x[k] != y[k]
                for k in (
                    "frame",
                    "measurement_timestamp_ns",
                    "record_timestamp_ns",
                    "sensor_frame",
                    "topic",
                )
            ):
                identity_errors.append([x["frame"], y["frame"]])
                continue
            for key in fields:
                if difference(x.get(key), y.get(key), numeric):
                    changed[key] += 1
                    if len(examples[key]) < 30:
                        examples[key].append(x["frame"])
    return {
        "bag": entry["bag"],
        "before": str(before),
        "after": str(after),
        "before_sha256": digest(before),
        "after_sha256": digest(after),
        "before_statistics": old_stats.result(),
        "after_statistics": new_stats.result(),
        "missing_after": missing,
        "extra_after": extra,
        "measurement_identity_errors": identity_errors,
        "changed_frames_by_field": dict(changed),
        "first_changed_frames": {k: v for k, v in examples.items() if v},
        "largest_compared_float_difference": numeric[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report path")
    panel = json.loads(args.panel.read_text())
    report = {
        "panel": str(args.panel),
        "panel_sha256": digest(args.panel),
        "accuracy": None,
        "limitations": [
            "Alarm counts on nonexhaustively labelled development recordings are not precision, recall or false-alarm rate.",
            "Repeated frames and candidates are not independent object events.",
            "Timing is descriptive; concurrent workloads and visualization settings differ.",
            "Float comparison uses absolute and relative tolerance 1e-10; discrete decisions and IDs are exact.",
        ],
        "bags": [],
    }
    for entry in panel["bags"]:
        row = compare(entry)
        report["bags"].append(row)
        print(
            json.dumps(
                {
                    "bag": row["bag"],
                    "frames": row["after_statistics"]["frames"],
                    "changed_frames_by_field": row["changed_frames_by_field"],
                    "missing_after": len(row["missing_after"]),
                }
            ),
            flush=True,
        )
    write_json(args.output, report)


if __name__ == "__main__":
    main()
