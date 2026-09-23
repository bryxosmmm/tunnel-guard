"""Compare real detector runs without interpreting unlabelled alarms as accuracy.

Two modes share one streaming comparison engine:

* Historical ``--before DIR --after DIR --output FILE`` compares the result JSONL files
  found in two run directories, deriving the bag panel from the directories themselves.
* Explicit recipe ``--experiment FILE`` reads a fixed comparison recipe (run roots,
  absolute/relative tolerance, the fixed nine-bag panel, the detector config and the saved
  read-only input inventory), writes ``results/issue-5-full-regression-20260922.json`` and
  additionally validates processed/duplicate/subsample counts against that inventory.

Both runs are streamed row by row and paired by frame, so a panel with millions of objects
never has to fit in memory. Every material mismatch is written to a sidecar JSONL as
``bag/frame/path/old/new`` and aggregated per normalized field path with counts and
maximum absolute/relative deviations; exact-but-within-tolerance float differences are
aggregated as counts and maxima only. Missing, extra, duplicate, out-of-order or malformed
rows are failures, never a silent intersection. Only continuous floating estimates use the
declared tolerance; identifiers, counters, booleans, list lengths, keys and source/evidence
timestamps compare exactly. The only known exclusions are the added top-level
``sensor_attributes`` summary and the operational timing fields, which are reported rather
than compared. Timing, throughput and RSS are reported, never used to assert a detector
decision, field latency or recall.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from .run import digest, write_json

# Operational timing fields: measured per run, reported, never treated as detector evidence.
TIMING_FIELDS = (
    "processing_s", "motion_s", "mounting_observation_s", "deserialize_s", "decode_s",
    "ingestion_s", "inference_s", "read_and_process_s", "diagnostic_write_s", "visualization_s",
)
TIMING_FIELD_SET = frozenset(TIMING_FIELDS)
# The only other known top-level exclusion: the optional summary added by the sensor fix.
SENSOR_ATTRIBUTE_FIELD = "sensor_attributes"
SENSOR_ATTRIBUTE_KEYS = ("intensity", "ring", "raw_time")
# A summary of preserved sensor fields, not a calibration or detector-promotion claim.
SENSOR_ATTRIBUTE_UNKNOWN_FLAGS = {
    "profile_confirmed": False,
    "firing_identity": "unknown",
    "return_multiplicity": "unknown",
    "intensity_calibration": "unverified",
}
SENSOR_FIELD_COUNT_KEYS = ("dtype", "units", "count", "valid_count", "invalid_count")
# Statuses that hold an alarm-like episode open, matching the run summary's episode definition.
ACTIVE_STATUSES = ("obstacle", "unresolved_obstacle")
WARNING = (
    "Counts are algorithm outputs, not accuracy or false-alarm rates. Timing, throughput and "
    "RSS are reported, never used to assert a detector decision, field latency or recall. Check "
    "export settings and concurrent local workloads before interpreting timing."
)


class _Missing:
    """Sentinel for a key or element present on one side only."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<absent>"


MISSING = _Missing()


def _json_safe(value):
    """Recursively convert a comparison value into a JSON-serializable form."""
    if value is MISSING:
        return "<absent>"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


_INDEX = re.compile(r"\[\d+\]")


def _normalize(path: str) -> str:
    """Collapse concrete list indices so field paths aggregate across frames."""
    return _INDEX.sub("[]", path)


@lru_cache(maxsize=None)
def _timestamp_key(key: str) -> bool:
    return "timestamp" in key or key in {"last_observed_s", "start_s", "end_s"}


def _is_timestamp_path(path: str) -> bool:
    key = path.rsplit(".", 1)[-1].split("[", 1)[0]
    return _timestamp_key(key)


def diff_values(old, new, path, material, tolerated, atol, rtol):
    """Diff two decoded values, routing exact and tolerance comparisons to callbacks."""
    if isinstance(old, dict) or isinstance(new, dict):
        if not (isinstance(old, dict) and isinstance(new, dict)):
            material(path, old, new, "type")
            return
        old_keys, new_keys = set(old), set(new)
        for key in sorted(old_keys - new_keys):
            material(f"{path}.{key}", old[key], MISSING, "missing_key")
        for key in sorted(new_keys - old_keys):
            material(f"{path}.{key}", MISSING, new[key], "extra_key")
        for key in sorted(old_keys & new_keys):
            diff_values(old[key], new[key], f"{path}.{key}", material, tolerated, atol, rtol)
        return
    if isinstance(old, list) or isinstance(new, list):
        if not (isinstance(old, list) and isinstance(new, list)):
            material(path, old, new, "type")
            return
        if len(old) != len(new):
            material(path, len(old), len(new), "list_length")
        for index in range(min(len(old), len(new))):
            diff_values(old[index], new[index], f"{path}[{index}]", material, tolerated, atol, rtol)
        for index in range(len(new), len(old)):
            material(f"{path}[{index}]", old[index], MISSING, "missing_element")
        for index in range(len(old), len(new)):
            material(f"{path}[{index}]", MISSING, new[index], "extra_element")
        return
    if old is MISSING or new is MISSING:
        material(path, old, new, "missing")
        return
    if isinstance(old, bool) or isinstance(new, bool):
        if type(old) is not type(new) or old != new:
            material(path, old, new, "bool")
        return
    if isinstance(old, int) and isinstance(new, int):
        if old != new:
            material(path, old, new, "integer")
        return
    if isinstance(old, float) or isinstance(new, float):
        if type(old) is not float or type(new) is not float:
            material(path, old, new, "type")
            return
        first, second = float(old), float(new)
        if first == second:
            return
        if _is_timestamp_path(path):
            # Source and evidence timestamps are evidence: exact, never tolerated.
            if first != second:
                material(path, old, new, "timestamp")
            return
        if math.isnan(first) or math.isnan(second):
            if not (math.isnan(first) and math.isnan(second)):
                material(path, old, new, "nan")
            return
        if math.isinf(first) or math.isinf(second):
            if first != second:
                material(path, old, new, "infinite")
            return
        absolute = abs(first - second)
        relative = absolute / abs(second) if second else (0.0 if absolute == 0.0 else math.inf)
        if absolute <= atol + rtol * abs(second):
            tolerated(path, absolute, relative)
        else:
            material(path, old, new, "float", absolute, relative)
        return
    if type(old) is not type(new) or old != new:
        material(path, old, new, "value")


class BagComparison:
    """Per-bag streaming sink: sidecar records, aggregates and fail-closed evidence."""

    def __init__(self, bag: str, atol: float, rtol: float, writer, strict: bool = True):
        self.bag = bag
        self.atol = atol
        self.rtol = rtol
        self.writer = writer
        self.strict = strict
        self.frame = None
        self.material = {}
        self.tolerated = {}
        self.material_total = 0
        self.tolerated_total = 0
        self.frames_compared = 0
        self.missing_after = []
        self.extra_after = []
        self.gaps = []
        self.failures = Counter()
        self.failure_total = 0
        self.excluded_present = {"before": Counter(), "after": Counter()}
        self.sensor_present = {"before": 0, "after": 0}
        self.sensor_added = 0
        self.sensor_removed = 0

    def failure(self, kind: str, detail: str):
        self.failures[(kind, detail)] += 1
        self.failure_total += 1

    def on_error(self, message: str):
        self.failure("malformed", message)

    def on_material(self, path, old, new, kind, abs_diff=None, rel_diff=None):
        self.material_total += 1
        norm = _normalize(path)
        record = self.material.get(norm)
        if record is None:
            record = {"count": 0, "kinds": Counter(), "max_abs": None, "max_rel": None,
                      "example": None}
            self.material[norm] = record
        record["count"] += 1
        record["kinds"][kind] += 1
        if abs_diff is not None:
            record["max_abs"] = abs_diff if record["max_abs"] is None else max(record["max_abs"], abs_diff)
        if rel_diff is not None:
            record["max_rel"] = rel_diff if record["max_rel"] is None else max(record["max_rel"], rel_diff)
        if record["example"] is None:
            record["example"] = {"frame": self.frame, "path": path,
                                 "old": _json_safe(old), "new": _json_safe(new)}
        if self.writer is not None:
            self.writer.write(json.dumps({
                "bag": self.bag, "frame": self.frame, "path": path, "kind": kind,
                "old": _json_safe(old), "new": _json_safe(new),
                "abs_diff": _json_safe(abs_diff), "rel_diff": _json_safe(rel_diff),
            }, allow_nan=False) + "\n")

    def on_tolerated(self, path, abs_diff, rel_diff):
        self.tolerated_total += 1
        norm = _normalize(path)
        record = self.tolerated.get(norm)
        if record is None:
            record = {"count": 0, "max_abs": None, "max_rel": None}
            self.tolerated[norm] = record
        record["count"] += 1
        record["max_abs"] = abs_diff if record["max_abs"] is None else max(record["max_abs"], abs_diff)
        record["max_rel"] = rel_diff if record["max_rel"] is None else max(record["max_rel"], rel_diff)

    def ok(self) -> bool:
        return (self.failure_total == 0 and self.material_total == 0
                and not self.missing_after and not self.extra_after
                and (not self.gaps or not self.strict))

    def aggregate(self, records):
        return {
            path: {
                "count": record["count"],
                "kinds": dict(record["kinds"]) if "kinds" in record else None,
                "max_abs": record["max_abs"],
                "max_rel": record["max_rel"],
                "example": record.get("example"),
            }
            for path, record in sorted(records.items())
        }


class Coverage:
    """Accumulate actual state coverage over every frame, never only alarm frames."""

    def __init__(self):
        self.frames = 0
        self.first_frame = None
        self.last_frame = None
        self.status = Counter()
        self.status_transitions = Counter()
        self.previous_status = None
        self.active_frames = 0
        self.episodes = []
        self.current_episode = None
        self.motion_overlap_count = 0
        self.motion_overlap_min = None
        self.motion_overlap_max = None
        self.motion_residual_count = 0
        self.motion_residual_min = None
        self.motion_residual_max = None
        self.motion_deskew_frames = 0
        self.geometry_reason = Counter()
        self.geometry_invalid_reason = Counter()
        self.motion_reason = Counter()
        self.weak_translation_axes = Counter()
        self.relations = Counter()
        self.confirmations = Counter()
        self.intersection_confirmations = Counter()
        self.path_relation_reasons = Counter()
        self.objects = 0
        self.confirmed_objects = 0
        self.intersection_confirmed_objects = 0
        self.support_voxels = 0
        self.claim_voxels = 0
        self.interior_voxels = 0
        self.accumulated_support_voxels = 0
        self.track_ids = set()
        self.hits = Counter()
        self.association_candidates = 0
        self.association_confirmed = 0
        self.history_retained_for_motion = 0
        self.object_distance_count = 0
        self.object_distance_min = None
        self.object_distance_max = None
        self.distance_methods = Counter()
        self.frames_with_nearest_obstacle = 0
        self.evidence_observations = 0
        self.evidence_duplicate_observations = 0
        self.evidence_future_observations = 0
        self.interior_evidence_observations = 0
        self.interior_duplicate_observations = 0
        self.interior_future_observations = 0
        self.raw_points = 0
        self.invalid_points = 0
        self.input_valid_points = 0
        self.geometry_points = 0
        self.geometry_valid_frames = 0
        self.motion_valid_frames = 0
        self.sensor_rows = 0
        self.sensor_count_mismatch_rows = 0
        self.sensor_fields = {key: self._empty_field() for key in SENSOR_ATTRIBUTE_KEYS}
        self.sensor_flags = {flag: Counter() for flag in SENSOR_ATTRIBUTE_UNKNOWN_FLAGS}
        self.sensor_source_time_fields = Counter()

    @staticmethod
    def _empty_field():
        return {"available_rows": 0, "count": 0, "valid_count": 0, "invalid_count": 0,
                "dtypes": Counter(), "units": Counter()}

    def update(self, row: dict):
        self.frames += 1
        frame = row.get("frame")
        if self.first_frame is None:
            self.first_frame = frame
        self.last_frame = frame
        status = row.get("status")
        self.status[status] += 1
        if self.previous_status is not None:
            self.status_transitions[f"{self.previous_status} -> {status}"] += 1
        self.previous_status = status
        if status in ACTIVE_STATUSES:
            self.active_frames += 1
            if self.current_episode is None:
                self.current_episode = {"start_frame": frame, "start_s": row.get("timestamp_s"),
                                        "end_frame": frame, "end_s": row.get("timestamp_s"),
                                        "frames": 0, "statuses": Counter()}
            self.current_episode["end_frame"] = frame
            self.current_episode["end_s"] = row.get("timestamp_s")
            self.current_episode["frames"] += 1
            self.current_episode["statuses"][status] += 1
        elif self.current_episode is not None:
            self.episodes.append(self.current_episode)
            self.current_episode = None
        geometry = row.get("geometry") or {}
        reason = geometry.get("reason")
        self.geometry_reason[reason] += 1
        if geometry.get("valid"):
            self.geometry_valid_frames += 1
        else:
            self.geometry_invalid_reason[reason] += 1
        motion = row.get("motion") or {}
        self.motion_reason[motion.get("reason")] += 1
        if motion.get("valid"):
            self.motion_valid_frames += 1
        self.weak_translation_axes[motion.get("weak_translation_axes")] += 1
        if motion.get("deskew_timestamps"):
            self.motion_deskew_frames += 1
        overlap = motion.get("overlap")
        if isinstance(overlap, (int, float)) and not isinstance(overlap, bool):
            self.motion_overlap_count += 1
            self.motion_overlap_min = overlap if self.motion_overlap_min is None else min(self.motion_overlap_min, overlap)
            self.motion_overlap_max = overlap if self.motion_overlap_max is None else max(self.motion_overlap_max, overlap)
        residual = motion.get("median_residual_m")
        if isinstance(residual, (int, float)) and not isinstance(residual, bool):
            self.motion_residual_count += 1
            self.motion_residual_min = residual if self.motion_residual_min is None else min(self.motion_residual_min, residual)
            self.motion_residual_max = residual if self.motion_residual_max is None else max(self.motion_residual_max, residual)
        association = (row.get("pipeline") or {}).get("association") or {}
        self.association_candidates += association.get("candidates") or 0
        self.association_confirmed += association.get("confirmed") or 0
        if association.get("history_retained_for_motion"):
            self.history_retained_for_motion += 1
        self.raw_points += row.get("raw_points") or 0
        self.invalid_points += row.get("invalid_points") or 0
        self.input_valid_points += row.get("input_valid_points") or 0
        self.geometry_points += row.get("geometry_points") or 0
        if row.get("nearest_obstacle_m") is not None:
            self.frames_with_nearest_obstacle += 1
        if row.get("distance_method"):
            self.distance_methods[row["distance_method"]] += 1
        decoded = (row.get("raw_points") or 0) - (row.get("invalid_points") or 0)
        timestamp_s = row.get("timestamp_s")
        attributes = row.get(SENSOR_ATTRIBUTE_FIELD)
        if isinstance(attributes, dict):
            self.sensor_rows += 1
            for flag in SENSOR_ATTRIBUTE_UNKNOWN_FLAGS:
                self.sensor_flags[flag][_json_safe(attributes.get(flag))] += 1
            self.sensor_source_time_fields[_json_safe(attributes.get("source_time_field"))] += 1
            fields = attributes.get("fields")
            count_mismatch = False
            if isinstance(fields, dict):
                for key, info in fields.items():
                    stats = self.sensor_fields.setdefault(key, self._empty_field())
                    if not isinstance(info, dict) or not info.get("available"):
                        continue
                    stats["available_rows"] += 1
                    stats["count"] += info.get("count") or 0
                    stats["valid_count"] += info.get("valid_count") or 0
                    stats["invalid_count"] += info.get("invalid_count") or 0
                    if info.get("dtype"):
                        stats["dtypes"][info["dtype"]] += 1
                    if info.get("units"):
                        stats["units"][info["units"]] += 1
                    if info.get("count") != decoded:
                        count_mismatch = True
            self.sensor_count_mismatch_rows += count_mismatch
        for obj in row.get("objects") or []:
            self.objects += 1
            if obj.get("confirmed"):
                self.confirmed_objects += 1
            if obj.get("intersection_confirmed"):
                self.intersection_confirmed_objects += 1
            self.relations[obj.get("path_relation")] += 1
            self.confirmations[obj.get("confirmation")] += 1
            self.intersection_confirmations[obj.get("intersection_confirmation")] += 1
            self.path_relation_reasons[obj.get("path_relation_reason")] += 1
            self.support_voxels += obj.get("support_voxels") or 0
            self.claim_voxels += obj.get("claim_voxels") or 0
            self.interior_voxels += obj.get("interior_voxels") or 0
            self.accumulated_support_voxels += obj.get("accumulated_support_voxels") or 0
            self.hits[obj.get("hits")] += 1
            if obj.get("track_id") is not None:
                self.track_ids.add(obj["track_id"])
            distance = obj.get("distance_m")
            if isinstance(distance, (int, float)) and not isinstance(distance, bool):
                self.object_distance_count += 1
                self.object_distance_min = (distance if self.object_distance_min is None
                                            else min(self.object_distance_min, distance))
                self.object_distance_max = (distance if self.object_distance_max is None
                                            else max(self.object_distance_max, distance))
            if "evidence_timestamps_s" in obj:
                stamps = obj.get("evidence_timestamps_s") or []
                self.evidence_observations += 1
                if len(stamps) != len(set(stamps)):
                    self.evidence_duplicate_observations += 1
                if timestamp_s is not None and any(stamp > timestamp_s for stamp in stamps):
                    self.evidence_future_observations += 1
            if "intersection_evidence_timestamps_s" in obj:
                stamps = obj.get("intersection_evidence_timestamps_s") or []
                self.interior_evidence_observations += 1
                if len(stamps) != len(set(stamps)):
                    self.interior_duplicate_observations += 1
                if timestamp_s is not None and any(stamp > timestamp_s for stamp in stamps):
                    self.interior_future_observations += 1

    def to_dict(self) -> dict:
        episodes = list(self.episodes)
        if self.current_episode is not None:
            episodes.append(self.current_episode)
        return {
            "frames": self.frames,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "status_frames": dict(self.status),
            "status_transitions": dict(self.status_transitions),
            "active_frames": self.active_frames,
            "status_episodes": [
                {"start_frame": episode["start_frame"], "start_s": episode["start_s"],
                 "end_frame": episode["end_frame"], "end_s": episode["end_s"],
                 "frames": episode["frames"], "statuses": dict(episode["statuses"])}
                for episode in episodes
            ],
            "geometry_reason": dict(self.geometry_reason),
            "geometry_invalid_reason": dict(self.geometry_invalid_reason),
            "geometry_valid_frames": self.geometry_valid_frames,
            "motion_reason": dict(self.motion_reason),
            "motion_valid_frames": self.motion_valid_frames,
            "motion_deskew_frames": self.motion_deskew_frames,
            "motion_overlap_count": self.motion_overlap_count,
            "motion_overlap_min": self.motion_overlap_min,
            "motion_overlap_max": self.motion_overlap_max,
            "motion_residual_count": self.motion_residual_count,
            "motion_residual_min": self.motion_residual_min,
            "motion_residual_max": self.motion_residual_max,
            "weak_translation_axes": {str(key): count for key, count in self.weak_translation_axes.items()},
            "objects": self.objects,
            "confirmed_objects": self.confirmed_objects,
            "intersection_confirmed_objects": self.intersection_confirmed_objects,
            "relations": dict(self.relations),
            "confirmations": dict(self.confirmations),
            "intersection_confirmations": dict(self.intersection_confirmations),
            "path_relation_reasons": dict(self.path_relation_reasons),
            "support_voxels": self.support_voxels,
            "claim_voxels": self.claim_voxels,
            "interior_voxels": self.interior_voxels,
            "accumulated_support_voxels": self.accumulated_support_voxels,
            "unique_track_ids": len(self.track_ids),
            "hits": {str(key): count for key, count in self.hits.items()},
            "association_candidates": self.association_candidates,
            "association_confirmed": self.association_confirmed,
            "history_retained_for_motion_frames": self.history_retained_for_motion,
            "object_distance_count": self.object_distance_count,
            "object_distance_min": self.object_distance_min,
            "object_distance_max": self.object_distance_max,
            "distance_methods": dict(self.distance_methods),
            "frames_with_nearest_obstacle": self.frames_with_nearest_obstacle,
            "evidence_observations": self.evidence_observations,
            "evidence_duplicate_observations": self.evidence_duplicate_observations,
            "evidence_future_observations": self.evidence_future_observations,
            "interior_evidence_observations": self.interior_evidence_observations,
            "interior_duplicate_observations": self.interior_duplicate_observations,
            "interior_future_observations": self.interior_future_observations,
            "raw_points": self.raw_points,
            "invalid_points": self.invalid_points,
            "input_valid_points": self.input_valid_points,
            "geometry_points": self.geometry_points,
            "sensor_rows": self.sensor_rows,
            "sensor_flags": {flag: {str(value): count for value, count in values.items()}
                             for flag, values in self.sensor_flags.items()},
            "sensor_source_time_fields": {str(value): count
                                          for value, count in self.sensor_source_time_fields.items()},
            "sensor_fields": {
                key: {
                    "available_rows": stats["available_rows"],
                    "count": stats["count"],
                    "valid_count": stats["valid_count"],
                    "invalid_count": stats["invalid_count"],
                    "dtypes": dict(stats["dtypes"]),
                    "units": dict(stats["units"]),
                }
                for key, stats in sorted(self.sensor_fields.items())
            },
        }


def _iter_rows(path: Path, on_error):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                on_error(f"malformed JSON in {path.name} line {line_number}: {error}")
                continue
            if not isinstance(row, dict):
                on_error(f"non-object JSON in {path.name} line {line_number}")
                continue
            yield row


def _ordered_rows(path: Path, side: str, comparison: BagComparison):
    """Yield rows with strictly increasing frame numbers; record violations as failures.

    Frame numbers are the bag's raw message index, so a skipped duplicate leaves a gap. The
    per-row cumulative ``skipped_duplicate_scans`` accounts for it: the duplicate-corrected
    position must advance by exactly one per emitted row. Genuine gaps are recorded and are a
    verdict failure only in strict mode, where the recipe declares every=1 and no truncation.
    """
    last = None
    expected = 0 if comparison.strict else None
    for row in _iter_rows(path, comparison.on_error):
        frame = row.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool):
            comparison.failure("frame", f"{side} row without an integer frame in {path.name}")
            continue
        if last is not None and frame <= last:
            comparison.failure("ordering", f"{side} frame {frame} not strictly after {last} in {path.name}")
            continue
        skipped = row.get("skipped_duplicate_scans")
        logical = frame - skipped if isinstance(skipped, int) and not isinstance(skipped, bool) else frame
        if expected is not None and logical != expected:
            comparison.gaps.append({"side": side, "frame": frame, "logical_index": logical,
                                    "expected_logical_index": expected})
        last = frame
        expected = logical + 1
        yield row


def _validate_sensor_attributes(attributes, comparison: BagComparison):
    """Strict recipe check: the optional summary is present, unknown, and count-consistent."""
    if not isinstance(attributes, dict):
        comparison.failure("sensor_attributes", "sensor_attributes is not an object")
        return
    fields = attributes.get("fields")
    if not isinstance(fields, dict):
        comparison.failure("sensor_attributes", "sensor_attributes.fields is not an object")
        return
    for key in SENSOR_ATTRIBUTE_KEYS:
        info = fields.get(key)
        if not isinstance(info, dict) or type(info.get("available")) is not bool:
            comparison.failure("sensor_attributes", f"fields.{key} lacks an availability flag")
            continue
        if not info["available"]:
            continue
        missing = [name for name in SENSOR_FIELD_COUNT_KEYS if name not in info]
        if missing:
            comparison.failure("sensor_attributes", f"fields.{key} lacks {missing}")
            continue
        if not all(type(info.get(name)) is int and info[name] >= 0
                   for name in ("count", "valid_count", "invalid_count")):
            comparison.failure("sensor_attributes", f"fields.{key} counts are not nonnegative integers")
        elif info["count"] != info["valid_count"] + info["invalid_count"]:
            comparison.failure("sensor_attributes", f"fields.{key} counts are inconsistent")
        if not all(isinstance(info[name], str) for name in ("dtype", "units")):
            comparison.failure("sensor_attributes", f"fields.{key} dtype/units are not strings")
    for flag, expected in SENSOR_ATTRIBUTE_UNKNOWN_FLAGS.items():
        if type(attributes.get(flag)) is not type(expected) or attributes.get(flag) != expected:
            comparison.failure("sensor_attributes",
                               f"{flag}={attributes.get(flag)!r} is not the unknown value {expected!r}")
    for key in set(attributes) - (set(SENSOR_ATTRIBUTE_UNKNOWN_FLAGS) | {"fields", "source_time_field"}):
        comparison.failure("sensor_attributes", f"unexpected summary field {key!r}")
    for key in set(fields) - set(SENSOR_ATTRIBUTE_KEYS):
        comparison.failure("sensor_attributes", f"unexpected sensor channel {key!r}")


def compare_row(old: dict, new: dict, comparison: BagComparison, strict: bool):
    if strict:
        _validate_sensor_attributes(new.get(SENSOR_ATTRIBUTE_FIELD), comparison)
    old_keys, new_keys = set(old), set(new)
    for key in sorted(old_keys - new_keys):
        if key in TIMING_FIELD_SET:
            comparison.excluded_present["before"][key] += 1
            continue
        if key == SENSOR_ATTRIBUTE_FIELD:
            comparison.sensor_present["before"] += 1
            comparison.sensor_removed += 1
            continue
        comparison.on_material(f".{key}", old[key], MISSING, "missing_after")
    for key in sorted(new_keys - old_keys):
        if key in TIMING_FIELD_SET:
            comparison.excluded_present["after"][key] += 1
            continue
        if key == SENSOR_ATTRIBUTE_FIELD:
            comparison.sensor_present["after"] += 1
            comparison.sensor_added += 1
            continue
        comparison.on_material(f".{key}", MISSING, new[key], "extra_after")
    for key in sorted(old_keys & new_keys):
        if key in TIMING_FIELD_SET:
            comparison.excluded_present["before"][key] += 1
            comparison.excluded_present["after"][key] += 1
            continue
        if key == SENSOR_ATTRIBUTE_FIELD:
            before_attr, after_attr = old[key], new[key]
            comparison.sensor_present["before"] += 1
            comparison.sensor_present["after"] += 1
            if before_attr is None and after_attr is None:
                continue
            if before_attr is None or after_attr is None:
                continue
            diff_values(before_attr, after_attr, f".{SENSOR_ATTRIBUTE_FIELD}",
                        comparison.on_material, comparison.on_tolerated,
                        comparison.atol, comparison.rtol)
            continue
        diff_values(old[key], new[key], f".{key}", comparison.on_material,
                    comparison.on_tolerated, comparison.atol, comparison.rtol)


def compare_bag(bag: str, before_file: Path, after_file: Path, writer, atol: float,
                rtol: float, strict: bool) -> dict:
    """Stream and pair one bag's result rows, returning its report record."""
    comparison = BagComparison(bag, atol, rtol, writer, strict)
    before_coverage, after_coverage = Coverage(), Coverage()
    before_rows = _ordered_rows(before_file, "before", comparison)
    after_rows = _ordered_rows(after_file, "after", comparison)
    before = next(before_rows, None)
    after = next(after_rows, None)
    while before is not None or after is not None:
        if before is not None and after is not None:
            before_frame, after_frame = before["frame"], after["frame"]
            if before_frame == after_frame:
                comparison.frame = before_frame
                comparison.frames_compared += 1
                before_coverage.update(before)
                after_coverage.update(after)
                compare_row(before, after, comparison, strict)
                before = next(before_rows, None)
                after = next(after_rows, None)
            elif before_frame < after_frame:
                comparison.frame = before_frame
                comparison.missing_after.append(before_frame)
                before_coverage.update(before)
                comparison.on_material(".frame", before_frame, MISSING, "missing_after")
                before = next(before_rows, None)
            else:
                comparison.frame = after_frame
                comparison.extra_after.append(after_frame)
                after_coverage.update(after)
                comparison.on_material(".frame", MISSING, after_frame, "extra_after")
                after = next(after_rows, None)
        elif before is not None:
            comparison.frame = before["frame"]
            comparison.missing_after.append(before["frame"])
            before_coverage.update(before)
            comparison.on_material(".frame", before["frame"], MISSING, "missing_after")
            before = next(before_rows, None)
        else:
            comparison.frame = after["frame"]
            comparison.extra_after.append(after["frame"])
            after_coverage.update(after)
            comparison.on_material(".frame", MISSING, after["frame"], "extra_after")
            after = next(after_rows, None)
    if not before_coverage.frames or not after_coverage.frames:
        comparison.failure("empty_result", "A recording contains no usable result rows.")
    if strict and after_coverage.sensor_count_mismatch_rows:
        comparison.failure(
            "sensor_attributes",
            f"{after_coverage.sensor_count_mismatch_rows} after rows have a field count that is not "
            "raw_points - invalid_points")
    before_cover = _json_safe(before_coverage.to_dict())
    after_cover = _json_safe(after_coverage.to_dict())
    coverage_material, coverage_tolerated = {}, {}
    diff_values({k: v for k, v in before_cover.items() if not k.startswith("sensor")},
                {k: v for k, v in after_cover.items() if not k.startswith("sensor")},
                ".coverage", lambda path, old, new, kind, *rest: coverage_material.setdefault(
                    _normalize(path), {"old": _json_safe(old), "new": _json_safe(new), "kind": kind}),
                lambda *args: coverage_tolerated.setdefault(_normalize(args[0]), True),
                atol, rtol)
    if strict:
        for side, cover in (("before", before_cover), ("after", after_cover)):
            for key in ("evidence_duplicate_observations", "evidence_future_observations",
                        "interior_duplicate_observations", "interior_future_observations"):
                if cover[key]:
                    comparison.failure("temporal_evidence", f"{side} {key}={cover[key]}")
    print(f"{bag}: {comparison.frames_compared} frames compared, "
          f"{comparison.material_total} material mismatches, {comparison.tolerated_total} tolerated",
          flush=True)
    return {
        "ok": comparison.ok(),
        "before_file": str(before_file),
        "after_file": str(after_file),
        "before_sha256": digest(before_file),
        "after_sha256": digest(after_file),
        "frames_compared": comparison.frames_compared,
        "missing_after_frames": comparison.missing_after,
        "extra_after_frames": comparison.extra_after,
        "frame_progression_gaps": comparison.gaps,
        "material_mismatch_total": comparison.material_total,
        "material_mismatches": comparison.aggregate(comparison.material),
        "within_tolerance_total": comparison.tolerated_total,
        "within_tolerance_float_differences": comparison.aggregate(comparison.tolerated),
        "excluded_timing_field_rows": {
            side: dict(counter) for side, counter in comparison.excluded_present.items()},
        "sensor_attributes_presence": {
            "before_rows": comparison.sensor_present["before"],
            "after_rows": comparison.sensor_present["after"],
            "added_rows": comparison.sensor_added,
            "removed_rows": comparison.sensor_removed,
        },
        "coverage": {"before": before_cover, "after": after_cover},
        "coverage_mismatch_paths": sorted(coverage_material),
        "coverage_tolerated_paths": sorted(coverage_tolerated),
        "failures": [{"kind": kind, "detail": detail, "count": count}
                     for (kind, detail), count in sorted(comparison.failures.items())],
    }


def _load_json(path: Path, failures, required: bool, label: str):
    if not path.is_file():
        if required:
            failures.append({"kind": "missing_input", "detail": f"{label} missing: {path}"})
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        failures.append({"kind": "malformed_input", "detail": f"{label} unreadable: {path}: {error}"})
        return None


def run_identity(root: Path, failures, required: bool) -> dict:
    identity = {"root": str(root), "files": {}}
    for name in ("detector.json", "experiment.json", "manifest.json", "summary.json"):
        path = root / name
        identity["files"][name] = digest(path) if path.is_file() else None
        if not path.is_file() and required:
            failures.append({"kind": "missing_input", "detail": f"{name} missing in {root}"})
    manifest = _load_json(root / "manifest.json", failures, required, "manifest.json")
    identity["manifest"] = None
    if isinstance(manifest, dict):
        source_hashes = manifest.get("source_sha256") or {}
        identity["manifest"] = {
            "config_sha256": manifest.get("config_sha256"),
            "git_revision": manifest.get("git_revision"),
            "git_status": manifest.get("git_status"),
            "command": manifest.get("command"),
            "started_unix_s": manifest.get("started_unix_s"),
            "finished_unix_s": manifest.get("finished_unix_s"),
            "native_accelerator": manifest.get("native_accelerator"),
            "source_tree_digest": digest_source_map(source_hashes),
            "source_sha256": source_hashes,
            "bags": manifest.get("bags"),
        }
    return identity


def digest_source_map(source_hashes: dict) -> str | None:
    if not source_hashes:
        return None
    canonical = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def summarize_resources(summary: list | None) -> dict:
    resources = {}
    if not isinstance(summary, list):
        return resources
    for entry in summary:
        if not isinstance(entry, dict):
            continue
        resources[entry.get("bag")] = {
            "frames": entry.get("frames"),
            "split": entry.get("split"),
            "wall_s": entry.get("wall_s"),
            "process_peak_rss_bytes": entry.get("process_peak_rss_bytes"),
            "ingestion": entry.get("ingestion"),
            "processing_ms": entry.get("processing_ms"),
            "read_and_process_ms": entry.get("read_and_process_ms"),
            "inference_ms": entry.get("inference_ms"),
            "ingestion_wait_ms": entry.get("ingestion_wait_ms"),
            "stage_ms": entry.get("stage_ms"),
            "diagnostic_write_total_s": entry.get("diagnostic_write_total_s"),
            "visualization_total_s": entry.get("visualization_total_s"),
            "status_frames": entry.get("status_frames"),
            "geometry_valid_frames": entry.get("geometry_valid_frames"),
            "motion_valid_frames": entry.get("motion_valid_frames"),
            "alarm_episode_count": len(entry.get("alarm_episodes_unlabelled") or []),
            "latency_scope": entry.get("latency_scope"),
            "accuracy": entry.get("accuracy"),
            "accuracy_validity": entry.get("accuracy_validity"),
        }
    return resources


def compare_resources(before: dict, after: dict, bags) -> dict:
    comparison = {}
    for bag in bags:
        old, new = before.get(bag), after.get(bag)
        if not isinstance(old, dict) or not isinstance(new, dict):
            comparison[bag] = None
            continue
        old_wall, new_wall = old.get("wall_s"), new.get("wall_s")
        old_rss, new_rss = old.get("process_peak_rss_bytes"), new.get("process_peak_rss_bytes")
        old_p50 = (old.get("processing_ms") or {}).get("p50")
        new_p50 = (new.get("processing_ms") or {}).get("p50")
        comparison[bag] = {
            "frames_delta": (new.get("frames") or 0) - (old.get("frames") or 0),
            "wall_s_before": old_wall,
            "wall_s_after": new_wall,
            "wall_ratio_after_over_before": (new_wall / old_wall) if old_wall else None,
            "process_peak_rss_bytes_before": old_rss,
            "process_peak_rss_bytes_after": new_rss,
            "process_peak_rss_delta_bytes": (new_rss - old_rss) if isinstance(old_rss, int) and isinstance(new_rss, int) else None,
            "processing_ms_p50_before": old_p50,
            "processing_ms_p50_after": new_p50,
            "processing_ms_p50_ratio": (new_p50 / old_p50) if old_p50 else None,
            "ingestion_before": old.get("ingestion"),
            "ingestion_after": new.get("ingestion"),
        }
    return comparison


def validate_inventory(bag_panel, inventory, side, root: Path, resources: dict, failures,
                       strict: bool, manifest_metadata: dict | None = None):
    """Validate processed/duplicate/subsample counts against the saved SQLite inventory."""
    by_path = {entry.get("path"): entry for entry in (inventory or []) if isinstance(entry, dict)}
    records = {}
    for bag in bag_panel:
        entry = by_path.get(bag.get("path"))
        resource = resources.get(bag["name"]) or {}
        ingestion = resource.get("ingestion") or {}
        expected_raw = entry.get("raw_cloud_messages") if isinstance(entry, dict) else None
        frames = resource.get("frames")
        record = {
            "bag": bag["name"], "path": bag.get("path"), "side": side,
            "inventory_raw_cloud_messages": expected_raw,
            "inventory_metadata_sha256": entry.get("metadata_sha256") if isinstance(entry, dict) else None,
            "manifest_metadata_sha256": (manifest_metadata or {}).get(bag.get("path")),
            "summary_frames": frames,
            "ingestion": ingestion,
        }
        records[bag["name"]] = record
        if not strict:
            continue
        if entry is None:
            failures.append({"kind": "inventory_missing_bag", "side": side, "bag": bag["name"],
                             "detail": f"no inventory entry for {bag.get('path')}"})
            continue
        manifest_hash = (manifest_metadata or {}).get(bag.get("path"))
        if not entry.get("metadata_sha256") or not manifest_hash or entry["metadata_sha256"] != manifest_hash:
            failures.append({"kind": "input_metadata_mismatch", "side": side, "bag": bag["name"],
                             "detail": f"inventory metadata {entry['metadata_sha256']} != "
                                       f"manifest metadata {manifest_hash}"})
        if not isinstance(ingestion, dict):
            failures.append({"kind": "inventory_missing_ingestion", "side": side, "bag": bag["name"],
                             "detail": "summary.json has no ingestion record"})
            continue
        source = ingestion.get("source_messages")
        duplicates = ingestion.get("duplicate_measurements")
        subsampled = ingestion.get("subsampled_measurements")
        emitted = ingestion.get("emitted_scans")
        for label, value in (("source_messages", source), ("duplicate_measurements", duplicates),
                             ("subsampled_measurements", subsampled), ("emitted_scans", emitted)):
            if not isinstance(value, int):
                failures.append({"kind": "inventory_missing_count", "side": side, "bag": bag["name"],
                                 "detail": f"ingestion.{label} is not an integer"})
        if source != expected_raw:
            failures.append({"kind": "inventory_source_mismatch", "side": side, "bag": bag["name"],
                             "detail": f"processed source_messages={source}, inventory raw_cloud_messages={expected_raw}"})
        if isinstance(emitted, int) and isinstance(frames, int) and emitted != frames:
            failures.append({"kind": "inventory_emitted_mismatch", "side": side, "bag": bag["name"],
                             "detail": f"ingestion emitted_scans={emitted}, result rows={frames}"})
        if all(isinstance(value, int) for value in (source, duplicates, subsampled, emitted, frames)):
            if frames + duplicates + subsampled != source:
                failures.append({"kind": "inventory_accounting", "side": side, "bag": bag["name"],
                                 "detail": f"{frames} processed + {duplicates} duplicates + {subsampled} "
                                           f"subsampled != {source} source messages"})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
                        help="Comparison recipe: run roots, tolerance, panel and inventory.")
    parser.add_argument("--before", type=Path, help="Historical mode: before run directory.")
    parser.add_argument("--after", type=Path, help="Historical mode: after run directory.")
    parser.add_argument("--output", type=Path, help="Historical mode: report output file.")
    args = parser.parse_args()
    reporter_sources = {name: digest(Path(__file__).with_name(name))
                        for name in ("compare_alarm_runs.py", "decode_benchmark.py")}

    if args.experiment is not None:
        if any(value is not None for value in (args.before, args.after, args.output)):
            parser.error("--experiment cannot be combined with --before/--after/--output")
        mode = "experiment"
        recipe_path = args.experiment
        if not recipe_path.is_file():
            raise SystemExit(f"Comparison recipe missing: {recipe_path}")
        recipe = json.loads(recipe_path.read_text())
        before_root = Path(recipe["before"])
        after_root = Path(recipe["after"])
        output = Path(recipe["output"])
        atol = float(recipe.get("atol", 1e-9))
        rtol = float(recipe.get("rtol", 1e-9))
        expected = [dict(entry) for entry in recipe["expected_bags"]]
        inventory_path = Path(recipe["input_inventory"]) if recipe.get("input_inventory") else None
        detector_config = Path(recipe["detector_config"]) if recipe.get("detector_config") else None
        declared_revisions = recipe.get("declared_revisions") or {}
        source_binding_path = Path(recipe["source_binding"]) if recipe.get("source_binding") else None
        strict = True
    else:
        if any(value is None for value in (args.before, args.after, args.output)):
            parser.error("provide either --experiment or all of --before/--after/--output")
        mode = "historical"
        recipe = None
        before_root, after_root, output = args.before, args.after, args.output
        atol = rtol = 1e-9
        inventory_path = detector_config = None
        declared_revisions = {}
        source_binding_path = None
        strict = False
        names = sorted({path.stem for root in (before_root, after_root)
                        for path in root.glob("*.jsonl")
                        if path.is_file() and not path.name.endswith("-timing.jsonl")})
        expected = [{"name": name, "path": None, "split": None} for name in names]

    sidecar = output.with_name(output.stem + "-mismatches.jsonl")
    if output.exists() or sidecar.exists():
        raise SystemExit(f"Refusing to overwrite existing evidence: {output} or {sidecar}")
    output.parent.mkdir(parents=True, exist_ok=True)

    failures = []
    required = mode == "experiment"

    # Expected bag panel and processed records.
    expected_names = [entry["name"] for entry in expected]
    if not expected_names:
        failures.append({"kind": "empty_panel", "detail": "No recording results were selected."})
    processed = {}
    for side, root in (("before", before_root), ("after", after_root)):
        if not root.is_dir():
            failures.append({"kind": "missing_input", "detail": f"{side} run directory missing: {root}"})
            processed[side] = []
            continue
        processed[side] = sorted({path.stem for path in root.glob("*.jsonl")
                                  if path.is_file() and not path.name.endswith("-timing.jsonl")})
    expected_vs_processed = {
        "expected": expected_names,
        "before": processed["before"],
        "after": processed["after"],
        "missing_before": sorted(set(expected_names) - set(processed["before"])),
        "missing_after": sorted(set(expected_names) - set(processed["after"])),
        "extra_before": sorted(set(processed["before"]) - set(expected_names)),
        "extra_after": sorted(set(processed["after"]) - set(expected_names)),
    }
    for key in ("missing_before", "missing_after", "extra_before", "extra_after"):
        for bag in expected_vs_processed[key]:
            failures.append({"kind": "bag_panel_mismatch", "detail": f"{key}: {bag}"})

    # Run identities and resources.
    identities = {side: run_identity(root, failures, required)
                  for side, root in (("before", before_root), ("after", after_root))}
    summaries = {}
    for side, root in (("before", before_root), ("after", after_root)):
        summaries[side] = _load_json(root / "summary.json", failures, required, "summary.json")
    resources = {side: summarize_resources(summaries[side]) for side in ("before", "after")}
    resources["comparison"] = compare_resources(resources["before"], resources["after"], expected_names)

    detector_config_mismatch = None
    before_config_hash = (identities["before"].get("manifest") or {}).get("config_sha256")
    after_config_hash = (identities["after"].get("manifest") or {}).get("config_sha256")
    if before_config_hash and after_config_hash and before_config_hash != after_config_hash:
        detector_config_mismatch = {"before": before_config_hash, "after": after_config_hash}
        failures.append({"kind": "detector_config_mismatch",
                         "detail": "before and after manifests record different detector config hashes"})
    if strict:
        configured_hash = digest(detector_config) if detector_config and detector_config.is_file() else None
        if not configured_hash or before_config_hash != configured_hash or after_config_hash != configured_hash:
            failures.append({"kind": "detector_config_mismatch",
                             "detail": "The recorded configuration does not match the declared recipe."})
        if identities["before"]["files"]["detector.json"] != identities["after"]["files"]["detector.json"]:
            failures.append({"kind": "captured_detector_config_mismatch",
                             "detail": "The effective captured detector configurations differ."})

    # Native kernel source identity must agree; the compiled binary may differ by build path and is
    # retained as a hash only, never required to be byte-equal.
    def _native(identity):
        return (identity.get("manifest") or {}).get("native_accelerator") or {}
    before_native, after_native = _native(identities["before"]), _native(identities["after"])
    before_native_sources = before_native.get("sources_sha256") or {}
    after_native_sources = after_native.get("sources_sha256") or {}
    native_sources_equal = (before_native_sources == after_native_sources) if (before_native_sources or after_native_sources) else None
    if native_sources_equal is False:
        failures.append({"kind": "native_source_mismatch",
                         "detail": "before and after native kernel source hashes differ"})
    native_identity = {
        "before_binary_sha256": before_native.get("binary_sha256"),
        "after_binary_sha256": after_native.get("binary_sha256"),
        "binaries_equal": (before_native.get("binary_sha256") == after_native.get("binary_sha256")
                           if before_native.get("binary_sha256") and after_native.get("binary_sha256") else None),
        "before_sources_sha256": before_native_sources,
        "after_sources_sha256": after_native_sources,
        "sources_equal": native_sources_equal,
        "note": "Binary equality is not required: the compiled artifact may differ by build path. "
                "Kernel source identity is compared; binary hashes are retained as evidence.",
    }

    # Provenance: the captured manifest git_revision is observed, not authoritative. A run may have
    # used an archived source snapshot nested inside the current worktree, so the manifest can report
    # the enclosing branch revision while the captured Python/C++ is an older commit. The recipe's
    # declared revisions and the source-binding file are the identity of record.
    source_binding = None
    if source_binding_path is not None:
        source_binding = _load_json(source_binding_path, failures, True, "source binding")
    binding_revisions = {}
    if isinstance(source_binding, dict):
        binding_revisions = {
            "before": source_binding.get("declared_upstream_revision"),
            "after": source_binding.get("declared_submitted_fix_revision"),
        }
    declared_match = None
    if binding_revisions.get("before") and binding_revisions.get("after"):
        declared_match = (declared_revisions.get("before") == binding_revisions["before"]
                          and declared_revisions.get("after") == binding_revisions["after"])
        if declared_match is False:
            failures.append({"kind": "provenance_mismatch",
                             "detail": f"recipe declared_revisions={declared_revisions} disagree with "
                                       f"the source binding {binding_revisions}"})
    if strict:
        variants = (source_binding or {}).get("variants", {})
        for label, root in (("upstream", before_root), ("submitted_fix", after_root)):
            variant = variants.get(label, {})
            bindings = variant.get("files") or []
            if variant.get("production_source_matches_declared_revision") is not True or not bindings:
                failures.append({"kind": "source_binding_unverified", "detail": label})
            for binding in bindings:
                captured = root / "source" / binding["file"]
                actual_hash = digest(captured) if captured.is_file() else None
                if actual_hash != binding.get("captured_sha256"):
                    failures.append({"kind": "captured_source_changed", "detail": f"{label}: {binding['file']}"})
                if not binding.get("reporting_only") and actual_hash != binding.get("commit_object_sha256"):
                    failures.append({"kind": "source_revision_mismatch", "detail": f"{label}: {binding['file']}"})
    provenance = {
        "declared_revisions": declared_revisions,
        "binding_declared_revisions": binding_revisions,
        "declared_revisions_match_binding": declared_match,
        "source_binding_path": str(source_binding_path) if source_binding_path else None,
        "source_binding_sha256": (digest(source_binding_path)
                                  if source_binding_path is not None and source_binding_path.is_file() else None),
        "source_binding": source_binding,
        "observed_manifest_git_revision": {
            side: (identities[side].get("manifest") or {}).get("git_revision")
            for side in ("before", "after")},
        "observed_manifest_command": {
            side: (identities[side].get("manifest") or {}).get("command")
            for side in ("before", "after")},
        "captured_source_tree_digest": {
            side: (identities[side].get("manifest") or {}).get("source_tree_digest")
            for side in ("before", "after")},
        "note": "manifest git_revision is reported exactly as observed and is NOT treated as the "
                "implementation revision: an archived snapshot nested in the worktree makes "
                "run.git_revision() report the enclosing branch. Equality or inequality of the two "
                "manifest revisions is not a verdict criterion; captured source hashes bound in the "
                "source binding are.",
    }


    # Saved input inventory: processed + duplicate + subsample counts must account for the source.
    inventory = _load_json(inventory_path, failures, True, "input inventory") if inventory_path else None
    inventory_records = {}
    if inventory_path is not None:
        if isinstance(inventory, dict):
            inventory = inventory.get("inventory")
        for side, root in (("before", before_root), ("after", after_root)):
            manifest_bags = ((identities[side].get("manifest") or {}).get("bags")) or []
            manifest_metadata = {entry.get("path"): entry.get("metadata_sha256")
                                 for entry in manifest_bags if isinstance(entry, dict)}
            inventory_records[side] = validate_inventory(
                expected, inventory, side, root, resources[side], failures, strict, manifest_metadata)

    # Streaming row comparison.
    bag_records = {}
    with sidecar.open("x", encoding="utf-8") as writer:
        for entry in expected:
            bag = entry["name"]
            before_file = before_root / f"{bag}.jsonl"
            after_file = after_root / f"{bag}.jsonl"
            if not before_file.is_file() or not after_file.is_file():
                failures.append({"kind": "missing_result", "bag": bag,
                                 "detail": f"missing result file: {before_file} / {after_file}"})
                continue
            if before_file.stat().st_size == 0 or after_file.stat().st_size == 0:
                failures.append({"kind": "empty_result", "bag": bag,
                                 "detail": f"empty result file: {before_file} / {after_file}"})
                continue
            record = compare_bag(bag, before_file, after_file, writer, atol, rtol, strict)
            bag_records[bag] = record
            if strict:
                for side in ("before", "after"):
                    summary_frames = (resources[side].get(bag) or {}).get("frames")
                    if isinstance(summary_frames, int) and summary_frames != record["frames_compared"]:
                        failures.append({
                            "kind": "frame_count_mismatch", "bag": bag, "side": side,
                            "detail": f"paired result rows={record['frames_compared']}, "
                                      f"summary frames={summary_frames}"})
            if not record["ok"]:
                failures.append({"kind": "bag_not_equal", "bag": bag,
                                 "detail": f"{record['material_mismatch_total']} material mismatches"})

    # Diagnostic archives: the sibling helper owns array-level comparison.
    diagnostics = None
    try:
        from .decode_benchmark import compare_diagnostic_archives
    except Exception as error:
        compare_diagnostic_archives = None
        failures.append({"kind": "diagnostics_unavailable",
                         "detail": f"compare_diagnostic_archives unavailable: {error}"})
    if compare_diagnostic_archives is not None:
        before_diag = (before_root / "diagnostics").is_dir()
        after_diag = (after_root / "diagnostics").is_dir()
        if strict or before_diag or after_diag:
            try:
                diagnostics = compare_diagnostic_archives(
                    before_root, after_root, require_sensor_attributes=strict)
            except Exception as error:
                failures.append({"kind": "diagnostics_error", "detail": str(error)})
            else:
                if not diagnostics.get("ok"):
                    for item in diagnostics.get("failures") or []:
                        failures.append({"kind": "diagnostics", "detail": json.dumps(item, sort_keys=True)})
        else:
            diagnostics = {"ok": None, "skipped": True,
                           "reason": "diagnostics directories absent; not requested"}

    report = {
        "schema_version": 2,
        "reporter_source_sha256": reporter_sources,
        "recipe_sha256": digest(recipe_path) if mode == "experiment" else None,
        "mode": mode,
        "hypothesis": (recipe or {}).get("hypothesis"),
        "recipe": str(recipe_path) if mode == "experiment" else None,
        "before": str(before_root),
        "after": str(after_root),
        "tolerances": {"atol": atol, "rtol": rtol,
                       "scope": "absolute/relative tolerance applies only to continuous floating "
                                "estimates; identifiers, counters, booleans, list lengths, keys and "
                                "source/evidence timestamps compare exactly"},
        "excluded_fields": list(TIMING_FIELDS),
        "excluded_field_note": "Operational timing fields are reported, not compared. The top-level "
                               "sensor_attributes summary is a known addition/removal, validated for "
                               "presence and unknown flags but excluded from the equality verdict.",
        "strict_sensor_attributes": strict,
        "detector_config": str(detector_config) if detector_config else None,
        "detector_config_sha256": digest(detector_config) if detector_config and detector_config.is_file() else None,
        "detector_config_mismatch": detector_config_mismatch,
        "native_identity": native_identity,
        "provenance": provenance,
        "expected_bags": expected,
        "expected_vs_processed": expected_vs_processed,
        "identities": identities,
        "input_inventory": {
            "path": str(inventory_path) if inventory_path else None,
            "records": inventory_records,
        },
        "resources": resources,
        "diagnostics": diagnostics,
        "bags": bag_records,
        "sidecar": str(sidecar),
        "failures": failures,
        "warning": WARNING,
    }
    report["ok"] = (not failures
                    and all(record["ok"] for record in bag_records.values())
                    and len(bag_records) == len(expected)
                    and diagnostics is not None
                    and diagnostics.get("ok") is not False)
    report = _json_safe(report)
    write_json(output, report)
    total_material = sum(record["material_mismatch_total"] for record in bag_records.values())
    print(json.dumps({
        "ok": report["ok"],
        "mode": mode,
        "bags_compared": len(bag_records),
        "expected_bags": len(expected),
        "material_mismatches": total_material,
        "failures": len(failures),
        "diagnostics_ok": (diagnostics or {}).get("ok"),
        "output": str(output),
        "sidecar": str(sidecar),
    }, indent=2), flush=True)
    if strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
