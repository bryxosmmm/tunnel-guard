"""Paired decoder timing and measurement preservation on configured real bags.

Compares a retained baseline decoder against the current one on real messages and,
when the current decoder returns the five-tuple carrying :class:`tunnel_guard.io.PointAttributes`,
measures the preserved sensor attributes directly against the raw PointCloud2 layout:
``source_indices`` to original flattened row-major slots, ``xyz_valid_mask`` over every
slot, and each retained value array byte-for-byte against the field it came from.
The first four decoder outputs must stay bit-identical (dtype, shape and bytes).
Every configured recording must be present: a missing bag is an error, not a skipped
measurement, and every deviation is recorded as a failure rather than downgraded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import zipfile

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from . import io
from .run import digest, environment, write_json


_DATATYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
_ATTRIBUTE_KEYS = ("intensity", "ring", "raw_time")


def unpack(result):
    """Split a decoder return value of either arity into its five components."""
    if len(result) == 5:
        return result[0], result[1], result[2], result[3], result[4]
    if len(result) == 4:
        return result[0], result[1], result[2], result[3], None
    raise ValueError(f"Decoder returned {len(result)} values; expected the four- or five-tuple")


def finite_max(difference) -> tuple[float, int]:
    """Largest finite absolute difference and the count of non-finite comparisons."""
    difference = np.abs(np.asarray(difference, dtype=np.float64))
    finite = np.isfinite(difference)
    if not finite.any():
        return 0.0, int(difference.size)
    return float(np.max(difference[finite])), int(np.count_nonzero(~finite))


def bit_identical(first, second) -> bool:
    """Exact equality: arrays by dtype, shape and bytes; scalars by type and raw bytes.

    Comparing scalar encodings is deliberate: ``-0.0 == 0.0`` is true in Python, so the
    duration and invalid-count outputs must be compared as their stored representations.
    """
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
                and first.dtype == second.dtype and first.shape == second.shape
                and first.tobytes() == second.tobytes())
    if type(first) is not type(second):
        return False
    first_scalar, second_scalar = np.asarray(first), np.asarray(second)
    return (first_scalar.dtype == second_scalar.dtype
            and first_scalar.tobytes() == second_scalar.tobytes())


def raw_view(message):
    """Structured scalar view of the real message, honouring offsets, row/point strides and endianness."""
    endian = ">" if message.is_bigendian else "<"
    names, formats, offsets, skipped = [], [], [], []
    for field in message.fields:
        if field.count != 1 or field.datatype not in _DATATYPES:
            skipped.append(field.name)
            continue
        names.append(field.name)
        formats.append(endian + _DATATYPES[field.datatype])
        offsets.append(field.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": message.point_step})
    records = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                         strides=(message.row_step, message.point_step))
    return records.reshape(-1), skipped


def independent_valid(message):
    """Recompute the decoder's XYZ validity from the raw fields, slot by slot."""
    flat, _ = raw_view(message)
    xyz = np.stack([flat[name].astype(np.float64) for name in ("x", "y", "z")], axis=1)
    return (np.isfinite(xyz).all(axis=1) & (np.einsum("ij,ij->i", xyz, xyz) > 1e-6))


def channel_stats(name, values):
    """JSON-safe availability, range and dtype summary for one retained array."""
    stats = {"dtype": str(values.dtype), "count": int(values.size)}
    numeric = values.dtype.kind in "iufc"
    if numeric and values.size:
        finite = np.isfinite(values)
        stats["invalid_count"] = int(np.count_nonzero(~finite))
        if finite.any():
            stats["min"] = float(np.min(values[finite]))
            stats["max"] = float(np.max(values[finite]))
        else:
            stats["min"] = stats["max"] = None
    else:
        stats["invalid_count"] = 0
    if name == "ring" and values.size:
        usable = values[np.isfinite(values)] if numeric else values
        if usable.size:
            stats["unique_count"] = int(np.unique(usable).size)
            stats["negative_or_fractional"] = int(np.count_nonzero(
                np.asarray(usable < 0) | np.asarray(np.mod(usable, 1) != 0)))
    return stats


def attribute_valid(name, values) -> np.ndarray:
    """Independent finite/(ring: nonnegative integral) validity for real decoded rows."""
    values = np.asarray(values)
    if values.dtype.kind in "iuf":
        valid = np.isfinite(values)
        if name == "ring":
            valid = valid & (values >= 0) & (values == np.floor(values))
        return valid
    return np.zeros(values.shape, dtype=bool)


def measure_attributes(message, attributes):
    """Compare retained attributes against the raw structured message view.

    Retained arrays must be aligned to the shared decoded rows: a length that matches only
    the raw slot count is a failure, not an alternative alignment. Every deviation -- a
    missing raw source field, an omitted source field, an unexpected value key, a
    wrong-length array or a validity array that disagrees with the finite/(ring:
    nonnegative integral) criteria on the real decoded rows -- is recorded in ``failures``.
    """
    flat, skipped = raw_view(message)
    slots = message.height * message.width
    valid = independent_valid(message)
    source_indices = np.asarray(attributes.source_indices)
    mask = np.asarray(attributes.xyz_valid_mask)
    rows = int(source_indices.size)
    mask_ok = mask.dtype == np.bool_ and mask.shape == (slots,)
    index_ok = (source_indices.dtype.kind in "iu"
                and source_indices.shape == (int(np.count_nonzero(valid)),)
                and np.array_equal(source_indices, np.flatnonzero(valid)))
    failures = []
    expected_time_field = next((name for name in ("timestamp", "time", "t")
                                if name in flat.dtype.names), None)
    if attributes.time_field != expected_time_field:
        failures.append("time_field")
    if not mask_ok or not np.array_equal(mask, valid):
        failures.append("xyz_valid_mask")
    if not index_ok:
        failures.append("source_indices")
    values = attributes.values
    measurement = {
        "slots": int(slots),
        "skipped_raw_fields": skipped,
        "valid_slots": int(np.count_nonzero(valid)),
        "decoded_rows": rows,
        "xyz_mask_shape": list(mask.shape),
        "xyz_mask_agrees": bool(mask_ok and np.array_equal(mask, valid)),
        "xyz_mask_mismatches": int(np.count_nonzero(mask != valid)) if mask_ok else None,
        "source_index_agrees": bool(index_ok),
        "time_field": attributes.time_field,
        "failures": failures,
        "channels": {},
    }
    emitted = attributes.arrays()
    retained_bytes = int(source_indices.nbytes + mask.nbytes)
    for name in sorted(set(values) - set(_ATTRIBUTE_KEYS)):
        failures.append(f"unexpected_value:{name}")
    for name in _ATTRIBUTE_KEYS:
        field_name = expected_time_field if name == "raw_time" else name
        if name not in values and field_name is not None and field_name in flat.dtype.names:
            failures.append(f"omitted_value:{name}")
    for name in sorted(set(values) | set(_ATTRIBUTE_KEYS)):
        field_name = expected_time_field if name == "raw_time" else name
        if name not in values:
            measurement["channels"][name] = {
                "available": False,
                "raw_field_name": field_name,
                "raw_field_present": bool(field_name is not None and field_name in flat.dtype.names)}
            continue
        raw = np.asarray(values[name])
        present = field_name is not None and field_name in flat.dtype.names
        entry = {"available": True, "raw_field_name": field_name, "raw_field_present": bool(present),
                 "aligned_rows": int(raw.size), "shared_decoded_rows": bool(raw.size == rows)}
        entry.update(channel_stats(name, raw))
        if not present:
            failures.append(f"missing_raw_field:{name}")
            entry.update({"raw_field_dtype": None, "matches_raw_field": False,
                          "max_abs_difference": None, "nonfinite_compare": None})
        elif raw.size != rows:
            failures.append(f"length:{name}")
            entry.update({"raw_field_dtype": str(np.asarray(flat[field_name]).dtype),
                          "matches_raw_field": False, "max_abs_difference": None,
                          "nonfinite_compare": None})
        else:
            expected = np.asarray(flat[field_name])[source_indices]
            matches = (raw.dtype == expected.dtype and raw.shape == expected.shape
                       and raw.tobytes() == expected.tobytes())
            if not matches:
                failures.append(f"value:{name}")
            maximum, nonfinite = 0.0, 0
            if raw.size:
                maximum, nonfinite = finite_max(raw.astype(np.float64) - expected.astype(np.float64))
            entry.update({"raw_field_dtype": str(np.asarray(flat[field_name]).dtype),
                          "matches_raw_field": bool(matches), "max_abs_difference": maximum,
                          "nonfinite_compare": nonfinite})
        independent = attribute_valid(name, raw)
        entry["invalid_count"] = int(np.count_nonzero(~independent))
        emitted_valid = emitted.get(f"{name}_valid")
        if emitted_valid is None:
            failures.append(f"validity_missing:{name}")
            entry["validity_mismatches"] = None
        else:
            emitted_valid = np.asarray(emitted_valid)
            if emitted_valid.dtype != np.bool_ or emitted_valid.shape != independent.shape:
                failures.append(f"validity_shape:{name}")
                entry["validity_mismatches"] = None
            else:
                mismatches = int(np.count_nonzero(emitted_valid != independent))
                entry["validity_mismatches"] = mismatches
                if mismatches:
                    failures.append(f"validity:{name}")
        entry["retained_bytes"] = int(raw.nbytes)
        retained_bytes += int(raw.nbytes)
        measurement["channels"][name] = entry
    measurement["retained_bytes"] = retained_bytes
    measurement["summary"] = attributes.summary()
    return measurement


# --- Saved-archive comparison -------------------------------------------------------------
# The keys are fixed by the detector's diagnostic contract: the legacy point/stage arrays are
# what every earlier run already saved, and the sensor-attribute arrays are the additions the
# #5 fix introduced. A key outside the two sets is an unannounced schema change, not an option.
_CLUSTER_METADATA_KEYS = ("cluster_labels", "cluster_core", "cluster_observed",
                          "cluster_nominal_overlap", "cluster_boundary_uncertain")
_STAGE_POINTS = {"decoded": "decoded_points", "range": "range_points",
                 "geometry_voxel": "geometry_voxel_points", "cluster": "cluster_points"}
_SENSOR_STAGES = ("decoded", "range", "geometry_voxel", "cluster")
_SENSOR_MASK = "decoded_xyz_valid_mask"
_SENSOR_SUFFIXES = ("source_indices",) + tuple(
    suffix for name in _ATTRIBUTE_KEYS for suffix in (name, f"{name}_valid"))
_SENSOR_KEYS = frozenset({_SENSOR_MASK}
                         | {f"{stage}_{suffix}" for stage in _SENSOR_STAGES
                            for suffix in _SENSOR_SUFFIXES})


def _stage_state(keys, archive, point_key, metadata_keys):
    """Whether a saved stage ran: captured, metadata-only, or absent.

    Zero-length metadata with no point array is NOT evidence the stage ran: it is recorded
    as ``not_executed`` so it can never be counted as successful coverage.
    """
    if point_key in keys:
        return {"state": "captured", "rows": int(archive[point_key].shape[0])}
    present = [key for key in metadata_keys if key in keys]
    rows = [int(archive[key].shape[0]) for key in present]
    return {"state": "metadata_without_points" if any(rows) else "not_executed",
            "metadata_keys": present, "metadata_rows": max(rows) if rows else 0}


def _slot_positions(decoded_indices, stage_indices):
    """Decoded row positions of `stage_indices` and whether every slot was actually found.

    Membership is checked against the exact decoded original slots, so an index that names a
    slot the decoded stage dropped is a failure rather than an approximate match.
    """
    positions = np.searchsorted(decoded_indices, stage_indices)
    inside = positions < decoded_indices.size
    found = np.zeros(positions.shape, dtype=bool)
    if inside.any():
        found[inside] = decoded_indices[positions[inside]] == stage_indices[inside]
    return positions, found


def _compare_stage(stage, archive, keys, decoded_points, decoded_indices,
                   present_channels, present_validity, name, failures, report_points):
    """Exact provenance for one captured stage against the decoded stage it came from.

    Every channel, validity array and the stage's own point array must byte-reproduce the
    decoded array at the stage's original slots: a re-associated, averaged or merely nearby
    value is a failure, and there is no tolerance for timestamps, ids or counts.
    """
    record = {"captured": True, "rows": None, "source_indices": False, "membership": False,
              "points_bytes_equal": None, "channels": {}, "validity": {}, "ok": True}
    index_key = f"{stage}_source_indices"
    if index_key not in keys:
        failures.append({"file": name, "kind": "stage_source_indices_missing", "key": index_key,
                         "detail": "Captured stage carries no original-slot index."})
        record["ok"] = False
        return record
    indices = archive[index_key]
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        failures.append({"file": name, "kind": "stage_source_indices_invalid", "key": index_key,
                         "detail": f"Expected 1-D integer slots, got {indices.dtype}{indices.shape}."})
        record["ok"] = False
        return record
    record["source_indices"] = True
    record["rows"] = int(indices.size)
    if decoded_indices is None:
        failures.append({"file": name, "kind": "decoded_indices_unavailable", "key": index_key,
                         "detail": "No decoded_source_indices to resolve stage rows against."})
        record["ok"] = False
        return record
    positions, found = _slot_positions(decoded_indices, indices)
    record["membership"] = bool(found.all())
    if not record["membership"]:
        failures.append({"file": name, "kind": "stage_source_index_not_in_decoded", "key": index_key,
                         "detail": f"{int((~found).sum())} of {indices.size} stage slots are not decoded rows."})
        record["ok"] = False
        return record
    point_key = _STAGE_POINTS[stage]
    points = archive[point_key]
    expected_points = decoded_points[positions] if decoded_points is not None else None
    equal = (expected_points is not None and points.ndim == 2
             and bit_identical(points, expected_points))
    record["points_bytes_equal"] = bool(equal)
    if not equal and report_points:
        failures.append({"file": name, "kind": "stage_points_provenance", "key": point_key,
                         "detail": "Stage points do not reproduce decoded_points at their source slots."})
        record["ok"] = False
    for attribute in _ATTRIBUTE_KEYS:
        value_key = f"{stage}_{attribute}"
        decoded_has_value = f"decoded_{attribute}" in keys
        if decoded_has_value:
            if value_key not in keys:
                record["channels"][attribute] = {"present": False, "bytes_equal": False}
                failures.append({"file": name, "kind": "stage_channel_missing", "key": value_key,
                                 "detail": "Decoded channel is absent from a captured stage."})
                record["ok"] = False
            else:
                values = archive[value_key]
                decoded_values = present_channels[attribute]
                reproduced = (values.dtype == decoded_values.dtype
                              and bit_identical(values, decoded_values[positions]))
                record["channels"][attribute] = {"present": True, "dtype": str(values.dtype),
                                                 "rows": int(values.size),
                                                 "bytes_equal": bool(reproduced)}
                if not reproduced:
                    failures.append({"file": name, "kind": "stage_channel_provenance", "key": value_key,
                                     "detail": "Stage channel does not reproduce the decoded channel at its source slots."})
                    record["ok"] = False
        elif value_key in keys:
            record["channels"][attribute] = {"present": True, "bytes_equal": False}
            failures.append({"file": name, "kind": "stage_channel_without_decoded", "key": value_key,
                             "detail": "Stage channel exists although the decoded stage does not carry it."})
            record["ok"] = False
        else:
            record["channels"][attribute] = {"present": False, "reason": "absent_in_both_stages_not_zero_filled"}
        validity_key = f"{stage}_{attribute}_valid"
        decoded_has_validity = f"decoded_{attribute}_valid" in keys
        if decoded_has_validity:
            if validity_key not in keys:
                record["validity"][attribute] = {"present": False, "bytes_equal": False}
                failures.append({"file": name, "kind": "stage_validity_missing", "key": validity_key,
                                 "detail": "Decoded validity array is absent from a captured stage."})
                record["ok"] = False
            else:
                validity = archive[validity_key]
                decoded_validity = present_validity[attribute]
                reproduced = (validity.dtype == np.bool_
                              and bit_identical(validity, decoded_validity[positions]))
                record["validity"][attribute] = {"present": True, "rows": int(validity.size),
                                                 "bytes_equal": bool(reproduced)}
                if not reproduced:
                    failures.append({"file": name, "kind": "stage_validity_provenance", "key": validity_key,
                                     "detail": "Stage validity does not reproduce the decoded validity at its source slots."})
                    record["ok"] = False
        elif validity_key in keys:
            record["validity"][attribute] = {"present": True, "bytes_equal": False}
            failures.append({"file": name, "kind": "stage_validity_without_decoded", "key": validity_key,
                             "detail": "Stage validity exists although the decoded stage does not carry it."})
            record["ok"] = False
        else:
            record["validity"][attribute] = {"present": False, "reason": "absent_in_both_stages_not_zero_filled"}
    return record


def _compare_archive(before_path, after_path, deskew_enabled, failures, require_sensor_attributes):
    """Compare one pair of saved diagnostic archives; returns a JSON-safe per-file record."""
    name = before_path.name
    start = len(failures)
    record = {"file": name, "before_sha256": digest(before_path), "after_sha256": None,
              "before_keys": [], "after_keys": [], "legacy_keys": [], "added_keys": [],
              "legacy": {}, "stages": {}, "channel_availability": {},
              "cluster_stage": {}, "representative_provenance": None,
              "sensor_attributes": "absent", "mask_consistent": None, "monotonic": None,
              "counts": {}, "ok": False}
    if not after_path.exists():
        failures.append({"file": name, "kind": "missing_diagnostic_file",
                         "detail": "No matching after diagnostic NPZ exists; the panel is incomplete."})
        return record
    record["after_sha256"] = digest(after_path)
    try:
        before = np.load(before_path)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        failures.append({"file": name, "kind": "diagnostic_archive_unreadable", "path": str(before_path),
                         "detail": f"{type(error).__name__}: {error}"})
        return record
    try:
        after = np.load(after_path)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        before.close()
        failures.append({"file": name, "kind": "diagnostic_archive_unreadable", "path": str(after_path),
                         "detail": f"{type(error).__name__}: {error}"})
        return record
    with before, after:
        before_keys = list(before.files)
        after_key_list = list(after.files)
        after_keys = set(after_key_list)
        record["before_keys"] = sorted(before_keys)
        record["after_keys"] = sorted(after_key_list)
        legacy_keys = [key for key in before_keys if key not in _SENSOR_KEYS]
        record["legacy_keys"] = sorted(legacy_keys)
        added_keys = sorted(key for key in after_key_list if key not in set(before_keys))
        record["added_keys"] = added_keys
        for key in legacy_keys:
            if key not in after_keys:
                record["legacy"][key] = {"present": False}
                failures.append({"file": name, "kind": "legacy_key_missing", "key": key,
                                 "detail": "Legacy diagnostic array absent from the after archive."})
                continue
            old, new = before[key], after[key]
            equal = bit_identical(old, new)
            record["legacy"][key] = {"present": True, "dtype_equal": bool(old.dtype == new.dtype),
                                     "shape_equal": bool(old.shape == new.shape),
                                     "bytes_equal": bool(equal)}
            if not equal:
                if old.dtype != new.dtype:
                    kind = "legacy_dtype_changed"
                elif old.shape != new.shape:
                    kind = "legacy_shape_changed"
                else:
                    kind = "legacy_bytes_changed"
                failures.append({"file": name, "kind": kind, "key": key,
                                 "detail": f"before {old.dtype}{old.shape}, after {new.dtype}{new.shape}"})
            del old, new
        for key in added_keys:
            if key not in _SENSOR_KEYS:
                failures.append({"file": name, "kind": "unexpected_key", "key": key,
                                 "detail": "Added diagnostic key is not an allowed sensor-attribute array."})
        cluster_after = _stage_state(after_keys, after, "cluster_points", _CLUSTER_METADATA_KEYS)
        cluster_before = _stage_state(set(before_keys), before, "cluster_points", _CLUSTER_METADATA_KEYS)
        record["cluster_stage"] = {"before": cluster_before, "after": cluster_after}
        if cluster_after["state"] == "metadata_without_points":
            failures.append({"file": name, "kind": "cluster_metadata_without_points",
                             "detail": f"{cluster_after['metadata_rows']} cluster metadata rows saved without cluster_points."})
        if cluster_before["state"] != cluster_after["state"]:
            failures.append({"file": name, "kind": "stage_execution_changed", "key": "cluster_points",
                             "detail": f"cluster stage before={cluster_before['state']}, after={cluster_after['state']}"})
        if not require_sensor_attributes and not (after_keys & _SENSOR_KEYS):
            record["counts"] = {
                "legacy_keys": len(legacy_keys),
                "legacy_bytes_equal": sum(bool(item.get("bytes_equal")) for item in record["legacy"].values()),
                "added_keys": len(added_keys), "sensor_keys": 0}
            record["ok"] = len(failures) == start
            return record

        mask = after[_SENSOR_MASK] if _SENSOR_MASK in after_keys else None
        decoded_indices = after["decoded_source_indices"] if "decoded_source_indices" in after_keys else None
        if (mask is None) != (decoded_indices is None):
            failures.append({"file": name, "kind": "partial_mask_index",
                             "detail": "decoded_xyz_valid_mask and decoded_source_indices must be saved together."})
        if mask is not None and decoded_indices is not None:
            record["sensor_attributes"] = "present"
            malformed = not (mask.dtype == np.bool_ and mask.ndim == 1
                             and decoded_indices.ndim == 1 and decoded_indices.dtype.kind in "iu")
            if malformed:
                failures.append({"file": name, "kind": "decoded_mask_index_malformed",
                                 "detail": f"mask {mask.dtype}{mask.shape}, indices {decoded_indices.dtype}{decoded_indices.shape}"})
            else:
                record["mask_consistent"] = bool(np.array_equal(decoded_indices, np.flatnonzero(mask)))
                record["monotonic"] = bool(decoded_indices.size == 0
                                           or np.all(np.diff(decoded_indices) > 0))
                if not record["mask_consistent"]:
                    failures.append({"file": name, "kind": "decoded_source_indices_mismatch",
                                     "detail": "decoded_source_indices are not the slots selected by decoded_xyz_valid_mask."})
                if not record["monotonic"]:
                    failures.append({"file": name, "kind": "decoded_source_indices_not_monotonic",
                                     "detail": "decoded_source_indices do not increase with original-slot order."})
        elif mask is None and decoded_indices is None:
            failures.append({"file": name, "kind": "sensor_attributes_missing",
                             "detail": "After archive carries no decoded_xyz_valid_mask / decoded_source_indices."})
        for attribute in _ATTRIBUTE_KEYS:
            present_value = f"decoded_{attribute}" in after_keys
            present_validity = f"decoded_{attribute}_valid" in after_keys
            record["channel_availability"][attribute] = "present" if present_value else "absent"
            if present_validity and not present_value:
                failures.append({"file": name, "kind": "decoded_orphan_validity",
                                 "key": f"decoded_{attribute}_valid",
                                 "detail": "Validity array saved without its channel; absence must be explicit, not zero-filled."})
        present_channels = {attribute: after[f"decoded_{attribute}"] for attribute in _ATTRIBUTE_KEYS
                            if f"decoded_{attribute}" in after_keys}
        present_validity = {attribute: after[f"decoded_{attribute}_valid"] for attribute in _ATTRIBUTE_KEYS
                            if f"decoded_{attribute}_valid" in after_keys}
        decoded_points = after["decoded_points"] if "decoded_points" in after_keys else None
        decoded_rows = decoded_points.shape[0] if decoded_points is not None else None
        if decoded_indices is not None and decoded_rows is not None and decoded_indices.size != decoded_rows:
            failures.append({"file": name, "kind": "decoded_rows_mismatch",
                             "detail": f"decoded_points has {decoded_rows} rows but decoded_source_indices has {decoded_indices.size}."})
        if decoded_rows is not None:
            for attribute, values in present_channels.items():
                if values.ndim != 1 or values.shape[0] != decoded_rows:
                    failures.append({"file": name, "kind": "decoded_channel_rows",
                                     "key": f"decoded_{attribute}",
                                     "detail": f"{values.shape} does not align with {decoded_rows} decoded rows."})
                validity = present_validity.get(attribute)
                if validity is None:
                    failures.append({"file": name, "kind": "decoded_validity_missing",
                                     "key": f"decoded_{attribute}_valid",
                                     "detail": "Every present decoded channel requires an explicit validity array."})
                elif validity.dtype != np.bool_ or validity.shape != values.shape:
                    failures.append({"file": name, "kind": "decoded_validity_shape",
                                     "key": f"decoded_{attribute}_valid",
                                     "detail": f"expected bool{values.shape}, got {validity.dtype}{validity.shape}."})
                elif not bit_identical(validity, attribute_valid(attribute, values)):
                    failures.append({"file": name, "kind": "decoded_validity_values",
                                     "key": f"decoded_{attribute}_valid",
                                     "detail": "Decoded validity differs from independently computed channel validity."})
        record["stages"]["decoded"] = {
            "captured": decoded_points is not None, "rows": decoded_rows,
            "index_rows": int(decoded_indices.size) if decoded_indices is not None else None,
            "mask_consistent": record["mask_consistent"], "monotonic": record["monotonic"]}
        for stage in ("range", "geometry_voxel", "cluster"):
            point_key = _STAGE_POINTS[stage]
            if point_key not in after_keys:
                record["stages"][stage] = {"captured": False, "state": "not_executed",
                                           "note": "no point array saved; not counted as coverage"}
                metadata = [key for key in _SENSOR_KEYS
                            if key.startswith(f"{stage}_") and key in after_keys]
                nonempty = [key for key in metadata if after[key].size]
                if nonempty or (metadata and stage != "cluster"):
                    failures.append({"file": name, "kind": "stage_metadata_without_points",
                                     "keys": nonempty or metadata,
                                     "detail": "Only empty cluster metadata is allowed when its stage did not execute."})
                continue
            record["stages"][stage] = _compare_stage(
                stage, after, after_keys, decoded_points, decoded_indices,
                present_channels, present_validity, name, failures, report_points=(stage != "cluster"))
        cluster_record = record["stages"]["cluster"]
        if not cluster_record.get("captured"):
            record["representative_provenance"] = "not_executed"
        elif deskew_enabled is True:
            record["representative_provenance"] = "unsupported_deskew_enabled"
            failures.append({"file": name, "kind": "representative_provenance_unsupported_scope",
                             "detail": "deskew enabled: equality with raw decoded points cannot establish representative provenance."})
        elif deskew_enabled is False:
            verified = bool(cluster_record.get("points_bytes_equal"))
            record["representative_provenance"] = "verified" if verified else "failed"
            if not verified:
                failures.append({"file": name, "kind": "representative_xyz_provenance",
                                 "detail": "cluster_points do not reproduce decoded points at their source slots."})
        else:
            record["representative_provenance"] = "unavailable"
            failures.append({"file": name, "kind": "deskew_flag_unavailable",
                             "detail": "No boolean deskew_enabled in detector.json; provenance scope cannot be stated."})
        record["counts"] = {
            "legacy_keys": len(legacy_keys),
            "legacy_bytes_equal": sum(1 for entry in record["legacy"].values() if entry.get("bytes_equal")),
            "added_keys": len(added_keys),
            "sensor_keys": sum(1 for key in after_key_list if key in _SENSOR_KEYS),
        }
    record["ok"] = len(failures) == start
    return record


def compare_diagnostic_archives(before_directory: Path, after_directory: Path, *,
                                require_sensor_attributes: bool = True) -> dict:
    """Compare the saved detector diagnostics of two run roots, exactly and fail-closed.

    ``before_directory`` and ``after_directory`` are RUN ROOT directories: the diagnostics
    live under ``<root>/diagnostics/*.npz`` and the captured detector configuration in
    ``<root>/detector.json``. Both runs must save the same diagnostic inventory - a file
    either run is missing is a failure, never a silent skip and never an intersection-only
    acceptance.

    For every legacy key the before archive contains, the after archive must hold the array
    with the same dtype, shape and bytes. The only added keys permitted are
    ``decoded_xyz_valid_mask`` and the ``decoded_/range_/geometry_voxel_/cluster_``
    ``source_indices``, ``intensity``, ``ring``, ``raw_time`` and ``*_valid`` arrays; anything
    else is an unannounced schema change. The full mask is checked against
    ``decoded_source_indices`` and against monotonic original-slot order, and each captured
    stage must resolve its indices into the decoded original slots and byte-reproduce the
    decoded channel, validity and point arrays at exactly those slots - no averaging or
    nearest-neighbour reassociation is accepted.

    With deskew disabled the saved representatives must equal the original decoded points.
    With deskew enabled, motion resamples the cloud, so equality with the raw points cannot
    establish provenance: the claim is reported as an unsupported scope and fails rather than
    passing. Calibration and firing identity of the retained fields are unknown here, and
    this function makes no safety or recall claim.

    Historical comparisons may set ``require_sensor_attributes=False``: archives without
    the newly introduced fields still receive the complete legacy-array comparison, but
    contribute no sensor-provenance coverage.

    Returns ``{"ok": bool, "files": [per-file records], "failures": [actionable records]}``.
    Files are opened one at a time so a full panel does not need to fit in memory at once.
    """
    before_root, after_root = Path(before_directory), Path(after_directory)
    failures = []
    before_config, before_error = _read_config(before_root)
    after_config, after_error = _read_config(after_root)
    if before_error is not None:
        failures.append({"kind": "detector_config_unreadable",
                         "path": str(before_root / "detector.json"), "detail": before_error})
    if after_error is not None:
        failures.append({"kind": "detector_config_unreadable",
                         "path": str(after_root / "detector.json"), "detail": after_error})
    before_deskew = before_config.get("deskew_enabled") if isinstance(before_config, dict) else None
    after_deskew = after_config.get("deskew_enabled") if isinstance(after_config, dict) else None
    if isinstance(before_deskew, bool) and isinstance(after_deskew, bool) and before_deskew != after_deskew:
        failures.append({"kind": "deskew_flag_mismatch",
                         "detail": f"before deskew_enabled={before_deskew}, after deskew_enabled={after_deskew}"})
    before_dir, after_dir = before_root / "diagnostics", after_root / "diagnostics"
    if not before_dir.is_dir():
        failures.append({"kind": "diagnostics_directory_missing", "path": str(before_dir)})
    if not after_dir.is_dir():
        failures.append({"kind": "diagnostics_directory_missing", "path": str(after_dir)})
    before_files = sorted(path.name for path in before_dir.glob("*.npz")) if before_dir.is_dir() else []
    after_files = sorted(path.name for path in after_dir.glob("*.npz")) if after_dir.is_dir() else []
    if not before_files and not after_files:
        failures.append({"kind": "no_diagnostic_coverage",
                         "detail": "Neither run saved a diagnostic NPZ; no coverage can be claimed."})
        return {"ok": False, "files": [], "failures": failures,
                "coverage": {"before_files": 0, "after_files": 0,
                             "message": "no saved diagnostic archives"}}
    for name in sorted(set(after_files) - set(before_files)):
        failures.append({"file": name, "kind": "unexpected_diagnostic_file",
                         "detail": "After run saved a diagnostic the before run did not; the inventories must match."})
    coverage = {"before_files": len(before_files), "after_files": len(after_files), "compared": 0,
                "legacy_keys": 0, "legacy_bytes_equal": 0, "files_with_sensor_attributes": 0,
                "representative_verified": 0, "representative_not_executed": 0}
    files = []
    for name in before_files:
        record = _compare_archive(before_dir / name, after_dir / name, after_deskew, failures,
                                  require_sensor_attributes)
        files.append(record)
        coverage["compared"] += 1
        coverage["legacy_keys"] += record["counts"].get("legacy_keys", 0)
        coverage["legacy_bytes_equal"] += record["counts"].get("legacy_bytes_equal", 0)
        coverage["files_with_sensor_attributes"] += record["sensor_attributes"] == "present"
        coverage["representative_verified"] += record["representative_provenance"] == "verified"
        coverage["representative_not_executed"] += record["representative_provenance"] == "not_executed"
    return {"ok": not failures, "files": files, "failures": failures, "coverage": coverage}


def _read_config(root: Path):
    """Read a run root's captured detector configuration, or the reason it is unusable."""
    path = Path(root) / "detector.json"
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    if recipe.get("every", 1) != 1:
        raise ValueError("Paired decoding measures every raw message; every must be 1")
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(recipe["baseline_source"])
    spec = importlib.util.spec_from_file_location("_decode_baseline", baseline)
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    config_path = Path(recipe["detector_config"])
    config = json.loads(config_path.read_text())
    if recipe.get("seed", config["seed"]) != config["seed"]:
        raise ValueError("Experiment and detector seed disagree")
    rotation, translation = np.asarray(config["sensor_rotation"]), np.asarray(config["sensor_translation"])
    store = get_typestore(Stores.ROS2_HUMBLE)
    shutil.copyfile(baseline, output / "before_io.py")
    shutil.copyfile(Path(io.__file__), output / "after_io.py")
    shutil.copyfile(Path(__file__), output / "decode_benchmark.py")
    write_json(output / "experiment.json", recipe)
    write_json(output / "detector.json", config)
    manifest = environment() | {"command": sys.argv, "baseline_sha256": digest(baseline),
                                "config_sha256": digest(config_path),
                                "before_io_sha256": digest(baseline),
                                "after_io_sha256": digest(Path(io.__file__)),
                                "bags": []}
    manifest.update(benchmark_sha256=digest(Path(__file__)),
                    recipe_sha256=digest(args.experiment), started_unix_s=time.time())
    write_json(output / "manifest.json", manifest)
    report = {"frames": 0, "point_observations": 0, "changed_frames": [], "layouts": [],
              "repetitions": recipe["repetitions"], "max_coordinate_difference": 0.0,
              "max_time_difference": 0.0, "attributes_available": False,
              "attribute_failures": [],
              "attribute_channels": {}, "retained_attribute_bytes": 0, "frames_with_attributes": 0,
              "attribute_summary_example": None,
              "timing_scope": "decode_cloud only, alternating per-message order; no deserialization or comparison cost"}
    diagnostic_root = (Path(recipe["diagnostic_run"]) / "diagnostics"
                       if recipe.get("diagnostic_run") else None)
    diagnostic_files = (set(path.name for path in diagnostic_root.glob("*.npz"))
                        if diagnostic_root is not None else set())
    if diagnostic_root is not None and not diagnostic_files:
        raise ValueError(f"No source diagnostics to compare: {diagnostic_root}")
    unmatched_diagnostics = diagnostic_files.copy()
    report["diagnostic_source_files"] = []
    report["diagnostic_source_failures"] = []
    elapsed = {"before": [], "after": []}
    layouts = set()
    with (output / "frames.jsonl").open("x") as stream:
        for entry in recipe["bags"]:
            bag = Path(entry["path"])
            if not bag.exists():
                raise FileNotFoundError(f"Configured bag is missing: {bag}")
            manifest["bags"].append({"path": str(bag), "available": True, "split": entry.get("split"),
                                     "metadata_sha256": digest(bag / "metadata.yaml")})
            with Reader(bag) as reader:
                connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"
                               and (not entry.get("topic") or c.topic == entry["topic"])]
                if len({c.topic for c in connections}) != 1:
                    raise ValueError("Select exactly one cloud topic")
                for index, (connection, record_ns, raw) in enumerate(reader.messages(connections=connections)):
                    if recipe.get("max_frames") is not None and index >= recipe["max_frames"]:
                        break
                    message = store.deserialize_cdr(raw, connection.msgtype)
                    layout = {"height": message.height, "width": message.width,
                              "point_step": message.point_step, "row_step": message.row_step,
                              "bigendian": message.is_bigendian,
                              "fields": [(f.name, f.offset, f.datatype, f.count) for f in message.fields]}
                    signature = json.dumps(layout, sort_keys=True)
                    if signature not in layouts:
                        layouts.add(signature)
                        report["layouts"].append(layout)
                    variants = [("before", previous.decode_cloud), ("after", io.decode_cloud)]
                    if index == 0:
                        for _, function in variants:
                            function(message, rotation, translation)
                    decoded = {}
                    frame_elapsed = {"before": [], "after": []}
                    for repeat in range(recipe["repetitions"]):
                        for name, function in (variants if (index + repeat) % 2 == 0 else variants[::-1]):
                            started = time.perf_counter()
                            value = function(message, rotation, translation)
                            duration = time.perf_counter() - started
                            decoded[name] = value
                            elapsed[name].append(duration)
                            frame_elapsed[name].append(duration)
                    before = unpack(decoded["before"])
                    after = unpack(decoded["after"])
                    changed = []
                    for offset, field in ((0, "coordinates"), (1, "normalized_times")):
                        if not bit_identical(before[offset], after[offset]):
                            changed.append(field)
                        if before[offset].shape == after[offset].shape and before[offset].size:
                            key = "max_coordinate_difference" if offset == 0 else "max_time_difference"
                            report[key] = max(report[key], finite_max(before[offset].astype(np.float64)
                                                                     - after[offset].astype(np.float64))[0])
                    if not bit_identical(before[2], after[2]):
                        changed.append("invalid_count")
                    if not bit_identical(before[3], after[3]):
                        changed.append("duration_s")
                    if changed:
                        report["changed_frames"].append({"bag": bag.name, "frame": index, "fields": changed,
                                                         "source_message_sha256": hashlib.sha256(raw).hexdigest()})
                    attributes = after[4]
                    measurement = None
                    frame_attributes = None
                    if attributes is None:
                        report["attribute_failures"].append({"bag": bag.name, "frame": index,
                                                             "fields": ["missing_point_attributes"]})
                    else:
                        report["attributes_available"] = True
                        report["frames_with_attributes"] += 1
                        measurement = measure_attributes(message, attributes)
                        frame_attributes = {name: channel.get("matches_raw_field")
                                            for name, channel in measurement["channels"].items()
                                            if channel.get("available")}
                        report["retained_attribute_bytes"] = max(report["retained_attribute_bytes"],
                                                                 measurement["retained_bytes"])
                        if measurement["failures"]:
                            report["attribute_failures"].append({"bag": bag.name, "frame": index,
                                                                 "fields": measurement["failures"]})
                        if report["attribute_summary_example"] is None:
                            report["attribute_summary_example"] = {"bag": bag.name, "frame": index,
                                                                   "summary": measurement["summary"]}
                        for name, channel in measurement["channels"].items():
                            if not channel.get("available"):
                                continue
                            aggregate = report["attribute_channels"].setdefault(name, {"frames": 0, "dtypes": {},
                                                                                        "invalid_total": 0,
                                                                                        "min": None, "max": None})
                            aggregate["frames"] += 1
                            aggregate["dtypes"][channel["dtype"]] = aggregate["dtypes"].get(channel["dtype"], 0) + 1
                            aggregate["invalid_total"] += channel.get("invalid_count", 0)
                            if channel.get("min") is not None:
                                aggregate["min"] = channel["min"] if aggregate["min"] is None else min(aggregate["min"], channel["min"])
                                aggregate["max"] = channel["max"] if aggregate["max"] is None else max(aggregate["max"], channel["max"])
                    diagnostic_name = f"{bag.name}_{index:06d}.npz"
                    if diagnostic_name in diagnostic_files:
                        unmatched_diagnostics.remove(diagnostic_name)
                        expected_arrays = {"decoded_points": after[0]}
                        if attributes is not None:
                            expected_arrays["decoded_xyz_valid_mask"] = attributes.xyz_valid_mask
                            expected_arrays.update({f"decoded_{key}": value
                                                    for key, value in attributes.arrays().items()})
                        with np.load(diagnostic_root / diagnostic_name, allow_pickle=False) as archive:
                            differences = [key for key, expected in expected_arrays.items()
                                           if key not in archive or not bit_identical(archive[key], expected)]
                        report["diagnostic_source_files"].append(diagnostic_name)
                        if differences or attributes is None:
                            report["diagnostic_source_failures"].append({
                                "file": diagnostic_name, "fields": differences,
                                "attributes_missing": attributes is None})
                    report["frames"] += 1
                    report["point_observations"] += len(after[0])
                    stream.write(json.dumps({"bag": bag.name, "frame": index, "record_ns": record_ns,
                        "measurement_ns": message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec,
                        "source_message_sha256": hashlib.sha256(raw).hexdigest(),
                        "points": len(after[0]), "invalid": after[2], "duration_s": after[3],
                        "changed": changed, "elapsed_s": frame_elapsed,
                        "attribute_channels": frame_attributes,
                        "diagnostic_source_file": diagnostic_name if diagnostic_name in diagnostic_files else None,
                        "source_indices_sha256": hashlib.sha256(
                            np.asarray(attributes.source_indices).tobytes()).hexdigest() if attributes is not None else None,
                        "xyz_valid_mask_sha256": hashlib.sha256(
                            np.asarray(attributes.xyz_valid_mask).tobytes()).hexdigest() if attributes is not None else None,
                        "retained_attribute_bytes": measurement["retained_bytes"] if measurement is not None else 0}) + "\n")
    if not report["frames"]:
        raise ValueError("No real clouds decoded")
    for name in sorted(unmatched_diagnostics):
        report["diagnostic_source_failures"].append({
            "file": name, "fields": ["source_message_not_decoded"]})
    report["decode_ms"] = {name: {label: float(np.quantile(values, q) * 1000)
                                  for label, q in (("p50", .5), ("p95", .95))}
                           for name, values in elapsed.items()}
    report["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024)
    report["decode_samples_per_arm"] = {name: len(values) for name, values in elapsed.items()}
    report["limitations"] = ["Only recorded layouts/transforms exercised; no generated layout cases.",
                              "No safety/accuracy claim or target hardware measurement.",
                              "Pooled local timing, not a controlled deployment benchmark; after includes attribute extraction.",
                              "Point observations count decoded XYZ-valid returns over raw messages, not independent events.",
                              "Retained-array bytes report the peak over measured frames, not a cumulative allocation or resident-set cost.",
                              "Attribute availability is reported as measured; firing identity and return multiplicity are not inferred."]
    manifest["finished_unix_s"] = time.time()
    write_json(output / "manifest.json", manifest)
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("layouts", "attribute_channels")}, indent=2))


if __name__ == "__main__":
    main()
