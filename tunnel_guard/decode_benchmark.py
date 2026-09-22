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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(recipe["baseline_source"])
    spec = importlib.util.spec_from_file_location("_decode_baseline", baseline)
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    config_path = Path(recipe["detector_config"])
    config = json.loads(config_path.read_text())
    rotation, translation = np.asarray(config["sensor_rotation"]), np.asarray(config["sensor_translation"])
    store = get_typestore(Stores.ROS2_HUMBLE)
    shutil.copyfile(baseline, output / "before_io.py")
    shutil.copyfile(Path(io.__file__), output / "after_io.py")
    write_json(output / "experiment.json", recipe)
    write_json(output / "detector.json", config)
    manifest = environment() | {"command": sys.argv, "baseline_sha256": digest(baseline),
                                "config_sha256": digest(config_path),
                                "before_io_sha256": digest(baseline),
                                "after_io_sha256": digest(Path(io.__file__)),
                                "bags": []}
    report = {"frames": 0, "point_observations": 0, "changed_frames": [], "layouts": [],
              "repetitions": recipe["repetitions"], "max_coordinate_difference": 0.0,
              "max_time_difference": 0.0, "attributes_available": False,
              "attribute_failures": [],
              "attribute_channels": {}, "retained_attribute_bytes": 0, "frames_with_attributes": 0,
              "attribute_summary_example": None,
              "timing_scope": "decode_cloud only, alternating per-message order; no deserialization or comparison cost"}
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
                    report["frames"] += 1
                    report["point_observations"] += len(after[0])
                    stream.write(json.dumps({"bag": bag.name, "frame": index, "record_ns": record_ns,
                        "measurement_ns": message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec,
                        "source_message_sha256": hashlib.sha256(raw).hexdigest(),
                        "points": len(after[0]), "invalid": after[2], "duration_s": after[3],
                        "changed": changed, "elapsed_s": frame_elapsed,
                        "attribute_channels": frame_attributes,
                        "source_indices_sha256": hashlib.sha256(
                            np.asarray(attributes.source_indices).tobytes()).hexdigest() if attributes is not None else None,
                        "xyz_valid_mask_sha256": hashlib.sha256(
                            np.asarray(attributes.xyz_valid_mask).tobytes()).hexdigest() if attributes is not None else None,
                        "retained_attribute_bytes": measurement["retained_bytes"] if measurement is not None else 0}) + "\n")
    if not report["frames"]:
        raise ValueError("No real clouds decoded")
    report["decode_ms"] = {name: {label: float(np.quantile(values, q) * 1000)
                                  for label, q in (("p50", .5), ("p95", .95))}
                           for name, values in elapsed.items()}
    report["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report["limitations"] = ["Only recorded layouts/transforms exercised; no generated layout cases.",
                              "No safety/accuracy claim or target hardware measurement.",
                              "Retained-array bytes report the peak over measured frames, not a cumulative allocation or resident-set cost.",
                              "Attribute availability is reported as measured; firing identity and return multiplicity are not inferred."]
    write_json(output / "manifest.json", manifest)
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("layouts", "attribute_channels")}, indent=2))


if __name__ == "__main__":
    main()
