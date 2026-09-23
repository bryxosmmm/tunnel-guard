"""Paired decoder timing and measurement preservation on configured real bags."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from . import io
from .run import digest, environment, write_json


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
                                "config_sha256": digest(config_path), "bags": []}
    report = {"frames": 0, "point_observations": 0, "changed_frames": [], "layouts": [],
              "repetitions": recipe["repetitions"], "max_coordinate_difference": 0.0,
              "max_time_difference": 0.0, "timing_scope": "decode_cloud only, alternating per-message order; no deserialization or comparison cost"}
    elapsed = {"before": [], "after": []}
    layouts = set()
    with (output / "frames.jsonl").open("x") as stream:
        for entry in recipe["bags"]:
            bag = Path(entry["path"])
            manifest["bags"].append({"path": str(bag), "metadata_sha256": digest(bag / "metadata.yaml")})
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
                    a, b = decoded["before"], decoded["after"]
                    changed = []
                    for offset, field in ((0, "coordinates"), (1, "normalized_times")):
                        if not np.array_equal(a[offset], b[offset]):
                            changed.append(field)
                        if a[offset].shape == b[offset].shape and a[offset].size:
                            key = "max_coordinate_difference" if offset == 0 else "max_time_difference"
                            report[key] = max(report[key], float(np.max(np.abs(a[offset] - b[offset]))))
                    if a[2:4] != b[2:4]:
                        changed.append("invalid_count_or_duration")
                    if changed:
                        report["changed_frames"].append({"bag": bag.name, "frame": index, "fields": changed})
                    report["frames"] += 1
                    report["point_observations"] += len(b[0])
                    stream.write(json.dumps({"bag": bag.name, "frame": index, "record_ns": record_ns,
                        "measurement_ns": message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec,
                        "points": len(b[0]), "invalid": b[2], "duration_s": b[3],
                        "changed": changed, "elapsed_s": frame_elapsed}) + "\n")
    if not report["frames"]:
        raise ValueError("No real clouds decoded")
    report["decode_ms"] = {name: {label: float(np.quantile(values, q) * 1000)
                                  for label, q in (("p50", .5), ("p95", .95))}
                           for name, values in elapsed.items()}
    report["limitations"] = ["Only recorded layouts/transforms exercised; no generated layout cases.",
                              "No safety/accuracy claim or target hardware measurement."]
    write_json(output / "manifest.json", manifest)
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "layouts"}))


if __name__ == "__main__":
    main()
