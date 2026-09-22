"""Stream a bounded bag interval and report sensor layout and acquisition clocks."""
import argparse
from pathlib import Path
import json

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .detector import load_config
from .io import decode_cloud
from .run import digest, write_json


def coincident_groups(points, ring, timestamps, voxel_size, labels=None):
    """Measure exact channel/time coincidences, never infer physical firing identity."""
    valid = np.isfinite(ring) & np.isfinite(timestamps)
    if labels is not None:
        valid &= labels >= 0
    points, ring, timestamps = points[valid], ring[valid], timestamps[valid]
    dtype = [("ring", ring.dtype), ("time", timestamps.dtype)]
    if labels is not None:
        dtype.append(("component", labels.dtype))
    keys = np.empty(len(points), dtype=dtype)
    keys["ring"], keys["time"] = ring, timestamps
    if labels is not None:
        keys["component"] = labels[valid]
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    minimum = np.full((len(unique), 3), np.inf)
    maximum = np.full((len(unique), 3), -np.inf)
    np.minimum.at(minimum, inverse, points)
    np.maximum.at(maximum, inverse, points)
    repeated = counts > 1
    distinct_xyz = np.any(minimum != maximum, axis=1)
    multiple_cells = np.any(np.floor(minimum / voxel_size) != np.floor(maximum / voxel_size), axis=1)
    histogram_keys, histogram_counts = np.unique(counts, return_counts=True)
    result = {
        "eligible_points": len(points), "groups": len(unique),
        "group_size_histogram": dict(zip(map(str, histogram_keys), map(int, histogram_counts))),
        "repeated_groups": int(repeated.sum()),
        "repeated_groups_with_distinct_xyz": int((repeated & distinct_xyz).sum()),
        "repeated_groups_spanning_voxel_cells": int((repeated & multiple_cells).sum()),
        "maximum_group_size": int(counts.max()) if len(counts) else 0,
        "maximum_group_axis_span_m": float((maximum - minimum).max()) if len(counts) else 0.,
    }
    if labels is not None:
        affected = repeated & multiple_cells
        result["components_with_spanning_groups"] = np.unique(unique["component"][affected]).tolist()
    return result


def audit_returns(recipe_path):
    """Inspect saved real scans and their emitted objects without altering the detector."""
    recipe = json.loads(recipe_path.read_text())
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(Path(recipe["detector_config"]))
    if recipe["seed"] != config["seed"]:
        raise ValueError("Recipe and detector seeds differ")
    write_json(output / "experiment.json", recipe)
    manifest = {"inspector_sha256": digest(Path(__file__)),
                "recipe_sha256": digest(recipe_path), "runs": []}
    records = []
    for run_name in recipe["diagnostic_runs"]:
        root = Path(run_name)
        captured_config = json.loads((root / "detector.json").read_text())
        if captured_config != config or config["deskew_enabled"]:
            raise ValueError("Requires the declared, deskew-disabled captured configuration")
        experiment = json.loads((root / "experiment.json").read_text())
        files = sorted((root / "diagnostics").glob("*.npz"))
        if not files:
            raise ValueError(f"No captured real scans in {root}")
        manifest["runs"].append({"path": str(root), "manifest_sha256": digest(root / "manifest.json"),
                                 "detector_sha256": digest(root / "detector.json")})
        by_file = {}
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                record = {"run": str(root), "file": path.name, "sha256": digest(path),
                          "decoded": coincident_groups(data["decoded_points"], data["decoded_ring"],
                              data["decoded_raw_time"], config["cluster_voxel_m"]),
                          "cluster": None, "objects_with_spanning_groups": []}
                if "cluster_points" in data:
                    record["cluster"] = coincident_groups(data["cluster_points"], data["cluster_ring"],
                        data["cluster_raw_time"], config["cluster_voxel_m"], data["cluster_labels"])
                records.append(record)
                by_file[path.name] = record
        matched = set()
        for entry in experiment["bags"]:
            bag = Path(entry["path"]).name
            with (root / f"{bag}.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    name = Path(row.get("diagnostic_points", "")).name
                    if name not in by_file:
                        continue
                    if name in matched:
                        raise ValueError(f"Duplicate diagnostic reference: {name}")
                    matched.add(name)
                    record = by_file[name]
                    record.update(bag=bag, frame=row["frame"], status=row["status"])
                    component_ids = set((record["cluster"] or {}).get("components_with_spanning_groups", []))
                    for obj in row["objects"]:
                        if obj["component_id"] not in component_ids:
                            continue
                        record["objects_with_spanning_groups"].append({key: obj[key] for key in
                            ("component_id", "track_id", "path_relation", "confirmed",
                             "intersection_confirmed", "intersection_immediate", "support_voxels",
                             "uncertain_voxels", "accumulated_support_voxels", "hits")})
        if matched != set(by_file):
            raise ValueError(f"Unmatched captured scans: {set(by_file) - matched}")
    summary = {"captured_scans": len(records), "cluster_stages_executed": sum(r["cluster"] is not None for r in records),
               "decoded_points": sum(r["decoded"]["eligible_points"] for r in records),
               "decoded_spanning_groups": sum(r["decoded"]["repeated_groups_spanning_voxel_cells"] for r in records),
               "within_component_spanning_groups": sum((r["cluster"] or {}).get("repeated_groups_spanning_voxel_cells", 0) for r in records),
               "affected_object_observations": sum(len(r["objects_with_spanning_groups"]) for r in records),
               "affected_confirmed_observations": sum(bool(o["confirmed"]) for r in records for o in r["objects_with_spanning_groups"]),
               "affected_intersection_confirmed_observations": sum(bool(o["intersection_confirmed"]) for r in records for o in r["objects_with_spanning_groups"]),
               "firing_identity": "unknown", "return_multiplicity": "unknown",
               "interpretation": "Exact ring/time coincidences can span spatial support; these groups are NOT established firings. No deduplication, counterfactual detection or causal confirmation change is claimed."}
    write_json(output / "manifest.json", manifest)
    write_json(output / "records.json", records)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, nargs="?")
    parser.add_argument("--experiment", type=Path, help="Configured saved-scan return ambiguity audit")
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.experiment:
        if args.bag is not None or args.output is not None:
            parser.error("Use either a bag inspection or --experiment")
        audit_returns(args.experiment)
        return
    if args.bag is None or args.output is None:
        parser.error("Bag inspection requires a bag and --output")
    if args.max_frames < 1 or args.output.exists():
        parser.error("Use positive --max-frames and a new output path")
    config = load_config(args.config)
    store = get_typestore(Stores.ROS2_HUMBLE)
    records = []
    with Reader(args.bag) as reader:
        topics = [{"topic": c.topic, "type": c.msgtype, "messages": c.msgcount} for c in reader.connections]
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
        for c, record_ns, raw in reader.messages(connections=connections):
            if len(records) >= args.max_frames:
                break
            message = store.deserialize_cdr(raw, c.msgtype)
            points, times, invalid, duration, attributes = decode_cloud(message, np.asarray(config["sensor_rotation"]),
                                                                        np.asarray(config["sensor_translation"]))
            header_ns = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
            records.append({"topic": c.topic, "frame_id": message.header.frame_id,
                "record_ns": record_ns, "measurement_ns": header_ns, "record_minus_header_s": (record_ns-header_ns)/1e9,
                "width": message.width, "height": message.height, "point_step": message.point_step,
                "row_step": message.row_step, "bigendian": message.is_bigendian,
                "fields": [{"name": f.name, "offset": f.offset, "datatype": f.datatype, "count": f.count} for f in message.fields],
                "valid_points": len(points), "invalid_points": invalid, "normalized_point_times": len(times),
                "inferred_scan_duration_s": duration,
                "sensor_attributes": attributes.summary(),
                "range_counts": np.histogram(np.linalg.norm(points, axis=1), bins=config["range_bins_m"])[0].tolist()})
    report = {"bag": str(args.bag), "metadata_sha256": digest(args.bag / "metadata.yaml"),
              "config_sha256": digest(args.config), "topics": topics, "frames": records,
              "range_bins_m": config["range_bins_m"],
              "note": "Fields and clocks measured from this prefix only; no calibration, deskew provenance, or range validation implied."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for topic in sorted({r["topic"] for r in records}):
        rows = [r for r in records if r["topic"] == topic]
        delta = np.diff([r["measurement_ns"] for r in rows]) / 1e9
        print(json.dumps({"topic": topic, "frames": len(rows), "first": rows[0],
                          "acquisition_delta_s": delta.tolist()}, indent=2))


if __name__ == "__main__":
    main()
