"""Inspect bounded raw scans, layout, acquisition clocks and fitted geometry evidence."""
import argparse
from itertools import islice
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .detector import load_config
from .diagnostics import distribution
from .geometry import TrackGeometry, voxel_representatives
from .io import decode_cloud
from .run import bag_entry, digest, environment, write_json


def clock_summary(rows, max_gap_s):
    stamps = np.asarray([r["measurement_ns"] for r in rows], dtype=np.int64)
    delta = np.diff(stamps) / 1e9
    positive = delta[delta > 0]
    period = float(np.median(positive)) if len(positive) else None
    span = int(stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0
    return {"frames": len(rows), "first_measurement_ns": int(stamps[0]), "last_measurement_ns": int(stamps[-1]),
            "acquisition_span_s": span, "delta_s": distribution(delta),
            "median_period_hz": 1 / period if period else None,
            "messages_per_acquisition_span_hz": (len(rows) - 1) / span if span > 0 else None,
            "duplicate_timestamps": int(np.count_nonzero(delta == 0)),
            "backward_timestamps": int(np.count_nonzero(delta < 0)),
            "gap_rule": {"detector_reset_s": max_gap_s, "irregular_above_median_multiplier": 1.5},
            "timestamp_discontinuities": [{"frame": rows[i + 1]["frame"], "previous_ns": int(stamps[i]),
                 "measurement_ns": int(stamps[i+1]), "delta_s": float(dt),
                 "would_reset_detector": bool(dt > max_gap_s)} for i, dt in enumerate(delta)
                 if dt <= 0 or dt > max_gap_s or (period is not None and dt > 1.5 * period)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--topic")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.output.with_suffix(".source")
    if args.max_frames < 1 or args.output.exists() or source_dir.exists():
        parser.error("Use positive --max-frames and new output/source paths")
    config = load_config(args.config)
    # No odometry or surface rejection is needed for raw geometry inspection.
    geometry_config = config | {"background": config["background"] | {"enabled": False}}
    store = get_typestore(Stores.ROS2_HUMBLE)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, source_dir / source.name)
    write_json(source_dir / "detector.json", config)
    provenance = environment() | {"command": sys.argv,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)}
    (source_dir / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--", "tunnel_guard", "configs"]))
    write_json(source_dir / "manifest.json", provenance)
    with Reader(args.bag) as reader:
        topics = [{"topic": c.topic, "type": c.msgtype, "messages": c.msgcount} for c in reader.connections]
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"
                       and (args.topic is None or c.topic == args.topic)]
        if len({c.topic for c in connections}) != 1:
            raise ValueError("Select exactly one PointCloud2 topic with --topic")
        for index, (c, record_ns, raw) in enumerate(islice(reader.messages(connections=connections), args.max_frames)):
            message = store.deserialize_cdr(raw, c.msgtype)
            points, times, invalid, duration = decode_cloud(message, np.asarray(config["sensor_rotation"]),
                                                           np.asarray(config["sensor_translation"]))
            radii = np.linalg.norm(points, axis=1)
            mask = ((radii >= config["min_range_m"]) & (radii <= config["max_range_m"])
                    & (points[:, 0] >= config["min_forward_m"]) & (np.abs(points[:, 1]) < config["context_half_width_m"]))
            geometry = TrackGeometry(voxel_representatives(points[mask], config["geometry_voxel_m"]), geometry_config)
            header_ns = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
            field_names = [f.name for f in message.fields]
            records.append({"frame": index, "topic": c.topic, "frame_id": message.header.frame_id,
                "record_ns": record_ns, "measurement_ns": header_ns, "record_minus_header_s": (record_ns-header_ns)/1e9,
                "width": message.width, "height": message.height, "point_step": message.point_step,
                "row_step": message.row_step, "bigendian": message.is_bigendian,
                "fields": [{"name": f.name, "offset": f.offset, "datatype": f.datatype, "count": f.count} for f in message.fields],
                "timestamp_fields": [n for n in field_names if n in ("timestamp", "time", "t")],
                "return_fields": [n for n in field_names if "return" in n.lower()],
                "valid_points": len(points), "invalid_points": invalid, "normalized_point_times": len(times),
                "inferred_scan_duration_s": duration,
                "range_counts": np.histogram(radii, bins=config["range_bins_m"])[0].tolist(),
                "geometry": geometry.describe()})
    if not records:
        raise ValueError("No PointCloud2 messages found")
    report = {"bag": str(args.bag), "source": bag_entry({"path": str(args.bag)}),
              "config_sha256": digest(args.config), "topics": topics, "frames": records,
              "summary": clock_summary(records, config["frame_max_gap_s"]) | {
                  "geometry_valid_frames": sum(r["geometry"]["valid"] for r in records),
                  "frame_ids": sorted({r["frame_id"] for r in records}),
                  "path_horizon_m": distribution(r["geometry"]["path_horizon_m"] for r in records),
                  "gauge_inner_m": distribution(r["geometry"]["gauge_inner_median_m"] for r in records)},
              "range_bins_m": config["range_bins_m"],
              "evidence": {"measured": "Prefix message layout, header/record clocks, finite nonzero XYZ, radial histogram.",
                  "inferred": "Geometry fits after configured transform/range/crop/voxel. No ICP/deskew/background rejection in inspection. Rail gauge and head heights are fitted proxies, not survey measurements.",
                  "unknown": ["exact sensor model", "return mode and independent firing multiplicity", "point timestamp units/provenance",
                              "prior deskew", "extrinsics", "surveyed gauge and cant", "vehicle swept envelope", "obstacle detection range"]},
              "note": "Prefix coverage only. path_horizon_m is lateral path support, not a validated clearance horizon; consult ground uncertainty too. Record-minus-header time is not latency."}
    write_json(args.output, report)
    print(json.dumps({"bag": str(args.bag), "output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
