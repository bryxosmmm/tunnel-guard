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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
            points, times, invalid, duration = decode_cloud(message, np.asarray(config["sensor_rotation"]),
                                                           np.asarray(config["sensor_translation"]))
            header_ns = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
            records.append({"topic": c.topic, "frame_id": message.header.frame_id,
                "record_ns": record_ns, "measurement_ns": header_ns, "record_minus_header_s": (record_ns-header_ns)/1e9,
                "width": message.width, "height": message.height, "point_step": message.point_step,
                "row_step": message.row_step, "bigendian": message.is_bigendian,
                "fields": [{"name": f.name, "offset": f.offset, "datatype": f.datatype, "count": f.count} for f in message.fields],
                "valid_points": len(points), "invalid_points": invalid, "normalized_point_times": len(times),
                "inferred_scan_duration_s": duration,
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
