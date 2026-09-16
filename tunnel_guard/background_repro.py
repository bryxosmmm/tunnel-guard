"""Measure repeatability on selected real scans with fixed seeds and fresh state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .detector import Detector, load_config
from .io import decode_cloud
from .run import environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(recipe["detector_config"])
    source_rows = {
        r["frame"]: r
        for r in map(
            json.loads, Path(recipe["reference_jsonl"]).read_text().splitlines()
        )
    }
    write_json(output / "experiment.json", recipe)
    write_json(output / "detector.json", config)
    write_json(
        output / "manifest.json",
        environment() | {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
    )
    shutil.copyfile(__file__, output / "experiment_source.py")
    store = get_typestore(Stores.ROS2_HUMBLE)
    records = []
    with Reader(Path(recipe["bag"])) as reader:
        for frame in recipe["frames"]:
            reference = source_rows[frame]
            connections = [
                c for c in reader.connections if c.topic == reference["topic"]
            ]
            stamp = reference["record_timestamp_ns"]
            connection, _, raw = next(
                reader.messages(connections=connections, start=stamp, stop=stamp + 1)
            )
            message = store.deserialize_cdr(raw, connection.msgtype)
            measured = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if measured != reference["measurement_timestamp_ns"]:
                raise ValueError("Source measurement does not match reference row")
            points, times, _, _ = decode_cloud(
                message,
                np.asarray(config["sensor_rotation"]),
                np.asarray(config["sensor_translation"]),
            )
            for repetition in range(recipe["repetitions"]):
                detector = Detector(config)
                row = detector.process(
                    points, measured * 1e-9, times, capture_diagnostics=True
                )
                row.update(
                    source_frame_index=frame,
                    repetition=repetition,
                    measurement_timestamp_ns=measured,
                )
                arrays = detector.diagnostic_arrays
                np.savez_compressed(
                    output / f"frame_{frame:06d}_repeat_{repetition}.npz", **arrays
                )
                write_json(output / f"frame_{frame:06d}_repeat_{repetition}.json", row)
                record = {
                    "frame": frame,
                    "repetition": repetition,
                    "status": row["status"],
                    "background": row.get("geometry", {}).get("background"),
                    "cluster_points": row["pipeline"]["segmentation"].get(
                        "cluster_points"
                    ),
                    "candidates": len(row["objects"]),
                    "processing_s": row["processing_s"],
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    write_json(
        output / "summary.json",
        {
            "observations": records,
            "interpretation": "Fresh detector for each repeat; fixed configuration and same measured scan. This measures within-scan repeatability, not tracking accuracy or object truth.",
        },
    )


if __name__ == "__main__":
    main()
