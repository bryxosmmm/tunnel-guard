"""Run the configured detector against bags; preserve evidence, not inferred labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

from .detector import Detector, load_config
from .io import iter_bag


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def environment() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "rosbags", "kiss-icp", "open3d", "travel-seg", "hdbscan", "pypatchworkpp"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "packages": packages,
            "source_sha256": {str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def summarize(rows: list[dict]) -> dict:
    processing = np.asarray([r["processing_s"] for r in rows])
    e2e = np.asarray([r["read_and_process_s"] for r in rows])
    counts = Counter(r["status"] for r in rows)
    duration = rows[-1]["timestamp_s"] - rows[0]["timestamp_s"] if len(rows) > 1 else 0.0
    alarms = []
    current = None
    for row in rows:
        active = row["status"] in ("obstacle", "unresolved_obstacle")
        if active and current is None:
            current = {"start_frame": row["frame"], "start_s": row["timestamp_s"], "end_s": row["timestamp_s"], "frames": 0}
        if active:
            current["end_s"] = row["timestamp_s"]
            current["frames"] += 1
        elif current is not None:
            alarms.append(current)
            current = None
    if current is not None:
        alarms.append(current)
    return {"frames": len(rows), "recording_duration_s": duration, "status_frames": dict(counts),
            "geometry_valid_frames": sum(r.get("geometry", {}).get("valid", False) for r in rows),
            "motion_valid_frames": sum(r.get("motion", {}).get("valid", False) for r in rows),
            "processing_ms": {name: float(np.quantile(processing, q) * 1000) for name, q in (("p50", .5), ("p95", .95), ("p99", .99))},
            "read_and_process_ms": {name: float(np.quantile(e2e, q) * 1000) for name, q in (("p50", .5), ("p95", .95), ("p99", .99))},
            "alarm_episodes_unlabelled": alarms,
            "accuracy": None, "accuracy_validity": "No ground truth implied by bag name or alarm count"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    config_path = Path(experiment["detector_config"])
    config = load_config(config_path)
    if experiment["seed"] != config["seed"]:
        raise ValueError("Experiment and detector seed disagree")
    output = Path(experiment["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", experiment)
    write_json(output / "detector.json", config)
    source_dir = output / "source" / "tunnel_guard"
    source_dir.mkdir(parents=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, source_dir / source.name)
    manifest = environment() | {"command": sys.argv, "config_sha256": digest(config_path),
                                "started_unix_s": time.time(), "bags": []}
    write_json(output / "manifest.json", manifest)
    summaries = []
    for entry in experiment["bags"]:
        bag = Path(entry["path"])
        metadata = bag / "metadata.yaml"
        manifest["bags"].append(entry | {"metadata_sha256": digest(metadata),
            "files": [{"name": str(p.name), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                      for p in sorted(bag.glob("*.db3"))]})
        write_json(output / "manifest.json", manifest)
        detector = Detector(config)
        rows = []
        start = time.perf_counter()
        iterator = iter_bag(bag, config, every=experiment["every"], max_frames=experiment["max_frames"], topic=entry.get("topic"))
        with (output / f"{bag.name}.jsonl").open("x") as stream:
            while True:
                frame_start = time.perf_counter()
                try:
                    scan = next(iterator)
                except StopIteration:
                    break
                row = detector.process(scan.points, scan.timestamp_s, scan.point_times)
                row.update(frame=scan.index, bag=bag.name, raw_points=scan.raw_points,
                           invalid_points=scan.invalid_points, sensor_frame=scan.frame_id,
                           topic=scan.topic, scan_duration_s=scan.scan_duration_s,
                           read_and_process_s=time.perf_counter() - frame_start)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if len(rows) % 50 == 0:
                    stream.flush()
                    print(f"{bag.name}: {len(rows)} frames, {row['status']}, nearest={row['nearest_obstacle_m']}", flush=True)
        if not rows:
            raise ValueError(f"No scans processed from {bag}")
        summary = summarize(rows) | {"bag": bag.name, "split": entry["split"], "wall_s": time.perf_counter() - start}
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
        print(json.dumps({k: summary[k] for k in ("bag", "frames", "status_frames", "processing_ms", "wall_s")}), flush=True)
    manifest["finished_unix_s"] = time.time()
    write_json(output / "manifest.json", manifest)
    print(f"Evidence: {output}")


if __name__ == "__main__":
    main()
