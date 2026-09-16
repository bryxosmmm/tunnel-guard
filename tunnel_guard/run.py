"""Run the configured detector against bags; preserve evidence, not inferred labels."""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import hashlib
import importlib.metadata
import json
import multiprocessing
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


def ordered_parallel(items: list, worker, workers: int):
    """Yield worker(item) results in input order, spreading independent work over processes."""
    if workers <= 1 or len(items) <= 1:
        for item in items:
            yield worker(item)
        return
    pool = multiprocessing.get_context("fork").Pool(min(workers, len(items)))
    with pool:
        yield from pool.imap(worker, items, chunksize=1)


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


def run_bag(output: Path, config: dict, experiment: dict, entry: dict) -> dict:
    """Process one recording end to end; each bag carries its own detector state."""
    bag = Path(entry["path"])
    detector = Detector(config)
    rows = []
    start = time.perf_counter()
    ingestion = {}
    iterator = iter_bag(bag, config, every=experiment["every"], max_frames=experiment["max_frames"],
                        topic=entry.get("topic"), diagnostics=ingestion)
    from contextlib import nullcontext
    from .visualization import ResultBag
    visual = (ResultBag(output / f"{bag.name}_rviz", config,
                        experiment.get("display_max_points", 100000))
              if experiment.get("visualization", False) else nullcontext())
    with (output / f"{bag.name}.jsonl").open("x") as stream, visual as display:
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
                       measurement_timestamp_ns=scan.measurement_timestamp_ns,
                       record_timestamp_ns=scan.record_timestamp_ns,
                       source_scan_id=f"{scan.topic}:{scan.frame_id}:{scan.measurement_timestamp_ns}",
                       skipped_duplicate_scans=scan.skipped_duplicate_scans,
                       read_and_process_s=time.perf_counter() - frame_start)
            if display is not None:
                display_started = time.perf_counter()
                display.write(row, detector.display_points, scan.measurement_timestamp_ns, detector.display_support)
                row["visualization_s"] = time.perf_counter() - display_started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            if len(rows) % 50 == 0:
                stream.flush()
                print(f"{bag.name}: {len(rows)} frames, {row['status']}, nearest={row['nearest_obstacle_m']}", flush=True)
    if not rows:
        raise ValueError(f"No scans processed from {bag}")
    return summarize(rows) | {"bag": bag.name, "split": entry["split"], "wall_s": time.perf_counter() - start,
                              "ingestion": ingestion,
                              "visualization_total_s": sum(r.get("visualization_s", 0) for r in rows)}


def bag_entry(entry: dict) -> dict:
    bag = Path(entry["path"])
    return entry | {"metadata_sha256": digest(bag / "metadata.yaml"),
                    "files": [{"name": str(p.name), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                              for p in sorted(bag.glob("*.db3"))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="processes for independent recordings; each bag keeps its own detector state")
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    config_path = Path(experiment["detector_config"])
    config = load_config(config_path)
    if experiment["seed"] != config["seed"]:
        raise ValueError("Experiment and detector seed disagree")
    for entry in experiment["bags"]:
        if not (Path(entry["path"]) / "metadata.yaml").is_file():
            raise FileNotFoundError(f"ROS bag metadata missing: {entry['path']}/metadata.yaml")
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
    manifest["bags"] = [bag_entry(entry) for entry in experiment["bags"]]
    write_json(output / "manifest.json", manifest)
    summaries = []
    workers = max(1, min(args.workers, len(experiment["bags"])))
    worker = functools.partial(run_bag, output, config, experiment)
    for summary in ordered_parallel(experiment["bags"], worker, workers):
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
        print(json.dumps({k: summary[k] for k in ("bag", "frames", "status_frames", "processing_ms", "wall_s")}), flush=True)
    manifest["finished_unix_s"] = time.time()
    write_json(output / "manifest.json", manifest)
    print(f"Evidence: {output}")


if __name__ == "__main__":
    main()
