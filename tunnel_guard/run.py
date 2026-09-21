"""Run the configured detector against bags; preserve evidence, not inferred labels.

`prefetch_depth` in the experiment config (default 1) reads and decodes the next scan on one
producer thread while the detector processes the current one. It changes no measurement and no
decision; it removes the reader from the per-frame critical path. See `io.prefetch` and
`summary.latency_scope`."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

from .detector import Detector, load_config
from .io import iter_bag, prefetch


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def git_revision() -> str | None:
    """Installed wheels/containers need not contain a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


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


def capture_native_sources(destination: Path) -> dict:
    """Retain every translation unit/header and the build recipe, with hashes."""
    root = Path(__file__).parent.parent
    hashes = {}
    paths = sorted((root / "cpp").glob("*.cpp")) + sorted((root / "cpp").glob("*.h"))
    paths += [root / "setup.py", root / "MANIFEST.in"]
    for source in paths:
        if source.is_file():
            relative = source.relative_to(root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = digest(source)
    return hashes


def summarize(rows: list[dict], prefetch_depth: int = 0) -> dict:
    processing = np.asarray([r["processing_s"] for r in rows])
    e2e = np.asarray([r["read_and_process_s"] for r in rows])
    ingestion = np.asarray([r["ingestion_s"] for r in rows])
    inference = np.asarray([r["inference_s"] for r in rows])
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
            "inference_ms": {name: float(np.quantile(inference, q) * 1000) for name, q in (("p50", .5), ("p95", .95), ("p99", .99))},
            "ingestion_wait_ms": {name: float(np.quantile(ingestion, q) * 1000) for name, q in (("p50", .5), ("p95", .95), ("p99", .99))},
            # With prefetch_depth > 0 the scan is read and decoded on a producer thread, so
            # ingestion_wait_ms is the consumer's block on that thread, not the cost of
            # reading. read_and_process_ms then measures loop cost per frame, not the age of
            # a decision; full-bag wall_s / frames is the throughput that overlap buys.
            "prefetch_depth": prefetch_depth,
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
    revision = git_revision()
    source_root = Path(__file__).parent.parent
    manifest = environment() | {"command": sys.argv, "config_sha256": digest(config_path),
                                "started_unix_s": time.time(), "bags": [],
                                "git_revision": revision,
                                "git_status": (subprocess.check_output(["git", "status", "--porcelain=v1"],
                                    cwd=source_root, text=True) if revision is not None else None)}
    if revision is not None:
        (output / "working-tree.patch").write_bytes(subprocess.check_output(
            ["git", "diff", "HEAD", "--", "tunnel_guard", "configs"], cwd=source_root))
    if config.get("voxel_backend", "numpy") == "cpp" or config.get("native_kernels", False):
        from . import _native
        manifest["native_accelerator"] = {"binary_sha256": digest(Path(_native.__file__)),
            "module": "tunnel_guard._native", "backend": "cpp"}
        manifest["native_accelerator"]["sources_sha256"] = capture_native_sources(output / "source")
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
        ingestion = {}
        diagnostic_frames = set(experiment.get("diagnostic_frames", []))
        prefetch_depth = int(experiment.get("prefetch_depth", 1))
        iterator = prefetch(iter_bag(bag, config, every=experiment["every"],
                                     max_frames=experiment["max_frames"],
                                     topic=entry.get("topic"), diagnostics=ingestion),
                            depth=prefetch_depth)
        from contextlib import nullcontext
        from .visualization import ResultBag
        visual = (ResultBag(output / f"{bag.name}_rviz", config,
                            experiment.get("display_max_points", 100000))
                  if experiment.get("visualization", False) else nullcontext())
        with (output / f"{bag.name}.jsonl").open("x") as stream, visual as display, \
                (output / f"{bag.name}-timing.jsonl").open("x") as timing_stream:
            while True:
                frame_start = time.perf_counter()
                try:
                    scan = next(iterator)
                except StopIteration:
                    break
                inference_start = time.perf_counter()
                ingestion_s = inference_start - frame_start
                row = detector.process(scan.points, scan.timestamp_s, scan.point_times,
                                       capture_diagnostics=scan.index in diagnostic_frames)
                inference_s = time.perf_counter() - inference_start
                row.update(frame=scan.index, bag=bag.name, raw_points=scan.raw_points,
                           duplicate_return_points=scan.duplicate_return_points,
                           invalid_points=scan.invalid_points, sensor_frame=scan.frame_id,
                           topic=scan.topic, scan_duration_s=scan.scan_duration_s,
                           measurement_timestamp_ns=scan.measurement_timestamp_ns,
                           record_timestamp_ns=scan.record_timestamp_ns,
                           source_scan_id=f"{scan.topic}:{scan.frame_id}:{scan.measurement_timestamp_ns}",
                           skipped_duplicate_scans=scan.skipped_duplicate_scans,
                           deserialize_s=scan.deserialize_s, decode_s=scan.decode_s,
                           ingestion_s=ingestion_s, inference_s=inference_s,
                           read_and_process_s=time.perf_counter() - frame_start)
                if scan.index in diagnostic_frames:
                    diagnostic_start = time.perf_counter()
                    folder = output / "diagnostics"
                    folder.mkdir(exist_ok=True)
                    filename = f"{bag.name}_{scan.index:06d}.npz"
                    np.savez_compressed(folder / filename, **detector.diagnostic_arrays)
                    row["diagnostic_points"] = f"diagnostics/{filename}"
                    row["diagnostic_write_s"] = time.perf_counter() - diagnostic_start
                if display is not None:
                    display_started = time.perf_counter()
                    display.write(row, detector.display_points, scan.measurement_timestamp_ns, detector.display_support)
                    row["visualization_s"] = time.perf_counter() - display_started
                write_start = time.perf_counter()
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                result_write_s = time.perf_counter() - write_start
                iteration_s = time.perf_counter() - frame_start
                timing_stream.write(json.dumps({
                    "frame": scan.index, "measurement_timestamp_ns": scan.measurement_timestamp_ns,
                    "prefetch_depth": prefetch_depth,
                    "ingestion_s": ingestion_s, "deserialize_s": scan.deserialize_s,
                    "decode_s": scan.decode_s, "inference_s": inference_s,
                    "visualization_s": row.get("visualization_s", 0),
                    "diagnostic_write_s": row.get("diagnostic_write_s", 0),
                    "result_serialize_and_buffer_write_s": result_write_s,
                    "offline_iteration_s": iteration_s,
                }) + "\n")
                # Full object/support records are already durable in JSONL.
                # Keep only fields used by summarize(), not every track history
                # and covariance from the entire recording.
                rows.append({key: row[key] for key in (
                    "frame", "timestamp_s", "status", "processing_s", "read_and_process_s",
                    "ingestion_s", "inference_s")}
                    | {"geometry": {"valid": row.get("geometry", {}).get("valid", False)},
                       "motion": {"valid": row.get("motion", {}).get("valid", False)},
                       "diagnostic_write_s": row.get("diagnostic_write_s", 0),
                       "visualization_s": row.get("visualization_s", 0),
                       "decode_s": scan.decode_s, "offline_iteration_s": iteration_s})
                if len(rows) % 50 == 0:
                    stream.flush()
                    print(f"{bag.name}: {len(rows)} frames, {row['status']}, nearest={row['nearest_obstacle_m']}", flush=True)
        if not rows:
            raise ValueError(f"No scans processed from {bag}")
        summary = summarize(rows, prefetch_depth) | {"bag": bag.name, "split": entry["split"],
                                                     "wall_s": time.perf_counter() - start,
                                    "ingestion": ingestion,
                                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                                    "diagnostic_write_total_s": sum(r.get("diagnostic_write_s", 0) for r in rows),
                                    "visualization_total_s": sum(r.get("visualization_s", 0) for r in rows)}
        summary["stage_ms"] = {key: {name: float(np.quantile([r[key] for r in rows], q) * 1000)
                                             for name, q in (("p50", .5), ("p95", .95))}
                               for key in ("decode_s", "offline_iteration_s")}
        summary["latency_scope"] = (
            "Offline iteration: bag read/deserialization/decode, inference, optional diagnostic/"
            "RViz-bag writes, result JSON serialization and buffered write. Excludes timing-log "
            "write, fsync, live DDS queues/transport and viewer rendering; not sensor-to-display age. "
            "With prefetch_depth > 0 the read/deserialization/decode of the next scan runs on one "
            "producer thread while inference runs on this one, so read_and_process_ms is the cost of "
            "one loop iteration, not the age of a decision: a scan still waits for the scan ahead of "
            "it to finish. Wall_s / frames is the throughput that overlap buys.")
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
        print(json.dumps({k: summary[k] for k in ("bag", "frames", "status_frames", "processing_ms", "wall_s")}), flush=True)
    manifest["finished_unix_s"] = time.time()
    write_json(output / "manifest.json", manifest)
    print(f"Evidence: {output}")


if __name__ == "__main__":
    main()
