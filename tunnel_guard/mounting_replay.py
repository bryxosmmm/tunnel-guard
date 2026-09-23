"""Re-estimate railhead support after a prefix-frozen rotation on later real scans.

No height/center translation is forced and no production transform is installed.
This evaluates a local-track orientation hypothesis, not vehicle calibration.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag
from .mounting import observe_mounting
from .mounting_report import partition_stats
from .run import capture_native_sources, digest, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    source = Path(plan["source_run"])
    config = load_config(source / "detector.json")
    if config.get("deskew_enabled", False):
        raise ValueError("This stateless replay requires deskew-disabled source observations")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", plan)
    write_json(output / "detector.json", config)
    shutil.copytree(Path(__file__).parent, output / "source" / "tunnel_guard",
                    ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd", "*.dylib"))
    native_sources = capture_native_sources(output / "source")
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = environment() | {"source_manifest_sha256": digest(source / "manifest.json"),
                               "native_sources_sha256": native_sources, "candidates": []}
    from . import _native
    manifest["native_binary_sha256"] = digest(Path(_native.__file__))
    cfg = copy.deepcopy(config)
    cfg["background"]["enabled"] = False
    reference_origin = np.asarray(cfg["sensor_translation"])
    report_recipe = json.loads((Path(plan["candidate_report"]) / "experiment.json").read_text())
    summaries = []
    for entry in plan["bags"]:
        bag = Path(entry["path"])
        original_bag = next(b for b in source_manifest["bags"] if Path(b["path"]).name == bag.name)
        current_files = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                         for p in sorted(bag.glob("*.db3"))]
        if digest(bag / "metadata.yaml") != original_bag["metadata_sha256"] or current_files != original_bag["files"]:
            raise ValueError("Source bag identity changed since mounting observations")
        candidate_path = Path(plan["candidate_report"]) / f"{bag.name}.json"
        candidate = json.loads(candidate_path.read_text())
        if candidate["source_sha256"] != digest(source / f"{bag.name}.jsonl"):
            raise ValueError("Candidate and source observations disagree")
        matrix = candidate["frozen_processing_to_track_rotation"]
        if matrix is None:
            summaries.append({"bag": bag.name, "reason": "no_supported_prefix_rotation"})
            continue
        correction = np.asarray(matrix)
        cutoff = candidate["fit_cutoff_measurement_ns"]
        manifest["candidates"].append({"bag": bag.name, "candidate_sha256": digest(candidate_path),
                                       "fit_cutoff_measurement_ns": cutoff,
                                       "metadata_sha256": digest(bag / "metadata.yaml")})
        original = {}
        with (source / f"{bag.name}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["measurement_timestamp_ns"] > cutoff:
                    original[row["measurement_timestamp_ns"]] = {
                        k: row[k] for k in ("frame", "sensor_frame", "record_timestamp_ns", "mounting")}
        before, after = [], []
        with (output / f"{bag.name}.jsonl").open("x") as stream:
            for scan in iter_bag(bag, config, topic=entry.get("topic")):
                if scan.measurement_timestamp_ns <= cutoff:
                    continue
                if plan.get("max_validation_frames") and len(after) >= plan["max_validation_frames"]:
                    break
                saved = original.get(scan.measurement_timestamp_ns)
                if saved is None or (saved["frame"], saved["sensor_frame"], saved["record_timestamp_ns"]) != (
                        scan.index, scan.frame_id, scan.record_timestamp_ns):
                    raise ValueError("Validation acquisition identity differs from source")
                points = (scan.points - reference_origin) @ correction.T + reference_origin
                radii = np.linalg.norm(points, axis=1)
                keep = ((radii >= cfg["min_range_m"]) & (radii <= cfg["max_range_m"])
                        & (points[:, 0] >= cfg["min_forward_m"])
                        & (np.abs(points[:, 1]) < cfg["context_half_width_m"]))
                reduced = voxel_representatives(points[keep], cfg["geometry_voxel_m"])
                geometry = TrackGeometry(reduced, cfg)
                observation = observe_mounting(reduced, geometry, cfg)
                record = {"frame": scan.index, "measurement_timestamp_ns": scan.measurement_timestamp_ns,
                          "fit_cutoff_measurement_ns": cutoff, "before": saved["mounting"], "after": observation}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                before.append({"mounting": saved["mounting"]})
                after.append({"mounting": observation})
        if not after:
            raise ValueError("No strictly later validation acquisitions")
        summary = {"bag": bag.name, "validation_frames": len(after), "fit_cutoff_measurement_ns": cutoff,
                   "before": partition_stats(before, Rotation.from_matrix(correction), report_recipe),
                   "after": partition_stats(after, Rotation.identity(), report_recipe),
                   "calibration_installed": False,
                   "scope": "Frozen rotation about sensor origin, re-estimated on strictly later clouds; no translation or detector/accuracy evaluation"}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    main()
