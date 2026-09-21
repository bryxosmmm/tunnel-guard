"""Compare registration quality references on a complete recorded sequence."""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .detector import Detector, load_config
from .io import iter_bag
from .run import digest, environment, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(a.config)
    detector = Detector(cfg)
    previous_dense = None
    previous_pose = None
    records = []
    write_json(a.output / "config.json", cfg)
    write_json(
        a.output / "manifest.json",
        environment()
        | {
            "bag": str(a.bag),
            "config_sha256": digest(a.config),
            "metadata_sha256": digest(a.bag / "metadata.yaml"),
        },
    )
    with (a.output / "motion.jsonl").open("x") as stream:
        for scan in iter_bag(a.bag, cfg):
            start = time.perf_counter()
            radii = np.linalg.norm(scan.points, axis=1)
            keep = (radii >= cfg["min_range_m"]) & (radii <= cfg["max_range_m"])
            frame, pose, old = detector._motion(
                scan.points[keep],
                scan.point_times[keep] if len(scan.point_times) else scan.point_times,
            )
            source = detector.previous_source
            _, dense = detector.odometry.voxelize(frame)
            row = {
                "frame": scan.index,
                "measurement_timestamp_ns": scan.measurement_timestamp_ns,
                "sparse_reference": old,
                "source_points": len(source),
                "dense_points": len(dense),
                "pose": pose.tolist(),
            }
            if previous_dense is not None:
                transform = np.linalg.inv(previous_pose) @ pose
                aligned = source @ transform[:3, :3].T + transform[:3, 3]
                distances = cKDTree(previous_dense).query(aligned)[0]
                median = float(np.median(distances))
                overlap = float(np.mean(distances < cfg["odometry_max_residual_m"] * 2))
                row["dense_reference"] = {
                    "median_residual_m": median,
                    "overlap": overlap,
                    "valid": bool(
                        np.isfinite(pose).all()
                        and median <= cfg["odometry_max_residual_m"]
                        and overlap >= cfg["odometry_min_overlap"]
                    ),
                }
                row["delta_translation_m"] = float(np.linalg.norm(transform[:3, 3]))
            previous_dense, previous_pose = dense, pose
            row["processing_s"] = time.perf_counter() - start
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            records.append(row)
            if len(records) % 50 == 0:
                stream.flush()
                print("motion frames", len(records), flush=True)
    valid = [r for r in records if "dense_reference" in r]
    summary = {
        "frames": len(records),
        "compared": len(valid),
        "sparse_rejected": sum(not r["sparse_reference"]["valid"] for r in valid),
        "dense_rejected": sum(not r["dense_reference"]["valid"] for r in valid),
        "changed_to_valid_frames": [
            r["frame"]
            for r in valid
            if not r["sparse_reference"]["valid"] and r["dense_reference"]["valid"]
        ],
        "changed_to_invalid_frames": [
            r["frame"]
            for r in valid
            if r["sparse_reference"]["valid"] and not r["dense_reference"]["valid"]
        ],
        "note": "Same estimated poses and same thresholds. Dense reference contains only previous-scan measured points. No pose ground truth or calibrated validity is implied.",
    }
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
