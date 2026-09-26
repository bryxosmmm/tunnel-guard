"""Replay one saved real ICP input without updating the map or prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from kiss_icp.pybind import kiss_icp_pybind

from .detector import Detector, load_config
from .run import digest, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    config = load_config(Path(plan["detector_config"]))
    if config["seed"] != plan["seed"]:
        raise ValueError("Seed mismatch")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    registration = Detector(config).odometry
    with np.load(plan["input"], allow_pickle=False) as data:
        points = data["motion_source"]
        initial = data["motion_initial_guess"]
        mapped = data["motion_map"]
        sigma = float(data["motion_sigma"])
    registration.local_map.add_points(mapped)
    restored = registration.local_map.point_cloud()
    # Exact point multiset, independent of hash-map traversal order.
    sort_rows = lambda array: array[np.lexsort(array.T[::-1])]
    if not np.array_equal(sort_rows(mapped), sort_rows(restored)):
        raise ValueError("Restored real map differs from captured map")
    poses, durations = [], []
    for _ in range(plan["repetitions"]):
        started = time.perf_counter()
        poses.append(registration.registration.align_points_to_map(
            points=points, voxel_map=registration.local_map, initial_guess=initial,
            max_correspondance_distance=3 * sigma, kernel=sigma))
        durations.append(time.perf_counter() - started)
    poses = np.asarray(poses)
    np.savez_compressed(output / "poses.npz", poses=poses)
    report = {"plan": plan, "environment": environment(), "input_sha256": digest(Path(plan["input"])),
              "config_sha256": digest(Path(plan["detector_config"])),
              "kiss_binary_sha256": digest(Path(kiss_icp_pybind.__file__)),
              "odometry_threads": config["odometry_threads"],
              "distinct_pose_bit_patterns": len({pose.tobytes() for pose in poses}),
              "max_abs_difference_from_first": float(np.max(np.abs(poses - poses[0]))),
              "alignment_seconds": durations, "map_points": len(mapped), "source_points": len(points),
              "scope": "Same real source, restored map, initial guess and scale; no map or predictor updates. Each thread configuration runs in a fresh process because KISS 1.3.0 holds static TBB global_control."}
    write_json(output / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("odometry_threads", "distinct_pose_bit_patterns", "max_abs_difference_from_first")}, indent=2))


if __name__ == "__main__":
    main()
