"""Measure an operator-review policy on preserved real replay outputs, not inferred labels."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from .operator_channel import operator_decision
from .run import digest, environment, write_json


def measure(path: Path, config: dict, minimum_distance_m: float, stream):
    actions, states = Counter(), Counter()
    frames = episodes = gaps = invalid_motion_intervals = 0
    observed_s = alert_s = pose_distance = valid_distance = 0.0
    first_stamp = last_stamp = first_alert = None
    previous = None
    active_before = False
    examples = {}
    input_hash = hashlib.sha256()
    started = time.perf_counter()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        for line in source:
            input_hash.update(line)
            row = json.loads(line)
            stamp = row["timestamp_s"]
            if not math.isfinite(stamp) or (last_stamp is not None and stamp <= last_stamp):
                raise ValueError(f"Invalid acquisition order in {path}")
            if first_stamp is None:
                first_stamp = stamp
            decision = operator_decision(row, config)
            action = decision["operator_action"]
            active = action in ("inspect_obstacle", "inspect_unresolved")
            dt = stamp - last_stamp if last_stamp is not None else 0.0
            contiguous = last_stamp is not None and dt <= config["frame_max_gap_s"]
            if last_stamp is not None and not contiguous:
                gaps += 1
            if contiguous:
                observed_s += dt
                alert_s += dt if active_before else 0.0
                if "pose" in row and "pose" in previous:
                    distance = math.sqrt(sum((row["pose"][i][3] - previous["pose"][i][3])**2 for i in range(3)))
                    if not math.isfinite(distance):
                        raise ValueError(f"Nonfinite pose in {path}")
                    pose_distance += distance
                    if row.get("motion", {}).get("valid") and previous.get("motion", {}).get("valid"):
                        valid_distance += distance
                    else:
                        invalid_motion_intervals += 1
                else:
                    invalid_motion_intervals += 1
            if active and (not active_before or not contiguous):
                episodes += 1
            if active and first_alert is None:
                first_alert = stamp
            actions[action] += 1
            states[row["status"]] += 1
            record = {"bag": path.name.split(".jsonl")[0], "frame": row["frame"],
                      "timestamp_s": stamp, "detector_status": row["status"], "decision": decision}
            stream.write((json.dumps(record, allow_nan=False) + "\n").encode())
            examples.setdefault(row["status"], record)
            previous, last_stamp, active_before = row, stamp, active
            frames += 1
    if not frames:
        raise ValueError(f"Empty replay: {path}")
    alert_frames = actions["inspect_obstacle"] + actions["inspect_unresolved"]
    return {"input": str(path), "input_archive_sha256": digest(path),
            "input_uncompressed_sha256": input_hash.hexdigest(), "frames": frames,
            "detector_status_frames": dict(states), "operator_action_frames": dict(actions),
            "alert_frames": alert_frames, "alert_episodes": episodes,
            "elapsed_s": last_stamp - first_stamp, "observed_interval_s": observed_s,
            "excluded_gaps": gaps, "alert_time_s": alert_s,
            "alert_time_fraction": alert_s / observed_s if observed_s else None,
            "alert_episodes_per_minute": episodes * 60 / observed_s if observed_s else None,
            "alert_frames_per_minute": alert_frames * 60 / observed_s if observed_s else None,
            "pose_distance_m": pose_distance, "valid_motion_distance_m": valid_distance,
            "invalid_motion_intervals": invalid_motion_intervals,
            "alert_episodes_per_estimated_km": episodes * 1000 / pose_distance if pose_distance >= minimum_distance_m else None,
            "alert_frames_per_estimated_km": alert_frames * 1000 / pose_distance if pose_distance >= minimum_distance_m else None,
            "km_rate_validity": "odometry_proxy_not_calibrated_distance" if pose_distance >= minimum_distance_m else "insufficient_travel",
            "first_alert_from_recording_start_s": first_alert - first_stamp if first_alert is not None else None,
            "first_alert_validity": "Not object-onset latency; labels do not establish onset",
            "false_alarms_per_km": None, "precision": None, "calibrated_probability": None,
            "label_validity": "No exhaustive negative intervals; alerts are not adjudicated false positives",
            "audit_wall_s": time.perf_counter() - started, "examples_by_status": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    root = Path(recipe["input_run"])
    config = json.loads((root / "detector.json").read_text())
    if recipe["seed"] != config["seed"]:
        raise ValueError("Recipe and captured detector seeds differ")
    minimum = float(recipe["minimum_reporting_distance_m"])
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("Positive minimum_reporting_distance_m required")
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", recipe)
    manifest = environment() | {"recipe_sha256": digest(args.experiment),
        "captured_manifest_sha256": digest(root / "manifest.json"),
        "captured_config_sha256": digest(root / "detector.json"),
        "policy_sha256": digest(Path(__file__).with_name("operator_channel.py")),
        "scope": "Policy evaluation on recorded outputs; no new detector replay or target-runtime benchmark"}
    write_json(output / "manifest.json", manifest)
    summaries = []
    with gzip.GzipFile(filename=str(output / "decisions.jsonl.gz"), mode="wb", mtime=0) as stream:
        for bag in recipe["bags"]:
            path = root / f"{bag}.jsonl.gz"
            if not path.is_file():
                path = root / f"{bag}.jsonl"
            summaries.append(measure(path, config, minimum, stream))
    report = {"runs": summaries, "frames": sum(s["frames"] for s in summaries),
              "policy": "operator_review_only", "target_latency": {"state": "not_measured", "required_cpu": "Intel i7-9700E"},
              "interval_convention": "Hold previous alert to next scan, exclude gaps above frame_max_gap_s, omit unknown terminal interval",
              "episode_convention": "Contiguous inspect_obstacle/inspect_unresolved frames; not physical object events",
              "distance_convention": "Sum pose translation norms, report valid-motion subset separately; never stitch distinct recordings",
              "decision_archive_sha256": digest(output / "decisions.jsonl.gz")}
    write_json(output / "summary.json", report)
    print(json.dumps({"frames": report["frames"], "bags": len(summaries), "output": str(output),
                      "alert_frames": sum(s["alert_frames"] for s in summaries),
                      "alert_episodes": sum(s["alert_episodes"] for s in summaries)}, indent=2))


if __name__ == "__main__":
    main()
