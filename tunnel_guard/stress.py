"""Deterministic, occlusion-correct synthetic tests; explicitly not field accuracy."""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import subprocess
import sys
import os
from pathlib import Path
import time

import numpy as np

from .detector import Detector, load_config
from .evaluate import evaluate_frames
from .run import digest, environment, ordered_parallel, write_json


def beam_directions(config: dict, rng: np.random.Generator) -> np.ndarray:
    azimuth = (np.arange(config["azimuth_bins"]) + rng.uniform(-0.5, 0.5)) * 2 * np.pi / config["azimuth_bins"] - np.pi
    elevation = np.deg2rad(np.concatenate([np.linspace(lo, hi, int(n), endpoint=False)
                                         for lo, hi, n in config["elevation_segments_deg"]]))
    a, e = np.meshgrid(azimuth, elevation)
    return np.column_stack((np.cos(e.ravel()) * np.cos(a.ravel()),
                            np.cos(e.ravel()) * np.sin(a.ravel()), np.sin(e.ravel())))


def ray_box(directions: np.ndarray, origin: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    parallel = np.abs(directions) < 1e-12
    safe = np.where(parallel, 1.0, directions)
    low = (minimum - origin) / safe
    high = (maximum - origin) / safe
    entries, exits = np.minimum(low, high), np.maximum(low, high)
    outside = parallel & ((origin < minimum) | (origin > maximum))
    entries[parallel] = -np.inf
    exits[parallel] = np.inf
    near, far = entries.max(axis=1), exits.min(axis=1)
    valid = (far >= np.maximum(near, 0)) & ~outside.any(axis=1)
    distance = np.where(near > 0, near, far)
    return np.where(valid & (distance > 0), distance, np.inf)


def scene_scan(config: dict, directions: np.ndarray, origin_x: float, obj: dict | None,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    origin = np.array([origin_x, 0.0, 0.0])
    ground, ceiling = config["ground_z_m"], config["tunnel_ceiling_z_m"]
    width, end = config["tunnel_half_width_m"], config["scene_end_m"]
    # The sensor is inside this box: the exit is the first floor/wall/ceiling hit.
    background = ray_box(directions, origin, np.array([-5.0, -width, ground]), np.array([end, width, ceiling]))
    for y in [-config["rail_gauge_m"] / 2, config["rail_gauge_m"] / 2]:
        rail = ray_box(directions, origin,
                       np.array([-5.0, y - config["rail_width_m"] / 2, ground]),
                       np.array([end, y + config["rail_width_m"] / 2, ground + config["rail_height_m"]]))
        background = np.minimum(background, rail)
    # A legitimate low trackside installation, present in every negative scene.
    fixture = ray_box(directions, origin, np.array([8.0, -2.25, ground + 0.7]),
                      np.array([60.0, -2.0, ground + 0.9]))
    background = np.minimum(background, fixture)
    target = np.full(len(directions), np.inf)
    if obj is not None:
        target = ray_box(directions, origin, np.asarray(obj["bbox_min"]), np.asarray(obj["bbox_max"]))
    target_visible = target < background
    distance = np.minimum(background, target)
    keep = np.isfinite(distance) & (rng.random(len(distance)) >= config["random_drop_probability"])
    distance = distance[keep] + rng.normal(0, config["range_noise_sigma_m"], int(keep.sum()))
    return directions[keep] * distance[:, None], target_visible[keep]


def cases(config: dict):
    for step in config["sensor_step_m"]:
        yield {"range_m": None, "lateral_m": None, "dimensions_m": None, "step_m": step, "hazard": False}
    for distance, lateral, dims, step in itertools.product(config["ranges_m"], config["lateral_m"],
                                                          config["object_dimensions_m"], config["sensor_step_m"]):
        yield {"range_m": distance, "lateral_m": lateral, "dimensions_m": dims, "step_m": step,
               "hazard": abs(lateral) < config["rail_gauge_m"] / 2}


def run_case(case: dict, stress: dict, detector_config: dict, index: int):
    rng = np.random.default_rng(np.random.SeedSequence([stress["seed"], index]))
    rays = beam_directions(stress, rng)
    obj = None
    if case["range_m"] is not None:
        dx, dy, dz = case["dimensions_m"]
        lo = np.array([case["range_m"], case["lateral_m"] - dy / 2, stress["ground_z_m"] + stress["rail_height_m"]])
        obj = {"bbox_min": lo.tolist(), "bbox_max": (lo + [dx, dy, dz]).tolist()}
    detector = Detector(detector_config)
    rows, labels, visible_counts = [], [], []
    for frame in range(stress["frames_per_case"]):
        origin_x = case["step_m"] * frame
        cloud, visible = scene_scan(stress, rays, origin_x, obj, rng)
        row = detector.process(cloud, frame * stress["frame_period_s"],
                               capture_association=stress.get("association_diagnostics", False))
        row.update(frame=frame, bag=f"case_{index:03d}")
        if "tracking_diagnostics" in row:
            for event in row["tracking_diagnostics"]["events"]:
                event["frame"] = frame
        rows.append(row)
        visible_counts.append(int(visible.sum()))
        truth = []
        if case["hazard"]:
            if visible.any():
                # Detector outputs observed support, not an amodal object shape.
                # Compare to independently ray-cast visible support, including noise.
                support = cloud[visible]
                lo, hi = support.min(axis=0), support.max(axis=0)
                hi = np.maximum(hi, lo + 1e-3)
            else:
                # A physically present but invisible object still counts as a miss.
                lo, hi = np.asarray(obj["bbox_min"]) - [origin_x, 0, 0], np.asarray(obj["bbox_max"]) - [origin_x, 0, 0]
            truth = [{"event_id": "inserted_object", "bbox_min": lo.tolist(), "bbox_max": hi.tolist()}]
        labels.append({"bag": row["bag"], "frame": frame, "exhaustive": True, "objects": truth})
    return rows, labels, visible_counts


def run_case_indexed(item: tuple[int, dict], stress: dict, detector_config: dict):
    index, case = item
    rows, labels, visible = run_case(case, stress, detector_config, index)
    return index, rows, labels, visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="processes for independent scenes; each scene is deterministic from the plan seed")
    args = parser.parse_args()
    stress = json.loads(args.experiment.read_text())
    detector_config = load_config(stress["detector_config"])
    if stress["seed"] != detector_config["seed"]:
        raise ValueError("Stress and detector seeds disagree")
    output = Path(stress["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", stress)
    write_json(output / "detector.json", detector_config)
    source_dir = output / "source" / "tunnel_guard"
    source_dir.mkdir(parents=True)
    for source in Path(__file__).parent.glob("*.py"):
        (source_dir / source.name).write_bytes(source.read_bytes())
    manifest = environment() | {"config_sha256": digest(Path(stress["detector_config"])),
        "command": sys.argv, "started_unix_s": time.time(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)}
    write_json(output / "manifest.json", manifest)
    (output / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--", "tunnel_guard", "configs"]))
    case_list = list(cases(stress))
    workers = max(1, min(args.workers, len(case_list)))
    predictions, annotations, records = {}, [], []
    started = time.perf_counter()
    worker = functools.partial(run_case_indexed, stress=stress, detector_config=detector_config)
    with (output / "predictions.jsonl").open("x") as stream:
        for index, rows, labels, visible in ordered_parallel(list(enumerate(case_list)), worker, workers):
            case = case_list[index]
            for row in rows:
                predictions[(row["bag"], row["frame"])] = row
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            annotations.extend(labels)
            panel = {"label_status": "synthetic_exact", "prediction_scope": "collision_hazards",
                     "minimum_iou": stress["success_criterion"]["minimum_iou"], "frames": labels}
            score = evaluate_frames(predictions, panel)
            record = case | {"index": index, "visible_points_per_frame": visible,
                             "event_detected": score["event_recall"], "tp": score["tp"], "fn": score["fn"],
                             "fp": score["fp_exhaustive_only"], "negative_alarm": not case["hazard"] and any(r["status"] in ("obstacle", "unresolved_obstacle") for r in rows),
                             "unknown_frames": sum(r["status"] == "unknown" for r in rows)}
            records.append(record)
            print(json.dumps(record), flush=True)
    panel = {"label_status": "synthetic_exact", "prediction_scope": "collision_hazards",
             "minimum_iou": stress["success_criterion"]["minimum_iou"], "frames": annotations}
    score = evaluate_frames(predictions, panel)
    negatives = [r for r in records if not r["hazard"]]
    negative_rate = sum(r["negative_alarm"] for r in negatives) / len(negatives)
    by_range = []
    for distance in stress["ranges_m"]:
        group = [r for r in records if r["hazard"] and r["range_m"] == distance]
        by_range.append({"range_m": distance, "cases": len(group),
                         "event_recall": float(np.mean([r["event_detected"] for r in group])),
                         "invisible_cases": sum(max(r["visible_points_per_frame"]) == 0 for r in group)})
    score.update(negative_episode_rate=negative_rate, range_results=by_range, wall_s=time.perf_counter() - started,
                 pass_criterion=(score["event_recall"] is not None and score["event_recall"] >= stress["success_criterion"]["event_recall_min"]
                                 and negative_rate <= stress["success_criterion"]["negative_episode_rate_max"]),
                 note=stress["note"] + " Localization IoU uses visible support, not amodal boxes.")
    write_json(output / "annotations.json", panel)
    write_json(output / "cases.json", records)
    write_json(output / "metrics.json", score)
    manifest["finished_unix_s"] = time.time()
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in score.items() if k not in ("frames", "events")}, indent=2))


if __name__ == "__main__":
    main()
