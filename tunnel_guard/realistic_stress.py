"""HYBRID SYNTHETIC replay: opaque boxes on measured empty-tunnel return rays."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .detector import Detector, load_config
from .io import iter_bag
from .run import digest, environment, write_json
from .stress import ray_box


def insert_box(scan, pose: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], angular_cell_deg: float):
    """Use only measured return directions; nearer measured returns occlude the box.

    The original cloud is retained except for returns on rays intercepted by the
    opaque box. One synthetic first return is emitted per occupied angular cell.
    This intentionally cannot model beams that supplied no measured return.
    """
    points = scan.points
    radius = np.linalg.norm(points, axis=1)
    directions = points / radius[:, None]
    world_directions = directions @ pose[:3, :3].T
    hit = ray_box(world_directions, pose[:3, 3], *bounds)
    intercepted = np.isfinite(hit)
    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    elevation = np.degrees(np.arcsin(directions[:, 2]))
    cells = np.column_stack((np.rint(azimuth / angular_cell_deg).astype(np.int32),
                             np.rint(elevation / angular_cell_deg).astype(np.int32)))
    _, cell_index = np.unique(cells, axis=0, return_inverse=True)
    cell_count = int(cell_index.max()) + 1
    nearest_real = np.full(cell_count, np.inf)
    nearest_box = np.full(cell_count, np.inf)
    np.minimum.at(nearest_real, cell_index, radius)
    np.minimum.at(nearest_box, cell_index, hit)
    visible_cell = nearest_box < nearest_real - 0.02
    visible = intercepted & visible_cell[cell_index]
    # A foreground return in the same angular cell vetoes the synthetic hit.
    remove = np.flatnonzero(visible_cell[cell_index] & (radius > nearest_box[cell_index] + 0.02))
    candidates = np.flatnonzero(visible)
    if len(candidates):
        order = np.lexsort((hit[candidates], cell_index[candidates]))
        sorted_cells = cell_index[candidates[order]]
        first = np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
        selected = candidates[order[first]]
    else:
        selected = np.empty(0, dtype=np.int64)
    inserted = directions[selected] * hit[selected, None]
    keep = np.ones(len(points), dtype=bool)
    keep[remove] = False
    cloud = np.vstack((points[keep], inserted))
    point_times = (np.r_[scan.point_times[keep], scan.point_times[selected]]
                   if len(scan.point_times) else np.empty(0))
    provenance = {"intersecting_measured_return_rays": int(intercepted.sum()),
                  "foreground_occluded_rays": int((intercepted & ~visible).sum()),
                  "removed_background_returns": len(remove),
                  "visible_inserted_returns": len(selected),
                  "source_returns": len(points), "output_returns": len(cloud)}
    return cloud, point_times, remove, selected, inserted, provenance


def matched_object(row: dict, support: np.ndarray, tolerance_m: float) -> dict | None:
    if not len(support):
        return None
    lo, hi = support.min(axis=0) - tolerance_m, support.max(axis=0) + tolerance_m
    choices = []
    for obj in row.get("objects", []):
        lower, upper = np.asarray(obj["bbox_min"]), np.asarray(obj["bbox_max"])
        overlap = np.maximum(0, np.minimum(hi, upper) - np.maximum(lo, lower))
        if np.all(overlap > 0):
            choices.append((float(np.prod(overlap)), obj))
    return max(choices, key=lambda x: x[0])[1] if choices else None


def scene_cases(split: dict):
    for shape, distance, position, motion in itertools.product(
            split["shapes"], split["ranges_m"], split["positions"], split["motions"]):
        yield {"shape": shape, "range_m": distance, "position": position, "motion": motion}


def run_split(split: dict, recipe: dict, detector_config: dict, output: Path):
    bag = Path(split["bag"])
    scans = list(iter_bag(bag, detector_config, max_frames=split["frames"]))
    if len(scans) != split["frames"]:
        raise ValueError(f"Insufficient frames in {bag}")
    baseline = Detector(detector_config)
    baseline_rows = [baseline.process(s.points, s.timestamp_s, s.point_times) for s in scans]
    poses = [np.asarray(row["pose"]) for row in baseline_rows]
    ground = baseline_rows[0]["geometry"]["ground_plane"]
    rail_height = baseline_rows[0]["geometry"]["rail_head_height_m"]
    if ground is None or rail_height is None:
        raise ValueError(f"No supported near ground/rail geometry in {bag}")
    with (output / f"baseline-{split['name']}.jsonl").open("x") as stream:
        for scan, row in zip(scans, baseline_rows):
            stream.write(json.dumps(row | {"source_frame": scan.index, "label": "HYBRID SYNTHETIC paired real background"}, allow_nan=False) + "\n")
    identity = {"split": split["name"], "bag": str(bag), "metadata_sha256": digest(bag / "metadata.yaml"),
                "db3": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                        for p in sorted(bag.glob("*.db3"))],
                "source_frame_indices": [s.index for s in scans],
                "measurement_timestamp_ns": [s.measurement_timestamp_ns for s in scans],
                "baseline_pose_valid": [r["motion"]["valid"] for r in baseline_rows],
                "ground_plane_at_first_frame": ground, "rail_height_m_at_first_frame": rail_height,
                "source_frame_id": scans[0].frame_id, "topic": scans[0].topic}
    evidence = output / "provenance" / split["name"]
    evidence.mkdir(parents=True)
    records = []
    with (output / f"predictions-{split['name']}.jsonl").open("x") as stream:
        for number, case in enumerate(scene_cases(split)):
            shape = case["shape"]
            x, y = case["range_m"], case["position"]["lateral_m"]
            dx, dy, dz = shape["dimensions_m"]
            bed_z = ground[0] * x + ground[1] * y + ground[2]
            bottom = bed_z + rail_height
            center0 = np.array([x, y, bottom])
            center_world = center0 @ poses[0][:3, :3].T + poses[0][:3, 3]
            rows, supports, frame_evidence = [], [], []
            detector = Detector(detector_config)
            for j, scan in enumerate(scans):
                dt = scan.timestamp_s - scans[0].timestamp_s
                center = center_world + np.asarray(case["motion"]["velocity_world_mps"]) * dt
                low = center + np.array([0, -dy / 2, 0])
                high = low + np.array([dx, dy, dz])
                cloud, times, removed, selected, inserted, visibility = insert_box(
                    scan, poses[j], (low, high), recipe["angular_cell_deg"])
                row = detector.process(cloud, scan.timestamp_s, times)
                support = inserted
                match = matched_object(row, support, recipe["association_tolerance_m"])
                baseline_match = matched_object(baseline_rows[j], support, recipe["association_tolerance_m"])
                matched = match is not None and baseline_match is None
                hazard = bool(match and match["confirmed"] and match["path_relation"] == "intersecting")
                hazard = hazard and matched
                artifact = evidence / f"case-{number:03d}-frame-{j:02d}.npz"
                np.savez_compressed(artifact, removed_source_indices=removed,
                                    inserted_from_source_indices=selected, inserted_points=inserted,
                                    amodal_world_min=low, amodal_world_max=high)
                row.update(label="HYBRID SYNTHETIC", split=split["name"], case=number,
                           source_frame=scan.index, source_measurement_timestamp_ns=scan.measurement_timestamp_ns,
                           insertion=visibility, provenance_npz=str(artifact.relative_to(output)),
                           amodal_world_bbox=[low.tolist(), high.tolist()],
                           observed_support_bbox=([support.min(axis=0).tolist(), support.max(axis=0).tolist()]
                                                  if len(support) else None),
                           matched_object=match, baseline_spatial_match=baseline_match,
                           object_detected=matched, confirmed_hazard=hazard)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                supports.append(support)
                frame_evidence.append(visibility)
            first_visible = next((j for j, v in enumerate(frame_evidence) if v["visible_inserted_returns"]), None)
            first_detect = next((j for j, r in enumerate(rows) if r["object_detected"]), None)
            first_hazard = next((j for j, r in enumerate(rows) if r["confirmed_hazard"]), None)
            records.append({"label": "HYBRID SYNTHETIC", "split": split["name"], "case": number,
                            "shape": shape["name"], "dimensions_m": shape["dimensions_m"],
                            "range_m": x, "position": case["position"]["name"], "lateral_m": y,
                            "motion": case["motion"]["name"], "visibility_mode": recipe["occlusion"]["mode"],
                            "visible_returns_per_frame": [v["visible_inserted_returns"] for v in frame_evidence],
                            "intersecting_rays_per_frame": [v["intersecting_measured_return_rays"] for v in frame_evidence],
                            "foreground_occluded_rays_per_frame": [v["foreground_occluded_rays"] for v in frame_evidence],
                            "removed_returns_per_frame": [v["removed_background_returns"] for v in frame_evidence],
                            "first_visible_frame": first_visible, "first_detection_frame": first_detect,
                            "first_hazard_frame": first_hazard,
                            "detection_latency_s": (scans[first_detect].timestamp_s - scans[first_visible].timestamp_s
                                                    if first_visible is not None and first_detect is not None else None),
                            "hazard_latency_s": (scans[first_hazard].timestamp_s - scans[first_visible].timestamp_s
                                                 if first_visible is not None and first_hazard is not None else None),
                            "object_detected": first_detect is not None, "hazard_confirmed": first_hazard is not None,
                            "statuses": [r["status"] for r in rows],
                            "matched_path_relations": [(r["matched_object"] or {}).get("path_relation") for r in rows],
                            "matched_distances_m": [(r["matched_object"] or {}).get("distance_m") for r in rows],
                            "baseline_statuses": [r["status"] for r in baseline_rows],
                            "background_alarm_frames": sum(r["status"] in ("obstacle", "unresolved_obstacle") for r in baseline_rows),
                            "unmatched_alarm_frames": sum(r["status"] in ("obstacle", "unresolved_obstacle")
                                                          and not r["confirmed_hazard"] for r in rows)})
            print(json.dumps({k: records[-1][k] for k in ("split", "case", "range_m", "position", "shape", "visible_returns_per_frame", "object_detected", "hazard_confirmed")}), flush=True)
    return identity, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new immutable run directory; leaves the frozen matrix unchanged")
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    if args.output is not None:
        recipe["output"] = str(args.output)
    config_path = Path(recipe["detector_config"])
    detector_config = load_config(config_path)
    if recipe["label"] != "HYBRID SYNTHETIC" or recipe["seed"] == detector_config["seed"]:
        raise ValueError("Explicit hybrid label and independent experiment seed required")
    if recipe["max_range_policy"] != "reject_beyond_detector_input":
        raise ValueError("Unsupported range policy")
    for split in recipe["splits"]:
        if split["seed"] == recipe["seed"] or any(x["seed"] == split["seed"] for x in recipe["splits"] if x is not split):
            raise ValueError("Split seeds must be distinct")
        if max(split["ranges_m"]) >= detector_config["max_range_m"]:
            raise ValueError("Range exceeds detector input ceiling")
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", recipe)
    shutil.copyfile(config_path, output / "detector.json")
    source = output / "source"
    source.mkdir()
    for name in ("realistic_stress.py", "realistic_report.py", "stress.py", "io.py", "detector.py"):
        shutil.copyfile(Path(__file__).parent / name, source / name)
    manifest = {"label": recipe["label"], "command": sys.argv, "started_unix_s": time.time(),
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain=v1"], text=True),
                "recipe_sha256": digest(args.experiment), "detector_sha256": digest(config_path),
                "environment": environment(), "sources": []}
    write_json(output / "manifest.json", manifest)
    all_records = []
    for split in recipe["splits"]:
        identity, records = run_split(split, recipe, detector_config, output)
        manifest["sources"].append(identity)
        write_json(output / "manifest.json", manifest)
        all_records.extend(records)
    write_json(output / "cases.json", all_records)
    manifest["finished_unix_s"] = time.time()
    write_json(output / "manifest.json", manifest)
    from .realistic_report import make_report
    make_report(output)


if __name__ == "__main__":
    main()
