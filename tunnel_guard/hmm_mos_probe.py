"""Probe the published HMM-MOS implementation on this project's tunnel data.

The method is Bhandari, James, Phillips, McAree, "Moving Object Segmentation in
Point Cloud Data using Hidden Markov Models" (IJRR 45(8), 2026; arXiv:2410.18638).
The code under test is the authors' own implementation, built unmodified from
github.com/vb44/HMM-MOS; this file only generates inputs and reads the labels it writes.

Two modes, each with an explicit recipe:

  synth  our measured-pattern tunnel scene (``tunnel_guard.stress.scene_scan``) with one
         inserted box; the sensor steps 1 m/frame and the box is static, moving, or appears.
  real   real scans from a bag plus the poses a recorded detector run already produced,
         converted to KITTI ``.bin`` + ``poses.txt``.

Nothing here changes detector thresholds and no score is claimed: the outputs are dynamic
point counts, which the review document interprets.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .stress import beam_directions, scene_scan

ROOT = Path(__file__).resolve().parent.parent
DYNAMIC_LABEL = 251


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def kitti_pose(origin_x: float) -> list[float]:
    """Row-major 3x4 sensor-to-map transform: the sensor only translates along the track axis."""
    return [1, 0, 0, origin_x, 0, 1, 0, 0.0, 0, 0, 1, 0.0]


def object_at(case: dict, frame: int, probe: dict, scene: dict):
    """Inserted-box world footprint for this frame, or None when the case has no box yet."""
    if case["kind"] == "none":
        return None
    if case["kind"] == "appearing" and frame < case["appear_frame"]:
        return None
    dx, dy, dz = probe["object_dimensions_m"]
    near_x = case["start_m"]
    if case["kind"] == "moving":
        stop = case.get("stop_frame")
        steps = frame if stop is None else min(frame, stop)
        near_x += case["velocity_mps"] * steps * probe["frame_period_s"]
    lateral = case.get("lateral_m", 0.0)
    low = np.array([near_x, lateral - dy / 2, scene["ground_z_m"] + scene["rail_height_m"]])
    return {"bbox_min": low.tolist(), "bbox_max": (low + [dx, dy, dz]).tolist()}, near_x, lateral


def region_masks(points: np.ndarray, scene: dict) -> dict:
    """Classify dynamic points by authored scene region, to name the nuisance source."""
    ground, ceiling = scene["ground_z_m"], scene["tunnel_ceiling_z_m"]
    width = scene["tunnel_half_width_m"]
    half_gauge, rail_top = scene["rail_gauge_m"] / 2, ground + scene["rail_height_m"]
    on_rail = (np.abs(np.abs(points[:, 1]) - half_gauge) < 0.15) & (points[:, 2] <= rail_top + 0.05)
    floor = points[:, 2] <= ground + 0.25
    wall = np.abs(points[:, 1]) >= width - 0.3
    top = points[:, 2] >= ceiling - 0.3
    rest = ~(on_rail | floor | wall | top)
    return {"rail": on_rail, "floor": floor & ~on_rail, "wall": wall, "ceiling": top, "interior": rest}


def hmm_config(path: Path, bins: Path, poses: Path, labels: Path, max_range: float, hmm: dict) -> None:
    path.write_text(f"""---
scanPath: {bins}/
posePath: {poses}
minRange: {hmm['minRange']}
maxRange: {max_range}
outputFile: false
scanNumsToPrint: []
outputFileName: /dev/null
outputLabels: true
outputLabelFolder: {labels}/

voxelSize: {hmm['voxelSize']}
occupancySigma: {hmm['occupancySigma']}
beliefThreshold: {hmm['beliefThreshold']}
convSize: {hmm['convSize']}
localWindowSize: {hmm['localWindowSize']}
globalWindowSize: {hmm['globalWindowSize']}
minOtsu: {hmm['minOtsu']}
""")


def read_labels(labels: Path, count: int) -> np.ndarray:
    """Per-point dynamic mask from the authors' SemanticKITTI-style .label output."""
    masks = []
    for index in range(count):
        raw = np.frombuffer((labels / f"{index:06d}.label").read_bytes(), dtype=np.uint32)
        masks.append((raw & 0xFFFF) == DYNAMIC_LABEL)
    return masks


def run_hmm_mos(binary: Path, config: Path, workdir: Path) -> dict:
    """Run the authors' binary once; /usr/bin/time -l supplies the child peak RSS on macOS."""
    timer = Path("/usr/bin/time")
    command = [str(timer), "-l", str(binary), str(config)] if timer.exists() else [str(binary), str(config)]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"HMM-MOS failed ({result.returncode}): {result.stdout[-2000:]} {result.stderr[-2000:]}")
    peak = None
    for line in result.stderr.splitlines():
        if "maximum resident set size" in line:
            peak = int(line.split()[0])
    return {"seconds": elapsed, "peak_rss_bytes": peak, "stderr_tail": result.stderr[-500:]}


def synth(args) -> dict:
    probe = json.loads(Path(args.experiment).read_text())
    scene = json.loads(Path(probe["scene_config"]).read_text())
    output = Path(probe["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", probe)
    frames = args.limit_frames or probe["frames"]
    cases = []
    for index, case in enumerate(probe["cases"]):
        step = case.get("sensor_step_m", probe["sensor_step_m"])
        case_dir = output / case["id"]
        case_frames = case.get("frames", frames)
        bins, labels = case_dir / "bins", case_dir / "labels"
        bins.mkdir(parents=True)
        labels.mkdir()
        rng = np.random.default_rng(np.random.SeedSequence([probe["seed"], index]))
        case_scene = dict(scene) | case.get("scene_overrides", {})
        rays = beam_directions(case_scene, rng)
        order, visible_masks = [], []
        poses = []
        with (case_dir / "poses.txt").open("w") as stream:
            for frame in range(case_frames):
                origin_x = step * frame
                inserted = object_at(case, frame, probe, scene)
                obj = inserted[0] if inserted else None
                # "matched" reuses one noise/dropout draw per frame across the whole case, so a
                # moving-sensor case isolates pose-driven change. "independent" draws fresh noise,
                # which is what real scans do even when the platform is stationary.
                frame_rng = rng
                if probe.get("independent_frame_noise"):
                    frame_rng = np.random.default_rng(np.random.SeedSequence([probe["seed"], index, 7, frame]))
                # scene_scan already returns the cloud in the sensor-centred frame with world axes
                # (origin + direction * range, with the origin subtracted), so the pose below is
                # exactly what maps these points back into the scene frame.
                cloud, visible = scene_scan(case_scene, rays, origin_x, obj, frame_rng)
                points = cloud.astype(np.float32)
                intensity = np.ones((len(points), 1), dtype=np.float32)
                np.column_stack((points, intensity)).tofile(bins / f"{frame:06d}.bin")
                # The written pose is the estimate the method sees; the points were generated
                # with the true pose, so a non-zero sigma is an ego-motion error, not noise on input.
                sigma = case.get("pose_noise_sigma_m", 0.0)
                estimate = origin_x + (0.0 if not sigma else
                                       float(np.random.default_rng(np.random.SeedSequence(
                                           [probe["seed"], index, 11, frame])).normal(0, sigma)))
                stream.write(" ".join(str(v) for v in kitti_pose(estimate)) + "\n")
                order.append(inserted)
                visible_masks.append(visible)
                poses.append(origin_x)
        config = case_dir / "hmm-mos.yaml"
        max_range = case.get("maxRange", probe["hmm_mos_config"]["maxRange"])
        hmm_config(config, bins, case_dir / "poses.txt", labels, max_range, probe["hmm_mos_config"])
        config = config.resolve()
        timing = run_hmm_mos(Path(args.binary).resolve(), config, case_dir)
        masks = read_labels(labels, case_frames)
        records = []
        regions = {}
        for frame in range(case_frames):
            inserted = order[frame]
            mask = masks[frame]
            visible = visible_masks[frame]
            points = np.fromfile(bins / f"{frame:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
            dynamic = points[mask]
            if len(dynamic):
                np.save(case_dir / f"dynamic_{frame:06d}.npy", dynamic)
                for name, selected in region_masks(dynamic, case_scene).items():
                    regions[name] = regions.get(name, 0) + int(selected.sum())
            records.append({
                "frame": frame,
                "sensor_x_m": poses[frame],
                "object_frame_distance_m": None if not inserted else inserted[1] - poses[frame],
                "visible_object_points": int(visible.sum()),
                "dynamic_points": int(mask.sum()),
                "dynamic_object_points": int((mask & visible).sum()),
                "dynamic_min_radius_m": None if not len(dynamic) else
                    float(np.linalg.norm(dynamic, axis=1).min()),
                "dynamic_max_radius_m": None if not len(dynamic) else
                    float(np.linalg.norm(dynamic, axis=1).max()),
            })
        cases.append({"id": case["id"], "case": case, "seconds": timing["seconds"],
                      "peak_rss_bytes": timing["peak_rss_bytes"], "max_range_m": max_range,
                      "points_per_frame": len(masks[0]), "dynamic_regions": regions, "frames": records})
        print(json.dumps({"case": case["id"], "seconds": round(timing["seconds"], 1),
                          "dynamic_total": int(sum(r["dynamic_points"] for r in records)),
                          "dynamic_object_total": int(sum(r["dynamic_object_points"] for r in records)),
                          "visible_object_total": int(sum(r["visible_object_points"] for r in records))}),
              flush=True)
        shutil.rmtree(bins)
    summary = {"mode": "synth", "source": probe["source"], "scene_config": probe["scene_config"],
               "hmm_mos_config": probe["hmm_mos_config"], "frames": frames,
               "measured_beam_pattern": {k: scene[k] for k in ("azimuth_bins", "ground_z_m",
                                                               "tunnel_half_width_m", "tunnel_ceiling_z_m",
                                                               "rail_gauge_m", "rail_height_m")},
               "cases": cases}
    write_json(output / "summary.json", summary)
    return summary


def real(args) -> dict:
    from .detector import load_config
    from .io import iter_bag

    probe = json.loads(Path(args.experiment).read_text())
    rows = [json.loads(line) for line in Path(probe["recorded_run"]).read_text().splitlines() if line]
    first, last = probe["first_frame"], probe["last_frame"]
    rows = [r for r in rows if first <= r["frame"] <= last]
    if not rows:
        raise ValueError("no recorded rows in the requested frame range")
    config = load_config(probe["detector_config"])
    output = Path(probe["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", probe)
    bins, labels = output / "bins", output / "labels"
    bins.mkdir()
    labels.mkdir()
    wanted = {r["frame"]: r for r in rows}
    decoded, poses = [], []
    with (output / "poses.txt").open("w") as stream:
        for scan in iter_bag(Path(probe["bag"]), config, max_frames=last + 1):
            if scan.index not in wanted:
                continue
            np.column_stack((scan.points.astype(np.float32),
                             np.ones((len(scan.points), 1), dtype=np.float32))).tofile(
                bins / f"{len(decoded):06d}.bin")
            pose = np.asarray(wanted[scan.index]["pose"], dtype=float).reshape(4, 4)
            stream.write(" ".join(str(v) for v in pose[:3].reshape(-1)) + "\n")
            decoded.append({"frame": scan.index, "points": len(scan.points),
                            "recorded_points": wanted[scan.index]["input_valid_points"],
                            "objects": wanted[scan.index]["objects"],
                            "recorded_status": wanted[scan.index]["status"]})
            poses.append(pose)
    config_path = output / "hmm-mos.yaml"
    hmm_config(config_path, bins, output / "poses.txt", labels, probe["hmm_mos_config"]["maxRange"],
               probe["hmm_mos_config"])
    timing = run_hmm_mos(Path(args.binary).resolve(), config_path.resolve(), output)
    masks = read_labels(labels, len(decoded))
    reference = {}
    if probe.get("annotation"):
        panel = json.loads(Path(probe["annotation"]).read_text())
        reference = {f["frame"]: f["objects"] for f in panel["frames"] if f["bag"] == Path(probe["bag"]).name}

    # Second pass: attribute the dynamic labels to points, since the labels are per point index.
    frame_records = []
    dynamic_by_range = {edge: 0 for edge in probe["range_edges_m"]}
    for index, scan in enumerate(iter_bag(Path(probe["bag"]), config, max_frames=last + 1)):
        if scan.index not in wanted:
            continue
        item = decoded[len(frame_records)]
        mask = masks[len(frame_records)]
        if len(mask) != len(scan.points):
            raise ValueError(f"label count {len(mask)} != point count {len(scan.points)} at frame {scan.index}")
        dynamic = scan.points[mask]
        radius = np.linalg.norm(dynamic, axis=1) if len(dynamic) else np.zeros(0)
        histogram = {}
        for edge in sorted(dynamic_by_range):
            histogram[str(edge)] = int((radius <= edge).sum())
            dynamic_by_range[edge] += int((radius <= edge).sum())
        inside, matched = 0, []
        for obj in reference.get(scan.index, []):
            low, high = np.asarray(obj["bbox_min"]), np.asarray(obj["bbox_max"])
            if len(dynamic):
                inside += int(np.all((dynamic >= low) & (dynamic <= high), axis=1).sum())
        # Independent check against the recorded detector's own candidate boxes: shared evidence
        # is not ground truth, but it says whether the two methods point at the same measurements.
        in_candidate = 0
        for obj in item["objects"]:
            low, high = np.asarray(obj["bbox_min"]), np.asarray(obj["bbox_max"])
            if len(dynamic):
                hits = int(np.all((dynamic >= low) & (dynamic <= high), axis=1).sum())
                if hits:
                    in_candidate += hits
                    matched.append({"track_id": obj.get("track_id"), "distance_m": obj.get("distance_m"),
                                    "dynamic_points": hits})
        if len(dynamic):
            np.save(output / f"dynamic_{len(frame_records):06d}.npy", dynamic.astype(np.float32))
        frame_records.append({
            "frame": scan.index,
            "points": len(scan.points),
            "recorded_points": item["recorded_points"],
            "dynamic_points": int(mask.sum()),
            "dynamic_within_reference_box": inside,
            "reference_boxes": len(reference.get(scan.index, [])),
            "dynamic_in_detector_candidate_boxes": in_candidate,
            "matching_detector_objects": matched,
            "dynamic_min_radius_m": None if not len(dynamic) else float(radius.min()),
            "dynamic_max_radius_m": None if not len(dynamic) else float(radius.max()),
            "recorded_status": item["recorded_status"],
            "recorded_objects": len(item["objects"]),
        })
    summary = {"mode": "real", "bag": probe["bag"], "recorded_run": probe["recorded_run"],
               "annotation": probe.get("annotation"),
               "hmm_mos_config": probe["hmm_mos_config"], "point_count_match":
                   all(r["points"] == r["recorded_points"] for r in frame_records),
               "seconds": timing["seconds"], "peak_rss_bytes": timing["peak_rss_bytes"],
               "frames": len(frame_records),
               "dynamic_points_total": int(sum(r["dynamic_points"] for r in frame_records)),
               "dynamic_within_reference_box_total": int(
                   sum(r["dynamic_within_reference_box"] for r in frame_records)),
               "frames_with_any_dynamic": int(sum(r["dynamic_points"] > 0 for r in frame_records)),
               "dynamic_cumulative_by_range_m": dynamic_by_range,
               "frame_records": frame_records}
    write_json(output / "summary.json", summary)
    shutil.rmtree(bins)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("synth", "real"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--binary", default="/tmp/hmm-mos-probe/HMM-MOS/build/hmmMOS")
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args()
    summary = synth(args) if args.mode == "synth" else real(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, default=str)[:2000])


if __name__ == "__main__":
    main()
