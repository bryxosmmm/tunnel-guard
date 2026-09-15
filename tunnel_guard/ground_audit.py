"""Evaluate released ground implementations across height/tilt, retaining raw trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .detector import load_config
from .geometry import TrackGeometry, robust_plane, voxel_representatives
from .io import iter_bag
from .run import digest, environment, write_json
from .segmentation import level_rotation
from .stress import beam_directions, scene_scan


def masks(points, plane, plan):
    import pypatchworkpp as pw
    import travel_seg as travel
    rotation = level_rotation(plane)
    aligned = points @ rotation.T
    height = abs(plane[2]) / np.sqrt(1 + np.sum(plane[:2]**2))
    for name in plan["methods"]:
        start = time.perf_counter()
        domain = np.ones(len(points), dtype=bool)
        if name == "local_geometry":
            residual = (points[:, 2] - points[:, :2] @ plane[:2] - plane[2]) / np.sqrt(1 + np.sum(plane[:2]**2))
            ground = np.abs(residual) <= .065
        else:
            source = aligned if "leveled" in name else points
            if name.endswith("corridor"):
                domain = np.abs(source[:, 1]) <= 3.
            data = source[domain].astype(np.float32)
            ground = np.zeros(len(points), dtype=bool)
            if name.startswith("patchwork"):
                params = pw.Parameters()
                for key, value in plan["patchwork"].items():
                    setattr(params, key, value)
                params.sensor_height = height
                segmenter = pw.patchworkpp(params)
                segmenter.estimateGround(np.column_stack((data, np.zeros(len(data), dtype=np.float32))))
                indices = np.asarray(segmenter.getGroundIndices()).astype(int).ravel()
                ground[np.flatnonzero(domain)[indices]] = True
            else:
                segmenter = travel.TravelGroundSeg(travel.GroundSegConfig(**plan["travel"]))
                result, _ = segmenter.estimate_ground(data)
                ground[domain] = result
        yield name, ground, domain, time.perf_counter() - start


def metrics(predicted, truth, domain):
    tp = np.count_nonzero(predicted & truth & domain)
    fp = np.count_nonzero(predicted & ~truth & domain)
    fn = np.count_nonzero(~predicted & truth & domain)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(plan["detector_config"])
    scene = json.loads(Path(plan["scene_config"]).read_text())
    scene.update(random_drop_probability=0., range_noise_sigma_m=0.)
    write_json(out / "experiment.json", plan)
    write_json(out / "manifest.json", environment() | {"detector_sha256": digest(Path(plan["detector_config"]))})
    source_dir = out / "source" / "tunnel_guard"
    source_dir.mkdir(parents=True)
    for source in Path(__file__).parent.glob("*.py"):
        (source_dir / source.name).write_bytes(source.read_bytes())
    rng = np.random.default_rng(plan["seed"])
    rays = beam_directions(scene, rng)
    synthetic, real = [], []
    for height in plan["sensor_heights_m"]:
        scene["ground_z_m"] = -height
        clean, _ = scene_scan(scene, rays, 0., None, rng)
        truth_all = np.isclose(clean[:, 2], -height, atol=1e-7)
        for roll, pitch in plan["tilt_roll_pitch_deg"]:
            distance = np.linalg.norm(clean, axis=1)
            keep = (distance >= 1) & (distance <= plan["max_range_m"]) & (rng.random(len(clean)) >= plan["random_drop_probability"])
            cloud = clean[keep] * (1 + rng.normal(0, plan["range_noise_sigma_m"], keep.sum()) / distance[keep])[:, None]
            cloud = cloud @ Rotation.from_euler("xy", [roll, pitch], degrees=True).as_matrix().T
            truth = truth_all[keep]
            plane, quality = robust_plane(voxel_representatives(cloud, cfg["geometry_voxel_m"]), cfg)
            if plane is None:
                synthetic.append({"sensor_height_m": height, "roll_deg": roll, "pitch_deg": pitch, "geometry_failure": quality})
                continue
            expected = clean[keep] @ Rotation.from_euler("xy", [roll, pitch], degrees=True).as_matrix().T
            bed = expected[truth]
            reference_error = np.abs(bed[:, 2] - bed[:, :2] @ plane[:2] - plane[2]) / np.sqrt(1 + np.sum(plane[:2]**2))
            near = (bed[:, 0] >= 2) & (bed[:, 0] <= 25) & (np.abs(bed[:, 1]) < 2)
            common_domain = np.abs((cloud @ level_rotation(plane).T)[:, 1]) <= 3.
            for method, mask, domain, elapsed in masks(cloud, plane, plan):
                row = {"method": method, "sensor_height_m": height, "roll_deg": roll, "pitch_deg": pitch,
                       "runtime_s": elapsed, "local_plane_height_mae_m": float(reference_error[near].mean()),
                       "full_range_plane_height_mae_m": float(reference_error.mean()),
                       "method_input_domain_metrics": metrics(mask, truth, domain),
                       "score_domain": "identical_leveled_plus_minus_3m_corridor"} | metrics(mask, truth, common_domain)
                synthetic.append(row)
                write_json(out / "synthetic.json", synthetic)
    for bag in plan["real_bags"]:
        scan = next(iter_bag(Path(plan["bag_root"]) / bag, cfg))
        points = scan.points
        plane, quality = robust_plane(voxel_representatives(points, cfg["geometry_voxel_m"]), cfg)
        if plane is None:
            real.append({"bag": bag, "geometry_failure": quality})
            continue
        residual = np.abs(points[:, 2] - points[:, :2] @ plane[:2] - plane[2]) / np.sqrt(1 + np.sum(plane[:2]**2))
        near = (points[:, 0] >= 2) & (points[:, 0] <= 25) & (np.abs(points[:, 1]) < 2)
        for method, mask, domain, elapsed in masks(points, plane, plan):
            selected = mask & near & domain
            real.append({"bag": bag, "frame": scan.index, "method": method, "runtime_s": elapsed,
                         "ground_points": int(mask.sum()), "near_corridor_ground_points": int(selected.sum()),
                         "near_reference_residual_median_m": float(np.median(residual[selected])) if selected.any() else None,
                         "near_reference_residual_p95_m": float(np.quantile(residual[selected], .95)) if selected.any() else None,
                         "label_status": "unlabeled_reference_diagnostic_not_accuracy"})
        write_json(out / "real.json", real)
    summary = []
    for method in plan["methods"]:
        rows = [r for r in synthetic if r.get("method") == method]
        total = {key: sum(r[key] for r in rows) for key in ["tp", "fp", "fn"]}
        denominator = 2 * total["tp"] + total["fp"] + total["fn"]
        summary.append({"method": method, "synthetic_scenes": len(rows), "ground_f1": 2 * total["tp"] / denominator if denominator else None,
                        "worst_scene_f1": min(r["f1"] for r in rows) if rows else None,
                        "runtime_median_s": float(np.median([r["runtime_s"] for r in rows])) if rows else None})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
