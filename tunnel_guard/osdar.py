"""Evaluate OSDaR23 original cuboids with a fixed coordinate adapter."""
from __future__ import annotations

import argparse
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from .detector import Detector, load_config
from .evaluate import evaluate_frames
from .run import digest, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(plan["detector_config"])
    cfg.update(seed=plan["seed"], rail_gauge_m=plan["rail_gauge_m"], sensor_rotation=np.eye(3).tolist(),
               sensor_translation=[0., 0., -plan["virtual_sensor_height_m"]])
    write_json(out / "experiment.json", plan)
    write_json(out / "detector.json", cfg)
    write_json(out / "manifest.json", environment() | {"archive_sha256": digest(Path(plan["archive"]))})
    source_dir = out / "source" / "tunnel_guard"
    source_dir.mkdir(parents=True)
    for source in Path(__file__).parent.glob("*.py"):
        (source_dir / source.name).write_bytes(source.read_bytes())
    detector = Detector(cfg)
    predictions, frames, visibility = {}, [], []
    offset = np.asarray(cfg["sensor_translation"])
    with zipfile.ZipFile(plan["archive"]) as archive, (out / "predictions.jsonl").open("x") as stream:
        label_bytes = archive.read(plan["sequence"] + "_labels.json")
        (out / "original-labels.json").write_bytes(label_bytes)
        (out / "sensor-data-license.txt").write_bytes(archive.read("license.md"))
        labels = json.loads(label_bytes)["openlabel"]
        for frame_id in sorted(labels["frames"], key=int):
            original = labels["frames"][frame_id]
            props = original["frame_properties"]
            member = props["streams"]["lidar"]["uri"].lstrip("/")
            raw = archive.read(member)
            marker = raw.find(b"DATA ascii")
            if marker < 0:
                raise ValueError("This adapter requires the documented ASCII PCD sequence")
            start = raw.index(b"\n", marker) + 1
            values = np.loadtxt(io.BytesIO(raw[start:]))
            points = values[:, :3] + offset
            # The fused sensors have distinct acquisition origins; skip fake deskew.
            row = detector.process(points, float(props["timestamp"]))
            row.update(bag=plan["sequence"], frame=int(frame_id))
            predictions[(row["bag"], row["frame"])] = row
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            truth = []
            for identity, obj in original["objects"].items():
                category = labels["objects"][identity]["type"]
                if category not in plan["classes"]:
                    continue
                for box in obj.get("object_data", {}).get("cuboid", []):
                    if box["coordinate_system"] != "lidar":
                        raise ValueError("Unrecognized cuboid coordinate system")
                    val = np.asarray(box["val"], dtype=float)
                    center, quaternion, extent = val[:3], val[3:7], val[7:10]
                    if not (plan["evaluation_forward_range_m"][0] <= center[0] <= plan["evaluation_forward_range_m"][1]
                            and abs(center[1]) <= plan["evaluation_half_width_m"]):
                        continue
                    rotation = Rotation.from_quat(quaternion).as_matrix()
                    corners = np.asarray(list(itertools.product([-0.5, .5], repeat=3))) * extent
                    corners = corners @ rotation.T + center + offset
                    local = (points - center - offset) @ rotation
                    visible = np.count_nonzero(np.all(np.abs(local) <= extent / 2, axis=1))
                    truth.append({"event_id": identity, "category": category,
                                  "bbox_min": corners.min(axis=0).tolist(), "bbox_max": corners.max(axis=0).tolist()})
                    visibility.append({"frame": int(frame_id), "event_id": identity, "category": category,
                                       "range_m": float(center[0]), "points_inside_original_cuboid": int(visible)})
            frames.append({"bag": row["bag"], "frame": row["frame"], "exhaustive": False, "objects": truth})
    panel = {"label_status": "human_verified", "prediction_scope": "near_track_objects",
             "minimum_iou": plan["minimum_iou"], "frames": frames,
             "provenance": "Original FusionSystems/DZSF/DB OSDaR23 annotations, CC0-1.0; cuboids converted to enclosing AABB, no manual relabeling."}
    write_json(out / "annotations.json", panel)
    write_json(out / "visibility.json", visibility)
    score = evaluate_frames(predictions, panel)
    score["validity_warnings"].append("Independent outdoor railway subset, not metro safety validation; class subset is nonexhaustive for class-agnostic detection")
    write_json(out / "metrics.json", score)
    print(json.dumps({k: v for k, v in score.items() if k not in ["frames", "events"]}, indent=2))


if __name__ == "__main__":
    main()
