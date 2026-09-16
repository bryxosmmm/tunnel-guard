"""Convert SUSTechPOINTS label files into the annotations the evaluator validates.

The tool stores one JSON file per frame under ``<scene>/label/`` with boxes in
``position / rotation / scale`` form (ZYX Euler radians, metres, box centre).  This
command reads those files back into the ``annotations/*.json`` schema that
``tunnel_guard.evaluate`` checks, so human labels can be scored against a detector run.

Frames the reviewer declares exhaustive become scored frames even when they hold no
object: that is what makes negative episodes measurable, and it is a claim about the
reviewer, not about the detector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import validate_annotations
from .run import digest, write_json


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict):
        raise ValueError("Config must be a JSON object")
    return plan


def read_box_files(label_dir: Path) -> dict[int, list[dict]]:
    frames = {}
    for entry in sorted(label_dir.glob("*.json")):
        if not entry.stem.isdigit():
            continue
        boxed = json.loads(entry.read_text())
        if not isinstance(boxed, list):
            raise ValueError(f"{entry} must contain a list of boxes")
        if int(entry.stem) in frames:
            raise ValueError(f"Multiple label files resolve to frame {int(entry.stem)}")
        frames[int(entry.stem)] = boxed
    return frames


def rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Match the tool's ``euler_angle_to_rotate_matrix``: R = Rx @ Ry @ Rz (order ZYX)."""
    cx, cy, cz = np.cos(rotation)
    sx, sy, sz = np.sin(rotation)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_x @ rot_y @ rot_z


def box_aabb(box: dict) -> tuple[list[float], list[float]]:
    psr = box["psr"]
    center = np.asarray([psr["position"][axis] for axis in "xyz"], dtype=float)
    scale = np.asarray([psr["scale"][axis] for axis in "xyz"], dtype=float)
    rotation = np.asarray([psr["rotation"][axis] for axis in "xyz"], dtype=float)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or not np.isfinite(rotation).all():
        raise ValueError("Box centre, scale and rotation must be finite")
    if np.any(scale <= 0):
        raise ValueError("Box scale must be positive on every axis")
    half = scale / 2.0
    corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = corners @ rotation_matrix(rotation).T + center
    return world.min(axis=0).tolist(), world.max(axis=0).tolist()


def export(plan: dict, config_path: Path) -> dict:
    scene_root = Path(plan["scene_root"])
    output = Path(plan["output"])
    scenes = plan.get("scenes") or {}
    if not scenes:
        raise ValueError("Export config must list at least one scene")
    annotation = {
        "label_status": plan["label_status"],
        "prediction_scope": plan["prediction_scope"],
        "minimum_iou": float(plan["minimum_iou"]),
        "provenance": plan["provenance"],
        "box_semantics": plan.get("box_semantics", "unspecified"),
        "frames": [],
    }
    summary = {}
    for scene, scene_plan in sorted(scenes.items()):
        label_dir = scene_root / scene / str(scene_plan.get("label_directory", "label"))
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory {label_dir}")
        annotated = read_box_files(label_dir)
        sources = {int(p.stem): p for p in label_dir.glob("*.json") if p.stem.isdigit()}
        exhaustive = {int(f) for f in scene_plan.get("exhaustive_frames") or []}
        available = {int(p.stem) for p in (scene_root / scene / "lidar").iterdir() if p.stem.isdigit()}
        if not available:
            raise FileNotFoundError(f"Scene {scene} exposes no lidar frames")
        for kind, frames in (("labelled", sorted(set(annotated))), ("exhaustive", sorted(exhaustive))):
            unknown = sorted(set(frames) - available)
            if unknown:
                raise ValueError(f"{scene}: {kind} frames absent from the recording: {unknown}")
        missing = sorted(exhaustive - set(annotated))
        for frame in sorted(set(annotated) | exhaustive):
            objects = []
            for box in annotated.get(frame, []):
                low, high = box_aabb(box)
                objects.append({
                    "event_id": str(box["obj_id"]),
                    "bbox_min": low,
                    "bbox_max": high,
                    "class": str(box["obj_type"]),
                    "path_intersection": str(plan.get("path_intersection", "unverified")),
                    "annotation_origin": scene_plan.get("annotation_origins", {}).get(
                        str(frame), {}).get(str(box["obj_id"]), "unspecified"),
                    "source_box_psr": box["psr"],
                    "source_label_sha256": digest(sources[frame]),
                })
            annotation["frames"].append({
                "bag": scene, "frame": frame, "exhaustive": frame in exhaustive, "objects": objects,
            })
        summary[scene] = {
            "annotated_frames": len(annotated),
            "exhaustive_frames": len(exhaustive),
            "exhaustive_frames_without_labels": missing,
            "boxes": sum(len(v) for v in annotated.values()),
        }
    if not annotation["frames"]:
        raise ValueError(
            "Nothing to export: no scene has label files or declared exhaustive frames. An empty "
            "annotation file would read as 'no object anywhere'."
        )
    validate_annotations(annotation)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, annotation)
    return {
        "output": str(output),
        "config_sha256": digest(config_path),
        "warning": (
            "Rotated boxes are exported as their axis-aligned envelope, which is what the IoU evaluator "
            "consumes. Exhaustiveness and class are copied from the declaration and the human labels; "
            "neither this command nor the score verifies them."
        ),
        "scenes": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = export(load_plan(args.config), args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
