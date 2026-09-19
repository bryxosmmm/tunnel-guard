"""Labelled curved-tunnel panel: does the corridor follow the track, measured with exact labels?

The shipped synthetic panel has straight rails, so it cannot show whether the reference contour follows
a curve - the very thing the corridor's curvature continuation exists for. This scene is analytic and
exactly labelled: the tunnel centre-line is a circular arc of radius R in the horizontal plane, the
walls are two coaxial cylinders, floor and ceiling are planes, the rails are thin cylinders at gauge
offset and head height, and an object is a world-frame box placed on the arc at a given arc distance.

Scoring uses the project's own evaluator on exact labels, so recall on a curve at 60-200 m can be
compared between the shipped continuation and the previous straight one (path_curve_window_m: 0),
on identical scenes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import Detector, load_config
from .evaluate import evaluate_frames
from .run import write_json
from .stress import beam_directions, ray_box


def _cylinder_hits(directions: np.ndarray, origin: np.ndarray, centre_xy: np.ndarray, radius: float):
    """Smallest positive range where each ray meets a vertical cylinder, else inf."""
    ox, oy = origin[0] - centre_xy[0], origin[1] - centre_xy[1]
    dx, dy = directions[:, 0], directions[:, 1]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4 * a * c
    hit = disc > 0
    root = np.sqrt(np.where(hit, disc, 0.0))
    t1 = np.where(a > 0, (-b - root) / (2 * np.where(a > 0, a, 1.0)), np.inf)
    t2 = np.where(a > 0, (-b + root) / (2 * np.where(a > 0, a, 1.0)), np.inf)
    t = np.where(t1 > 1e-6, t1, t2)
    return np.where(hit & (t > 1e-6), t, np.inf)


def _plane_hits(directions: np.ndarray, origin: np.ndarray, height: float):
    dz = directions[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (height - origin[2]) / dz
    return np.where((np.abs(dz) > 1e-12) & (t > 1e-6), t, np.inf)


def scene_scan_curved(config: dict, directions: np.ndarray, origin: np.ndarray, obj: dict | None):
    """Nearest hit of every ray against the curved tunnel, its rails and the object."""
    R = float(config["alignment_arc_m"])                 # tunnel centre-line radius
    half_width = float(config["tunnel_half_width_m"])
    ground, ceiling = float(config["ground_z_m"]), float(config["tunnel_ceiling_z_m"])
    gauge, head = float(config["rail_gauge_m"]), float(config["ground_z_m"]) + float(config["rail_height_m"])
    centre = np.array([0.0, R])                          # arc centre in the horizontal plane
    span = np.deg2rad(float(config["arc_span_deg"]))

    distance = np.full(len(directions), np.inf)
    for radius in (R - half_width, R + half_width):      # walls
        distance = np.minimum(distance, _cylinder_hits(directions, origin, centre, radius))
    for height in (ground, ceiling):                     # floor, ceiling
        distance = np.minimum(distance, _plane_hits(directions, origin, height))
    for offset in (-gauge / 2, gauge / 2):               # rails: thin cylinders at head height
        rail = _cylinder_hits(directions, origin, centre, R + offset)
        z_at = origin[2] + rail * directions[:, 2]
        near_head = np.abs(z_at - head) <= 0.20
        distance = np.minimum(distance, np.where(near_head, rail, np.inf))

    # Keep only the angular span the arc covers. The sensor sits at angle -pi/2 about the arc centre
    # (y_sensor - y_centre = -R) and the alignment runs towards increasing angle, so the valid window
    # is [-pi/2, -pi/2 + span]; using +pi/2 here masked every surface out and left only the object.
    angle = np.arctan2((origin[1] + directions[:, 1] * distance) - centre[1],
                       (origin[0] + directions[:, 0] * distance) - centre[0])
    inside = (angle >= -np.pi / 2 - 1e-9) & (angle <= -np.pi / 2 + span)
    distance = np.where(inside, distance, np.inf)

    background = distance
    target = np.full(len(directions), np.inf)
    if obj is not None:
        target = ray_box(directions, origin, np.asarray(obj["bbox_min"]), np.asarray(obj["bbox_max"]))
        distance = np.minimum(distance, target)
    # The object is only *visible* where it is the nearest hit; a ray that would reach it through a wall
    # returns the wall. Marking those as object support put wall points into the labels, which is why an
    # earlier panel scored zero detection on scenes where the object was plainly present.
    visible = target < background
    keep = np.isfinite(distance)
    return directions[keep] * distance[keep][:, None], visible[keep]


def coverage_support(nearby: list[dict], lo: np.ndarray, hi: np.ndarray, grid: int = 8):
    """Fraction of the label box sampled on a grid that falls inside any nearby predicted box.

    The detector's boxes are observed support, and on this scene an object's support can split across
    clusters, so a union measure answers what one-to-one IoU cannot: is the object's volume covered by
    what was reported nearby at all.
    """
    axes = [np.linspace(lo[i], hi[i], grid) for i in range(3)]
    sample = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if not nearby:
        return len(sample), 0, 0.0
    boxes = [(np.asarray(o["bbox_min"]), np.asarray(o["bbox_max"])) for o in nearby]
    inside = np.zeros(len(sample), dtype=bool)
    for low, high in boxes:
        inside |= np.all((sample >= low) & (sample <= high), axis=1)
    return len(sample), int(inside.sum()), float(inside.mean())


def object_on_arc(config: dict, arc_m: float, lateral_m: float, dimensions) -> dict:
    """Axis-aligned envelope of a box standing on the arc at `arc_m` of arc length."""
    R = float(config["alignment_arc_m"])
    ground = float(config["ground_z_m"]) + float(config["rail_height_m"])
    phi = arc_m / R
    # point at arc length arc_m along the centre-line, offset laterally by lateral_m
    x = R * np.sin(phi)
    y = R - R * np.cos(phi)
    yaw = phi
    half_x, half_y = (lateral_m + dimensions[0] / 2), dimensions[1] / 2
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx = x + sx * half_x * np.cos(yaw) - sy * half_y * np.sin(yaw)
            cy = y + sx * half_x * np.sin(yaw) + sy * half_y * np.cos(yaw)
            corners.append((cx, cy))
    corners = np.asarray(corners)
    return {"bbox_min": [corners[:, 0].min(), corners[:, 1].min(), ground],
            "bbox_max": [corners[:, 0].max(), corners[:, 1].max(), ground + dimensions[2]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    detector_config = load_config(experiment["detector_config"])
    output = Path(experiment["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", experiment)
    write_json(output / "detector.json", detector_config)

    rng = np.random.default_rng(experiment["seed"])
    rays = beam_directions(experiment, rng)
    predictions, panel = {}, []
    for index, arc_m in enumerate(experiment["arc_distances_m"]):
        for lateral in experiment["lateral_m"]:
            box = object_on_arc(experiment, arc_m, lateral, experiment["object_dimensions_m"])
            detector = Detector(detector_config)
            for frame in range(experiment["frames"]):
                origin = np.zeros(3)   # stationary sensor: geometry, not motion, is under test
                cloud, visible = scene_scan_curved(experiment, rays, origin, box)
                row = detector.process(cloud, frame * 0.1, None)
                bag = f"arc{index:02d}_lat{lateral:+.1f}"
                row.update(frame=frame, bag=bag)
                predictions[(bag, frame)] = row
                if visible.any():
                    support = cloud[visible]
                    lo, hi = support.min(axis=0), support.max(axis=0)
                else:
                    lo, hi = np.asarray(box["bbox_min"]), np.asarray(box["bbox_max"])
                panel.append({"bag": bag, "frame": frame, "exhaustive": True,
                              "objects": [{"event_id": "on_arc", "bbox_min": lo.tolist(),
                                           "bbox_max": (np.maximum(hi, lo + 1e-3)).tolist()}]})
    annotations = {"label_status": "synthetic_exact", "prediction_scope": "collision_hazards",
                   "minimum_iou": experiment["minimum_iou"], "frames": panel}
    write_json(output / "annotations.json", annotations)
    score = evaluate_frames(predictions, annotations)

    # The evaluator matches one prediction to one label at IoU >= minimum_iou. On this scene an object's
    # visible support can split into two clusters (its near face and its top), each below that IoU against
    # a label spanning both, so the one-to-one score under-reports detection. This complementary measure
    # asks the question the corridor exists to answer: how much of the labelled support is covered by the
    # union of predictions that lie within the object's neighbourhood - support covered, per case.
    coverage = []
    for key, row in predictions.items():
        label = next((f for f in panel if (f["bag"], f["frame"]) == key), None)
        if label is None or not label["objects"]:
            continue
        lo, hi = np.asarray(label["objects"][0]["bbox_min"]), np.asarray(label["objects"][0]["bbox_max"])
        centre, radius = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)), 1.0)
        nearby = [o for o in row["objects"]
                  if np.linalg.norm(np.asarray(o["center"]) - centre) <= 2 * radius]
        support = coverage_support(nearby, lo, hi)
        coverage.append({"bag": key[0], "frame": key[1], "label_points": int(support[0]),
                         "covered_points": int(support[1]), "coverage": float(support[2])})
    covered = [c["coverage"] for c in coverage]
    score["union_coverage"] = {"median": float(np.median(covered)) if covered else None,
                               "cases": len(covered),
                               "cases_at_least_half": int(sum(1 for c in covered if c >= 0.5)),
                               "note": "fraction of the object's visible support covered by the union of "
                                       "nearby predictions; one-to-one IoU is reported separately"}
    score["arc_m"] = experiment["alignment_arc_m"]
    print(json.dumps({k: v for k, v in score.items() if k not in ("frames", "events")}, indent=2))
    write_json(output / "metrics.json", score)


if __name__ == "__main__":
    main()
