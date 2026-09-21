"""Synthetic scenes built from a real scan instead of invented by hand.

`stress` ray-casts an analytic tunnel with an invented sampling pattern, uniform
dropout and uniform noise. Each of those choices decides the answer: the angular
sampling decides whether an object is resolved at range at all, the scene decides
what occludes it, and the return behaviour decides how much evidence survives.
This module takes all three from the recording:

* rays - the measured directions of a real frame, so the sampling is the
  hardware's, including its non-uniform elevation ladder and dead channels;
* background - the measured range of every shot of that frame, so the tunnel's
  real cross-section, curvature and occlusion are present;
* returns - an inserted object occupies shots that the recording returned from,
  and, when the recipe enables ``empty_slots_return_object``, also slots the
  sensor left empty. An empty slot means there was no surface there, or none
  inside the demonstrated range; putting an object there replaces both causes, so
  the object is taken to return where the sensor demonstrably can - within the
  longest range it returned from in that frame, and at an intensity no higher than
  the recording's own upper decile. That assumption is stated, not hidden, and a
  panel that leaves it off is the pessimistic bound;
* intensity and acquisition time - carried from the same frame, so an object's
  returns can be given a measured-plausible brightness and the scan can be
  deskewed or not.

Frames come from the recording, so the ego motion, the motion distortion and the
frame-to-frame scene change are the real ones; the object is placed once in the
world frame and the recorded poses carry the sensor towards it.

What this is still not: the background is one recorded pass, not a surveyed
scene, so a frame's remaining content is unknown. Panels built here therefore
score recall and report unmatched hazards separately, and they do not claim
precision.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .detector import Detector, load_config
from .evaluate import evaluate_frames
from .run import digest, environment, write_json

POINT_FIELDS = ("x", "y", "z", "intensity", "ring", "timestamp")
FIELD_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def read_frame(bag: Path, index: int, config: dict) -> dict:
    """One recorded frame, deduplicated, with the fields the pipeline discards."""
    store = get_typestore(Stores.ROS2_HUMBLE)
    rotation = np.asarray(config["sensor_rotation"], dtype=float)
    translation = np.asarray(config["sensor_translation"], dtype=float)
    with Reader(bag) as reader:
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
        if len(connections) != 1:
            raise ValueError(f"Expected one PointCloud2 topic in {bag}, found {len(connections)}")
        for position, (connection, _, raw) in enumerate(reader.messages(connections=connections)):
            if position != index:
                continue
            message = store.deserialize_cdr(raw, connection.msgtype)
            fields = {f.name: f for f in message.fields}
            names = [name for name in POINT_FIELDS if name in fields]
            endian = ">" if message.is_bigendian else "<"
            dtype = np.dtype({"names": names, "formats": [endian + FIELD_TYPES[fields[n].datatype] for n in names],
                              "offsets": [fields[n].offset for n in names], "itemsize": message.point_step})
            records = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                                 strides=(message.row_step, message.point_step))
            points = np.column_stack([records[n].ravel() for n in ("x", "y", "z")]).astype(np.float64)
            valid = np.isfinite(points).all(axis=1) & (np.einsum("ij,ij->i", points, points) > 1e-6)
            ring = records["ring"].ravel()[valid] if "ring" in names else np.zeros(int(valid.sum()), dtype=np.uint16)
            intensity = (records["intensity"].ravel()[valid] if "intensity" in names
                         else np.zeros(int(valid.sum())))
            time_s = records["timestamp"].ravel()[valid] if "timestamp" in names else np.zeros(int(valid.sum()))
            # The recordings carry each return twice with identical channel,
            # position, intensity and acquisition time; keep one copy so every
            # point- and neighbourhood-based statistic means what it says.
            keys = np.column_stack([ring, points[valid]])
            _, first = np.unique(keys, axis=0, return_index=True)
            keep = np.sort(first)
            duplicates = int(valid.sum()) - len(keep)
            stamp = message.header.stamp
            return {"index": index, "topic": connection.topic, "frame_id": message.header.frame_id,
                    "slots": message.height * message.width, "stamp_s": stamp.sec + stamp.nanosec * 1e-9,
                    "points": points[valid][keep] @ rotation.T + translation,
                    "ring": ring[keep], "intensity": np.asarray(intensity, dtype=float)[keep],
                    "time_s": np.asarray(time_s, dtype=float)[keep], "duplicates": duplicates,
                    "invalid": int((~valid).sum())}
    raise IndexError(f"{bag} has no frame {index}")


def sensor_pattern(frame: dict) -> dict:
    """The frame reduced to its sampling pattern: one entry per measured shot."""
    ranges = np.linalg.norm(frame["points"], axis=1)
    return {"points": frame["points"], "ranges": ranges, "directions": frame["points"] / ranges[:, None],
            "intensity": frame["intensity"], "time_s": frame["time_s"]}


def bed_height(pattern: dict, pose: np.ndarray, along_m: float, lateral_m: float) -> float:
    """Local bed height where the object is to stand, from the recording itself.

    A band average over tens of metres is not the ground under the object: the
    track has grade and the bed is not flat, and an object placed a few
    centimetres below the local surface is occluded by it and never measured.
    The lower returns inside the object's own footprint are the local ground.
    """
    points = pattern["points"]
    forward = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    lateral = pose[:3, :3] @ np.array([0.0, 1.0, 0.0])
    up = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    local = points - pose[:3, 3]
    u, v, w = local @ forward, local @ lateral, local @ up
    for window, spread in ((1.5, 1.2), (4.0, 2.0), (10.0, 3.0)):
        near = (np.abs(u - along_m) < window) & (np.abs(v - lateral_m) < spread)
        if int(near.sum()) >= 8:
            return float(np.percentile(w[near], 10))
    return float(np.percentile(w, 10))


def ray_box(directions: np.ndarray, origin: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """First entry parameter of each ray into an axis-aligned box; inf when missed."""
    parallel = np.abs(directions) < 1e-12
    safe = np.where(parallel, 1.0, directions)
    low = (minimum - origin) / safe
    high = (maximum - origin) / safe
    entries, exits = np.minimum(low, high), np.maximum(low, high)
    outside = parallel & ((origin < minimum) | (origin > maximum))
    entries[parallel] = -np.inf
    exits[parallel] = np.inf
    near, far = entries.max(axis=1), exits.min(axis=1)
    usable = (far >= np.maximum(near, 0)) & ~outside.any(axis=1)
    distance = np.where(near > 0, near, far)
    return np.where(usable & (distance > 0), distance, np.inf)


def insert(pattern: dict, target_world: dict | None, pose: np.ndarray, rng: np.random.Generator,
           jitter_m: float, object_intensity: float, empty_slots_return: bool = False) -> dict:
    """Merge an object into a recorded frame, with the occlusion it would cause."""
    points, intensity = pattern["points"].copy(), pattern["intensity"].copy()
    labelled = np.zeros(len(points), dtype=bool)
    along = np.full(len(points), np.inf)  # no object: no ray reaches one
    hits = 0
    if target_world is not None:
        rotation, origin = pose[:3, :3], pose[:3, 3]
        along = ray_box(pattern["directions"] @ rotation.T, origin, np.asarray(target_world["bbox_min"]),
                        np.asarray(target_world["bbox_max"]))
        # Only measured shots can carry the object: an empty slot stays empty.
        occluded = np.isfinite(along) & (along < pattern["ranges"])
        hits = int(np.isfinite(along).sum())
        if occluded.any():
            # The recordings' own surfaces carry their own spread; an inserted
            # return gets the measured slot plus this stated jitter. A calibrated
            # noise model is not available: the sensor profile is unverified.
            reached = along[occluded] + rng.normal(0.0, jitter_m, int(occluded.sum()))
            points[occluded] = pattern["directions"][occluded] * reached[:, None]
            intensity[occluded] = object_intensity
            labelled[occluded] = True
    if empty_slots_return:
        # A slot the recording left empty becomes a slot with a surface in it. The
        # sensor demonstrably returns from those ranges in this frame, and the
        # object is assumed no brighter than the recording's own upper decile.
        longest = float(np.max(pattern["ranges"]))
        reachable = ~np.isfinite(pattern["ranges"]) & np.isfinite(along) & (along <= longest) & ~labelled
        if reachable.any():
            reached = along[reachable] + rng.normal(0.0, jitter_m, int(reachable.sum()))
            points[reachable] = pattern["directions"][reachable] * reached[:, None]
            intensity[reachable] = object_intensity
            labelled[reachable] = True
    label = None
    if labelled.any():
        support = points[labelled]
        low, high = support.min(axis=0), support.max(axis=0)
        label = {"bbox_min": low.tolist(), "bbox_max": np.maximum(high, low + 1e-3).tolist()}
    return {"points": points, "intensity": intensity, "time_s": pattern["time_s"], "labelled": labelled,
            "rays_hitting_object": hits, "returns_from_object": int(labelled.sum()), "label": label}


def cases(config: dict):
    """Every case places one object on the corridor, or none for a scene-negative."""
    yield {"range_m": None, "lateral_m": None, "dimensions_m": None}
    for distance, lateral, dims in itertools.product(config["ranges_m"], config["lateral_m"],
                                                     config["object_dimensions_m"]):
        yield {"range_m": distance, "lateral_m": lateral, "dimensions_m": dims,
               "hazard": abs(lateral) < config["rail_gauge_m"] / 2}


def load_poses(run: Path, bag: str) -> dict:
    """Recorded sensor poses and the fitted track centreline for the same frames."""
    poses = {}
    for path in sorted(run.glob("*.jsonl")):
        if path.stem != bag:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            anchors = np.asarray(row.get("geometry", {}).get("rail_anchors") or [], dtype=float)
            poses[row["frame"]] = (np.asarray(row["pose"], dtype=float), row["timestamp_s"], anchors)
    return poses


def world_target(case: dict, pose: np.ndarray, pattern: dict, height_m: float,
                 anchors: np.ndarray) -> dict | None:
    """Place the object on the recorded track, at the requested distance along it.

    A straight line from the sensor leaves the tunnel on any curve: at 100 m it is
    already inside the lining, and an object placed there is occluded by the wall
    rather than being seen down the track. The fitted rail chain is where the track
    actually goes, so the object is placed on that centreline and offset from it.
    """
    if case["range_m"] is None:
        return None
    dx, dy, dz = case["dimensions_m"]
    along_m, lateral_m = case["range_m"], case["lateral_m"]
    centre_lateral = float(np.interp(along_m, anchors[:, 0], anchors[:, 1])) if len(anchors) >= 2 else 0.0
    forward = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    lateral = pose[:3, :3] @ np.array([0.0, 1.0, 0.0])
    up = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    bed = bed_height(pattern, pose, along_m, centre_lateral + lateral_m)
    sensor_point = np.array([along_m, centre_lateral + lateral_m, 0.0])
    centre = pose[:3, 3] + pose[:3, :3] @ sensor_point
    centre = centre + up * (pose[:3, 3] @ up + bed + height_m + dz / 2 - float(centre @ up))
    return {"bbox_min": (centre - forward * dx / 2 - lateral * dy / 2 - up * dz / 2).tolist(),
            "bbox_max": (centre + forward * dx / 2 + lateral * dy / 2 + up * dz / 2).tolist()}


def run_case(case: dict, stress: dict, detector_config: dict, index: int, sources: list[dict]):
    rng = np.random.default_rng(np.random.SeedSequence([stress["seed"], index]))
    detector = Detector(detector_config)
    target = world_target(case, sources[0]["pose"], sources[0]["pattern"],
                          stress["object_rest_height_m"], sources[0]["anchors"])  # placed once, on the track
    rows, labels, statistics = [], [], []
    for position, source in enumerate(sources):
        pose, stamp = source["pose"], source["stamp_s"]
        merged = insert(source["pattern"], target, pose, rng, stress["object_range_noise_m"],
                        stress["object_intensity"], bool(stress.get("empty_slots_return_object", False)))
        times = merged["time_s"]
        normalised = ((times - times.min()) / np.ptp(times)) if len(times) and np.ptp(times) > 0 else np.empty(0)
        row = detector.process(merged["points"], stamp, normalised)
        row.update(frame=position, bag=f"case_{index:03d}")
        rows.append(row)
        truth = []
        if case.get("hazard", False):
            if merged["label"] is not None:
                truth = [{"event_id": "inserted_object", "bbox_min": merged["label"]["bbox_min"],
                          "bbox_max": merged["label"]["bbox_max"]}]
            else:
                # Present in the world but not measured in this frame: a miss.
                rotation = pose[:3, :3]
                middle = (np.asarray(target["bbox_min"]) + np.asarray(target["bbox_max"])) / 2 - pose[:3, 3]
                centre = rotation.T @ middle
                half = np.abs(rotation.T @ (np.asarray(target["bbox_max"]) - np.asarray(target["bbox_min"]))) / 2
                truth = [{"event_id": "inserted_object", "bbox_min": (centre - half).tolist(),
                          "bbox_max": (centre + half).tolist()}]
        labels.append({"bag": row["bag"], "frame": position, "exhaustive": False, "objects": truth})
        statistics.append({"case": index, "frame": position, "range_m": case["range_m"],
                           "lateral_m": case["lateral_m"], "dimensions_m": case["dimensions_m"],
                           "rays_hitting_object": merged["rays_hitting_object"],
                           "returns_from_object": merged["returns_from_object"],
                           "detections": len(row["objects"]), "status": row["status"],
                           "nearest_obstacle_m": row["nearest_obstacle_m"]})
    return rows, labels, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
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
    write_json(output / "manifest.json", environment() | {"config_sha256": digest(Path(stress["detector_config"]))})
    bag = Path(stress["background_bag"])
    poses = load_poses(Path(stress["pose_run"]), bag.name)
    if not poses:
        raise ValueError(f"No recorded poses for {bag.name} in {stress['pose_run']}")
    frames = sorted(poses)
    start = int(stress.get("frame_start", 0))
    stride = int(stress.get("frame_stride", 1))
    chosen = frames[start::stride][:stress["source_frames"]]
    if len(chosen) < stress["source_frames"]:
        raise ValueError(f"{bag.name} has {len(frames)} frames; asked for {stress['source_frames']} from {start}")
    sources, profile = [], []
    for frame_index in chosen:
        frame = read_frame(bag, frame_index, detector_config)
        sources.append({"frame_index": frame_index, "pattern": sensor_pattern(frame),
                        "pose": poses[frame_index][0], "stamp_s": frame["stamp_s"],
                        "anchors": poses[frame_index][2]})
        profile.append({"frame": frame_index, "returns": len(frame["points"]),
                        "duplicates_dropped": frame["duplicates"], "invalid": frame["invalid"],
                        "slots": frame["slots"]})
    write_json(output / "sensor_profile.json",
               {"background_bag": str(bag), "recorded_frames": profile,
                "object_range_noise_m": stress["object_range_noise_m"],
                "object_rest_height_m": stress["object_rest_height_m"],
                "empty_slots_return_object": bool(stress.get("empty_slots_return_object", False)),
                "longest_demonstrated_range_m": float(max(np.max(s["pattern"]["ranges"]) for s in sources)),
                "object_intensity": stress["object_intensity"]})
    predictions, annotations, records = {}, [], []
    started = time.perf_counter()
    with (output / "predictions.jsonl").open("x") as stream:
        for index, case in enumerate(cases(stress)):
            rows, labels, statistics = run_case(case, stress, detector_config, index, sources)
            for row in rows:
                predictions[(row["bag"], row["frame"])] = row
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            annotations.extend(labels)
            records.extend(statistics)
    panel = {"label_status": "synthetic_exact", "prediction_scope": "collision_hazards",
             "minimum_iou": stress["minimum_iou"], "box_semantics": "observed_support", "frames": annotations}
    score = evaluate_frames(predictions, panel)
    score.update(cases=sum(1 for _ in cases(stress)), frames_per_case=len(sources), background=str(bag),
                 elapsed_s=time.perf_counter() - started)
    write_json(output / "cases.json", records)
    write_json(output / "annotations.json", panel)
    write_json(output / "metrics.json", score)
    print(json.dumps({k: v for k, v in score.items() if k not in ("frames", "events")}, indent=2))


if __name__ == "__main__":
    main()
