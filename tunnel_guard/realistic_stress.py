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
from .geometry import TrackGeometry, voxel_representatives
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
                    "invalid": int((~valid).sum()), "rotation": rotation}
    raise IndexError(f"{bag} has no frame {index}")


def sensor_pattern(frame: dict) -> dict:
    """The frame reduced to its sampling pattern: one entry per measured shot, plus the shots that
    were emitted but returned nothing.

    A recording contains only the slots that returned. The sensor emitted far more: on these scans
    a third to two thirds of the emission grid is empty, and an empty slot means there was no
    surface in it, not that the sensor cannot see there. Painting an inserted object only onto
    returned slots therefore hides the object's body wherever the empty tunnel returned nothing -
    mid-height at range, which is exactly where a standing object is. The emitted-but-empty slots
    are reconstructed on the sensor's own lattice so an object can occupy them, bounded by the
    longest range the recording demonstrably returned from in the same frame.
    """
    ranges = np.linalg.norm(frame["points"], axis=1)
    pattern = {"points": frame["points"], "ranges": ranges, "directions": frame["points"] / ranges[:, None],
               "intensity": frame["intensity"], "time_s": frame["time_s"],
               "empty_directions": np.empty((0, 3)), "empty_time_s": np.empty(0),
               "slots_observed": int(len(ranges))}
    ring = frame.get("ring")
    if ring is None or not len(ranges):
        return pattern
    # Elevation ladder in the frame the detector works in: the measured median elevation of each
    # ring, which is what the driver's fixed table produces, measured rather than assumed.
    elevation = np.arctan2(frame["points"][:, 2], np.hypot(frame["points"][:, 0], frame["points"][:, 1]))
    azimuth = np.arctan2(frame["points"][:, 1], frame["points"][:, 0])
    ladder, occupied = {}, {}
    for value in np.unique(ring):
        member = ring == value
        ladder[int(value)] = float(np.median(elevation[member]))
        occupied[int(value)] = azimuth[member]
    # Each ring samples azimuth on its own comb and the rings' combs are interleaved, so the step has
    # to be measured: the modal gap between consecutive returns is one emission step, gaps of twice
    # or three times that are emissions that returned nothing, and a handful of zero gaps are
    # duplicate azimuths. A minimum would be degenerate on those duplicates and a median of a
    # sparsely returning ring's gaps would collapse the lattice onto the observations.
    pooled = np.concatenate([np.diff(np.sort(occupied[value])) for value in ladder]) if ladder else np.empty(0)
    pooled = pooled[pooled > 1e-6]
    if not len(pooled):
        return pattern
    binned = np.round(pooled / 1e-5).astype(np.int64)
    values, counts = np.unique(binned, return_counts=True)
    step = float(values[int(counts.argmax())]) * 1e-5
    if not np.isfinite(step) or step <= 0:
        return pattern
    directions, times = [], []
    median_time = float(np.median(frame["time_s"])) if len(frame["time_s"]) else 0.0
    for value, height in ladder.items():
        observed = np.sort(occupied[value])
        if len(observed) < 2:
            continue
        count = int(np.floor((observed[-1] - observed[0]) / step)) + 1
        if count < 2 or count > 65536:
            continue
        grid = observed[0] + step * np.arange(count)
        index = np.clip(np.round((observed - observed[0]) / step).astype(np.int64), 0, count - 1)
        seen = np.zeros(count, dtype=bool)
        seen[index] = True
        missing = grid[~seen]
        if not len(missing):
            continue
        cos_h = np.cos(height)
        directions.append(np.column_stack((cos_h * np.cos(missing), cos_h * np.sin(missing),
                                           np.full(len(missing), np.sin(height)))))
        times.append(np.full(len(missing), median_time))
    if directions:
        # The ladder and the azimuth grid were measured in the frame the detector works in, so the
        # directions are reconstructed directly in that frame; no second rotation is applied.
        pattern["empty_directions"] = np.vstack(directions)
        pattern["empty_time_s"] = np.concatenate(times)
        pattern["slots_emitted_estimated"] = int(len(pattern["empty_directions"]) + len(ranges))
    return pattern


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
    """Merge an object into a recorded frame, with the occlusion it would cause.

    Two kinds of slot can carry the object. A slot the recording returned from is replaced when the
    object stands in front of the recorded surface. A slot the sensor emitted but returned nothing
    from becomes a slot with a surface in it, which is what an object standing in an empty tunnel
    actually produces - but only where the sensor demonstrably can return, meaning at a range no
    longer than the longest this frame returned from, and at an intensity no higher than the
    recording's own upper decile. That second rule is off unless the recipe asks for it, and a panel
    that leaves it off is the pessimistic bound: it measures how much of the object the empty
    tunnel's own returns happen to expose, not how much of it the sensor could see.
    """
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
        if empty_slots_return and len(pattern["empty_directions"]):
            low, high = np.asarray(target_world["bbox_min"]), np.asarray(target_world["bbox_max"])
            centre, radius = 0.5 * (low + high), 0.5 * float(np.linalg.norm(high - low))
            empty_world = pattern["empty_directions"] @ rotation.T
            to_centre = centre - origin
            # A ray can only meet the object if the box centre lies within its bounding sphere of
            # the ray: one vectorised test replaces an exact slab test over every emitted slot,
            # which was the whole cost of this rule.
            projection = empty_world @ to_centre
            perpendicular = np.linalg.norm(to_centre - projection[:, None] * empty_world, axis=1)
            candidate = np.flatnonzero((projection > 0) & (perpendicular <= radius))
            empty_along = np.full(len(empty_world), np.inf)
            if len(candidate):
                empty_along[candidate] = ray_box(empty_world[candidate], origin, low, high)
            longest = float(np.max(pattern["ranges"])) if len(pattern["ranges"]) else np.inf
            reachable = np.isfinite(empty_along) & (empty_along <= longest)
            if reachable.any():
                hits += int(reachable.sum())
                reached = empty_along[reachable] + rng.normal(0.0, jitter_m, int(reachable.sum()))
                points = np.vstack((points, pattern["empty_directions"][reachable] * reached[:, None]))
                intensity = np.concatenate((intensity, np.full(int(reachable.sum()), object_intensity)))
                times_extra = pattern["empty_time_s"][reachable]
                pattern = dict(pattern, time_s=np.concatenate((pattern["time_s"], times_extra)))
                labelled = np.concatenate((labelled, np.ones(int(reachable.sum()), dtype=bool)))
    label = None
    if labelled.any():
        support = points[labelled]
        low, high = support.min(axis=0), support.max(axis=0)
        label = {"bbox_min": low.tolist(), "bbox_max": np.maximum(high, low + 1e-3).tolist()}
    return {"points": points, "intensity": intensity, "time_s": pattern["time_s"], "labelled": labelled,
            "rays_hitting_object": hits, "returns_from_object": int(labelled.sum()), "label": label,
            # The object's own measured returns. The bounding box of these spans volume the sensor
            # never observed, so a box-only score cannot tell "reported, boxed differently" from
            # "not reported"; the points can.
            "labelled_points": points[labelled]}


def contour_width_over_height(low: float, high: float, config: dict) -> float:
    """Largest configured reference-contour half-width touched by a box height interval.

    This is only geometry of the selected reference contour, not a vehicle swept
    envelope or a field collision label. Its extrema are at the endpoints of
    the overlapping piecewise-linear segments.
    """
    widths = []
    for bottom, top, start, end in np.asarray(config["envelope_segments_m"], dtype=float):
        left, right = max(low, bottom), min(high, top)
        if left > right:
            continue
        for height in (left, right):
            widths.append(float(start + (end - start) * (height - bottom) / (top - bottom)))
    return max(widths) if widths else 0.0


def contour_intersects(lateral_m: float, dimensions_m: tuple[float, float, float] | list[float],
                       reference_config: dict, rest_height_m: float) -> bool:
    """Whether the injected full box intersects the selected reference contour.

    The previous panel used half the rail gauge as a proxy. That is narrower
    than the configured contour and labels true edge cases as adjacent.
    """
    _, lateral_size, vertical_size = map(float, dimensions_m)
    low = float(rest_height_m)
    width = contour_width_over_height(low, low + vertical_size, reference_config)
    nearest_lateral = max(0.0, abs(float(lateral_m)) - lateral_size / 2.0)
    return bool(nearest_lateral <= width)


def contour_laterals(dimensions_m: tuple[float, float, float] | list[float], scenario_config: dict,
                     reference_config: dict, rest_height_m: float):
    """Resolve frozen inside/edge/adjacent modes against the configured contour.

    Existing recipes retain their literal ``lateral_m`` values. New recipes may
    use modes so changing dimensions cannot silently make an edge panel empty.
    """
    modes = scenario_config.get("contour_lateral_modes")
    if modes is None:
        for lateral in scenario_config["lateral_m"]:
            yield float(lateral), "explicit_lateral"
        return
    low = float(rest_height_m)
    width = contour_width_over_height(low, low + float(dimensions_m[2]), reference_config)
    half = float(dimensions_m[1]) / 2.0
    overlap = float(scenario_config.get("contour_edge_overlap_m", 0.05))
    clearance = float(scenario_config.get("contour_adjacent_clearance_m", 0.05))
    for mode in modes:
        if mode == "inside":
            magnitude = 0.0
        elif mode == "edge":
            magnitude = width + half - overlap
        elif mode == "adjacent":
            magnitude = width + half + clearance
        else:
            raise ValueError(f"Unknown contour lateral mode: {mode}")
        for side in scenario_config.get("contour_lateral_sides", [1.0]):
            if float(side) not in (-1.0, 1.0):
                raise ValueError("contour_lateral_sides must contain only -1 or 1")
            yield float(side) * magnitude, mode


def cases(config: dict, reference_config: dict | None = None):
    """Every case records object presence and contour intersection separately."""
    yield {"range_m": None, "lateral_m": None, "dimensions_m": None,
           "object_present": False, "reference_contour_intersects": False,
           "lateral_mode": "background"}
    reference_config = config if reference_config is None else reference_config
    heights = config.get("object_rest_heights_m", [config["object_rest_height_m"]])
    for distance, dims, rest_height in itertools.product(config["ranges_m"], config["object_dimensions_m"], heights):
        for lateral, mode in contour_laterals(dims, config, reference_config, float(rest_height)):
            yield {"range_m": distance, "lateral_m": lateral, "dimensions_m": dims,
                   "rest_height_m": float(rest_height), "object_present": True,
                   "reference_contour_intersects": contour_intersects(
                       lateral, dims, reference_config, float(rest_height)),
                   "lateral_mode": mode}


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


def world_target(case: dict, pose: np.ndarray, geometry, height_m: float) -> dict | None:
    """Place the object on the recorded track, at the requested distance along it.

    A straight line from the sensor leaves the tunnel on any curve: at 100 m it is
    already inside the lining, and an object placed there is occluded by the wall
    rather than being seen down the track. The fitted rail chain is where the track
    actually goes, so the object is placed on that centreline and offset from it.

    Both the centre-line and the floor come from the detector's own geometry for the same frame, so a
    result cannot be a disagreement between the injector and the classifier about where the track is:
    an interpolated anchor chain clamps beyond its last anchor while the classifier extrapolates the
    fitted slope and curvature, and at 150 m those two centres differ by metres, which put a
    centre-line object 2.85 m off the corridor and out of the envelope. Where either is unsupported at
    that station the case is left unsupported instead of being scored.
    """
    if case["range_m"] is None:
        return None
    dx, dy, dz = case["dimensions_m"]
    along_m, lateral_m = case["range_m"], case["lateral_m"]
    centre, _, _ = geometry.path(np.array([along_m]))
    target_lateral = float(centre[0]) + lateral_m
    bed, _ = geometry.ground(np.array([[along_m, target_lateral, 0.0]]))
    if not np.isfinite(bed[0]) or not np.isfinite(centre[0]):
        return None
    forward = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    lateral = pose[:3, :3] @ np.array([0.0, 1.0, 0.0])
    up = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    centre_world = pose[:3, 3] + pose[:3, :3] @ np.array([along_m, target_lateral,
                                                          bed[0] + height_m + dz / 2])
    return {"bbox_min": (centre_world - forward * dx / 2 - lateral * dy / 2 - up * dz / 2).tolist(),
            "bbox_max": (centre_world + forward * dx / 2 + lateral * dy / 2 + up * dz / 2).tolist()}


def run_case(case: dict, stress: dict, detector_config: dict, index: int, sources: list[dict]):
    rng = np.random.default_rng(np.random.SeedSequence([stress["seed"], index]))
    detector = Detector(detector_config)
    target = world_target(case, sources[0]["pose"], sources[0]["geometry"],
                          case.get("rest_height_m", stress["object_rest_height_m"]))  # placed once, on the track
    if target is None and case.get("range_m") is not None:
        # No floor was observed at that station, so the object cannot be placed on the track there.
        # Reporting it as an unsupported case keeps a placement failure out of the miss count.
        return [], [], [], [], False
    rows, labels, statistics, inserted = [], [], [], []
    for position, source in enumerate(sources):
        pose, stamp = source["pose"], source["stamp_s"]
        merged = insert(source["pattern"], target, pose, rng, stress["object_range_noise_m"],
                        stress["object_intensity"], bool(stress.get("empty_slots_return_object", False)))
        times = merged["time_s"]
        normalised = ((times - times.min()) / np.ptp(times)) if len(times) and np.ptp(times) > 0 else np.empty(0)
        row = detector.process(merged["points"], stamp, normalised)
        row.update(frame=position, bag=f"case_{index:03d}")
        rows.append(row)
        # Truth at the level the sensor actually resolved: the object's own returns. Recorded at the
        # precision the detector itself saw: rounding them to 0.1 mm while the detector's box is the
        # exact bbox of those same returns left 4 of 7 returns testing as outside their own box by
        # <=5e-5 m, which scored a box identical to the label as a miss.
        recorded = (np.asarray(merged["labelled_points"], dtype=float)
                    if len(merged["labelled_points"]) else None)
        if recorded is not None:
            inserted.append({"case": index, "frame": position, "points": recorded.tolist()})
        truth = []
        observed_support_intersects = False
        if recorded is not None:
            # Ask the same current-frame classifier about the support actually
            # created by the insertion. A full form can touch the contour while
            # all its measured returns stay outside it; that is observability,
            # not a detector miss.
            support_masks, _ = source["geometry"].classify_with_section(
                recorded, remove_background=False, include_boundary=True)
            observed_support_intersects = bool(np.any(support_masks[4]))
        if case["reference_contour_intersects"] and observed_support_intersects:
            low = recorded.min(axis=0)
            high = np.maximum(recorded.max(axis=0), low + 1e-3)
            truth = [{"event_id": "inserted_object", "bbox_min": low.tolist(),
                      "bbox_max": high.tolist()}]
        labels.append({"bag": row["bag"], "frame": position, "exhaustive": False, "objects": truth})
        statistics.append({"case": index, "frame": position, "range_m": case["range_m"],
                           "lateral_m": case["lateral_m"], "dimensions_m": case["dimensions_m"],
                           "rest_height_m": case.get("rest_height_m"),
                           "lateral_mode": case["lateral_mode"],
                           "object_present": case["object_present"],
                           "full_shape_reference_contour_intersects": case["reference_contour_intersects"],
                           "observed_support_reference_contour_intersects": observed_support_intersects,
                           "rays_hitting_object": merged["rays_hitting_object"],
                           "returns_from_object": merged["returns_from_object"],
                           "detections": len(row["objects"]), "status": row["status"],
                           "nearest_obstacle_m": row["nearest_obstacle_m"]})
    return rows, labels, statistics, inserted, True


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
        # The same reduction the detector applies, so the geometry the object is placed on is the
        # geometry the detector will use for that frame.
        cloud = frame["points"]
        crop = ((cloud[:, 0] >= detector_config["min_forward_m"])
                & (np.abs(cloud[:, 1]) < detector_config["context_half_width_m"]))
        reduced = voxel_representatives(cloud[crop], detector_config["geometry_voxel_m"])
        geometry = TrackGeometry(reduced, detector_config)
        sources.append({"frame_index": frame_index, "pattern": sensor_pattern(frame),
                        "pose": poses[frame_index][0], "stamp_s": frame["stamp_s"],
                        "anchors": poses[frame_index][2], "geometry": geometry})
        profile.append({"frame": frame_index, "returns": len(frame["points"]),
                        "duplicates_dropped": frame["duplicates"], "invalid": frame["invalid"],
                        "slots": frame["slots"]})
    write_json(output / "sensor_profile.json",
               {"background_bag": str(bag), "recorded_frames": profile,
                "object_range_noise_m": stress["object_range_noise_m"],
                "object_rest_height_m": stress["object_rest_height_m"],
                "object_rest_heights_m": stress.get("object_rest_heights_m", [stress["object_rest_height_m"]]),
                "empty_slots_return_object": bool(stress.get("empty_slots_return_object", False)),
                "longest_demonstrated_range_m": float(max(np.max(s["pattern"]["ranges"]) for s in sources)),
                "object_intensity": stress["object_intensity"]})
    predictions, annotations, records, inserted_rows = {}, [], [], []
    unsupported_cases = []
    started = time.perf_counter()
    with (output / "predictions.jsonl").open("x") as stream:
        for index, case in enumerate(cases(stress, detector_config)):
            rows, labels, statistics, inserted, supported = run_case(case, stress, detector_config,
                                                                     index, sources)
            if not supported:
                unsupported_cases.append({"case": index, "range_m": case["range_m"],
                                          "lateral_m": case["lateral_m"]})
                continue
            for row in rows:
                predictions[(row["bag"], row["frame"])] = row
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            annotations.extend(labels)
            records.extend(statistics)
            inserted_rows.extend(inserted)
    write_json(output / "unsupported_cases.json",
               {"count": len(unsupported_cases), "cases": unsupported_cases,
                "reason": ("No floor was observed at the station the object needed to stand on, so the "
                           "object was not placed and the case is not scored. Scored objects are "
                           "reported as objects.injected by realistic_report.")})
    with (output / "inserted.jsonl").open("x") as stream:
        for record in inserted_rows:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    panel = {"label_status": "synthetic_exact", "prediction_scope": "collision_hazards",
             "minimum_iou": stress["minimum_iou"], "box_semantics": "observed_support", "frames": annotations}
    score = evaluate_frames(predictions, panel)
    score.update(cases=sum(1 for _ in cases(stress, detector_config)), frames_per_case=len(sources), background=str(bag),
                 elapsed_s=time.perf_counter() - started)
    write_json(output / "cases.json", records)
    write_json(output / "annotations.json", panel)
    write_json(output / "metrics.json", score)
    print(json.dumps({k: v for k, v in score.items() if k not in ("frames", "events")}, indent=2))


if __name__ == "__main__":
    main()
