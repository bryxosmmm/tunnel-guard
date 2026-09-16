"""Find objects with point support inside the reference clearance envelope.

The detector's ``path_relation=intersecting`` comes from cluster bounding boxes, so a wall
cluster whose box reaches into the envelope counts as a hazard. This command works on
returns instead: only points that actually lie inside the GOST contour are clustered, and a
cluster is rejected when it is a face of a large connected surface, or when the same lateral
and vertical profile runs continuously (or periodically) along the tunnel — cable runs,
linings, trays and posts. What survives is a short list of compact intrusions, grouped into
events across frames so one object is reported once rather than once per frame.

Thresholds live in ``configs/on-track-probe.json``. Output is the per-frame clusters plus the
grouped events; a candidate list is evidence for review, not a verified hazard, and on curved
recordings the envelope itself is only as good as the track-centre estimate behind it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag

DEFAULTS = {
    "cluster_eps_m": 0.30,
    "cluster_voxel_m": 0.10,
    "min_inside_voxels": 3,
    "min_height_extent_m": 0.10,
    "max_object_height_m": 3.0,
    "max_component_voxels": 1500,
    "max_compact_length_m": 2.0,
    "profile_tolerance_m": 0.40,
    "max_continuous_coverage_m": 4.0,
    "min_periodic_spacing_m": 5.0,
    "event_max_frame_gap": 3,
    "event_lateral_tolerance_m": 0.8,
    "event_vertical_tolerance_m": 0.8,
}


def components(points: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    tree = cKDTree(points)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    n = len(points)
    if pairs.size == 0:
        return np.arange(n), n
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    count, labels = connected_components(graph, directed=False)
    return labels, count


def voxel_reduce(points: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """One representative per occupied voxel, plus the voxel index of every point."""
    index = np.floor(points / size).astype(np.int64)
    _, inverse = np.unique(index, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centers = np.zeros((len(counts), 3))
    for axis in range(3):
        centers[:, axis] = np.bincount(inverse, weights=points[:, axis]) / counts
    return centers, inverse


def frame_candidates(points: np.ndarray, detector: dict, thresholds: dict) -> list[dict]:
    th = {**DEFAULTS, **thresholds}
    # Only ahead of the train: the near field holds the vehicle's own structure.
    forward = ((points[:, 0] >= detector["min_forward_m"]) & (points[:, 0] <= detector["max_range_m"])
               & (np.abs(points[:, 1]) <= detector["context_half_width_m"]))
    points = points[forward]
    if len(points) < th["min_inside_voxels"]:
        return []
    reduced = voxel_representatives(points, detector["geometry_voxel_m"])
    geometry = TrackGeometry(reduced, detector)
    if not geometry.valid:
        return []
    _, _, height, observed, nominal_overlap = geometry.classify(points, remove_background=False)
    inside = observed & nominal_overlap
    if inside.sum() < th["min_inside_voxels"]:
        return []

    centers, inverse = voxel_reduce(points, th["cluster_voxel_m"])
    n_voxels = len(centers)
    inside_voxels = np.zeros(n_voxels, dtype=bool)
    inside_voxels[np.unique(inverse[inside])] = True
    counts = np.bincount(inverse, minlength=n_voxels)
    heights = np.bincount(inverse, weights=height, minlength=n_voxels) / counts
    inside_centers = centers[inside_voxels]

    labels, count = components(centers, th["cluster_eps_m"])
    out = []
    for label in range(count):
        members = labels == label
        n_inside = int(inside_voxels[members].sum())
        if n_inside < th["min_inside_voxels"]:
            continue
        cluster = centers[members]
        inside_points = cluster[inside_voxels[members]]
        inside_extent = inside_points.max(axis=0) - inside_points.min(axis=0)
        height_extent = float(inside_extent[2])
        if height_extent < th["min_height_extent_m"] or height_extent > th["max_object_height_m"]:
            continue
        # Infrastructure either runs continuously along the tunnel or repeats at intervals.
        # A foreign object covers one short stretch.
        profile_y = float(np.median(inside_points[:, 1]))
        profile_z = float(np.median(inside_points[:, 2]))
        same_profile = ((np.abs(inside_centers[:, 1] - profile_y) < th["profile_tolerance_m"])
                        & (np.abs(inside_centers[:, 2] - profile_z) < th["profile_tolerance_m"]))
        profile_x = np.sort(inside_centers[same_profile, 0])
        coverage = float(profile_x.max() - profile_x.min()) if len(profile_x) else 0.0
        repeats = int((np.diff(profile_x) > th["min_periodic_spacing_m"]).sum())
        out.append({
            "profile_coverage_m": round(coverage, 2),
            "profile_repeat_groups": repeats,
            "continuous_structure": bool(coverage > th["max_continuous_coverage_m"]),
            "periodic_structure": bool(repeats >= 1
                                       and coverage > th["max_continuous_coverage_m"] * 0.5),
            "component_voxels": int(members.sum()),
            "inside_voxels": n_inside,
            "center": inside_points.mean(axis=0).round(3).tolist(),
            "distance_m": round(float(inside_points[:, 0].min()), 2),
            "extent_m": (cluster.max(axis=0) - cluster.min(axis=0)).round(2).tolist(),
            "inside_extent_m": inside_extent.round(2).tolist(),
            "height_extent_m": round(height_extent, 2),
            "kind": "long_structure" if float(inside_extent[0]) > th["max_compact_length_m"] else "compact",
            "median_height_above_bed_m": round(float(np.median(heights[members][inside_voxels[members]])), 2),
            "attached_to_surface": bool(int(members.sum()) > th["max_component_voxels"]),
        })
    return sorted(out, key=lambda c: c["distance_m"])


def is_candidate(cluster: dict, thresholds: dict) -> bool:
    th = {**DEFAULTS, **thresholds}
    return (cluster["inside_voxels"] >= th["min_inside_voxels"]
            and cluster["kind"] == "compact"
            and not cluster["attached_to_surface"]
            and not cluster["continuous_structure"]
            and not cluster["periodic_structure"])


def group_events(candidates: list[dict], thresholds: dict) -> list[dict]:
    """Join per-frame candidates into events: an object keeps its lateral and vertical
    position across frames while x changes, so match on (y, z) with a small frame gap."""
    th = {**DEFAULTS, **thresholds}
    events = []
    for candidate in sorted(candidates, key=lambda c: c["frame"]):
        best = None
        for event in events:
            gap = candidate["frame"] - event["last_frame"]
            if gap <= 0 or gap > th["event_max_frame_gap"]:
                continue
            dy = abs(candidate["center"][1] - event["last_center"][1])
            dz = abs(candidate["center"][2] - event["last_center"][2])
            within_tolerance = (dy <= th["event_lateral_tolerance_m"]
                                and dz <= th["event_vertical_tolerance_m"])
            if within_tolerance and (best is None or gap < candidate["frame"] - best["last_frame"]):
                best = event
        if best is None:
            events.append({"first_frame": candidate["frame"], "last_frame": candidate["frame"],
                           "last_center": candidate["center"], "frames": [candidate["frame"]],
                           "occurrences": [candidate], "min_distance_m": candidate["distance_m"]})
        else:
            best["last_frame"] = candidate["frame"]
            best["last_center"] = candidate["center"]
            best["frames"].append(candidate["frame"])
            best["occurrences"].append(candidate)
            best["min_distance_m"] = min(best["min_distance_m"], candidate["distance_m"])
    for event in events:
        occurrences = event.pop("occurrences")
        event.pop("last_center")
        centers = np.asarray([o["center"] for o in occurrences])
        event["median_center"] = np.median(centers, axis=0).round(3).tolist()
        event["lateral_drift_m"] = round(float(centers[:, 1].max() - centers[:, 1].min()), 2)
        event["max_inside_voxels"] = max(o["inside_voxels"] for o in occurrences)
        event["median_inside_extent_m"] = np.median(
            [o["inside_extent_m"] for o in occurrences], axis=0).round(2).tolist()
    return events


def report_candidates(events: list[dict], plan: dict) -> list[dict]:
    """Events that pass the reporting bar: real support, several frames, visible size, and
    inside a plausible lateral band. The band guards against envelope-estimate drift, which
    is what puts impossible |y| values on curved recordings."""
    bar = {"min_inside_voxels": 20, "min_frames": 3, "min_extent_m": 0.25, "max_lateral_m": 1.6}
    bar.update(plan.get("report", {}))
    out = []
    for event in events:
        if abs(event["median_center"][1]) > bar["max_lateral_m"]:
            continue
        if event["max_inside_voxels"] < bar["min_inside_voxels"] or len(event["frames"]) < bar["min_frames"]:
            continue
        if max(event["median_inside_extent_m"]) < bar["min_extent_m"]:
            continue
        out.append(event)
    return sorted(out, key=lambda e: e["min_distance_m"])


def scan(bag: Path, plan: dict, start_frame: int = 0, max_frames: int | None = None) -> dict:
    detector = load_config(Path(plan.get("detector_config", "configs/detector.json")))
    thresholds = plan.get("thresholds", {})
    clusters, selected = [], []
    for scan_row in iter_bag(bag, detector):
        if scan_row.index < start_frame:
            continue
        if max_frames is not None and scan_row.index >= max_frames:
            break
        for cluster in frame_candidates(scan_row.points, detector, thresholds):
            cluster["frame"] = scan_row.index
            cluster["timestamp_s"] = scan_row.timestamp_s
            clusters.append(cluster)
            if is_candidate(cluster, thresholds):
                selected.append(cluster)
    events = group_events(selected, thresholds)
    return {"bag": str(bag), "clusters": clusters, "events": events,
            "candidates": report_candidates(events, plan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/on-track-probe.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    plan = json.loads(args.config.read_text())
    result = scan(args.bag, plan, args.start_frame, args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")

    print(f"{args.bag.name}: {len(result['clusters'])} in-envelope clusters -> "
          f"{len(result['events'])} events -> {len(result['candidates'])} candidates")
    for event in result["candidates"][:20]:
        print(f"  frames {event['first_frame']:4d}-{event['last_frame']:<4d} ({len(event['frames']):3d})"
              f"  d={event['min_distance_m']:6.2f} m  center={event['median_center']}")
    print(f"Evidence: {args.output}")


if __name__ == "__main__":
    main()
