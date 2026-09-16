"""Check the rail-based track centre against the tunnel bore, independently of the rails.

``TrackGeometry`` derives the track centre from one source: the paired-rail match. When that
match takes a wrong pair there is nothing to contradict it, and ``center_stability`` shows the
centre then sits 2-3 m off axis for runs of frames. The bore is a second source, present in
the same scan, needing no odometry, no map and no empty-tunnel reference.

The bore is used through its mirror symmetry rather than a fitted shape, so a round and a
boxed section are treated alike and nothing is assumed about the lining profile. A symmetry
axis is not the track centre, though: in a two-track tunnel it sits between the tracks. That
offset is a property of the tunnel and the track being run, constant within a scan, so it is
calibrated in the near field, where the rail chain is still trustworthy -- across this subset
no excursion departs before 20 m -- and then carried out to the ranges under test.

The symmetry score says where the method applies. A platform breaks the bore's symmetry, and
the score falls with it, so frames the reference cannot judge are declared rather than
guessed. Read the score as a precondition, not as a confidence: on a recording holding both a
platform and a switch it still admits frames it gets wrong.

Nothing in ``geometry.py`` is used beyond its public output, and nothing is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag

NEAR_RANGES_M = (10.0, 15.0)
FAR_RANGES_M = (25.0, 30.0, 35.0, 40.0)
SLAB_HALF_M = 2.5
BIN_M = 0.05
SEARCH_HALF_M = 3.5
MIN_HEIGHT_M = 1.2
EDGES = np.arange(-8.0, 8.0 + BIN_M, BIN_M)
BIN_CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
SHIFTS = np.arange(-SEARCH_HALF_M, SEARCH_HALF_M + BIN_M, BIN_M)


def symmetry_axis(lateral: np.ndarray) -> tuple[float | None, float | None]:
    """Lateral offset whose mirror image best reproduces the cross-section, and the overlap.

    The score is the share of the lateral profile that survives mirroring, so a symmetric bore
    scores high and a platform on one side scores low without either case being named.
    """
    if len(lateral) < 60:
        return None, None
    profile = np.histogram(lateral, bins=EDGES)[0].astype(float)
    total = profile.sum()
    if total < 60:
        return None, None
    overlap = [np.minimum(profile, np.interp(2 * shift - BIN_CENTRES, BIN_CENTRES, profile,
                                             left=0.0, right=0.0)).sum() for shift in SHIFTS]
    best = int(np.argmax(overlap))
    return float(SHIFTS[best]), float(overlap[best] / total)


def frame_row(scan_points: np.ndarray, geometry: TrackGeometry) -> dict:
    height = scan_points[:, 2] - geometry.ground(scan_points)[0]
    anchors = geometry.rail_anchors
    probes = {}
    for probe in NEAR_RANGES_M + FAR_RANGES_M:
        slab = ((np.abs(scan_points[:, 0] - probe) < SLAB_HALF_M) & (height > MIN_HEIGHT_M)
                & (np.abs(scan_points[:, 1]) < 8))
        axis, score = symmetry_axis(scan_points[slab, 1])
        anchored = len(anchors) >= 2 and anchors[0, 0] <= probe <= anchors[-1, 0]
        probes[str(probe)] = {
            "bore_axis_m": axis, "symmetry_score": score,
            "rail_center_m": (float(np.interp(probe, anchors[:, 0], anchors[:, 1]))
                              if anchored else None)}
    return probes


def calibrated_disagreement(probes: dict, min_score: float) -> dict:
    """Bore prediction of the track centre, and how far the rails sit from it.

    Returns nothing usable when the near field cannot calibrate the offset or when the
    section is not symmetric enough to be read, which is the point of reporting it.
    """
    offsets, near_scores = [], []
    for probe in NEAR_RANGES_M:
        p = probes[str(probe)]
        if p["bore_axis_m"] is not None and p["rail_center_m"] is not None:
            offsets.append(p["bore_axis_m"] - p["rail_center_m"])
            near_scores.append(p["symmetry_score"])
    if not offsets:
        return {"usable": False, "reason": "near_field_offset_unavailable"}
    offset = float(np.median(offsets))
    worst_near = float(min(near_scores))
    out = {"usable": False, "bore_to_track_offset_m": round(offset, 3),
           "near_symmetry_score": round(worst_near, 3), "reason": "section_not_symmetric",
           "by_range": {}}
    for probe in FAR_RANGES_M:
        p = probes[str(probe)]
        if p["bore_axis_m"] is None or p["rail_center_m"] is None:
            continue
        score = min(p["symmetry_score"], worst_near)
        if score < min_score:
            continue
        predicted = p["bore_axis_m"] - offset
        out["by_range"][str(probe)] = {
            "predicted_center_m": round(predicted, 3),
            "rail_center_m": round(p["rail_center_m"], 3),
            "disagreement_m": round(abs(predicted - p["rail_center_m"]), 3),
            "symmetry_score": round(score, 3)}
    if out["by_range"]:
        out["usable"] = True
        out["reason"] = "bore_readable"
        out["max_disagreement_m"] = max(v["disagreement_m"] for v in out["by_range"].values())
    return out


def measure(bag: Path, detector: dict, every: int, max_frames: int | None, min_score: float) -> dict:
    rows = []
    for scan in iter_bag(bag, detector, every=every, max_frames=max_frames):
        forward = ((scan.points[:, 0] >= detector["min_forward_m"])
                   & (scan.points[:, 0] <= detector["max_range_m"])
                   & (np.abs(scan.points[:, 1]) <= detector["context_half_width_m"]))
        geometry = TrackGeometry(voxel_representatives(scan.points[forward],
                                                       detector["geometry_voxel_m"]), detector)
        if not geometry.valid:
            continue
        probes = frame_row(scan.points, geometry)
        rows.append({"frame": scan.index, "probes": probes,
                     "check": calibrated_disagreement(probes, min_score)})
    usable = [r for r in rows if r["check"]["usable"]]
    disagreement = np.asarray([r["check"]["max_disagreement_m"] for r in usable])
    return {
        "bag": str(bag), "frames": len(rows), "frames_readable": len(usable),
        "readable_share": round(len(usable) / max(len(rows), 1), 3),
        "disagreement_m": ({"p50": round(float(np.median(disagreement)), 3),
                            "p90": round(float(np.quantile(disagreement, 0.9)), 3),
                            "max": round(float(disagreement.max()), 3)}
                           if len(disagreement) else None),
        "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-symmetry-score", type=float, default=0.50)
    args = parser.parse_args()

    detector = load_config(args.config)
    detector = {**detector, "background": {**detector["background"], "enabled": False}}
    results = [measure(bag, detector, args.every, args.max_frames, args.min_symmetry_score)
               for bag in args.bag]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": str(args.config), "min_symmetry_score": args.min_symmetry_score,
        "near_ranges_m": list(NEAR_RANGES_M), "far_ranges_m": list(FAR_RANGES_M),
        "seed": detector["seed"], "recordings": results}, indent=1) + "\n")

    for result in results:
        d = result["disagreement_m"]
        print(f"{Path(result['bag']).name:38s} {result['frames']:4d} frames, bore readable in "
              f"{result['readable_share']:.0%}"
              + (f", rails disagree p50 {d['p50']:.2f} p90 {d['p90']:.2f} max {d['max']:.2f} m"
                 if d else ", no readable section"))
    print(f"\nEvidence: {args.output}")


if __name__ == "__main__":
    main()
