"""Forward prediction of the track centre-line from each alignment model (go/no-go harness).

Take a source frame, predict the centre-line at a look-ahead distance, express that world point in the
axes of a later frame that has driven over the same ground, and compare with what its rails measure
there. Label-free: the reference is the sensor's own later measurement over a short, well-supported
lever. Each model is scored independently, so a model that has no prediction at a range is reported as
absent rather than suppressing the row.

Models
  straight   : interpolated anchors, then a clipped two-point tangent (the original continuation)
  window     : the local quadratic from the anchor window (path_curve_window_m), shipped default
  long-lever : rails anchored, continued with the shared curvature fitted to the surface traces
  axis       : cross-section circle centres along the tunnel, offset by the measured rail offset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import axis_trace, cross_section_axis, fit_alignment, fit_spline, surface_traces
from .detector import load_config
from .geometry import TrackGeometry
from .io import iter_bag

STEP_M = 1.72          # mean travel per frame in these recordings


def _frame_from_rails(geometry: TrackGeometry) -> tuple[np.ndarray, np.ndarray, float]:
    """Bootstrap frame: along/lateral unit vectors and heading from the near-field rail anchors."""
    heading = float(np.polyfit(geometry.rail_anchors[:, 0], geometry.rail_anchors[:, 1], 1)[0])
    along = np.array([1.0, heading]) / np.hypot(1.0, heading)
    return along, np.array([-along[1], along[0]]), heading


def straight_centre(geometry: TrackGeometry, x: np.ndarray) -> np.ndarray:
    a = geometry.rail_anchors
    center = np.interp(x, a[:, 0], a[:, 1])
    for edge, other, mask in [(0, 1, x < a[0, 0]), (-1, -2, x > a[-1, 0])]:
        slope = np.clip((a[edge, 1] - a[other, 1]) / (a[edge, 0] - a[other, 0]),
                        -geometry.config["rail_max_heading"], geometry.config["rail_max_heading"])
        center[mask] = a[edge, 1] + slope * (x[mask] - a[edge, 0])
    return center


def long_lever_centre(geometry: TrackGeometry, alignment, x: float) -> float:
    a = geometry.rail_anchors
    x_edge, y_edge = float(a[-1, 0]), float(a[-1, 1])
    slope = geometry._continuation(-1)[2]
    distance = x - x_edge
    return y_edge + slope * distance + alignment.curvature * (x * x - x_edge * x_edge) / 2


def axis_centre(geometry: TrackGeometry, axis, lateral_dir: np.ndarray, along: np.ndarray, look: float,
                calibrate_within_m: float = 40.0):
    """Centre-line lateral position at `look` from the measured cross-section axis.

    The bore is not concentric with the track: a circle fitted through two walls, a flat floor and an
    arched ceiling sits about a metre off the track centre, and that bias changes where the section
    changes. So the axis is never used as an absolute centre. Its offset is *calibrated* against the
    rails over the range where both are visible, and only the calibrated axis is carried outward -
    assuming an offset persists is far more benign than assuming a curvature does.
    """
    if not axis.valid or len(axis.samples) < 4:
        return None
    stations = np.array([sample.station_m for sample in axis.samples])
    laterals = np.array([sample.lateral_m for sample in axis.samples])
    if not (stations.min() <= look <= stations.max()):
        return None
    rail_lateral = geometry.rail_anchors[:, :2] @ lateral_dir
    rail_station = geometry.rail_anchors[:, :2] @ along
    overlap = (rail_station <= calibrate_within_m) & (rail_station >= stations.min())
    if overlap.sum() < 3:
        return None
    offset = float(np.median(rail_lateral[overlap] - np.interp(rail_station[overlap], stations, laterals)))
    return float(np.interp(look, stations, laterals)) + offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--lookahead", type=float, nargs="+", default=[60.0, 100.0, 150.0, 200.0, 250.0])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    with open(args.poses) as stream:
        poses = {json.loads(line)["frame"]: np.asarray(json.loads(line)["pose"], float) for line in stream}

    geometry, fits, axes, splines = {}, {}, {}, {}
    anchors = {}
    sources = set(args.frames)
    for scan in iter_bag(args.bag, cfg, max_frames=max(args.frames) + 1):
        g = TrackGeometry(scan.points, cfg)
        anchors[scan.index] = g.rail_anchors
        if scan.index in sources:
            geometry[scan.index] = g
            fits[scan.index] = fit_alignment(surface_traces(scan.points, -1.32, cfg),
                                             g.rail_anchors[:, :2], cfg)
            axes[scan.index] = cross_section_axis(scan.points, g.rail_anchors, cfg)
            trace = axis_trace(axes[scan.index])
            splines[scan.index] = (fit_spline([trace], g.rail_anchors[:, :2], cfg)
                                   if trace is not None else None)

    results = {}
    print("look-ahead   n    straight   window   long-lever   axis+calib   (|median| error, m)")
    for look in args.lookahead:
        per_model: dict[str, list[float]] = {}
        for k in sorted(geometry):
            g = geometry[k]
            if len(g.rail_anchors) < 3 or look <= g.rail_anchors[-1, 0]:
                continue
            along, lateral_dir, _ = _frame_from_rails(g)
            for n in range(4, 200):
                later = k + n
                if later not in anchors or later not in poses:
                    continue
                travelled = n * STEP_M
                if look - travelled < 2.0 or look - travelled > anchors[later][-1, 0]:
                    continue
                bed = float(np.interp(look, g.ground_anchors[:, 0], g.ground_anchors[:, 1])) \
                    if len(g.ground_anchors) else 0.0
                predicted = {"straight": float(straight_centre(g, np.array([look]))[0]),
                             "window": float(g.path(np.array([look]))[0][0])}
                if fits[k].valid:
                    predicted["long-lever"] = long_lever_centre(g, fits[k], look)
                axis_value = axis_centre(g, axes[k], lateral_dir, along, look) if axes.get(k) and axes[k].valid else None
                if axis_value is not None:
                    predicted["axis+calib"] = axis_value
                compared = False
                for name, lateral in predicted.items():
                    # world point: station along the bootstrap line, lateral offset, bed height
                    world = poses[k][:3, :3] @ np.array([look, lateral, bed]) + poses[k][:3, 3]
                    local = (world - poses[later][:3, 3]) @ poses[later][:3, :3]
                    if not (2.0 <= local[0] <= anchors[later][-1, 0]):
                        continue
                    measured = float(np.interp(local[0], anchors[later][:, 0], anchors[later][:, 1]))
                    per_model.setdefault(name, []).append(abs(local[1] - measured))
                    compared = True
                if compared:
                    break        # only stop searching for a later frame once one was comparable
        row = {}
        for name in ("straight", "window", "long-lever", "axis+calib"):
            values = per_model.get(name, [])
            row[name] = float(np.median(values)) if values else None
        results[str(look)] = {"n": max((len(v) for v in per_model.values()), default=0), **row}
        cells = "   ".join(f"{row[name]:8.3f}" if row[name] is not None else "     n/a" for name in
                           ("straight", "window", "long-lever", "axis+calib"))
        print(f"  {look:5.0f} m  {results[str(look)]['n']:3d}   {cells}")

    axes_summary = {k: {"valid": a.valid, "reason": a.reason, "furthest_m": a.furthest_m,
                        "samples": len(a.samples), "breaks": len(a.breaks),
                        "sigma_median_m": float(np.median([s.sigma_m for s in a.samples])) if a.samples else None,
                        "rail_offset_m": a.rail_offset_median_m, "rail_offset_spread_m": a.rail_offset_spread_m}
                    for k, a in axes.items()}
    print("\ncross-section axis per source frame:")
    for k, summary in sorted(axes_summary.items()):
        if summary["valid"]:
            print(f"   frame {k:3d}: {summary['samples']:2d} slabs to {summary['furthest_m']:5.0f} m, "
                  f"{summary['breaks']} breaks, sigma median {summary['sigma_median_m']:.3f} m, "
                  f"rail offset {summary['rail_offset_m']:+.3f} m (spread {summary['rail_offset_spread_m']:.3f})")
        else:
            print(f"   frame {k:3d}: {summary['reason']}")
    if args.json_out:
        args.json_out.write_text(json.dumps({"prediction": results, "axis": axes_summary,
                                             "bag": str(args.bag), "config": str(args.config)}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
