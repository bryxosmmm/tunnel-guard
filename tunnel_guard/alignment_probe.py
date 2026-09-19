"""Go/no-go for the long-lever alignment, measured against later frames' rails.

Three models predict the track centre-line at 60-300 m look-ahead; each prediction is expressed in
the axes of a later frame that has driven over that ground, where the rails measure the same physical
centre over a short, well-supported lever. Label-free.

  straight  : the previous continuation (interpolated anchors, then a clipped two-point tangent)
  window    : the shipped local quadratic from the anchor window (path_curve_window_m)
  long-lever: rails as the near-field anchor line continued with the shared curvature fitted to the
              tunnel surfaces (tunnel_guard.alignment)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import fit_alignment, surface_traces
from .detector import load_config
from .geometry import TrackGeometry
from .io import iter_bag


def old_centre(geometry: TrackGeometry, x: np.ndarray) -> np.ndarray:
    a = geometry.rail_anchors
    center = np.interp(x, a[:, 0], a[:, 1])
    for edge, other, mask in [(0, 1, x < a[0, 0]), (-1, -2, x > a[-1, 0])]:
        slope = np.clip((a[edge, 1] - a[other, 1]) / (a[edge, 0] - a[other, 0]),
                        -geometry.config["rail_max_heading"], geometry.config["rail_max_heading"])
        center[mask] = a[edge, 1] + slope * (x[mask] - a[edge, 0])
    return center


def long_lever_centre(geometry: TrackGeometry, alignment, x: float) -> float:
    """Anchor line from the measured rails, continued with the shared surface curvature."""
    a = geometry.rail_anchors
    x_edge, y_edge, slope = float(a[-1, 0]), float(a[-1, 1]), 0.0
    fitted = geometry._continuation(-1)
    slope = fitted[2]
    distance = x - x_edge
    return y_edge + slope * distance + alignment.curvature * (x * x - x_edge * x_edge) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detector-native.json"))
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--lookahead", type=float, nargs="+", default=[60.0, 100.0, 150.0, 200.0])
    args = parser.parse_args()
    cfg = load_config(args.config)
    poses = {json.loads(l)["frame"]: np.asarray(json.loads(l)["pose"], float) for l in open(args.poses)}
    # Every frame's rails can serve as a later measurement of ground an earlier frame predicted, so
    # anchors are kept for all of them; the fitted alignment is only needed at the source frames.
    geometry, fits, anchors = {}, {}, {}
    sources = set(args.frames)
    for scan in iter_bag(args.bag, cfg, max_frames=max(args.frames) + 1):
        g = TrackGeometry(scan.points, cfg)
        anchors[scan.index] = g.rail_anchors
        if scan.index in sources:
            geometry[scan.index] = g
            fits[scan.index] = fit_alignment(surface_traces(scan.points, -1.32, cfg),
                                             g.rail_anchors[:, :2] if len(g.rail_anchors) else None, cfg)
    print("look-ahead  n   straight    window   long-lever   (|median| error, m)")
    for look in args.lookahead:
        rows = []
        for k in sorted(geometry):
            g = geometry[k]
            if len(g.rail_anchors) < 3 or look <= g.rail_anchors[-1, 0]:
                continue
            for n in range(4, 150):
                later = k + n
                if later not in anchors or later not in poses:
                    continue
                travelled = n * 1.72
                if look - travelled < 2.0 or look - travelled > anchors[later][-1, 0]:
                    continue
                z = float(np.interp(look, g.ground_anchors[:, 0], g.ground_anchors[:, 1])) \
                    if len(g.ground_anchors) else 0.0
                predicted = {
                    "straight": float(old_centre(g, np.array([look]))[0]),
                    "window": float(g.path(np.array([look]))[0][0]),
                }
                if fits[k].valid:
                    predicted["long-lever"] = long_lever_centre(g, fits[k], look)
                errors = {}
                for name, y in predicted.items():
                    world = poses[k][:3, :3] @ np.array([look, y, z]) + poses[k][:3, 3]
                    local = (world - poses[later][:3, 3]) @ poses[later][:3, :3]
                    if not (2.0 <= local[0] <= anchors[later][-1, 0]):
                        break
                    measured = float(np.interp(local[0], anchors[later][:, 0], anchors[later][:, 1]))
                    errors[name] = abs(local[1] - measured)
                if len(errors) == len(predicted):
                    rows.append(errors)
                    break
        if not rows:
            print(f"  {look:5.0f} m   0   (no frame pair qualified)")
            continue
        def median(name):
            values = [r[name] for r in rows if name in r]
            return f"{np.median(values):6.3f}" if values else "   n/a"
        print(f"  {look:5.0f} m {len(rows):3d}   {median('straight')}   {median('window')}   {median('long-lever')}")
    fitted = [f for f in fits.values() if f.valid]
    print(f"\nframes with a fitted alignment: {len(fitted)}/{len(fits)}")
    for k, f in sorted(fits.items()):
        if f.valid:
            print(f"   frame {k:3d}: surfaces {f.surfaces} spans {['%.0f' % s for s in f.spans_m[:4]]} "
                  f"residual {f.residual_m:.3f} m curvature {f.curvature:+.6f} sigma {f.curvature_sigma:.2e} "
                  f"horizon(0.5 m) {f.horizon_m(0.5):5.0f} m")
        else:
            print(f"   frame {k:3d}: not fitted ({f.reason})")


if __name__ == "__main__":
    main()
