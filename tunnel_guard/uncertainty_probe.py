"""Is the propagated path uncertainty honest? Calibrate sigma_path against measured error.

The corridor is currently cut off a fixed 25 m beyond the last rail anchor because the previous
continuation had no error model. The curvature continuation does have one: the anchor-window fit
returns slope and curvature standard errors, and `TrackGeometry.path` already adds
sigma_path(x) = sqrt((d*sigma_slope)^2 + (d^2*sigma_curv)^2) beyond the anchor edge.

If that sigma is honest (covers the measured error, without being wildly conservative), the corridor
can be modelled as far as its own error budget allows instead of a fixed horizon - which is the only
way to reach further on this data without certifying unmeasured coverage.

Method: the same label-free pair test as `alignment_probe`, but recording both the measured absolute
error of the modelled centre and the sigma the model claims at that range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import load_config
from .geometry import TrackGeometry
from .io import iter_bag

STEP_M = 1.72


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detector-native.json"))
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--lookahead", type=float, nargs="+", default=[60.0, 80.0, 100.0, 120.0, 150.0])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    with open(args.poses) as stream:
        poses = {json.loads(line)["frame"]: np.asarray(json.loads(line)["pose"], float) for line in stream}

    geometry, anchors = {}, {}
    sources = set(args.frames)
    for scan in iter_bag(args.bag, cfg, max_frames=max(args.frames) + 1):
        g = TrackGeometry(scan.points, cfg)
        anchors[scan.index] = g.rail_anchors
        if scan.index in sources:
            geometry[scan.index] = g

    print("look-ahead   n   measured |error| (m)   claimed sigma (m)   ratio   coverage")
    records = {}
    for look in args.lookahead:
        errors, sigmas = [], []
        for k in sorted(geometry):
            g = geometry[k]
            if len(g.rail_anchors) < 3 or look <= g.rail_anchors[-1, 0]:
                continue
            x_edge, _, _, _, slope_sigma, curvature_sigma = g._continuation(-1)
            distance = look - x_edge
            extension = float(np.sqrt((distance * slope_sigma) ** 2 + (distance * distance * curvature_sigma) ** 2))
            for n in range(4, 200):
                later = k + n
                if later not in anchors or later not in poses:
                    continue
                travelled = n * STEP_M
                if look - travelled < 2.0 or look - travelled > anchors[later][-1, 0]:
                    continue
                bed = float(np.interp(look, g.ground_anchors[:, 0], g.ground_anchors[:, 1])) \
                    if len(g.ground_anchors) else 0.0
                predicted = float(g.path(np.array([look]))[0][0])
                world = poses[k][:3, :3] @ np.array([look, predicted, bed]) + poses[k][:3, 3]
                local = (world - poses[later][:3, 3]) @ poses[later][:3, :3]
                if not (2.0 <= local[0] <= anchors[later][-1, 0]):
                    continue
                measured = float(np.interp(local[0], anchors[later][:, 0], anchors[later][:, 1]))
                errors.append(abs(local[1] - measured))
                sigmas.append(extension)
                break
        if not errors:
            print(f"  {look:5.0f} m   0   (no frame pair qualified)")
            continue
        errors, sigmas = np.array(errors), np.array(sigmas)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sigmas > 0, errors / sigmas, np.inf)
        coverage = float(np.mean(errors <= sigmas))
        records[str(look)] = {"n": len(errors), "error_median_m": float(np.median(errors)),
                              "sigma_median_m": float(np.median(sigmas)),
                              "ratio_median": float(np.median(ratio)), "coverage": coverage,
                              "errors": errors.tolist(), "sigmas": sigmas.tolist()}
        print(f"  {look:5.0f} m {len(errors):3d}   {np.median(errors):14.3f}   {np.median(sigmas):14.3f}   "
              f"{np.median(ratio):6.2f}   {coverage:6.2f}")

    print("\nratio = measured error / claimed sigma. >1 means the model is over-confident.")
    if args.json_out:
        args.json_out.write_text(json.dumps({"bag": str(args.bag), "config": str(args.config),
                                             "calibration": records}, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
