"""Throwaway: array-level comparison of the native geometry kernels with NumPy.

Runs the detector over the same real scans twice -- once with the NumPy recipe, once with the
native kernels -- recording every classification, coordinate, mask and structural verdict,
then compares the arrays. No detector decision is interpreted here.

The project has no test suite by rule, so this is the only place a silent divergence between
the two backends could show itself, and it is run by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tunnel_guard import background as background_mod
from tunnel_guard import geometry as geometry_mod
from tunnel_guard.detector import Detector, load_config
from tunnel_guard.io import iter_bag

CAPTURE: dict = {}


def record(tag: str, label: str, payload):
    CAPTURE.setdefault(tag, {}).setdefault(label, []).append(payload)


def run(config_path: Path, bag: Path, frames: int, tag: str) -> dict:
    CAPTURE[tag] = {}
    real_init = geometry_mod.TrackGeometry.__init__
    real_classify = geometry_mod.TrackGeometry.classify_with_section
    real_structural = geometry_mod.TrackGeometry.structural_mask
    real_mask = background_mod.TunnelBackground.mask

    def init(self, points, config):
        real_init(self, points, config)
        self._captured_points = points

    def classify_with_section(self, points, *, remove_background=True, include_boundary=False, ground=None):
        masks, section = real_classify(self, points, remove_background=remove_background,
                                      include_boundary=include_boundary, ground=ground)
        record(tag, f"state@{len(CAPTURE[tag].get('plane', []))}",
               {"plane": None if self.plane is None else self.plane.copy(),
                "ground_anchors": self.ground_anchors.copy(),
                "rail_anchors": self.rail_anchors.copy(),
                "rail_head": self.rail_head_height_m})
        record(tag, f"classify_{len(points)}_{remove_background}_{include_boundary}",
               tuple(np.array(part, copy=True) for part in masks))
        record(tag, f"section_{len(points)}",
               tuple(np.array(part, copy=True) for part in section))
        return masks, section

    def structural_mask(self, points, section=None):
        result = real_structural(self, points, section)
        record(tag, f"structural_{len(points)}", np.array(result, copy=True))
        return result

    def mask(self, points, protected):
        result = real_mask(self, points, protected)
        record(tag, f"mask_{len(points)}", np.array(result, copy=True))
        return result

    geometry_mod.TrackGeometry.__init__ = init
    geometry_mod.TrackGeometry.classify_with_section = classify_with_section
    geometry_mod.TrackGeometry.structural_mask = structural_mask
    background_mod.TunnelBackground.mask = mask
    config = load_config(config_path)
    detector = Detector(config)
    for scan in iter_bag(bag, config, max_frames=frames):
        detector.process(scan.points, scan.timestamp_s, scan.point_times)
    geometry_mod.TrackGeometry.__init__ = real_init
    geometry_mod.TrackGeometry.classify_with_section = real_classify
    geometry_mod.TrackGeometry.structural_mask = real_structural
    background_mod.TunnelBackground.mask = real_mask
    return CAPTURE[tag]


def compare_arrays(a, b) -> str | None:
    if isinstance(a, dict):
        if set(a) != set(b):
            return "keys differ"
        for key in a:
            detail = compare_arrays(a[key], b[key])
            if detail:
                return f"{key}: {detail}"
        return None
    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.ndim == 0):
        if np.array_equal(a, b, equal_nan=True):
            return None
        return f"scalar {a!r} vs {b!r}"
    for index, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            return f"part{index}: shape {x.shape} vs {y.shape}"
        if x.dtype == bool or y.dtype == bool:
            count = int(np.count_nonzero(x != y))
            if count:
                return f"part{index}: {count} of {x.size} differ"
        else:
            if not np.array_equal(x, y, equal_nan=True):
                difference = np.abs(np.asarray(x, float) - np.asarray(y, float))
                difference = difference[np.isfinite(difference)]
                return (f"part{index}: max |diff| "
                        f"{float(difference.max()) if difference.size else float('nan'):.3g}")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()

    reference = run(Path("configs/detector.json"), args.bag, args.frames, "numpy")
    native = run(Path("configs/detector-native.json"), args.bag, args.frames, "native")

    failures = 0
    compared = 0
    labels = sorted(set(reference) | set(native))
    for label in labels:
        left, right = reference.get(label), native.get(label)
        if left is None or right is None or len(left) != len(right):
            print(f"FAIL {label}: missing or count mismatch "
                  f"({0 if left is None else len(left)} vs {0 if right is None else len(right)})")
            failures += 1
            continue
        for index, (a, b) in enumerate(zip(left, right)):
            compared += 1
            detail = compare_arrays(a, b)
            if detail:
                print(f"FAIL {label}[{index}] {detail}")
                failures += 1
    print(f"{compared - failures}/{compared} recorded arrays identical across the two recipes")
    print(json.dumps({"compared": compared, "failures": failures,
                      "labels": {label: len(reference.get(label, [])) for label in labels}}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
