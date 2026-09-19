"""Throwaway: array-level comparison of the native geometry kernels with NumPy.

Runs the detector over the same real scans twice -- once with the NumPy recipe,
once with the native kernels -- recording every classification and mask result,
then compares the arrays. No detector decision is interpreted here.
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
    real_classify = geometry_mod.TrackGeometry.classify
    real_mask = background_mod.TunnelBackground.mask

    def init(self, points, config):
        real_init(self, points, config)
        self._captured_points = points

    def classify(self, points, *, remove_background=True, include_boundary=False, ground=None):
        result = real_classify(self, points, remove_background=remove_background,
                               include_boundary=include_boundary, ground=ground)
        record(tag, f"state@{len(CAPTURE[tag].get('plane', []))}",
               {"plane": None if self.plane is None else self.plane.copy(),
                "ground_anchors": self.ground_anchors.copy(),
                "rail_anchors": self.rail_anchors.copy(),
                "rail_head": self.rail_head_height_m})
        record(tag, f"classify_{len(points)}_{remove_background}_{include_boundary}",
               tuple(np.array(part, copy=True) for part in result))
        record(tag, f"ground_{len(points)}", tuple(np.array(part, copy=True)
                                                   for part in self.ground(points)))
        return result

    def mask(self, points, protected):
        result = real_mask(self, points, protected)
        record(tag, f"mask_{len(points)}", np.array(result, copy=True))
        return result

    geometry_mod.TrackGeometry.__init__ = init
    geometry_mod.TrackGeometry.classify = classify
    background_mod.TunnelBackground.mask = mask
    config = load_config(config_path)
    detector = Detector(config)
    for scan in iter_bag(bag, config, max_frames=frames):
        detector.process(scan.points, scan.timestamp_s, scan.point_times)
    geometry_mod.TrackGeometry.__init__ = real_init
    geometry_mod.TrackGeometry.classify = real_classify
    background_mod.TunnelBackground.mask = real_mask
    return CAPTURE[tag]


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
            print(f"FAIL {label}: missing or count mismatch ({0 if left is None else len(left)} vs "
                  f"{0 if right is None else len(right)})")
            failures += 1
            continue
        for index, (a, b) in enumerate(zip(left, right)):
            compared += 1
            if isinstance(a, dict if not isinstance(a, tuple) else tuple) and isinstance(a, dict):
                same = all(np.array_equal(a[key], b[key], equal_nan=True) for key in a)
            else:
                same = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))
            if not same:
                detail = []
                if isinstance(a, tuple):
                    for name, x, y in zip(range(len(a)), a, b):
                        if not np.array_equal(x, y, equal_nan=True):
                            detail.append(f"part{name}: {int(np.count_nonzero(x != y)) if x.dtype == bool else float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))))} differ")
                print(f"FAIL {label}[{index}] {'; '.join(detail)}")
                failures += 1
    print(f"{compared - failures}/{compared} recorded arrays identical across the two recipes")
    print(json.dumps({"compared": compared, "failures": failures,
                      "labels": {label: len(reference.get(label, [])) for label in labels}}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
