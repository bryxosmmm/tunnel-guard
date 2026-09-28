"""Measure inclusive wall time of actual replay stages, retaining normal outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import accelerator, background, detector, geometry, io, mounting, run, segmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    samples = defaultdict(list)
    targets = [
        (detector.Detector, "process"), (detector.Detector, "_motion"),
        (detector.Detector, "_associate"), (detector, "cluster_candidates"),
        (detector, "density_labels"), (geometry.TrackGeometry, "__init__"),
        (geometry.TrackGeometry, "classify"), (geometry, "robust_plane"),
        (background.TunnelBackground, "__init__"), (background.TunnelBackground, "mask"),
        (accelerator, "normal_statistics"), (accelerator, "density_graph"),
        (accelerator, "segment_plane"),
        (io, "decode_cloud"),
        # The per-frame refits and third-party kernels that the stages above contain.
        (detector, "_far_field_bounds"),
        (geometry.TrackGeometry, "_ground_profile"), (geometry.TrackGeometry, "_rail_profile"),
        (geometry.TrackGeometry, "ground"), (geometry.TrackGeometry, "path"),
        (geometry.TrackGeometry, "structural_mask"), (geometry.TrackGeometry, "supported_range_m"),
        (geometry.TrackGeometry, "classify_with_section"),
        (geometry, "voxel_representative_indices"),
        # The consumers the coarse stages above cannot separate: the native classifier and
        # component builder, the crop/range reducers, and the association backends.
        (accelerator, "classify_geometry"), (accelerator, "cluster_objects"),
        (accelerator, "crop_voxels"), (accelerator, "range_indices"),
        (accelerator, "range_indices_open"), (accelerator, "range_summary"),
        (accelerator, "voxel_counts"), (accelerator, "component_labels"),
        (accelerator, "induced_subgraph"), (accelerator, "keep_outside_radius"),
        (detector, "linear_sum_assignment"),
        (segmentation, "_separate_running_surface"), (mounting, "observe_mounting"),
    ]
    import open3d as o3d
    targets += [(o3d.geometry.PointCloud, "segment_plane"),
                (o3d.geometry.PointCloud, "voxel_down_sample")]
    originals = []

    def instrument(function, label):
        @wraps(function)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                samples[label].append(time.perf_counter() - started)
        return measured

    original_argv = sys.argv
    try:
        for owner, name in targets:
            original = getattr(owner, name)
            originals.append((owner, name, original))
            setattr(owner, name, instrument(original, f"{owner.__name__}.{name}"))
        sys.argv = [__file__, "--experiment", str(args.experiment)]
        run.main()
    finally:
        sys.argv = original_argv
        for owner, name, original in originals:
            setattr(owner, name, original)
    run.write_json(Path(recipe["output"]) / "wall-profile.json", {
        "command": original_argv,
        "limitations": ["Inclusive nested times must not be summed.",
                        "Includes first-frame initialization; profiling has overhead.",
                        "No inference about accuracy or unmeasured target hardware."],
        "stages": {label: {"calls": len(values), "total_s": sum(values),
                           "p50_ms": float(np.median(values) * 1000),
                           "samples_s": values} for label, values in samples.items()},
    })


if __name__ == "__main__":
    main()
