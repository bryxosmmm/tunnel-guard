"""Plot recorded authored-scene rail surface errors and measured support gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import TrackGeometry
from .track_scene_experiment import sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads((args.run / "experiment.json").read_text())
    scene = json.loads((args.run / "scene.json").read_text())
    configs = json.loads((args.run / "detectors.json").read_text())
    cases = recipe["cases"][:4]
    fig, axes = plt.subplots(2, len(cases), figsize=(15, 7), squeeze=False)
    colors = {"old_local3d": "#c56c23", "joint": "#8a74b7", "surface": "#8a74b7", "local3d": "#008d9c"}
    labels = {"old_local3d": "Upper quantile + lines", "joint": "Joint robust fit",
              "surface": "Surface fit, fixed stations", "local3d": "Dominant surface + joint fit"}
    for column, case in enumerate(cases):
        stations = np.arange(0., 90., .25)
        center, _, _, normal = sections(stations, scene, case)
        truth = center + scene["rail_height_m"] * normal
        for name, config in configs.items():
            with (args.run / (case["id"] + "-" + name + ".jsonl")).open() as stream:
                row = json.loads(next(stream))
            description = row["geometry"]
            geometry = TrackGeometry.__new__(TrackGeometry)
            geometry.config = config
            geometry.rail_frames = description["rail_frames"]
            geometry.rail_frame_version = description.get("rail_frame_version", 1)
            geometry.rail_support_diagnostics = description["rail_support_diagnostics"]
            height, angle = np.full(len(stations), np.nan), np.full(len(stations), np.nan)
            for a, b, _, _, n, *_ in geometry.frame_segments():
                valid = (stations >= a[0]) & (stations <= b[0])
                x = stations[valid]
                predicted = a[2] + (b[2]-a[2]) * (x-a[0])/(b[0]-a[0])
                height[valid] = 1000 * (predicted - np.interp(x, truth[:, 0], truth[:, 2]))
                actual_n = np.column_stack([np.interp(x, truth[:, 0], normal[:, k]) for k in range(3)])
                actual_n /= np.linalg.norm(actual_n, axis=1)[:, None]
                angle[valid] = np.degrees(np.arccos(np.clip(actual_n @ n, -1, 1)))
            axes[0, column].plot(stations, height, color=colors[name], label=labels[name])
            axes[1, column].plot(stations, angle, color=colors[name])
        axes[0, column].set_title(case["id"])
        axes[0, column].axhline(0, color="black", linewidth=.6)
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
            ax.set_xlabel("Forward coordinate (m)")
    axes[0, 0].set_ylabel("Rail-center height error (mm)")
    axes[1, 0].set_ylabel("Section normal error (degrees)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3)
    fig.suptitle("Authored ray-cast scenes, frame 0 — gaps mean unavailable rail sections\n"
                 "Each estimator's own support is shown; this is not field accuracy", fontsize=12)
    fig.tight_layout(rect=(0, .06, 1, .92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
