"""Render real low-level rail returns without choosing a box from detector proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import iter_bag
from .object_evidence_report import rows
from .review_remaining_alerts import geometry
from .run import digest, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    run = Path(plan["source_run"])
    bag = Path(plan["bag"])
    config = json.loads((run / "detector.json").read_text())
    recorded = {r["frame"]: r for r in rows(run, bag.name)}
    frames = plan["frames"]
    bounds = plan["forward_bounds_m"]
    lateral = plan["track_half_width_m"]
    height_bounds = plan["height_view_bounds_m"]
    fig, axes = plt.subplots(len(frames), 2, figsize=(17, 3 * len(frames)), layout="constrained", squeeze=False)
    details = []
    for scan in iter_bag(bag, config):
        if scan.index > max(frames):
            break
        row = recorded[scan.index]
        scan_id = f"{scan.topic}:{scan.frame_id}:{scan.measurement_timestamp_ns}"
        if row["source_scan_id"] != scan_id:
            raise ValueError("Recorded geometry belongs to a different measurement")
        g = geometry(row, config)
        if not row["geometry"]["valid"] or len(g.rail_anchors) < 2:
            raise ValueError("Review requires measured path and bed anchors")
        points = scan.points[(scan.points[:, 0] >= bounds[0]) & (scan.points[:, 0] <= bounds[1])]
        center, gauge, path_uncertainty = g.path(points[:, 0])
        bed, bed_uncertainty = g.ground(points)
        height = points[:, 2] - bed
        offset = points[:, 1] - center
        review = (np.abs(offset) <= lateral) & (height >= height_bounds[0]) & (height <= height_bounds[1])
        selected, h, y = points[review], height[review], offset[review]
        above = h > g.rail_head_height_m + plan["above_railhead_review_margin_m"]
        if scan.index in frames:
            np.savez_compressed(output / f"raw-{scan.index:06d}.npz", points=selected, bed_height=h, lateral_offset=y,
                                bed_uncertainty=bed_uncertainty[review], path_uncertainty=path_uncertainty[review])
            ix = frames.index(scan.index)
            color = axes[ix, 0].scatter(selected[:, 0], y, c=h, s=3, cmap="turbo", vmin=height_bounds[0], vmax=height_bounds[1])
            x = np.linspace(bounds[0], bounds[1], 250)
            _, widths, _ = g.path(x)
            for sign in (-1, 1):
                axes[ix, 0].plot(x, sign * widths / 2, "k--", lw=0.7)
            axes[ix, 0].set(xlim=bounds, ylim=(-lateral, lateral), ylabel="offset from estimated track center [m]",
                           title=f"frame {scan.index}: raw low returns; dashed = estimated rails")
            axes[ix, 1].scatter(selected[:, 0], h, c=y, s=3, cmap="coolwarm", vmin=-lateral, vmax=lateral)
            axes[ix, 1].axhline(g.rail_head_height_m, color="black", ls="--", lw=0.7)
            axes[ix, 1].set(xlim=bounds, ylim=height_bounds, ylabel="height above estimated bed [m]",
                           title=f"{len(selected)} returns in explicitly bounded review slice")
            fig.colorbar(color, ax=axes[ix, 0], label="height above estimated bed [m]")
        details.append({"frame": scan.index, "timestamp_s": scan.timestamp_s, "source_scan_id": scan_id,
                        "review_returns": len(selected), "above_railhead_plus_margin_returns": int(above.sum()),
                        "above_railhead_plus_margin_xyz": selected[above].tolist(),
                        "rail_head_height_m": g.rail_head_height_m,
                        "max_height_in_view_m": float(h.max()) if len(h) else None,
                        "max_bed_uncertainty_m": float(bed_uncertainty[review].max()) if len(h) else None,
                        "detector_interval_overlaps": [o for o in row["objects"] if o["bbox_min"][0] <= bounds[1]
                            and o["bbox_max"][0] >= bounds[0] and o["lateral_m"][0] <= lateral
                            and o["lateral_m"][1] >= -lateral and o["height_above_bed_m"][0] <= height_bounds[1]
                            and o["height_above_bed_m"][1] >= height_bounds[0]],
                        "geometry": row["geometry"]})
    if len(details) != plan["expected_frames"]:
        raise ValueError("Recording coverage is incomplete")
    for ax in axes[-1]:
        ax.set_xlabel("forward x from configured processing origin [m]")
    fig.savefig(output / "raw-rail-views.png", dpi=150)
    plt.close(fig)
    report = {"plan": plan, "recipe_sha256": digest(args.experiment), "environment": environment(),
              "source_manifest": json.loads((run / "manifest.json").read_text()), "frames": details,
              "limits": "Exploratory review, not a label or an exhaustive negative. Low returns below the estimated railhead can belong to sleepers, bed irregularities, or a low object. The visualization does not establish object identity, true railhead geometry, swept clearance, or first physical visibility."}
    write_json(output / "summary.json", report)
    print(json.dumps({"reviewed_frames": len(details),
                      "above_railhead_plus_margin_returns": sum(d["above_railhead_plus_margin_returns"] for d in details),
                      "max_height_in_view_m": max(d["max_height_in_view_m"] for d in details),
                      "rendered_frames": frames}, indent=2))


if __name__ == "__main__":
    main()
