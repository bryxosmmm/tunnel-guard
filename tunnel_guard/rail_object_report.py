"""Report a raw-reviewed region without promoting it to an independently labelled object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .object_evidence_report import rows
from .review_remaining_alerts import geometry, transform
from .run import digest, write_json


def plot_membership(plan, trace, captured, output):
    frames = plan["view_frames"]
    fig, axes = plt.subplots(len(frames), 2, figsize=(12, 4 * len(frames)), layout="constrained", squeeze=False)
    low, high = np.array(plan["context_roi_min"]), np.array(plan["context_roi_max"])
    roi_low, roi_high = np.array(plan["review_roi_min"]), np.array(plan["review_roi_max"])
    for index, frame in enumerate(frames):
        with np.load(trace / "diagnostics" / f"{plan['bag']}_{frame:06d}.npz") as arrays:
            raw = arrays["decoded_points"]
            raw = raw[np.all((raw >= low) & (raw <= high), axis=1)]
            for axis, dimension in zip(axes[index], (1, 2)):
                axis.scatter(raw[:, 0], raw[:, dimension], s=5, c="0.75", label="raw returns")
                for obj in captured[frame]["objects"]:
                    points = arrays[f"support_{obj['track_id']}"]
                    visible = np.all((points >= low) & (points <= high), axis=1)
                    if visible.any():
                        axis.scatter(points[visible, 0], points[visible, dimension], s=18, label=f"track {obj['track_id']}")
                axis.add_patch(Rectangle((roi_low[0], roi_low[dimension]), roi_high[0] - roi_low[0],
                                         roi_high[dimension] - roi_low[dimension], fill=False, linestyle="--", color="black"))
                axis.set(xlim=(low[0], high[0]), ylim=(low[dimension], high[dimension]), xlabel="forward x [m]",
                         ylabel=("lateral y [m]" if dimension == 1 else "processing z [m]"),
                         title=f"frame {frame}: actual support membership; dashed = review ROI, not GT")
                axis.legend(fontsize=8)
    fig.savefig(output / "support-membership.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    root, trace = Path(plan["run"]), Path(plan["trace"])
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "reporter.py").write_bytes(Path(__file__).read_bytes())
    (output / "experiment.json").write_bytes(args.experiment.read_bytes())
    full = list(rows(root, plan["bag"]))
    captured = {r["frame"]: r for r in rows(trace, plan["bag"])}
    config = json.loads((trace / "detector.json").read_text())
    observations = [{"frame": r["frame"], "timestamp_s": r["timestamp_s"], "measurement_timestamp_ns": r["measurement_timestamp_ns"], **o}
                    for r in full for o in r["objects"] if o["track_id"] == plan["track_id"]]
    origin_ns = full[0]["measurement_timestamp_ns"]
    firsts = {}
    for name, field in (("candidate", None), ("presence", "presence_confirmed"), ("hazard_policy", "confirmed"), ("intersection", "intersection_confirmed")):
        first = next((o for o in observations if field is None or o[field]), None)
        firsts[name] = None if first is None else {"frame": first["frame"], "since_first_available_scan_s": (first["measurement_timestamp_ns"] - origin_ns) / 1e9,
                    "forward_distance_m": first["distance_m"], "path_relation": first["path_relation"],
                    "confirmed": first["confirmed"], "intersection_confirmed": first["intersection_confirmed"]}
    matrix = []
    for source in plan["counterfactual_frames"]:
        with np.load(trace / "diagnostics" / f"{plan['bag']}_{source:06d}.npz") as arrays:
            support = arrays[f"support_{plan['track_id']}"]
        for target in plan["counterfactual_frames"]:
            points = support if source == target else transform(support, captured[source], captured[target])
            g = geometry(captured[target], config)
            masks, section = g.classify_with_section(points, remove_background=False, include_boundary=True)
            core, _, _, observed, nominal, boundary = masks
            _, _, path_uncertainty = g.path(points[:, 0])
            matrix.append({"support_frame": source, "geometry_frame": target, "support_voxels": len(points),
                           "certified_interior": int(core.sum()), "nominal_interior": int(nominal.sum()),
                           "boundary_uncertain": int(boundary.sum()), "observed": int(observed.sum()),
                           "lateral_min_max_m": [float(section.lateral.min()), float(section.lateral.max())],
                           "path_uncertainty_min_max_m": [float(path_uncertainty.min()), float(path_uncertainty.max())]})
    low, high = np.array(plan["review_roi_min"]), np.array(plan["review_roi_max"])
    membership = []
    for frame in plan["membership_frames"]:
        row = captured[frame]
        with np.load(trace / "diagnostics" / f"{plan['bag']}_{frame:06d}.npz") as arrays:
            raw = arrays["decoded_points"]
            selected = raw[np.all((raw >= low) & (raw <= high), axis=1)]
            components = []
            for obj in row["objects"]:
                support = arrays[f"support_{obj['track_id']}"]
                inside = np.all((support >= low) & (support <= high), axis=1)
                if inside.any():
                    components.append({"support_in_review_roi": int(inside.sum()), **obj})
            membership.append({"frame": frame, "raw_slots_in_review_roi": len(selected),
                               "unique_xyz_in_review_roi": len(np.unique(selected, axis=0)), "components": components})
    report = {"plan": plan, "recipe_sha256": digest(output / "experiment.json"), "reporter_sha256": digest(output / "reporter.py"),
              "run_manifest": json.loads((root / "manifest.json").read_text()),
              "trace_manifest": json.loads((trace / "manifest.json").read_text()),
              "firsts": firsts, "initial_track_observations": observations,
              "static_support_geometry_counterfactual": matrix, "actual_support_membership": membership,
              "identity_status": "Unclassified persistent low group near a separate upright moving group. Correspondence to the organizer's reported foreign object is not established.",
              "limits": "The ROI is a review window, not an object annotation or amodal box. First available scan is left-censored physical visibility. Forward x is not maximum detection range. Counterfactual reclassification assumes static support and estimated registration; structural masks are not reconstructed. A region can merge into a person's component. Whole-scene obstacle status is not object-specific detection evidence. No precision, recall or collision-safety claim."}
    write_json(output / "summary.json", report)
    plot_membership(plan, trace, captured, output)
    print(json.dumps({"firsts": firsts, "counterfactual": matrix,
                      "membership": [{"frame": r["frame"], "raw_slots": r["raw_slots_in_review_roi"], "unique_xyz": r["unique_xyz_in_review_roi"],
                                      "components": [{"track_id": o["track_id"], "roi_voxels": o["support_in_review_roi"], "total_voxels": o["support_voxels"]} for o in r["components"]]} for r in membership]}, indent=2))


if __name__ == "__main__":
    main()
