"""Chronological mounting consistency report from real, already-recorded observations.

Freeze local-track orientation on a prefix; later measurements never modify it.
This report cannot certify vehicle extrinsics or silently install a transform.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .run import digest, environment, write_json


def partition_stats(records, rotation, recipe):
    valid = [r for r in records if r["mounting"]["state"] == "observed"]
    result = {"frames": len(records), "observed_frames": len(valid),
              "observed_fraction": len(valid) / len(records) if records else 0.,
              "states": dict(Counter(r["mounting"]["state"] for r in records))}
    if not valid:
        return result
    h = np.array([r["mounting"]["height_above_support_plane_m"] for r in valid])
    y = np.array([r["mounting"]["lateral_offset_from_support_center_m"] for r in valid])
    result.update(height_m={"p05": float(np.quantile(h, .05)), "p50": float(np.median(h)),
                            "p95": float(np.quantile(h, .95)), "span": float(np.ptp(h))},
                  lateral_m={"p05": float(np.quantile(y, .05)), "p50": float(np.median(y)),
                             "p95": float(np.quantile(y, .95)), "span": float(np.ptp(y))},
                  reference_comparisons=dict(Counter(r["mounting"]["reference_comparison"] for r in valid)))
    if rotation is not None:
        local = Rotation.from_matrix([r["mounting"]["processing_to_local_track_rotation"] for r in valid])
        result["max_rotation_deviation_deg"] = float(np.max((local * rotation.inv()).magnitude()) * 180 / np.pi)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    source = Path(recipe["run"])
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "detector.json").read_text())
    write_json(output / "experiment.json", recipe)
    write_json(output / "manifest.json", environment() | {
        "run_manifest_sha256": digest(source / "manifest.json"),
        "detector_sha256": digest(source / "detector.json"),
        "report_source_sha256": digest(Path(__file__)),
    })
    summaries, plot_rows = [], []
    for name in recipe["bags"]:
        path = source / f"{name}.jsonl"
        records = []
        watermark = None
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                stamp = row["measurement_timestamp_ns"]
                if watermark is not None and stamp <= watermark:
                    raise ValueError("Repeated or noncausal measurement in mounting report")
                watermark = stamp
                records.append({"frame": row["frame"], "measurement_timestamp_ns": stamp,
                                "mounting": row["mounting"]})
        if len(records) <= recipe["fit_frames"]:
            raise ValueError("Need observations strictly after the fitting prefix")
        fit = records[:recipe["fit_frames"]]
        heldout = records[recipe["fit_frames"]:]
        rotations = [r["mounting"]["processing_to_local_track_rotation"] for r in fit
                     if r["mounting"]["state"] == "observed"]
        frozen = Rotation.from_matrix(rotations).mean() if rotations else None
        partitions = {"fit": partition_stats(fit, frozen, recipe),
                      "later": partition_stats(heldout, frozen, recipe),
                      "all": partition_stats(records, frozen, recipe)}
        failures = []
        for label in ("fit", "later"):
            values = partitions[label]
            if values["observed_fraction"] < recipe["min_observed_fraction"]:
                failures.append(label + ":insufficient_observed_frames")
            if values.get("max_rotation_deviation_deg", float("inf")) > recipe["max_rotation_deviation_deg"]:
                failures.append(label + ":orientation_instability")
            if values.get("height_m", {}).get("span", float("inf")) > recipe["max_height_span_m"]:
                failures.append(label + ":height_instability")
            if values.get("lateral_m", {}).get("span", float("inf")) > recipe["max_lateral_span_m"]:
                failures.append(label + ":lateral_instability")
            if values.get("reference_comparisons", {}).get("outside_diagnostic_band", 0):
                failures.append(label + ":reported_reference_disagreement")
        summary = {"bag": name, "source_sha256": digest(path), "partitions": partitions,
                   "fit_cutoff_measurement_ns": fit[-1]["measurement_timestamp_ns"],
                   "later_first_measurement_ns": heldout[0]["measurement_timestamp_ns"],
                   "numerical_consistency_accepted": not failures, "rejection_reasons": failures,
                   "frozen_processing_to_track_rotation": frozen.as_matrix().tolist() if frozen else None,
                   "frozen_xyz_deg": frozen.as_euler("xyz", degrees=True).tolist() if frozen else None,
                   "calibration_installed": False, "vehicle_extrinsics_verified": False,
                   "installation_blockers": ["reference_recording_load_and_stationarity_unverified",
                                             "railhead_support_is_not_independently_surveyed_geometry",
                                             "longitudinal_vehicle_offset_and_body_envelope_unknown"],
                   "limitations": ["Existing rail model seeds candidates; no independent rail-identity proof.",
                                   "Near track must be locally straight/planar; curvature/cant changes can invalidate a fixed orientation.",
                                   "Upper observed support can be below the physical running surface.",
                                   "Reported height direction/uncertainty unspecified; plane-normal height is a proxy.",
                                   "Temporal frames correlated; ranges are not statistical confidence bounds."]}
        write_json(output / f"{name}.json", summary)
        summaries.append(summary)
        plot_rows.append((name, records))
        print(json.dumps({"bag": name, "all": partitions["all"], "rejection_reasons": failures}), flush=True)
    write_json(output / "summary.json", summaries)
    if recipe.get("plot", False):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(len(plot_rows), 2, figsize=(12, 3 * len(plot_rows)), squeeze=False)
        reference = config["installation_reference"]
        heights = [r["mounting"]["height_above_support_plane_m"] for _, rows in plot_rows for r in rows
                   if r["mounting"]["state"] == "observed"] + [reference["height_above_railheads_m"]]
        height_limits = (min(heights) - .03, max(heights) + .03)
        for panel, (name, records) in enumerate(plot_rows):
            observed = [r for r in records if r["mounting"]["state"] == "observed"]
            unsupported = [r for r in records if r["mounting"]["state"] != "observed"]
            for column, (key, nominal, label) in enumerate([
                    ("height_above_support_plane_m", reference["height_above_railheads_m"], "Support-plane height [m]"),
                    ("lateral_offset_from_support_center_m", reference["lateral_offset_from_track_center_m"], "Sensor lateral offset [m]")]):
                axis = axes[panel, column]
                axis.scatter([r["frame"] for r in observed], [r["mounting"][key] for r in observed], s=6, label="Measured support proxy")
                axis.axhline(nominal, color="darkorange", linestyle="--", label="Reported empty/stationary reference")
                axis.axvline(records[recipe["fit_frames"]]["frame"], color="grey", linestyle=":", label="Frozen prefix cutoff")
                if unsupported:
                    axis.plot([r["frame"] for r in unsupported], [nominal] * len(unsupported), "rx", ms=3, label="Unsupported (not a height estimate)")
                axis.set(title=name, xlabel="Recorded frame", ylabel=label)
                axis.grid(alpha=.2)
                if column == 0:
                    axis.set_ylim(*height_limits)
                if panel == 0:
                    axis.legend(fontsize=7)
        figure.suptitle("Real recordings: railhead support vs reported mounting — applicability unverified", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, .97))
        figure.savefig(output / "mounting-observations.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
