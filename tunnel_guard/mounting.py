"""Observe railhead support relative to the sensor; do not calibrate by assertion.

Existing bed/rail estimates only seed candidate selection. The reported mounting
height never selects points or fits the support plane. No result feeds detection.
"""
from __future__ import annotations

import numpy as np


def validate_mounting_config(config):
    recipe = config.get("mounting_observer")
    reference = config.get("installation_reference")
    if recipe is None:
        return
    if not isinstance(recipe, dict) or not isinstance(reference, dict):
        raise ValueError("mounting_observer requires an installation_reference object")
    for name in ("height_above_railheads_m", "lateral_offset_from_track_center_m"):
        if not np.isfinite(reference[name]):
            raise ValueError(f"Nonfinite installation reference: {name}")
    if reference["height_above_railheads_m"] <= 0:
        raise ValueError("Reference railhead height must be positive")
    if not isinstance(reference["recording_applicability_verified"], bool):
        raise ValueError("Reference applicability must be explicitly boolean")
    lo, hi = recipe["fit_range_m"]
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
        raise ValueError("Invalid mounting observation range")
    for name in ("bin_m", "min_span_m", "max_fit_residual_m", "reference_height_band_m", "reference_lateral_band_m"):
        if not np.isfinite(recipe[name]) or recipe[name] <= 0:
            raise ValueError(f"Mounting observer {name} must be positive")
    for name in ("min_points_per_side_bin", "min_upper_points_per_side_bin", "min_paired_bins"):
        if type(recipe[name]) is not int or recipe[name] < 2:
            raise ValueError(f"Mounting observer {name} must be an integer >= 2")
    if not 0 <= recipe["upper_support_quantile"] < 1:
        raise ValueError("Invalid upper support quantile")


def observe_mounting(points, geometry, config):
    recipe = config.get("mounting_observer")
    reference = config.get("installation_reference")
    if not recipe or not reference:
        return None
    result = {
        "state": "unavailable", "reason": geometry.reason,
        "reference_height_m": reference["height_above_railheads_m"],
        "reference_lateral_offset_m": reference["lateral_offset_from_track_center_m"],
        "reference_condition": reference["condition"],
        "reference_applicability_verified": reference["recording_applicability_verified"],
        "vehicle_extrinsics_verified": False,
        "method": "paired_railhead_support_plane_seeded_by_existing_rail_model",
        "support_points": [], "paired_head_representatives": [],
    }
    if not geometry.valid:
        return result
    lo, hi = recipe["fit_range_m"]
    near = points[(points[:, 0] >= lo) & (points[:, 0] < hi) &
                  (np.abs(points[:, 1]) < config["ground_fit_half_width_m"])]
    if not len(near):
        return result | {"reason": "no_near_track_returns"}
    bed, bed_uncertainty = geometry.ground(near)
    center, spacing, path_uncertainty = geometry.path(near[:, 0])
    height = near[:, 2] - bed
    eligible = ((bed_uncertainty <= config["ground_max_uncertainty_m"])
                & (path_uncertainty <= config["path_max_uncertainty_m"])
                & (height > config["rail_height_bounds_m"][0])
                & (height < config["rail_height_bounds_m"][1]))
    pairs, supports = [], []
    for start in np.arange(lo, hi, recipe["bin_m"]):
        in_bin = eligible & (near[:, 0] >= start) & (near[:, 0] < min(start + recipe["bin_m"], hi))
        heads, selected = [], []
        for side in (-1, 1):
            mask = in_bin & (np.abs(near[:, 1] - center - side * spacing / 2) < config["rail_half_width_m"])
            q, h = near[mask], height[mask]
            if len(q) < recipe["min_points_per_side_bin"]:
                break
            head = q[h >= np.quantile(h, recipe["upper_support_quantile"])]
            if len(head) < recipe["min_upper_points_per_side_bin"]:
                break
            heads.append(np.median(head, axis=0))
            selected.append(head)
        if len(heads) == 2:
            pairs.append(heads)
            supports.extend(selected)
    result["paired_bins"] = len(pairs)
    if len(pairs) < recipe["min_paired_bins"]:
        return result | {"reason": "insufficient_paired_head_bins"}
    heads = np.asarray(pairs)
    middle = heads.mean(axis=1)
    span = float(np.ptp(middle[:, 0]))
    result["longitudinal_span_m"] = span
    if span < recipe["min_span_m"]:
        return result | {"reason": "insufficient_paired_head_span"}
    q = heads.reshape(-1, 3)
    design = np.column_stack((q[:, :2], np.ones(len(q))))
    plane, _, rank, _ = np.linalg.lstsq(design, q[:, 2], rcond=None)
    if rank < 3:
        return result | {"reason": "degenerate_head_plane"}
    line_design = np.column_stack((middle[:, 0], np.ones(len(middle))))
    lateral = np.linalg.lstsq(line_design, middle[:, 1], rcond=None)[0]
    up = np.r_[-plane[:2], 1.0]
    up /= np.linalg.norm(up)
    forward = np.array([1., lateral[0], plane[0] + plane[1] * lateral[0]])
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    correction = np.stack((forward, left, up))
    # Configured translation is the raw sensor origin in this processing frame.
    origin = np.asarray(config["sensor_translation"], dtype=float)
    y = lateral[0] * origin[0] + lateral[1]
    track = np.array([origin[0], y, plane[0] * origin[0] + plane[1] * y + plane[2]])
    sensor_height = float(np.dot(up, origin - track))
    sensor_lateral = float(np.dot(left, origin - track))
    vertical_residual = float(np.max(np.abs(q[:, 2] - design @ plane)) / np.linalg.norm(np.r_[plane[:2], 1.]))
    lateral_residual = float(np.max(np.abs(middle[:, 1] - line_design @ lateral)))
    result.update(
        state="observed", reason="supported_railhead_proxy",
        height_above_support_plane_m=sensor_height,
        lateral_offset_from_support_center_m=sensor_lateral,
        height_residual_to_reference_m=sensor_height - result["reference_height_m"],
        lateral_residual_to_reference_m=sensor_lateral - result["reference_lateral_offset_m"],
        max_plane_residual_m=vertical_residual, max_centerline_residual_m=lateral_residual,
        support_plane_z_coefficients=plane.tolist(), centerline_y_coefficients=lateral.tolist(),
        processing_to_local_track_rotation=correction.tolist(),
        head_support_separation_m=float(np.median((heads[:, 1] - heads[:, 0]) @ left)),
        support_points=np.vstack(supports).tolist(), paired_head_representatives=heads.tolist(),
        support_point_count=sum(len(p) for p in supports),
        reference_comparison="within_diagnostic_band" if (
            abs(sensor_height - result["reference_height_m"]) <= recipe["reference_height_band_m"]
            and abs(sensor_lateral - result["reference_lateral_offset_m"]) <= recipe["reference_lateral_band_m"]
        ) else "outside_diagnostic_band",
        diagnostic_bands_m={"height": recipe["reference_height_band_m"], "lateral": recipe["reference_lateral_band_m"]},
        interpretation="Support-plane proxy, not surveyed rail top; head separation is not inner-face gauge. Reported static empty-wagon reference has unknown per-recording applicability.",
    )
    if max(vertical_residual, lateral_residual) > recipe["max_fit_residual_m"]:
        result.update(state="unsupported_fit", reason="near_track_not_straight_or_planar")
    return result
