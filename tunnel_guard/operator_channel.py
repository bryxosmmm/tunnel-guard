"""Operator review recommendations, never train-control or route-clearance authority."""
from __future__ import annotations


def observation_groups(objects: list[dict], config: dict):
    """The detector's existing partition; no new threshold or object suppression."""
    hazards, doubt, range_unknown = [], [], []
    claim_floor = int(config.get("claim_min_support_voxels", 2))
    isolation = float(config.get("structure_isolation_m", 0.3))
    for obj in objects:
        if obj["path_relation"] in ("intersecting", "unresolved"):
            hazards.append(obj)
            continue
        distance = obj["structure_distance_m"]
        if (obj["boundary_unexplained_voxels"] >= claim_floor
                and (distance is None or distance > isolation)):
            doubt.append(obj)
        if obj["interior_unmeasured_voxels"]:
            range_unknown.append(obj)
    return hazards, doubt, range_unknown


def operator_decision(row: dict, config: dict, *, groups=None) -> dict:
    """One proposal: request human review, retain unresolved evidence, issue no authority.

    Counts are object observations, not independent physical objects or calibrated risk.
    Categories may overlap; the unresolved total counts each observation only once.
    """
    hazards, doubt, range_unknown = (observation_groups(row["objects"], config)
                                   if groups is None else groups)
    certain = [o for o in hazards if o.get("intersection_confirmed", False)]
    unresolved_hazards = [o for o in hazards if not o.get("intersection_confirmed", False)]
    unresolved = {id(o): o for o in unresolved_hazards + doubt + range_unknown}
    categories = {
        "confirmed_boundary": [o for o in unresolved_hazards if o["confirmed"]],
        "tentative_intrusion": [o for o in unresolved_hazards if not o["confirmed"]],
        "isolated_boundary": doubt,
        "unsupported_range": range_unknown,
    }
    unavailable = row.get("health") == "unavailable" or row["status"] == "unknown"
    if certain:
        action, reason = "inspect_obstacle", "confirmed_reference_contour_intrusion"
    elif unresolved:
        action, reason = "inspect_unresolved", "unresolved_observations_require_review"
    elif unavailable or row.get("health") == "degraded":
        action, reason = "check_perception", "perception_unavailable" if unavailable else "perception_degraded"
    else:
        action, reason = "monitor_observations", "no_hazard_observed_not_route_clearance"
    return {
        "policy": "operator_review_only",
        "operator_action": action,
        "reason": reason,
        "movement_authority": "not_issued",
        "braking_decision": "not_computed",
        "train_control_approval": "not_validated",
        "evidence_kind": "uncalibrated_not_probability",
        "adjudication": "not_adjudicated",
        "obstacle_detected": True if certain else (None if unresolved or unavailable else False),
        "nearest_obstacle_m": min((o["distance_m"] for o in certain), default=None),
        "unresolved": {
            "objects": len(unresolved),
            "nearest_m": min((o["distance_m"] for o in unresolved.values()), default=None),
            "categories": {name: {"objects": len(items),
                "nearest_m": min((o["distance_m"] for o in items), default=None)}
                for name, items in categories.items()},
        },
        "distance_origin": row.get("distance_origin", "configured_processing_frame_origin"),
        "distance_kind": "forward_coordinate_not_braking_distance",
        "supported_forward_x_m": row.get("supported_range_m"),
        "route_clearance": "not_established",
        "health": row.get("health", "unavailable"),
        "health_reasons": row.get("health_reasons", []),
        "timing": {"freshness": "unverified", "sensor_to_result_s": None,
                   "reason": "acquisition_to_consumer_age_not_measured"},
    }


def live_timing(decision: dict, callback_elapsed_s: float, input_period_s: float | None):
    """Report a measured lower bound, not sensor age or an invented safe deadline."""
    overrun = input_period_s is not None and callback_elapsed_s > input_period_s
    decision["timing"] = {
        "freshness": "unverified",
        "sensor_to_result_s": None,
        "callback_elapsed_s": callback_elapsed_s,
        "observed_input_period_s": input_period_s,
        "processing_exceeds_input_period": overrun if input_period_s is not None else None,
        "reason": "processing_exceeds_input_period" if overrun else "sensor_host_clock_relation_unverified",
        "scope": "Callback entry to message construction; excludes DDS queue, publication and display",
    }
    if overrun and decision["operator_action"] == "monitor_observations":
        decision.update(operator_action="check_perception", reason="processing_exceeds_input_period")
