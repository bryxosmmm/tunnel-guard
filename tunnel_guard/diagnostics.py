"""Read-only association evidence and observation summaries, never identity ground truth."""
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree


def distribution(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {"count": len(a), **{name: float(np.quantile(a, q)) if len(a) else None
            for name, q in (("min", 0), ("p50", .5), ("p95", .95), ("max", 1))}}


def observation_status(obj):
    if not obj["confirmed"]:
        return "candidate"
    return {"intersecting": "confirmed_obstacle", "unresolved": "unresolved_obstacle",
            "adjacent": "observed_structure"}[obj["path_relation"]]


def association_evidence(objects, tracks, predicted, matched, world, pose, measurement_cov, config, stamp, frame_number):
    """Inspect the actual prediction BEFORE the Kalman update mutates it.

    Nearest alternatives are review leads. A change of numeric ID is not proof of
    physical identity, occlusion, split or merge. No result feeds association.
    """
    ids = list(predicted)
    matched_ids = set(matched.values())
    events = []
    pairs = set()
    if ids and len(objects):
        centers = np.asarray([predicted[k][0][:3] for k in ids])
        distance = np.linalg.norm(centers[:, None, :] - world[None, :, :], axis=2)
        # Every birth gets its nearest active predecessor, including rejected gates.
        pairs.update((ids[int(np.argmin(distance[:, j]))], j) for j in range(len(objects)) if j not in matched)
        # An unmatched predecessor may have been merged into an assigned observation.
        pairs.update((k, int(np.argmin(distance[i]))) for i, k in enumerate(ids) if k not in matched_ids)
        # Reacquisition under the same ID exposes temporal gaps without inventing a switch.
        pairs.update((k, j) for j, k in matched.items() if tracks[k]["history"][-1] < frame_number - 1)
        for key, j in sorted(pairs):
            track = tracks[key]
            state, covariance = predicted[key]
            delta = world[j] - state[:3]
            innovation = covariance[:3, :3] + measurement_cov
            mahalanobis = float(delta @ np.linalg.solve(innovation, delta))
            shape = float(np.linalg.norm(np.log((np.asarray(objects[j]["extent_m"]) + .1) / (track["extent"] + .1))))
            old = track["evidence"][-1][1] + state[:3]
            new = objects[j]["_support_points"] @ pose[:3, :3].T + pose[:3, 3]
            lo, hi = np.maximum(old.min(axis=0), new.min(axis=0)), np.minimum(old.max(axis=0), new.max(axis=0))
            intersection = float(np.prod(np.maximum(0, hi - lo)))
            union = float(np.prod(np.ptp(old, axis=0)) + np.prod(np.ptp(new, axis=0)) - intersection)
            tolerance = config["cluster_voxel_m"]
            old_fraction = float(np.mean(cKDTree(new).query(old)[0] <= tolerance))
            new_fraction = float(np.mean(cKDTree(old).query(new)[0] <= tolerance))
            overlap = old_fraction > 0 or new_fraction > 0
            gates = []
            if mahalanobis > config["tracking_mahalanobis_gate"]:
                gates.append("mahalanobis_gate")
            if shape > config["tracking_extent_log_gate"]:
                gates.append("extent_gate")
            same = matched.get(j) == key
            reason = ("occlusion_or_missed_segmentation" if same else
                      "split" if key in matched_ids and overlap else
                      "merge" if j in matched and overlap else
                      "fragmentation_or_occlusion" if overlap else "new_object_or_unrelated_neighbor")
            events.append({"old_track_id": key, "object_index": j,
                           "old_last_observed_s": track["stamp"], "timestamp_s": stamp,
                           "temporal_gap_s": stamp - track["stamp"],
                           "missing_processed_frames": frame_number - track["history"][-1] - 1,
                           "spatial_gap_m": float(np.linalg.norm(delta)),
                           "association_mahalanobis_squared": mahalanobis, "extent_log_distance": shape,
                           "rejected_by": gates, "old_assigned_elsewhere": key in matched_ids and not same,
                           "predicted_position_sigma_m": np.sqrt(np.diag(covariance)[:3]).tolist(),
                           "innovation_sigma_m": np.sqrt(np.diag(innovation)).tolist(),
                           "bbox_iou_world_predicted_support": intersection / union if union > 0 else None,
                           "old_support_overlap_fraction": old_fraction, "new_support_overlap_fraction": new_fraction,
                           "support_overlap_tolerance_m": tolerance, "possible_cause": reason,
                           "identity_verified": False})
    return {"state": "measured", "matched": len(matched), "births": len(objects) - len(matched),
            "unmatched_active_tracks": len(ids) - len(matched_ids), "events": events,
            "note": "Nearest alternatives only, not exhaustive identity switches. Support is translated by the predicted state in estimated world coordinates; covariance is heuristic."}


def summarize_observations(rows):
    objects = [o for r in rows for o in r["objects"]]
    events = [e for r in rows for e in r.get("tracking_diagnostics", {}).get("events", [])]
    geometry = [r.get("geometry", {}) for r in rows]
    diagnosed = sum("tracking_diagnostics" in r for r in rows)
    bins = {}
    uncertainty = {}
    for g in geometry:
        for sample in g.get("uncertainty_by_range", []):
            uncertainty.setdefault(sample["x_m"], []).append(sample)
    for row in rows:
        for item in row.get("range_observability", []):
            key = tuple(item["range_m"])
            acc = bins.setdefault(key, {"range_m": list(key), "frames": 0, "returns": 0,
                                       "returns_before_geometry_voxel": 0, "geometry_supported_returns": 0})
            acc["frames"] += 1
            for name in ("returns", "returns_before_geometry_voxel", "geometry_supported_returns"):
                acc[name] += item[name]
    return {"frame_observation_status": dict(Counter("confirmed_obstacle" if r["status"] == "obstacle" else r["status"] for r in rows)),
            "object_observation_status": dict(Counter(observation_status(o) for o in objects)),
            "candidate_observations": len(objects), "confirmed_object_observations": sum(o["confirmed"] for o in objects),
            "path_horizon_m": distribution(g.get("path_horizon_m") for g in geometry),
            "gauge_inner_m": distribution(g.get("gauge_inner_median_m") for g in geometry),
            "rail_head_height_above_bed_m": distribution(g.get("rail_head_height_m") for g in geometry),
            "ground_fit_median_residual_m": distribution(g.get("ground_quality", {}).get("median_residual_m") for g in geometry),
            "nearest_obstacle_sensor_x_m": distribution(r["nearest_obstacle_m"] for r in rows),
            "cluster_support_min_x_m": distribution(o.get("cluster_nearest_x_m") for o in objects),
            "distance_support_x_m": distribution(o["distance_support_point"][0] for o in objects if o.get("distance_support_point") is not None),
            "supported_envelope_min_x_m": distribution(o.get("supported_envelope_nearest_x_m") for o in objects),
            "unresolved_envelope_min_x_m": distribution(o.get("unresolved_envelope_nearest_x_m") for o in objects),
            "tracking": {"diagnosed_frames": diagnosed,
                         "births": sum(r.get("tracking_diagnostics", {}).get("births", 0) for r in rows) if diagnosed else None,
                         "diagnostic_processing_ms": distribution(r["tracking_diagnostics"]["processing_s"] * 1000 for r in rows if r.get("tracking_diagnostics", {}).get("processing_s") is not None),
                         "gap_reset_frames": sum(r.get("gap_reset", False) for r in rows),
                         "removed_track_observations": sum(len(r.get("tracking_diagnostics", {}).get("removed_track_ids", [])) for r in rows) if diagnosed else None,
                         "hypotheses_by_cause": dict(Counter(e["possible_cause"] for e in events)),
                         "different_id_hypotheses": sum(e["old_track_id"] != e["new_track_id"] for e in events) if diagnosed else None,
                         "verified_identity_switches": None},
            "range_observability": list(bins.values()),
            "uncertainty_by_range": [{"x_m": x, "frames_with_profile": len(samples),
                "geometry_supported_frames": sum(s["geometry_supported"] for s in samples),
                "path_sigma_m": distribution(s["path_sigma_m"] for s in samples),
                "ground_sigma_m": distribution(s["ground_sigma_m"] for s in samples)}
                for x, samples in sorted(uncertainty.items())],
            "note": "Object counts are correlated frame observations. Alarm counts are unlabeled observations, not false positives. no_obstacle_observed never asserts a clear route. Distances are sensor-frame x, not bumper or along-track distance."}
