"""Attribute real recorded components to reviewed person boxes, not frame statuses."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from .evaluate import box_iou
from .object_evidence_report import rows
from .run import digest, write_json
from .sustech import box_aabb, read_box_files, rotation_matrix
from .trace_analysis import STAGES


def box_geometry(box):
    psr = box["psr"]
    center, scale, angles = [np.array([psr[key][a] for a in "xyz"])
                             for key in ("position", "scale", "rotation")]
    rotation = rotation_matrix(angles)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)]) * scale / 2
    return center, scale, rotation, corners @ rotation.T + center


def in_oriented_box(points, center, scale, rotation):
    return np.all(np.abs((points - center) @ rotation) <= scale / 2, axis=1)


def oriented_iou(obj, center, scale, rotation, epsilon):
    """Intersect the twelve box half-spaces; no padded predictions or yaw removal."""
    low, high = np.array(obj["bbox_min"]), np.array(obj["bbox_max"])
    normals = np.concatenate((np.eye(3), -np.eye(3), rotation.T, -rotation.T))
    limits = np.concatenate((high, -low, rotation.T @ center + scale / 2,
                             -rotation.T @ center + scale / 2))
    triples = np.array(list(combinations(range(12), 3)))
    matrices = normals[triples]
    nonsingular = np.abs(np.linalg.det(matrices)) > epsilon
    vertices = np.linalg.solve(matrices[nonsingular], limits[triples[nonsingular], None])[..., 0]
    vertices = vertices[np.all(vertices @ normals.T <= limits + epsilon, axis=1)]
    intersection = 0.0
    if len(vertices) >= 4 and np.linalg.matrix_rank(vertices - vertices[0], tol=epsilon) == 3:
        intersection = float(ConvexHull(vertices).volume)
    volumes = (float(np.prod(high - low)), float(np.prod(scale)))
    if intersection < 0 or intersection > min(volumes) + epsilon:
        raise ValueError("Invalid real-box intersection volume")
    return intersection / (sum(volumes) - intersection)


def draw_box(ax, corners, axes, color, linestyle="-"):
    for i, j in combinations(range(8), 2):
        if (i ^ j).bit_count() == 1:
            ax.plot(corners[[i, j], axes[0]], corners[[i, j], axes[1]],
                    color=color, linewidth=.8, linestyle=linestyle)


def aabb_corners(obj):
    return np.array([[x, y, z] for x in (obj["bbox_min"][0], obj["bbox_max"][0])
                     for y in (obj["bbox_min"][1], obj["bbox_max"][1])
                     for z in (obj["bbox_min"][2], obj["bbox_max"][2])])


def render_person(batch, output):
    fig, axes = plt.subplots(len(batch), 2, figsize=(12, 3.2 * len(batch)), squeeze=False)
    for panels, entry in zip(axes, batch):
        record, raw, support, corners = entry
        for ax, dims in zip(panels, ((0, 1), (0, 2))):
            ax.scatter(raw[:, dims[0]], raw[:, dims[1]], s=.3, c="gray")
            ax.scatter(support[:, dims[0]], support[:, dims[1]], s=2, c="darkorange")
            draw_box(ax, corners, dims, "royalblue")
            if record["component"]:
                draw_box(ax, aabb_corners(record["component"]), dims, "red", "--")
            ax.set_aspect("equal")
            ax.set_xlabel("xyz"[dims[0]] + " m")
            ax.set_ylabel("xyz"[dims[1]] + " m")
            ax.set_title(f'frame {record["frame"]}, track {record["track_id"]}, '
                         f'points in OBB {record["component_points_inside_obb"]}\n'
                         f'AABB IoU {record["aabb_iou"]:.3f}; oriented IoU {record["oriented_iou"]:.3f}')
    fig.suptitle("REAL measured points: gray raw / orange selected component / blue reviewed full-person box\n"
                 "Red dashed: observed-support AABB. Selection uses point membership, not IoU; inspect ambiguity.")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / f'person_{batch[0][0]["frame"]}.png', dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    run, output = Path(plan["run"]), Path(plan["output"])
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "detector.json").read_text())
    if "finished_unix_s" not in manifest or config["seed"] != plan["seed"]:
        raise ValueError("Require completed real replay with the declared seed")
    recorded = list(rows(run, plan["bag"]))
    if [r["frame"] for r in recorded] != list(range(plan["expected_frames"])):
        raise ValueError("Missing or reordered recording frames")
    annotations = json.loads(Path(plan["annotations"]).read_text())
    labels = read_box_files(Path(plan["source_labels"]))
    if sorted(labels) != plan["label_frames"]:
        raise ValueError("Authored panel changed")
    exported = {f["frame"]: f["objects"][0] for f in annotations["frames"]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", plan)
    shutil.copy2(__file__, output / Path(__file__).name)
    observations, batch, source_hashes = [], [], {}
    for frame in plan["label_frames"]:
        row = recorded[frame]
        source = labels[frame]
        if len(source) != 1 or str(source[0]["obj_id"]) != str(plan["event_id"]):
            raise ValueError("Expected the single authored person")
        box = source[0]
        low, high = box_aabb(box)
        if low != exported[frame]["bbox_min"] or high != exported[frame]["bbox_max"]:
            raise ValueError("Source PSR and exported annotations disagree")
        center, scale, rotation, corners = box_geometry(box)
        diagnostic = run / row["diagnostic_points"]
        source_hashes[str(diagnostic)] = digest(diagnostic)
        with np.load(diagnostic) as arrays:
            points, cluster_labels = arrays["cluster_points"], arrays["cluster_labels"]
            mask = in_oriented_box(points, center, scale, rotation)
            counts = Counter(int(v) for v in cluster_labels[mask] if v >= 0)
            ranked = sorted((o for o in row["objects"] if counts[o["component_id"]]),
                            key=lambda o: (-counts[o["component_id"]], o["component_id"]))
            best = ranked[0] if ranked else None
            support = points[cluster_labels == best["component_id"]] if best else points[:0]
            inside_count = counts[best["component_id"]] if best else 0
            record = {"frame": frame, "source_scan_id": row["source_scan_id"],
                      "event_id": plan["event_id"], "annotation_origin": exported[frame]["annotation_origin"],
                      "component": best, "track_id": best["track_id"] if best else None,
                      "stage_points_in_obb": {s: int(in_oriented_box(arrays[s], center, scale, rotation).sum())
                                              for s in STAGES if s in arrays},
                      "component_points_inside_obb": inside_count,
                      "component_points_total": len(support),
                      "inside_fraction": inside_count / len(support) if len(support) else None,
                      "ranked_component_support": [{"component_id": o["component_id"], "track_id": o["track_id"],
                                                     "points_inside_obb": counts[o["component_id"]]} for o in ranked],
                      "aabb_iou": box_iou(best, exported[frame]) if best else 0.0,
                      "oriented_iou": oriented_iou(best, center, scale, rotation, plan["numerical_epsilon"]) if best else 0.0,
                      "center_error_m": float(np.linalg.norm(np.array(best["center"]) - center)) if best else None,
                      "support_to_full_volume_ratio": float(np.prod(best["extent_m"]) / np.prod(scale)) if best else None}
            observations.append(record)
            raw = arrays["decoded_points"]
            nearby = np.all((raw >= np.array(low) - plan["plot_margin_m"]) &
                            (raw <= np.array(high) + plan["plot_margin_m"]), axis=1)
            batch.append((record, raw[nearby], support, corners))
        if len(batch) == 6 or frame == plan["label_frames"][-1]:
            render_person(batch, output)
            batch.clear()
    by_frame = {r["frame"]: r for r in observations}
    alarms, alarm_tracks = [], defaultdict(list)
    alarms_by_frame = defaultdict(list)
    for row in recorded:
        for obj in row["objects"]:
            if not (obj["confirmed"] and obj["path_relation"] in ("intersecting", "unresolved")):
                continue
            person = by_frame.get(row["frame"])
            attribution = ("unlabelled_frame" if person is None else
                           "selected_person_component" if person["component"] and
                           obj["component_id"] == person["component"]["component_id"] else
                           "other_component_overlaps_person_aabb" if box_iou(obj, exported[row["frame"]]) > 0 else
                           "disjoint_from_person_label")
            alarm = {"frame": row["frame"], "timestamp_s": row["timestamp_s"],
                     "attribution": attribution, "object": obj}
            alarms.append(alarm)
            alarm_tracks[obj["track_id"]].append(alarm)
            alarms_by_frame[row["frame"]].append(alarm)
    for frame, entries in alarms_by_frame.items():
        row = recorded[frame]
        if not row.get("diagnostic_points"):
            continue
        diagnostic = run / row["diagnostic_points"]
        source_hashes[str(diagnostic)] = digest(diagnostic)
        with np.load(diagnostic) as arrays:
            fields = {key: arrays[key] for key in
                      ("cluster_points", "cluster_ring", "cluster_raw_time", "cluster_intensity",
                       "cluster_source_indices", "cluster_ring_valid", "cluster_raw_time_valid",
                       "cluster_intensity_valid")}
            core, component_ids = arrays["cluster_core"], arrays["cluster_labels"]
            for alarm in entries:
                mask = core & (component_ids == alarm["object"]["component_id"])
                alarm["certified_witnesses"] = {key: value[mask].tolist() for key, value in fields.items()}
                alarm["certified_witness_ring_count"] = len(np.unique(fields["cluster_ring"][mask]))
    write_json(output / "person_observations.json", observations)
    write_json(output / "alarm_observations.json", alarms)
    track_summary = []
    for track, entries in alarm_tracks.items():
        track_summary.append({"track_id": track, "observations": len(entries),
                              "first_frame": entries[0]["frame"], "last_frame": entries[-1]["frame"],
                              "center_min": np.min([e["object"]["center"] for e in entries], axis=0).tolist(),
                              "center_max": np.max([e["object"]["center"] for e in entries], axis=0).tolist(),
                              "attribution": dict(Counter(e["attribution"] for e in entries))})
    for frame in plan["alarm_review_frames"]:
        row = recorded[frame]
        with np.load(run / row["diagnostic_points"]) as arrays:
            raw = arrays["decoded_points"]
            roi = np.all((raw >= plan["alarm_roi_min"]) & (raw <= plan["alarm_roi_max"]), axis=1)
            raw = raw[roi]
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax, dims in zip(axes, ((0, 1), (0, 2), (1, 2))):
                ax.scatter(raw[:, dims[0]], raw[:, dims[1]], s=1, c="gray")
                for alarm in (a for a in alarms if a["frame"] == frame):
                    obj = alarm["object"]
                    support = arrays["cluster_points"][arrays["cluster_labels"] == obj["component_id"]]
                    ax.scatter(support[:, dims[0]], support[:, dims[1]], s=10, c="red")
                    draw_box(ax, aabb_corners(obj), dims, "red")
                ax.set_xlim(plan["alarm_roi_min"][dims[0]], plan["alarm_roi_max"][dims[0]])
                ax.set_ylim(plan["alarm_roi_min"][dims[1]], plan["alarm_roi_max"][dims[1]])
                ax.set_aspect("equal")
                ax.set_xlabel("xyz"[dims[0]] + " m")
                ax.set_ylabel("xyz"[dims[1]] + " m")
            fig.suptitle(f'Real far-field review frame {frame}: gray raw returns / red confirmed hazard support\n'
                         'Fixed ROI, not another labelled person; no physical identity inferred from shape alone.')
            fig.tight_layout(rect=(0, 0, 1, .90))
            fig.savefig(output / f"alarm_{frame}.png", dpi=110)
            plt.close(fig)
    threshold = annotations["minimum_iou"]
    non_target = [a for a in alarms if a["attribution"] == "disjoint_from_person_label"]
    center_errors = [o["center_error_m"] for o in observations if o["component"]]
    summary = {"scope": plan["scene_truth"], "manifest": manifest,
               "person_observations": len(observations),
               "frames_with_measured_component": sum(o["component"] is not None for o in observations),
               "presence_confirmed_frames": sum(bool(o["component"] and o["component"]["presence_confirmed"]) for o in observations),
               "person_intersection_confirmed_frames": sum(bool(o["component"] and o["component"]["intersection_confirmed"]) for o in observations),
               "person_tracks": dict(Counter(o["track_id"] for o in observations)),
               "person_id_switches": sum(a["track_id"] != b["track_id"] for a, b in zip(observations, observations[1:])
                                        if a["component"] and b["component"]),
               "person_relations": dict(Counter(o["component"]["path_relation"] if o["component"] else "missing" for o in observations)),
               "aabb_iou_pass_frames": sum(o["aabb_iou"] >= threshold for o in observations),
               "oriented_iou_pass_frames": sum(o["oriented_iou"] >= threshold for o in observations),
               "iou_threshold_unchanged": threshold,
               "center_error_median_m": float(np.median(center_errors)) if center_errors else None,
               "alarms": len(alarms), "alarm_frames": len({a["frame"] for a in alarms}),
               "alarm_attribution": dict(Counter(a["attribution"] for a in alarms)),
               "alarm_tracks": track_summary,
               "alarm_reasons": dict(Counter(a["object"]["path_relation_reason"] for a in alarms)),
               "alarm_without_unexplained_interior_support": sum(a["object"]["certified_unexplained_voxels"] == 0 for a in alarms),
               "labelled_non_target_observations": len(non_target),
               "labelled_non_target_frames": len({a["frame"] for a in non_target}),
               "labelled_non_target_reasons": dict(Counter(a["object"]["path_relation_reason"] for a in non_target)),
               "labelled_non_target_core_counts": dict(Counter(a["object"]["in_envelope_voxels"] for a in non_target)),
               "labelled_non_target_ring_counts": dict(Counter(a["certified_witness_ring_count"] for a in non_target)),
               "labelled_non_target_without_unexplained_support": sum(a["object"]["certified_unexplained_voxels"] == 0 for a in non_target),
               "provenance": {"annotations_sha256": digest(Path(plan["annotations"])),
                              "original_labels_sha256": {p.name: digest(p) for p in sorted(Path(plan["source_labels"]).glob("*.json"))},
                              "diagnostic_sha256": source_hashes, "recorded_rows_sha256": digest(run / f'{plan["bag"]}.jsonl'),
                              "reporter_sha256": digest(Path(__file__))},
               "limits": ["User states one target obstacle, a person; this does not supply unlabelled-frame person positions.",
                          "All 36 positions were author-reviewed after detector-assisted propagation; not 36 independent events.",
                          "OBB membership includes background and is a correspondence diagnostic, not a semantic point label or recall score.",
                          "Full-person volume and visible-return support have different box semantics; oriented IoU does not remove that mismatch.",
                          "Disjoint confirmed alarms on labelled frames are non-target alarms under the supplied scene truth, not proof of safe physical clearance.",
                          "Track IDs do not prove physical identity outside the reviewed panel; no arbitrary matching radius or relabelling of unknown frames."]}
    write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("manifest", "provenance", "alarm_tracks")}, indent=2))


if __name__ == "__main__":
    main()
