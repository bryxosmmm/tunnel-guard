"""Record actual detector results for RViz2 replay without importing ROS runtime."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from .geometry import TrackGeometry


def distance_summary(row: dict) -> dict:
    """Do not present an uncertain nearby object as a confirmed intrusion."""
    hazards = [obj for obj in row["objects"] if obj["path_relation"] in ("intersecting", "unresolved")]
    return {
        "confirmed_intersection_m": min((obj["distance_m"] for obj in hazards
            if obj.get("intersection_confirmed", False)), default=None),
        "unresolved_confirmed_m": min((obj["distance_m"] for obj in hazards
            if obj["confirmed"] and not obj.get("intersection_confirmed", False)), default=None),
        "tentative_m": min((obj["distance_m"] for obj in hazards if not obj["confirmed"]), default=None),
        "geometry_unknown_m": min((obj["cluster_nearest_x_m"] for obj in row["objects"]
                                  if obj["path_relation"] == "unknown"), default=None),
        "method": row.get("distance_method", "unavailable"),
        "origin": row.get("distance_origin", "configured_processing_frame_origin"),
    }


def corridor_edges(description: dict, config: dict) -> np.ndarray:
    """Draw the classifier's reference contour only where geometry is supported."""
    if not description.get("valid"):
        return np.empty((0, 3))
    geometry = object.__new__(TrackGeometry)
    geometry.config = config
    geometry.plane = np.asarray(description["ground_plane"])
    geometry.ground_anchors = np.asarray(description["ground_anchors"])
    geometry.rail_anchors = np.asarray(description["rail_anchors"])
    geometry.rail_head_height_m = description["rail_head_height_m"]
    geometry.rail_support_diagnostics = description.get("rail_support_diagnostics", [])
    segments = np.asarray(config["envelope_segments_m"])
    # Retain both sides of width discontinuities at adjacent segment boundaries.
    contour = [(float(h), float(w + config["envelope_margin_m"]))
               for low, high, wlow, whigh in segments for h, w in ((low, wlow), (high, whigh))]
    contour = [(h, -w) for h, w in contour] + [(h, w) for h, w in reversed(contour)]
    edges, previous = [], None
    slope = geometry.plane[1]
    normal = np.sqrt(1 + np.sum(geometry.plane[:2] ** 2))
    for x in np.arange(config["min_forward_m"], config["max_range_m"], 2.0):
        center, _, path_uncertainty = geometry.path(np.array([x]))
        ring = []
        supported = path_uncertainty[0] <= config["path_max_uncertainty_m"]
        for h, lateral in contour:
            dy = (lateral * np.sqrt(1 + slope**2) - slope * h * normal) / (1 + slope**2)
            p = np.array([[x, center[0] + dy, 0.0]])
            ground, uncertainty = geometry.ground(p)
            supported &= uncertainty[0] <= config["ground_max_uncertainty_m"]
            ring.append([x, p[0, 1], ground[0] + geometry.rail_head_height_m + h * normal])
        if not supported:
            previous = None
            continue
        ring = np.asarray(ring)
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            edges.extend((a, b))
        if previous is not None:
            for a, b in zip(previous, ring):
                edges.extend((a, b))
        previous = ring
    return np.asarray(edges).reshape(-1, 3)


def box_edges(obj: dict) -> np.ndarray:
    corners = np.asarray(list(product(*zip(obj["bbox_min"], obj["bbox_max"]))))
    return np.asarray([corners[j] for i in range(8) for bit in (1, 2, 4)
                       if i < (i ^ bit) for j in (i, i ^ bit)])


class ResultMessages:
    """Shared display messages for recorded replay and the live ROS adapter."""

    def __init__(self, config: dict, max_points: int = 100000, *, presentation="recorded_result_replay"):
        if not isinstance(max_points, int) or max_points < 1:
            raise ValueError("display_max_points must be a positive integer")
        self.config, self.max_points = config, max_points
        self.presentation = presentation
        self.store = get_typestore(Stores.ROS2_HUMBLE)

    def message(self, kind, *args, **kwargs):
        return self.store.types[kind](*args, **kwargs)

    def marker(self, header, ns, identity, kind, xyz=(), color=(0., .8, 1., .45), text="", position=(0., 0., 0.), action=0):
        m = self.message
        empty = np.empty(0, dtype=np.uint8)
        return m("visualization_msgs/msg/Marker", header=header, ns=ns, id=identity, type=kind, action=action,
                 pose=m("geometry_msgs/msg/Pose", m("geometry_msgs/msg/Point", *map(float, position)),
                        m("geometry_msgs/msg/Quaternion", 0., 0., 0., 1.)),
                 scale=m("geometry_msgs/msg/Vector3", .035, .035, .35 if kind == 9 else .035),
                 color=m("std_msgs/msg/ColorRGBA", *color),
                 lifetime=m("builtin_interfaces/msg/Duration", 0, 300000000), frame_locked=False,
                 points=[m("geometry_msgs/msg/Point", *map(float, p)) for p in xyz], colors=[],
                 texture_resource="", texture=m("sensor_msgs/msg/CompressedImage", header, "", empty),
                 uv_coordinates=[], text=text, mesh_resource="",
                 mesh_file=m("visualization_msgs/msg/MeshFile", "", empty), mesh_use_embedded_materials=False)

    def build(self, row: dict, points: np.ndarray, timestamp_ns: int, support: dict | None = None):
        m = self.message
        header = m("std_msgs/msg/Header", m("builtin_interfaces/msg/Time", *divmod(timestamp_ns, 1000000000)),
                   row["coordinate_frame"])
        # Only the display copy is sampled. Detector input and boxes stay unchanged.
        step = max(1, int(np.ceil(len(points) / self.max_points)))
        cloud = np.ascontiguousarray(points[::step], dtype="<f4")
        fields = [m("sensor_msgs/msg/PointField", name, i * 4, 7, 1) for i, name in enumerate(("x", "y", "z"))]
        point_message = m("sensor_msgs/msg/PointCloud2", header, 1, len(cloud), fields, False, 12,
                          len(cloud) * 12, cloud.view(np.uint8).reshape(-1), True)
        markers = [self.marker(header, "clear", 0, 5, action=3)]
        mounting = row.get("mounting")
        if mounting and mounting.get("support_points"):
            markers.append(self.marker(header, "railhead_support_for_mounting", 0, 8,
                                       mounting["support_points"], (.1, 1., .6, 1.)))
        corridor = corridor_edges(row.get("geometry", {}), self.config)
        if len(corridor):
            markers.append(self.marker(header, "reference_envelope", 0, 5, corridor))
        for obj in row["objects"]:
            hazard = obj["path_relation"] in ("intersecting", "unresolved")
            intersection_confirmed = obj.get("intersection_confirmed", obj["confirmed"] and obj["path_relation"] == "intersecting")
            color = ((1., .15, .1, 1.) if intersection_confirmed else
                     (1., .55, .05, 1.) if obj["confirmed"] and hazard else
                     ((1., .8, .1, 1.) if hazard or obj["path_relation"] == "unknown" else (.5, .5, .5, 1.)))
            markers.append(self.marker(header, "observed_support", obj["track_id"], 5, box_edges(obj), color))
            if support is not None and obj["track_id"] in support:
                markers.append(self.marker(header, "candidate_measurements", obj["track_id"], 8,
                                           support[obj["track_id"]], color))
            label = f"#{obj['track_id']} {obj['distance_m']:.2f} m | {obj['path_relation']} | {obj['confirmation']} | hits={obj['hits']}"
            if "intersection_confirmation" in obj:
                label += f" | intrusion={obj['intersection_confirmation']} | inside_hits={obj['intersection_hits']}"
            if obj.get("boundary_uncertain_voxels", 0):
                label += f" | boundary={obj['boundary_uncertain_voxels']}"
            if hazard and "distance_support_point" in obj:
                markers.append(self.marker(header, "distance_witness", obj["track_id"], 8,
                                           [obj["distance_support_point"]], (0., 1., 1., 1.)))
                label += f" | {obj['distance_method']}"
            markers.append(self.marker(header, "object_labels", obj["track_id"], 9, color=color,
                                       text=label, position=obj["bbox_max"]))
        distances = distance_summary(row)
        text = (f"{self.presentation.upper()} | {row['status']}\n"
                f"Intrusion={distances['confirmed_intersection_m']} m | uncertain={distances['unresolved_confirmed_m']} m\n"
                f"Object without corridor relation={distances['geometry_unknown_m']} m\n"
                f"{row['health']}: {', '.join(row['health_reasons'])}\n"
                "Reference envelope; observed support only; route clearance unknown")
        if mounting:
            measured = mounting.get("height_above_support_plane_m") if mounting["state"] == "observed" else None
            estimate = "unavailable" if measured is None else f"{measured:.3f} m"
            text += (f"\nRail support height: {estimate}; reported static height: "
                     f"{mounting['reference_height_m']:.3f} m"
                     "\nRecording applicability unverified; vehicle calibration unverified")
        markers.append(self.marker(header, "quality", 0, 9, color=(1., 1., 1., 1.), text=text, position=(4., 0., 2.)))
        payloads = {"points_display": point_message,
                    "debug_markers": m("visualization_msgs/msg/MarkerArray", markers),
                    "status": m("std_msgs/msg/String", json.dumps(row | {"presentation": self.presentation,
                        "display_points": len(cloud), "distance_summary": distances}, allow_nan=False)),
                    "attention_required": m("std_msgs/msg/Bool", row["status"] != "no_obstacle_observed"),
                    "nearest_obstacle_m": m("std_msgs/msg/Float32", float(
                        row["nearest_obstacle_m"] if row["nearest_obstacle_m"] is not None else np.nan)),
                    }
        return payloads


class ResultBag(ResultMessages):
    """Write the same display messages to a ROS bag without a ROS runtime."""

    def __init__(self, path: Path, config: dict, max_points: int = 100000):
        super().__init__(config, max_points)
        self.writer = Writer(path, version=8)
        self.connections = {}

    def __enter__(self):
        self.writer.open()
        for topic, kind in (("points_display", "sensor_msgs/msg/PointCloud2"),
                            ("debug_markers", "visualization_msgs/msg/MarkerArray"),
                            ("status", "std_msgs/msg/String"),
                            ("attention_required", "std_msgs/msg/Bool"),
                            ("nearest_obstacle_m", "std_msgs/msg/Float32")):
            self.connections[topic] = self.writer.add_connection("/perception/" + topic, kind, typestore=self.store)
        return self

    def __exit__(self, *args):
        self.writer.close()

    def write(self, row: dict, points: np.ndarray, timestamp_ns: int, support: dict | None = None):
        payloads = self.build(row, points, timestamp_ns, support)
        for topic, message in payloads.items():
            connection = self.connections[topic]
            self.writer.write(connection, timestamp_ns, self.store.serialize_cdr(message, connection.msgtype))


def replay_frame(run: Path, bag: str, frame: int):
    """Read actual saved messages for a static review; does not run inference."""
    from rosbags.rosbag2 import Reader
    with (run / f"{bag}.jsonl").open() as stream:
        row = next(json.loads(line) for line in stream if json.loads(line)["frame"] == frame)
    store = get_typestore(Stores.ROS2_HUMBLE)
    cloud, markers = None, None
    stamp = row["measurement_timestamp_ns"]
    with Reader(run / f"{bag}_rviz") as reader:
        for connection, _, raw in reader.messages(start=stamp, stop=stamp + 1):
            if connection.topic.endswith("points_display"):
                message = store.deserialize_cdr(raw, connection.msgtype)
                cloud = np.frombuffer(message.data, dtype="<f4").reshape(-1, 3).copy()
            elif connection.topic.endswith("debug_markers"):
                markers = store.deserialize_cdr(raw, connection.msgtype).markers
    if cloud is None or markers is None:
        raise ValueError(f"Missing recorded visualization for {bag}:{frame}")
    return row, cloud, markers


def render_comparison(before: Path, after: Path, cases: dict, output: Path):
    """Same-frame, same-scale PNGs from the existing ResultBag path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    output.mkdir(parents=True, exist_ok=False)
    for index, case in enumerate(cases["cases"]):
        fig, axes = plt.subplots(3, 2, figsize=(16, 11), layout="constrained")
        fig.suptitle(f"{case['bag']} / frame {case['frame']} — {case['title']}\nRecorded real measurements; provisional annotation is not verified hazard ground truth", fontsize=13)
        for column, (name, run) in enumerate((("BEFORE", before), ("AFTER", after))):
            row, cloud, markers = replay_frame(run, case["bag"], case["frame"])
            for level, dimension in enumerate((1, 1, 2)):
                ax = axes[level, column]
                limits = case["overview_xy"] if level == 0 else case["zoom_xy"] if level == 1 else case["zoom_xz"]
                x0, x1, y0, y1 = limits
                visible = ((cloud[:, 0] >= x0) & (cloud[:, 0] <= x1)
                           & (cloud[:, dimension] >= y0) & (cloud[:, dimension] <= y1))
                q = cloud[visible]
                ax.scatter(q[:, 0], q[:, dimension], s=.5, c="#a6abb0", rasterized=True)
                for marker in markers:
                    xyz = np.array([[p.x, p.y, p.z] for p in marker.points]).reshape(-1, 3)
                    color = (marker.color.r, marker.color.g, marker.color.b)
                    if marker.type == 5 and len(xyz) and marker.ns in ("reference_envelope", "observed_support"):
                        edges = xyz[:, [0, dimension]].reshape(-1, 2, 2)
                        ax.add_collection(LineCollection(edges, colors=[color], linewidths=.6, alpha=.7))
                    elif marker.type == 8 and len(xyz) and marker.ns == "candidate_measurements" and level > 0:
                        ax.scatter(xyz[:, 0], xyz[:, dimension], s=3, c=[color], rasterized=True)
                target = next((o for o in row["objects"] if o["component_id"] == case.get("component_id")), None)
                if target is not None and level > 0:
                    old = target["cluster_nearest_x_m"]
                    supported = target["supported_envelope_nearest_x_m"]
                    ax.axvline(old, c="#ad4bbc", linestyle=":", linewidth=1.3, label=f"cluster min x: {old:.3f} m")
                    if supported is not None:
                        ax.axvline(supported, c="#007e95", linestyle="--", linewidth=1.3,
                                   label=f"supported min x: {supported:.3f} m")
                    ax.legend(loc="upper right", fontsize=8)
                if "annotation_box" in case and level > 0:
                    box = case["annotation_box"]
                    lo, hi = box["bbox_min"], box["bbox_max"]
                    ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]],
                            [lo[dimension], lo[dimension], hi[dimension], hi[dimension], lo[dimension]],
                            c="#ad4bbc", linestyle="--", linewidth=1, label="provisional box")
                ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="forward x [m]",
                       ylabel="lateral y [m]" if dimension == 1 else "processing-frame z [m]")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=.2)
                if level == 0:
                    # `nearest_obstacle_m` is None whenever the frame has no confirmed hazard, which
                    # is most frames now that the claim requires a measured coordinate.
                    nearest = ("none" if row["nearest_obstacle_m"] is None
                               else f"{row['nearest_obstacle_m']:.3f} m")
                    candidate = ("none" if row.get("nearest_candidate_m") is None
                                 else f"{row['nearest_candidate_m']:.3f} m")
                    ax.set_title(f"{name} — {row['status']}; nearest hazard={nearest}, nearest candidate={candidate}"
                                 f"\nhealth={row['health']}; confirmed denotes algorithmic evidence", fontsize=10)
                elif level == 1:
                    detail = (f"component {target['component_id']}: reported={target['distance_m']:.3f} m; {target['path_relation']}"
                              if target is not None else "Provisional structure region; observed support only")
                    ax.set_title(detail, fontsize=10)
        fig.savefig(output / f"case_{index:02d}_frame_{case['frame']:06d}.png", dpi=140)
        plt.close(fig)
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Render matched PNG comparisons from actual ResultBag exports")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render_comparison(a.before, a.after, json.loads(a.cases.read_text()), a.output)
    print(f"Recorded-result comparisons: {a.output}")


if __name__ == "__main__":
    main()
