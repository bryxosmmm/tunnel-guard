"""Record actual detector results for RViz2 replay without importing ROS runtime."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from .geometry import TrackGeometry


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
    geometry.rail_head_anchors = np.asarray(description.get("rail_head_anchors") or np.empty((0, 3)))
    geometry.rail_sigma = np.asarray(description.get("rail_sigmas_m") or np.empty(0))
    geometry.rail_gauge_inner = np.empty(0)
    geometry.path_growth_m_per_m = description.get("path_growth_m_per_m")
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
            ring.append([x, p[0, 1], ground[0] + geometry.rail_head_profile(np.array([x]))[0] + h * normal])
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


class ResultBag:
    """Bounded display copy; no feedback into detection or temporal state."""

    def __init__(self, path: Path, config: dict, max_points: int = 100000):
        if not isinstance(max_points, int) or max_points < 1:
            raise ValueError("display_max_points must be a positive integer")
        self.config, self.max_points = config, max_points
        self.store = get_typestore(Stores.ROS2_HUMBLE)
        self.writer = Writer(path, version=8)
        self.connections = {}

    def __enter__(self):
        self.writer.open()
        for topic, kind in (("points_display", "sensor_msgs/msg/PointCloud2"),
                            ("debug_markers", "visualization_msgs/msg/MarkerArray"),
                            ("status", "std_msgs/msg/String")):
            self.connections[topic] = self.writer.add_connection("/perception/" + topic, kind, typestore=self.store)
        return self

    def __exit__(self, *args):
        self.writer.close()

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

    def write(self, row: dict, points: np.ndarray, timestamp_ns: int, support: dict | None = None):
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
        corridor = corridor_edges(row.get("geometry", {}), self.config)
        if len(corridor):
            markers.append(self.marker(header, "reference_envelope", 0, 5, corridor))
        for obj in row["objects"]:
            hazard = obj["path_relation"] in ("intersecting", "unresolved")
            color = ((1., .15, .1, 1.) if obj["confirmed"] and hazard else
                     ((1., .8, .1, 1.) if hazard else (.5, .5, .5, 1.)))
            markers.append(self.marker(header, "observed_support", obj["track_id"], 5, box_edges(obj), color))
            if support is not None and obj["track_id"] in support:
                markers.append(self.marker(header, "candidate_measurements", obj["track_id"], 8,
                                           support[obj["track_id"]], color))
            label = f"#{obj['track_id']} {obj['distance_m']:.2f} m | {obj['path_relation']} | {obj['confirmation']} | hits={obj['hits']}"
            markers.append(self.marker(header, "object_labels", obj["track_id"], 9, color=color,
                                       text=label, position=obj["bbox_max"]))
        text = (f"RECORDED RESULT REPLAY | {row['status']} | nearest={row['nearest_obstacle_m']} m\n"
                f"{row['health']}: {', '.join(row['health_reasons'])}\n"
                "Reference envelope; observed support only; route clearance unknown")
        markers.append(self.marker(header, "quality", 0, 9, color=(1., 1., 1., 1.), text=text, position=(4., 0., 2.)))
        payloads = {"points_display": point_message,
                    "debug_markers": m("visualization_msgs/msg/MarkerArray", markers),
                    "status": m("std_msgs/msg/String", json.dumps(row | {"presentation": "recorded_result_replay",
                        "display_points": len(cloud)}, allow_nan=False))}
        for topic, message in payloads.items():
            connection = self.connections[topic]
            self.writer.write(connection, timestamp_ns, self.store.serialize_cdr(message, connection.msgtype))
