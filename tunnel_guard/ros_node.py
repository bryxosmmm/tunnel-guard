"""Live ROS 2 subscriber for the rail-relative Tunnel Guard detector.

The node deliberately keeps the detector's safety semantics explicit:
``obstacle`` is a confirmed intersection with the configured reference
envelope, while ``unresolved_obstacle`` is published as an alert on the
boolean topic but remains distinguishable in the JSON result and status.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .detector import Detector, load_config
from .io import decode_cloud

try:  # Keep the offline bag runner importable on machines without ROS 2.
    import rclpy
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point, Quaternion
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Bool, Float32, Header, String
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError as exc:  # pragma: no cover - exercised only outside ROS 2.
    rclpy = None
    Node = object
    _ROS_IMPORT_ERROR = exc


def _json_default(value):
    """Serialize numpy scalar values produced by geometry and ICP diagnostics."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stamp_seconds(stamp) -> float:
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
        raise ValueError("PointCloud2 header contains an invalid ROS time")
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class TunnelGuardNode(Node):
    """Subscribe to PointCloud2, run one stateful detector, and publish evidence."""

    def __init__(self):
        if rclpy is None:  # pragma: no cover - protects accidental non-ROS use.
            raise RuntimeError(
                "The live ROS 2 node requires rclpy and sensor_msgs; use tunnel-guard for offline bags"
            ) from _ROS_IMPORT_ERROR
        super().__init__("tunnel_guard")
        self.declare_parameter("config_path", "/opt/tunnel-guard/configs/detector.json")
        self.declare_parameter("input_topic", "/lidar_points")
        self.declare_parameter("result_topic", "/tunnel_guard/result")
        self.declare_parameter("obstacle_topic", "/tunnel_guard/obstacle")
        self.declare_parameter("nearest_topic", "/tunnel_guard/nearest_obstacle_m")
        self.declare_parameter("cloud_topic", "/tunnel_guard/points")
        self.declare_parameter("markers_topic", "/tunnel_guard/markers")
        self.declare_parameter("output_frame", "tunnel_guard_local")
        self.declare_parameter("publish_cloud", True)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("max_cloud_points", 100_000)

        config_path = Path(str(self.get_parameter("config_path").value)).expanduser()
        self.config = load_config(config_path)
        self.detector = Detector(self.config)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.publish_cloud = bool(self.get_parameter("publish_cloud").value)
        self.publish_markers = bool(self.get_parameter("publish_markers").value)
        self.max_cloud_points = int(self.get_parameter("max_cloud_points").value)
        if self.max_cloud_points < 1:
            raise ValueError("max_cloud_points must be positive")
        self.rotation = np.asarray(self.config["sensor_rotation"], dtype=float)
        self.translation = np.asarray(self.config["sensor_translation"], dtype=float)
        self._sensor_frame: str | None = None
        self._last_measurement_ns: int | None = None
        self._frame_number = 0

        self.result_pub = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), 10
        )
        self.obstacle_pub = self.create_publisher(
            Bool, str(self.get_parameter("obstacle_topic").value), 10
        )
        self.nearest_pub = self.create_publisher(
            Float32, str(self.get_parameter("nearest_topic").value), 10
        )
        self.cloud_pub = (
            self.create_publisher(PointCloud2, str(self.get_parameter("cloud_topic").value), 10)
            if self.publish_cloud else None
        )
        self.marker_pub = (
            self.create_publisher(MarkerArray, str(self.get_parameter("markers_topic").value), 10)
            if self.publish_markers else None
        )
        self.subscription = self.create_subscription(
            PointCloud2, self.input_topic, self._on_cloud, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Listening on {self.input_topic}; config={config_path}; output_frame={self.output_frame}"
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        started = time.perf_counter()
        source_frame = str(message.header.frame_id)
        if not source_frame:
            self.get_logger().error("Dropping PointCloud2 with empty header.frame_id")
            return
        if self._sensor_frame is None:
            self._sensor_frame = source_frame
        elif source_frame != self._sensor_frame:
            self.get_logger().error(
                f"Dropping scan with changed sensor frame {source_frame!r}; expected {self._sensor_frame!r}"
            )
            return
        try:
            timestamp_s = _stamp_seconds(message.header.stamp)
            measurement_ns = _stamp_ns(message.header.stamp)
            if self._last_measurement_ns is not None and measurement_ns < self._last_measurement_ns:
                raise ValueError("acquisition timestamp moved backwards")
            if self._last_measurement_ns is not None and measurement_ns == self._last_measurement_ns:
                self.get_logger().warn("Skipping duplicate PointCloud2 acquisition timestamp")
                return
            points, point_times, invalid_points, scan_duration_s = decode_cloud(
                message, self.rotation, self.translation
            )
            # Unknown per-point time units must never be passed as deskew evidence.
            usable_times = point_times if self.config.get("deskew_enabled", False) and scan_duration_s > 0 else np.empty(0)
            result = self.detector.process(points, timestamp_s, usable_times)
        except (TypeError, ValueError, OverflowError) as exc:
            self.get_logger().error(f"Dropping invalid PointCloud2 scan: {exc}")
            return

        self._last_measurement_ns = measurement_ns
        self._frame_number += 1
        result = result | {
            "frame": self._frame_number - 1,
            "sensor_frame": source_frame,
            "topic": self.input_topic,
            "raw_points": int(message.height * message.width),
            "invalid_points": int(invalid_points),
            "scan_duration_s": float(scan_duration_s),
            "measurement_timestamp_ns": measurement_ns,
            "read_and_process_s": time.perf_counter() - started,
            "source_scan_id": f"{self.input_topic}:{source_frame}:{measurement_ns}",
            "transport": "live_ros2_subscription",
        }
        payload = json.dumps(result, allow_nan=False, default=_json_default)
        self.result_pub.publish(String(data=payload))
        # The boolean is an alert channel: unresolved geometry is intentionally visible
        # to a braking/monitoring consumer, while the JSON status preserves the distinction.
        self.obstacle_pub.publish(Bool(data=result["status"] in ("obstacle", "unresolved_obstacle")))
        nearest = result.get("nearest_obstacle_m")
        self.nearest_pub.publish(Float32(data=float(nearest) if nearest is not None else math.nan))
        if self.cloud_pub is not None:
            # Detector.display_points is the range-filtered, registered frame used by
            # geometry and therefore matches the advertised tunnel_guard_local frame.
            self.cloud_pub.publish(self._cloud_message(message.header, self.detector.display_points))
        if self.marker_pub is not None:
            self.marker_pub.publish(self._marker_message(message.header, result))

    def _cloud_message(self, input_header: Header, points: np.ndarray) -> PointCloud2:
        step = max(1, int(math.ceil(len(points) / self.max_cloud_points)))
        cloud = np.ascontiguousarray(points[::step], dtype="<f4")
        header = Header(stamp=input_header.stamp, frame_id=self.output_frame)
        fields = [
            PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
            for index, name in enumerate(("x", "y", "z"))
        ]
        return PointCloud2(
            header=header,
            height=1,
            width=len(cloud),
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=len(cloud) * 12,
            data=cloud.view(np.uint8).reshape(-1).tobytes(),
            is_dense=True,
        )

    def _marker_message(self, input_header: Header, result: dict) -> MarkerArray:
        header = Header(stamp=input_header.stamp, frame_id=self.output_frame)
        delete = Marker(header=header, ns="tunnel_guard", id=0, action=Marker.DELETEALL)
        markers = [delete]
        for obj in result.get("objects", []):
            minimum = np.asarray(obj["bbox_min"], dtype=float)
            maximum = np.asarray(obj["bbox_max"], dtype=float)
            extent = np.maximum(maximum - minimum, 0.01)
            center = (minimum + maximum) / 2.0
            hazard = obj.get("path_relation") in ("intersecting", "unresolved")
            confirmed = bool(obj.get("confirmed", False))
            marker = Marker(header=header, ns="observed_support", id=int(obj["track_id"]), action=Marker.ADD,
                            type=Marker.CUBE)
            marker.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            marker.scale.x, marker.scale.y, marker.scale.z = map(float, extent)
            marker.color.r, marker.color.g, marker.color.b = (
                (1.0, 0.1, 0.05) if confirmed and hazard else
                ((1.0, 0.8, 0.05) if hazard else (0.5, 0.5, 0.5))
            )
            marker.color.a = 1.0
            marker.lifetime = Duration(sec=0, nanosec=300_000_000)
            markers.append(marker)
            label = Marker(header=header, ns="object_labels", id=100_000 + int(obj["track_id"]),
                           action=Marker.ADD, type=Marker.TEXT_VIEW_FACING)
            label.pose.position = Point(x=float(maximum[0]), y=float(maximum[1]), z=float(maximum[2]))
            label.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            label.scale.z = 0.25
            label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 1.0, 1.0
            label.text = f"#{obj['track_id']} {obj['distance_m']:.1f} m {obj['path_relation']}"
            label.lifetime = Duration(sec=0, nanosec=300_000_000)
            markers.append(label)
        status = Marker(header=header, ns="quality", id=0, action=Marker.ADD,
                        type=Marker.TEXT_VIEW_FACING)
        status.pose.position = Point(x=3.0, y=0.0, z=3.0)
        status.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        status.scale.z = 0.3
        status.color.r, status.color.g, status.color.b, status.color.a = 1.0, 1.0, 1.0, 1.0
        status.text = f"LIVE | {result['status']} | health={result['health']}"
        status.lifetime = Duration(sec=0, nanosec=300_000_000)
        markers.append(status)
        return MarkerArray(markers=markers)


def main(args=None):
    """Console entry point used by ``ros2 run tunnel_guard_ros tunnel_guard_node``."""
    if rclpy is None:  # pragma: no cover - protects accidental non-ROS use.
        raise RuntimeError("ROS 2 Humble is not installed") from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node = TunnelGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
