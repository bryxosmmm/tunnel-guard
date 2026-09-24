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
from .visualization import corridor_edges

try:  # Keep the offline bag runner importable on machines without ROS 2.
    import rclpy
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point, Quaternion
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
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
        self.declare_parameter("config_path", "/opt/tunnel-guard/configs/detector-native.json")
        self.declare_parameter("input_topic", "/lidar_points")
        self.declare_parameter("result_topic", "/tunnel_guard/result")
        self.declare_parameter("obstacle_topic", "/tunnel_guard/obstacle")
        self.declare_parameter("nearest_topic", "/tunnel_guard/nearest_obstacle_m")
        self.declare_parameter("cloud_topic", "/tunnel_guard/points")
        self.declare_parameter("markers_topic", "/tunnel_guard/markers")
        self.declare_parameter("output_frame", "tunnel_guard_local")
        self.declare_parameter("input_timeout_s", 3.0)
        self.declare_parameter("input_reliability", "reliable")
        self.declare_parameter("publish_cloud", True)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("max_cloud_points", 100_000)

        config_path = Path(str(self.get_parameter("config_path").value)).expanduser()
        self.config = load_config(config_path)
        self.detector = Detector(self.config)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        if self.output_frame != "tunnel_guard_local":
            raise ValueError(
                "output_frame must remain tunnel_guard_local: this node publishes detector-local coordinates "
                "and has no surveyed TF/extrinsic transform"
            )
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
        self.input_timeout_s = float(self.get_parameter("input_timeout_s").value)
        if self.input_timeout_s <= 0:
            raise ValueError("input_timeout_s must be positive")
        self.input_reliability = str(self.get_parameter("input_reliability").value)
        if self.input_reliability not in ("reliable", "best_effort"):
            raise ValueError("input_reliability must be reliable or best_effort")
        self._last_received_monotonic = time.monotonic()
        self._watchdog_fired = False
        self._received_scans = 0
        self._invalid_scans = 0
        self._duplicate_scans = 0
        self._out_of_order_scans = 0

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=(ReliabilityPolicy.RELIABLE if self.input_reliability == "reliable"
                         else ReliabilityPolicy.BEST_EFFORT),
        )

        self.result_pub = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), output_qos
        )
        self.obstacle_pub = self.create_publisher(
            Bool, str(self.get_parameter("obstacle_topic").value), output_qos
        )
        self.nearest_pub = self.create_publisher(
            Float32, str(self.get_parameter("nearest_topic").value), output_qos
        )
        self.cloud_pub = (
            self.create_publisher(PointCloud2, str(self.get_parameter("cloud_topic").value), output_qos)
            if self.publish_cloud else None
        )
        self.marker_pub = (
            self.create_publisher(MarkerArray, str(self.get_parameter("markers_topic").value), output_qos)
            if self.publish_markers else None
        )
        self.subscription = self.create_subscription(
            PointCloud2, self.input_topic, self._on_cloud, input_qos
        )
        self.watchdog_timer = self.create_timer(min(self.input_timeout_s, 0.5), self._watchdog)
        self.get_logger().info(
            f"Listening on {self.input_topic}; input QoS={self.input_reliability}, depth=1; "
            f"config={config_path}; output_frame={self.output_frame}"
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        started = time.perf_counter()
        self._received_scans += 1
        self._last_received_monotonic = time.monotonic()
        self._watchdog_fired = False
        source_frame = str(message.header.frame_id)
        if not source_frame:
            self._invalid_scans += 1
            self.detector = Detector(self.config)
            self.get_logger().error("Dropping PointCloud2 with empty header.frame_id")
            self._publish_unavailable("invalid_pointcloud_missing_frame")
            return
        if self._sensor_frame is None:
            self._sensor_frame = source_frame
        elif source_frame != self._sensor_frame:
            self._invalid_scans += 1
            self.detector = Detector(self.config)
            self.get_logger().error(
                f"Dropping scan with changed sensor frame {source_frame!r}; expected {self._sensor_frame!r}"
            )
            self._publish_unavailable("sensor_frame_changed_restart_required")
            return
        try:
            timestamp_s = _stamp_seconds(message.header.stamp)
            measurement_ns = _stamp_ns(message.header.stamp)
            if self._last_measurement_ns is not None and measurement_ns < self._last_measurement_ns:
                self._out_of_order_scans += 1
                raise ValueError("acquisition timestamp moved backwards")
            if self._last_measurement_ns is not None and measurement_ns == self._last_measurement_ns:
                self._duplicate_scans += 1
                self.get_logger().warn("Skipping duplicate PointCloud2 acquisition timestamp")
                return
            points, point_times, invalid_points, scan_duration_s = decode_cloud(
                message, self.rotation, self.translation
            )
            # Unknown per-point time units must never be passed as deskew evidence.
            usable_times = point_times if self.config.get("deskew_enabled", False) and scan_duration_s > 0 else np.empty(0)
            result = self.detector.process(points, timestamp_s, usable_times)
        except (TypeError, ValueError, OverflowError) as exc:
            self._invalid_scans += 1
            self.detector = Detector(self.config)
            self.get_logger().error(f"Dropping invalid PointCloud2 scan: {exc}")
            self._publish_unavailable(f"invalid_pointcloud:{exc}")
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
            "input_received_scans": self._received_scans,
            "processed_scans": self._frame_number,
            "dropped_invalid_scans": self._invalid_scans,
            "dropped_duplicate_scans": self._duplicate_scans,
            "dropped_out_of_order_scans": self._out_of_order_scans,
            "input_loss_count": None,
            "input_loss_note": "DDS has no source sequence counter; depth=1 deliberately retains only the newest queued scan.",
            "input_qos": {"reliability": self.input_reliability, "history": "keep_last", "depth": 1},
            "result_age_s": None,
            "result_age_note": "Not computed: acquisition and host clocks are not proven comparable. callback_to_publish_s is local monotonic elapsed time.",
        }
        self._publish(message.header, result, self.detector.display_points, started)

    def _watchdog(self) -> None:
        elapsed = time.monotonic() - self._last_received_monotonic
        if not self._watchdog_fired and elapsed > self.input_timeout_s:
            self._watchdog_fired = True
            self.detector = Detector(self.config)
            self.get_logger().error(
                f"No PointCloud2 received on {self.input_topic!r} for {elapsed:.3f} s "
                f"(timeout {self.input_timeout_s:g} s); publishing unknown and clearing live display."
            )
            self._publish_unavailable("input_timeout", elapsed)

    def _publish_unavailable(self, reason: str, elapsed_s: float | None = None) -> None:
        result = {
            "status": "unknown",
            "reason": reason,
            "objects": [],
            "nearest_obstacle_m": None,
            "nearest_candidate_m": None,
            "nearest_unresolved_range_m": None,
            "coordinate_frame": self.output_frame,
            "health": "unavailable",
            "health_reasons": [reason],
            "measurement_timestamp_ns": None,
            "transport": "live_ros2_subscription",
            "input_received_scans": self._received_scans,
            "processed_scans": self._frame_number,
            "dropped_invalid_scans": self._invalid_scans,
            "dropped_duplicate_scans": self._duplicate_scans,
            "dropped_out_of_order_scans": self._out_of_order_scans,
            "input_loss_count": None,
            "input_loss_note": "DDS has no source sequence counter; depth=1 deliberately retains only the newest queued scan.",
            "input_qos": {"reliability": self.input_reliability, "history": "keep_last", "depth": 1},
            "time_since_last_received_s": elapsed_s,
            "result_age_s": None,
            "result_age_note": "No current measurement exists; route clearance is unknown.",
        }
        self._publish(Header(frame_id=self.output_frame), result, np.empty((0, 3)), None)

    def _publish(self, header: Header, result: dict, display_points: np.ndarray, started: float | None) -> None:
        if started is not None:
            result["callback_to_publish_s"] = time.perf_counter() - started
        payload = json.dumps(result, allow_nan=False, default=_json_default)
        self.result_pub.publish(String(data=payload))
        # This is an attention signal, not an automated braking command.  Unknown
        # stays visible as attention-required instead of looking like a clear route.
        self.obstacle_pub.publish(Bool(data=result["status"] != "no_obstacle_observed"))
        nearest = result.get("nearest_obstacle_m")
        self.nearest_pub.publish(Float32(data=float(nearest) if nearest is not None else math.nan))
        if self.cloud_pub is not None:
            self.cloud_pub.publish(self._cloud_message(header, display_points))
        if self.marker_pub is not None:
            self.marker_pub.publish(self._marker_message(header, result))

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
        corridor = corridor_edges(result.get("geometry", {}), self.config)
        if len(corridor):
            envelope = Marker(header=header, ns="reference_envelope", id=0, action=Marker.ADD,
                              type=Marker.LINE_LIST)
            envelope.scale.x = 0.035
            envelope.color.r, envelope.color.g, envelope.color.b, envelope.color.a = 0.0, 0.8, 1.0, 0.45
            envelope.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in corridor]
            envelope.lifetime = Duration(sec=0, nanosec=300_000_000)
            markers.append(envelope)
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
        status.text = (f"LIVE | {result['status']} | health={result['health']}\n"
                       f"confirmed={result.get('nearest_obstacle_m')} m | "
                       f"candidate={result.get('nearest_candidate_m')} m | "
                       f"unresolved={result.get('nearest_unresolved_range_m')} m\n"
                       "Observed support and reference contour; route clearance unknown")
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
