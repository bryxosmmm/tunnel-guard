"""ROS2 Humble adapter. Measurement time drives inference; wall time detects silence."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from .detector import Detector, load_config
from .io import decode_cloud
from .visualization import ResultMessages


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("tunnel_guard")
        self.declare_parameter("config", "/opt/tunnel-guard/configs/detector-native.json")
        self.declare_parameter("input_topic", "/lidar_points")
        self.declare_parameter("display_max_points", 50000)
        self.declare_parameter("input_timeout_s", 3.0)
        self.declare_parameter("input_reliability", "reliable")
        self.config = load_config(Path(self.get_parameter("config").value))
        self.timeout = float(self.get_parameter("input_timeout_s").value)
        if self.timeout <= 0:
            raise ValueError("input_timeout_s must be positive")
        self.detector = Detector(self.config)
        self.messages = ResultMessages(
            self.config,
            self.get_parameter("display_max_points").value,
            presentation="live_detector",
        )
        self.rotation = np.asarray(self.config["sensor_rotation"])
        self.translation = np.asarray(self.config["sensor_translation"])
        self.kinds = {
            "points_display": ("sensor_msgs/msg/PointCloud2", PointCloud2),
            "debug_markers": ("visualization_msgs/msg/MarkerArray", MarkerArray),
            "status": ("std_msgs/msg/String", String),
        }
        self.publishers_by_topic = {
            topic: self.create_publisher(kind, "/perception/" + topic, 1)
            for topic, (_, kind) in self.kinds.items()
        }
        # A slow detector must not build an unbounded queue of obsolete scans.
        reliability = self.get_parameter("input_reliability").value
        if reliability not in ("reliable", "best_effort"):
            raise ValueError("input_reliability must be reliable or best_effort")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=(
                ReliabilityPolicy.RELIABLE
                if reliability == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
        )
        self.subscription = self.create_subscription(
            PointCloud2, self.get_parameter("input_topic").value, self.on_cloud, qos
        )
        self.last_received = time.monotonic()
        self.last_stamp = None
        self.source_frame = None
        self.processed = 0
        self.duplicates = 0
        self.silent = False
        self.timer = self.create_timer(min(self.timeout, 0.5), self.watchdog)
        self.get_logger().info(
            f"Ready. Reliability={reliability}, queue depth=1; use slow replay for complete evaluation."
        )

    def publish(self, row, points, stamp, support=None):
        payloads = self.messages.build(row, points, stamp, support)
        for topic, message in payloads.items():
            typename, native = self.kinds[topic]
            serialized = self.messages.store.serialize_cdr(message, typename)
            self.publishers_by_topic[topic].publish(
                deserialize_message(bytes(serialized), native)
            )

    def unavailable(self, reason, stamp=0):
        row = {
            "coordinate_frame": "tunnel_guard_local",
            "status": "unknown",
            "objects": [],
            "nearest_obstacle_m": None,
            "health": "unavailable",
            "health_reasons": [reason],
            "processed_scans": self.processed,
            "measurement_timestamp_ns": stamp,
        }
        self.publish(row, np.empty((0, 3)), stamp)

    def watchdog(self):
        if not self.silent and time.monotonic() - self.last_received > self.timeout:
            self.silent = True
            self.detector = Detector(self.config)
            # Keep the acquisition watermark: silence does not turn an old
            # measurement into new evidence. A new epoch requires a restart.
            self.unavailable("input_timeout")

    def on_cloud(self, message):
        received = time.monotonic()
        stamp = message.header.stamp
        ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        try:
            if (
                stamp.sec < 0
                or not 0 <= stamp.nanosec < 1_000_000_000
                or not message.header.frame_id
            ):
                raise ValueError(
                    "invalid acquisition timestamp or missing sensor frame"
                )
            if self.last_stamp == ns:
                self.duplicates += 1
                return
            if self.last_stamp is not None and ns < self.last_stamp:
                raise ValueError(
                    "measurement time moved backwards; restart for a new bag epoch"
                )
            if (
                self.source_frame is not None
                and self.source_frame != message.header.frame_id
            ):
                raise ValueError(
                    "sensor frame changed; reselect calibration and restart"
                )
            points, times, invalid, duration = decode_cloud(
                message, self.rotation, self.translation
            )
            self.last_received = received
            self.silent = False
            row = self.detector.process(points, ns * 1e-9, times)
            self.last_stamp, self.source_frame = ns, message.header.frame_id
            self.processed += 1
            row.update(
                measurement_timestamp_ns=ns,
                sensor_frame=message.header.frame_id,
                input_topic=self.get_parameter("input_topic").value,
                invalid_points=invalid,
                scan_duration_s=duration,
                processed_scans=self.processed,
                skipped_duplicate_scans=self.duplicates,
                input_queue_depth=1,
                input_reliability=self.get_parameter("input_reliability").value,
                input_loss_count=None,
                input_loss_note="DDS latest-scan policy; source has no sequence counter",
                callback_processing_s=time.monotonic() - received,
            )
            if row["callback_processing_s"] > 0.1:
                row["health_reasons"].append("processing_exceeds_10hz_input_period")
                if row["health"] == "normal":
                    row["health"] = "degraded"
            self.publish(
                row, self.detector.display_points, ns, self.detector.display_support
            )
            self.get_logger().info(
                json.dumps(
                    {
                        "scan": self.processed,
                        "stamp_ns": ns,
                        "status": row["status"],
                        "nearest_m": row["nearest_obstacle_m"],
                        "processing_s": row["processing_s"],
                    }
                )
            )
        except (ValueError, RuntimeError) as error:
            self.detector = Detector(self.config)
            self.get_logger().error(str(error))
            self.unavailable(str(error), max(0, ns))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
