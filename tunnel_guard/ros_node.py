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
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import MarkerArray

from .detector import Detector, load_config
from .io import decode_cloud
from .visualization import ResultMessages


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("tunnel_guard")
        self.declare_parameter("config", "/opt/tunnel-guard/configs/detector.json")
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
        # The RViz configuration's fixed frame is `tunnel_guard_local`, and nothing published it: a
        # demonstration would come up with a missing fixed frame and show nothing at all. The frame is
        # real, not invented - it is the sensor frame of this node's own outputs - so it is published as
        # an identity transform to whatever frame the incoming clouds declare, once per source frame.
        # Mounting and extrinsics remain unverified, as the README says; this names the frame the messages
        # are already expressed in rather than claiming a calibration.
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_frames: set = set()
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
            # Say WHY nothing arrived. A bag whose point-cloud topic differs from `input_topic` otherwise
            # looks like a broken detector: the most common cause is a mismatched topic, and our own
            # recordings do not share one - one of them publishes on
            # /sensing/lidar/hesai128/pointcloud while the default is /lidar_points - so the available
            # point-cloud topics are named here instead of leaving the operator to guess.
            available = [name for name, kinds in self.get_topic_names_and_types()
                         if "sensor_msgs/msg/PointCloud2" in kinds]
            self.get_logger().error(
                f"No cloud on {self.get_parameter('input_topic').value!r} for "
                f"{self.timeout:g} s. Point-cloud topics currently available: {available or 'none'}. "
                "Restart with input_topic:=<one of those>."
            )
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
            decode_started = time.monotonic()
            points, times, invalid, duration, attributes = decode_cloud(
                message, self.rotation, self.translation
            )
            decode_s = time.monotonic() - decode_started
            self.last_received = received
            self.silent = False
            row = self.detector.process(points, ns * 1e-9, times, point_attributes=attributes)
            self.last_stamp, self.source_frame = ns, message.header.frame_id
            if message.header.frame_id and message.header.frame_id not in self.broadcast_frames:
                transform = TransformStamped()
                transform.header.stamp = self.get_clock().now().to_msg()
                transform.header.frame_id = "tunnel_guard_local"
                transform.child_frame_id = message.header.frame_id
                transform.transform.rotation.w = 1.0
                self.static_broadcaster.sendTransform(transform)
                self.broadcast_frames.add(message.header.frame_id)
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
                decode_s=decode_s,
                sensor_to_result_s=None,
                latency_scope="Callback entry to result; excludes DDS queue and publication. Sensor/host clock relation unverified.",
                callback_processing_s=time.monotonic() - received,
            )
            if row["callback_processing_s"] > 0.1:
                row["health_reasons"].append("processing_exceeds_10hz_input_period")
                if row["health"] == "normal":
                    row["health"] = "degraded"
            publish_started = time.monotonic()
            self.publish(
                row, self.detector.display_points, ns, self.detector.display_support
            )
            publish_s = time.monotonic() - publish_started
            callback_to_publish_return_s = time.monotonic() - received
            self.get_logger().info(
                json.dumps(
                    {
                        "scan": self.processed,
                        "stamp_ns": ns,
                        "status": row["status"],
                        "nearest_m": row["nearest_obstacle_m"],
                        "processing_s": row["processing_s"],
                        "decode_s": decode_s,
                        "callback_processing_s": row["callback_processing_s"],
                        "publish_s": publish_s,
                        "callback_to_publish_return_s": callback_to_publish_return_s,
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
