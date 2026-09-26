"""ROS2 Humble adapter. Measurement time drives inference; wall time detects silence."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, String
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
        self.rotation = np.asarray(self.config["sensor_rotation"])
        self.translation = np.asarray(self.config["sensor_translation"])
        self.kinds = {
            "points_display": ("sensor_msgs/msg/PointCloud2", PointCloud2),
            "debug_markers": ("visualization_msgs/msg/MarkerArray", MarkerArray),
            "status": ("std_msgs/msg/String", String),
            "attention_required": ("std_msgs/msg/Bool", Bool),
            "nearest_obstacle_m": ("std_msgs/msg/Float32", Float32),
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
        self.received = 0
        self.invalid = 0
        self.out_of_order = 0
        self.silent = False
        # Input freshness must keep expiring when bag playback /clock is paused.
        self.timer = self.create_timer(
            min(self.timeout, 0.5), self.watchdog,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(
            f"Ready. Reliability={reliability}, queue depth=1; use slow replay for complete evaluation."
        )

    def publish(self, row, points, stamp, support=None):
        payloads = self.messages.build(row, points, stamp, support)
        conversion_s = publication_s = 0.0
        for topic, message in payloads.items():
            started = time.monotonic()
            typename, native = self.kinds[topic]
            serialized = self.messages.store.serialize_cdr(message, typename)
            converted = deserialize_message(bytes(serialized), native)
            converted_at = time.monotonic()
            self.publishers_by_topic[topic].publish(converted)
            conversion_s += converted_at - started
            publication_s += time.monotonic() - converted_at
        return self.messages.last_timings | {
            "cdr_roundtrip_s": conversion_s, "publication_calls_s": publication_s}

    def unavailable(self, reason):
        row = {
            "coordinate_frame": "tunnel_guard_local",
            "status": "unknown",
            "objects": [],
            "nearest_obstacle_m": None,
            "nearest_candidate_m": None,
            "nearest_unresolved_range_m": None,
            "health": "unavailable",
            "health_reasons": [reason],
            "processed_scans": self.processed,
            "input_received_scans": self.received,
            "dropped_invalid_scans": self.invalid,
            "dropped_duplicate_scans": self.duplicates,
            "dropped_out_of_order_scans": self.out_of_order,
            "input_loss_count": None,
            "input_loss_note": "DDS has no source sequence counter; depth=1 retains only the newest queued scan.",
            "measurement_timestamp_ns": None,
            "result_age_s": None,
            "result_age_note": "No current measurement exists; route clearance is unknown.",
        }
        self.publish(row, np.empty((0, 3)), 0)

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
                f"No fresh cloud on {self.get_parameter('input_topic').value!r} for "
                f"{self.timeout:g} s. Point-cloud topics currently available: {available or 'none'}. "
                "Check the input topic and acquisition timestamp progression."
            )
            self.detector = Detector(self.config)
            # Keep the acquisition watermark: silence does not turn an old
            # measurement into new evidence. A new epoch requires a restart.
            self.unavailable("input_timeout")

    def on_cloud(self, message):
        received = time.monotonic()
        self.received += 1
        out_of_order = False
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
                self.out_of_order += 1
                out_of_order = True
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
            detector_started = time.monotonic()
            row = self.detector.process(points, ns * 1e-9, times, point_attributes=attributes)
            detector_process_s = time.monotonic() - detector_started
            self.last_stamp, self.source_frame = ns, message.header.frame_id
            # Receiving a duplicate does not renew the last usable measurement.
            self.last_received = received
            self.silent = False
            self.processed += 1
            row.update(
                measurement_timestamp_ns=ns,
                sensor_frame=message.header.frame_id,
                input_topic=self.get_parameter("input_topic").value,
                invalid_points=invalid,
                scan_duration_s=duration,
                input_received_scans=self.received,
                processed_scans=self.processed,
                skipped_duplicate_scans=self.duplicates,
                dropped_invalid_scans=self.invalid,
                dropped_duplicate_scans=self.duplicates,
                dropped_out_of_order_scans=self.out_of_order,
                input_queue_depth=1,
                input_reliability=self.get_parameter("input_reliability").value,
                input_loss_count=None,
                input_loss_note="DDS has no source sequence counter; depth=1 retains only the newest queued scan.",
                decode_s=decode_s,
                result_age_s=None,
                result_age_note="Not computed: acquisition and host clocks are not proven comparable.",
                latency_scope="Callback entry to publication; excludes DDS queue. Sensor/host clock relation unverified.",
                callback_processing_s=time.monotonic() - received,
            )
            if row["callback_processing_s"] > 0.1:
                row["health_reasons"].append("processing_exceeds_10hz_input_period")
                if row["health"] == "normal":
                    row["health"] = "degraded"
            publish_started = time.monotonic()
            display_timings = self.publish(
                row, self.detector.display_points, ns, self.detector.display_support
            )
            publish_s = time.monotonic() - publish_started
            callback_to_publish_return_s = time.monotonic() - received
            self.get_logger().info(
                json.dumps(
                    {
                        **display_timings,
                        "detector_process_s": detector_process_s,
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
            if not out_of_order:
                self.invalid += 1
            self.detector = Detector(self.config)
            self.get_logger().error(str(error))
            self.unavailable(str(error))


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
