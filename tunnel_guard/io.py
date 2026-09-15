"""ROS-free bag ingestion with the same PointCloud2 decoder usable in a ROS node."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


@dataclass
class Scan:
    index: int
    timestamp_s: float
    points: np.ndarray
    point_times: np.ndarray
    frame_id: str
    topic: str
    raw_points: int
    invalid_points: int
    scan_duration_s: float


def decode_cloud(message, rotation: np.ndarray, translation: np.ndarray):
    """Respect padding, endianness, invalid returns and optional acquisition times.

    Point timestamps are normalized within the acquisition, never confused with bag
    record time. Missing/constant timestamps disable deskew rather than invent time.
    """
    fields = {f.name: f for f in message.fields}
    endian = ">" if message.is_bigendian else "<"
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    required = ["x", "y", "z"]
    time_name = next((n for n in ("timestamp", "time", "t") if n in fields), None)
    names = required + ([time_name] if time_name else [])
    for name in names:
        if name not in fields or fields[name].count != 1 or fields[name].datatype not in types:
            raise ValueError(f"Missing or unsupported scalar PointCloud2 field: {name}")
        f = fields[name]
        if f.offset < 0 or f.offset + np.dtype(types[f.datatype]).itemsize > message.point_step:
            raise ValueError(f"PointCloud2 field exceeds point_step: {name}")
    if message.row_step < message.width * message.point_step:
        raise ValueError("PointCloud2 row_step is shorter than its row")
    if len(message.data) < message.height * message.row_step:
        raise ValueError("Truncated PointCloud2 data")
    dtype = np.dtype({"names": names, "formats": [endian + types[fields[n].datatype] for n in names],
                      "offsets": [fields[n].offset for n in names], "itemsize": message.point_step})
    records = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                         strides=(message.row_step, message.point_step))
    points = np.column_stack([records[n].ravel() for n in required]).astype(np.float64)
    valid = np.isfinite(points).all(axis=1) & (np.einsum("ij,ij->i", points, points) > 1e-6)
    times = np.empty(0, dtype=np.float64)
    duration = 0.0
    if time_name and valid.any():
        raw_time = records[time_name].ravel().astype(np.float64)[valid]
        if np.isfinite(raw_time).all() and np.ptp(raw_time) > 0:
            duration = float(np.ptp(raw_time))
            times = (raw_time - raw_time.min()) / duration
            # Only a field named timestamp is interpreted as absolute seconds.
            # Other time fields retain normalized ordering but unknown units.
            if time_name != "timestamp":
                duration = 0.0
    return points[valid] @ rotation.T + translation, times, int((~valid).sum()), duration


def iter_bag(path: Path, config: dict, *, topic: str | None = None, every: int = 1,
             max_frames: int | None = None):
    if every < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("every and max_frames must be positive")
    store = get_typestore(Stores.ROS2_HUMBLE)
    rotation = np.asarray(config["sensor_rotation"], dtype=float)
    translation = np.asarray(config["sensor_translation"], dtype=float)
    with Reader(path) as reader:
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"
                       and (topic is None or c.topic == topic)]
        topics = {c.topic for c in connections}
        if len(topics) != 1:
            raise ValueError(f"Select one PointCloud2 topic; found {sorted(topics)} in {path}")
        emitted = 0
        for index, (connection, timestamp_ns, raw) in enumerate(reader.messages(connections=connections)):
            if index % every:
                continue
            if max_frames is not None and emitted >= max_frames:
                break
            message = store.deserialize_cdr(raw, connection.msgtype)
            points, times, invalid, duration = decode_cloud(message, rotation, translation)
            yield Scan(index, timestamp_ns * 1e-9, points, times, message.header.frame_id,
                       connection.topic, message.height * message.width, invalid, duration)
            emitted += 1
