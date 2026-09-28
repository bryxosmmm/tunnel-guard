"""ROS-free bag ingestion with the same PointCloud2 decoder usable in a ROS node."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

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
    measurement_timestamp_ns: int = 0
    record_timestamp_ns: int = 0
    skipped_duplicate_scans: int = 0
    deserialize_s: float = 0.0
    decode_s: float = 0.0
    attributes: PointAttributes | None = None
    # `detector.prepare_scan` for exactly these `points` and `point_times`, computed here because
    # this thread is otherwise idle while the detector works on the previous scan. It changes no
    # value: the same functions on the same inputs produce the same arrays.
    prepared: tuple | None = None


_ATTRIBUTE_KEYS = ("intensity", "ring", "raw_time")
_ATTRIBUTE_UNITS = {"intensity": "unverified_raw_counts", "ring": "ring_index"}


def _valid_values(key: str, values: np.ndarray) -> np.ndarray:
    """Per-element attribute validity: finite, and for ring also nonnegative integral."""
    if np.issubdtype(values.dtype, np.floating):
        valid = np.isfinite(values)
        if key == "ring":
            valid &= (values >= 0) & (values == np.floor(values))
        return valid
    if np.issubdtype(values.dtype, np.integer):
        # Integers are always finite and integral; only ring adds a sign constraint.
        return values >= 0 if key == "ring" else np.ones(values.shape, dtype=bool)
    return np.zeros(values.shape, dtype=bool)


@dataclass
class PointAttributes:
    """Sensor fields carried through decode without changing geometry decisions.

    ``values`` holds the raw source columns (original dtype and values) for whichever
    optional fields exist, aligned with the decoded, XYZ-valid rows. Invalid attribute
    values are preserved and flagged, never used to drop a point or alter the XYZ mask.
    """
    values: dict[str, np.ndarray]
    source_indices: np.ndarray
    xyz_valid_mask: np.ndarray
    time_field: str | None = None

    def summary(self) -> dict:
        """JSON-safe availability, counts and units; no identity or calibration claim."""
        fields = {}
        for key in _ATTRIBUTE_KEYS:
            values = self.values.get(key)
            if values is None:
                fields[key] = {"available": False}
                continue
            valid = _valid_values(key, values)
            valid_count = int(np.count_nonzero(valid))
            units = _ATTRIBUTE_UNITS.get(key)
            if units is None:
                units = "seconds" if self.time_field == "timestamp" else "unknown"
            fields[key] = {
                "available": True,
                "dtype": values.dtype.name,
                "units": units,
                "count": int(values.size),
                "valid_count": valid_count,
                "invalid_count": int(values.size) - valid_count,
            }
        return {
            "fields": fields,
            "source_time_field": self.time_field,
            "profile_confirmed": False,
            "firing_identity": "unknown",
            "return_multiplicity": "unknown",
            "intensity_calibration": "unverified",
        }

    def arrays(self, indices=None) -> dict:
        """NPZ-ready arrays for the selected decoded rows (all rows when indices is None).

        Excludes the full ``xyz_valid_mask``, which the caller stores once.
        """
        source_indices = self.source_indices
        if indices is not None:
            source_indices = source_indices[indices]
        out = {"source_indices": np.asarray(source_indices)}
        for key, values in self.values.items():
            selected = values if indices is None else values[indices]
            out[key] = selected
            out[f"{key}_valid"] = _valid_values(key, selected)
        return out


def prefetch(iterable, depth: int = 1):
    """Consume `iterable` on one producer thread, at most `depth` items ahead.

    Reading a bag costs decompression and decoding that do not depend on the detector, so
    a consumer that computes is idle while the next scan is prepared. This runs the same
    iteration the caller would have run, one thread earlier and bounded: order, duplicate
    skipping, subsampling and the value of every yielded item are unchanged, and
    StopIteration or an exception surfaces at the `next()` call that would have raised it.

    Bounded depth is deliberate. Depth 1 keeps one unprocessed scan in memory and limits
    how far the producer may run ahead of the consumer, so a slow consumer cannot build an
    unbounded queue of stale scans. It overlaps work; it does not make the consumer fast.
    Closing the consumer stops the producer, so a long-lived process cannot leak threads.
    """
    if depth < 1:
        # This function is a generator whether or not it prefetches, so returning an
        # iterator here would end it without yielding a single scan: `prefetch_depth: 0`
        # produced an empty run and `iter_bag` then raised "No scans processed".
        yield from iterable
        return
    items: queue.Queue = queue.Queue(maxsize=depth)
    done, stop = object(), threading.Event()

    def publish(item) -> bool:
        """Offer an item until it is queued or the consumer has stopped.

        Every put goes through here, including the exception and the end marker. A
        blocking put with no timeout cannot be released by `stop`, so a consumer that
        closes on a full queue would leave the producer thread parked forever.
        """
        while not stop.is_set():
            try:
                items.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for item in iterable:
                if not publish(item):
                    return
        except Exception as error:  # delivered to the consumer, never swallowed
            publish(error)
        finally:
            publish(done)

    threading.Thread(target=produce, daemon=True, name="iter_bag_prefetch").start()
    try:
        while True:
            item = items.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


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
    attribute_names = [n for n in ("intensity", "ring")
                       if n in fields and n not in required and n != time_name]
    names = required + ([time_name] if time_name else []) + attribute_names
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
    # Copy strided fields directly into their final float64 array. Building a
    # float32 column_stack first made an extra full cloud and padded-row copies.
    points = np.empty((message.height, message.width, 3), dtype=np.float64)
    for axis, name in enumerate(required):
        points[:, :, axis] = records[name]
    points = points.reshape(-1, 3)
    valid = (np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
             & (np.einsum("ij,ij->i", points, points) > 1e-6))
    invalid = len(points) - int(np.count_nonzero(valid))
    times = np.empty(0, dtype=np.float64)
    duration = 0.0
    if time_name and valid.any():
        raw_time = records[time_name].ravel().astype(np.float64, copy=False)
        if invalid:
            raw_time = raw_time[valid]
        if np.isfinite(raw_time).all():
            duration = float(np.ptp(raw_time))
        if duration > 0:
            times = (raw_time - raw_time.min()) / duration
            # Only a field named timestamp is interpreted as absolute seconds.
            # Other time fields retain normalized ordering but unknown units.
            if time_name != "timestamp":
                duration = 0.0
    # Optional sensor fields travel with the points they came from: the same XYZ-valid
    # mask selects both, so attribute rows stay aligned. Absent fields stay absent.
    mask_2d = valid.reshape(message.height, message.width)
    values = {name: records[name][mask_2d] for name in attribute_names}
    if time_name:
        values["raw_time"] = records[time_name][mask_2d]
    attributes = PointAttributes(values, np.flatnonzero(valid), valid, time_name)
    if invalid:
        points = points[valid]
    # Exact signed axis permutations cover the current mounting recipe. General
    # calibrated rotations keep the matrix product and its arithmetic order.
    axes = np.argmax(np.abs(rotation), axis=1)
    signs = rotation[np.arange(3), axes]
    permutation = np.zeros((3, 3))
    permutation[np.arange(3), axes] = signs
    if len(set(axes)) == 3 and np.all(np.abs(signs) == 1) and np.array_equal(rotation, permutation):
        transformed = np.empty_like(points)
        for axis, source in enumerate(axes):
            np.multiply(points[:, source], signs[axis], out=transformed[:, axis])
    else:
        transformed = points @ rotation.T
    transformed += translation
    return transformed, times, invalid, duration, attributes


def iter_bag(path: Path, config: dict, *, topic: str | None = None, every: int = 1,
             max_frames: int | None = None, diagnostics: dict | None = None,
             prepare: bool = True):
    if every < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("every and max_frames must be positive")
    store = get_typestore(Stores.ROS2_HUMBLE)
    from . import accelerator
    from .detector import prepare_scan
    # The reader thread only ever calls the CPU kernels: the device module's scratch is shared,
    # so a device call from here would corrupt a scan the detector is still working on.
    module = accelerator.cpu_native(config)
    rotation = np.asarray(config["sensor_rotation"], dtype=float)
    translation = np.asarray(config["sensor_translation"], dtype=float)
    stats = diagnostics if diagnostics is not None else {}
    stats.update(source_messages=0, duplicate_measurements=0, subsampled_measurements=0, emitted_scans=0)
    with Reader(path) as reader:
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"
                       and (topic is None or c.topic == topic)]
        topics = {c.topic for c in connections}
        if len(topics) != 1:
            raise ValueError(f"Select one PointCloud2 topic; found {sorted(topics)} in {path}")
        emitted = 0
        last_stamp = None
        source_frame = None
        duplicates = 0
        for index, (connection, timestamp_ns, raw) in enumerate(reader.messages(connections=connections)):
            if max_frames is not None and emitted >= max_frames:
                break
            stats["source_messages"] += 1
            deserialize_start = time.perf_counter()
            message = store.deserialize_cdr(raw, connection.msgtype)
            deserialize_s = time.perf_counter() - deserialize_start
            stamp = message.header.stamp
            if not 0 <= stamp.nanosec < 1_000_000_000 or stamp.sec < 0:
                raise ValueError(f"Invalid acquisition timestamp at scan {index}")
            measurement_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if source_frame is not None and source_frame != message.header.frame_id:
                raise ValueError(f"Sensor frame changed at scan {index}; calibration must be reselected")
            source_frame = message.header.frame_id
            if not source_frame:
                raise ValueError(f"Missing sensor frame at scan {index}")
            if last_stamp is not None and measurement_ns < last_stamp:
                raise ValueError(f"Acquisition time moved backwards at scan {index}; split recording into epochs")
            if measurement_ns == last_stamp:
                duplicates += 1
                stats["duplicate_measurements"] += 1
                continue
            last_stamp = measurement_ns
            if index % every:
                stats["subsampled_measurements"] += 1
                continue
            decode_start = time.perf_counter()
            points, times, invalid, duration, attributes = decode_cloud(message, rotation, translation)
            decode_s = time.perf_counter() - decode_start
            # Only worth doing when a reader thread exists to run ahead of the detector;
            # without one this runs on the consumer, where the device kernels are faster
            # (measured, the crop/reduce pair costs 12.5 ms on the CPU against 2.2 ms there).
            prepared = prepare_scan(points, times, config, module) if prepare else None
            stats["emitted_scans"] += 1
            yield Scan(index, measurement_ns * 1e-9, points, times, message.header.frame_id,
                       connection.topic, message.height * message.width, invalid, duration,
                       measurement_ns, timestamp_ns, duplicates, deserialize_s, decode_s,
                       attributes, prepared)
            emitted += 1
