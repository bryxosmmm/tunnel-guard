"""Characterise sensor identity, clocks, observability and mounting across whole recordings.

``inspect_bag`` dumps the PointCloud2 layout and acquisition clocks of a bounded prefix of one
recording. This command answers the cross-recording questions instead: which capture
configuration a bag belongs to, whether the two configurations share a ring and azimuth
pattern, how the clocks behave over an entire recording rather than a prefix, and how the
sensor sits relative to the track bed it observes.

Everything reported here is measured from the bags. Nothing is taken from a manufacturer
datasheet: a family manual does not establish what this particular unit was configured to do,
and no ``verified`` flag is set from a document. Fields the bags cannot answer are reported as
unknown rather than guessed, because that list is what has to go to the organisers.

Mounting is estimated through the detector's own ground plane and rail anchors, so the numbers
describe the frame the detector actually works in, including its configured ``sensor_rotation``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag
from .run import digest

RAW_FIELDS = ("x", "y", "z", "intensity", "ring", "timestamp")
TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def raw_records(message) -> np.ndarray:
    """Every slot of the cloud in sensor coordinates, invalid returns included.

    The detector drops invalid slots before it ever sees a cloud; the ring and azimuth pattern
    is only legible with them kept, because an unreturned slot still occupies its emission.
    """
    fields = {f.name: f for f in message.fields}
    present = [n for n in RAW_FIELDS if n in fields]
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({"names": present,
                      "formats": [endian + TYPES[fields[n].datatype] for n in present],
                      "offsets": [fields[n].offset for n in present],
                      "itemsize": message.point_step})
    return np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                      strides=(message.row_step, message.point_step)).ravel()


def scan_structure(message) -> dict:
    """Ring table, azimuth grid and return multiplicity of one scan."""
    records = raw_records(message)
    x, y, z = (records[n].astype(np.float64) for n in ("x", "y", "z"))
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    valid = finite & ((x**2 + y**2 + z**2) > 1e-6)
    out = {"slots": int(records.size), "valid_slots": int(valid.sum()),
           "invalid_slots": int((~valid).sum())}

    if "ring" in records.dtype.names:
        rings = records["ring"].astype(np.int64)
        out["rings_present"] = int(len(np.unique(rings)))
        out["slots_per_ring"] = int(np.median(np.bincount(rings)))
        elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
        table, spread = [], []
        for r in np.unique(rings):
            values = elevation[(rings == r) & valid]
            if len(values) < 10:
                continue
            table.append(float(np.median(values)))
            spread.append(float(np.quantile(values, 0.95) - np.quantile(values, 0.05)))
        out["ring_elevation_deg"] = [round(v, 4) for v in table]
        out["max_ring_elevation_spread_deg"] = round(float(max(spread)), 6) if spread else None
        out["elevation_span_deg"] = ([round(min(table), 3), round(max(table), 3)] if table else None)

    # Azimuth is measured inside one ring: pooling rings mixes their small mounting offsets and
    # destroys the emission grid. Coverage is 360 minus the largest gap, so a cropped sector is
    # measured correctly and a full circle does not read as a 360 deg span by accident.
    azimuth = np.degrees(np.arctan2(y, x))
    if "ring" in records.dtype.names:
        rings = records["ring"].astype(np.int64)
        busiest = int(np.bincount(rings[valid]).argmax()) if valid.any() else 0
        sector = np.sort(np.unique(azimuth[valid & (rings == busiest)]))
        if len(sector) > 2:
            # Close the circle before taking gaps, so a cropped sector and a full turn with a
            # blind arc are both measured as 360 minus the largest gap, with no special case.
            steps = np.diff(np.r_[sector, sector[0] + 360.0])
            out["azimuth_reference_ring"] = busiest
            out["azimuth_coverage_deg"] = round(360.0 - float(steps.max()), 1)
            out["largest_azimuth_gap_deg"] = round(float(steps.max()), 1)
            out["median_azimuth_step_deg"] = round(float(np.median(steps)), 4)

    if "timestamp" in records.dtype.names:
        stamps = records["timestamp"].astype(np.float64)
        finite_stamps = stamps[np.isfinite(stamps)]
        if len(finite_stamps):
            out["point_time_span_s"] = round(float(np.ptp(finite_stamps)), 9)
            out["unique_point_times"] = int(len(np.unique(finite_stamps)))
        # One emission is one (ring, emission time) pair, so its slot count is the return
        # multiplicity. Counting by azimuth instead would fold quantisation noise into it.
        if "ring" in records.dtype.names:
            order = np.lexsort((stamps, records["ring"].astype(np.int64)))
            boundary = np.ones(len(order), dtype=bool)
            boundary[1:] = ((records["ring"].astype(np.int64)[order][1:] != records["ring"].astype(np.int64)[order][:-1])
                            | (stamps[order][1:] != stamps[order][:-1]))
            per_emission = np.diff(np.flatnonzero(np.r_[boundary, True]))
            counts = Counter(per_emission.tolist())
            out["slots_per_emission"] = {str(k): int(v) for k, v in sorted(counts.items())}
            out["emissions"] = int(len(per_emission))
    return out


def clock_stats(bag: Path, store) -> dict:
    """Header stamps, record stamps and their relationship over the whole recording."""
    headers, records, spans, starts, ends = [], [], [], [], []
    with Reader(bag) as reader:
        connections = [c for c in reader.connections
                       if c.msgtype == "sensor_msgs/msg/PointCloud2"]
        topics = sorted({c.topic for c in connections})
        other = sorted({c.topic for c in reader.connections
                        if c.msgtype != "sensor_msgs/msg/PointCloud2"})
        for connection, record_ns, raw in reader.messages(connections=connections):
            message = store.deserialize_cdr(raw, connection.msgtype)
            header_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            headers.append(header_ns)
            records.append(record_ns)
            fields = {f.name: f for f in message.fields}
            if "timestamp" in fields:
                stamps = raw_records(message)["timestamp"].astype(np.float64)
                stamps = stamps[np.isfinite(stamps)]
                spans.append(float(np.ptp(stamps)))
                # Which instant of the acquisition the header names decides what a measurement
                # age means and what deskew has to rotate points about.
                starts.append(header_ns * 1e-9 - float(stamps.min()))
                ends.append(header_ns * 1e-9 - float(stamps.max()))
    header = np.asarray(headers, dtype=np.int64)
    record = np.asarray(records, dtype=np.int64)
    delta = np.diff(header) / 1e9
    return {
        "pointcloud_topics": topics, "other_topics": other, "messages": len(header),
        "duplicate_header_stamps": int(len(header) - len(np.unique(header))),
        "header_monotonic": bool(np.all(np.diff(header) >= 0)),
        "header_period_s": {"p50": round(float(np.median(delta)), 6),
                            "p95": round(float(np.quantile(delta, 0.95)), 6),
                            "min": round(float(delta.min()), 6),
                            "max": round(float(delta.max()), 6)} if len(delta) else None,
        "gaps_over_1_5_period": int(np.count_nonzero(delta > 1.5 * np.median(delta))) if len(delta) else 0,
        "record_minus_header_s": {"p50": round(float(np.median((record - header) / 1e9)), 3),
                                  "min": round(float(((record - header) / 1e9).min()), 3),
                                  "max": round(float(((record - header) / 1e9).max()), 3)},
        "point_time_span_s": {"p50": round(float(np.median(spans)), 6),
                              "min": round(float(np.min(spans)), 6),
                              "max": round(float(np.max(spans)), 6)} if spans else None,
        # KISS-ICP stretches the supplied [0, 1] point-time range over one full inter-frame
        # motion increment, so a span shorter than the period would be over-corrected by 1/ratio.
        "acquisition_span_over_header_period": (
            round(float(np.median(spans) / np.median(delta)), 3)
            if spans and len(delta) and np.median(delta) > 0 else None),
        "header_minus_first_point_s": round(float(np.median(starts)), 6) if starts else None,
        "header_minus_last_point_s": round(float(np.median(ends)), 6) if ends else None,
    }


def mounting(bag: Path, detector: dict, every: int, limit: int) -> dict:
    """Ground plane and track heading in the frame the detector works in.

    ``robust_plane`` returns z = a*x + b*y + c for the track bed, so a and b are the pitch and
    roll of the sensor relative to it and -c is the height above it. The track heading comes
    from the rail anchors. A mounting difference between recordings shows up here; a curve does
    not, because the heading is read at the near anchors.
    """
    pitch, roll, height, yaw = [], [], [], []
    for scan in iter_bag(bag, detector, every=every, max_frames=limit):
        forward = ((scan.points[:, 0] >= detector["min_forward_m"])
                   & (scan.points[:, 0] <= detector["max_range_m"])
                   & (np.abs(scan.points[:, 1]) <= detector["context_half_width_m"]))
        reduced = voxel_representatives(scan.points[forward], detector["geometry_voxel_m"])
        geometry = TrackGeometry(reduced, {**detector, "background": {**detector["background"],
                                                                     "enabled": False}})
        if geometry.plane is None:
            continue
        pitch.append(np.degrees(np.arctan(geometry.plane[0])))
        roll.append(np.degrees(np.arctan(geometry.plane[1])))
        height.append(-geometry.plane[2])
        near = geometry.rail_anchors[geometry.rail_anchors[:, 0] <= 45]
        if len(near) >= 4:
            yaw.append(np.degrees(np.arctan(np.polyfit(near[:, 0], near[:, 1], 1)[0])))

    def summary(values, digits=3):
        return ({"p50": round(float(np.median(values)), digits),
                 "p10": round(float(np.quantile(values, 0.1)), digits),
                 "p90": round(float(np.quantile(values, 0.9)), digits),
                 "frames": len(values)} if len(values) else None)
    return {"frames_used": len(pitch), "bed_pitch_deg": summary(pitch),
            "bed_roll_deg": summary(roll), "sensor_height_above_bed_m": summary(height),
            "track_heading_deg": summary(yaw)}


def profile(bag: Path, detector: dict, store, structure_every: int, structure_limit: int,
            mounting_every: int, mounting_limit: int) -> dict:
    scans = []
    with Reader(bag) as reader:
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
        first = None
        for index, (connection, _, raw) in enumerate(reader.messages(connections=connections)):
            if len(scans) >= structure_limit:
                break
            if index % structure_every:
                continue
            message = store.deserialize_cdr(raw, connection.msgtype)
            if first is None:
                first = {"topic": connection.topic, "frame_id": message.header.frame_id,
                         "height": message.height, "width": message.width,
                         "point_step": message.point_step, "row_step": message.row_step,
                         "is_bigendian": bool(message.is_bigendian), "is_dense": bool(message.is_dense),
                         "fields": [{"name": f.name, "offset": f.offset,
                                     "datatype": f.datatype, "count": f.count}
                                    for f in message.fields]}
            scans.append(scan_structure(message))
    return {"bag": str(bag), "metadata_sha256": digest(bag / "metadata.yaml"),
            "layout": first, "structure_scans": scans,
            "clocks": clock_stats(bag, store),
            "mounting": mounting(bag, detector, mounting_every, mounting_limit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure-every", type=int, default=50)
    parser.add_argument("--structure-limit", type=int, default=6)
    parser.add_argument("--mounting-every", type=int, default=10)
    parser.add_argument("--mounting-limit", type=int, default=25)
    args = parser.parse_args()

    detector = load_config(args.config)
    store = get_typestore(Stores.ROS2_HUMBLE)
    results = [profile(bag, detector, store, args.structure_every, args.structure_limit,
                       args.mounting_every, args.mounting_limit) for bag in args.bag]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": str(args.config), "config_sha256": digest(args.config),
        "note": ("Measured from these bags only. No manufacturer datasheet was used and no "
                 "verified flag is implied. Detection range, calibration provenance and prior "
                 "deskew cannot be established from a recording alone."),
        "recordings": results}, indent=1) + "\n")

    for result in results:
        layout, clocks, first = result["layout"], result["clocks"], result["structure_scans"][0]
        print(f"\n{Path(result['bag']).name}")
        print(f"  topic {layout['topic']}  frame_id {layout['frame_id']}  "
              f"{layout['height']}x{layout['width']} slots, point_step {layout['point_step']}")
        print(f"  rings {first.get('rings_present')}  elevation {first.get('elevation_span_deg')} deg  "
              f"azimuth {first.get('azimuth_coverage_deg')} deg @ {first.get('median_azimuth_step_deg')} deg")
        print(f"  slots/emission {first.get('slots_per_emission')} over {first.get('emissions')} emissions  "
              f"valid {first['valid_slots']}/{first['slots']}")
        print(f"  header period p50 {clocks['header_period_s']['p50']} s, gaps {clocks['gaps_over_1_5_period']}, "
              f"duplicates {clocks['duplicate_header_stamps']}, point span p50 "
              f"{clocks['point_time_span_s']['p50'] if clocks['point_time_span_s'] else None} s")
        print(f"  record-header p50 {clocks['record_minus_header_s']['p50']} s")
        mount = result["mounting"]
        print(f"  bed pitch {mount['bed_pitch_deg']}\n  bed roll  {mount['bed_roll_deg']}")
        print(f"  height    {mount['sensor_height_above_bed_m']}\n  heading   {mount['track_heading_deg']}")
    print(f"\nEvidence: {args.output}")


if __name__ == "__main__":
    main()
