"""Local browser review of complete recorded results and their source measurements."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .io import decode_cloud
from .visualization import corridor_edges, distance_summary


def _channel_payload(values: np.ndarray, valid: np.ndarray):
    """JSON-safe channel row: unusable samples become null, never zero.

    A zero that the sensor actually reported stays a zero; only samples the
    validity mask rejects (non-finite, or a ring that is not a nonnegative
    integer) are replaced by null so the browser cannot mistake them for data.
    """
    payload = values.tolist()
    flags = valid.tolist()
    for position, ok in enumerate(flags):
        if not ok:
            payload[position] = None
    return payload, flags


def _displayed_attributes(attributes, selection: np.ndarray) -> dict:
    """Channel arrays and provenance aligned to exactly the displayed rows."""
    if attributes is None:
        return {"available": False, "summary": None, "channels": {},
                "displayed_source_indices": None}
    arrays = attributes.arrays(selection)
    channels = {}
    source_indices = None
    for name, array in arrays.items():
        if name == "source_indices":
            source_indices = array.tolist()
            continue
        if name.endswith("_valid") and name[: -len("_valid")] in arrays:
            continue
        valid = arrays.get(name + "_valid")
        if valid is None:
            valid = np.isfinite(array)
        values, flags = _channel_payload(array, valid)
        channels[name] = {"values": values, "valid": flags,
                          "dtype": str(array.dtype)}
    return {"available": True, "summary": attributes.summary(),
            "channels": channels, "displayed_source_indices": source_indices}


class ResultIndex:
    """Index complete JSONL records by byte offset, keeping objects on disk."""

    def __init__(self, path):
        self.path = path
        self.offsets = []
        self.frames = []
        self.position = 0
        stat = path.stat()
        self.identity = (stat.st_dev, stat.st_ino)
        self.lock = threading.Lock()
        self.refresh()

    def refresh(self):
        with self.lock:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self.identity or stat.st_size < self.position:
                raise ValueError("Indexed result file was replaced or truncated")
            with self.path.open("rb") as stream:
                stream.seek(self.position)
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line.endswith(b"\n"):
                        break  # A running detector may not have finished this row.
                    row = json.loads(line)
                    self.offsets.append(offset)
                    self.frames.append(row["frame"])
                    self.position = stream.tell()
            return list(self.frames)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.lock:
            if index < 0 or index >= len(self.offsets):
                raise IndexError("Frame index outside recorded run")
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self.identity or stat.st_size < self.position:
                raise ValueError("Indexed result file was replaced or truncated")
            with self.path.open("rb") as stream:
                stream.seek(self.offsets[index])
                row = json.loads(stream.readline())
            if row["frame"] != self.frames[index]:
                raise ValueError("Indexed result frame changed")
            return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--display-max-points", type=int, default=30000)
    args = parser.parse_args()
    if args.display_max_points < 1:
        parser.error("display-max-points must be positive")
    config = json.loads((args.run / "detector.json").read_text())
    if config.get("deskew_enabled", False):
        parser.error(
            "Source cloud replay is exact only with deskew disabled; use the recorded RViz bag for deskewed runs"
        )
    rows = ResultIndex(args.run / (args.bag.name + ".jsonl"))
    if not rows:
        parser.error("Result contains no frames")
    manifest = json.loads((args.run / "manifest.json").read_text())
    entry = next(
        (b for b in manifest["bags"] if Path(b["path"]).name == args.bag.name), None
    )
    if (
        entry is None
        or hashlib.sha256((args.bag / "metadata.yaml").read_bytes()).hexdigest()
        != entry["metadata_sha256"]
    ):
        parser.error("Source bag metadata does not match the recorded run")
    store = get_typestore(Stores.ROS2_HUMBLE)
    rotation, translation = (
        np.asarray(config["sensor_rotation"]),
        np.asarray(config["sensor_translation"]),
    )

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    payload, kind = (
                        Path(__file__).with_name("review.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                elif url.path == "/metadata":
                    payload = json.dumps(
                        {"bag": args.bag.name, "run": args.run.name,
                         "frames": rows.refresh()}
                    ).encode()
                    kind = "application/json"
                elif url.path == "/object":
                    query = parse_qs(url.query)
                    index, identity = int(query["index"][0]), int(query["id"][0])
                    if index < 0 or index >= len(rows):
                        raise ValueError("Frame index outside recorded run")
                    row = rows[index]
                    obj = next((o for o in row["objects"] if o["track_id"] == identity), None)
                    if obj is None:
                        raise ValueError("Object ID is absent from this frame")
                    response = {"available": False, "points": [], "reason": "diagnostics_not_recorded"}
                    diagnostic = row.get("diagnostic_points")
                    if diagnostic:
                        path = (args.run / diagnostic).resolve()
                        if not path.is_relative_to(args.run.resolve()):
                            raise ValueError("Diagnostic path outside run")
                        if path.is_file():
                            with np.load(path, allow_pickle=False) as arrays:
                                support = arrays["cluster_points"][arrays["cluster_labels"] == obj["component_id"]]
                            if len(support) != obj["support_voxels"]:
                                raise ValueError("Recorded support count differs from object record")
                            response = {"available": True, "points": support.tolist(),
                                        "reason": "exact_current_scan_voxel_representatives"}
                    payload, kind = json.dumps(response, allow_nan=False).encode(), "application/json"
                elif url.path == "/frame":
                    index = int(parse_qs(url.query)["index"][0])
                    if index < 0 or index >= len(rows):
                        raise ValueError("Frame index outside recorded run")
                    row = rows[index]
                    cloud = None
                    attributes = None
                    with Reader(args.bag) as reader:
                        connections = [
                            c for c in reader.connections if c.topic == row["topic"]
                        ]
                        for connection, _, raw in reader.messages(
                            connections=connections,
                            start=row["record_timestamp_ns"],
                            stop=row["record_timestamp_ns"] + 1,
                        ):
                            msg = store.deserialize_cdr(raw, connection.msgtype)
                            ns = (
                                msg.header.stamp.sec * 1_000_000_000
                                + msg.header.stamp.nanosec
                            )
                            if (
                                ns == row["measurement_timestamp_ns"]
                                and msg.header.frame_id == row["sensor_frame"]
                            ):
                                cloud, _, _, _, attributes = decode_cloud(
                                    msg, rotation, translation
                                )
                                break
                    if cloud is None:
                        raise ValueError(
                            "Exact source measurement not found; refusing a different frame"
                        )
                    count = len(cloud)
                    step = max(1, int(np.ceil(count / args.display_max_points)))
                    # One explicit index list drives both the displayed XYZ and the
                    # attribute channels, so subsampling cannot desynchronize them.
                    selection = np.arange(0, count, step, dtype=np.int64)
                    response = {
                        "row": row
                        | {
                            "measurement_timestamp_ns_string": str(
                                row["measurement_timestamp_ns"]
                            )
                        },
                        "points": cloud[selection].round(4).tolist(),
                        "decoded_points": count,
                        "display_step": step,
                        "distances": distance_summary(row),
                        "corridor": corridor_edges(row.get("geometry", {}), config)
                        .round(4)
                        .tolist(),
                        "attributes": _displayed_attributes(attributes, selection),
                    }
                    payload, kind = (
                        json.dumps(response, allow_nan=False).encode(),
                        "application/json",
                    )
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, KeyError, IndexError) as error:
                payload = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"Recorded review: http://127.0.0.1:{args.port} — {len(rows)} frames; Ctrl-C to stop",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
