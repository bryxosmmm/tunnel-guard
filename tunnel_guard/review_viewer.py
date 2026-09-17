"""Local browser review of complete recorded results and their source measurements."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .io import decode_cloud
from .visualization import corridor_edges, distance_summary


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
    rows = [
        json.loads(line)
        for line in (args.run / (args.bag.name + ".jsonl")).read_text().splitlines()
    ]
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
                        {"bag": args.bag.name, "frames": [r["frame"] for r in rows]}
                    ).encode()
                    kind = "application/json"
                elif url.path == "/frame":
                    index = int(parse_qs(url.query)["index"][0])
                    if index < 0 or index >= len(rows):
                        raise ValueError("Frame index outside recorded run")
                    row = rows[index]
                    with Reader(args.bag) as reader:
                        connections = [
                            c for c in reader.connections if c.topic == row["topic"]
                        ]
                        cloud = None
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
                                cloud, _, _, _ = decode_cloud(
                                    msg, rotation, translation
                                )
                                break
                    if cloud is None:
                        raise ValueError(
                            "Exact source measurement not found; refusing a different frame"
                        )
                    count = len(cloud)
                    cloud = cloud[
                        :: max(1, int(np.ceil(count / args.display_max_points)))
                    ].round(4)
                    response = {
                        "row": row
                        | {
                            "measurement_timestamp_ns_string": str(
                                row["measurement_timestamp_ns"]
                            )
                        },
                        "points": cloud.tolist(),
                        "decoded_points": count,
                        "distances": distance_summary(row),
                        "corridor": corridor_edges(row.get("geometry", {}), config)
                        .round(4)
                        .tolist(),
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
