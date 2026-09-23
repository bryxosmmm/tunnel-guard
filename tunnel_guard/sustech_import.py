"""Import recordings into SUSTechPOINTS scenes.

Writes one ``.pcd`` per bag message into ``<scene-root>/<bag>/lidar/`` with the same
`tunnel_guard_local` transform the detector applies, keeps the intensity field, and creates an
empty ``label`` directory. Frame ``00000N.pcd`` is bag message index ``N``, which is the index
used by the run JSONL, so detector output, proposals and labels all refer to the same frame
number. Nothing is downsampled, cropped, deskewed or filtered; ring and per-point timestamps
stay in the original bags.

The scene root must not contain anything but scenes afterwards: the tool's ``scene_reader``
treats every entry of that directory as a scene directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from .io import decode_cloud


def write_pcd(path: Path, points: np.ndarray, intensity: np.ndarray):
    header = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
              f"COUNT 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS {len(points)}\nDATA binary\n")
    payload = np.empty((len(points), 4), dtype="<f4")
    payload[:, :3] = points
    payload[:, 3] = intensity
    with path.open("xb") as out:
        out.write(header.encode("ascii"))
        payload.tofile(out)


def import_bag(bag: Path, scene: Path, config: dict, max_frames: int | None = None) -> dict:
    rotation = np.asarray(config["sensor_rotation"])
    translation = np.asarray(config["sensor_translation"])
    store = get_typestore(Stores.ROS2_HUMBLE)
    scene.mkdir(parents=True, exist_ok=False)
    (scene / "lidar").mkdir()
    (scene / "label").mkdir()
    frames = points = 0
    started = time.perf_counter()
    with Reader(bag) as reader:
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
        if len({c.topic for c in connections}) != 1:
            raise ValueError(f"Ambiguous point cloud topic in {bag}")
        for index, (connection, _, raw) in enumerate(reader.messages(connections=connections)):
            if max_frames is not None and index >= max_frames:
                break
            message = store.deserialize_cdr(raw, connection.msgtype)
            xyz, _, _, _, attributes = decode_cloud(message, rotation, translation)
            intensity = attributes.values.get("intensity")
            if intensity is None:
                raise ValueError(f"PointCloud2 in {bag} has no scalar intensity field")
            write_pcd(scene / "lidar" / f"{index:06d}.pcd", xyz, intensity)
            frames += 1
            points += len(xyz)
    shutil.copyfile(bag / "metadata.yaml", scene / "source-metadata.yaml")
    return {"bag": bag.name, "frames": frames, "points": points,
            "seconds": round(time.perf_counter() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("configs/evaluation-quality.json"))
    parser.add_argument("--scene-root", type=Path, default=Path("SUSTechPOINTS/data"))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    plan = json.loads(args.experiment.read_text())
    config = json.loads(Path(plan["detector_config"]).read_text())
    args.scene_root.mkdir(parents=True, exist_ok=True)
    for entry in plan["bags"]:
        bag = Path(entry["path"])
        result = import_bag(bag, args.scene_root / bag.name, config, args.max_frames)
        print(f"{result['bag']}: {result['frames']} frames, {result['points']} points, "
              f"{result['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
