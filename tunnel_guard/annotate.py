"""Extract raw-cloud evidence for independently authored annotation boxes.

This command never invokes the detector or turns an unreviewed bag into a negative.
Box annotations are authored as JSON and validated by tunnel_guard.evaluate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import load_config
from .evaluate import validate_annotations
from .io import iter_bag
from .run import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotation = json.loads(args.annotations.read_text())
    validate_annotations(annotation)
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = []
    for bag in sorted({f["bag"] for f in annotation["frames"]}):
        if Path(bag).name != bag:
            raise ValueError("Annotation bag names must be simple directory names")
        selected = {f["frame"]: f for f in annotation["frames"] if f["bag"] == bag}
        found = set()
        for scan in iter_bag(args.bag_root / bag, config):
            if scan.index not in selected:
                continue
            found.add(scan.index)
            frame = selected[scan.index]
            filename = f"{bag}_{scan.index:06d}.npz"
            arrays = {"points": scan.points, "timestamp_s": scan.timestamp_s,
                      "point_times": scan.point_times}
            if scan.attributes is not None:
                # The decoded tuple metadata travels with the raw cloud it was decoded from, under
                # the same decoded_* naming the detector's diagnostic capture uses, so an
                # independent annotator reads the sensor fields rather than reparsing the payload.
                arrays["decoded_xyz_valid_mask"] = scan.attributes.xyz_valid_mask
                arrays.update({f"decoded_{key}": value
                               for key, value in scan.attributes.arrays().items()})
            np.savez_compressed(args.output / filename, **arrays)
            object_evidence = []
            for obj in frame["objects"]:
                inside = np.all((scan.points >= obj["bbox_min"]) & (scan.points <= obj["bbox_max"]), axis=1)
                cloud = scan.points[inside]
                object_evidence.append({"event_id": obj["event_id"], "points_in_authored_box": len(cloud),
                                        "observed_min": cloud.min(axis=0).tolist() if len(cloud) else None,
                                        "observed_max": cloud.max(axis=0).tolist() if len(cloud) else None})
            evidence.append({"bag": bag, "frame": scan.index, "raw_cloud": filename,
                             "timestamp_s": scan.timestamp_s, "objects": object_evidence,
                             "sensor_attributes": scan.attributes.summary() if scan.attributes is not None else None})
            if found == set(selected):
                break
        if found != set(selected):
            raise ValueError(f"Annotation frames absent from {bag}: {sorted(set(selected) - found)}")
    write_json(args.output / "evidence.json", {"annotations_sha256": digest(args.annotations),
               "config_sha256": digest(args.config), "label_status": annotation["label_status"],
               "warning": "Point support validates extraction, not object identity, exhaustiveness, or path intersection.",
               "frames": evidence})
    print(f"Extracted {len(evidence)} raw annotation frames to {args.output}")


if __name__ == "__main__":
    main()
