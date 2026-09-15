"""Compare original published backends on fixed labels and a fixed stress panel."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from .detector import Detector, load_config
from .evaluate import evaluate_frames
from .io import iter_bag
from .run import digest, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(plan["detector_config"])
    annotations = json.loads(Path(plan["annotations"]).read_text())
    write_json(output / "experiment.json", plan)
    write_json(output / "manifest.json", environment() | {"annotation_sha256": digest(Path(plan["annotations"])),
                                                        "experiment_sha256": digest(args.experiment)})
    source_dir = output / "source" / "tunnel_guard"
    source_dir.mkdir(parents=True)
    for source in Path(__file__).parent.glob("*.py"):
        (source_dir / source.name).write_bytes(source.read_bytes())
    summary = []
    for method in plan["methods"]:
        candidate = copy.deepcopy(config)
        candidate["segmentation_method"] = method
        config_path = output / f"detector-{method}.json"
        write_json(config_path, candidate)
        predictions = {}
        started = time.perf_counter()
        with (output / f"real-{method}.jsonl").open("x") as stream:
            for bag in sorted({f["bag"] for f in annotations["frames"]}):
                detector = Detector(candidate)
                for scan in iter_bag(Path(plan["bag_root"]) / bag, candidate,
                                     every=plan["real_every"], max_frames=plan["real_max_frames"]):
                    row = detector.process(scan.points, scan.timestamp_s, scan.point_times)
                    row.update(bag=bag, frame=scan.index)
                    predictions[(bag, scan.index)] = row
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        score = evaluate_frames(predictions, annotations)
        score.update(method=method, wall_s=time.perf_counter() - started,
                     annotations_sha256=digest(Path(plan["annotations"])))
        write_json(output / f"real-{method}-metrics.json", score)
        stress = json.loads(Path(plan["stress_config"]).read_text())
        stress["detector_config"] = str(config_path)
        stress["output"] = str(output / f"stress-{method}")
        stress_path = output / f"stress-{method}.json"
        write_json(stress_path, stress)
        with (output / f"stress-{method}.log").open("x") as log:
            subprocess.run([sys.executable, "-m", "tunnel_guard.stress", "--experiment", str(stress_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        synthetic = json.loads((output / f"stress-{method}" / "metrics.json").read_text())
        entry = {"method": method, "real_localization_recall": score["annotated_object_recall"],
                 "real_mean_iou_matched": score["matched_mean_3d_iou"], "real_precision": score["precision_exhaustive_only"],
                 "synthetic_event_recall": synthetic["event_recall"],
                 "synthetic_precision": synthetic["precision_exhaustive_only"],
                 "synthetic_recall": synthetic["annotated_object_recall"], "synthetic_f1": synthetic["f1_exhaustive_only"],
                 "synthetic_negative_episode_rate": synthetic["negative_episode_rate"],
                 "real_validity_warnings": score["validity_warnings"]}
        criteria = plan["success_criterion"]
        entry["pass"] = (entry["real_localization_recall"] >= criteria["real_localization_recall_min"]
                         and entry["synthetic_event_recall"] >= criteria["synthetic_event_recall_min"]
                         and entry["synthetic_negative_episode_rate"] <= criteria["negative_episode_rate_max"])
        summary.append(entry)
        write_json(output / "comparison.json", summary)
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
