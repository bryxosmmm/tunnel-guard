"""Summary artifact for a corridor-claim change: before and after, per corpus.

Reads recorded run directories only (no detector run), and writes one JSON with the status
counts, the hazard anatomy and the frame-level losses on the labelled recording. Every number
in it is a count over recorded frames; none of them is a quality claim. `--markdown` prints the
table a document quotes, so the two cannot drift apart.

    python -m tunnel_guard.run --experiment configs/real-gate-6tunnels.json     # the after run
    python -m tunnel_guard.corridor_report --before-six <dir> --after-six <dir> \
        --before-extended <dir> --after-extended <dir> --output results/<name>.json --markdown
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

HAZARDS = ("intersecting", "unresolved")


def load(root: Path) -> dict[str, list[dict]]:
    runs = {}
    for path in sorted(glob.glob(str(root / "*.jsonl"))):
        if "timing" in path:
            continue
        with open(path) as stream:
            runs[os.path.basename(path)[:-6]] = [json.loads(line) for line in stream]
    return runs


def summarize(rows: list[dict]) -> dict:
    statuses = Counter(row["status"] for row in rows)
    nearest = [row["nearest_obstacle_m"] for row in rows if row["nearest_obstacle_m"] is not None]
    hazards = [o for row in rows for o in row["objects"] if o["confirmed"] and o["path_relation"] in HAZARDS]
    reasons = Counter(o["path_relation_reason"] for row in rows for o in row["objects"])
    nearest_hazard = []
    for row in rows:
        candidate = [o for o in row["objects"] if o["confirmed"] and o["path_relation"] in HAZARDS]
        if candidate:
            nearest_hazard.append(min(candidate, key=lambda o: o["distance_m"]))
    return {
        "frames": len(rows),
        "status_frames": dict(statuses),
        "nearest_obstacle_m": ({"n": len(nearest),
                                "p10": round(float(np.percentile(nearest, 10)), 3),
                                "median": round(float(np.median(nearest)), 3),
                                "p90": round(float(np.percentile(nearest, 90)), 3)}
                               if nearest else {"n": 0}),
        "objects_per_frame_median": int(np.median([len(row["objects"]) for row in rows])),
        "hazard_observations": len(hazards),
        "nearest_hazard_support_voxels_median": (int(np.median([o["support_voxels"] for o in nearest_hazard]))
                                                 if nearest_hazard else None),
        "nearest_hazard_in_envelope_voxels_median": (int(np.median([o["in_envelope_voxels"] for o in nearest_hazard]))
                                                     if nearest_hazard else None),
        "object_reasons": dict(reasons),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-six", type=Path, required=True)
    parser.add_argument("--after-six", type=Path, required=True)
    parser.add_argument("--before-extended", type=Path, required=True)
    parser.add_argument("--after-extended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", action="store_true", help="print the README table instead of JSON")
    args = parser.parse_args()

    before_six, after_six = load(args.before_six), load(args.after_six)
    before_ext, after_ext = load(args.before_extended), load(args.after_extended)
    report = {
        "what": "Corridor-claim change of 2026-09-22: the intrusion claim now requires a measured "
                "coordinate and evidence that is not the tunnel's own cross-section, the tunnel's own "
                "cross-section is recognised per return, and the frame status distinguishes a claim "
                "from doubt. Recorded frames only; no quality claim.",
        "before": str(args.before_six.parent), "after": str(args.after_six.parent),
        "six_tunnels": {
            "before": summarize([row for rows in before_six.values() for row in rows]),
            "after": summarize([row for rows in after_six.values() for row in rows]),
            "per_recording": {
                name: {"before": summarize(before_six[name])["status_frames"],
                       "after": summarize(after_six[name])["status_frames"],
                       "before_nearest_median_m": summarize(before_six[name])["nearest_obstacle_m"]["median"],
                       "after_nearest_median_m": summarize(after_six[name])["nearest_obstacle_m"].get("median")}
                for name in sorted(before_six)},
        },
        "extended_corpus": {
            "before": summarize([row for rows in before_ext.values() for row in rows]),
            "after": summarize([row for rows in after_ext.values() for row in rows]),
        },
    }
    labelled = "doubleT_obstacle"
    if labelled in before_six and labelled in after_six:
        lost = [index for index, (a, b) in enumerate(zip(before_six[labelled], after_six[labelled]))
                if a["status"] == "obstacle" and b["status"] != "obstacle"]
        report["labelled_recording"] = {
            "frames_losing_obstacle_status": lost,
            "note": "Each frame still reports the object with its distance; only the status differs. "
                    "Frame 75 is the one the trailing-anchor guard costs, and the other two are frames "
                    "where the certified intrusion is three voxels, so instant confirmation - which now "
                    "reads interior geometry instead of whole-component density - does not fire.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        print(markdown(report))
    else:
        print(json.dumps(report, indent=2))


def markdown(report: dict) -> str:
    """The table the README quotes, generated from the same artifact."""
    lines = ["| | before | after |", "|---|---|---|"]
    for label, key in (("six tunnels, frames", "six_tunnels"), ("extended corpus, frames", "extended_corpus")):
        before, after = report[key]["before"], report[key]["after"]
        lines.append(f"| {label} | {before['frames']} | {after['frames']} |")
        for field in ("status_frames", "objects_per_frame_median", "hazard_observations",
                      "nearest_hazard_support_voxels_median"):
            left, right = before[field], after[field]
            if isinstance(left, dict):
                left = " ".join(f"{k.split('_')[0]} {v}" for k, v in sorted(left.items()))
                right = " ".join(f"{k.split('_')[0]} {v}" for k, v in sorted(right.items()))
            lines.append(f"| {field.replace('_', ' ')} | {left} | {right} |")
        lines.append(f"| nearest obstacle m (median) | {before['nearest_obstacle_m'].get('median')} | "
                     f"{after['nearest_obstacle_m'].get('median')} |")
    labelled = report.get("labelled_recording")
    if labelled:
        lines.append(f"| labelled recording, frames losing obstacle status | - | "
                     f"{labelled['frames_losing_obstacle_status']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
