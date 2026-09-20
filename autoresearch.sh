#!/usr/bin/env bash
# Canonical benchmark for this session: misses and nuisance detections of the detector on scenes
# whose rays, surfaces, occlusion, intensity and ego motion come from a real recording, with
# objects of several types inserted on the recorded track.
#
# Prints METRIC lines only. Exits non-zero if any stage fails, so a broken run can never be
# mistaken for a good score.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
EXPERIMENT="configs/realistic-autoresearch-20260920.json"
RUN="build/autoresearch-panel"
REPORT="build/autoresearch-report.json"
GATE_EXPERIMENT="configs/real-gate-20260920.json"
GATE_RUN="build/autoresearch-real"

rm -rf "$RUN" "$REPORT" "$GATE_RUN"
"$PY" -m tunnel_guard.realistic_stress --experiment "$EXPERIMENT" >/dev/null
"$PY" -m tunnel_guard.realistic_report --run "$RUN" --output "$REPORT" >/dev/null
# The synthetic panel cannot see what a change costs on real clutter. This replay is the gate: a
# change that lowers the obstacle-frame count on the labelled recording is rejected whatever the
# panel says, so the number has to be in the same METRIC stream as the panel's own.
"$PY" -m tunnel_guard.run --experiment "$GATE_EXPERIMENT" >/dev/null

"$PY" - "$REPORT" "$GATE_RUN" <<'PY'
import json, sys
from collections import Counter

report = json.load(open(sys.argv[1]))
recall = report["recall"]
counts = recall["counts"]
nuisance = report["nuisance"]
rows = [
    # Primary: share of the objects whose own returns were reported by a comparably sized box.
    # Higher is better. False negatives are the fatal error, so misses are the headline.
    ("recall_visible_objects", recall["point_detected_of_frames_with_returns"]),
    ("missed_visible_objects", counts["with_returns"] - counts["point_detected"]),
    ("objects_with_returns", counts["with_returns"]),
    ("objects_no_returns_visibility_outcome",
     report["objects"]["no_returns_visibility_outcome"]),
    # The box-IoU view, which is what a scorer comparing boxes would see.
    ("strict_iou_recall_visible", recall["detected_of_visible"]),
    # Nuisance detections on frames where nothing was injected, under both scorer conventions.
    ("injection_free_frames_claiming_hazard", nuisance["frames_claiming_an_obstacle"]),
    ("unexplained_hazard_objects", nuisance["unexplained_hazard_objects_on_injected_frames"]),
]
# Real-recording gate. The strict obstacle count and the any-hazard count are reported separately:
# the rejected floor change moved 14 frames from obstacle to unresolved_obstacle, so a sum of the
# two would have hidden exactly the cost this gate exists to catch.
for bag in ("doubleT_obstacle", "roundT_doubleT"):
    statuses, objects = Counter(), 0
    with open(f"{sys.argv[2]}/{bag}.jsonl") as stream:
        for line in stream:
            row = json.loads(line)
            statuses[row["status"]] += 1
            objects += len(row["objects"])
    rows.append((f"{bag}_strict_obstacle_frames", statuses["obstacle"]))
    rows.append((f"{bag}_any_hazard_frames",
                 statuses["obstacle"] + statuses["unresolved_obstacle"]))
    rows.append((f"{bag}_objects", objects))

for name, value in rows:
    if value is None:
        raise SystemExit(f"METRIC {name} is None: the run did not produce it")
    print(f"METRIC {name}={value}")
PY
