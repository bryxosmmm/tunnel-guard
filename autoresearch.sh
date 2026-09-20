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

rm -rf "$RUN" "$REPORT"
"$PY" -m tunnel_guard.realistic_stress --experiment "$EXPERIMENT" >/dev/null
"$PY" -m tunnel_guard.realistic_report --run "$RUN" --output "$REPORT" >/dev/null

"$PY" - "$REPORT" <<'PY'
import json, sys
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
for name, value in rows:
    if value is None:
        raise SystemExit(f"METRIC {name} is None: the panel did not produce it")
    print(f"METRIC {name}={value}")
PY
