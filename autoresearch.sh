#!/usr/bin/env bash
# Canonical benchmark for this session, segment 4: EVERY hackathon tunnel, more frames.
#
# Segment 3 measured the primary metric on six frames (40-45) of a single tunnel, doubleT_obstacle,
# and replayed two tunnels in the gate. A second tunnel measured out of band showed the guarantee does
# not transfer: point recall held (0.935 vs 0.975) while the box view collapsed (0.444 vs 0.818),
# traceable to track continuity rather than to the corridor. The panel therefore now runs on all six
# tunnels with twenty-four consecutive frames per case, and the gate replays every frame each
# recording holds.
#
# Stages, in order, each of which must succeed or the run fails:
#   1. pose run      - one row per frame per tunnel carrying the sensor pose and the fitted rail
#                      anchors. Also the cheap smoke test: a bag that will not decode fails here.
#   2. panel         - per tunnel: insert objects into real frames on the recorded track and report.
#   3. gate          - replay every tunnel end to end and report the hazard load.
#   4. pool          - sum the six reports into one primary metric, and report each tunnel separately.
#
# Parallelism (2026-09-20). The three workload stages are per-tunnel independent: a tunnel's frames,
# poses, geometry, tracker and rng never read another tunnel's, and every stage writes one file per
# tunnel named after that tunnel. Each unit therefore runs in its own process with its own recipe and
# its own output directory, and the shared directories the later stages read are assembled from
# symlinks to those per-tunnel files. No recipe, frame, seed or metric definition changes - only the
# order in which independent work executes changes - so the pooled numbers are the numbers the serial
# harness produced. Measured on this machine: the panel stage took 19.2 min of detector time serially
# and 6.4 min with the six tunnels concurrent.
#
# Prints METRIC lines only. Exits non-zero if any stage fails, so a broken run can never be mistaken
# for a good score.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
POSE_RECIPE="configs/pose-6tunnels.json"
POSE_RUN="build/pose-6tunnels"
POSE_PARTS="build/pose-parts"
PANEL_TEMPLATE="configs/panel-6tunnels-template.json"
PANEL_ROOT="build/autoresearch-panels"
REPORT_ROOT="build/autoresearch-reports"
GATE_EXPERIMENT="configs/real-gate-6tunnels.json"
GATE_RUN="build/autoresearch-real"
GATE_PARTS="build/gate-parts"

# Independent tunnels launched at once. Each process already threads its own odometry (8) and
# nearest-neighbour queries (all cores), so this is the point where more units stop paying.
JOBS="${AR_JOBS:-6}"

rm -rf "$POSE_RUN" "$POSE_PARTS" "$PANEL_ROOT" "$REPORT_ROOT" "$GATE_RUN" "$GATE_PARTS"
mkdir -p "$PANEL_ROOT/recipes" "$REPORT_ROOT" "$POSE_RUN" "$GATE_RUN"

mapfile -t BAGS < <("$PY" -c "
import json
print('\n'.join(b['path'].split('/')[-1] for b in json.load(open('$POSE_RECIPE'))['bags']))")

# Launch one shell command per line of work, at most JOBS at a time, and fail the stage if any fails.
run_parallel() {
  local -a pids=()
  local failed=0
  for command in "$@"; do
    bash -c "$command" &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "$JOBS" ]; then
      wait "${pids[0]}" || failed=1
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]:-}"; do
    [ -n "${pid:-}" ] && { wait "$pid" || failed=1; }
  done
  if [ "$failed" -ne 0 ]; then
    echo "a parallel stage command failed" >&2
    exit 1
  fi
}

# One recipe per tunnel for a multi-bag recipe, so each tunnel can run as its own process.
split_recipe() {
  "$PY" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
source, destination = sys.argv[1], sys.argv[2]
recipe = json.loads(Path(source).read_text())
root = Path(destination)
root.mkdir(parents=True, exist_ok=True)
for entry in recipe["bags"]:
    bag = entry["path"].split("/")[-1]
    part = dict(recipe, bags=[entry], output=f"{root}/{bag}",
                note=f"per-tunnel part of {source} for {bag}, run as its own process")
    (root / f"{bag}.json").write_text(json.dumps(part, indent=2) + "\n")
PY
}

# 1. Poses and rail anchors per tunnel, one process per tunnel, then expose them as one directory.
split_recipe "$POSE_RECIPE" "$POSE_PARTS"
POSE_COMMANDS=()
for bag in "${BAGS[@]}"; do
  POSE_COMMANDS+=("$PY -m tunnel_guard.run --experiment '$POSE_PARTS/$bag.json' >/dev/null")
done
run_parallel "${POSE_COMMANDS[@]}"
for bag in "${BAGS[@]}"; do
  ln -sfn "$PWD/$POSE_PARTS/$bag/$bag.jsonl" "$POSE_RUN/$bag.jsonl"
done

# 2. One panel recipe per tunnel, then insert and report on each. The insertions are the expensive
#    half and run concurrently; reporting one tunnel reads only that tunnel's output.
PANEL_COMMANDS=()
for bag in "${BAGS[@]}"; do
  "$PY" - "$PANEL_TEMPLATE" "$PANEL_ROOT/recipes/$bag.json" "$bag" "$POSE_RUN" "$PANEL_ROOT/$bag" <<'PY'
import json, sys
from pathlib import Path

template, destination, bag, pose_run, output = sys.argv[1:]
Path(destination).parent.mkdir(parents=True, exist_ok=True)
recipe = json.load(open(template))
recipe.update(background_bag=f"data/sourcecraft_subset/for_hackathon/{bag}",
              pose_run=pose_run, output=output, note=f"generated from {template} for {bag}")
with open(destination, "w") as stream:
    stream.write(json.dumps(recipe, indent=2) + "\n")
PY
  PANEL_COMMANDS+=("$PY -m tunnel_guard.realistic_stress --experiment '$PANEL_ROOT/recipes/$bag.json' >/dev/null")
done
run_parallel "${PANEL_COMMANDS[@]}"
REPORT_COMMANDS=()
for bag in "${BAGS[@]}"; do
  REPORT_COMMANDS+=("$PY -m tunnel_guard.realistic_report --run '$PANEL_ROOT/$bag' --output '$REPORT_ROOT/$bag.json' >/dev/null")
done
run_parallel "${REPORT_COMMANDS[@]}"

# 3. Gate: every frame of every tunnel, one process per tunnel, exposed as one directory.
split_recipe "$GATE_EXPERIMENT" "$GATE_PARTS"
GATE_COMMANDS=()
for bag in "${BAGS[@]}"; do
  GATE_COMMANDS+=("$PY -m tunnel_guard.run --experiment '$GATE_PARTS/$bag.json' >/dev/null")
done
run_parallel "${GATE_COMMANDS[@]}"
for bag in "${BAGS[@]}"; do
  ln -sfn "$PWD/$GATE_PARTS/$bag/$bag.jsonl" "$GATE_RUN/$bag.jsonl"
done

# 4. Metrics: pooled primary plus per tunnel, then the gate.
"$PY" - "$POSE_RECIPE" "$REPORT_ROOT" "$GATE_RUN" <<'PY'
import json, sys
from collections import Counter

import numpy as np

pose_recipe, report_root, gate_run = sys.argv[1], sys.argv[2], sys.argv[3]
bags = [b["path"].split("/")[-1] for b in json.load(open(pose_recipe))["bags"]]

rows, per_tunnel = [], []
for bag in bags:
    report = json.load(open(f"{report_root}/{bag}.json"))
    counts = report["recall"]["counts"]
    objects = report["objects"]
    nuisance = report["nuisance"]
    frames = report["frames"]
    per_tunnel.append((bag, counts, objects, nuisance, frames))
    rows.append((bag,
                 counts["with_returns"] - counts["point_detected"],
                 counts["with_returns"],
                 objects["no_returns_visibility_outcome"],
                 counts["strict_detected"],
                 frames["without_injection"],
                 nuisance["frames_claiming_an_obstacle"],
                 nuisance["unexplained_hazard_objects_on_injected_frames"],
                 counts["injected"]))

injected = sum(r[8] for r in rows)
with_returns = sum(r[2] for r in rows)
missed = sum(r[1] for r in rows)
point_detected = with_returns - missed
strict_detected = sum(r[4] for r in rows)
no_returns = sum(r[3] for r in rows)
negative_frames = sum(r[5] for r in rows)
alarm_frames = sum(r[6] for r in rows)
unexplained = sum(r[7] for r in rows)

metrics = [
    # Primary: pooled over all six tunnels. Higher is better; false negatives are the fatal error.
    ("recall_visible_objects", point_detected / with_returns),
    ("missed_visible_objects", missed),
    ("objects_with_returns", with_returns),
    ("objects_injected", injected),
    ("objects_no_returns_visibility_outcome", no_returns),
    # The box-IoU view pooled the same way: what a scorer comparing boxes would see.
    ("strict_iou_recall_visible", strict_detected / with_returns),
    ("injection_free_frames", negative_frames),
    ("injection_free_frames_claiming_hazard", alarm_frames),
    ("unexplained_hazard_objects", unexplained),
]
# Per tunnel, so a pooled number cannot hide one tunnel carrying or breaking the result.
for bag, counts, objects, nuisance, frames in per_tunnel:
    metrics.append((f"recall_{bag}",
                    counts["point_detected"] / counts["with_returns"] if counts["with_returns"] else 0.0))
    metrics.append((f"with_returns_{bag}", counts["with_returns"]))
    metrics.append((f"injected_{bag}", counts["injected"]))

for bag in bags:
    statuses, objects = Counter(), 0
    with open(f"{gate_run}/{bag}.jsonl") as stream:
        for line in stream:
            row = json.loads(line)
            statuses[row["status"]] += 1
            objects += len(row["objects"])
    metrics.append((f"gate_frames_{bag}", sum(statuses.values())))
    metrics.append((f"gate_strict_obstacle_frames_{bag}", statuses["obstacle"]))
    metrics.append((f"gate_any_hazard_frames_{bag}",
                    statuses["obstacle"] + statuses["unresolved_obstacle"]))
    metrics.append((f"gate_objects_{bag}", objects))


def overlap(first, second):
    inter = 1.0
    for axis in range(3):
        low = max(first["bbox_min"][axis], second["bbox_min"][axis])
        high = min(first["bbox_max"][axis], second["bbox_max"][axis])
        inter *= max(0.0, high - low)
    if inter <= 0:
        return 0.0
    va = float(np.prod([first["bbox_max"][a] - first["bbox_min"][a] for a in range(3)]))
    vb = float(np.prod([second["bbox_max"][a] - second["bbox_min"][a] for a in range(3)]))
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


# The annotated recording is one tunnel; the label's own provenance says the box was propagated from
# the detector's own track, and the annotated object lies outside the reference contour, so the
# overlap below is reported and never claimed as verification.
try:
    annotations = json.load(open("annotations/doubleT-obstacle-person.json"))
except FileNotFoundError:
    annotations = None
if annotations is not None:
    labelled = {f["frame"]: f["objects"][0] for f in annotations["frames"]}
    rows_by_frame = {}
    with open(f"{gate_run}/doubleT_obstacle.jsonl") as stream:
        for line in stream:
            row = json.loads(line)
            rows_by_frame[row["frame"]] = row
    hazards, claimed, overlaps = 0, 0, []
    for frame_index, truth in labelled.items():
        row = rows_by_frame.get(frame_index)
        if row is None:
            continue
        if row["status"] in ("obstacle", "unresolved_obstacle"):
            claimed += 1
        hazard_objects = [o for o in row["objects"]
                          if o["confirmed"] and o["path_relation"] in ("intersecting", "unresolved")]
        if hazard_objects:
            hazards += 1
        overlaps.append(max((overlap(o, truth) for o in hazard_objects), default=0.0))
    metrics.append(("annotated_frames", len(labelled)))
    metrics.append(("annotated_frames_claiming_hazard", claimed))
    metrics.append(("annotated_frames_with_confirmed_hazard_object", hazards))
    metrics.append(("annotated_median_best_iou", round(float(sorted(overlaps)[len(overlaps) // 2]), 4)))

for name, value in metrics:
    if value is None:
        raise SystemExit(f"METRIC {name} is None: the run did not produce it")
    print(f"METRIC {name}={value}")

# Nuisance ceilings, frozen at the segment baseline of 2026-09-21. The session goal counts
# infrastructure and adjacent nuisance detections as errors, so a recall gain bought with a higher
# nuisance load is not a win - and until now nothing in the harness said so, which is how a run
# that raised the coverage metric by reporting more objects could be kept while the load grew.
# The tolerances sit far above the measured noise floor (4 objects in 150156 unexplained, threaded
# float rounding) and below every trade measured so far, so they block recall-for-nuisance trades
# without failing on run-to-run jitter. Raising a ceiling is a deliberate act that belongs in a
# segment bump with its justification written down, never a quiet side effect of a kept run.
CEILINGS = {
    "unexplained_hazard_objects": 151138 * 1.002,
    "gate_objects_doubleT_obstacle": 60371 * 1.01,
    "gate_objects_doubleT_platform": 105018 * 1.01,
    "gate_objects_roundT_doubleT": 66509 * 1.01,
    "gate_objects_roundT_pressureGate_roundT": 91524 * 1.01,
    "gate_objects_roundT_squareT_pressureGate_squareT": 156938 * 1.01,
    "gate_objects_squareT_platform_squareT_switch": 256540 * 1.01,
}
violations = [(name, value, CEILINGS[name]) for name, value in metrics
              if name in CEILINGS and value > CEILINGS[name]]
for name, value, ceiling in violations:
    print(f"CEILING {name}={value} exceeds {ceiling:.1f} "
          f"({100.0 * (value / ceiling - 1.0):+.2f}% over)")
if violations:
    raise SystemExit(f"nuisance ceiling exceeded on {len(violations)} metric(s): "
                     + ", ".join(name for name, _, _ in violations))
PY
