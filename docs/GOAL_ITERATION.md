# Continuous iteration: real-data coverage and usable execution

Completed 2026-09-17. **All six supplied recordings were processed completely:**
2,488 unique scans, plus the separately identified runtime and hybrid experiments.
The result is a usable research runtime/review tool, not a validated warning system.

## Final full-corpus comparison

Machine-readable evidence: [real comparison](../results/goal-real-20260917.json) and
[runtime / experiments](../results/goal-runtime-20260917.json).

| Recording | Scans | Obstacle before → after | Unresolved after | Geometry valid | Motion valid | p50 ms after |
|---|---:|---:|---:|---:|---:|---:|
| doubleT_obstacle | 201 | 141 → 141 | 60 | 201 | 200 | 641 |
| doubleT_platform | 345 | 87 → 87 | 258 | 345 | 240 | 624 |
| roundT_doubleT | 252 | 69 → 69 | 182 | 252 | 251 | 605 |
| roundT_pressureGate_roundT | 268 | 75 → 75 | 193 | 268 | 266 | 777 |
| roundT_squareT_pressureGate_squareT | 545 | 162 → 163 | 378 | 544 | 516 | 594 |
| squareT_platform_squareT_switch | 877 | 2 → 2 | 873 | 877 | 804 | 534 |

No missing/extra frames or measurement-identity mismatches. No future or duplicate
timestamps were found in either object or intersection histories across all 2,488
final frames. Status changes in exactly one frame (square-gate frame 503); the
gate geometry failure at frame 473 remains `unknown`. Five recordings retain all
their frame-level statuses.

This is **not full object-output equivalence** to parallel baseline: objects/IDs
differ on 188 round-to-double frames, 419 square-gate frames and 823 switch frames.
All changed `geometry` subfields are background summaries; rail/ground geometry
fields are identical. Background changes can propagate through association. Switch distance differs on
one frame. Do not interpret one additional obstacle-status frame as improved
recall, or equal alarm counts as correct detections. The original serial graph
control and fixed-seed repeatability experiments isolate the intended changes.

Candidate observations at ≥60 m total 130,216 before and 130,237 after. They are
repeated observations including infrastructure, not unique obstacles or field
recall. Sparse-support thresholds were not weakened or raised. The real obstacle
recording, including its previously inspected far candidate, retains all compared
non-timing outputs.

Native p50 processing spans 534–777 ms across these recordings. Concurrent work,
different baseline visualization settings and warm-up prevent a controlled speedup
claim. This still exceeds a 100 ms / 10 Hz budget. All 2,487 geometry-available
frames remain degraded by unverified sensor/extrinsics/timing; the remaining frame
is unavailable. No clear-route certification is produced.

## Fixed starting point

Branch `experiments/morev`, clean tree at
`3f83aebc7cf94b6e8bca4dce1d82d5df8f15f480`. Frozen Python source and configuration:
`build/goal-baseline-source/`. Current changes are incremental; no replacement
project or rewritten detector is introduced. No subagents or automated tests.
Verification uses actual detector runs, saved real measurements, ROS messages,
and inspection of displayed output.

Existing complete baseline runs can be reused without claiming they were rerun:

- `build/intersection-evidence-after/doubleT_obstacle.jsonl`: 201 frames.
- `build/coverage-expansion-real/doubleT_platform.jsonl`: 345 frames.
- `build/coverage-expansion-real/roundT_doubleT.jsonl`: 252 frames.

All three configurations match the frozen configuration, and the bytes of
`detector.py`, `geometry.py`, `background.py`, `segmentation.py`, and `io.py` match
the frozen source. The two gate recordings (268 and 545 scans) and switch recording (877 scans)
were processed completely from that frozen source. Baseline coverage is therefore
all six recordings, 2,488 scans. Sequential extraction kept within local disk space. Only newly extracted temporary copies
under `build/goal-data` may be released after their runs; the source archive and
existing user data are retained.

## Implemented and exercised changes

1. `segmentation.density_labels`: query each point's actual adaptive radius;
   retain the same mutual-radius edges, rather than first enumerating every pair
   at the largest radius in the entire cloud. No thresholds or support criteria
   change. On six actual saved stage clouds (platform and round-to-double frames
   0/100/200), label and core masks are identical. Single-pass summed stage time
   4.404 s → 1.728 s; one individual cloud is slower. Concurrent workloads mean
   these are diagnostic timings, not a controlled throughput benchmark.
2. `run.main`: stream full rows to JSONL and retain only scalar summary fields
   plus geometry/motion validity. Full object histories and covariances no longer
   accumulate redundantly in memory. Source and output snapshots remain intact.
3. `review_viewer`: complete-run local browser navigation, exact original
   measurement selection, XY/XZ cloud views with metric scale, contour, boxes,
   confirmation state, distance witnesses, data-quality reasons, pan/zoom and
   object focus. No CDN or cloud upload. Display sampling never enters inference.
4. `ResultMessages` shares visualization construction between recorded bags and
   a ROS2 adapter. Displayed confirmed-intersection distance and uncertain-object
   distance are separate; the legacy JSON distance field is retained. The live
   adapter bounds its queue, rejects repeated/backward acquisition time, reports
   processing overruns and clears output to `unknown` after input silence.
5. Humble/Python 3.10 packaging, container/launch recipe, and explicit provisional
   Q&A contour. See [Q&A implications](QA_IMPLICATIONS.md) and
   [run and review instructions](RUN_AND_REVIEW.md). Default contour and motion
   validity thresholds remain unchanged.

## Completed focused evidence

### Motion-reference hypothesis

`python -m tunnel_guard.motion_audit` processed **all 345 platform scans**.
Registration poses match the prior baseline within `7.55e-15`; original validity
flags match on every frame. The first frame has no preceding measurement.
On the remaining 344, sparse previous-frame reference rejects 104 registrations;
using the denser previous-frame sample with identical poses/thresholds rejects
66. Thirty-eight change to valid; none change in the opposite direction.

This demonstrates sampling sensitivity, **not validated pose accuracy**. There
is no motion ground truth; a smaller nearest-neighbor residual can still hide
wrong tunnel-axis motion. The production motion gate is not changed. Evidence:
`build/goal-motion-dense-reference/{motion.jsonl,summary.json,baseline-pose-comparison.json}`.

### Runner and shared display

`configs/goal-runner-real-prefix.json` ran the actual detector on ten original
`doubleT_obstacle` scans and wrote the existing RViz result bag. Status and geometry
match baseline exactly. Object, motion and pose fields match within absolute and
relative tolerance `1e-10`; largest float difference `2.18e-14` in estimated velocity.
All ten recorded ROS status messages contain distance summaries matching their
actual result rows. Full summary was recomputed from JSONL for comparison.
These ten scans validate the integration prefix, not complete-recording coverage.

### Browser inspection

The browser displayed real platform frame 0, stepped to frame 1, and sought directly
to frame 344 using its number field and Go button. Acquisition stamps remained
exact strings, avoiding JavaScript integer rounding. Screenshots were inspected
locally; source clouds/screenshots are not published to Git.

## Failed / superseded attempts retained

- A first parallel after-run was intentionally interrupted and kept at
  `build/goal-after-available-superseded-memory-runner/`, with its log and explicit
  `INCOMPLETE.json`. It is not accepted as a full comparison. This motivated the
  bounded summary retention above.
- Initial Docker build exited successfully but produced an `UNKNOWN` wheel without
  detector dependencies due to the old packaging environment. Actual inspection
  exposed the failure; it is not a working runtime result.
- Corrected native Linux ARM64 build failed: official Open3D 0.19 wheel is absent
  ([upstream issue](https://github.com/isl-org/Open3D/issues/7130)). The target
  Intel/Ubuntu `linux/amd64` build is used; no implicit library downgrade.
- First local HTTP start was denied by the shell sandbox; loopback-only restart
  with local execution permission succeeded. No external service is exposed.

## Commands and artifacts

Executed absolute baseline recipe is retained in
`build/goal-motion-audit/baseline-invocation.json`; the checked-in gate recipe is
portable when invoked from the frozen source directory:

```bash
(cd build/goal-baseline-source && ../../.venv-iteration/bin/python \
  -m tunnel_guard.run --experiment ../../configs/goal-baseline-gates.json)
.venv-iteration/bin/python -m tunnel_guard.motion_audit \
  --bag data/sourcecraft_subset/for_hackathon/doubleT_platform \
  --config configs/detector-native.json --output build/goal-motion-dense-reference
.venv-iteration/bin/python -m tunnel_guard.run \
  --experiment configs/goal-runner-real-prefix.json
docker build --platform linux/amd64 -t tunnel-guard:goal-amd64 .
uv lock
```

Existing output directories are intentionally not overwritten. Use a new output
path in a copied recipe to repeat a run. `uv lock` was updated for Python 3.10
compatibility without changing the active macOS environment.

## Full-run discrepancy and root cause

The first optimized full run (`build/goal-after-available`) completed 1,611 scans.
The obstacle, platform and round-gate recordings match their original decisions.
The round-to-double and round-to-square-gate recordings each change **one status**;
IDs differ after earlier small candidate changes. The original equivalence
criterion therefore **did not pass** for that run. The result is retained.

The earliest round discrepancy is frame 68: 22,201 versus 22,205 points enter
clustering, so the difference already exists upstream of the optimized neighbor
query. Repeated processing of the same real scans with fresh detector state and
the same seed reproduced nondeterminism: frame 76 had two different retained-point
sets; frame 232 had three. The geometry voxel input was identical in every repeat.

The [Open3D 0.19 implementation](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudSegmentation.cpp)
pre-generates samples, but evaluates them in parallel and changes its stopping
limit as results arrive. A fixed random seed alone does not fix which proposals
finish before that limit. This agrees with the observed retained-point variation.

`TunnelBackground.__init__` now limits OpenMP to one thread **only around
`segment_plane`**, using the loaded-library controller. Other geometry operations
and the configured KISS registration threading remain available. The recipe
explicitly sets `background.ransac_threads=1`; higher values remain a research
option with the repeatability limitation. `threadpoolctl==3.7.0` is now an explicit
dependency (BSD-3-Clause; the version already present in the native environment).

On frames 68/76/124/232, four fresh-state repeats per frame now produce identical
geometry voxel arrays, retained context, clustering inputs and labels. The
experiment with `OMP_NUM_THREADS=1` agrees with this stage-local fix. These are
real-scan repeatability measurements, not ground-truth detection measurements.
Evidence is in `build/goal-background-repro*`. The original-algorithm serial
control completed all 252 round-to-double scans. Compared with the final optimized
run, every discrete decision, ID, geometry and motion field agrees; floats agree
within `1e-10` (largest compared difference `6.13e-13`). This isolates the graph
optimization from the original parallel RANSAC instability. It is not evidence
that every old parallel-baseline candidate must remain identical.

## ROS delivery findings

The complete runtime chain has been exercised on the **first ten unchanged
serialized PointCloud2 messages** of `doubleT_platform`:

- best-effort subscriber: 1/10 received in the local emulated setup;
- reliable subscriber: 9/10, with the final message missing when the publisher
  exited;
- reliable plus `ros2 bag play --wait-for-all-acked 30000`: **10/10**, correct
  acquisition stamps, no duplicates and no future/duplicate intersection evidence.

The recorder captured actual PointCloud2, MarkerArray and JSON status messages.
After input stopped, the watchdog published `unknown`, an empty cloud and a
marker clear. The final complete replay had two timeout messages: before playback
and after playback. This does not measure native Intel throughput: the container
runs AMD64 under emulation on Apple Silicon, with explicit OpenMP=2/BLAS=1 settings.
The package imports and `pip check` also succeeded in the image. Results/logs:
`build/goal-ros-validation` (first failed delivery), `build/goal-ros-reliable`
(partial and complete delivery, separate recorded bags). The final image was then rebuilt with the RANSAC fix and replayed again.
`build/goal-ros-final/report.json` records **10/10** exact unique source stamps,
zero missing/unexpected stamps, zero temporal-history violations, 12 clouds and
12 MarkerArrays (ten measurements plus two watchdog clears). All ten scene
statuses are `unresolved_obstacle`. Final `pip check` reported no broken
requirements; package versions and image ID are retained. Container processing
was 1.45–4.87 s per scan under emulation and concurrent load; no real-time claim.

## Existing hybrid panel repeated

Both existing insertion recipes were rerun on final code: nine platform cases
and three obstacle-background cases, ten scans each. All per-case summary fields
match their previous results exactly (`build/goal-motion-audit/hybrid-comparison.json`).
The small 60 m box still produces a candidate in 1/10 frames and no confirmed
intersection; the upright 60 m platform box has candidates in 10/10 and confirmed
intersections in 7/10. Platform 100 m cases have no modeled returns. These remain
controlled opaque-box insertions using measured ray directions, not real hazards
or physical detection-range validation. No new model or synthetic training was
introduced.

## Review findings that remain unresolved

- `TrackGeometry._rail_profile`: raised railhead support and a locally fitted
  reference path do not establish the vehicle's swept envelope, flush gate rails,
  or the selected branch of a switch. The provisional Q&A rectangle is a separate
  profile; it ran on ten real obstacle scans (1 obstacle, 8 unresolved, 1 candidate).
  Narrowing it is not credited as a detector improvement.
- `Detector._motion` / KISS registration: internal motion estimates have no
  independent reference. Map update precedes the downstream validity gate;
  rejecting a pose does not undo that map update. Changing this requires a
  controlled map-state experiment; no threshold relaxation was accepted.
- `io.decode_cloud` and sensor configuration: processing axes, mounting and
  point timing are not independently calibrated. Deskew stays disabled and
  quality stays degraded. Two returns from the same firing are not two new frames.
- `cluster_candidates` and `_associate`: measured support boxes are not full
  object shapes; fragmented infrastructure and ID changes remain. Sparse far
  candidates must not be removed merely to make the output look clean.
- Evaluation: the person panel is one event with detector-assisted propagation,
  not exhaustive independent truth. Current alarm counts cannot establish
  precision, recall, false-alarm rate or safety.

The next validation step needs independent event/negative-interval labels and
per-recording sensor/vehicle geometry. Target CPU profiling and live queue loss
also remain necessary before 10 Hz delivery can be claimed. ML is not yet justified
as the solution to these missing measurements.

### Installed CLI and cross-platform boundary

A follow-up offline run inside the image exposed a packaging bug: provenance
collection unconditionally invoked Git in a directory without `.git`. The failed
invocation is retained in `build/goal-ros-final/offline-failed.log` and `offline/`.
`run.git_revision` now returns `null` outside a checkout, skips the unavailable
working-tree patch, and still records configuration/source hashes and snapshots.
A missing checkout is not represented as a fabricated clean revision.

The ten ROS display clouds match the corresponding transformed, range-filtered,
display-sampled source coordinates **exactly** as float32 arrays. The native and
container scene statuses and geometry/motion validity agree on all ten frames,
but candidate counts differ on nine. Platform/dependency differences therefore
remain a separate evaluation boundary; no cross-platform candidate equivalence
is claimed. Evidence: `build/goal-ros-final/coordinate-comparison.json`.

The corrected container offline CLI completed all ten scans. Within the same
Linux image, offline and ROS outputs agree in status, distance, all objects/IDs,
geometry, motion, pose, observability and gap-reset state within `1e-10`; largest
compared float difference `2.07e-14`. This isolates the adapter from platform
changes. The final image differs from the live-replay image only in offline
provenance handling and an additional experiment recipe; perception/ROS code is
unchanged. See `build/goal-ros-final/adapter-comparison.json`.

## Commands executed for this iteration

In addition to the commands above (all from repository root unless noted):

```bash
# Frozen baseline, cwd build/goal-baseline-source:
../../.venv-iteration/bin/python -m tunnel_guard.run --experiment ../../configs/goal-baseline-switch.json
OMP_NUM_THREADS=1 ../../.venv-iteration/bin/python -m tunnel_guard.run --experiment ../../configs/goal-baseline-round-serial.json

# Native detector and actual-scan experiments:
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/goal-after-available.json
.venv-iteration/bin/python -m tunnel_guard.background_repro --experiment configs/goal-background-repro.json
OMP_NUM_THREADS=1 .venv-iteration/bin/python -m tunnel_guard.background_repro --experiment configs/goal-background-repro-serial.json
.venv-iteration/bin/python -m tunnel_guard.background_repro --experiment configs/goal-background-repro-fixed.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/goal-final-real.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/goal-qa-profile-prefix.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/goal-final-cli-prefix.json
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.injection_experiment --experiment configs/goal-ray-insertion-expansion.json
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.injection_experiment --experiment configs/goal-ray-insertion-obstacle-background.json

# Inside the Humble image, unchanged real ten-scan bag mounted at /data:
ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py input_topic:=/lidar_points input_reliability:=reliable input_timeout_s:=15.0
ros2 bag record -o /evidence/ros_output /perception/status /perception/points_display /perception/debug_markers
ros2 bag play /data --rate 0.02 --delay 3 --wait-for-all-acked 30000
python3 -m pip check
python3 -m tunnel_guard.run --experiment /recipe.json

# Interactive native viewer:
.venv-iteration/bin/python -m tunnel_guard.review_viewer --run build/goal-final-real --bag data/sourcecraft_subset/for_hackathon/doubleT_platform --port 8766
```

`/recipe.json` was the mounted `configs/goal-ros-offline-prefix.json`; `/evidence`
was the local `build/goal-ros-final` directory. The ten-scan bag was made by copying
unchanged serialized source messages and original timestamps, not reconstructing
or synthesizing clouds. Its local provenance is retained.

Historical recipes above ran against their then-current source/configuration.
**The snapshots and hashes in each output directory are authoritative**; rerunning
an old recipe against today's changed default is a new experiment. The final
CLI recipe's checked-in criterion was clarified to say "ten-scan prefix" after
its execution; data-selection and detector settings are unchanged. No previous
output directory or failed result was overwritten.

The final native CLI prefix also completed ten actual scans after the provenance
fix. Decisions/objects/geometry/motion/pose/distances match the full final run;
Git revision, status and working-tree patch are present. Its real RViz bag is
retained in `build/goal-final-cli-prefix/`. This exercises the checkout branch
of provenance handling, separately from the installed-image branch.

Final report command completed successfully:

```bash
.venv-iteration/bin/python -m tunnel_guard.panel_report --panel configs/goal-final-comparison.json --output results/goal-real-20260917.json
git diff --check
```

No automated tests or test suites were created or run. Verification above is
actual detector execution, recorded-message inspection and interactive review.
