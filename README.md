# Tunnel Guard

Class-agnostic LiDAR obstacle-detection baseline for metro tunnels. Reads ROS 2 PointCloud2 bags directly; no ROS installation, Docker, or pretrained weights required for the default pipeline.

**Research baseline, not a validated collision-warning system.** Recall, infrastructure alarms, generalization, and runtime remain unresolved. `CASE.md` contains the original requirements. Recorded RViz2 export and a live ROS2 Humble adapter are implemented. The AMD64 Humble container processed a ten-scan real replay under Apple Silicon emulation; native target throughput and the RViz GUI remain unverified.

The initial review and its two real 30-frame prefixes are documented in [docs/AUDIT.md](docs/AUDIT.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Subsequent iterations and historical results below are separate evidence.

## Extended dataset: initial real runs

See [archive inventory and first comparison](docs/EXTENDED_FIRST_LOOK.md): 11,271 clouds in 221 segments, about 84 GiB unpacked. Three fixed segments (153 clouds) were processed by current and previous geometry without tuning. Both fail on the same one frame; no labelled accuracy is established. The initial sample extracted about 1.14 GiB; after disk cleanup, [the complete recording is now extracted and all acquisition headers audited](docs/EXTENDED_FULL_INGEST.md). [Continuous detector inference now covers all 11,271 clouds](docs/EXTENDED_FULL_RUN.md): 11,250 frames with supported geometry, 21 unavailable; median processing 132 ms on this Mac. No labelled accuracy is established.

## Usable runtime and full-corpus iteration

The [Gerasimov/HMM-MOS review and scenario runs](docs/REVIEW_GERASIMOV_20260918.md) add reproducible moving, stopped and appearing-object scenes for the actual detector. A first-rail-heading experiment was replayed on all 11,271 real clouds for geometry and 951 clouds end to end. It remains opt-in: nine geometry failures recovered, one new failure and unresolved path-selection changes. HMM-MOS is not used to suppress stationary obstacles.

The [3D track and clearance literature review](docs/TRACK_GEOMETRY_LITERATURE_20260918.md) maps published rail-pair estimation and local clearance coordinates to the remaining curve, grade and cant limitations. It distinguishes proposed adaptations from implemented and evaluated behavior.

[The first 3D geometry iteration](docs/TRACK_LOCAL3D_ITERATION.md) now enforces the configured heading bound at actual rail-anchor locations in the default Python/native recipes. An opt-in `configs/detector-local3d-experimental.json` estimates local head heights and tilted cross-sections shared by classification, viewer and RViz. Real replay and ray-cast curve/grade/cant experiments are recorded; height bias and unresolved alarms prevent promotion of the 3D mode.

See [run and review](docs/RUN_AND_REVIEW.md) for the local browser viewer and ROS2 launch commands, and [iteration evidence](docs/GOAL_ITERATION.md) for all six supplied recordings (2,488 scans), background repeatability, and remaining limitations. The browser shows original clouds, the reference corridor, candidates, confirmed intersections, distances and data quality. [Q&A implications](docs/QA_IMPLICATIONS.md) separates organizer statements from unresolved calibration assumptions.

## Native acceleration and calibration experiment

See [build, commands and evidence](docs/CALIBRATION_AND_NATIVE.md) and the [native integration report](docs/NATIVE_INTEGRATION.md). Native acceleration is integrated with the current interval-envelope policy. Use `configs/detector-native.json`; ROS container defaults select this recipe. The Python recipe remains available as a reference. Historical performance experiments are retained in `results/performance-20260917.json`; they are not evidence for the merged revision. Mounting calibration remains provisional: all three evaluated orientation candidates failed stability gates and were not installed.

## Native kernels follow the corrected contour

The corrected interval semantics above are implemented in **both** paths. The native classification kernel computes the same exact extrema of the piecewise-linear contour width over each point's height interval, in the same order, so the C++ recipe is not a frozen copy of the older rule. Verified two ways: the kernel and object checks compare the two recipes array by array (28/28 kernel checks, 20/20 geometry arrays, 624 and 550 candidate objects with no field mismatch), and the native recipe independently reproduces every corpus count the correction reports — `obstacle` frames 119 / 87 / 64 and 82 / 258 / 187 unresolved, identical to the review's table on 798 real scans. The three labelled panels are unchanged (development 162/54, holdout 159/57, measured 709/251), so the correction removes marginal confirmations in the unlabelled corpus, not in the annotated panels.

Their kernel-equivalence and corpus counts were measured on this branch's pre-merge recipes; the object counts move with the rail-heading and decode changes merged below, while the two-backend agreement is re-checked on the merged build.
The review's other finding — that the background model was queried for returns whose decision is never consumed — is now applied in both paths as well, and it is what closes the remaining latency gap: only segmentation-context returns that are not protected evidence are queried. No threshold changed.

## Measured rail anchors and corrected path geometry

The default recipes now fit paired rail heading and place anchors within actual measured support. See [implementation, real runs and limitations](docs/RAIL_GEOMETRY.md): 798 valid frames, 6093 supported anchors, improved withheld-point residuals on 8/9 saved clouds. Candidate grouping and alarms change; field accuracy and far-object recall remain unverified. The frozen previous recipe is `configs/detector-rail-baseline.json`.

## Reported 1.075 m mounting reference

See [railhead-support observations and chronological replay](docs/MOUNTING_REFERENCE.md). The reported empty/stationary height has unknown applicability to individual recordings. Direct support estimates are about 1.500 / 1.086 / 1.093 m across three recordings; no calibration is installed. All 798 compared detector outputs are preserved. A frozen-rotation replay on 738 later scans retains failures, including worse support in the round tunnel. The browser and RViz export now show the actual points supporting the estimate.

## Small detections and contour uncertainty

See [small-object review and calibration limits](docs/ENVELOPE_INTERVAL_REVIEW.md). Height uncertainty now propagates through the stepped contour width and both vertical boundaries. On 798 real scans, confirmed intersection observations changed from 718 to 649; small detections and nuisance alarms remain unresolved, and this is not a precision improvement claim. The browser exposes measured box sizes, support counts and exact diagnostic points for selected saved frames.

## Runtime reduction without reducing coverage

See [runtime profile and verification](docs/RUNTIME_CONTEXT_OPTIMIZATION.md). Avoiding unused background queries and repeated component scans preserved compared outputs on all 798 real scans. That pre-integration version measured 315–469 ms on the development Mac. See [native integration](docs/NATIVE_INTEGRATION.md) for current timings and remaining bottlenecks; target-hardware performance is unverified.

## Decode and complete offline latency

See [decoder preservation and latency scope](docs/DECODE_AND_LATENCY.md). Paired
real-cloud decoding decreased from 11.70 to 7.60 ms; all 174.5 million valid point
observations and normalized times matched exactly. The complete offline loop
measures 140–201 ms median across three recordings, including read/decode,
inference and result serialization. This is not live sensor-to-display latency.

## Coverage expansion and modeled insertions

See [coverage expansion](docs/COVERAGE_EXPANSION.md): complete platform and round-to-double tunnel runs, plus nine controlled cases on actual measured ray directions. Modeled support is traced through processing stages; missing rays, occlusion and candidate rejection are reported separately. Synthetic attribution is not field recall. The production detector is frozen for this experiment; its parameters were not tuned to inserted boxes.

## Review of new annotations

See [annotation review](docs/ANNOTATION_REVIEW.md) and [sensor evidence](docs/SENSOR_PROFILE.md). The new person panel contains one manual anchor and 35 detector-propagated boxes. Raw-cloud review and a fresh 201-frame run are complete; the panel is not accepted as independent ground truth. Original oriented label files are now included; their AABB conversion reproduces all 36 exported boxes exactly. The author reports reviewing all propagated positions. The box convention and oriented evaluation remain to be resolved.

## Latest real-sequence iteration

See [NEXT_ITERATION.md](NEXT_ITERATION.md): the complete 201-frame annotated sequence, stage-by-stage point evidence, and an optional envelope-support distance definition. All five provisional observations retained their localization; detection decisions stayed unchanged. This is a correction of distance semantics, not a measured detection-range improvement. The default recipe retains the original cluster-minimum distance; use `configs/iteration-envelope-distance.json` for the new mode.

## Latest alarm-cause correction

See [alarm-cause iteration](docs/ALARM_CAUSE_ITERATION.md). Lateral path uncertainty now
separates interior evidence from uncertain boundary crossings; unresolved points remain
candidates and appear orange in RViz. Two fixed structures retain their complete boxes
while losing unsupported certainty. A real candidate near 56 m remains detected. Across
402 real frames, definite-alarm frames decrease, but warning-free operation and field
false-alarm improvement are **not established**.

## External method review

See [HMM-MOS review](docs/HMM_MOS_REVIEW.md): the authors' own implementation of the IJRR moving-object
segmenter, built unmodified and run on our synthetic measured-pattern tunnel and on three real windows.
It segments objects that are genuinely moving (real walking person: 95% of its labels inside the one
hand-authored person box) and produced **0 labels on the static object in four synthetic cases at 15, 30,
60 and 100 m** - the class this project is scored on - because a state change is only counted for
occupied<->free transitions. Measured cost 0.22 s/frame and 0.30 GB at 60 m, 0.75 s/frame at 100 m, and
an empty tunnel at metro speed (1.5 m/frame) produced 70k false dynamic labels in 200 frames. Compact
numbers: `results/hmm-mos-probe-20260918.json`; recipes: `configs/hmm-mos-probe-*.json` and
`tunnel_guard/hmm_mos_probe.py`. Nothing in the pipeline was changed by this review.

## The input crop follows the track too (2026-09-19)

The detector cropped its input to a fixed 8 m lateral window in the *sensor* frame, so on a curve the track
itself left that window and both the returns and the rail anchors that estimate the curve were discarded
before classification: on the R=300 m arc panel an object standing on the track centre at 20-100 m was
absent from the objects list entirely. The window now follows the previous frame's remembered corridor and
is gated on that offset, so a straight run keeps the original crop exactly. Verified: the fixed
measured-pattern panel is **byte-identical** (709/251/176, precision 0.8011, zero empty-scene alarms) and
100-frame prefixes of both real recordings keep identical statuses; the R=300 m arc reports the on-track
object as `obstacle` from the first frame with history. Cost **+4-6 ms/frame** (127.1 -> 133.6,
139.1 -> 143.3 p50), measured under contention and to be re-measured idle;
`corridor_crop_threshold_m` disables it at the cost of curves.

## What the corridor can and cannot reach (2026-09-19)

Measured, not assumed. The **bed** is sampled to 65-105 m (13-18 anchors per frame - the floor is wide), so
heights above the running surface are known far out; inside its 15 m gate the linear bed extrapolation errs by
<=0.02 m even where the vertical curvature is R_v ~ 7 km. The **lateral** track centre is the binding unknown:
rail returns collapse 1995 -> 105 -> 17 -> 0 per 20 m bin from 10 m to 90 m, and the tunnel bore is a biased
proxy - robust circle fits to perpendicular slabs (16-23 slabs to 105-165 m, conditioned centre sigma
0.003-0.011 m) sit about a metre off the track centre, and calibrating that bias on the rails still predicts
only 1.32 m at 100 m.

The corridor's reach is therefore set by an uncertainty budget, `path_max_uncertainty_m` (0.4 m), which the
existing heuristic sigma reaches at 33.7 m past the last anchor - a ~74 m horizon, where the measured centre
error is 0.72 m, 47% of the 1.535 m half-width. `path_max_extrapolation_m` does not bind: raising it 25 -> 45 m
changed no classification at all. Raising the *budget* to 0.7 m would reach 86 m but was measured and rejected:
it turns two frames of `roundT_doubleT` into certified obstacles (intersecting 12 -> 20) in a band where the
centre is uncertain by ~1.0 m, for no measured gain. Fitted-curvature sigma propagated from the anchor window is
over-confident by 1.6x at 50-60 m and 4-8x at 80-150 m, so the heuristic term is the calibrated model.
Every object record now carries `far_field_lateral_bound_m`: the path uncertainty the classifier itself used at that object's distance - `sqrt(base^2 + extension^2)` with `base = 0.06 + 0.008r + 0.0003r^2`, calibrated to 0.46-1.34x the measured centre error over 10-110 m of extrapolation - or `null` beyond the modelled horizon, where no bounded claim exists. On 200 real frames this changed no status, no nearest distance and no object identity: 11,904 of 15,041 observations on `roundT_doubleT` carry a finite bound (max 0.45 m) and 3,137 state that their lateral track relation is unknown. A far-field detection therefore states what it does not know instead of implying that an unmeasured corridor is clear. Full evidence: `results/alignment-long-lever-20260919.json`, `build/uncertainty-calibration.json`.

## Reference corridor follows the curve

The reference contour used to continue past the last measured rail anchor along a straight tangent
whose slope was clipped at `rail_max_heading`, modelled for `path_max_extrapolation_m` beyond the
nearest anchor. Rails here are supported to a median 40 m, so the contour had a modelled horizon near
65 m — and inside it the true track leaves a straight line quadratically. Measured against what later
frames of `roundT_doubleT` see over the same ground, that continuation was 0.20 m off at 40 m, 0.47 m
at 50 m and 1.02 m at 60 m (p90 1.41 m), against a corridor half-width of 1.535 m.

`TrackGeometry._continuation` and its native mirror now continue along a local quadratic fitted to the
anchors inside `path_curve_window_m`, expressed in the edge anchor's frame so the centre-line stays
continuous there, with the curvature shrunk to zero unless it exceeds `path_curvature_significance`
standard errors (default 4), and the extrapolation uncertainty taken from the fit covariance in
quadrature with the previous base term. Inside the anchor span the corridor is bit-identical to before;
the model only acts beyond it. Forward-prediction error becomes **0.045 / 0.136 / 0.203 m** at 40 / 50 / 60 m.
Both backends stay identical (20/20 geometry arrays, 29/29 kernel checks), and 60-frame prefixes of
`roundT_doubleT` and `doubleT_obstacle` keep their statuses; the object set moves slightly
(14,305 → 14,014 on the straight recording, intersecting observations 64 → 70), which no available
label can adjudicate. `path_curve_window_m: 0` reproduces the previous continuation exactly.

The gate is not cosmetic. On the measured-pattern panel, whose rails are straight by construction, a
weaker 2-sigma gate bends the corridor off a geometry the old model already had exactly right:
precision 0.8011 -> 0.7647, tp 709 -> 702, fp 176 -> 216, and empty scenes start alarming
(negative-episode rate 0 -> 0.08). At 4 sigma the same panel is byte-identical to the baseline
(709/251/176, precision 0.8011, zero empty-scene alarms) while the real-curve prediction gain above is
kept, so 4 is the shipped default and `path_curvature_significance` is the knob to loosen deliberately.
Both panel runs are retained as evidence. Straight recordings are still not bit-identical — the fitted
slope replaces the noisy two-point tangent even where curvature is shrunk, which moves one frame of 60
from `unresolved_obstacle` to `candidate` on `doubleT_obstacle`.

This does **not** extend the modelled horizon: beyond anchors + 25 m the corridor is still `unknown`,
which is why the scored 100–300 m band needs a long-lever estimate. Walls and ceiling do return to
120–207 m on the curved recording, but per-bin medians of those returns are not an axis — they jump
5–9 m with platform edges — so that estimator has to be built on surface strips and validated with the
same forward-prediction test before it is allowed to widen the corridor. Evidence:
`results/curve-continuation-20260919.json`.

## Reader overlap in the offline runner

The bag reader (decompression plus PointCloud2 decode) ran between frames: 38.6 ms p50 on
`doubleT_obstacle`, 12.8 ms on `doubleT_platform`. One bounded producer thread now reads ahead by one
scan while inference runs. On the same 30-frame protocol with the same recipe,
`read_and_process` p50 falls 164.7 → **126.7 ms** and 118.8 → **105.2 ms**, and p95 185.2 → 142.0 ms,
with every compared field identical (status, nearest distance, each object's track id and distance, 60
frames). Inference is untouched (125.6 → 126.5 / 105.5 → 104.7 ms).

This removes the reader from the critical path; it does **not** shorten the age of a decision, and
inference at ~126 ms per frame still exceeds the 100 ms input period, so 10 Hz per frame is not met.
Recorded result: `results/reader-overlap-20260919.json`.

## Team work

See [next iteration assignments](docs/TEAM_TASKS.md): reviewed episodes, sensor/time evidence, Ubuntu/RViz validation, and oriented evaluation. Use separate branches from `experiments/morev`.

## Quick start

Python 3.10+; this iteration used native macOS Python 3.12.8 and container Python 3.10 (Humble). Historical audits used other versions. Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked
uv run python -m tunnel_guard.run --experiment configs/evaluation-audit.json
```
Agent policy lives in `AGENTS.md`: no subagents or automated tests. Verify changes through actual detector runs and configured evaluations; the repository intentionally has no test suite.

The audit recipe requires the two real bags described below. It processes the first 30 consecutive frames of each and writes JSONL, configuration, source snapshots and RViz result bags to `build/audit-reviewed/`. Set `visualization` to `false` in a copied experiment JSON for headless processing; detector decisions do not depend on the display consumer. No model download is needed at runtime.

**Output directories must not already exist.** To repeat a run, copy its experiment JSON and change `output`; do not delete evidence merely to rerun. Nix users can optionally use `nix develop`; everyone else can ignore `flake.nix`, `flake.lock`, and `.envrc`.

## Run the real bags

Obtain the organizer's dataset separately and put the extracted directories here:

```text
data/sourcecraft_subset/for_hackathon/
  doubleT_obstacle/
  doubleT_platform/
  roundT_doubleT/
  roundT_pressureGate_roundT/
  roundT_squareT_pressureGate_squareT/
  squareT_platform_squareT_switch/
```

Each directory must contain its `metadata.yaml` and SQLite `.db3` files. Recordings and dataset archives are deliberately not included in Git.

For the organizer archive supplied as `archive/for_hackathon.zst`, the two audit bags can be extracted without unpacking the whole dataset:

```sh
mkdir -p data/sourcecraft_subset
tar --zstd -xf archive/for_hackathon.zst -C data/sourcecraft_subset \
  for_hackathon/doubleT_obstacle for_hackathon/doubleT_platform
uv run python -m tunnel_guard.inspect_bag \
  data/sourcecraft_subset/for_hackathon/doubleT_obstacle \
  --max-frames 30 --output build/input-inspection.json
```

The original full evaluation below requires all six bags:

```sh
uv run python -m tunnel_guard.run --experiment configs/evaluation-quality.json
uv run python -m tunnel_guard.evaluate \
  --run build/tunnel-guard-real-quality \
  --annotations annotations/sourcecraft-provisional.json \
  --output build/tunnel-guard-real-quality/localization.json
```

The five supplied annotation boxes describe **one provisional upright structure**, not independent verified hazards. They are nonexhaustive: real precision cannot be computed from them.

## Label the recordings in SUSTechPOINTS

Human labels do not exist yet for the six recordings, and no score can be computed without them. The upstream [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS) annotation tool is used for that work: its source is vendored under `SUSTechPOINTS/` at upstream revision `50fa188`, with this project's taxonomy patch already applied (the delta is kept in `patches/`). What is not tracked, matching the tool's own rules and this project's data rule, is its `data/` directory, its virtualenv and the 14 MB model release.

```sh
cd SUSTechPOINTS
python3 -m venv .venv && .venv/bin/pip install -r requirement.txt
wget https://github.com/naurril/SUSTechPOINTS/releases/download/0.1/deep_annotation_inference.h5 -P algos/models
cd .. && uv run python -m tunnel_guard.sustech_import --experiment configs/evaluation-quality.json
```

`tunnel_guard.sustech_import` writes one `.pcd` per bag message into `SUSTechPOINTS/data/<bag>/lidar/`, keeping `tunnel_guard_local` coordinates, intensity and full density, and an empty `label/` beside it. Frame `00000N.pcd` is bag message index `N` — the same index the run JSONL uses, so detector output and labels refer to the same frame. It reproduces the current scene files byte for byte (checked on three frames), and it refuses to overwrite an existing scene. Start the tool with `python main.py` inside `SUSTechPOINTS` and open <http://127.0.0.1:8081>; `server.conf` listens on `0.0.0.0`, so it is also reachable over a private network such as Tailscale (the tool has no authentication).

Boxes are authored by hand: the detector's candidates are not good enough to seed a panel, and labels a detector supplies for its own scoring cannot measure that detector. After a labelling pass, convert the tool's label files into the schema the evaluator validates:

```sh
uv run python -m tunnel_guard.sustech --config configs/annotation-export.json
```

`configs/annotation-export.json` requires the reviewer to declare which frames are exhaustively labelled, including frames with no object: those become the scored negative frames, and `tunnel_guard.evaluate` counts nothing else as a false alarm. Rotated boxes are exported as their axis-aligned envelope, which is what the IoU matcher consumes.

### What is labelled so far

`annotations/doubleT-obstacle-person.json` holds the one object we have: `doubleT_obstacle` frames 165–200, one person-sized box, `event_id 7`, `class Person`. The box was authored by hand at frame 181 and propagated over the other 35 frames by a constant-velocity fit of the detector's own track of that object — it recedes at 0.31 m/frame along +x while the tunnel itself stays fixed in the sensor frame (a tracked ceiling fixture moves 0.001 m/frame), so the motion is the object's, not the train's. Every frame is `exhaustive: false`: this records where one object is, not that the frames contain nothing else, and "person" is the reviewer's judgement.

`annotations/sustech-raw/doubleT_obstacle/*.json` is the authoritative form of the same 36 frames, copied verbatim from the annotation tool: `position`, `rotation` and `scale` per box. The evaluator's schema has no rotation field, so the export stores the axis-aligned envelope of the rotated box and inflates the footprint — for this box, yaw 0.406 rad widens x by 43% and y by 22%. Comparing the label with the detector's own box for the same object over those frames: centre offset median 0.48 m, label/detector extent ratio 1.73 × 1.63 × 1.29 at true scale, median IoU 0.196 (6 of 36 frames at or above the 0.25 gate), falling to 0.147 (0 of 36) through the envelope conversion. The label is a full-person volume; the detector's box is the support of its returns. Those are different measurement conventions, and the mismatch is not evidence about either method's correctness.

### Objects inside the clearance envelope

`tunnel_guard/on_track.py` finds intrusions the way the detector should: only returns that actually fall inside the GOST contour are clustered, and a cluster is dropped when it is a face of a large surface, or when its lateral/vertical profile runs continuously or repeats along the tunnel — cable runs, linings, trays and posts. Over the six recordings that is **11,115 in-envelope clusters → 381 events → 15 candidates**, against 335–347 boxes per frame from the detector.

```sh
uv run python -m tunnel_guard.on_track \
  --bag data/sourcecraft_subset/for_hackathon/doubleT_obstacle \
  --config configs/on-track-probe.json --output build/on-track-events.json
```

Thresholds are in `configs/on-track-probe.json`. The strongest candidate is `doubleT_obstacle` around x ≈ 55.7 m, y ≈ −0.5 m: a compact 0.45 × 1.02 × 1.35 m mass standing on the bed, isolated from any surface, present in frames 2–75 and then gone. Two limits matter. Geometry cannot separate a permanent fixture that pokes into the contour from a foreign object — the ~34 m hit in the same recording is a post running through the whole recording. And on the `roundT_*`/`squareT_*` curves the track-centre estimate drifts laterally, putting 139 of 169 events in one recording at |y| = 1.6–3.2 m while the GOST half-width never exceeds 1.535 m; `report.max_lateral_m` drops those, which is why the count is 15 and not 381. That same drift is a large part of the detector's alarm load.

The surviving candidates are also written into the tool as separate scenes (`<bag>_candidates`, with `lidar` symlinked to the real clouds) so they can be stepped through without touching the human labels.

The tool's own auto-annotation is not usable for this data: `GET /auto_annotate` returns HTTP 500 because `annotate_file` is defined inside an `if False:` block and its clustering binary and discrimination model are absent, and `/predict_rotation` feeds uncentered coordinates to a model trained on centred object crops, so its angle barely depends on the selected object.

### Classes

The patch in `patches/` makes `public/js/obj_cfg.js` carry a metro taxonomy in place of the upstream driving classes: `Person`, `ForeignObject`, `Equipment` (things that must not be on the track), then `PlatformEdge`, `PressureGate`, `TrackSwitch`, `TrackFixture`, `Cable`, `WallLining` (content the recordings actually contain), then `Unknown` and `DontCare`. The same names are repeated in `tools/check_labels.py` (server-side `/checkscene`) and `tools/visualize-camera.py`; keep the three in sync. `Unknown` is the fallback of `get_obj_cfg_by_type`, so it must exist.

The organizers do not require a class and the detector is class-agnostic, so this field is not a supervised target. It exists so a reviewer can attribute alerts: the measured failure mode is nuisance alarms on tunnel infrastructure, and without a label for "this was the platform edge" those alarms cannot be explained. Labels written with the removed driving names still load and render through the `Unknown` configuration, but `/checkscene` reports them as unrecognisable.

Nothing except scene directories may live under `SUSTechPOINTS/data/`: `scene_reader` treats every entry as a scene and fails on any other file.

## Code map

| Path | Purpose |
|---|---|
| `tunnel_guard/io.py` | PointCloud2 decoding, invalid returns, acquisition timestamps |
| `tunnel_guard/geometry.py` | Track bed, paired rails, reference clearance envelope |
| `tunnel_guard/segmentation.py` | Density-core clustering; optional published backends |
| `tunnel_guard/background.py` | Open3D-supported tunnel surfaces and protrusion protection |
| `tunnel_guard/detector.py` | KISS-ICP motion, candidates, tracking and temporal evidence |
| `tunnel_guard/run.py` | Reproducible bag runner |
| `tunnel_guard/inspect_bag.py` | Bounded layout, acquisition-clock and density inspection |
| `tunnel_guard/visualization.py` | Actual PointCloud2 / MarkerArray / status export for RViz2 replay |
| `tunnel_guard/evaluate.py` | One-to-one IoU matching and annotation validity |
| `tunnel_guard/stress.py` | Occlusion-aware synthetic ray-cast evaluation |
| `tunnel_guard/annotate.py` | Extract raw frames for annotation review |
| `tunnel_guard/sustech.py` | SUSTechPOINTS human labels into the `annotations/*.json` schema |
| `tunnel_guard/on_track.py` | Objects with real point support inside the clearance envelope |
| `tunnel_guard/sustech_import.py` | Recordings into SUSTechPOINTS scenes |
| `patches/` | Modifications applied to the pinned SUSTechPOINTS revision |
| `configs/detector.json` | Default detector recipe; no bag-specific branches |
| `results/` | Small recorded result summaries; full artifacts remain local |

Pipeline: validated points → KISS-ICP pose (optional deskew) → local bed and rails → supported tunnel-surface rejection → density-core segmentation → rail-relative clearance classification → temporal state and spatial evidence.

**Deskew is disabled in the current recipes.** The observed point-time span differs from the frame period, especially in cropped clouds; KISS-ICP normalizes that span to a full previous motion increment. `deskew_enabled=true` restores the experimental mode, but needs verified timing and prior-deskew provenance. Turning it off leaves motion distortion unresolved. This change is not a claim of higher detection accuracy.

`configs/detector-native-fast.json` is the fastest recipe: it is `detector-native.json` with two background plane proposals per window instead of three. That is a behaviour change, so it is kept as a separate recipe: measured on 798 real scans it moves 297 definite alarms to 296 (one frame becomes unresolved) and leaves all three labelled panels unchanged, while `detector-native.json` keeps the original setting and is the integration recipe. Historical pre-kernel comparisons and current interval-policy comparisons must be distinguished.

`native_kernels` (C++ recipe only; the NumPy recipe keeps the reference path) selects the locally built `tunnel_guard._native` kernels: radial range selection, mutual-radius clustering graph, envelope classification, track-bed reference, background patch candidates, protrusion protection, strip membership and the evidence voxel count. `normal_covariances` replaces Open3D's `estimate_normals` plus `estimate_covariances` with one grid pass: neighbours within the radius capped to the nearest `max_nn`, mean-centred covariance over n, and — in the same call — the eigenvalues and the smallest eigenvector by fixed-sweep Jacobi rotations. Three labelled panels (development, seed holdout, measured beam pattern; 2,120 frames) reproduce their recorded tp/fn, event recall and precision exactly with the native kernels. Measured on a real sample: counts identical to the scipy radius count, planarity gate identical on all 31,298 points, normal agreement |dot| = 1.000000000000 on every reliable point and zero alignment differences across all patches — the points whose normals differ are exactly the ones the gate discards.

Native kernels cover voxel selection, neighbour graphs, geometry classification, background masking, component statistics and evidence counting. They use reusable buffers and preserve measurement support on the recorded integration panel. Empirical agreement does not establish identity for all unseen inputs. `query_workers` controls SciPy queries; native kernels also use their own worker threads. Open3D plane proposals remain serial for repeatability. See [integration evidence and limitations](docs/NATIVE_INTEGRATION.md).

Segmentation preserves object portions outside the clearance gate. Dense instances cannot merge through a thin chain of border points. Temporal matching uses velocity, heuristic covariance, shape, and distinct-frame evidence. Missing path support must not turn rail removal into an infinite-width exclusion zone.

Tunnel-background rejection uses Open3D plane fitting and local normals. Only observed longitudinal surface strips are removed; cross-track panels, supported clearance intersections, protruding faces and their attachment edges are protected. No generic sparse-outlier deletion is applied. It is a local planar approximation, not a complete curved-tunnel model.

Focused verification: the original tilted empty-tunnel wall alert disappears; narrow-tunnel wall retention drops from 96.2% to about 1.6%, ceiling retention from 100% to about 8%. Edge protection intentionally leaves some lining near surface intersections. The 30-frame obstacle-preservation panel retained 100% of target points entering the original ROI, with no confirmed-localization regressions. The intermediate real sequence retained 5/5 provisional matches, but median processing increased to 741 ms; the later removal of redundant queries has not had a full-sequence timing run. These checks do not establish general safety or solve all infrastructure alerts. The broader stress rerun was stopped at user request.

## Output semantics

- `obstacle`: intersection with the configured reference envelope has its own confirmation evidence; not a validated collision claim.
- `unresolved_obstacle`: a confirmed potential hazard has uncertain geometry or insufficient repeated interior evidence.
- `candidate`: insufficient confirmation evidence.
- `no_obstacle_observed`: no hazard reported; **does not mean the route is clear**.
- `unknown`: insufficient geometry or returns.

Boxes describe observed support, not inferred full object volume. Distance is the minimum forward x of the current cluster in the configured processing frame, including its out-of-envelope support; a noisy extreme may dominate it. It is not bumper distance or curve-integrated track distance. Uncertainty is heuristic, not a calibrated safety probability.

- `health` (`normal` / `degraded` / `unavailable`) and `health_reasons` are independent of detection status. The current unverified calibration keeps results degraded.
- `timestamp_s` uses acquisition header time. `measurement_timestamp_ns` and `record_timestamp_ns` preserve both exact clocks; do not interpret their difference as latency.
- `source_scan_id`, `last_observed_s`, `hits`, and `evidence_timestamps_s` expose the source and temporal evidence. Duplicate acquisition timestamps are skipped by the reader; backwards time or a changed sensor frame stops the run explicitly. A new bag creates a new detector.
- `confirmed` describes the object; `intersection_confirmed` separately describes its current envelope intrusion. Immediate confirmation uses interior support; weak intrusion requires distinct recent interior observations. See [intersection evidence](docs/INTERSECTION_EVIDENCE.md) for real-data diagnosis, fields and tradeoffs. RViz uses red for confirmed intrusion and orange for confirmed objects with unresolved/pending intrusion.
- `coordinate_frame=tunnel_guard_local` identifies the transformed current-scan coordinates. `sensor_frame` is source metadata. No global TF or verified vehicle extrinsics are implied.
- `processing_s`, `read_and_process_s` and optional `visualization_s` use monotonic timing; summary includes ingestion/drop counts and visualization time. `range_observability` reports support, not free-space coverage.
- The runner reads and decodes the next scan on one producer thread while the detector processes the
  current one: `prefetch_depth` in the experiment config, default 1, bounded to keep one unprocessed
  scan in memory. With overlap, `read_and_process_s` is the cost of one loop iteration and
  `ingestion_s` is the consumer's block on that thread, **not** the age of a decision — a scan still
  waits for the scan ahead of it. `prefetch_depth: 0` restores inline reading. No measurement and no
  decision changes either way; measured evidence and the equivalence check are in
  `results/reader-overlap-20260919.json` (`configs/perf-prefetch-on.json`).

## View actual results in RViz2

The audit recipe records `build/audit-reviewed/doubleT_obstacle_rviz/` and `doubleT_platform_rviz/`. Export and CDR readback were run on macOS; the commands below require a machine with **ROS 2 Humble and RViz2** and have not been executed on that runtime here.

Run from the repository root in two terminals after sourcing ROS:

```sh
source /opt/ros/humble/setup.bash
rviz2 -d "$PWD/rviz/tunnel_guard.rviz" --ros-args -p use_sim_time:=true
```

```sh
source /opt/ros/humble/setup.bash
ros2 bag play "$PWD/build/audit-reviewed/doubleT_obstacle_rviz" --clock
```

The result bag is **recorded inference replay**, not live inference. Play it by itself: it supplies the measurement timeline through `/clock`; do not run a second clock publisher or mix it with the original bag's different record-time epoch. Restart playback and reset RViz when switching recordings or seeking backwards. Replay never continues a detector with future state.

Displays: grey measured scene, cyan supported reference contour, yellow tentative candidates, red confirmed nominal intersections, grey adjacent objects, and explicit health/nearest text. `candidate_measurements` markers show the actual voxel representatives even when the background display copy is sampled. Each frame starts with DELETEALL; markers expire after 0.3 simulation seconds. Pausing replay pauses simulation time too; the scene is explicitly labelled replay.

Fixed Frame is `tunnel_guard_local`: all messages already share this local frame, so no invented `map` or identity TF is needed. Do not accumulate clouds across frames. For a top view choose RViz's top-down view; for candidate inspection set the Orbit focal point to the object's reported `center` and reduce view distance. This changes camera framing, not physical coordinates. Marker namespaces can be toggled to inspect points without boxes.

`display_max_points` limits only the background visual copy; candidate representatives remain separate. GUI-off/headless comparison preserved status, boxes, distances, IDs and confirmations on all 60 inspected frames. Full bitwise pose equality is not claimed. A static real-frame overview is saved locally as `build/audit-preview.png`; it is not RViz GUI validation.

The envelope follows the [GOST 23961-80 M reference contour](https://engenegr.ru/gost-23961-80), with a conservatively filled lower contour above 50 mm. Actual vehicle dynamics, curves, mirror/current-collector extensions and organizer-certified extrinsics remain unverified. Anisotropic clustering helps unequal beam spacing but can merge vertically adjacent structures; temporal confirmation delays weak detections. These are explicit tradeoffs, not solved guarantees.

## Recorded results

See `results/acceptance.json` and `results/panel-summary.csv`. **Acceptance: not promoted.** Original IoU thresholds and provisional annotations were retained.

| Panel | Event recall | Object-frame precision | Notes |
|---|---:|---:|---|
| Original synthetic panel | 65/72 (90.3%) | 88.5% | 0/38 negative alarm episodes |
| Untouched random seed | 64/72 (88.9%) | 86.4% | Same scenario families, not a domain holdout |
| Measured beam pattern, 10–300 m | 77/96 (80.2%) | 80.1% | 1,460 frames; 0/50 negative episodes |

All synthetic panels fail the declared 95% event-recall target. Measured-pattern event recall: 100% at 10–100 m, 66.7% at 150 m, 58.3% at 200 m, 16.7% at 300 m. Ideal raycasting is **not hardware range or reflectivity validation**.

Final metro localization: 5/5 unchanged provisional boxes at IoU ≥0.25, mean IoU 0.282; earlier baseline was 2/5. One object sampled five times does not establish generalization.

Full real run: 2,488 frames; 1,579 `obstacle`, 819 `unresolved_obstacle`. These are **not false-positive counts** without exhaustive labels. Per-bag median processing was 294–486 ms on Apple M4 before the latency work; the three re-run recordings measure 99–123 ms with the C++ backend and native kernels afterwards (see `results/performance-20260917.json`): still not real-time at the ~10 Hz recording rate, and not target Intel performance.

Published backend comparisons and height/tilt ground audits are summarized in `results/`. The saved segmentation comparison predates the last support-preservation correction; its original full source/config snapshots are local under `build/`. Running the current comparison recipe evaluates the current code, not that historical snapshot. Paths inside result summaries refer to these intentionally untracked original artifacts.

## Optional research evaluations

```sh
uv sync --locked --extra comparison
uv run python -m tunnel_guard.compare --experiment configs/comparison.json
uv run python -m tunnel_guard.ground_audit --experiment configs/ground-audit.json
uv run python -m tunnel_guard.stress --experiment configs/stress-quality-heldout.json
uv run python -m tunnel_guard.stress --experiment configs/stress-measured-pattern-final.json
```

The ground audit also needs the six organizer bags. TRAVEL and HDBSCAN comparisons share geometry/tracking to isolate segmentation; they are not complete neural SOTA benchmarks.

Target-metro exhaustive positives, negatives, and held-out recordings are still needed.

## Dependencies and data licenses

- [KISS-ICP](https://github.com/PRBonn/kiss-icp), MIT: used directly for motion/deskew.
- [Open3D](https://www.open3d.org/), MIT, pinned to 0.19.0: plane segmentation, voxel sampling, normal and covariance estimation for background rejection. Its standard distribution adds substantial transitive dependencies.
- [TRAVEL](https://github.com/url-kaist/TRAVEL), **GPL-3.0-or-later**: optional comparison dependency only. Review license obligations before distributing an integrated derivative.
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan), BSD; [Patchwork++](https://github.com/url-kaist/patchwork-plusplus), BSD-2-Clause: optional published comparisons.

No project-wide redistribution license is granted here. Keep organizer data, credentials, local agent configuration and generated artifacts out of commits.
