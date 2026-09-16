# Tunnel Guard

Class-agnostic LiDAR obstacle-detection baseline for metro tunnels. Reads ROS 2 PointCloud2 bags directly; no ROS installation, Docker, or pretrained weights required for the default pipeline.

**Research baseline, not a validated collision-warning system.** Recall, infrastructure alarms, generalization, and runtime remain unresolved. `CASE.md` contains the original requirements. Recorded RViz2 result export is implemented; target Ubuntu/Humble deployment and the RViz GUI remain unverified.

The current review, real-data comparison and limitations are in [docs/AUDIT.md](docs/AUDIT.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). The latest iteration processed two real 30-frame prefixes. Historical results below are separate evidence.

## Quick start

Python 3.11+; Python 3.13.5 was used for the latest audit (older results used 3.12). Install [uv](https://docs.astral.sh/uv/), then:

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

`tunnel_guard.sustech_import` writes one `.pcd` per bag message into `SUSTechPOINTS/data/<bag>/lidar/`, keeping `tunnel_guard_local` coordinates, intensity and full density, and an empty `label/` beside it. Frame `00000N.pcd` is bag message index `N` — the same index the run JSONL uses, so detector output and labels refer to the same frame. It reproduces the current scene files byte for byte (checked on three frames), and it refuses to overwrite an existing scene. Start the tool with `python main.py` inside `SUSTechPOINTS` and open <http://127.0.0.1:8081>; `server.conf` listens on `0.0.0.0`, so it is also reachable over a private network such as Tailscale (the tool has no authentication). Nothing but scene directories may live under `SUSTechPOINTS/data/`: the tool treats every entry there as a scene.

Boxes are authored by hand: the detector's candidates are not good enough to seed a panel, and labels a detector supplies for its own scoring cannot measure that detector. After a labelling pass, convert the tool's label files into the schema the evaluator validates:

```sh
uv run python -m tunnel_guard.sustech --config configs/annotation-export.json
```

`configs/annotation-export.json` requires the reviewer to declare which frames are exhaustively labelled, including frames with no object: those become the scored negative frames, and `tunnel_guard.evaluate` counts nothing else as a false alarm. Rotated boxes are exported as their axis-aligned envelope, which is what the IoU matcher consumes.

### What is labelled so far

`annotations/doubleT-obstacle-person.json` holds the one object we have: `doubleT_obstacle` frames 165–200, one person-sized box, `event_id 7`, `class Person`. The box was authored by hand at frame 181 and propagated over the other 35 frames by a constant-velocity fit of the detector's own track of that object — it recedes at 0.31 m/frame along +x while the tunnel itself stays fixed in the sensor frame (a tracked ceiling fixture moves 0.001 m/frame), so the motion is the object's, not the train's. Every frame is `exhaustive: false`: this records where one object is, not that the frames contain nothing else, and "person" is the reviewer's judgement. The same boxes are visible in the tool as `doubleT_obstacle` frames 165–200.

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

Segmentation preserves object portions outside the clearance gate. Dense instances cannot merge through a thin chain of border points. Temporal matching uses velocity, heuristic covariance, shape, and distinct-frame evidence. Missing path support must not turn rail removal into an infinite-width exclusion zone.

Tunnel-background rejection uses Open3D plane fitting and local normals. Only observed longitudinal surface strips are removed; cross-track panels, supported clearance intersections, protruding faces and their attachment edges are protected. No generic sparse-outlier deletion is applied. It is a local planar approximation, not a complete curved-tunnel model.

Focused verification: the original tilted empty-tunnel wall alert disappears; narrow-tunnel wall retention drops from 96.2% to about 1.6%, ceiling retention from 100% to about 8%. Edge protection intentionally leaves some lining near surface intersections. The 30-frame obstacle-preservation panel retained 100% of target points entering the original ROI, with no confirmed-localization regressions. The intermediate real sequence retained 5/5 provisional matches, but median processing increased to 741 ms; the later removal of redundant queries has not had a full-sequence timing run. These checks do not establish general safety or solve all infrastructure alerts. The broader stress rerun was stopped at user request.

## Output semantics

- `obstacle`: confirmed structure intersects the configured reference envelope.
- `unresolved_obstacle`: confirmed nominal intersection with insufficient geometry support; not a proven collision.
- `candidate`: insufficient confirmation evidence.
- `no_obstacle_observed`: no hazard reported; **does not mean the route is clear**.
- `unknown`: insufficient geometry or returns.

Boxes describe observed support, not inferred full object volume. Distance is the minimum forward x of the current cluster in the configured processing frame, including its out-of-envelope support; a noisy extreme may dominate it. It is not bumper distance or curve-integrated track distance. Uncertainty is heuristic, not a calibrated safety probability.

- `health` (`normal` / `degraded` / `unavailable`) and `health_reasons` are independent of detection status. The current unverified calibration keeps results degraded.
- `timestamp_s` uses acquisition header time. `measurement_timestamp_ns` and `record_timestamp_ns` preserve both exact clocks; do not interpret their difference as latency.
- `source_scan_id`, `last_observed_s`, `hits`, and `evidence_timestamps_s` expose the source and temporal evidence. Duplicate acquisition timestamps are skipped by the reader; backwards time or a changed sensor frame stops the run explicitly. A new bag creates a new detector.
- `coordinate_frame=tunnel_guard_local` identifies the transformed current-scan coordinates. `sensor_frame` is source metadata. No global TF or verified vehicle extrinsics are implied.
- `processing_s`, `read_and_process_s` and optional `visualization_s` use monotonic timing; summary includes ingestion/drop counts and visualization time. `range_observability` reports support, not free-space coverage.

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

Full real run: 2,488 frames; 1,579 `obstacle`, 819 `unresolved_obstacle`. These are **not false-positive counts** without exhaustive labels. Per-bag median processing was 294–486 ms on Apple M4: not real-time at the recording rate, and not target Intel performance.

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
