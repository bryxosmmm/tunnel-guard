# HMM-MOS review: the IJRR moving-object segmenter, tested on this project's data

Reviewed: V. Bhandari, J. James, T. G. Phillips, P. R. McAree, *Moving Object Segmentation in Point
Cloud Data Using Hidden Markov Models*, The International Journal of Robotics Research 45(8),
1165-1194, 2026 (DOI 10.1177/02783649251379929, first online 2025-10-06). The SAGE full text is not
accessible from here (HTTP 403), so the review uses the authors' open IROS 2024 workshop version
([arXiv:2410.18638](https://arxiv.org/abs/2410.18638)) and, for behaviour, their reference
implementation ([github.com/vb44/HMM-MOS](https://github.com/vb44/HMM-MOS) at commit `3f625031`),
built unmodified and run here.

## Verdict

**Not usable as this project's detector, and not a replacement for any part of the current pipeline.
Its one measured strength - segmenting objects that are genuinely moving during the scan - is a
capability this project's scoring does not primarily ask for, at a detection range 2-6x shorter than
the band we are scored on: the authors' own real-time envelope is 20-50 m, measured here as 20.4 Hz at
20 m falling to 1.3 Hz at 100 m, and the method is structurally blind to obstacles that were already
standing when the train came into range.**

Measured on our geometry and sensor sampling (measured sourcecraft ring/azimuth pattern, 128 rings,
0.1 deg azimuth, 10 Hz, 10 m/s platform):

| hazard class | HMM-MOS result |
|---|---|
| object moving during the scan (walking person, real recording) | labelled in 35/41 frames, 95% of the labels inside the one hand-authored person box |
| object moving, synthetic tunnel at 10 m/s platform speed | 94% of the object's returns labelled, 54/60 frames |
| **static object on the track bed** (our main class) | **0 labels on the object at 15, 30, 60 and 100 m**; at 30 m the only labels in the scene are on a trackside fixture |
| object that walks and then stands still | labelled while moving, silent from the frame it stops |
| object placed into space already observed free | labelled for 5 frames after it appears |

The static-object result is structural, not a tuning artifact: a state change is counted only when
*both* the previous and the new voxel state are non-`unobserved` (`src/Map.cpp`), so an obstacle that
comes into view for the first time can never seed a detection. A person standing on the track after
walking into a tunnel is exactly the case that fails.

## What the method actually does (verified in the reference code)

Stage | Implementation detail (commit `3f625031`)
---|---
Voxelise | `voxelSize` (0.2-0.25 m in their configs); states `{unobserved, occupied, free}`
Raycast free space | Bresenham from the sensor to every occupied voxel, `maxRange` per scan
Occupancy likelihood | Gaussian on the distance to the nearest measurement in a local window of scans, `occupancySigma`
Per-voxel filter | `xHat = B A xHat` normalised, with `A = [[1-2e,0,0],[e,1-e,e],[e,e,1-e]]`, `e = 0.005`, state committed at `beliefThreshold = 0.99`
Change detection | `maxElementIndex != currentState && maxElementIndex != 0 && currentState != 0` - **occupied<->free only**
Aggregation | 3D convolution (`convSize` 5) over a local window (`localWindowSize` 3-5), median filter, Otsu threshold with a `minOtsu` floor, then one-voxel dilation and carry-over of the previous scan's confident voxels
Map pruning | Voxels beyond `maxRange*1.5` or unseen for `globalWindowSize` (300) scans are deleted

Two consequences the paper states and that our measurements reproduce:

- The method detects *movement*, not *objects that have moved*; the authors report this as the main
  reason their recall looks low against datasets that label objects that moved at any time.
- The method is "not appearance-based, and any moving measurement relative to a fixed frame is
  labelled as dynamic" (their README, limitation I). In a tunnel that includes measurement changes
  that are not object motion.

## Experiments

Recipes are explicit and reproducible: `configs/hmm-mos-probe-synth*.json`, `configs/hmm-mos-probe-real-*.json`,
`configs/hmm-mos-range-*.json`, driven by `tunnel_guard/hmm_mos_probe.py` (it only generates inputs and reads
the labels the authors' binary writes; their source is unmodified). Full per-frame records are in
`build/hmm-mos-*/summary.json`; the compact numbers are in `results/hmm-mos-probe-20260918.json`.

### 1. Synthetic tunnel with our measured beam pattern

`configs/hmm-mos-probe-synth.json`: the scene is `tunnel_guard.stress.scene_scan` with the
measured-pattern stress settings (128 measured ring elevations, 3600 azimuth bins, 0.1 deg, rails on a
1.52 m gauge, a 0.2 m-tall trackside fixture, 1.5 cm ranging noise, 5% dropout), 60 frames at 10 Hz,
sensor stepping 1 m/frame along the track axis, one inserted 0.45x1.02x1.35 m box.
Poses are exact; the crate carries no pose error unless a case says so. One caveat on density: the
synthetic scene emits one ray per emission at the full measured pattern (460,800 rays/frame), whereas a
real deduplicated frame here carries 96,385-177,690 distinct returns, so the synthetic load is the
denser, more adverse regime; the 64-ring cases above probe the other direction.

| case | dynamic labels | frames with any label | object returns labelled |
|---|---:|---:|---:|
| empty tunnel | 0 | 0/60 | - |
| static box, 60 m, stationary sensor | 0 | 0/60 | 0% |
| static box, 60 m, sensor approaching | 0 | 0/60 | 0% |
| static box, 100 m, sensor approaching | 0 | 0/60 | 0% |
| static box, 30 m, sensor passing/over it | 9,954 | 9/60 | 0% (all labels on the trackside fixture) |
| box moving towards the sensor at 1.5 m/s | 207,303 | 54/60 | 94% |
| box moving 20 frames, then stopped | 3,864 | 19/60 | 2% - labels exist only while it moves |
| box appears at frame 10 in already-observed free space | 155,446 | 14/60 | 32% |

The 30 m case labels the fixture at y ~ -2.1 m, z ~ -0.9 m, 13-22 m ahead, in frames 28-36, i.e. when
the sensor is inside the fixture's span and the inserted box occludes the far side of it: a long thin
trackside structure flips voxel state when its occlusion changes. That is the same kind of object
(cables, trays, posts, platform edge) this project already over-alarms on.

### 2. Real recordings

Poses are the recorded detector run's own KISS-ICP estimates (`build/perf-cpp-opt-20260917/*.jsonl`);
points are the same deduplicated returns the detector consumes (the decoded count is 0.51-0.52x the
bag's `input_valid_points` because the recordings carry each measurement twice).

| recording, window | ego-motion | frames with labels | labels | attribution |
|---|---:|---:|---:|---|
| `doubleT_obstacle` 0-80, stationary platform | 2.8 mm/frame | 23/81 | 862 | 55.6-57.0 m in frames 16-24 and 56-80, inside the cluster our detector tracks as 147 (the on-track candidate at 55.7 m on the bed, drifting 0.006 m/frame); 1.8-2.2 m in frames 13, 14, 16, 21-24 on structure our detector does not single out |
| `doubleT_obstacle` 160-200, the annotated person | 2.8 mm/frame | 35/41 | 29,779 | 95% inside the hand-authored person box, 99% inside our detector's own candidate boxes; the person recedes 0.31 m/frame |
| `roundT_doubleT` 60-160, curved section | 1.72 m/frame (17 m/s) | 44/101 | 419,007 | 73% within 10 m and 90% within 20 m; bursts reach 23,884 points in one frame; 94% fall inside *confirmed* detector objects (trackside at y -3.0..-2.1 m) - near-field infrastructure, the class we already over-alarm on |

The moving-person window is the method working as advertised on our data. The moving-platform window is
its cost: at metro speed the output becomes a near-field flood that carries no information our detector
does not already have, and it cannot be separated from pose error without an independent trajectory
reference, which this project does not have.

### 3. Sensitivity ablations (empty tunnel, no inserted object, 60 or 200 frames)

| ablation | labels | note |
|---|---:|---|
| fresh measurement noise each frame, stationary sensor | 0 | measurement noise alone does not label static geometry |
| same, sensor moving 1 m/frame | 0 | pose-consistent motion of a static scene is also stable |
| pose error sigma = 0.02 m | 10,796 in a single frame | sporadic, not gradual |
| pose error sigma = 0.10 m | 0 | bursts are event-driven, so the ablation is not monotone |
| pose error sigma = 0.30 m | 10/60 frames, up to 23,629 | their stated limitation III, quantified |
| displacement 0.1 m/frame (1 m/s), 200 frames | 0 | |
| displacement 1.0 m/frame (10 m/s), 200 frames | 3,778 | one short transient, when the sensor leaves the trackside fixture it was travelling along |
| displacement 1.5 m/frame (15 m/s), 200 frames | 70,828 | 10 burst frames, up to 14,325 labels in one |
| displacement 2.0 m/frame (20 m/s), 200 frames | 44,704 | fewer frames inside the fixture, so fewer labels than 1.5 m/frame |
| displacement 1.5 m/frame at 64-ring/1024-azimuth density | 3,104 | 23x fewer labels than at our measured pattern density |
| 64-ring scan, moving object / static object | 24,187 / 0 | the lower count is not blindness: the moving object is still segmented |

The displacement sweep is the one that matters for us: at metro speed (1.5 m/frame) an **empty** tunnel
produces 70k false dynamic labels in 200 frames, concentrated in 10 burst frames (up to 14,325 in one). This matters here because our ego-motion
comes from KISS-ICP in a tunnel whose longitudinal geometry is close to degenerate, and the project has
no independent trajectory reference. A method whose failure mode is "10k false dynamic points in one
frame when the pose is off by 2 cm" cannot be added to a pipeline that is already alarm-dominated.

### 4. Cost and range envelope (real scans, `doubleT_obstacle` frames 0-20, 177k returns/frame, Apple M4)

`maxRange` | s/frame | implied rate | peak RSS | dynamic labels
---|---:|---:|---:|---:
20 m | 0.049 | 20.4 Hz | 193 MB | 146 |
40 m | 0.107 | 9.3 Hz | 251 MB | 146 |
60 m | 0.220 | 4.5 Hz | 296 MB | 159 |
80 m | 0.429 | 2.3 Hz | 360 MB | 159 |
100 m | 0.752 | 1.3 Hz | 441 MB | 0 (13-frame window, nothing labelled in it) |

This independently reproduces the authors' own statement that the method is real-time only within
20-50 m. Memory follows the per-scan dense observed-voxel grid, whose side is `2*maxRange/voxelSize+1`;
at the 100-300 m the organizers score, the same grid would be 3.4-30 GB and the raycast would take
seconds per scan. The current pipeline runs at 0.115 s/frame total, so this is not a channel we can
afford even as an auxiliary process.

## Why it does not fit this project

1. **Wrong hazard class.** The scored hazards are objects that must not be on the track - a fallen
   object, construction equipment, a person who is standing there. All of them are static in the world
   while the train approaches, and the method is structurally blind to them (code rule above, and 0
   labels in four synthetic static cases). Using it as the detector would trade our main class for a
   class the recordings barely contain.
2. **Range.** The method's detections come from occupancy changes observed inside the map; the authors
   report real-time results only within 20-50 m depending on point density. Our measured-beam-pattern
   synthetic panel has 100% event recall at 10-100 m and 58% at 200 m *with a single scan*, so replacing or gating our
   detector on a 20-50 m channel would cut the detection range the organizers explicitly score.
3. **Cost.** It runs a second map, a KD-tree, a per-scan dense voxel grid and a 4D convolution: measured
   in §4, 0.22 s/frame and 0.30 GB at 60 m and 0.75 s/frame and 0.44 GB at 100 m, i.e. alone 2-6x the
   100 ms budget the pipeline is chasing (current C++ recipe: 115 ms/frame), and it would add a pose
   dependency we cannot yet validate.
4. **Free-space dependence on unresolved timing.** The journal version's metadata states they excluded
   SemanticKITTI from the main analysis because acquisition and deskew problems affect methods that
   rely on accurate free space (the full text was not readable here; treat as reported, not verified).
   This project's sensor timing, deskew and extrinsics are still unverified, so we would inherit exactly
   that exposure.

## What is worth borrowing

1. **Vocabulary and evaluation discipline.** Their separation of "moving during the current scan" from
   "has moved" is the same distinction our empty-but-alarming frames and our `exhaustive: false`
   annotation panel need. It is a citation for the labeling rules we already enforce, not a code change.
2. **Occupancy history as a static-structure prior.** Our background rejection is a plane fit plus
   observed-strip membership. A per-voxel count of distinct scans in which a voxel was occupied would
   give an explicit "how long has this structure been here" statistic that is exactly what rejecting
   infrastructure needs. This is the inverse of their use (evidence of *stability*, not of motion) and
   is a hypothesis, not a measured improvement: it must be tested against the same fixed panels and
   would need exhaustive labels we do not have.
3. **Spatiotemporal aggregation.** Their 4D convolution plus Otsu threshold grows a detection to the
   whole object and suppresses single-voxel noise. Our pipeline already does object-level grouping and
   distinct-frame confirmation, so the honest position is that this is not needed until the current
   nuisance-alarm problem is measured against exhaustive labels.

## Threats to this review

- The IJRR version could not be read (SAGE 403). Its nine-dataset evaluation, the ground-truth
  convention discussion and any sensitivity analysis in the published version are unexamined beyond the
  abstract and search summaries; the workshop version and the code were read in full.
- The synthetic scene is an ideal tunnel with perfect poses, a fixed beam pattern and no multipath; it
  establishes a capability boundary, not field performance. Its labels are still exact by construction.
- The real windows have no per-scan ground truth. The person box is one hand-authored box propagated
  over 36 frames by the detector's own track, so the 95% figure is agreement with a non-independent
  reference, and no precision can be computed anywhere.
- Detection thresholds were left at the authors' defaults; nothing was tuned to make their method look
  worse, and the one harness bug found during this work (a wrong frame convention that injected a
  1 m/frame pose error) was fixed before any of the numbers above were recorded.

## Reproduction

The build is fiddly only because nixpkgs' `onetbb` ships no CMake config; a four-line shim fixes it.
Everything else is the authors' code unchanged.

```sh
git clone --depth 1 https://github.com/vb44/HMM-MOS /tmp/hmm-mos-probe/HMM-MOS   # commit 3f625031

mkdir -p /tmp/hmm-mos-probe/cmake-modules
cat > /tmp/hmm-mos-probe/cmake-modules/FindTBB.cmake <<'EOF'
find_path(TBB_INCLUDE_DIR tbb/parallel_for.h HINTS ${TBB_ROOT} PATH_SUFFIXES include)
find_library(TBB_LIBRARY NAMES tbb HINTS ${TBB_ROOT} PATH_SUFFIXES lib lib64)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG TBB_INCLUDE_DIR TBB_LIBRARY)
if(TBB_FOUND AND NOT TARGET TBB::tbb)
  add_library(TBB::tbb UNKNOWN IMPORTED)
  set_target_properties(TBB::tbb PROPERTIES IMPORTED_LOCATION "${TBB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}")
endif()
EOF

DEPS=$(nix build --no-link --print-out-paths nixpkgs#onetbb nixpkgs#onetbb.dev nixpkgs#yaml-cpp \
  nixpkgs#nanoflann nixpkgs#unordered_dense nixpkgs#eigen nixpkgs#boost nixpkgs#boost.dev | paste -sd';')
cd /tmp/hmm-mos-probe/HMM-MOS
nix shell nixpkgs#cmake nixpkgs#clang --command bash -c "
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$(which clang++) \
    -DCMAKE_PREFIX_PATH='$DEPS' -DCMAKE_MODULE_PATH=/tmp/hmm-mos-probe/cmake-modules \
    -DTBB_ROOT=\$(echo '$DEPS' | tr ';' '\n' | grep onetbb | grep dev) && cmake --build build -j8"
```

clang is required, not gcc: the nix dependencies are built against libc++, and linking them from gcc
fails on `YAML::LoadFile`'s `std::__cxx11` symbols.

Then the probes (the third script default `--binary` already points at the build above):

```sh
uv run python -m tunnel_guard.hmm_mos_probe synth --experiment configs/hmm-mos-probe-synth.json
uv run python -m tunnel_guard.hmm_mos_probe synth --experiment configs/hmm-mos-probe-synth2.json
uv run python -m tunnel_guard.hmm_mos_probe synth --experiment configs/hmm-mos-probe-synth3.json
uv run python -m tunnel_guard.hmm_mos_probe synth --experiment configs/hmm-mos-probe-synth4b.json
uv run python -m tunnel_guard.hmm_mos_probe synth --experiment configs/hmm-mos-probe-synth5.json
uv run python -m tunnel_guard.hmm_mos_probe real  --experiment configs/hmm-mos-probe-real-obstacle.json
uv run python -m tunnel_guard.hmm_mos_probe real  --experiment configs/hmm-mos-probe-real-160-200.json
uv run python -m tunnel_guard.hmm_mos_probe real  --experiment configs/hmm-mos-probe-real-roundt-60-160.json
uv run python -m tunnel_guard.hmm_mos_probe real  --experiment configs/hmm-mos-range-60.json   # cost sweep: 20/40/60/80/100
```

Every `real` recipe needs the recordings under `data/sourcecraft_subset/for_hackathon/` and the recorded
detector run under `build/perf-cpp-opt-20260917/` for the poses (they are the KISS-ICP poses that run
already produced, not a new registration). The `synth` recipes only need `configs/stress-measured-pattern-final.json`
for the scene and beam pattern.
