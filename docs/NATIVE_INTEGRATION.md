# Native integration, 2026-09-17

Integrated Gerasimov `bb8c20c` into our `48977ad`; merge commit `22e6a55`.
The production recipe is `configs/detector.json`. Preserve three background
plane proposals; the separate `native-fast` recipe is not the accepted baseline.

**Current architecture (2026-09-22):** one detector decision path uses the
required `tunnel_guard._native` extension. This report records integration
history; its Python-reference comparisons and dual-backend commands describe
the earlier revision and are not current verification. Build with
`python setup.py build_ext --inplace`. Without the extension, configuration load
fails with this command in its error. Backend-selection keys have been removed. The
Python-only experimental `local_3d` classifier was removed; `bed` is supported.

## Changes

- Native voxel selection, neighbour graph, background operations, component
  statistics, batched evidence counting and association are integrated.
- Ported our height/lateral uncertainty and stepped-contour interval bounds to
  native classification. Preserved context-only background
  queries and exact-support viewer; current component grouping is native.
- Invalid geometry now returns unsupported observability and `unknown` instead
  of indexing an empty classification cache.
- Skip redundant cluster voxelization only for identical grids. Coarse-grid
  uniqueness alone does not guarantee the same fine-grid ordering/cluster IDs.
- Added header packaging, extension header dependency and Docker header allowlist.
  Container installation explicitly imports `_native`; ROS defaults select the
  native recipe. The sole detector recipe is required.
- Replay/benchmark provenance now captures all C++ sources, headers and build
  recipes. Paired benchmarking can load the historical binary independently.
- Added `tunnel_guard.profile_run` for inclusive stage wall timings around an
  actual configured replay; this is instrumentation, not a test suite.

## Verification

Actual runs and machine-readable results: `results/integrated-native-20260917.json`.
No automated tests or suites were created or run.

1. Built the extension and ran a three-real-frame prefix with diagnostic points.
2. Replayed **798 real frames** from three recordings, every frame, seed 20260915.
   Existing panel evaluation found no changed compared fields against
   `build/runtime-context-real`: objects, IDs, distances, geometry, pose, health,
   observability and temporal evidence. Maximum float difference 3.50e-13 at
   tolerance 1e-10; no missing/extra frames or measurement identity errors.
   All **44,291 candidate observations at ≥60 m** retained; no future or duplicate
   timestamps in object evidence histories. These are observations, not events.
3. Three-frame Python reference and native prefix agree within 1e-10.
4. Replayed a real scan with deliberately impossible rail acceptance:
   `unknown`, no crash. Config is explicitly saved; this is not claimed to be a
   naturally failing frame in the default panel.
5. Built an sdist, extracted it, compiled there and imported its native module.
   Docker/ROS2 and target Intel execution were not available and are unverified.
6. Viewer `/metadata`, `/frame?index=100`, `/object` returned the integrated
   platform recording, cloud, corridor, distance and the exact 4653 support points
   for track 11788. HTTP/data behavior was inspected; no new visual-layout claim.

### Performance

Three alternating repetitions, 20 cached real scans each, separate historical
and integrated native binaries, one warmup per variant:

| Variant | Median of run medians |
|---|---:|
| Before (`48977ad`) | 475.68 ms |
| Integrated | 146.51 ms |

**3.25× throughput, 69.2% less processing time.** All three paired output
comparisons agree. This excludes decoding, ROS transport and visualization;
not an end-to-end 10 Hz result or a target-CPU claim.

Full replay processing medians: obstacle 155 ms, platform 126 ms, round-to-double
124 ms; p95 respectively 201, 231, 161 ms. These descriptive timings include
some concurrent build/inspection work and diagnostic capture on nine frames.

### Remaining costs

Explicit wall timers on 10 real scans report per-call medians:

| Stage | Time |
|---|---:|
| Detector.process | 163 ms |
| TrackGeometry construction | 66 ms |
| Background construction (inside geometry) | 56 ms |
| Odometry | 40 ms |
| Clustering including classification | 26 ms |
| Association/evidence | 13 ms |
| Cloud decoding (outside Detector.process) | 29 ms |

Nested medians must not be added. Normal statistics consume about 11 ms of the
background stage. Includes initialization and profiler overhead; only ten scans.
Initial cProfile output had inconsistent parent call counts and is retained in
`build/integrated-native-profile`; it is not used to attribute total stage costs.
The usable wall report is `build/integrated-native-wall-profile/wall-profile.json`.

## Reproduce

From the repository root, with the local environment installed:

```sh
.venv-iteration/bin/python setup.py build_ext --inplace
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/integrated-native-prefix.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/integrated-native-real.json
.venv-iteration/bin/python -m tunnel_guard.panel_report --panel configs/integrated-native-panel.json --output build/integrated-native-comparison.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/integrated-native-no-rails.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/integrated-python-prefix.json
.venv-iteration/bin/python -m tunnel_guard.native_benchmark --experiment configs/integrated-native-paired.json
.venv-iteration/bin/python -m tunnel_guard.profile_run --experiment configs/integrated-native-wall-profile.json
.venv-iteration/bin/python setup.py sdist --dist-dir build/integrated-native-dist
.venv-iteration/bin/python -m tunnel_guard.review_viewer --run build/integrated-native-real --bag data/sourcecraft_subset/for_hackathon/doubleT_platform --port 8768
```

Runner output paths must be new; existing evidence is deliberately not overwritten.
Paired recipe expects the local `build/integration-baseline` export of `48977ad`,
compiled with that revision's setup.py. Its binary filename is environment-specific;
update `baseline_native` when reproducing elsewhere. The baseline source/binary,
recipes, hashes, failed profiler output and logs are retained locally.

## Next priorities and extended dataset

No new dataset has been downloaded or evaluated. Suggested order:

1. **Freeze this baseline and qualify incoming data before tuning.** Inventory
   metadata, PointCloud2 layout/returns, measurement/record clocks, frames,
   gaps, intensity/rings, IMU/odometry and mounting changes with the existing
   inspection CLI. Split by independent run/site/install, not neighbouring frames.
   Reserve a blind portion before inspecting detector outputs; missing metadata
   stays unknown. Compare the frozen recipe first on the development portion.
2. **Measure real end-to-end latency and remove decoding copies.** Current decoding
   costs ~29 ms beyond inference. Profile conversions and allocations, then use
   layout-aware views/buffers while preserving all coordinates and timestamps.
   Validate packed/padded fields, endian/layout variants and multiple returns on
   real inputs. Measure receipt-to-result age, queue length and dropped scans on
   the target ROS machine; never disguise dropping as full-rate processing.
3. **Profile background construction and motion on the broader panel.** They are
   larger costs than graph construction now. First remove repeated searches,
   copies or thread setup with unchanged outputs. Reusing background across
   frames is a separate algorithmic experiment requiring causal pose transforms,
   uncertainty growth, expiration and immediate invalidation on degraded motion.
   Do not simply reuse the previous frame's obstacle mask or reduce distant points.
4. **Build independent event-level labels and evaluate useful alarms.** Mark
   object support, rail-relative position, visible time interval, visibility and
   uncertainty independently of our predicted tracks. Record whether a reviewed
   interval is exhaustive. Annotate clearly empty observed corridors as negatives;
   an unseen/unsupported corridor is not a negative. Evaluate event detection,
   first reliable range, confirmation delay, nuisance duration and ID switches.
   Do not count the 36 propagated person boxes as 36 independent events.

ML/learned nuisance rejection becomes a justified next experiment if independent
labels show recurring geometric ambiguities across installations. Synthetic
insertion remains useful for controlled failure exploration and scarce cases,
but cannot establish real detection range or field safety. No model training is
required to accept and diagnose the incoming dataset.

## Optional CUDA and odometry-overlap evaluation — 2026-09-27

This is **not** a replacement for `configs/detector.json` or evidence of a
200 ms deployment deadline. The measured host was an EPYC 7742 with an RTX 4080;
the paired six-recording runs were pinned to CPU cores 0–7, Python 3.12,
seed 20260915. `processing_s` covers `Detector.process`, not bag decode,
serialization, ROS delivery, or sensor-to-warning age. These recordings have
no exhaustive negatives or independent swept-envelope ground truth.

The device `mutual_graph` port enumerates the same mutual-radius pairs as the
CPU, with sorted CSR rows. Cell starts must be **sorted point positions**, not
the exclusive-scan rank of the cell head. The CUDA path also retains the
existing native plane proposal. In Python, matched tracks reuse the inverse
innovation covariance already calculated for assignment; matched covariance
updates are batched without changing the per-track state-vector multiplication.
None of these changes alters the configured geometric thresholds.

The pinned KISS-ICP 1.3.0 binding patch releases the Python GIL during native
voxel downsampling, registration, and map update. With deskew disabled and
native odometry preprocessing, `overlap_motion_geometry=true` computes a
pose-independent, freshly fitted geometry while odometry runs on one worker.
The registered frame is compared to the exact strict-range frame before
association; a different frame aborts. Deskew or cached background models
are incompatible with this scheduling. An unpatched wheel rejects the option
at configuration load. This custom wheel is **optional** and is not installed
by the CPU-only ROS container.

| Complete recorded scene | Frames | CPU p50 ms | CUDA overlap, ICP 1 p50 ms | CUDA overlap, ICP 8 p50/p95 ms |
|---|---:|---:|---:|---:|
| doubleT_obstacle | 201 | 664 | 264 | 285 / 358 |
| doubleT_platform | 345 | 625 | 338 | 246 / 299 |
| roundT_doubleT | 252 | 701 | 315 | 259 / 317 |
| roundT_pressureGate_roundT | 268 | 1128 | 718 | 297 / 368 |
| roundT_squareT_pressureGate_squareT | 545 | 832 | 452 | 256 / 324 |
| squareT_platform_squareT_switch | 877 | 515 | 222 | 218 / 289 |

`configs/latency-full-six-{cpu,gil-overlap,gil-odom8}-20260927.json` fix the
same **2,488 scans**. The recorded-run comparator found zero changed discrete
fields, zero missing/extra frames, and maximum compared numeric difference
**0.0** for CUDA overlap/one ICP worker versus CPU. Eight ICP workers kept
the same decisions but varied compared numeric values up to **1.15e-12**
relative to CPU; TBB reduction scheduling loses bitwise pose repeatability.
An exact six-recording replay checks *this* corpus, not unseen sensors,
calibration, target hardware, or field recall/precision.

The one-worker profile on the pressure-gate recording exposed a **633 ms**
median odometry stage, versus 68 ms geometry, 101 ms clustering, and 33 ms
association (inclusive stage timings cannot be summed). Eight workers reduced
that scene to 283 ms p50 in an isolated run; the full-panel p50 was 297 ms.
On the obstacle recording eight workers were slightly slower than one
(285 versus 264 ms in the complete-panel runs). Pinning sixteen CPU cores
instead of eight reduced the gate scene to **227 ms p50, 301 ms p95**, but
requires twice the core allocation and still misses 200 ms. A trial overlapping
motion with the stock GIL-holding wheel was slower; its negative outputs and
recipe remain in `build/latency-20260927/cuda-overlap`.

Rebuild the optional upstream MIT-licensed binding from the exact v1.3.0 tag
(commit `b16835283aee62f7d5e2bdf6c1c3bb2930de74ff`), retaining its license:

```sh
git clone --branch v1.3.0 https://github.com/PRBonn/kiss-icp.git build/kiss-icp-v1.3.0
git -C build/kiss-icp-v1.3.0 apply "$PWD/patches/kiss-icp-1.3.0-release-gil.patch"
uv build --wheel --directory "$PWD/build/kiss-icp-v1.3.0/python" \
  --out-dir "$PWD/build/kiss-wheel" --build-constraints "$PWD/patches/kiss-icp-build-constraints.txt"
uv pip install --python .venv-remote/bin/python --no-deps --reinstall \
  build/kiss-wheel/kiss_icp-1.3.0-*.whl
.venv-remote/bin/python setup_cuda.py build_ext --inplace
PYTHONPATH=. taskset -c 0-7 .venv-remote/bin/python -m tunnel_guard.run \
  --experiment configs/latency-full-six-gil-odom8-20260927.json
```

The measured Python-3.12 Linux wheel is retained locally under
`build/latency-20260927/kiss-wheel/`; its SHA-256 is
`169403762d157a758e0748db5c19bdba4e466bcaa9980593beedbbded95fd765`.
Every new run manifest records the installed binding's binary SHA-256 and
GIL-release marker, native binary hashes, patched source, recipe, and seed.
The fixed comparison artifacts are
`build/latency-20260927/full-six-{comparison,odom8-comparison}.json`.
**Decision:** preserve the exact one-worker option and the faster, non-bitwise
eight-worker research recipe; neither meets a 200 ms median on all scenes or
a 200 ms tail on any measured scene. Do not meet the number by reusing stale
background, discarding sparse returns, weakening ICP validity, or changing
the object/clearance policy.

### Review corrections — 2026-09-28

The KISS-ICP patch now includes context and applies with the documented plain
`git apply`. Existing deskew configurations keep the library preprocessor when
`odometry_native_preprocess` is omitted; explicitly enabling both remains an error.
The CUDA graph's out-of-range CPU fallback now runs after reacquiring the GIL.
CUDA slab/staging access and plane worker-pool submissions are serialized across
callers, with no mutex wait holding the GIL; proposal generators are thread-local.
Native proposals require exactly three samples. `cuda_entry_points` lists GPU-capable
kernels, not actual call counts, and no longer lists the CPU-only plane proposal.
The Open3D MIT notice, CUDA source/build recipe and dependency patch are included
in source distributions and the native-source capture.

These CUDA corrections require a fresh real-GPU replay; the earlier GPU measurements
above predate them. Local deskew/recorded-delivery and installed-viewer evidence is
reported separately in `results/review-integration-20260928.json`.

### Follow-up: the actual acceptance gate is p95 ≤200 ms

The user selected **p95 processing ≤200 ms on the measured RTX 4080
host**, rather than median ≤200 ms. On a second verified RTX 4080 / EPYC 7742
Vast instance with the same eight-core pinning, the unchanged eight-worker
detector replay measured **354 ms p95** on all 268 pressure-gate scans and
**343 ms p95** on all 201 obstacle scans. Its compared decisions match the
earlier host's saved run; cross-host numeric drift stayed ≤8.29e-14.
The two per-scene recipes and fresh outputs are under
`configs/latency-tail200-baseline-{gate,obstacle}.json` and
`build/latency-tail200/baseline-{gate,obstacle}-odom8/`.

An instrumented actual pressure-gate replay assigned **156 ms p50 / 243 ms
p95** to the ICP registration call, versus voxelization 11 / 27 ms and map
update 3 / 4 ms; source and samples:
`build/latency-tail200/profile-gate/motion-profile.json`. On the obstacle
scene, the *serial* geometry-construction, clustering and association stages
alone consumed 229 ms p50 and 296 ms p95 across 201 frames in the actual
stage-instrumented run. Those are same-frame sequential calls and exclude
the odometry parallel path, but include profiler overhead. Individual p95
values were 116, 155 and 40 ms respectively (not additive quantiles).
Thus speeding up ICP alone cannot meet the obstacle-scene target.

We measured a semantics-preserving KISS-ICP nearest-neighbor cache (the
strict first-minimum comparison still uses the same floating-point norm).
Gate p95 improved **354 → 335 ms**; obstacle p95 changed **343 → 341 ms**.
The two recorded-scene comparisons kept decisions, with compared numeric
drift ≤9.23e-14, but the target was missed by >130 ms. A four-worker ICP
trial with the cached version was substantially worse: gate/obstacle p95
**566/542 ms**. The cache variant is **not adopted**; source patch, wheel,
per-frame outputs and verdicts are retained under `build/latency-tail200/`.
That patch is an experiment artifact, not a second deployment dependency.

No honest 200 ms p95 claim follows from median improvements or from
GPU equivalence. On this eight-core host the remaining serial geometry and
segmentation critical path requires major exact-algorithm acceleration,
while ICP is independently long on pressure-gate frames. Porting these
without losing candidate support, train-clearance evidence, and hard
unknown-state behavior needs a new independently replayed implementation,
not an unbenchmarked threshold or fewer accepted points.

### Exact-kernel latency iteration — 2026-09-28

**Gate not met.** On the verified RTX 4080 / EPYC 7742 proxy (eight CPU cores
`0-7`, Python 3.12.3, CUDA 12.6, seed `20260915`), the final uninstrumented
`Detector.process` p95 is still above 200 ms in **all six** complete development
recordings. The CASE.md stand specifies an i7-9700E and RTX 4070 Ti SUPER
running Ubuntu 22.04 / ROS 2 Humble; an RTX 4080 with EPYC on Ubuntu 24.04
does **not** establish latency on that stand. Timings exclude bag decode; the
fixed panel uses 2,488 actual scans and reports per-scene, not independent-event,
percentiles. Recipes: `configs/latency-200-six-nocopy-20260928.json` and
`configs/latency-200-six-nocopy-comparison-20260928.json`. The run, original
per-frame outputs, hashes, and failed trials are retained under
`build/latency-200-20260928/`; the machine-readable failed promotion decision
is `results/latency-200-20260928.json`.

The saved six-scene eight-worker reference is from a previous same-class
Vast host; its p95 is historical context, not a controlled paired speedup.
The 201-frame unchanged baseline **on this host** measured 353.35 ms p95,
versus 309.62 ms in the final run. Replays on the same host varied by several
milliseconds. The fixed-panel comparator excludes timing fields.

| Recording | Frames | Saved eight-worker reference p95 (ms) | Final p50 / p95 (ms) |
|---|---:|---:|---:|
| doubleT_obstacle | 201 | 358.4 | 244.6 / 309.6 |
| doubleT_platform | 345 | 299.4 | 190.1 / 238.9 |
| roundT_doubleT | 252 | 317.3 | 202.7 / 267.9 |
| roundT_pressureGate_roundT | 268 | 368.4 | 213.1 / 276.3 |
| roundT_squareT_pressureGate_squareT | 545 | 324.4 | 198.7 / 258.7 |
| squareT_platform_squareT_switch | 877 | 288.9 | 197.8 / 262.8 |

The final comparison at `build/latency-200-20260928/six-nocopy-comparison.json`
has **zero changed compared fields**, zero missing/extra frames or measurement
identity errors, and identical candidate and support-voxel observation totals
on every recording. Largest compared numeric difference is `1.02e-12`.
The comparison establishes output equivalence on these development recordings,
**not** field recall, precision, blind-set behavior, or a clear-route guarantee.
The obstacle recording preserves its 155 `obstacle` frames; pressure-gate
preserves 254 `no_obstacle_observed`, one `unknown`, seven
`unresolved_obstacle`, three `obstacle`, and three `candidate` frames.
No thresholds, confirmation intervals, or return support were relaxed.

Changes measured together:

- Induce unit-weight CSR subgraphs by visiting selected native graph rows
  instead of SciPy's double slice. The isolated split stage median fell
  37.95 → 33.77 ms; a deferred-construction attempt made no useful p95
  improvement and was reverted.
- Move the entire radius-neighbor selection, FP64 covariance and fixed
  eight-sweep 3×3 Jacobi normal calculation to CUDA for large windows,
  preserving the CPU path for small/out-of-range inputs. On the 201-frame
  obstacle scene, `normal_statistics` median fell 26.5 → 1.9 ms and
  uninstrumented process p95 353.35 → 324.32 ms. GPU MODE's publicly
  benchmarked `eigh` kernels target larger FP32 matrices; substituting an
  unrelated analytic eigensolver risks degenerate-normal behavior. No
  competition code was copied.
- Preserve pinned MIT-licensed KISS-ICP v1.3.0's 27-voxel order and strict
  first-minimum tie rule, but skip a voxel's hash probe only when its
  conservatively guarded AABB distance lower bound exceeds the current
  closest distance. On the 268-frame gate scene, the instrumented native
  search p95 fell 234 → 82 ms; the *instrumented* detector p95 340 → 293 ms.
  Native profiling adds overhead; compare uninstrumented panel numbers above
  for the gate. The exact source patch is
  `patches/kiss-icp-1.3.0-gil-aabb.patch`; the resulting Python 3.12 Linux
  wheel (SHA-256 `e8b49a37269cdf8df64bc8481b2dbc0d581f424e4f9e85024b0ab602ee6bcf1b`)
  is at `build/latency-200-20260928/final-wheel/`.
- Let the existing native crop/voxel reducer filter the full registered
  frame rather than materializing a duplicate cropped cloud. A three-frame
  real-bag diagnostic verified every geometry voxel maps back to its exact
  decoded PointCloud2 source slot; all 2,488 final-panel outputs still match.
  A label-separated batched KD-tree candidate was slower (gate p95 319 ms
  versus 312 ms) and reverted.

To reproduce the wheel, clone the v1.3.0 tag (commit
`b16835283aee62f7d5e2bdf6c1c3bb2930de74ff`), apply **only**
`patches/kiss-icp-1.3.0-gil-aabb.patch`, then build with the same constrained
`uv build` command above and reinstall it before running the final recipe.
Do not apply the older GIL patch on top: the new patch includes it. The wheel
hash captures the measured compiler/dependency build, and the replay manifest
captures the installed binding hash (`8f11f56218950d3c260a92fb304d0d7e5c3ce6259a97affe7dbba5554532ea50`);
rebuilding is not a bitwise guarantee. The full-panel manifest was recorded
before its source-snapshot list included the new patch; a separate three-scan
actual replay under `build/latency-200-20260928/source-capture/` preserves
the patch bytes, SHA-256 `6930d093fbcc153c199fb28134f55e3be65f707681ad394bd9fd6501d7cff76c`,
and confirms the same installed binding hash. Its startup-frame latency is
not a substitute for the complete-panel p95.

KISS-ICP is MIT-licensed, and the optional CUDA backend additionally requires
NVIDIA's CUDA toolkit/device runtime. The exact algorithm avoids changing
recall/nuisance behavior **on the fixed panel**, at the cost of GPU/compiled
wheel portability and duplicated host/device transfers.

The measured remaining obstacle critical path is geometry p50 74 ms, clustering
92 ms (density 51 ms, split 34 ms), association 35 ms and approximately 72 ms
bag decode *outside* `Detector.process`; motion runs concurrently at p50
105 ms. These are inclusive, overlapping and scene-specific times, **not
additive**. Pressure-gate motion is p95 140 ms after search pruning, while
serial geometry+clustering+association remains the bottleneck. Meeting the
200-ms p95 gate requires an independently validated acceleration of these
serial stages; turning off surface analysis, reducing point support, changing
the odometry iteration cap, or claiming unknown coverage as clear would
misrepresent safety rather than solve latency.

### Exact ground-proposal batching and external-kernel screening — 2026-09-28

**The 200 ms gate remains unmet.** On a second RTX 4080 proxy with an EPYC
7R13 pinned to eight cores (`taskset -c 0-7`, Python 3.12.3, CUDA 12.6,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=8`), the complete 2,488-scan
development panel yielded these uninstrumented `Detector.process` p95 values:

| Recording | Frames | p50 / p95 (ms) |
|---|---:|---:|
| doubleT_obstacle | 201 | 188.8 / 248.4 |
| doubleT_platform | 345 | 145.9 / 178.7 |
| roundT_doubleT | 252 | 155.2 / 202.4 |
| roundT_pressureGate_roundT | 268 | 160.7 / 212.7 |
| roundT_squareT_pressureGate_squareT | 545 | 148.8 / 197.6 |
| squareT_platform_squareT_switch | 877 | 147.9 / 197.3 |

The sole retained source change batches the **same 180 seeded three-point
ground-plane draws** in `robust_plane` into NumPy's batched 3×3 solve, with the
original ordered per-draw solve as a fallback if any matrix is singular. The
inlier predicate, first-maximum tie ordering, refit, railhead measurement,
surface rejection, density support and object decisions are unchanged. An
instrumented 469-scan stage replay measured ground fitting p50/p95
13.56/31.52 → 6.62/9.40 ms on `doubleT_obstacle` and 10.52/13.33 →
6.69/9.07 ms on `roundT_pressureGate_roundT`; instrumented times include
profiling overhead. The same-host complete uninstrumented paired replay was
207.16/273.46 → 188.32/246.14 ms on the obstacle recording and
170.37/234.53 → 159.21/223.20 ms on the gate recording; those end-to-end
differences also contain scheduling variance and must not all be attributed
to the solver. All 2,488 full-panel frames have zero changed compared
decision/support fields, missing/extra frames or measurement-identity errors
against the previous six-scene output; the largest compared numerical
difference was 5.30e-13. The panel is nonexhaustively annotated: unchanged
outputs establish equivalence here, **not** recall, precision or field safety.

Primary-source screening found no plug-in GPU clustering with matching
adaptive mutual-radius edges, running-surface support and instance behavior:
[`cuda_cone_rush`](https://github.com/leonardonels/cuda_cone_rush) uses fixed
voxel connectivity and FP32 clock-seeded ground proposals;
[`cuPCL`](https://github.com/NVIDIA-AI-IOT/cuPCL) distributes precompiled
CUDA 10.2 libraries and its published GPU/PCL clustering counts differ;
[`cuML DBSCAN`](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.DBSCAN/)
uses a global `eps` and `min_samples`. None was silently substituted.
BSD-licensed [`nanoflann`](https://github.com/jlblancoc/nanoflann) was
benchmarked instead on ten exact elevated-core/border inputs preserved from
60 real scans. The five largest pairs saved about 1.0–1.2 ms per tree
build/query (SciPy 2.97–3.48 ms, nanoflann 1.80–2.48 ms), with no mismatched
nearest indices or distances on those samples. This is too little evidence
or total-tail benefit to justify vendoring a new dependency, especially
without auditing exact-distance ties on the complete panel. Brute-force
nearest search is worse: the observed 839 split-tree calls imply
893,574,987 candidate core/border pairs. The earlier fused child-degree
trial changed no compared fields but improved p95 only 1–2 ms within
run-to-run variation; it was reverted. Recipes, positive and negative
measurements, replay manifests and original frame outputs are under
`configs/latency-200-*20260928.json`,
`build/latency-200-split-20260928/`, and
`results/latency-200-ground-batched-20260928.json`.

The EPYC 7R13 is faster than the prior EPYC 7742 reference host; cross-host
p95 differences are **not** speedup evidence. Neither RTX 4080 proxy is the
specified i7-9700E / RTX 4070 Ti SUPER stand. Geometry, clustering and
association remain serial on the current scan; eliminating the entire
measured split stage would still leave the obstacle scene above 200 ms p95
on this proxy. Further latency reduction requires independently verified
acceleration of more than one stage without discarding uncertain or weak
support, and final acceptance still requires the actual stand.

### Measured attribution on a second proxy, and the exact reductions it justified — 2026-09-28

A second proxy (RTX 4080, AMD Ryzen Threadripper PRO 3955WX pinned to eight
cores, same seed and pinned dependencies, built by
`build/latency-200-radius-20260928/setup_remote.sh`) reproduced the gate
failure: **five independent runs of the unchanged detector on
`doubleT_obstacle` gave p50 191.8-195.4 ms and p95 249.9-252.1 ms.** Sampling
(`py-spy`, 201 scans, low overhead) plus a widened `profile_run` closed the
accounting, in milliseconds per scan, inclusive and **not additive**:

| Consumer | median | p95 band |
|---|---:|---:|
| `TrackGeometry.__init__` (background 34, rail 12.9, robust plane 6.5, ground profile 3.4) | 57.6 | 59.7 |
| `cluster_candidates` (density 42.1 incl. split 28.6, `structural_mask` 10.8, `cluster_objects` 6.2, own loop 21.0) | 74.8 | 112.0 |
| `_associate` | 28.3 | 47.6 |
| `process` own lines (range/crop/reduce, mounting, bins, summary) | 21.0 | 21.5 |
| `_motion` (concurrent) | 73.6 | 75.1 |

Inside the split, `induced_subgraph` is 11.4 ms, `connected_components`
4.2 ms and the KD query 2.7 ms; the per-scan `cluster_objects` call, the
association stage and the split are the only consumers that move with the
p95 band (`+24.9`, `+19.3`, `+11.7`), while segment-plane, the KD builds,
voxel down-sample, rail profile, structural mask, robust plane and motion are
flat to within 0.1 ms. **What is not explained by any input size is the
stall**: `cluster_objects` measured 61.3-62.6 ms on 8 of 201 scans (median
6.04, p95 7.46) with identical inputs (cloud 22.9k-24.7k, 1417-1591 labels,
273-330 accepted components), and the slowest scans differ between runs.
A ten-fold stall at constant input size is host-side scheduling or memory
behaviour, not an algorithmic term, and it is the reason p95 sits ~57 ms
above p50 on this proxy.

Retained, all verified as **zero changed compared fields** on real scans:

- Per-component evidence counts (six predicates plus two extrema) are taken
  in one pass over the concatenated support rows instead of per object:
  310 objects had made ~3 100 NumPy calls per scan, 7.0 s of the 39.1 s
  main-thread profile.
- The raw scan is no longer copied when the range filter keeps every row.
- The running-surface partition no longer enters a label that provably cannot
  split: an anchored body needs `immediate_min_voxels` points of one elevated
  child spanning `immediate_min_height_m`, so a label holding fewer, or whose
  elevated core does not span that height, is excluded by two reductions.
  Measured on 201 scans: **12.3 -> 9.6 candidates per scan, 526 of 2 465
  (21.3 %) removed.**
- Accepted structure cells are counted with `np.bincount` instead of the
  unbuffered `np.logical_or.at` accumulation, and an unused per-call index
  fill was dropped from the native `within_radius`.

Rejected on measurement, not on preference: an exact cell grid inside
`within_radius` (the caller makes 88 calls per recording, at most 96 x 96
pairs, 1.7 ms in total - reverted); a box-maintaining footprint merge (151
calls, median 2 and maximum 10 child groups - reverted as unmeasured
complexity); five thread allocations for odometry and query workers (p95
249.9-252.1 ms in every one); `nanoflann` (about 1.0-1.2 ms per build/query
on the largest preserved split inputs); and every screened GPU clustering
kernel, none of which preserves these decisions' adaptive edges and support
rules.

The retained bundle is verified bitwise-equivalent in decisions and support
(largest compared numerical difference 8.6e-14) and 1.5 ms faster at p50 on
one local CPU configuration; on the local configuration it improved p50
140.66 -> 139.15 ms and p95 164.29 -> 161.88 ms. That is far short of the
~50 ms the proxy needs, and **the six-scene CUDA panel could not be re-run**:
the rental account ran out of credit, the stopped instance could not be
restarted, and it was destroyed after its artifacts were collected.
`results/latency-200-radius-20260928.json` records the evidence, the negative
results and the untested leads (chief among them that the ~60 ms stalls may
be interference from the bag-decode producer thread - a fixed
`prefetch_depth` is exactly equivalent and testable). No proxy result
substitutes for the stand.

### Second proxy: reader-thread offload, a concurrency defect, and a device port rejected on measurement — 2026-09-28

**Five of six recordings now meet p95 ≤ 200 ms on the proxy; `doubleT_obstacle`
does not.** Host: RTX 4080 on an AMD Ryzen 7 5700 pinned to eight *physical*
cores (`taskset -c 0-7`; `thread_siblings_list` is `0,8`, so those are eight
distinct cores), 31 GiB, Ubuntu 24.04, Python 3.12.3, CUDA 12.6. This host is
faster per core than either earlier proxy, so the numbers below flatter the
result and are **not** stand latency. Every comparison is paired: the
pre-change tree was rebuilt from the run's own `source/` snapshot into
`/workspace/lct-baseline` with the same venv, so both trees ran on the same
host, data and configuration.

| Recording | Frames | p50 / p95 (ms) | p95 gate |
|---|---:|---:|---|
| doubleT_obstacle | 201 | 159.0 / 225.3 | over |
| doubleT_platform | 345 | 117.7 / 164.5 | met |
| roundT_doubleT | 252 | 127.2 / 180.6 | met |
| roundT_pressureGate_roundT | 268 | 131.0 / 194.9 | met |
| roundT_squareT_pressureGate_squareT | 545 | 121.3 / 181.2 | met |
| squareT_platform_squareT_switch | 877 | 118.0 / 171.5 | met |

The complete 2 488-frame panel has **zero changed compared fields**, zero
missing frames, zero measurement-identity errors and a largest compared
numerical difference of 5.6e-13 against the previously saved six-scene
reference, so every decision, support count and reported field is unchanged.
Recipes: `configs/latency-200-six-fixed-20260928.json` and
`configs/latency-200-six-fixed-comparison-20260928.json`; evidence:
`results/latency-200-20260928-second-proxy.json`.

**One retained runtime change.** The bag reader now computes the scan
derivations that depend on the configuration alone - the range selection, the
open-range subset the geometry stage classifies, its context mask and the
voxel reduction - and hands them to `Detector.process`, which no longer repeats
them. The reader is idle for most of the detector's per-scan work, so this
moves about 11 ms off the critical path: consumer p50 172 → 159 ms and wall
44.9 → 34.5 s for the 201 obstacle scans. At `prefetch_depth: 0` there is no
reader thread, the same code measures 139-148 ms p50, and the pipeline is 22 %
slower overall - so the offload trades 10-16 ms of the gate metric for 23 % of
throughput, and both configurations still miss the obstacle gate. Both are
honest; the throughput-favouring one is the default because a real vehicle
consumes scans, not p50 numbers.

**A concurrency defect, found by the panel and fixed.** The first version had
the reader call the *device* module for two of those kernels. The device
module keeps its scratch in a shared slab, so two threads calling a device
entry point at once corrupt each other: on `doubleT_obstacle` the frame states
changed from `{34 no_obstacle, 10 unresolved, 155 obstacle, 2 candidate}` to
`{40, 14, 137, 6}` and every frame's object list changed, while the same code
at `prefetch_depth: 0` reproduced the reference exactly. `accelerator.cpu_native`
now exists to name this rule: **only the detector thread may marshal the device
entry points**; a reader or any other thread must use the CPU kernels, whose
scratch is thread-local. This is a latent hazard for the project, not a fact
about one call site.

**A device port was written, verified, measured and reverted.** `cluster_objects`
is the largest tail contributor on this host (+31 ms of the +64 ms p95 band),
and it spikes to 75-82 ms at *constant* input size, so a device implementation
was the natural response. The port was completed: a radix sort keyed on
`(label, point index)` so a component's members stay in ascending point order
(the order the CPU's counting sort produces, which the witness selection
depends on), one thread per component applying the same admission rules, and
the same 26 returned blocks in the same order. It reproduces the CPU kernel
**byte for byte on all 26 blocks over 40 real scans**. It is also **2.2×
slower per scan inside the frame**: 7.93 → 17.10 ms per scan (p50 5.53 →
14.94), while in isolation it is only 0.61 → 0.81 ms. About fifty synchronous
copies and four device synchronisations per call cost more than this kernel's
CPU work at 24 500 points and ~1 500 components, which is the module's own
stated policy for small calls ("small calls stay on the CPU ... a documented
policy, not a silent fallback"). It was reverted, not kept behind a threshold
that this workload never crosses.

**Measured neutral, and kept anyway** (each exact): the component builder's
arena storage (`cluster_objects` 5.59 → 5.53 ms median), the split rewrite
(`_separate_running_surface` 12.96 → 13.08 ms; `induced_subgraph` is called
705 times in *both* trees — 3.5 per scan, not the 54 per scan measured on the
earlier host, so the premise of that change did not hold on this
configuration), and the association-loop hoists. The split rewrite also
carried a real defect that the panel caught and the obstacle-only local checks
had not: a core child holds the border points assigned to it, and those are
not elevated, so the anchored body must use the elevated-only neighbour count
(`group_degrees` now returns both, and `check_global_labels.py` verifies the
per-parent relabelling on random graphs). With that corrected, the panel
returns zero changed fields on all 2 488 frames.

Rejected without a run, on analysis: the grid nearest-neighbour replacement
(at the density query's 0.75 m cell size the 27-cell scan visits hundreds of
candidates per query and is slower than the tree; the background query's net
saving is ~3-4 ms against a tie-break `cKDTree` does not document).

The **remaining obstacle tail** is unchanged in character: p95 = median + 64 ms,
of which `cluster_objects` +31 and the association loop +23, while the segment
plane, the KD builds, the voxel down-sample, the rail profile, the structural
mask and motion are flat to within 0.3 ms. One recording out of six is over
the gate on a host faster than the stand; the stand itself remains unmeasured,
and no proxy substitutes for it.

#### Running the gate on the stand

The acceptance measurement has not been taken on the i7-9700E / RTX 4070 Ti
SUPER, and nothing in this file substitutes for it. What the stand run needs,
so that it can be taken without re-deriving it:

1. **The device backend must exist in the container.** `Dockerfile` builds only
   `python3 setup.py build_ext --inplace`, which produces the required CPU
   extension. The measured recipes ask for `native_backend: "cuda"`, and
   `load_config` refuses that request when `tunnel_guard._native_cuda` is not
   importable, so the container fails loudly rather than silently running the
   CPU backend - but it *will* fail. Build `python3 setup_cuda.py build_ext
   --inplace` in an image that has the CUDA toolkit (the RTX 4070 Ti SUPER is
   `sm_89`, the same architecture every measured recipe was built for).
2. **Run the panel**, on the stand, with the eight cores the stand has and no
   CPU pinning needed (8C/8T, no SMT to exclude):
   `python3 -m tunnel_guard.run --experiment configs/latency-200-six-fixed-20260928.json`.
   Timings that matter are the per-recording `processing_ms` in each
   `<bag>.jsonl` summary line: the gate is p95 ≤ 200 ms on `Detector.process`
   for every one of the six recordings.
3. **Compare the outputs**, which must be *host-independent*: the same run's
   jsonl against the saved panel
   (`python3 -m tunnel_guard.panel_report --panel configs/latency-200-six-fixed-comparison-20260928.json ...`
   with the `after` paths repointed at the stand's outputs). Zero changed
   compared fields is the quality gate; the reference's field values do not
   depend on which core produced them.
4. **Expected result, from measured components rather than a guess.** The
   proxy's eight physical cores sustain roughly 4.0-4.2 GHz (65 W part); a
   35 W eight-core Coffee Lake sustains roughly 3.0-3.3 GHz all-core, so the
   stand should be about **0.76 of the proxy per core** and this workload,
   which is ~85 % per-core-bound, should land near **1.3× the proxy's times**:
   obstacle ≈ 210 / 297 ms p50/p95, platform ≈ 155 / 217, roundT_doubleT ≈
   168 / 238, gate ≈ 173 / 257, roundT_squareT ≈ 160 / 239, squareT ≈ 156 /
   226. On that model **all six recordings are over the gate on the stand**,
   and the work still needed is ~1.4-1.5× on the serial path, concentrated in
   the two measured tail items above. The all-core clock is the load-bearing
   assumption and it is not measured: at 3.5 GHz the factor is ~1.15, at
   2.6 GHz ~1.5.



