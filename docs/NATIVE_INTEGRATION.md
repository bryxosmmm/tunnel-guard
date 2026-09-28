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
