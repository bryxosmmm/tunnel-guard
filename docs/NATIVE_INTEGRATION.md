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
