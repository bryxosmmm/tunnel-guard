# Processing budget on Apple M1 Pro — 2026-09-20

**100 ms/frame is not achieved.** This change reduces detector processing cost
without changing the production recipe. The current machine is an Apple M1 Pro
with 10 logical CPUs; earlier M4 measurements in this branch are not its baseline.
Original measurement baseline: `0371f20`. Aggregate evidence and source/config/native hashes:
[results/runtime-100ms-m1pro-20260920.json](../results/runtime-100ms-m1pro-20260920.json).

## Measured improvement

Three alternating before/after pairs per recording processed the same 30 cached,
consecutive real scans with fresh detectors. Libraries were loaded before timing.
Both variants used the unchanged `configs/detector-native.json` and the same
native binary: the only C++ addition is the rail-bin median function used by the
new Python implementation. Old Python source was loaded from the saved baseline.

| Recording | Before median, ms | After median, ms | Reduction |
|---|---:|---:|---:|
| doubleT_obstacle | 216.3 | 182.6 | 15.6% |
| roundT_doubleT | 180.2 | 146.5 | 18.7% |

Values are medians of the three run medians. The timed region is `Detector.process`,
excluding bag decoding, result serialization and display. Every pair retained
all compared output fields within the comparison tolerance.

Separate continuous replays processed 201 scans per recording, including the
annotated frames 165–200 of `doubleT_obstacle`. Reader prefetch was enabled, so
reader activity can contend with the detector even though decoding is outside
its processing timer.

| Recording | Before p50/p95, ms | After p50/p95, ms |
|---|---:|---:|
| doubleT_obstacle | 237.4 / 266.2 | 206.5 / 223.4 |
| roundT_doubleT | 188.7 / 231.9 | 162.7 / 196.4 |

All **402** final replay frames exceeded 100 ms. The first obstacle frame took
1,041 ms, including lazy library initialization; the round-tunnel maximum was
223 ms. These are processing measurements, not sensor-to-result latency.
Neither the median nor a per-frame deadline passes the requested budget.

## Changes and their scope

- Evaluate the fixed current-scan ground/path model over all candidate support
  together, then retain each candidate's own median. Evaluate reported lateral
  uncertainty at all object distances in one call.
- Reuse the six pointwise classification arrays computed during background
  construction for segmentation of the same input array. Copy the context mask
  before applying background removal, and slice results when the cluster voxel
  grid is identical. A geometry instance using an older cached background has
  no initial classification and computes its own; no classification crosses
  acquisitions.
- Compute longitudinal rail-bin x/y medians in C++, in ascending bin order.
  The two refinement iterations, least-squares solver, rank checks and rejection
  thresholds remain in Python unchanged. The Python median implementation
  remains available when native kernels are disabled.
- Compute association shape costs only for pairs passing the existing position
  gate. Keep the same assignment matrix and rejection costs.
- Avoid the intermediate crop copy for the fixed corridor, and avoid
  computing Python ground arrays that the native ground-profile path discards.

No range limit, point sampling, voxel size, geometry threshold, background refit
cadence, seed or confirmation rule changed. Odometry thread counts 1, 2 and 4
with serial query workers did not improve this workstation's timing; those
experimental recipes/results remain local under `build/` and were not promoted.

## Verification and limits

The actual runner and `panel_report` compared all 402 frames: no missing/extra
frames, measurement-identity errors or changed compared fields. Status, object
IDs, counts, boxes, distances, support counts/witnesses, confirmation, geometry,
motion, pose, observability and health were preserved. Maximum compared floating
difference was 2.54e-13; tolerance is 1e-10 absolute/relative. Discrete decisions
and IDs are exact. This is not a hash comparison of every internal support array.

The Python-only fallback was also replayed before/after on three real scans;
its compared outputs remained unchanged. No automated tests or synthetic
experiments were created or run. These recordings are development data with
nonexhaustive labels: unchanged output is not an accuracy or safety result.

The final 30-frame profile remains under `build/perf-100ms-profile-final/`.
Background construction and KISS-ICP motion dominate. A separate detailed
profile attributed about 33 ms/frame to Open3D plane segmentation alone and
6.4 ms to its voxel downsampling. Inclusive background/geometry stage times
must not be added together. Reaching 100 ms on this machine needs further work
on those stages; reducing plane trials or reusing stale geometry would change
measurements/decisions and is not justified by this performance comparison.

## Reproduction

Before opening the PR on 2026-09-21, this change was rebased onto `8ac5077`.
The upstream removal of corridor-following state and the corrected
`supported_range_m` report are preserved. An additional before/after replay of
30 consecutive real scans from each recording found no changed compared fields
or changed `supported_range_m` values. Evidence:
[post-rebase comparison](../results/runtime-100ms-rebase-20260921.json).
The original 402-frame measurements above remain historical evidence against
`0371f20`; the additional 60 frames verify integration with the newer base.

```sh
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m tunnel_guard.run --experiment configs/perf-100ms-after.json
.venv/bin/python -m tunnel_guard.profile_run --experiment configs/perf-100ms-profile.json
.venv/bin/python -m tunnel_guard.panel_report \
  --panel configs/perf-100ms-comparison.json \
  --output build/perf-100ms-after/comparison.json
.venv/bin/python -m tunnel_guard.native_benchmark \
  --experiment configs/perf-100ms-paired-doubleT_obstacle.json
.venv/bin/python -m tunnel_guard.native_benchmark \
  --experiment configs/perf-100ms-paired-roundT_doubleT.json
```

Use fresh output paths for another run. `perf-100ms-before.json` alone does not
select old code: the original baseline ran at `0371f20` and its source is retained
in `build/perf-100ms-before/source/tunnel_guard`. The paired recipes require that
snapshot. All intermediate/failed timings, detailed profiler script, manifests,
per-frame outputs and source snapshots remain under `build/perf-100ms-*`.
