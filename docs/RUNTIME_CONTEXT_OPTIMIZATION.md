# Avoid unused background work — 2026-09-17

Base revision: `2c69efd`. This is a runtime change, not a new detection rule.
No maximum-range, sampling, clustering, uncertainty or confirmation parameter was
relaxed. The previously corrected contour and sparse-return policy remain active.

## Profile and implementation

A fresh cProfile run on ten consecutive real `doubleT_obstacle` scans attributed
1.003 seconds to `TunnelBackground.mask` and 1.181 seconds to `density_labels`.
These are cumulative stage costs, excluding most startup/import work; profiling
and warmup affect full-run totals.

In diagnostic frame 0 there are 72,647 geometry representatives, but only 22,722
belong to segmentation context before background removal. The old implementation
searched the surface-normal tree and evaluated plane masks for every geometry
point even though the result was only consumed for context points. Protected
returns cannot be removed either.

`TrackGeometry.classify` now calls the frozen background model only for
`context & ~protected`. This is safe because `TunnelBackground.mask` computes each
query point's decision against the already fitted surface model; it does not fit
or pool new geometry from the query set. Background fitting, normal estimation,
protrusion support and protection rules are unchanged. Unused and protected
points remain in the original input and observability records.

`cluster_candidates` now performs one stable sort of component labels instead of
scanning the full label array for each component. Stable grouping preserves
component order and original point order within components, including tied
minimum-distance witnesses and temporal spatial evidence.

After these changes the same profiled ten-scan run used **0.318 seconds** for the
background mask, about **100 → 32 ms per frame**. Clustering as a whole, including
background classification, changed from 2.626 to 1.895 seconds. The neighbour
construction stage itself remained about 1.19 seconds, identifying the next
substantial bottleneck. Profiled median detector processing changed from 582 to
498 ms; this cold-prefix comparison is not the controlled timing result below.

## Full-recording verification

All **798 real scans** completed. Against `build/envelope-interval-real`, no
compared status, object/ID/support, distance, geometry, motion, pose, observability,
health or gap-reset field changed at the established absolute/relative tolerance
of 1e-10. Maximum compared floating difference was 4.30e-13. No missing/extra
frames, acquisition-identity mismatches, future evidence or duplicate evidence
timestamps were reported. All 44,291 far candidate observations (≥60 m) remain.
These are repeated observations, not independent obstacles or accuracy labels.

| Recording | Prior p50, ms | Current p50, ms |
|---|---:|---:|
| doubleT_obstacle | 543 | 469 |
| doubleT_platform | 361 | 315 |
| roundT_doubleT | 441 | 372 |

This table describes separate full-recording runs, not an isolated performance
comparison. Source decoding, diagnostic writes and machine load differ. The
controlled cached-scan benchmark below isolates the Python implementation change.

## Paired timing outcome

After full-recording processing finished, three alternating before/after runs
used the same cached 20 real scans, fresh detectors and the same C++ voxel
backend. Both variants were warmed on an actual scan before measurement.

Median of run medians: **529.68 → 460.82 ms**, **13.0% less processing time**
(1.149× throughput). Every paired comparison retained the same compared outputs
within 1e-10. The timed region excludes decoding, result serialization, display
and ROS transport. This is one development prefix, not target-hardware latency
or a general 13% guarantee across all scenes.

Public aggregate evidence: `results/runtime-context-20260917.json`. Original
per-frame timings and comparisons: `build/runtime-context-paired/report.json`.

## Reproduction and evidence

```sh
.venv-iteration/bin/python -m cProfile -o /private/tmp/tg-runtime-before.pstats \
  -m tunnel_guard.run --experiment configs/runtime-profile-before.json
.venv-iteration/bin/python -m cProfile -o /private/tmp/tg-runtime-after.pstats \
  -m tunnel_guard.run --experiment configs/runtime-profile-after.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/runtime-context-real.json
.venv-iteration/bin/python -m tunnel_guard.panel_report \
  --panel configs/runtime-context-panel.json --output build/runtime-context-real/comparison.json
.venv-iteration/bin/python -m tunnel_guard.native_benchmark --experiment configs/runtime-context-paired.json
```

The `before` recipe does not select old code. Its completed run has a Python
source snapshot under `build/runtime-profile-before/source/tunnel_guard`; use that
snapshot or the base revision for historical reproduction. A current checkout
running the old recipe is not a baseline.

The benchmark's optional `baseline_source` compares that saved package with the
current package, using the **same unchanged native voxel module** in both. The
old package is imported under a separate name; its files are not rewritten. It
retains source hashes for both variants, native binary/source hashes, input
measurement identity, all per-frame decisions and alternating run order. The
original NumPy-versus-C++ benchmark mode remains available without this option.

New output paths are required when repeating runs. Local artifacts include
profile files, logs, source snapshots, full-recording outputs and paired cached
scan measurements. No automated test suite, synthetic dataset, new dependency,
Docker build or bag extraction was used.

## Remaining path to 10 Hz

1. Move adaptive neighbourhood/graph construction out of Python object lists,
   preserving mutual-radius edges, core/border policy, deterministic component
   ordering and weak components. Compare actual decisions, not just aggregate
   runtime. This has not been implemented in this iteration.
2. Profile ground/background fitting and KISS-ICP separately after clustering
   improvement. Much of Open3D and KISS-ICP is already C++; translating Python
   orchestration alone cannot remove their computation cost.
3. Evaluate decoding/processing overlap and target hardware separately from
   detector compute time. Bounded input queues can prevent stale results but do
   not make an expensive detector faster. Dropping scans changes confirmation
   latency and cannot be counted as a cost-free speedup.
4. If model reuse is needed, reuse only past supported geometry under accepted
   motion and uncertainty checks. Never precompute from future frames or assume
   a static map solves registration degeneracy. This is a separate algorithmic
   experiment, not enabled here.

A 100 ms budget is not yet met. GPU use is not automatically beneficial for this
geometric pipeline; data transfer and available kernels must be measured. No
4× speedup or target-hardware throughput is promised from the current changes.
