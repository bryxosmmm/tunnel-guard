# Decode preservation and complete offline latency — 2026-09-17

Baseline: `839d9fa`. Fixed native detector recipe and seed 20260915. No changes to
geometry, thresholds, sparsity policy, tracking or confirmation rules.

## Implementation

`io.decode_cloud` now copies strided XYZ fields directly into a float64 array,
avoiding an intermediate column stack. It avoids the Nx3 finite-mask reduction,
unnecessary all-valid compaction and redundant time-range computation. Exact
signed axis permutations use column operations; arbitrary rotations retain the
matrix product. Measurement order, multiplicity and invalid-return policy remain
unchanged. Structured dtype offsets and row strides still handle padding/endian
layouts; only the actual layouts below were exercised, not generated cases.

`iter_bag` records deserialization and decode duration. `run` writes separate
`<bag>-timing.jsonl` records after result serialization/buffered writes, plus
summary quantiles. Timing records contain ingestion, decode, deserialization,
inference, optional diagnostic/RViz-bag output, result writing and the complete
offline iteration. Nested timings overlap and must not be added twice.

The offline iteration excludes timing-log writes, fsync, live DDS queues,
transport and GUI rendering. The ROS adapter separately logs decode time,
callback-to-result, publication duration and callback-to-publication-return.
Sensor-to-result age stays null because the clock relationship is unverified.
The ROS changes were inspected but **not run on ROS2 or target hardware**.

## Real-data evidence

Artifact: `results/decode-latency-20260917.json`.

- First ran decoder evaluation on three frames from each of three real bags and
  the actual detector on a three-frame prefix.
- Paired decoder benchmark: all **798 frames**, three alternating repetitions per
  message, warmup on the first message of each recording. Both implementations
  receive the identical deserialized real message. Timing excludes deserialization
  and comparison work; source snapshots/hashes and individual timings retained.
- **174,539,002 valid point observations** match exactly in values and order.
  Normalized point times, invalid counts and scan durations also match exactly;
  maximum coordinate/time difference zero. No missing distant support inferred
  from aggregate timing alone: the actual arrays were compared.
- Decoder p50 **11.704 → 7.603 ms** (35.0% reduction); p95 **29.678 → 20.796 ms**.
  These are cached-message measurements, not the cost of loading each real frame.
- Replayed all 798 frames through the native detector. All fields compared by
  panel_report match `build/integrated-native-real` within 1e-10; no missing/extra
  frames or measurement identity errors. All 44,291 ≥60 m candidate observations
  retained; no future/duplicate timestamps in saved evidence histories.
- Recorded layout: single row, little endian, 26-byte points; float32 XYZ,
  float32 intensity, uint16 ring, unaligned float64 timestamp at offset 18.
  Two observed widths. Other layouts and arbitrary calibrated rotations were
  not exercised by this real-data panel.

### Measured complete offline loop

| Recording | Frames | Detector p50 | Offline iteration p50 | Offline iteration p95 |
|---|---:|---:|---:|---:|
| doubleT_obstacle | 201 | 153 ms | 201 ms | 277 ms |
| doubleT_platform | 345 | 124 ms | 140 ms | 224 ms |
| roundT_doubleT | 252 | 132 ms | 150 ms | 221 ms |

Visualization and diagnostic capture disabled for this run. Includes first-frame
startup. Descriptive timings on the development Mac, not a paired end-to-end
speedup measurement. Detector-only timing must not be presented as live latency.
The complete pipeline does not yet sustain 10 Hz on this measurement.

## Executed commands

```sh
mkdir -p build/decode-baseline
git archive 839d9fa tunnel_guard/io.py | tar -x -C build/decode-baseline
.venv-iteration/bin/python -m tunnel_guard.decode_benchmark --experiment configs/decode-paired-prefix.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/decode-prefix.json
.venv-iteration/bin/python -m tunnel_guard.decode_benchmark --experiment configs/decode-paired.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/decode-real.json
.venv-iteration/bin/python -m tunnel_guard.panel_report --panel configs/decode-panel.json --output build/decode-comparison.json
git diff --check
```

Output directories must be new for another run. Recipes/source snapshots/logs are
retained locally. No automated tests/suites, synthetic inputs, container builds,
dataset extraction or additional packages were used.

## Pause boundary and unresolved fundamentals

This is a reasonable frozen research baseline pending the extended dataset.
There is still useful engineering work possible, but further tuning to these
few development recordings is not evidence of generalization.

1. **Vehicle reference is not established.** Rails and bed constrain a local
   track frame under visibility/geometry assumptions. They do not identify the
   vehicle nose relative to the sensor, actual body clearance, wheel recesses or
   suspension envelope. Existing provisional calibration candidates failed their
   declared stability gates and were not installed. More unlabeled clouds alone
   do not necessarily resolve these quantities.
2. **Alarm quality is not established.** Small boxes enclose observed support,
   not a proven physical object volume. Repetition confirms measurements, not
   semantic identity or collision danger. The nonexhaustive/detector-propagated
   annotations cannot establish nuisance rates or independent obstacle recall.
3. **Motion and timing provenance remain uncertain.** Driver transforms,
   per-point timestamp meaning and prior deskew/aggregation need confirmation.
   No independently verified trajectory or live sensor-to-result age is available.
4. **Real-time deployment remains unverified.** Stable 10 Hz, Linux/ROS behavior,
   queue loss and target-CPU latency are not established by these Mac replays.

When new data arrive: inventory layout/clock/frame changes first; split independent
runs/installations before tuning; evaluate the frozen baseline; then obtain
independent event labels and prioritize errors by event/range/visibility. Keep
unobserved intervals unknown. Revisit calibration only with observable references;
do not turn a generic sensor manual or more repeated frames into missing ground
truth. No ML training or new synthetic campaign is needed before that audit.

## Attribute retention after #11 and #13 — 2026-09-23

The compact implementation `e48304c` is compared with `f2dd7c4`: the accepted presence
separation plus the numerical support fix and serial ICP. No detector threshold, transform,
deskew setting or alarm policy differs between the variants.

The unchanged nine-recording panel contains **2641 frames**: six complete recordings and
three extended-data segments, not nine independent field trials. Seed: **20260915**.
Both frozen variants ran to EOF, in separate sequential processes, with the same rebuilt
native binary. The input inventory and source bindings are retained with the audit.

- **Zero differences** in every prior non-timing result field across **698799 component-frame
  observations**, with `atol=rtol=0`. This includes integer support, timestamps, poses, geometry,
  IDs, presence, confirmation, intersection and frame decisions. Only operational timings and
  the newly added `sensor_attributes` summary are outside equality; the latter is independently
  validated.
- The first four decoder outputs are **byte-identical** over **501204616 XYZ-valid return
  observations**. Raw optional fields, original-slot indices and validity masks were checked
  against the real PointCloud2 messages. These are not independent physical emissions.
- The full replay saved **81 diagnostic archives**: all **113589 prior arrays** are byte-identical,
  and all 81 representative selections retain exact source-attribute alignment.
- State coverage is unchanged: 202 obstacle, 58 unresolved, 18 candidate, 3 unknown and
  2360 no-obstacle-observed frames. This is regression coverage, not accuracy or route clearance.
- A supplementary replay of the same 319 frames from two recordings captured four additional
  archives, including all three invalid-geometry frames. Combined: **85 archives, 114916 prior
  arrays byte-identical, 82 executed representative stages and 3 correctly marked not executed**.
  It also had zero result/attribute mismatches. These repeats do not enlarge the independent panel.

### Measured cost

| Measurement | Before | With attributes |
|---|---:|---:|
| Paired decode p50, ms | 18.405 | 21.756 |
| Paired decode p95, ms | 37.032 | 43.075 |
| Full replay process peak RSS, MiB | 741.203 | 856.422 |
| Sum of recording wall times, s | 2224.092 | 2149.588 |
| Diagnostic-write time within that replay, s | 164.411 | 205.911 |

Decoder timing alternates call order on each of 2641 real messages and excludes deserialization
and comparison work. The observed p50/p95 increases are **18.21% / 16.32%**. Peak retained
attribute arrays are **8555468 bytes**; that is neither total allocation nor process RSS.
The replay RSS includes prefetch, tracking, output retention and diagnostic payloads.
The wall-time difference is **not a speedup claim**: one sequential pair on a shared Mac does
not isolate thermal, scheduling or cache effects and is not an i7-9700E deployment benchmark.

The first smoke attempt failed because the cloned worktree carried an older native ABI
(`classify_geometry`: 17 arguments versus the source's 19). Its failure record and binary were
preserved; the unchanged C++ sources were rebuilt for both variants before the successful
44-frame-per-variant smoke and full replay.

**Enablement tradeoff:** retaining fields fixes data loss and enables traceable inspection;
it has measured decode/memory cost and no measured detection benefit. This evidence supports
reviewing the compact metadata-preservation change, not enabling intensity decisions, deskew,
return deduplication or train-control authority. Sensor provenance and issue #5 remain open.

The compact [summary](../results/attributes-summary-20260923.json) records source identities,
costs and artifact references. Replay recipes are
`configs/attributes-before-20260923.json` and `configs/attributes-after-20260923.json`.
The large audit implementation and detailed evidence are kept outside the functional PR.
