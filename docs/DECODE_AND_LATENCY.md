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

## Sensor attributes preserved through decode — 2026-09-22

Follow-on to the 2026-09-17 work above, which is unchanged. Baseline: upstream
`bc4f73b`, branch `fix/issue-5-sensor-attributes`, seed `20260915`, detector recipe
`configs/detector-native.json`. Issue #5: the decoder returned geometry and normalized
times but dropped the sensor's own per-point fields. Complete evidence:
[results/issue-5-sensor-attributes-20260922.json](../results/issue-5-sensor-attributes-20260922.json);
browser views real frames 0/1 of `doubleT_obstacle` as
[intensity](../results/issue-5-intensity.webp), [ring](../results/issue-5-ring.webp) and
[raw_time](../results/issue-5-raw-time.webp).

### Contract

- `io.decode_cloud` returns five values:
  `(points, normalized_times, invalid_count, duration_s, attributes)`. The first four are
  bit-identical, including scalar encodings (`-0.0` ≠ `0.0`).
- `io.PointAttributes`: `values: dict[str, np.ndarray]` for `intensity`, `ring` and
  `raw_time`, only when the source has the field, with its original dtype and values
  (`raw_time` is the raw acquisition field, not normalized time); `source_indices`, the
  flattened row-major slots of the XYZ-valid points; `xyz_valid_mask`, one flag per
  original slot; `time_field`. `summary()` is JSON-safe availability, counts and units
  and states `profile_confirmed: false`, `firing_identity: "unknown"`,
  `return_multiplicity: "unknown"`, `intensity_calibration: "unverified"`.
  `arrays(indices=None)` returns NPZ-ready arrays aligned to the selected decoded rows
  (`source_indices`, each field and its `<field>_valid`); it omits the full
  `xyz_valid_mask`, which the caller stores once, and never averages metadata across a
  selection.
- Invalid attribute values are preserved and reported separately. They never drop a
  point or change the XYZ mask.
- `Scan.attributes: PointAttributes | None = None` is appended at the end of the
  dataclass, so existing positional construction still works; `iter_bag` always supplies
  decoded attributes.
- `Detector.process(..., *, point_attributes=None)` checks row alignment before any
  detector state moves, adds `sensor_attributes` to the row and never lets these fields
  reach detection, support or temporal confirmation.
- Diagnostic NPZ keeps every existing array and adds `decoded_xyz_valid_mask`,
  `decoded_source_indices`, `decoded_<field>`, `decoded_<field>_valid`, plus the same
  suffixes at the exact representative rows with prefixes `range_`, `geometry_voxel_` and
  `cluster_`.
- `review_viewer` / `review.html` read these decoded attributes instead of re-parsing the
  payload.

### Measured decoder comparison

Artifact `build/issue-5-decode` (corrected after review and re-run; the earlier report is
retained in `build/issue-5-decode-before-review`), config `configs/issue-5-decode.json`.
Seven real recordings, first eight messages each, three alternating repetitions per
message; both implementations decode the identical deserialized message. Timing is
`decode_cloud` alone, excluding deserialization and comparison.

| Quantity | Retained baseline | Current source |
|---|---:|---:|
| Messages / valid point observations | 56 / 11,760,266 | 56 / 11,760,266 |
| Changed frames; max coordinate / time difference | — ; 0.0 / 0.0 | 0 ; 0.0 / 0.0 |
| Attribute failures; frames with attributes | n/a | 0 ; 56 of 56 |
| `decode_cloud` p50 / p95 (ms) | 6.577 / 17.473 | 7.872 / 20.450 |

Process peak RSS of the paired benchmark: 545,046,528 bytes. Peak retained attribute
arrays: 8,554,632 bytes. That byte count is the peak over measured frames, not a
cumulative allocation or a resident-set cost. Only the recorded layout was exercised:
one row, little-endian, 26-byte points, float32 XYZ and intensity, uint16 ring at
offset 16, unaligned float64 timestamp at 18; widths 921,600 and 307,200.

| Attribute | dtype (all frames) | min–max | invalid |
|---|---|---:|---:|
| intensity | float32 | 0–255 | 0 |
| ring | uint16 | 0–127 | 0 |
| raw_time | float64 | 946,687,297.216–946,693,351.066 s | 0 |

Per-frame summary example (`doubleT_obstacle`, frame 0): 346,808 points, every channel
numerically valid, `source_time_field` `timestamp`, profile unconfirmed, firing identity
and return multiplicity unknown, intensity calibration unverified.

### Detector replay

`configs/issue-5-before.json` (archived upstream tree) against
`configs/issue-5-after.json`, six frames from each of two recordings, diagnostic frames
`[0, 5]`, visualization disabled.

| Recording | Frames | Object observations | Discrete/structural diffs | Max numeric diff | Wall (s) | Peak RSS (MB) | Diagnostic writes (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| doubleT_obstacle | 6 | 1,814 | 0 | 2.66e-15 | 5.009 → 5.483 | 660 → 727 | 2.068 → 2.880 |
| new_data_200 | 6 | 1,784 | 0 | 4.33e-15 | 2.485 → 2.771 | 661 → 768 | 1.360 → 1.810 |

All 12 frames report `no_obstacle_observed`, geometry valid 6/6, motion valid 5/6 in both
arms. Diagnostic-write time covers two saved frames per recording and grows with the
added attribute arrays. The four saved NPZs report no changed legacy array; the decoded,
range, geometry-voxel and cluster representative rows all match their source slots in
source index, XYZ and attributes (`doubleT_obstacle` frame 0: 346,808 / 342,740 / 72,647 /
23,303; `new_data_200` frame 0: 189,478 / 189,478 / 44,327 / 27,157), and the full XYZ
mask agrees with the source indices. Consumer CLIs were exercised without change to
their output contract: `inspect_bag` on 2 frames, `sustech_import` on one frame per bag
(346,808 and 189,478 rows, XYZ and intensity matching), and `annotate` on 5 original
provisional frames with the raw attributes and time field (`timestamp`) carried into the
evidence.

A final independent repeat of both 12-frame recipes against a freshly reconstructed
upstream tree (the `git archive` snapshot with its native extension rebuilt in place, as
in Reproduction) again left zero discrete/structural differences and zero legacy
diagnostic-array differences, with a maximum numeric difference of 1.03e-14
(`1.61e-15` / `1.02e-14` per recording; outputs `build/issue-5-final-before.json` and
`build/issue-5-final-after.json`, record `final_reproduction_confirmation` in the
evidence JSON). The table above is the original paired measurement and is unchanged.

### Reproduction

These commands are a clean reconstruction of the acceptance runs, not the literal
sequence in which the first baseline attempt failed. In those runs the baseline replay's
`PYTHONPATH` pointed at the source tree the failed attempt had left in
`build/issue-5-before-native-mismatch/source`, which is retained
(`results/issue-5-sensor-attributes-20260922.json` records `failed_baseline`); the
reconstruction below recreates an equivalent upstream tree at a clean path.

All seven configured recordings must be present — a missing bag is an error, not a skip.
The three `for_hackathon` bags outside the original three-recording subset are not part of
the checkout and come from the organizer archive; `data/extended_subset/new_data_200` is
required as well.

```sh
mkdir -p data/sourcecraft_subset
tar --zstd -xf archive/for_hackathon.zst -C data/sourcecraft_subset \
  for_hackathon/roundT_pressureGate_roundT \
  for_hackathon/roundT_squareT_pressureGate_squareT \
  for_hackathon/squareT_platform_squareT_switch

# upstream source snapshot plus its native extension. The ignored _native binary inherited
# from the experiments/morev checkout has the wrong kernel ABI (classify_geometry takes 17
# arguments there, 19 here), so the first baseline attempt failed and is kept in
# build/issue-5-before-native-mismatch. Never rebuild this branch's source under the
# snapshot; the archived pyproject.toml is already the correct one, so nothing is copied
# into it.
mkdir -p build/issue-5-upstream-source
git archive bc4f73b tunnel_guard cpp setup.py MANIFEST.in pyproject.toml \
  | tar -x -C build/issue-5-upstream-source
VENV="$PWD/.venv-iteration/bin/python"
(cd build/issue-5-upstream-source && "$VENV" setup.py build_ext --inplace)

# baseline decoder source that configs/issue-5-decode.json names
mkdir -p build/issue-5-baseline-source
git show bc4f73b:tunnel_guard/io.py > build/issue-5-baseline-source/io.py

# this branch's own native extension, before any run on this source: the snapshot build
# above is a separate tree, and the ignored _native binary inherited from the
# experiments/morev checkout is still the wrong kernel ABI here
.venv-iteration/bin/python setup.py build_ext --inplace --force

# baseline replay against the archived upstream tree; -P keeps this worktree's package from
# shadowing it
env PYTHONPATH="$PWD/build/issue-5-upstream-source" .venv-iteration/bin/python -P \
  -m tunnel_guard.run --experiment configs/issue-5-before.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/issue-5-after.json
.venv-iteration/bin/python -m tunnel_guard.decode_benchmark --experiment configs/issue-5-decode.json

# browser on a fixed run and its source bag (default port 8765)
.venv-iteration/bin/python -m tunnel_guard.review_viewer \
  --run build/issue-5-after \
  --bag data/sourcecraft_subset/for_hackathon/doubleT_obstacle
```

Output directories must be new for another run. The colour selector offers neutral /
intensity / ring / raw_time; each channel is auto-scaled min→max over the displayed frame
(blue→red), unusable samples are magenta, an absent field is reported as absent rather
than as zero, and a real zero stays inside the scale. These colours are not calibrated
reflectivity or material.

### Limitations

- Numerical validity is not physical validity: no material labels or calibration.
  Firing identity and return multiplicity are unknown; the existing spatial voxel support
  is not a count of independent firings, and temporal hits are unchanged.
- The detector panel is 12 development frames, all `no_obstacle_observed` with adjacent
  objects — no hazard recall or safety validation.
- The missing/invalid attribute display was not exercised against a real recording that
  has those conditions; only real, fully finite channels were seen.
- One-row little-endian recordings only; no synthetic layouts are part of acceptance.
- No ROS2 runtime or target Intel CPU measurement; deskew remains disabled and
  timing/mounting unverified.
- The run is short: the decode and replay numbers describe this panel and are not a
  speedup claim.
- Initial workers ran out-of-contract synthetic/ad-hoc probes; these are excluded from
  acceptance, and no test files or framework were introduced.

### Full-recording regression audit — 2026-09-22

**Strict unchanged-output acceptance: FAILED, not waived.** The initial 12-frame
prefix above did not exercise alarms or invalid geometry. The follow-up compares upstream
`bc4f73b4f75d83540683716953afb30351839cd6` with submitted fix
`555250285d0b12ea2252fc14d5abf0a1a131e3f2` across every frame of all six original
recordings and the three available extended segments. Seed `20260915`,
`configs/detector-native.json`, eight odometry threads, deskew disabled, no threshold
changes. Every SQLite source message is accounted for: 2,641 messages/frames per arm,
zero duplicate or subsample skips, missing frames, extra frames or progression gaps.

The [audit summary](../results/issue-5-regression-summary-20260922.json) links the
[reviewed comparison](../results/issue-5-full-regression-reviewed-20260922.json),
[complete mismatch sidecar](../results/issue-5-full-regression-reviewed-20260922-mismatches.jsonl),
[decoder evidence](../results/issue-5-full-decode-20260922.json) and compressed
[per-message hashes and measurements](../results/issue-5-full-decode-20260922-frames.jsonl.gz).
The original failed comparison and failed reporting-tool attempt are retained.

| Recording | Frames | Fix-versus-upstream counter differences | Upstream-versus-itself counter differences |
|---|---:|---:|---:|
| doubleT_obstacle | 201 | 0 | 0 |
| doubleT_platform | 345 | 7 | 6 |
| roundT_doubleT | 252 | 9 | 3 |
| roundT_pressureGate_roundT | 268 | 2 | 2 |
| roundT_squareT_pressureGate_squareT | 545 | 5 | 2 |
| squareT_platform_squareT_switch | 877 | 1 | 1 |
| new_data_0 | 51 | 0 | 0 |
| new_data_100 | 51 | 0 | 0 |
| new_data_200 | 51 | 3 | 3 |
| **Total** | **2,641** | **27** | **17** |

Every material difference is an exact integer change of one in
`accumulated_support_voxels`. IDs, object ordering, boxes, relations, confirmations, hits,
source/evidence timestamps and alarm states are unchanged. Other unequal continuous
estimates differ by at most `8.730793865652231e-13`; their predeclared tolerance is
`atol=rtol=1e-9`. Tolerance is never applied to counts or timestamps. Operational timing
fields and the known `sensor_attributes` addition are the only exclusions.

Coverage is 698,799 **object-frame observations**, not independent objects/events:
202 `obstacle`, 58 `unresolved_obstacle`, 18 `candidate`, 3 `unknown`, and 2,360
`no_obstacle_observed` frames. Geometry is valid in 2,638 frames and motion in 2,426.
There are zero duplicate or future evidence timestamps in either evidence history.
These are unlabelled output states, not recall, precision or field-safety measurements.

**Numerical control and remaining risk.** The
[upstream self-comparison](../results/issue-5-upstream-control-20260922.json) uses
identical captured Python source and the **same already-built native binary**. It still
has 17 one-voxel differences and continuous differences up to `1.2327916465437738e-12`.
Fourteen of the original 27 changes recur exactly; 13 do not recur in this one control.
The [per-case investigation](../results/issue-5-counter-investigation-20260922.json)
retains all three values rather than attributing every difference to the patch or to noise.

The counter counts distinct `floor(track_local_evidence / 0.05)` keys after pose
transformation (`detector.py` association; `accelerator.voxel_counts`). Arbitrarily small
coordinate changes can cross a voxel boundary. Roundoff-level pose variation and this
discontinuous count are consistent with the observed signature; the exact registration/
reduction-order cause is not established. All 27 affected objects are adjacent/pending,
with 0 or 1 uncertain-support voxels, below the configured 2-voxel confirmation gate.
Their confirmation is therefore unaffected. **The accumulated count is used in
confirmation generally**: this observation does not make other boundary cases safe.
No point snapping, tolerance relaxation, counter exclusion or geometry change was made.

**Decode and original-slot provenance.** All 501,204,616 decoded XYZ-valid return
observations preserve XYZ, normalized time, invalid count and duration byte-for-byte:
zero changed messages and zero raw-field/slot/mask failures. Intensity, ring and raw time
remain metadata; sensor profile, firing identity, return multiplicity and intensity
calibration remain unverified. The full replay captures 81 diagnostic frames with
1,053 byte-identical legacy arrays.

The three actual invalid-geometry frames were outside that fixed capture selection:
`roundT_pressureGate_roundT` frames 130/249 and `new_data_200` frame 48, all
`insufficient_paired_rail_support`. Supplementary complete replays of those two recordings
capture all three failures plus one valid frame. All 28 additional legacy arrays are
byte-identical, and fresh raw-message decoding rebinds all four decoded archives exactly.
Across both panels: **85 archives, 1,081 legacy arrays unchanged, 82 executed cluster
stages verified and 3 correctly reported as not executed**. Empty cluster metadata is not
counted as observed cluster support. The supplementary 319 messages repeat the original
panel; they are not extra independent data.

**Observed costs, not a deployment benchmark.** One alternating decoder pair per raw
message gives p50 **6.581 → 7.927 ms**, p95 **16.910 → 20.040 ms**; the after arm includes
attribute extraction. Peak retained attribute arrays occupy **8,555,468 bytes**. These
are pooled local measurements, excluding deserialization/comparison, not guaranteed
streaming latency. Whole-replay wall/RSS/stage metrics remain in the reports. The initial
baseline overlapped short worker NPZ inspections and one cancelled inspection of unknown
duration, so its timing must not be used as a causal performance comparison. RSS is a
cumulative process peak across recordings, not an incremental per-recording cost.

**Provenance and reporting.** The baseline source archive is nested under the fix
checkout, so its runner reports the enclosing fix Git revision. The
[source bindings](../results/issue-5-regression-source-20260922.json) instead verify
captured production Python/C++ hashes against the actual Git objects. Reviewed reports
also record their comparer/helper source hashes and recipe hash. `compare_alarm_runs`
now emits schema version 2, streams full rows, retains every material mismatch, rejects
incomplete inputs and never overwrites previous evidence. Its historical CLI flags remain;
the older summary schema is intentionally replaced. Strict recipe mode exits 1 for this
failed criterion. The early report accumulator error was fixed and its attempt retained
at `build/issue-5-report-attempt-1`; it was not a detector failure.

Reproduction uses the upstream archive/native-build preparation above and **new output
directories**. All nine bags are required. Run the following from the repository root,
sequentially; expected failed comparisons must not stop collection of the remaining evidence:

```sh
PYTHONPATH="$PWD/build/issue-5-upstream-source" \
  .venv-iteration/bin/python -P -m tunnel_guard.run \
  --experiment configs/issue-5-regression-before-20260922.json
.venv-iteration/bin/python -m tunnel_guard.run \
  --experiment configs/issue-5-regression-after-20260922.json
.venv-iteration/bin/python -m tunnel_guard.decode_benchmark \
  --experiment configs/issue-5-regression-decode-20260922.json
.venv-iteration/bin/python -m tunnel_guard.compare_alarm_runs \
  --experiment configs/issue-5-regression-report-reviewed-20260922.json

PYTHONPATH="$PWD/build/issue-5-upstream-source" \
  .venv-iteration/bin/python -P -m tunnel_guard.run \
  --experiment configs/issue-5-regression-upstream-repeat-20260922.json
.venv-iteration/bin/python -m tunnel_guard.compare_alarm_runs \
  --before build/issue-5-regression-before-20260922 \
  --after build/issue-5-regression-upstream-repeat-20260922 \
  --output results/issue-5-upstream-control-20260922.json

PYTHONPATH="$PWD/build/issue-5-upstream-source" \
  .venv-iteration/bin/python -P -m tunnel_guard.run \
  --experiment configs/issue-5-invalid-geometry-before-20260922.json
.venv-iteration/bin/python -m tunnel_guard.run \
  --experiment configs/issue-5-invalid-geometry-after-20260922.json
.venv-iteration/bin/python -m tunnel_guard.decode_benchmark \
  --experiment configs/issue-5-invalid-geometry-decode-20260922.json
.venv-iteration/bin/python -m tunnel_guard.compare_alarm_runs \
  --before build/issue-5-invalid-geometry-before-20260922 \
  --after build/issue-5-invalid-geometry-after-20260922 \
  --output results/issue-5-invalid-geometry-comparison-20260922.json
```

For a new execution, regenerate source bindings from its captured source rather than
reusing the historical binding's reporter-file hashes. The committed summary and source
inventory describe this execution; they are evidence, not silently regenerated fixtures.
No automated tests, generated clouds, ROS2 runtime validation or new calibration claim
are part of this audit.

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
