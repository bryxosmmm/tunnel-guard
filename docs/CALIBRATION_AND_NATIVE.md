# Track-relative calibration experiment and native acceleration — 2026-09-17

The backend comparison below is historical evidence from 2026-09-17. The current detector has one required C++ spatial path. Build it with `python setup.py build_ext --inplace`; a missing extension stops recipe loading with that command in the error. The backend selector and reference detector recipe have been removed.

## What ships

- `tunnel_guard.calibrate`: chronological fitting and later-frame consistency evaluation of a **provisional track-relative orientation**. Raw measurements, per-frame acceptance reasons, frozen rotation and provenance are retained. No mounting transform is automatically installed.
- `cpp/voxel.cpp`: required C++17 first-point voxel selection, integrated into the detector's geometry, clustering and temporal-support paths. It retains the first measured return and lexicographic voxel order; it does not average away sparse returns.
- `configs/detector-native.json` is the sole shipped detector recipe. The paired benchmark figures below record the earlier two-backend state.

This iteration uses real organizer data. No automated tests, synthetic scenes, new Docker images or dataset extraction were used. The native extension was built locally with Clang 21 on Apple M3/macOS, Python 3.12.8; Linux/ROS native compilation and target throughput were **not** verified here.

## Build and use

From the repository root, using the existing environment:

```sh
.venv-iteration/bin/python setup.py build_ext --inplace
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/native-full-real.json
.venv-iteration/bin/python -m tunnel_guard.calibrate --experiment configs/calibration-track-real.json
```

The extension uses only the CPython buffer API and the standard C++ library; no new runtime package is added. A compiler and Python development headers are needed for a native build. Package installation fails if the required extension cannot be built. GCC/Clang flags are configured; Windows/MSVC is not supported by this build recipe. The Docker build context includes native sources.

**These output directories already contain evidence on the development machine.** For another run, copy the experiment JSON and choose a new `output`. Do not overwrite or delete the original run to repeat a measurement. `configs/native-full-real.json` requires the three currently extracted bags, not all six supplied bags.

For a new experiment, select `configs/detector-native.json` and a new `output` directory. Copy that detector recipe only for intentional detector settings. GUI, JSONL and ROS result formats are unchanged.

## Calibration: what is actually measured

The initial raw-to-processing axis mapping must already be roughly correct. This procedure cannot recover an arbitrary sensor orientation from an unknown scene.

1. Decode unique acquisitions in chronological order using the existing PointCloud2 decoder.
2. Use the first 20 scans for fitting. Estimate paired rails in the configured 5–20 m interval and local track-bed height; require at least three rail anchors spanning 10 m and at most 8 cm line residual.
3. Construct forward from the fitted local centerline and longitudinal bed slope. Project the estimated bed normal perpendicular to forward to obtain an orthonormal frame.
4. Freeze the mean of supported rotations at the end of the fitting prefix.
5. On the next 20 scans, measure consistency both before and after applying that frozen rotation to actual points. No later scan changes the fit. Record exact acquisition timestamps and reject non-later validation measurements.

Declared gates, fixed before the first run: at least 80% supported scans in each partition, maximum angular deviation 1°, height span 0.10 m, center span 0.15 m. These are engineering rejection criteria, **not a validated accuracy specification**. Mean rotations and observed spans are not confidence intervals; neighbouring scans are correlated.

The candidate file contains `raw_to_aligned_rotation` and `raw_to_aligned_translation` under `kind=provisional_track_relative_orientation`, with `vehicle_extrinsics_verified=false`. Translation is merely the original processing origin expressed in the rotated axes. It does **not** place the origin on railheads, the track center or the vehicle front. Even rejected candidates are retained for inspection and carry `numerical_stability_accepted=false`. They are not detector configuration files and must not be promoted automatically.

### Actual outcome: no profile accepted

Each row uses 20 fit + 20 later scans; 120 original scans total. This evaluates selected prefixes, not the full recordings.

| Recording | Fit supported | Later supported before rotation | Maximum later deviation before / after | Decision |
|---|---:|---:|---:|---|
| doubleT_obstacle | 15/20 | 14/20 | 0.247° / 0.354° | Reject: support fraction below 80% |
| doubleT_platform | 20/20 | 20/20 | 1.599° / 1.251° | Reject: angular variation above 1° |
| roundT_doubleT | 20/20 | 20/20 | 2.322° / 2.036° | Reject: angular variation above 1° |

For the obstacle prefix, corrected geometry was supported in 20/20 later scans, but this does not erase the original fit/support rejection. Thresholds were not relaxed. The failures show that this estimator and these prefixes do not justify a fixed calibration profile; they do not establish that the hardware moved or that the entire dataset is unusable.

Evidence: `build/calibration-track-real/<bag>/{frames.jsonl,orientation-candidate.json,summary.json}`. The summary, configuration and source snapshot remain available. The three candidates are deliberately not installed into production.

### What is needed for a genuine train installation profile

A static installation profile and continuously estimated track geometry are different quantities. For each installation:

- Measure the rigid sensor pose relative to an identified vehicle frame, including the longitudinal offset to the vehicle front. Surveyed targets or mounting measurements can supply this reference; unknown tunnel surfaces cannot establish it.
- Record a stationary calibration sequence with a known track/vehicle relationship, clearly visible paired rails and an independently defined reference plane. A track bed is not necessarily the plane of the railheads, especially with cant, flush rails, drainage or local irregularity.
- Retain measurement uncertainty and check the transform on a separate sequence. Recalibrate after moving the mount; do not silently refit the installation to each curve or grade.
- Time offset and per-return timing require a separate timing reference and suitable motion. This orientation experiment does not calibrate clocks or enable deskew.

As a useful observability reference, [Autoware's ground-plane calibration](https://docs.autoware.org/main/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/calibrating-sensors/ground-lidar-calibration/) estimates z, roll and pitch under its ground assumptions; x, y and yaw need additional information. The rail procedure here is our own experiment, not a reproduction of Autoware and not a substitute for a vehicle reference.

## Native implementation and evidence

The profile of a fresh 10-frame actual-detector prefix attributed 1.356 s to 2,185 voxel-representative calls. With C++, these calls took 0.316 s, about 4.3× faster in that profiled run. End-to-end gain is smaller because KISS-ICP, ground/background estimation and clustering still consume time. KISS-ICP and Open3D already execute substantial work natively.

A full run processed all 798 scans of the three available recordings. Against the saved NumPy/serial-RANSAC reference in `build/goal-final-real`:

- No changed compared status, object/ID/support, distance, geometry, motion, pose, range-observability, health or gap-reset fields at the established absolute/relative tolerance of 1e-10.
- No missing/extra frames or acquisition-identity mismatches. Maximum compared floating difference: 3.99e-13.
- All 44,246 candidate observations at ≥60 m retained. These are repeated observations, not independent objects or proven hazards.
- No future or duplicate evidence timestamps in the inspected output histories.
- All 798 frames still report degraded quality; acceleration does not validate sensor calibration or improve detection accuracy.

`build/native-full-real/comparison.json` contains the complete comparison. Per-recording processing medians were 517, 324 and 407 ms. Those full-run timings are descriptive: workloads and visualization settings differed from the historical run. Use the paired benchmark below for the speed comparison.

The native module uses the [CPython buffer API](https://docs.python.org/3/c-api/buffer.html), accepts contiguous native float64 triples, releases the GIL during computation and rejects nonfinite/out-of-range voxel coordinates. It uses no fast-math flags. Source and binary hashes are included in native run manifests; the build log and source snapshots are retained. All current detector recipes use the C++ kernels.

### Paired end-to-end timing

After the full run and calibration finished, `native_benchmark` cached the same first 20 real obstacle scans, warmed each backend on one actual scan, then ran fresh detectors in alternating order. No decoding, display, ROS transport or result serialization is included in the timed region.

| Repetition / order | NumPy p50 / p95, ms | C++ p50 / p95, ms |
|---|---:|---:|
| 0: NumPy → C++ | 590 / 608 | 504 / 528 |
| 1: C++ → NumPy | 602 / 632 | 500 / 505 |
| 2: NumPy → C++ | 603 / 628 | 504 / 510 |

Median of run medians: **602.46 → 503.76 ms**, 16.4% less processing time, 1.196× throughput. All three paired output comparisons retained the same decisions and compared fields. This is one cached development prefix, not independent scene generalization or target-hardware latency. It does not meet a 100 ms / 10 Hz budget. Per-frame outputs, timings, hashes and comparisons are in `build/native-paired-real`.

## Next priorities

1. Obtain an independent vehicle/mount reference and a suitable stationary calibration sequence; investigate unstable bed/rail estimates without relaxing acceptance gates after seeing these failures. No detector threshold tuning can supply missing vehicle pose or timing ground truth.
2. Profile the remaining density clustering, track association and geometry/background costs on target Linux hardware. Port the next measured bottleneck only if the output-equivalence and runtime benefit justify it. The whole pipeline does not need a rewrite to gain native speed.
3. Keep independent annotation and hazard/corridor evaluation separate from throughput and geometric consistency. This iteration does not demonstrate better obstacle recall or fewer nuisance alarms.

## Executed commands

```sh
.venv-iteration/bin/python -m cProfile -o /private/tmp/tg-native-before.pstats \
  -m tunnel_guard.run --experiment configs/native-before-prefix.json
.venv-iteration/bin/python setup.py build_ext --inplace
.venv-iteration/bin/python -m cProfile -o /private/tmp/tg-native-after.pstats \
  -m tunnel_guard.run --experiment configs/native-after-prefix.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/native-full-real.json
.venv-iteration/bin/python -m tunnel_guard.calibrate --experiment configs/calibration-track-real.json
.venv-iteration/bin/python -m tunnel_guard.panel_report \
  --panel configs/native-comparison-panel.json --output build/native-full-real/comparison.json
```

Run logs are copied into the corresponding `build/` directories. Prefix cProfile files and the full comparison are retained. Public aggregate evidence is in `results/calibration-native-20260917.json`; raw scans, binaries and large per-frame outputs remain untracked.
