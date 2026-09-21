# Exact-return deduplication: rejected for default use

Experiment based on `experiments/gerasimov` at `71b293e`, Apple M1 Pro,
10 logical CPUs, Python 3.12, KISS-ICP 1.3.0. The aim was lower frame-processing
latency without changing detector decisions. That acceptance criterion failed.
The 100 ms/frame target remains unmet.

The optional `deduplicate_returns` decoder setting defaults to false. The separate
`configs/detector-return-dedup.json` enables it for research. No thresholds,
extrinsics, deskew policy, ground model or default detector recipe were changed.

Within each ring, the decoder compares successive valid measurements in acquisition
order. It drops a later measurement only when raw point time, XYZ and intensity all
match. Stable channel grouping followed by an original-order mask retains the first
copy. Different returns are preserved. Missing identity fields, non-integer rings,
missing/non-finite/constant acquisition times leave the cloud intact. This is not
a general emission classifier: coarse timestamp quantization on another sensor can
make separate shots indistinguishable. It is deliberately experimental.

`raw_points` and `invalid_points` retain their original meanings. The new
`duplicate_return_points` counts valid slots removed. Time normalization and scan
duration are calculated before deduplication. Per-range
`returns_before_geometry_voxel` counts retained processing points, so those counts
are expected to change with this option; they are not raw return-slot counts.

## Real replay results

Each variant processed the same 240 distinct recorded frames: consecutive prefixes
of 60 frames from each of the first two bags and 30 from each remaining bag.
No automated tests were run. Prefetch and visualization were disabled; processing
excludes decode. These are single sequential exploratory runs, not paired repeated
latency measurements. Cold startup is included in the retained distributions.

| Recording | Processing p50, before → after (ms) | Read + process p50 (ms) | Maximum position disagreement (m) |
|---|---:|---:|---:|
| doubleT_obstacle | 200.0 → 174.0 | 243.8 → 236.5 | 0.0029 |
| roundT_doubleT | 164.8 → 149.4 | 180.4 → 173.1 | 5.3075 |
| doubleT_platform | 144.3 → 127.2 | 160.5 → 151.5 | 0.0576 |
| roundT_pressureGate_roundT | 141.4 → 134.1 | 157.1 → 156.3 | 0.0369 |
| roundT_squareT_pressureGate_squareT | 142.3 → 131.9 | 157.5 → 155.7 | 0.0166 |
| squareT_platform_squareT_switch | 145.2 → 135.0 | 160.1 → 165.9 | 0.5801 |

Frame status, current geometry and support-voxel counts matched across the panel.
However, track IDs, object confirmations, accumulated evidence and covariance
changed. Nearest-obstacle distance changed on two platform frames. The position
disagreement is between estimated trajectories, not error against ground truth.
Neither trajectory can be declared more accurate from this experiment.

Decode cost nearly doubled: e.g. 25.0 → 46.6 ms on `doubleT_obstacle` and
9.2 → 17.8 ms on `roundT_doubleT`. The full-loop p95 worsened on three recordings.
Fewer input points alone does not establish an end-to-end improvement.

## Why exact duplicates affect the result

[KISS-ICP 1.3.0 VoxelUtils.cpp](https://github.com/PRBonn/kiss-icp/blob/v1.3.0/cpp/kiss_icp/core/VoxelUtils.cpp)
reserves its hash table using input frame size and returns points in table iteration
order. KISS then applies another voxel downsampling stage. On the first real scan
of **all six bags**, calling the original installed KISS voxelization showed:

1. The first stage retained exactly the same point multiset, in different order.
2. The registration stage retained a different point multiset.

This is an observed mechanism, not merely floating-point noise. Changing the input
count changes downstream sampling, and the tunnel's weak motion constraints can
amplify the difference. A useful next experiment is stable ordering between KISS's
two sampling stages, with trajectory and tracking evaluation as an algorithm
change. It must not be advertised as equivalent to the existing odometry.

## Reproduction and evidence

Run from the repository with installed dependencies and the six bags available:

```sh
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m tunnel_guard.run --experiment configs/experiments/return-dedup-before.json
.venv/bin/python -m tunnel_guard.run --experiment configs/experiments/return-dedup-after.json
.venv/bin/python -m tunnel_guard.run --experiment configs/experiments/return-dedup-panel-before.json
.venv/bin/python -m tunnel_guard.run --experiment configs/experiments/return-dedup-panel-after.json
.venv/bin/python -m tunnel_guard.return_dedup_report --panel configs/experiments/return-dedup-report.json --output results/return-dedup-m1pro-20260921.json
```

Replay output directories must not already exist; use new output paths for a repeat
and update the report recipe accordingly. Existing evidence is retained in
`build/return-dedup-{before,after,panel-before,panel-after}` with source snapshots,
binary/source/config hashes, environment, logs and full frame results. The compact
report is `results/return-dedup-m1pro-20260921.json`.

After the edits, the actual default CLI was also replayed on the first three frames
of each of the two initial bags using `return-dedup-default-check.json`. All six
frames matched the baseline for status, objects, geometry, motion, pose, range
observability, health and supported range at the comparison tolerance of 1e-10;
no returns were removed. This bounded check does not establish every input layout.

ROS uses the same decoder option and publishes the removed-slot count, but the ROS
node was not exercised on this Mac. No exhaustive object labels or reference
trajectory were used; this work establishes neither improved recognition nor field
safety. The failed candidate remains opt-in so its failure can be reproduced.
