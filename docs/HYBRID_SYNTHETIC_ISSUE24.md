# HYBRID SYNTHETIC: measured empty-tunnel backgrounds, issue #24

The frozen recipe is `configs/hybrid-synthetic-issue24.json`. Run it from the repository root with the project Python environment:

```sh
.venv-tunnel-guard/bin/python -m tunnel_guard.realistic_stress --experiment configs/hybrid-synthetic-issue24.json
```

The checkout had `tunnel_guard/stress.py` for wholly ray-cast tunnels but no `realistic_stress.py` or `realistic_report.py`. The hybrid entrypoint reuses its box-ray intersection and the existing bag decoder and detector; it adds the measured-background insertion and paired reporting that the synthetic-only runner cannot express.

For a repeat, add `--output build/hybrid-synthetic-issue24-repeat-01`; the output directory must be new. The run writes an effective recipe, detector config, source snapshot, source file SHA-256 identities, frame timestamps, case provenance, paired background and insertion predictions, a CSV matrix, and a report. The compact reviewed result is in `results/hybrid-synthetic-issue24/`. The raw bags are intentionally not copied into Git. No automated test suite is involved.

## Frozen source split and question

Issue #21 records the organizer statement that the supplied corpus has one person episode and otherwise empty tunnels. Its versioned real panel and independent frame review remain open. This experiment uses the first three continuous frames of `doubleT_platform` for the tuning partition and the first three of `roundT_doubleT` for synthetic evaluation. Both are development sources already inspected in earlier project work. The person bag and its lone object are excluded. Shape identities and seeds are disjoint between partitions. No generator parameter was selected against evaluation detector responses. The initial locally fitted track bed and rail height set box bottoms; the extrapolation to 100/200 m is a modeling assumption.

The predeclared hypothesis and promotion criterion are in the JSON recipe. It asks whether measured return directions give enough support for opaque box obstacles while adjacent positions and unchanged backgrounds constrain nuisance alarms. The criterion requires 80% object detection across *all* evaluation cases, 70% confirmed hazard for nominal inside cases, zero confirmed hazard for adjacent cases, and zero alarm frames on the evaluation background. A zero-return case remains in the denominator.

## How the hybrid cloud is made

Each original XYZ return supplies a measured direction from the configured sensor origin. We ray-intersect a world-fixed or prescribed moving, axis-aligned opaque box, using baseline detector odometry to transform directions to world coordinates. Per 0.1-degree angular cell, a nearer real return vetoes the box. Otherwise a single idealized first return is inserted and measured background returns behind the box are removed. The source indices of every removed return and of each inserted return's source ray are saved alongside inserted XYZ, full world box bounds and the observed-support bounds. A cell with no measured return supplies no ray and cannot gain support merely by moving the box farther away. The same three original frames go through the unmodified detector as a paired background run.

The PointCloud2 decoder retains XYZ and normalized acquisition time; insertion carries the chosen source return's time. There is no calibrated ring, intensity, reflectivity, weather, multipath, material attenuation, or range-noise model. Multiple returns are not separable into verified emission slots from this XYZ input. The 0.1-degree angular cell is an approximate visibility model, **not** a calibrated Pandar128 firing or detection model. The first baseline odometry frame is unvalidated; later registration validity is recorded in the manifest. The nominal inside/edge/adjacent offset is not proof of corridor occupancy, especially beyond the measured rail-support horizon. This is a box-surface experiment, not a transplanted human cloud or an amodal object reconstruction.

## Reviewed run

The actual detector ran for three frames on each empty background and for every configured insertion case. Baseline poses were valid on frames 1–2 and unvalidated on frame 0 in both bags. Thirty-six evaluation cases cover two shapes, 30/100/200 m, inside/edge/adjacent, and stationary/lateral-moving world trajectories. The matrix carries per-frame support, occlusion, removed returns, detections, confirmation, path relation and distance. The frozen 300 m scenario was *excluded before running*: the detector's input maximum is 220 m.

| Distance | Cases | Object response attributed to insertion | Confirmed intersecting hazard | Zero visible support |
|---|---:|---:|---:|---:|
| 30 m | 12 | 8 | 8 | 0 |
| 100 m | 12 | 6 | 0 | 4 |
| 200 m | 12 | 0 | 0 | 12 |

Across all 36 cases, attributed object response was 14/36 and confirmed intersecting hazard was 8/36. Conditional on at least one visible inserted return, these were 14/20 and 8/20. The inside hazard figure was 4/12; adjacent confirmed hazard was 0/12 under the conservative paired attribution rule. The unchanged evaluation background already produced `obstacle` in 2/3 frames; the tuning background alarmed in 3/3. These are tiny 0.2-second clips, so neither alarms/minute nor independent-event false-alarm rates follow. Most detections took one 0.1-second frame after first support; case-level latency is in `matrix.csv`.

The promotion criterion **failed**. At 100 m support is sparse (often 1–18 returns per frame), and at 200 m it is absent in this source. This does not establish hardware range. At 30 m some adjacent boxes are close to measured wall/installation returns; the paired baseline has spatially overlapping detector objects, so the result is conservatively marked unattributable rather than credited. The baseline alarms and range/path uncertainty also prevent a clear-route or field-recall claim. The corpus has no independent object diversity and no exhaustive real annotation for precision.

I visually inspected `scene-review.png`: an inside 30 m case, a wall-adjacent 30 m case, and a zero-support 200 m case. The first has front-surface synthetic returns and real returns removed behind it; the adjacent case blends with nearby wall support; the 200 m crop contains no measured returns near the target. This is a representative scene review, not a sensor-validation study.

An artifact audit inspected all 37 scenes / 111 frames. Every inserted range preceded its source return, and no retained source return remained behind an inserted return in the same angular cell. Counts and approximately 0.1-second frame spacing are in `physical-audit.json`. This checks this renderer's emitted clouds; it cannot validate the assumed firing pattern, extrinsics, or target material.

All scores and artifacts are labeled **HYBRID SYNTHETIC**. They do not prove field recall, private-set recall, a calibrated hazard probability, or 100–300 m hardware detection range.
