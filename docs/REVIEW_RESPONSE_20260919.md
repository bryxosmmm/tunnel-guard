# Response to the corridor and performance review, 2026-09-19

The review answered here is `origin/experiments/morev` @ `8eb332c`,
`docs/REVIEW_GERASIMOV_20260919.md`, written against this branch at `5587685`. Every finding was checked
against the code first; where a finding was static reasoning about conditions, the effect was then measured
on the recordings. The review's three P1 findings are all correct, and all three are fixed.

The reviewer's headline caution is accepted and applied: *the failure of several estimators does not prove
the mathematical impossibility of other methods.* The far-field section of
[CURVED_CORRIDOR_AND_RANGE.md](CURVED_CORRIDOR_AND_RANGE.md) is restated accordingly.

## P1 — disabling prefetch produced an empty run: confirmed, fixed

`tunnel_guard/io.py`, `prefetch`. The function contains `yield`, so `return iter(iterable)` ended the
generator without yielding a scan. The reviewer's run reproduced it: `prefetch_depth: 0` exited 1 with
`ValueError: No scans processed`.

Fixed with `yield from iterable` then `return`, and every put — item, exception and end marker — now goes
through one stop-aware `publish` helper, which also closes the P2 race they found: a blocking `put` with no
timeout cannot be released by `stop`, so a consumer closing on a full queue parked the producer forever.

Measured after the fix, same 30 frames of `roundT_doubleT`: `prefetch_depth: 0` processes 30 frames with
statuses identical to `prefetch_depth: 1` (`{candidate: 1, unresolved_obstacle: 29}` in both). Before the
fix the same command exited 1.

Their proposed reader on/off comparison now runs, and it was done on identical geometry: `doubleT_obstacle`,
60 frames, same recipe, three repetitions per mode. Wall time per frame: 213.2 / 180.9 / 177.7 ms inline
against 169.1 / 165.7 / 164.9 ms with prefetch - a median gain of 15.2 ms, about 8%, and a large reduction in
spread (the inline repeats vary by 35 ms). The detector's own p50 is unchanged within noise (131.3 against
135.2 ms; the reader thread competes for a core). Neither mode reaches 10 Hz here, and absolute latencies on
this shared machine are not comparable with other sessions' numbers - only the on/off contrast is controlled.

## P1 — the widened corridor was clipped again: confirmed, fixed, measured as latent

`tunnel_guard/detector.py`, `process`. The mask keeps `|y − offset(x)| < base + |offset(x)|`, which at base
8 and offset 10 reaches y = 28, while the kernel's own limit `|y| < base + max|offset| = 18` cut that back.
Both implementations apply it: `accelerator.crop_voxels` uses `np.abs(frame[:, 1]) < half_width`, and
`cpp/kernels.cpp:select_crop_voxels` takes the same half-width.

Fixed by reading the kernel's limit off the masked subset itself — its largest `|y|`, plus one voxel to clear
the strict inequality — so the pass cannot drop a point the mask kept, and the grid stays as tight as the
data allows. No temporary array is allocated.

Effect, measured rather than inferred: **0 frames differ on either recording and on the arc panel.** The
reviewer's example needs `max|offset|` to exceed the local `2·offset(x) + base`, and on these recordings the
corridor's offset grows monotonically along the frame, so the old limit was always the looser one. The
defect was latent, not active: it would bite on a frame whose corridor bends back toward the axis, or on one
whose far extent is small. Fixed as a latent hazard, with the measurement stated rather than an implied
improvement.

## P1 — the continuation covariance was incomplete: confirmed, fixed

`tunnel_guard/geometry.py`, `_continuation` and `path`, and the mirror in `cpp/kernels.cpp`. The per-point
extension used `d² Var(a) + d⁴ Var(b)` and dropped `2 d³ Cov(a,b)`. The fit is one-sided, so `s12 < 0`,
`Cov(a, b) = −s² s12 / (det · window³) > 0`: the omitted term is positive and grows faster with distance
than either diagonal term.

Both backends now carry the full covariance, with the same summation order, and clamp the combined variance
at zero. Verified: 20/20 geometry arrays identical across the two recipes, 24/24 kernel checks passed.

Measured effect on 60 frames of each recording: one object on `doubleT_obstacle` moves `adjacent →
unresolved` (`adjacent` 10861 → 10860, `unresolved` 3230 → 3231); `roundT_doubleT` is unchanged; statuses
are unchanged on both. Small in magnitude because the propagated term is small against the heuristic 0.4 m
budget — but it is now the *correct* variance, and the direction is conservative.

The reviewer's remaining cautions stand and are not papered over: overlapping windows, edge-anchor error and
model error are still outside this bound, so it is not called a calibrated probabilistic boundary anywhere.
Independently, the measured coverage of the propagated σ is 0.11–0.29 at 50–150 m (over-confident by
1.6–8×), which is why the horizon decision uses the heuristic term, not this one.

## P2 — the side case in the arc panel moved the size, not the object: confirmed, fixed

`tunnel_guard/curved_stress.py`, `object_on_arc`. `lateral_m` was added to a half-extent, so the box grew
and stayed on the centre-line. The centre is now translated along the local normal and the box keeps its
dimensions, so the 0.6 m case actually tests an object beside the path.

The reviewer is also right about the metrics, and this changes what I may claim:

* `object_reported` is relabelled **candidate presence**: any prediction box intersecting the label region.
  It is not a count of confirmed intrusions.
* `object_confirmed_intersecting` is added: cases whose object is confirmed to intersect the reference
  corridor. This is the operator-facing decision, and it is the number that belongs in a headline.
* `union_coverage` is relabelled: it samples the label *box* on an 8³ grid, not the reflection points, so it
  is a box-overlap measure.

On the corrected panel (24 cases: an object of 0.8 × 0.8 × 1.0 m on an R = 300 m arc at 40/60/100/150 m,
lateral 0.0 and 0.6 m, three frames each), with the review fixes in place:

| configuration | candidate | confirmed intersecting | one-to-one tp | unmatched |
|---|---:|---:|---:|---:|
| old (straight continuation, old crop) | 9/24 | 5/24 | 3 | 502 |
| corridor crop only | 9/24 | 5/24 | 3 | 486 |
| curvature continuation only | 12/24 | 6/24 | 6 | 74 |
| **shipped (both)** | **14/24** | **6/24** | **6** | **74** |

Honest reading: candidate presence rises 9 → 14 of 24 and unmatched objects fall 502 → 74, while *confirmed
corridor intrusion* rises only 5 → 6. The earlier "12 → 16 of 24" figure came from the mis-placed panel and a
candidate-level metric; it is superseded by this table.

## Regression verification after the fixes

| check | result |
|---|---|
| measured-pattern panel, 1460 frames | **byte-identical**: tp 709, fn 251, fp 176, precision 0.8011, event recall 0.8021, matched mean IoU 0.8223, distance MAE 0.00111 m, zero empty-scene alarms; wall time 415 -> 430 s, which is load, not work |
| the panel's own declared criterion | `event_recall >= 0.95` is **not met** (0.802). It was not met before these fixes either: unchanged, and reported as the standing gap it is |
| NumPy vs native | 20/20 geometry arrays identical, 24/24 kernel checks passed |
| both recordings, 60 frames each | statuses identical; `roundT_doubleT` relations identical; `doubleT_obstacle` moves one object `adjacent -> unresolved` |
| `prefetch_depth: 0` vs `1`, 30 frames | runs now, statuses identical (`{candidate: 1, unresolved_obstacle: 29}` in both) |
| arc panel, 24 cases | candidate 9 -> 14, confirmed intersecting 5 -> 6, one-to-one tp 3 -> 6, unmatched 502 -> 74 |

## What the review is right about and this branch has not changed

* **Horizontal only.** These changes are about the path's lateral geometry. They do not address the rail-head
  height bias the reviewer's own work found, and they do not replace a full `local_3d` check.
* **Curvature on real turns: measured, and the curvature model wins at every depth tested.** The six
  sourcecraft recordings are near-straight, so the review was right that the fitted continuation was
  unvalidated where it matters. The extended recording supplies the curves: sampling it as 25 subsets of two
  consecutive splits each (102 frames, spread across the run) gives 2546 frames of which **541 are curved**
  (radius <= 800 m; quartiles p25 1123 m, median 5015 m, minimum 138 m). For each curved frame a look-ahead
  point is taken on each model's centre-line past the anchor end, carried into a later frame that has driven
  over the same ground, and compared with the rails that frame measures there — label-free, the sensor's own
  later measurement as reference:

  | past the anchor end | pairs | curvature model | tangent model | ratio |
  |---|---:|---:|---:|---:|
  | 5 m | 514 | **0.044 m** | 0.074 m | 1.7x |
  | 10 m | 352 | **0.090 m** | 0.198 m | 2.2x |
  | 20 m | 83 | **0.247 m** | 0.664 m | 2.7x |
  | 30 m | 4 | 0.450 m | 1.302 m | 2.9x |

  Two limits belong next to that table. **The pair counts are not independent places:** consecutive frames of
  one stretch are correlated, so the 514 pairs at 5 m rest on about a dozen moving stretches and the 20 m and
  30 m rows on three or four. And **the sample is half stationary:** total travel per 102-frame subset has a
  median of 6.6 m (range 0–61.2 m), because these stretches include stops, so the deep rows are bounded by
  travel rather than by method.

  The gain grows with distance, which is the curvature signature rather than a slope-only improvement, and it
  holds on radii down to 138 m. Stated plainly: the deep rows are thin (83 pairs at 20 m, 4 at 30 m, because
  30 m past the anchor end needs ~38 m of advance and a subset carries at most ~33 m); the reference frame's
  own rail error is common to both models and is not separated here; and this is a continuation-accuracy
  measurement, not recall or precision. Three things had to be measured before the test could run at all —
  the curvature distribution above, the vehicle's crawl (some subsets barely move — one
  nets 2.3 m over 102 frames — while others advance about 30 m, so pairing must be by travel and not by
  frame index), and the anchor span (5–53 m) — and each one
  changed the design rather than being assumed.

* **Curvature was unvalidated on real turns when the review was written.** The six sourcecraft recordings have almost no measurable
  curvature; the extended 20-minute recording (`data/new_data`, 11 271 scans, 22% of frames below 800 m
  radius) is where this must be tested. A sampler for it is added (`python -m tunnel_guard.extended_subset` builds one-split
  subsets so places spread along the run can be read without copying 84 GiB) — the measurement itself is
  still outstanding.
* **No universal speedup.** The reviewer's 90-frame comparison shows my branch slower in median processing
  on all three recordings (157.6→174.8, 129.4→139.1, 131.8→144.8 ms) and better in wall time per frame on
  two of three. Consistent with my own table, which claims a gain for the reader overlap (164.7→126.7 ms)
  and a cost for the corridor-aware crop (+4–6 ms/frame), both measured under contention and neither
  controlled for warm-up. 10 Hz is not reached.
* **Nothing about this is a field acceptance test.** No independent exhaustive annotation of the panel
  exists, so no precision claim is made, and the synthetic scenes are logic checks with the measured beam
  pattern, not a simulation of the sensor's detection range.

## Reproduction

The historical backend-equivalence commands were removed with the NumPy detector path;
their earlier results remain in the saved reports.

```sh
uv run python setup.py build_ext --inplace
uv run python -m tunnel_guard.run --experiment <experiment with prefetch_depth 0 and with 1>
uv run python -m tunnel_guard.curved_stress --experiment <arc panel experiment>
uv run python -m tunnel_guard.extended_subset --bag data/new_data --indices 0 9 18 --out data/extended_subset
```

Artifacts: `build/arc2-{old,crop,curve,shipped}/metrics.json` (corrected panel),
`build/perf-prefetch-{off,on}-fixed/` (reader fix), `build/stress-measured-pattern-review-fix/` (fixed panel),
`build/perf-review-after/` versus a worktree of `5587685`. Machine-readable record:
`results/review-response-20260919.json`.
