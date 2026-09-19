# Curves, the reference corridor, and how far it can honestly reach

The reported symptom was that in curved tunnels — turning left, right, up or down — the reference
clearance corridor did not follow the train. This is the investigation, the two defects it found, what was
fixed, what was measured, and the limit that no estimator can pass on this data.

Everything below is measured on this repository's recordings and panels; the artifacts are listed at the end.

## The symptom had two causes

**1. The corridor continued straight past the measured rails.** The reference contour was built from the
measured rail centre-line, which is supported to a median 40 m, and beyond the last anchor it continued
along a straight tangent whose slope was clipped at `rail_max_heading` (0.12 rad = 6.9°), modelled for
`path_max_extrapolation_m` beyond the nearest anchor. On a curve the true centre-line leaves that line
quadratically. Measured against what *later* frames see over the same ground (label-free pair test):

| look-ahead | straight continuation | curvature continuation |
|---|---:|---:|
| 40 m | 0.094 m | **0.039 m** |
| 50 m | 0.370 m | **0.110 m** |
| 60 m | 0.852 m | **0.165 m** |
| 100 m | 4.036 m | **1.372 m** |

(n = 13–20 per row, 24 source frames of `roundT_doubleT`.) The old numbers are against a corridor
half-width of 1.535 m, so at 60 m the old model was already 55% of the half-width wrong.

**2. The detector cropped away the curved track before looking at it.** The input was cropped to a fixed
`context_half_width_m` (8 m) lateral window in the *sensor* frame. On a curve the track leaves that window —
on R = 300 m the true centre-line departs from the sensor axis by s²/(2R): 0.67 m at 20 m, 2.67 m at
40 m, 16.7 m at 100 m — so a fixed 8 m half-window holds it only to ~69 m, and less on a tighter curve. So both the
returns the corridor must classify *and* the anchors that estimate the curve were discarded before
classification. This is upstream of cause 1 and, on tighter curves, masks it entirely.

## Fixes

**Corridor.** `TrackGeometry._continuation` (Python) and its mirror in `cpp/kernels.cpp` continue along a
local quadratic fitted to the anchors inside `path_curve_window_m` (30 m), expressed in the edge anchor's
own frame so `center(x)` stays continuous there. The curvature is shrunk to zero unless it exceeds
`path_curvature_significance` standard errors (4), and the extrapolation uncertainty is the fit covariance
added in quadrature to the previous base term. Inside the anchor span the corridor is bit-identical to
before; `path_curve_window_m: 0` reproduces the previous continuation exactly.

**Input crop.** The window now follows the previous frame's remembered corridor (offset = centre-line at
each point's along-track coordinate, half-width = base + |offset|), gated on `corridor_crop_threshold_m`
(the code default 0.5 m — the key is absent from the configs, so the default applies) so a straight run keeps
the original crop exactly. Causal: no extra pass, no future frames. The
native crop kernel returns indices into the array it is given, so it receives the already-masked subset with
a half-width that cannot re-clip it.

## What the numbers say

**No regression where the old model was already right.** The measured-pattern panel (1460 frames, straight
rails) is byte-identical: 709/251/176, precision 0.8011, event recall 0.8021, matched mean IoU 0.8223,
distance MAE 0.00111 m, zero empty-scene alarms. NumPy and native backends agree (20/20 geometry arrays,
24/24 kernel checks).

**Real recordings.** With the shipped code: `roundT_doubleT` (curved, 60 frames) has **0 frames changed** —
identical statuses, relations and objects. `doubleT_obstacle` (stationary platform, 60 frames) drops 10 of
14,169 objects (0.07%), moves 14 relations from unresolved to adjacent, leaves the 68 intersecting unchanged.

**Detection on curved scenes.** The fixed panel has straight rails, so it *cannot* show this fix — the
curvature shrinks to zero and the crop offset is ~0 there. A curved panel was therefore built
(`tunnel_guard/curved_stress.py`: analytic arc tunnel, walls as coaxial cylinders, floor/ceiling as planes,
rails as thin cylinders at gauge and head height, object a world-frame box on the arc, measured beam
pattern, exact labels). Over 24 cases (object on the track centre of an R = 300 m arc at 20/40/100/150 m):

| configuration | object reported | unmatched objects | union coverage |
|---|---:|---:|---:|
| old (straight continuation, old crop) | 12/24 (0.50) | 476 | 0.023 |
| crop fix only | 12/24 (0.50) | 470 | 0.023 |
| curvature fix only | 12/24 (0.50) | 84 | 0.047 |
| **shipped (both)** | **16/24 (0.67)** | **84** | **0.055** |

Neither fix alone achieves it: **+33% of cases report the object and 82% fewer spurious objects**, which is
what the two-cause analysis predicted.

**Vertical (up/down) reference.** The bed is sampled to **65–105 m** (13–18 anchors per frame — the floor is
wide), and inside its own 15 m gate the linear bed extrapolation errs by **≤0.02 m** even where the vertical
curvature is R_v ≈ 7 km. Heights above the running surface are therefore sound far beyond the lateral
horizon; the binding unknown at range is lateral.

**Horizon and uncertainty are a calibrated policy, not a round number.** Reach is set by
`path_max_uncertainty_m` (0.4 m): raising `path_max_extrapolation_m` 25 → 45 m changed no classification at
all. The heuristic σ (`0.06 + 0.008r + 0.0003r²`) tracks the measured centre error within 0.46–1.34× over
10–110 m of extrapolation, so the ~74 m horizon sits where the measured error is 0.72 m — 47% of the
half-width. Raising the budget to 0.7 m (86 m horizon) was measured and **rejected**: it turns two frames of
`roundT_doubleT` into certified obstacles (intersecting 12 → 20) where the centre is uncertain by ~1.0 m,
for no measured gain. The uncertainty propagated from the fit covariance is over-confident by 1.6–8× at
50–150 m (coverage 0.11–0.29), so the heuristic term is the calibrated model. Every object now carries
`far_field_lateral_bound_m` — the same σ the classifier used, or `null` beyond the horizon (verified
reporting-only: 200 frames, identical statuses and objects).

## The far-field limit, and why it is information, not algorithm

| evidence | measurement |
|---|---|
| rail returns per 20 m bin, 10 → 90 m | 1995 → 364 → 105 → 17 → **0**; the track is not measured beyond ~90 m |
| bed-band return fraction, 10 → 70 m | 0.050 → 0.015 → **0.000** on the gauge-wide strip |
| tunnel bore as a proxy | visible to 120–207 m, but sits ~1 m off the track centre and drifts with section changes; calibrated on the rails it predicts only 1.32 m at 100 m — **worse** than the shipped model |
| model class | at 100 m the best of six models is 1.4 m; at 120 m 3.5 m; at 180 m 9.9 m — against a 1.535 m half-width |
| penalized spline (correct class for changing curvature) | cannot fit the surface traces at all: 1.7–4.5 m residual with the penalty removed |
| uncertainty magnitude | the honest error at 150 m (~4.8 m) exceeds the tunnel width (2.5–4 m) |
| temporal history as the long lever | **worse** than a single scan at every range (60 m: 1.084–4.840 m vs 0.174; 200 m: 46–76 m vs 10.0) |

So the lateral track relation is not recoverable beyond the modelled horizon on this sensor and route: the
track's own signature dies at 40–90 m, the only surface that reaches 200 m is biased against the track
centre, and the alignment's *future* curvature is not a function of anything measured (these routes are
sequences of independent curves and junctions). Objects beyond the horizon are reported as `unresolved`
with a directly measured distance and a trustworthy height, and their lateral relation is stated as unknown
— never as clear, and never as a certified intrusion.

Three things would change that, none of them present in the current inputs: **more returns at range**
(denser beams or a longer baseline), **an external alignment** (a prior pass over the same route — a track
database rather than a per-scan estimate — or the survey), or **a vehicle swept-volume specification**
(overhang and sway limits, converting an unmeasurable centre-line into a bounded volume claim).

## Trade-offs taken, stated

- **Latency.** The corridor-aware crop costs **+4–6 ms/frame** (127.1 → 133.6 and 139.1 → 143.3 p50) because
  on these recordings the gate engages — a fitted curvature bends the continuation by metres within the
  observed range, which is the situation it exists for. `corridor_crop_threshold_m` disables it at the cost
  of curves. Measured while a panel run shared the machine; to be re-measured idle.
- **Straight-recordings delta.** The fitted slope replaces the noisy two-point tangent even where curvature
  is shrunk, which moves about 0.1% of object relations on the stationary recording.
- **Conservative under-bending.** The 4σ shrinkage keeps the straight continuation for a genuinely gentle
  curve a 30 m window cannot resolve. Under-bending, never over-bending.
- **The curved panel is work in progress.** Its labels and the detector's per-cluster support do not match
  under the project's one-to-one IoU convention (an object's support splits into near face and top face
  here), so `object reported` is a coarse measure and the panel is **not** comparable in absolute terms to
  the IoU panels, and is not used as acceptance evidence. 24 cases, one arc radius, stationary sensor, ideal
  surfaces with the measured beam pattern.

## Artifacts

- `tunnel_guard/geometry.py` (`_continuation`, `path`), `cpp/kernels.cpp` (mirror), `tunnel_guard/detector.py`
  (`_corridor_offset`, crop) — the shipped changes.
- `tunnel_guard/alignment.py`, `alignment_probe.py`, `uncertainty_probe.py`, `history_probe.py`,
  `curved_stress.py` — the estimators and probes, none selected by any recipe.
- Recipes: `configs/stress-measured-pattern-{final,curve,curve4,cropcheck}-*.json`, `configs/perf-curve-*.json`,
  `configs/perf-prefetch-on.json`.
- Recorded evidence: `results/curve-continuation-20260919.json`,
  `results/alignment-long-lever-20260919.json`, `results/reader-overlap-20260919.json`,
  `build/uncertainty-calibration.json`.
