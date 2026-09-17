# Small detections, height uncertainty and calibration observability

Date: 2026-09-17. Base: `59f90f8`. Real reference: `build/native-full-real`.

## What the small boxes actually mean

The detector boxes enclose **measured support**, not a reconstructed full object.
A small box can be a small item, a visible fragment of infrastructure, a sparse
return from a larger distant object, or an artifact. Size and repeated appearance
alone cannot distinguish these cases. Temporal persistence establishes repeated
measurements, not semantic identity or physical collision danger.

The previous 798-scan run contained 718 confirmed-intersection object-frame
observations. Of these, 57 had a maximum box dimension below 10 cm, and 312 below
30 cm. These are cumulative descriptive bins, not rejection thresholds or counts
of independent objects. None is automatically a labelled false positive.

The configured reference contour also conservatively fills its lower region,
omitting wheel recesses. A return inside this proxy is not necessarily inside the
actual vehicle body. Actual lower-body clearance cannot be recovered from the
reference contour alone; this is a separate source of potentially excessive alarms.

Concrete inspected example: platform frame 223, current track **17895**, about
54.92 m forward. Box **0.066 × 0.030 × 0.086 m**, two current voxel representatives,
three interior-support acquisitions in the confirmation window. It remains red
under the corrected geometry. The browser now displays its exact two points and
these numbers. This evidence does not identify a physical obstacle, and a tiny
sample is not a measurement of the full object's size.

## Review findings and implemented correction

1. `TrackGeometry.classify` previously used bed uncertainty only to raise the
   lower vertical threshold. It checked lateral intrusion against the envelope
   width at the nominal height. This ignored width changes across height steps,
   the upper vertical boundary, and the effect of bed-height error on the rolled
   lateral coordinate.
2. `Detector._associate` permits temporally confirmed sparse support; this is
   intentional for distant objects. Its output must not be interpreted as a
   semantic object classifier or calibrated collision probability. No size or
   confirmation threshold was raised to hide these detections.
3. The browser's sampled grey cloud could omit the few returns that supported a
   small box. It showed neither box dimensions nor support counts in its table.
   `review.html` and `review_viewer.py` now expose dimensions, counts, evidence
   reasons and exact saved current-frame representatives when available.

### Geometry

Let `u` be the already estimated bed-height error, `a,b` the fitted plane slopes,
`h` the nominal running height, and `s = sqrt(1+a²+b²)`. The running-height interval
is `[h-u/s, h+u/s]`. `envelope_width_bounds` obtains exact extrema of the configured
piecewise-linear half-width over this interval, including **both sides of width
jumps**. No sampled approximation or new uncertainty multiplier is introduced.

The lateral interval radius adds the propagated bed-height term
`abs(b)*u/sqrt(1+b²)` to the existing path-centre term. A point is interior only
when its full height interval lies within the contour and its lateral interval
fits within the minimum half-width. Possible marginal intersections remain
boundary evidence, including intervals that cross the lower/upper contour.
They are protected from background removal and retained for clustering.

The separate marginal bounds discard correlation. This is conservative for
interior certification and can retain extra unresolved points. It can also alter
cluster grouping and track IDs. This tradeoff is reported rather than hidden.

**Scope:** these remain heuristic path-centre and bed-height bounds. They do not
bound sensor extrinsics, railhead estimation error, roll/pitch uncertainty,
vehicle suspension, body shape or the true swept envelope. Health stays degraded.

## Full real-data evaluation

After a ten-scan actual-detector prefix, processed all three currently extracted
recordings, 798 scans. No new synthetic data, automated tests, Docker build or
archive extraction was used.

| Recording | Confirmed intersection observations before → after | `obstacle` frames before → after |
|---|---:|---:|
| doubleT_obstacle, 201 scans | 241 → 203 | 141 → 119 |
| doubleT_platform, 345 scans | 266 → 255 | 87 → 87 |
| roundT_doubleT, 252 scans | 211 → 191 | 69 → 64 |

Total: **718 → 649** confirmed observations. This is a correction of unsupported
certainty, not a measured improvement in field false-alarm rate. Very small
confirmed observations remain: below 10 cm, **57 → 58**; below 30 cm, **312 → 270**.
In particular, this correction does not solve the platform's alarm burden.
Some confirmations change because earlier interior evidence changes, even when
the current box and interior count are identical.

No frame/acquisition identity mismatch, future evidence timestamp or duplicate
history timestamp was found. On all **nine shared diagnostic frames** (0, 100,
200 in each recording), every prior geometry voxel, post-background context point
and clustering-input point is retained within 1e-8 m. This is a point-level
statement for those frames, not for every scan in the corpus.

Candidate observations at ≥60 m: **44,246 → 44,291**. Of the old far boxes, 44,229
retain identical rounded bounds. The remaining 17 all fit inside a new larger
box with increased support, consistent with expansion/merging; a containing box
alone is not a per-return preservation proof. The audit lists these cases.
Preserving more boundary support can merge clusters; no ID-invariance claim is
made for this change.

Recorded processing medians were 543 / 361 / 441 ms for the three recordings.
The extra interval work has a computation cost; this was not an isolated paired
timing benchmark and is still far from a 100 ms budget.

Results: `results/envelope-interval-20260917.json`. Complete decisions, selected
point diagnostics, original configuration, source snapshots and manifests:
`build/envelope-interval-real`. The unchanged thresholds and native backend are
from `configs/detector-native.json`. Old code/reference outputs remain available.

## Can mathematics recover the sensor position?

Yes, **some components relative to observed geometry**, under stated assumptions:

| Reference actually known | What it constrains |
|---|---|
| A plane whose relationship to the desired frame is known | Normal direction (two angles) and perpendicular distance |
| Correct paired rails with known correspondence | Additional track heading and lateral centre reference |
| Known longitudinal landmark or surveyed target | Position along the track relative to that landmark |
| Surveyed mount/body reference or sufficiently informative independent vehicle motion | Sensor-to-vehicle pose, with the appropriate calibration method |

A ground plane alone leaves translation within the plane and rotation about its
normal unconstrained. Parallel straight rails still leave translation along the
track unconstrained. A bed normal is not necessarily gravity or railhead cant.
A bend/grade ahead is not necessarily the instantaneous body heading/attitude.

The fundamental missing link is the **vehicle reference**. Writing
`p_vehicle = R * p_lidar + t` does not supply observations of `R,t`. If the vehicle
frame and its relation to the scene are unknown, different transforms and vehicle
poses explain the same LiDAR measurements. SLAM can estimate LiDAR motion and a
map; it does not alone identify the bumper's position. Known vehicle kinematics
and sufficiently varied independent motion may add constraints, but must be
specified and checked for degeneracy, not assumed from an approximately straight
railway recording.

The detector can therefore operate in the sensor frame with a clearly provisional
track-relative contour. It cannot honestly report bumper clearance or a verified
train swept-envelope collision until that reference is supplied. A sensor model
name alone would not resolve this missing reference.

The previous chronological calibration experiment estimated orientations but
rejected all three prefixes on fixed stability criteria. Those rejected estimates
were not installed. See `CALIBRATION_AND_NATIVE.md`.

Primary observability reference: [Autoware ground–LiDAR calibration](https://docs.autoware.org/main/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/calibrating-sensors/ground-lidar-calibration/)
explicitly restricts its ground-plane adjustment to z, roll and pitch; x, y and yaw
need other methods. Our rail estimator is not a reproduction of that method.

## Viewer and commands actually used

The updated browser table exposes XYZ support extent, total/interior voxel counts,
confirmation history and the reason for the path relation. Selecting an object
pauses playback and fetches **exact diagnostic representatives**, without display
sampling. If a frame has no saved diagnostic, the UI explicitly says so; it never
substitutes an approximate box crop and labels it detector support. The run name
is visible to distinguish historical results from current ones.

Browser inspection performed on real frame 0 (169 exact representatives), frame 1
(explicit unavailable-diagnostics state), and platform frame 223 (two exact
representatives, dimensions and three supporting acquisitions). The final local
viewer is `http://127.0.0.1:8767`, using the full updated platform run.

```sh
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/envelope-interval-prefix.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/envelope-interval-real.json
.venv-iteration/bin/python -m tunnel_guard.intrusion_audit \
  --before build/native-full-real --after build/envelope-interval-real \
  --output results/envelope-interval-20260917.json
.venv-iteration/bin/python -m tunnel_guard.review_viewer \
  --run build/envelope-interval-real \
  --bag data/sourcecraft_subset/for_hackathon/doubleT_platform --port 8767
git diff --check
```

For repeats, copy recipes and choose new output paths; do not overwrite retained
evidence. The initial audit before far-box expansion details is preserved locally
as `build/envelope-interval-real/intrusion-audit-initial.json`.

## What remains before a defensible obstacle MVP

- Resolve railhead-versus-bed reference, local cant and mounting/body reference;
  a scalar global railhead height and heuristic plane uncertainty are incomplete.
- Review representative persistent fragments and their surroundings, including
  the platform frame above, with independently defined hazard/annotation rules.
  An object being part of infrastructure does not alone prove clearance.
- Evaluate association and merging when protected boundary support joins nearby
  components. Preserve distant sparse evidence during any change.
- Establish sensor return/timing semantics and independent positive/negative
  coverage before making recall, nuisance-alarm or collision-range claims.

There is still software/geometry work possible without hardware specifications.
What is blocked is a **validated physical interpretation**, not all development.
