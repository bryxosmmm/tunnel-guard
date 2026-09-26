# Separate object persistence from path-intersection evidence

## Real-data diagnosis

Reviewed all ten remaining `obstacle` frames in the first 201 frames of
`doubleT_platform` after revision `38d1e91`: 9, 15, 47, 56, 60, 64, 186, 193, 199,
200. Extracted 27 scans including neighbours, exact candidate representatives from
recorded RViz markers, and the recorded geometry/poses. Also extracted 53 scans for
18 fixed views and neighbours around the ~56 m group in `doubleT_obstacle`.

Two concrete causes appear in code and measurements:

1. `cluster_candidates` used the **whole object's** density and vertical extent for
   immediate confirmation. Platform frames 47/56/60/64 contain a long surface with
   5517/2898/5138/5191 representatives, but only 7/16/4/3 interior representatives.
   Thousands of returns outside the envelope therefore lent confirmation to a weak
   instantaneous intrusion. In the top and side views these are continuous edge/surface
   structures, not visibly isolated newly appearing objects.
2. `Detector._associate` counted adjacent/unresolved observations towards object
   persistence, and `Detector.process` then promoted any current weak intersection of
   that object to `obstacle`. Persistence of the object did not establish persistence
   of its intersection.

For the four long platform surfaces, projecting the same current support into the
previous estimated pose/geometry changes the interior count to zero. Median distance
from projected support to actual neighbour returns is 0.016–0.023 m. This supports
boundary/geometry fluctuation as a cause. Other sparse far cases have much larger
nearest-return differences, so the same interpretation is less secure there.
Counterfactual neighbour projection assumes static support and estimated registration;
it is an offline diagnostic, not ground truth. Future neighbours never enter inference.

The ~56 m region contains a compact upright return group whose lateral position and
observed vertical shape change across the inspected frames. It remains visible in the
raw cloud at frames 80 and 100, beyond the probe's last reported fragment at frame 75.
Thus probe selection is not a reliable presence/absence label. A moving person is a
plausible interpretation, **not an established class or proven collision hazard**.
It has not been identified with the separately annotated nearby person at frames 165–200.

## Change

- Keep object confirmation, matching, clustering, full support and bounding boxes intact.
- For immediate **intersection** confirmation, require the existing 30 density-core
  representatives and 0.5 m processing-frame z span within the interior subset itself.
- Otherwise require interior evidence in two distinct scans of the last three, using
  the existing confirmation configuration. Earlier adjacent/unresolved observations
  cannot count. The current scan must still be intersecting and the object confirmed.
- Record `intersection_confirmed`, `intersection_immediate`, `intersection_hits`,
  `intersection_evidence_timestamps_s`, `intersection_confirmation`, interior density
  count and height span. Reader deduplication and the detector's strict timestamp order
  remain in force. Distinct acquisitions do not imply statistically independent errors.
- Reserve frame status `obstacle` and red RViz markers for confirmed intersections.
  Confirmed objects with uncertain/pending intersection remain `unresolved_obstacle`
  and orange. Tentative candidates remain yellow. Existing `confirmed` means object
  confirmation, not intersection confirmation.

Current contract clarification (2026-09-22): the later corridor-claim policy made
`confirmed` depend on interior/uncertain support, so it no longer represented
general object presence. `presence_confirmed` and `presence_confirmation` now
represent measured component presence separately. `near_track_objects` evaluation
uses those fields; hazard decisions retain the corridor policy and relation checks.
The original measurements below are historical, not measurements of this new field.
See [the point-chain diagnosis](DETECTOR.md#разрыв-между-присутствием-и-оценкой--2026-09-22).

This can delay weak detections until a second qualifying scan, and missed associations
can delay them further. Strong current interior geometry retains the immediate path.
Correlated contour errors can still persist across scans and get confirmed. The fix
neither establishes calibrated uncertainty nor removes the need for path calibration.
`nearest_obstacle_m` retains its previous meaning: nearest confirmed potential hazard,
including unresolved cases. The warning load is not eliminated.

## Reproduction and artifacts

Baseline: existing `build/alarm-cause-after`, generated at revision `38d1e91`'s detector
state. New run: `configs/intersection-evidence-after.json`; seed 20260915, frames 0–200
of each bag, every scan, same detector configuration and RViz export settings. The
obstacle bag is complete; the platform bag has 345 scans, of which 201 are evaluated.
These are development recordings, not labelled empty scenes or a held-out safety panel.
Source/config snapshots and manifests are under the run directories.

```sh
.venv-iteration/bin/python -m tunnel_guard.review_remaining_alerts --config configs/remaining-alert-review.json
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.review_remaining_alerts --config configs/remaining-alert-review.json --render-only
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/intersection-evidence-after.json
.venv-iteration/bin/python -m tunnel_guard.compare_alarm_runs --before build/alarm-cause-after --after build/intersection-evidence-after --output results/intersection-evidence-comparison.json
.venv-iteration/bin/python -m tunnel_guard.evaluate --run build/intersection-evidence-after --annotations annotations/doubleT-obstacle-person.json --output build/intersection-evidence-after/person-evaluation.json
.venv-iteration/bin/python -m tunnel_guard.evaluate --run build/alarm-cause-after --annotations annotations/doubleT-obstacle-person.json --output build/intersection-evidence-after/baseline-person-evaluation.json
.venv-iteration/bin/python -m tunnel_guard.trace_analysis --run build/intersection-evidence-after --annotations annotations/doubleT-obstacle-person.json --output build/intersection-evidence-after/person-trace.json
git diff --check
```

The review extraction and experiment require fresh output directories. Existing review
extractions can be redrawn with `--render-only`. Review PNGs and point clouds stay local
under `build/remaining-alert-review/`; the quantitative neighbour evidence is in
`evidence.json`. Four platform sheets cover XY/XZ views and neighbouring scans for all
ten alert frames; two obstacle sheets cover the fixed 56 m region. Red points in these
**baseline** sheets denote confirmed-object interior support, not the new intersection
confirmation rule. Whole-scene status in each title can be caused by another object.

No automated tests or synthetic data are used. Actual recorded inference and offline
measurement evaluation provide the evidence. ROS 2/RViz GUI execution remains unverified.

## Completed comparison

| Recording, frames 0–200 | Before obstacle / unresolved | After obstacle / unresolved |
|---|---:|---:|
| doubleT_obstacle | 174 / 27 | 141 / 60 |
| doubleT_platform | 10 / 191 | 1 / 200 |

The remaining platform alarm is frame 200: IDs 17376 and 17587 each have interior
evidence in frames 199 and 200. This is repeatable algorithmic intersection evidence,
not a verified physical hazard. Object-frame confirmed intersections change from
634 to 241 on obstacle and 20 to 2 on platform. All 402 frames remain warnings;
none is reclassified as a clear route.

Across all 402 frames, candidate IDs, boxes, support/interior counts, geometric relations
and object confirmations match the baseline exactly. Candidate observation counts stay
44515 and 34967. All acquisition timestamps match. Every baseline point in the three
compared processing stages is retained on all ten captured diagnostic frames (tolerance
1e-8 m). Point-level comparison covers those ten frames, not every point in all scans.
The 79482 object records contain no future or duplicate intersection-evidence timestamps.

At obstacle frame 25 the ~56 m candidate retains 74 support representatives, 68 interior
representatives and immediate interior confirmation. This does not establish its event
recall. The existing person-box evaluation remains **0/36 matches at IoU >= 0.25**, as in
the baseline: no improvement in localization is claimed. Its nonexhaustive, partly
detector-propagated boxes and unspecified volume convention remain evaluation limitations.
The `collision_hazards` evaluation scope still includes confirmed potential hazards
(unresolved and intersecting); it does not select only `intersection_confirmed` objects.

Read back actual ROS bag marker messages at obstacle frame 25 and platform frames 47,
200. Respectively: 1 red / 40 orange, 0 red / 1 orange, and 2 red / 80 orange object
markers; all 585 object labels include the new intrusion evidence fields. Results are
saved locally in `build/intersection-evidence-after/marker-review.json`. This verifies
message contents, not RViz GUI appearance.

The actual detector/export run exited successfully: obstacle 222.6 s, platform 113.2 s;
median detector processing 912.5 ms and 425.5 ms on this Mac. Local analysis ran
concurrently; these numbers are not a controlled performance comparison or real-time
acceptance. Full aggregate comparison: `results/intersection-evidence-comparison.json`.

## Next priorities

1. Diagnose temporal instability of the estimated rail centre/railhead using the fixed
   platform cases; avoid smoothing across unsupported geometry or switch transitions.
2. Review the moving ~56 m group as an event, with class and physical envelope relation
   marked independently; inspect all intervening frames before assigning ground truth.
3. Validate extrinsics, timing and real vehicle envelope on Ubuntu/Humble and additional
   recordings. The present contour is a configured reference, not a verified swept volume.
