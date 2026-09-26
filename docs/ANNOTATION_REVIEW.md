# Review of teammate annotations — 2026-09-16

## Person-specific evidence — 2026-09-24

The user confirms that `doubleT_obstacle` has one target obstacle, a person.
The existing teammate labels are retained unchanged. Their positions were reviewed
on all 36 frames after detector-assisted propagation; they are not unreviewed
automatic labels. The author describes full-person volume in the README, whereas
the detector reports measured-return support. A fresh annotation pass from scratch
is not the prerequisite for inspecting this known person.

A fresh 201-frame real replay of `9e7fc92`, seed `20260915`, one ICP thread,
captured every labelled frame 165–200 and nine fixed far-field review frames.
The original oriented boxes select components by the number of measured points
inside the box, not by maximum IoU. All positive alternative components are recorded.
All six contact sheets (all 36 observations) were inspected against raw points:
the selected component follows the visible upright body and legs, not the bed.

| Question | Actual result |
| --- | --- |
| Measured person component | 36/36 frames; 331–449 component points |
| Points of that component inside the original oriented box | 282–415; next-best component at most 11 |
| Explicit presence confirmation | 36/36 |
| Track continuity within the labelled interval | One track, `2535`; zero ID switches |
| Relation assigned by the detector | `adjacent` on all 36 frames; zero confirmed intersections |
| IoU ≥ 0.25 against exported AABB | 5/36 |
| IoU ≥ 0.25 against original oriented full-person box | 14/36 |
| Median support-box / annotation-centre displacement | 0.436 m |

The oriented comparison intersects the actual box half-spaces; it does not resize
the prediction or remove label yaw. Numerical tolerance is explicit in the recipe.
Neither 5/36 nor 14/36 is a count of actual person detections. The centre displacement
also compares different volume conventions, not a surveyed physical centre.
These are 36 observations of one event, not 36 independent detections.

### Confirmed alarms are not person detections

The recording contains 203 confirmed hazard-component observations on 172 frames.
On the labelled interval, 32 observations on 27/36 frames are disjoint from the
person label. Under the supplied one-target scene truth, these are non-target alarms.
The other 171 observations occur on frames without person positions; their identity
is not established by this audit. Nineteen alarm track IDs are not nineteen objects.

All 32 labelled-interval non-target alarms carry
`certified_interior_shared_with_structure`, with zero
`certified_unexplained_voxels`. Sixteen have two certified interior voxels and
sixteen have three. Twenty-four use one recorded ring value, eight use two;
ring semantics and physical return independence remain unverified.

The causal code path is explicit: `cluster_candidates` intentionally preserves
certified interior evidence despite cross-section cell sharing with structure;
`_associate` can confirm thin interior evidence through temporal component support.
This proves why these records become alarms, not what physical surface generated
every return or whether the reference contour is physically correct.
For frame 181, the alarm is at about 56.38 m, while the person is near 10 m.
Its two certified witnesses are about 0.099 m apart, both carry `ring=59`,
and differ in height by about 0.00049 m. The actual witness coordinates,
timestamps, intensities, validity masks and source indices are preserved.

**Decision:** retain this fixed non-target panel and the 36-frame person control
for the next foreground-evidence / contour investigation. Do not suppress all
structure-sharing returns, discard a ring, inflate boxes or raise support thresholds
to pass this recording: low and sparse genuine obstacles can share those properties.
No detector behavior was changed and no false-alarm fix is claimed here.
This labelled person is adjacent according to the detector; it is not a verified
positive control for a real on-track intrusion.

Reproduce with `configs/person-specific-replay-20260923.json` and
`configs/person-specific-report-20260923.json`, respectively:

```bash
python -m tunnel_guard.run --experiment configs/person-specific-replay-20260923.json
python -m tunnel_guard.person_evidence_report --experiment configs/person-specific-report-20260923.json
```

Published evidence: [summary](../results/person-specific-20260923.json),
[person observations](../results/person-specific-observations-20260923.json),
[alarm records and witnesses](../results/person-specific-alarms-20260923.json),
[inspection and replay control](../results/person-specific-inspection-20260923.json).
Contact sheets and far-field figures are under `results/person-specific-20260923/`.
Full replay and diagnostics remain under `build/person-specific-replay-20260923/`.
The sections below retain historical findings and historical detector scores.

## Update after commit 3e2d017

Original PSR files are now available under `annotations/sustech-raw/doubleT_obstacle/`.
All 36 were read and converted using the existing exporter: maximum difference from
stored AABBs is exactly 0 m. The annotation JSON now retains PSR and source hashes.
The author reports checking position on every frame; size and rotation remain fixed
from frame 181. The earlier uncertainty about whether propagated positions were reviewed
is resolved by this report, not by an independent second annotation pass. Full-object
versus observed-support box convention still needs explicit agreement. The author's
oriented-IoU numbers have not yet been reproduced here.

Contact sheets remain local under `build/annotation-review-20260916/`; regenerate with the command below.

Original missing-file findings below describe the earlier review, not current availability.
The merge has since been committed for team publication. Next tasks: `docs/TEAM_TASKS.md`.

## Original decision and evidence

Incoming commits `0f85612..57945f4` were fetched and merged into `experiments/morev`
(base `d7105dc`) without conflicts, without a commit or push. Prior distance and
visualization work is retained. The merge is intentionally pending a commit.

**The new panel is useful for review, but is not accepted as independent ground truth.**
Do not tune detector thresholds, enlarge predictions, or lower IoU to make it pass.
No automated tests were created or run. Verification below uses actual bag extraction,
a sequential detector run, offline evaluation and visual inspection.

## Findings tied to code and data

1. `annotations/doubleT-obstacle-person.json`: one event, 36 observations (165–200).
   Per the supplied provenance, only frame 181 is manually authored; 35 boxes use a
   constant-velocity fit of a detector track. Added machine-readable origins without
   altering coordinates, classes, IDs, exhaustiveness or IoU. These are not 36 independent
   manual labels. All frames remain nonexhaustive and path intersection unverified.
2. The annotation AABB extent is approximately 1.393 × 1.545 × 1.787 m. Review of all
   36 raw XY/XZ projections shows visible object support and substantial surrounding
   bed support within the box. Parts of observed support can also extend beyond the box.
   This is not enough to certify amodal object size or identity. Decide whether the label
   represents full object volume or observed support before interpreting localization scores.
3. `tunnel_guard/sustech.py::box_aabb` correctly follows the vendored utility's Rx Ry Rz
   matrix product, but AABB conversion discards orientation and expands rotated boxes.
   Future exports now preserve original PSR and source-file SHA256, plus declared annotation
   origin. Original label JSON/PCD scenes are absent on this machine; the original PSR cannot
   be recovered from the existing AABB. End-to-end re-export is **not verified**: the CLI
   fails explicitly on the missing label directory. Duplicate numeric frame filenames now
   raise an error rather than silently overwrite each other.
4. `tunnel_guard/on_track.py::report_candidates` rejects abs(sensor-frame y) > 1.6 m.
   A curved track can legitimately move away from the sensor axis. This is not a valid
   general guard against corridor-estimation drift. Candidate count reduction is not recall.
5. `tunnel_guard/on_track.py::group_events` associates by y/z and frame gap without a
   longitudinal gate or ego-motion transform: different objects at different ranges can
   be grouped together. Its events are not validated physical identities.
6. `on_track.py::is_candidate` rejects attachment to large surfaces and repeated profiles;
   these properties do not prove that an intrusion is safe. Fixed minimum support also
   excludes weak distant evidence. The probe remains a review aid, not the production
   detection replacement. Its docstring says the main detector uses bounding boxes for
   intersection, while `Detector.cluster_candidates` actually classifies point support.

## Real-data results

- Extracted all 36 labelled raw scans using the configured local-frame transform.
- Counts inside annotation AABBs: 1,088–3,216 returns, including background; these are
  neither isolated object counts nor independent beam/temporal confirmations.
- Visually inspected all six contact sheets (all 36 observations).
- Ran current detector from frame 0 through 200 on `doubleT_obstacle`, no warm start,
  no subsampling, no duplicate measurements, no future input.
- Both saved baseline and fresh run: **0/36 matches at unchanged 3D IoU >= 0.25**.
- Fresh run best-overlap candidate IoU: 0.0929–0.2284. The best-overlap candidate is confirmed track 2769 on all 36 frames (adjacent),
  following the visible object, so zero box matches must not be described as proven absence of detection.
- No exhaustive frames: precision, F1 and false-alarm rate remain unavailable.
- Status unchanged on all 201 frames: all report obstacle. This does not establish
  correctness of those alarms. No detector behavior was changed in this iteration.
- Fresh processing p50 934.5 ms, p95 1183.6 ms, wall 208.68 s for 20.40 s of recording.
  Other review work ran concurrently; this is not a controlled performance comparison.

Compact result: `results/annotation-review-20260916.json`.
Full run and snapshots: `build/annotation-detector-20260916/`.
Raw evidence, original annotation snapshot, evaluations and sheets:
`build/annotation-review-20260916/`. These build artifacts are not committed.

## Commands executed

```sh
git fetch origin
git merge --no-commit --no-ff origin/main
.venv-iteration/bin/python -m tunnel_guard.annotate --annotations annotations/doubleT-obstacle-person.json --bag-root data/sourcecraft_subset/for_hackathon --config configs/detector.json --output build/annotation-review-20260916
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/annotation-review-run.json
.venv-iteration/bin/python -m tunnel_guard.evaluate --run build/iteration-baseline --annotations annotations/doubleT-obstacle-person.json --output build/annotation-review-20260916/baseline-evaluation.json
.venv-iteration/bin/python -m tunnel_guard.evaluate --run build/annotation-detector-20260916 --annotations annotations/doubleT-obstacle-person.json --output build/annotation-review-20260916/current-evaluation.json
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.render_annotation_review --evidence build/annotation-review-20260916 --annotations annotations/doubleT-obstacle-person.json --predictions build/iteration-baseline/doubleT_obstacle.jsonl
.venv-iteration/bin/python -m tunnel_guard.sustech --config configs/annotation-export.json
git diff --check
```

The exporter command fails because original scenes are missing; other listed processing
commands completed. Initial baseline scoring/extraction preceded metadata-only enrichment;
the original JSON is saved in `source/annotations.json`, and box coordinates are unchanged.
The renderer visualizes measured data; it is not a test suite or synthetic evaluation.

## Next gate

Obtain the original `doubleT_obstacle/label/*.json` (small files) and the reviewer definition
of box extent. Recheck anchor 181, then all propagated frames against raw measurements;
retain original labels and review provenance. Add independently reviewed negative intervals
before judging nuisance alarms. Only then select detector geometry/segmentation changes.
For tracking diagnosis, compare observed support and association before treating changed IDs
as new events. The hypothesis that the recording is stationary is not a verified calibration.
