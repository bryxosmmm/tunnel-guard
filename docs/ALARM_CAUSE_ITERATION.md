# Alarm cause iteration — 2026-09-16

## Decision and implemented change

The classifier previously called nominal lateral crossings `intersecting` while ignoring
its own nonzero lateral path uncertainty. This is reproducible on real returns:

- Frame 200, near structure at ~2 m: three candidate voxel representatives cross the
  nominal boundary by only 0.00048–0.00143 m; path uncertainty is approximately 0.07–0.09 m.
- Frame 0, structure at ~34 m: 11 representatives cross by 0.021–0.083 m versus
  approximately 0.16 m of path uncertainty.

These are **unsupported claims of definite intersection**, not independently labelled
false obstacles. We have not established the exact vehicle envelope or fixture identity.

`TrackGeometry.classify` now projects the existing path-centre uncertainty through the
same roll transform used for lateral coordinates. A point must remain inside throughout
that interval to be interior evidence. Points straddling either side of the boundary remain
unresolved evidence. No uncertainty thresholds, clustering radii or confirmation counts
were tuned. There is no bag-name rule, sensor-axis clipping or new semantic filter.

`cluster_candidates` preserves mixed interior/boundary support (e.g. one point of each)
so splitting evidence into categories cannot erase a weak candidate. Boundary points are
protected from background removal. Candidate boxes retain full measured support. New
per-object fields explain the boundary uncertainty; RViz uses orange for confirmed
unresolved objects and red for confirmed interior evidence.

## Fixed cases

| Case | Before | After | Measured object support |
|---|---|---|---|
| ~2 m, frame 200 | intersecting | unresolved | Same 165 voxel representatives, identical bbox |
| ~34 m, frame 0 | intersecting | unresolved | Same 300 voxel representatives, identical bbox |
| ~56 m, frame 25 | nominal interior region | interior evidence retained | 133/134 raw interior returns remain interior; one becomes boundary-uncertain |

The ~56 m main-detector candidate remains confirmed/intersecting with 74 total support
voxels and 68 interior voxels after the change. It is not labelled a proven obstacle:
its identity and actual physical path intersection need human review/calibration.

The raw-window counts in `cases.json` differ from object counts because windows include
all returns (including background), while detection uses voxel representatives and clusters.

## Teammate probe reproduced

Ran the unchanged `on_track` probe before modifying the classifier, on the complete
201-frame `doubleT_obstacle` recording. Result: 989 in-envelope clusters -> 43 grouped
events -> 6 reported candidates. Four reported fragments are near 56 m, two near 34 m.
Their frame spans are 2–30, 45–57, 62–69, 73–75 and 140–146, 150–153 respectively.
These are outputs of its unvalidated grouping/filtering, not six physical objects.
This does not reproduce the earlier six-recording total: only this bag was run.

## Full before/after results

| Recording (frames 0–200) | Before: obstacle / unresolved | After: obstacle / unresolved |
|---|---:|---:|
| doubleT_obstacle | 201 / 0 | 174 / 27 |
| doubleT_platform | 197 / 4 | 10 / 191 |

Confirmed interior object-frame observations: 1950 -> 634 on obstacle and 1034 -> 20 on
platform. These are correlated observations, not independent events or a recall measure.
Unresolved observations increase (2608 -> 6998; 3074 -> 7667). Total candidate observations
also increase (43580 -> 44515; 31757 -> 34967). This is the explicit cost of retaining the
uncertain boundary, and the overall warning load has not been eliminated.

All 402 frame indices and acquisition timestamps match. On all ten fixed diagnostic
frames, every prior geometry point, post-background context point and clustering input
point is present after the change (nearest-point tolerance 1e-8 m). This support check
covers those ten frames, not every point in the full recordings.

The two fixed near-object bboxes match exactly (IoU 1.0), with unchanged support counts;
only their relation becomes unresolved. The remaining strong ~56 m candidate is preserved.
JSON evidence: `results/alarm-cause-comparison.json`. Diagnostic raw windows and visual
comparison: `build/alarm-cause-analysis/current-cases/cases.json` and `cases.png`.
The real RViz bag was reopened and marker messages decoded successfully; orange unresolved
markers and explanatory labels are present. This is format/content verification, not GUI QA.

## Reproduction

Base revision: `a508709`. Environment: existing `.venv-iteration`, Python 3.12; dependency
versions, source copies, working-tree patches and bag identities are in run manifests.
Both recipes use seed 20260915, no subsampling, a fresh detector per bag and frames 0–200.
This is the complete obstacle bag and the first 201 of 345 platform frames. Neither bag
is certified empty. Diagnostic frames were fixed to 0, 25, 75, 181 and 200 before editing.

```sh
.venv-iteration/bin/python -m tunnel_guard.on_track --bag data/sourcecraft_subset/for_hackathon/doubleT_obstacle --config configs/on-track-probe.json --output build/alarm-cause-probe.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/alarm-cause-before.json
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/alarm-cause-after.json
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.render_alarm_cases --before build/alarm-cause-before --cases configs/alarm-review-cases.json --output build/alarm-cause-analysis/current-cases
.venv-iteration/bin/python -m tunnel_guard.compare_alarm_runs --before build/alarm-cause-before --after build/alarm-cause-after --output results/alarm-cause-comparison.json
git diff --check
```

The `before` recipe alone does not select old code: use revision `a508709` or its saved
source snapshot to reproduce the baseline. Do not rerun it on changed code and call that
baseline. Output directories are intentionally non-overwriting. Rendering uses recorded
baseline path/ground geometry and actual raw points; it does not regenerate a synthetic scene.

The after run also exports actual RViz result bags, using 30k points for the display copy.
Candidate support points are exported separately. GUI launch remains unverified locally.

## Limits and tradeoffs

- This fixes an uncertainty-classification error; it does not prove that the path is correct,
  identify infrastructure, or establish a reduction in the field false-alarm rate.
- More unresolved candidates are expected because both sides of the uncertain boundary
  are retained. An unresolved route must not be presented as clear.
- The path uncertainty is heuristic, not calibrated confidence. The change handles lateral
  centre uncertainty; it does not solve full correlated roll/pitch/extrinsic/railhead error.
- Existing vertical support and rail exclusion logic remain assumptions requiring review.
- Runtime comparison is not controlled: runs overlapped with local review work, and only
  the after recipe exports RViz data. No speed improvement is claimed.
- No automated tests, synthetic obstacles, neural model or future-frame evidence were used.

## Next priority

Review the persistent ~56 m return group and remaining interior alerts against actual path
geometry. Ask a reviewer to label the three rendered regions as infrastructure/object/unknown,
with path intersection separately recorded. Then address rail-centre/railhead estimation
where the data demonstrates it is wrong. Do not restore a hard sensor-axis lateral cutoff
or make unresolved evidence disappear to improve the alarm count.
