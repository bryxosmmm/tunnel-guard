# Coverage expansion: real recordings and controlled insertions

## Scope and decision

Frozen production detector: revision `7e7c192`. No detector thresholds or architecture
were changed for this iteration. We expanded the evidence without waiting for annotation
work: full real-recording coverage plus controlled modeled shapes on actual return rays.
This is an experiment with measured outcomes, not an automated test suite or a claim of
field accuracy. No neural model was trained and no GPU was rented.

The archive contains six recordings. Added `roundT_doubleT` (252 scans, ~1.9 GiB extracted)
and extended `doubleT_platform` from its first 201 scans to all 345. The existing complete
`doubleT_obstacle` run contains 201 scans. Three other recordings remain outside the current
version's completed coverage. Historical runs of older versions are not substituted.

## Hybrid insertion model

`tunnel_guard.injection_experiment` reads the first ten real platform scans and runs a
fresh detector for each configured case, including an unchanged recorded control.
For each valid measured point, its direction from the configured sensor origin defines a
ray. An opaque axis-aligned box can replace a farther measured return with the first box
intersection. Returns sharing a point timestamp and direction quantized at 1e-6 are grouped; the nearest return in the group occludes the box. This grouping is an explicit proxy for beam identity, not verified firing calibration. Configured dropout removes the
intercepted return; it does not resurrect background hidden by the modeled box.

The original acquisition times are retained. No future scans supply inference evidence.
The experiment uses nine cases (90 processed scans): control; 0.3 x 0.3 x 0.4 m and
0.6 x 0.6 x 1.7 m boxes at 20/60/100 m; an off-axis 60 m box; and the upright 60 m box
with 70% return-slot dropout. Each lasts ten scans, about 0.9 seconds.

**Limits:** these are sensor-frame-fixed modeled boxes, not measured obstacles or
world-fixed trajectories. The explicit base z=-1.1 m is not surveyed ground. Valid-return
directions are conditional on what the original scene returned; missing emissions are
not reconstructed. Return multiplicity is retained, not independent beam evidence.
There is no reflectivity, material response, multipath, firing calibration or full sensor
simulation. Dropout is per return slot, not a physically calibrated beam model. The
background is unlabelled and cannot be called an empty scene. Timing/extrinsics and the
physical vehicle envelope remain unverified. No real accuracy follows from these cases.

## What is measured

- Modeled return slots and distinct inserted positions at 1 micrometre rounding.
- Surviving inserted representatives through range filtering, crop, geometry voxelization,
  background filtering and input to clustering. The membership tolerance is 1e-6 m;
  deskew must remain disabled for this coordinate-based accounting.
- Each candidate containing modeled support: inserted count/fraction, object confirmation,
  geometric relation and intersection confirmation.
- Frames with a target-dominated candidate (at least 50% modeled support), a confirmed
  target-dominated object, and a confirmed target-dominated intersection. This is an
  explicit diagnostic attribution convention, **not an independently validated matching
  rule or semantic recall metric**. Raw fractions are retained for inspection.
- `--sampling-only` distinguishes no measured ray through a box from occlusion by an
  existing nearer return. Neither proves the absence of an emitted beam.

The experiment retains source/config snapshots, exact source scan timestamps, data-file
identity, predictions, per-stage observations and actual cloud previews. The unchanged
control is compared with the preceding real run; whole-scene alarms are not attributed
blindly to the inserted target.

## Findings from the configured panel

- At 20 m the small box is a candidate in 10/10 frames, with object and intersection
  confirmation in 9/10; the upright box has both confirmations in 10/10.
- At 60 m the small box has only 0–4 inserted slots, corresponding to **0–2 unique
  positions**. These positions reach clustering input; only one frame has a
  target-dominated candidate, and none has object confirmation. Duplication must not be
  counted as evidence for lowering the minimum unique support.
- The upright 60 m box has 8–16 slots / 4–8 unique positions: target-dominated candidates
  in 10/10, confirmed objects in 9/10, confirmed intersections in 7/10.
- With 70% return-slot dropout, upright 60 m yields candidates in 7/10, confirmed objects
  in 5/10 and confirmed intersections in 3/10. This is one seeded trajectory, not a
  distribution of weather/material conditions.
- Both 100 m cases receive no modeled returns: there are no matching valid-return
  directions in these source frames. They do not measure the detector's 100 m capability.
- The off-axis box is heavily occluded by nearer recorded returns. This placement is
  not an adequate negative panel for judging false hazard alarms; absence of a candidate
  can be caused by missing visible support.

The useful next question is **whether causal accumulation can retain repeatedly observed
single-position candidates without promoting infrastructure fluctuations**. The current
panel provides controlled weak-support cases for that experiment. A calibrated beam
model or additional real viewpoints are required before extending conclusions to 100 m.
Simply increasing the number of generated points would change the question.

## Reproduction

Run from the repository root with the existing `.venv-iteration`. Output directories
are non-overwriting; preserve old runs and use a copied recipe for another experiment.

```sh
tar --zstd -tvf archive/for_hackathon.zst
tar --zstd -xf archive/for_hackathon.zst -C data/sourcecraft_subset for_hackathon/roundT_doubleT
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.injection_experiment --experiment configs/ray-insertion-expansion.json
.venv-iteration/bin/python -m tunnel_guard.injection_experiment --experiment configs/ray-insertion-expansion.json --sampling-only
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.injection_experiment --experiment configs/ray-insertion-obstacle-background.json
.venv-iteration/bin/python -m tunnel_guard.injection_experiment --experiment configs/ray-insertion-obstacle-background.json --sampling-only
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/coverage-expansion-real.json
MPLCONFIGDIR=/private/tmp/tunnel-guard-matplotlib .venv-iteration/bin/python -m tunnel_guard.expansion_report --real build/coverage-expansion-real --injection build/ray-insertion-expansion-grouped-returns --background-followup build/ray-insertion-obstacle-background-grouped-returns --control-baseline build/intersection-evidence-after/doubleT_platform.jsonl --output results/coverage-expansion.json --plot build/ray-insertion-expansion-grouped-returns/summary.png
git diff --check
```

Source snapshots are retained for each run. Sampling inspection and report generation
record their inputs separately. The detector is unchanged throughout. Actual/modeled cloud sheets are local at
`build/ray-insertion-expansion-grouped-returns/insertion-review.png`; aggregate charts at `summary.png`.
Magenta means modeled insertion, not a labelled real object. Orange rings show candidate
support, teal the estimated contour. The rendered frame is the last of each case;
complete temporal observations are in JSONL. Files with clouds and source bags stay local.

An adaptive follow-up keeps the upright box geometry at 60/100 m identical and changes
the source recording to the first ten `doubleT_obstacle` scans (control + two insertions,
30 additional processed scans). Its purpose is to distinguish source-scene sampling
limitations from a general range claim. It is not a blind holdout. Both backgrounds have
their own unchanged recorded control, compared with the preceding real-run predictions.

## Real extension: important counterexample to prefix-only conclusions

The complete platform run has **87 `obstacle` / 258 `unresolved_obstacle` frames**.
Only one confirmed-intersection frame was in the old 0–200 prefix; the remaining 86
are in frames 201–344. At frames 225 and 250, several far candidates have only 2–4
interior representatives but persistent temporal confirmation. Therefore the earlier
1/201 result is not representative of the entire recording. There are no verified
negative labels, so these counts cannot be called a false-positive rate. Confirmation
alone cannot resolve a persistent error in the reference contour.

Next implementation priorities are bounded experiments on (1) causal retention of
single-position tentative support, using unique new scans and conservative status;
(2) stable, supported rail/ground geometry on the expanded platform sequence. Keep the
new full real run fixed as a comparison panel. The other three supplied recordings
remain a coverage task; additional modeled scenarios do not replace them.

## Retained failed generator iteration

The initial per-return insertion model did not group simultaneous returns. Actual source
inspection found that it could place an object behind a closer return of the same modeled
beam: eight invalid intercepted slots over the platform off-axis case and 27 over the
obstacle-background 100 m case. That version's results are retained under
`build/ray-insertion-expansion/` and `build/ray-insertion-obstacle-background/`, but are
superseded by the `*-grouped-returns` runs. The generator now uses the nearest same-time,
quantized-direction return before insertion, and both complete synthetic panels were
rerun. These rejected results are not reported as accepted evidence.

## Completed evidence and comparison

The accepted hybrid runs contain **120 processed scans** (90 primary + 30 adaptive
follow-up); the superseded generator runs remain separate. On the obstacle background,
the corrected upright 100 m insertion has 28–34 return slots, whereas the platform
background has zero at the identical box coordinates. It produces a target-dominated
candidate in 5/10 frames and a confirmed object in 3/10, with zero confirmed intersections.
The 60 m upright box on this second background has 120–128 return slots and a confirmed
object in 9/10, also with zero confirmed intersections. Its relations include unresolved
and adjacent: the inserted box is not ground truth for the vehicle envelope. On frame 4
all 63 geometry-voxel representatives are removed by background filtering. This gives a
specific retention case to investigate without labelling the box a proven collision.
The corrected off-axis platform insertion is fully occluded in all ten frames.

| Real recording | Frames | obstacle | unresolved_obstacle | candidate | Motion-valid frames |
|---|---:|---:|---:|---:|---:|
| doubleT_platform | 345 | 87 | 258 | 0 | 240 |
| roundT_doubleT | 252 | 69 | 182 | 1 | 251 |

All 597 frames have algorithmically valid geometry and degraded health; valid geometry
is not verified geometry. In particular **105 platform frames fail the motion-validity
criterion**, so temporal accumulation cannot assume trustworthy alignment throughout.
Investigate those motion failures before promoting weak single-point histories.
The completed 201-frame obstacle baseline plus these recordings cover 798 real frames
on three recordings with the current detector. The remaining three are not yet covered
by this version's completed runs.

Both unchanged hybrid controls match their preceding real baselines on all ten frames.
The full platform's original 201-frame prefix also matches exactly for status, candidate
IDs, boxes, support/interior counts, object/intersection confirmation, geometric relation
and distance. This iteration adds observability and experimental coverage, **not a measured
improvement to the production detector**. Aggregate evidence and source hashes are in
`results/coverage-expansion.json`; no real precision or recall is inferred.

The real runner and both accepted insertion runners exited successfully. Real-run wall
times were 381.7 s (platform) and 386.8 s (round-to-double); median detector processing was
1072.8 ms and 1578.1 ms. Several local experiments overlapped, so these are not controlled
speed comparisons. No real-time or target-Intel claim is made. Static raw-data previews
for frames 0/100/200 were rendered and inspected; RViz GUI was not run. An initial standalone
preview invocation aborted with the macOS GUI plotting backend; the renderer now explicitly
selects headless Agg and the rerun succeeded. No automated test suite was run.
