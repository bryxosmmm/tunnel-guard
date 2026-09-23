# Tunnel Guard

Class-agnostic LiDAR obstacle-detection baseline for metro tunnels. Reads ROS 2 PointCloud2 bags directly; no ROS installation, Docker, or pretrained weights required for the default pipeline.

**Research baseline, not a validated collision-warning system.** Recall, infrastructure alarms, generalization, and runtime remain unresolved. `CASE.md` contains the original requirements. Recorded RViz2 export and a live ROS2 Humble adapter are implemented. The AMD64 Humble container processed a ten-scan real replay under Apple Silicon emulation; native target throughput and the RViz GUI remain unverified.

The initial review and its two real 30-frame prefixes are documented in [docs/AUDIT.md](docs/AUDIT.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Subsequent iterations and historical results below are separate evidence.

## Extended dataset: initial real runs

See [archive inventory and first comparison](docs/EXTENDED_FIRST_LOOK.md): 11,271 clouds in 221 segments, about 84 GiB unpacked. Three fixed segments (153 clouds) were processed by current and previous geometry without tuning. Both fail on the same one frame; no labelled accuracy is established. The initial sample extracted about 1.14 GiB; after disk cleanup, [the complete recording is now extracted and all acquisition headers audited](docs/EXTENDED_FULL_INGEST.md). [Continuous detector inference now covers all 11,271 clouds](docs/EXTENDED_FULL_RUN.md): 11,250 frames with supported geometry, 21 unavailable; median processing 132 ms on this Mac. No labelled accuracy is established.

## Usable runtime and full-corpus iteration

The [Gerasimov/HMM-MOS review and scenario runs](docs/REVIEW_GERASIMOV_20260918.md) add reproducible moving, stopped and appearing-object scenes for the actual detector. A first-rail-heading experiment was replayed on all 11,271 real clouds for geometry and 951 clouds end to end. It remains opt-in: nine geometry failures recovered, one new failure and unresolved path-selection changes. HMM-MOS is not used to suppress stationary obstacles.

The [3D track and clearance literature review](docs/TRACK_GEOMETRY_LITERATURE_20260918.md) maps published rail-pair estimation and local clearance coordinates to the remaining curve, grade and cant limitations. It distinguishes proposed adaptations from implemented and evaluated behavior.

The experimental `local_3d` rail frame was removed because classification had a Python-only fallback. The shipped `bed` frame remains the only detector mode; the historical experiments are retained in `docs/TRACK_LOCAL3D_ITERATION.md`.

See [run and review](docs/RUN_AND_REVIEW.md) for the local browser viewer and ROS2 launch commands, and [iteration evidence](docs/GOAL_ITERATION.md) for all six supplied recordings (2,488 scans), background repeatability, and remaining limitations. The browser shows original clouds, the reference corridor, candidates, confirmed intersections, distances and data quality. [Q&A implications](docs/QA_IMPLICATIONS.md) separates organizer statements from unresolved calibration assumptions.

## Native acceleration and calibration experiment

See [build, commands and evidence](docs/CALIBRATION_AND_NATIVE.md) and the [native integration report](docs/NATIVE_INTEGRATION.md). The detector has one decision path: the C++ extension, required by `configs/detector.json` and ROS container defaults. Build it with `python setup.py build_ext --inplace`; loading a detector recipe without it fails with that command. Historical performance results predate this consolidation and are not evidence of Python/C++ equivalence in this revision. Mounting calibration remains provisional: all three evaluated orientation candidates failed stability gates and were not installed.

## Native kernels follow the corrected contour

The corrected interval semantics above are implemented in **both** paths. The native classification kernel computes the same exact extrema of the piecewise-linear contour width over each point's height interval, in the same order, so the C++ recipe is not a frozen copy of the older rule. Verified two ways: the kernel and object checks compare the two recipes array by array (28/28 kernel checks, 20/20 geometry arrays, 624 and 550 candidate objects with no field mismatch), and the native recipe independently reproduces every corpus count the correction reports — `obstacle` frames 119 / 87 / 64 and 82 / 258 / 187 unresolved, identical to the review's table on 798 real scans. The three labelled panels are unchanged (development 162/54, holdout 159/57, measured 709/251), so the correction removes marginal confirmations in the unlabelled corpus, not in the annotated panels.

Their kernel-equivalence and corpus counts were measured on this branch's pre-merge recipes; the object counts move with the rail-heading and decode changes merged below, while the two-backend agreement is re-checked on the merged build.
The review's other finding — that the background model was queried for returns whose decision is never consumed — is now applied in both paths as well, and it is what closes the remaining latency gap: only segmentation-context returns that are not protected evidence are queried. No threshold changed.

## Measured rail anchors and corrected path geometry

The default recipes now fit paired rail heading and place anchors within actual measured support. See [implementation, real runs and limitations](docs/RAIL_GEOMETRY.md): 798 valid frames, 6093 supported anchors, improved withheld-point residuals on 8/9 saved clouds. Candidate grouping and alarms change; field accuracy and far-object recall remain unverified. The frozen previous recipe is `configs/detector-rail-baseline.json`.

## Reported 1.075 m mounting reference

See [railhead-support observations and chronological replay](docs/MOUNTING_REFERENCE.md). The reported empty/stationary height has unknown applicability to individual recordings. Direct support estimates are about 1.500 / 1.086 / 1.093 m across three recordings; no calibration is installed. All 798 compared detector outputs are preserved. A frozen-rotation replay on 738 later scans retains failures, including worse support in the round tunnel. The browser and RViz export now show the actual points supporting the estimate.

## Small detections and contour uncertainty

See [small-object review and calibration limits](docs/ENVELOPE_INTERVAL_REVIEW.md). Height uncertainty now propagates through the stepped contour width and both vertical boundaries. On 798 real scans, confirmed intersection observations changed from 718 to 649; small detections and nuisance alarms remain unresolved, and this is not a precision improvement claim. The browser exposes measured box sizes, support counts and exact diagnostic points for selected saved frames.

## Runtime reduction without reducing coverage

See [runtime profile and verification](docs/RUNTIME_CONTEXT_OPTIMIZATION.md). Avoiding unused background queries and repeated component scans preserved compared outputs on all 798 real scans. That pre-integration version measured 315–469 ms on the development Mac. See [native integration](docs/NATIVE_INTEGRATION.md) for current timings and remaining bottlenecks; target-hardware performance is unverified.

## Decode and complete offline latency

See [decoder preservation and latency scope](docs/DECODE_AND_LATENCY.md). Paired
real-cloud decoding decreased from 11.70 to 7.60 ms; all 174.5 million valid point
observations and normalized times matched exactly. The complete offline loop
measures 140–201 ms median across three recordings, including read/decode,
inference and result serialization. This is not live sensor-to-display latency.

## Coverage expansion and modeled insertions

See [coverage expansion](docs/COVERAGE_EXPANSION.md): complete platform and round-to-double tunnel runs, plus nine controlled cases on actual measured ray directions. Modeled support is traced through processing stages; missing rays, occlusion and candidate rejection are reported separately. Synthetic attribution is not field recall. The production detector is frozen for this experiment; its parameters were not tuned to inserted boxes.

## Review of new annotations

See [annotation review](docs/ANNOTATION_REVIEW.md) and [sensor evidence](docs/SENSOR_PROFILE.md). The teammate's original oriented person boxes cover frames 165–200; all positions were author-reviewed after detector-assisted propagation. A fresh person-specific replay and inspection finds the person's measured component on all 36 frames, one track and zero ID switches. The unchanged IoU gate passes 5/36 against exported AABBs and 14/36 against original oriented full-person boxes; these are localization scores, not person-detection counts. There are 32 confirmed non-target hazard observations on 27 of those frames, disjoint from the sole target person. Full evidence and limits are in the linked review; no detector behavior was changed.

## Latest real-sequence iteration

See [NEXT_ITERATION.md](NEXT_ITERATION.md): the complete 201-frame annotated sequence, stage-by-stage point evidence, and an optional envelope-support distance definition. All five provisional observations retained their localization; detection decisions stayed unchanged. This is a correction of distance semantics, not a measured detection-range improvement. The default recipe retains the original cluster-minimum distance; use `configs/iteration-envelope-distance.json` for the new mode.

## Latest alarm-cause correction

See [alarm-cause iteration](docs/ALARM_CAUSE_ITERATION.md). Lateral path uncertainty now
separates interior evidence from uncertain boundary crossings; unresolved points remain
candidates and appear orange in RViz. Two fixed structures retain their complete boxes
while losing unsupported certainty. A real candidate near 56 m remains detected. Across
402 real frames, definite-alarm frames decrease, but warning-free operation and field
false-alarm improvement are **not established**.

## External method review

See [HMM-MOS review](docs/HMM_MOS_REVIEW.md): the authors' own implementation of the IJRR moving-object
segmenter, built unmodified and run on our synthetic measured-pattern tunnel and on three real windows.
It segments objects that are genuinely moving (real walking person: 95% of its labels inside the one
hand-authored person box) and produced **0 labels on the static object in four synthetic cases at 15, 30,
60 and 100 m** - the class this project is scored on - because a state change is only counted for
occupied<->free transitions. Measured cost 0.22 s/frame and 0.30 GB at 60 m, 0.75 s/frame at 100 m, and
an empty tunnel at metro speed (1.5 m/frame) produced 70k false dynamic labels in 200 frames. Compact
numbers: `results/hmm-mos-probe-20260918.json`; recipes: `configs/hmm-mos-probe-*.json` and
`tunnel_guard/hmm_mos_probe.py`. Nothing in the pipeline was changed by this review.

## The input crop: a corridor-following window was tried, measured, and removed (2026-09-20)

The detector crops its input to a fixed 8 m lateral window in the *sensor* frame. The concern was that on a curve
the track itself leaves that window, so both the returns and the rail anchors that estimate the curve would be
discarded before classification. A corridor-following window — following the previous frame's remembered
centre-line, gated on its offset so a straight run kept the original crop — was implemented on 2026-09-19, then
silently disabled the same day when an unrelated commit about background-refit cadence deleted the state that
drives the gate while leaving the readers in place. **The README claimed a shipped, measured behaviour for a day
after it had stopped running.**

On 2026-09-20 the state was restored and the mechanism was measured on three instruments, off versus on:

| instrument | result |
|---|---|
| extended curved run, 5 splits, 255 frames | the gate engaged on **143 frames**; objects **+2.93 %**, confirmed hazards **+10.16 %** (3761 -> 4143), **zero frame status changes** |
| the two sourcecraft recordings | statuses essentially unchanged (one frame on `roundT_doubleT` moved `no_obstacle_observed` -> `unresolved_obstacle`); objects +0.55 % and +1.2 %; the measured-intrusion class `inside_heuristic_path_and_ground_interval` **unchanged** at 223 -> 223 and 45 -> 45 |
| analytic R=300 m arc, 120 cases | matched detections unchanged at **20 of 120** frames; unmatched hazards 5736 -> 5758 |

Cost 1–3 % per frame. The mechanism follows the previous frame's *extrapolated* centre-line, which is the least
reliable quantity this detector has on a curve, and the returns it adds arrive as ambiguous rather than as
intrusions. A numeric gate cannot robustly disable such a thing and leaving it silently dead is worse, so the
mechanism, its state and the `corridor_crop_threshold_m` key were **removed**. The fixed band is used
unconditionally, and removing it reproduced the crop-off replay on every decision field of all 402 frames.
Record: `results/corridor-crop-removed-20260920.json`. The earlier claim above this section — that the R=300 m
arc reported the on-track object as `obstacle` because of the crop — was **not reproducible** by us and is
withdrawn; the arc scene generates on the order of 4700 unmatched hazards per 120 frames, so it cannot resolve
this question either way.

## Review response (2026-09-19)

A teammate's review of this branch's corridor and performance work (on `origin/experiments/morev`,
`docs/REVIEW_GERASIMOV_20260919.md`, against `5587685`) found three P1 defects and three P2 ones. All were
verified against the code and fixed: a generator that made `prefetch_depth: 0` produce an empty run, a
corridor crop that re-clipped the widened window symmetrically, and a continuation covariance that omitted
the `2 d^3 Cov(a,b)` cross term whose sign makes it grow fastest with distance. The measurements, the fixed
panel and the corrected metric definitions are in
[docs/REVIEW_RESPONSE_20260919.md](docs/REVIEW_RESPONSE_20260919.md).

### Input topic

The node subscribes to `/lidar_points` by default. Our own recordings do not share one topic: `doubleT_obstacle`
publishes on `/sensing/lidar/hesai128/pointcloud`, while the tunnel recordings and the extended run use `/lidar_points`.
Run with `input_topic:=<the bag's topic>` when they differ - `ros2 bag info <bag>` prints it. If nothing arrives within
`input_timeout_s`, the node degrades to `unavailable` and now logs an error naming the configured topic, the point-cloud
topics that are actually present, and the parameter to restart with, so the cause is visible rather than silent.

### The audit behind those instructions

Every link of the delivery path was checked on 2026-09-19 and five defects were found and fixed - an unpublished fixed
frame, a mismatched input topic, an undeclared `tf2_ros` dependency, a build check that did not import the node, and the
absence of these container instructions together with the host-networking requirement. The checks that came back clean,
the exact commands to run first, and everything that remains unverified are in
[docs/DELIVERY_PATH_AUDIT.md](docs/DELIVERY_PATH_AUDIT.md).

## Running it in the container (2026-09-19)

The submission requires build and run instructions for the container, and the README carried none: it documented the
offline runner and the ROS adapter separately, but never the sequence an evaluator actually performs. That is now here,
together with the one requirement that is easy to miss.

```sh
# 1. build (the image installs rviz2 and tf2, builds the native extension and imports the node, so a missing
#    dependency fails the build rather than the demonstration)
docker build -t tunnel-guard .

# 2. run the detector. --network host is REQUIRED: the node discovers ROS 2 traffic over DDS on the host
#    network, so without it `ros2 bag play` on the host is invisible to the container and nothing arrives.
docker run --rm -it --network host tunnel-guard

# 3. on the host, find the bag's point-cloud topic and play it (topics differ between recordings:
#    the tunnel recordings use /lidar_points, doubleT_obstacle uses /sensing/lidar/hesai128/pointcloud)
ros2 bag info <bag>
ros2 bag play <bag>
```

If the bag's topic differs from the node's default, run the node with the override - inside the container, or through the
launch file, which exposes the same parameters:

```sh
python3 -m tunnel_guard.ros_node --ros-args -p input_topic:=/sensing/lidar/hesai128/pointcloud
ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py input_topic:=/sensing/lidar/hesai128/pointcloud rviz:=true
```

`rviz:=true` starts RViz inside the container with `rviz/tunnel_guard.rviz`, which shows measured points, the reference
envelope, candidates, confirmed intrusions and their distances. For a machine without a display, run `rviz2` on the host
instead: it subscribes to the same `/perception/...` topics over the shared DDS network. Use slow replay
(`ros2 bag play -r 0.3 <bag>`) for a complete evaluation: the queue is depth 1 and the offline frame time is above the
10 Hz stream rate, so fast replay drops scans by design rather than silently.

Without ROS at all, the same detector runs offline and writes per-frame JSON with the measurement timestamp:

```sh
uv run python -m tunnel_guard.run --experiment configs/<recipe>.json
```

**What is verified and what is not:** the container path has been exercised only for a ten-scan replay under emulation,
and the changes of 2026-09-19 in it - the published fixed frame, the input-topic error message, the declared tf2
dependency and the build-time node import - are standard usage that this development machine cannot execute, because it
has no ROS 2 and no Docker daemon. They must be confirmed inside the container before the demonstration.

## Demonstration path, checked statically (2026-09-19)

`rviz/tunnel_guard.rviz` and the node were consistent on topics - the config listens to `/perception/points_display` and
`/perception/debug_markers`, which the node publishes - but the config's fixed frame, `tunnel_guard_local`, was published
by nothing, so RViz would come up with a missing fixed frame and render nothing. The node now publishes an identity
static transform from `tunnel_guard_local` to the frame the incoming clouds declare, once per source frame: the frame is
the sensor frame its outputs are already expressed in, not an invented one, and mounting and extrinsics stay unverified.
This change is standard tf2 usage but is **not exercised here** - no ROS 2 and no Docker daemon on the development
machine - so it must be confirmed inside the container before the demonstration.

## Where the frame time goes (2026-09-19)

Measured on `doubleT_obstacle`, per-frame medians, before the two shipped runtime changes:

| stage | ms | note |
|---|---:|---|
| background model construction | 40.3 | about 22 windows x 3 RANSAC planes; now refitted every metre of travel instead |
| geometry estimators | 54.7 | of which rail-pair refinement 13.7, surface normals 7.3 (they feed background removal), plane fit 4.3 |
| candidate clustering | 20.3 | over roughly 21k context points |
| KISS-ICP motion | 18.1 | registration, now 8 threads |
| association | 10.1 | per-track bookkeeping, not the assignment (measured by decomposing it) |
| classification (3 calls) | 8.0 | |
| input crop and voxel pass | 5.6 | |
| frame, total | ~123 | |

Per-frame medians, measured configuration by configuration, p50 on each recording:

| configuration | `roundT_doubleT` | `doubleT_obstacle` |
|---|---:|---:|
| before the 2026-09-19 changes | 122.2 | 134.7 |
| + the object-chain filter | 133.6 | 146.8 |
| + eight registration threads (**shipped**) | **114.6** | **110.7** |
| + the background refit cadence (opt-in, off) | 114.4 | 105.9 |

So the object-chain filter costs about 9 per cent, the threads return 14 and 25 per cent, and the cadence is worth nothing on
one recording and 4 per cent on the other - an earlier entry claimed 12 and 26 for it, comparing against a configuration that
also differed in thread count and in the chain filter, and that claim was wrong. Net of the shipped changes: 6 per cent on the
tunnel recording and 18 per cent on the station recording. Latency here is offline processing time
on an Apple M4 while the machine is shared; the deployment stand is an 8-core i7-9700E, so these figures bound the
shape of the budget rather than the field number. Two runtime ideas were measured and rejected rather than assumed: a
lateral band for the rail estimator (its 4 m search band is deliberate for rail-pair selection, and narrowing it moved
an anchor 3.6 cm) and a decomposition of the gated assignment (identical output, no gain).

## The corridor claim (2026-09-22)

The detector's alarm was its own crop floor. On all six organizer recordings the reported
`nearest_obstacle_m` had a **median of 2.00 m** - the value of `min_forward_m`, the near edge of the
input crop - and the object behind it was the tunnel's own surface: a component of 1253 voxels (median;
p90 6984) spanning 4-16 m along the track with **zero** voxels certified inside the reference contour.
`candidate` and `no_obstacle_observed` never appeared once in 3763 real frames. Four rules interacted:

1. **The boundary class asked the wrong question.** `possible` accepted a return whose *uncertainty
   interval* reached the contour whether or not its nominal position was inside, so the tripwire was a
   band on both sides of the contour edge. The document always said the class means "nominal position
   inside, interval not".
2. **A component's relation was an OR over its voxels**, so one return in that band made a
   thousand-voxel surface a hazard.
3. **Instant confirmation read the whole component's density and height**, not its interior support, so
   any dense wall or arch fragment confirmed on a single frame whatever its relation to the corridor.
   Measured, that is what put a 37-voxel component at 52 m into `nearest_obstacle_m` with one interior
   voxel.
4. **The infrastructure rule could not demote the head of a chain**: it required the chain to continue
   12 m beyond the candidate on *both* sides, so the nearest member - the one that sets
   `nearest_obstacle_m` - was exempt by construction. Measured: a chain at the same lateral existed in
   340 of 345 frames, its recorded span began at the crop floor, and no demoted member ever sat there.

What changed:

- **A claim needs a measured coordinate.** Where the bed or the centre-line is beyond its own
  uncertainty budget the lateral coordinate is our own extrapolation, and it cannot carry a claim on
  the corridor. Measured: 94-98 % of nominal-interior returns on real frames sit beyond that horizon,
  and they are what made the tunnel's own arch read as an intrusion at 60-200 m. Those objects are
  still reported, with their distance and their own `far_field_lateral_bound_m`, and the frame states
  how far its lateral reference reaches.
- **A claim needs to be an object, not the tunnel.** Every return is tested against the tunnel's own
  cross-section of the same scan, in track coordinates: longitudinal support (a coarse cell occupied
  over a span of at least `structure_min_length_m`), and outward reach (at that height the occupied
  region runs outward past the contour's half-width). Nothing is removed from the object pool - a
  return the contour clips at its edge is still reported, with the reason `structure_crossing_envelope`.
- **Doubt is a state, not an alarm.** Measured edge-uncertain evidence that is neither the tunnel's
  cross-section nor attached to it (further than `structure_isolation_m`) makes the frame `candidate`.
  Attachment is what separates a few returns of the tunnel's own surface that the cell test could not
  hold from an object standing in the corridor.
- **Instant confirmation is interior geometry**, as its own documentation always said.
- **The distance is the distance to the claiming evidence**: the train meets an object where the
  object's support first lies inside the contour, not at the nearest point of a cluster that extends
  out of the envelope.
- **A rejected registration no longer clears the history**; it marks the frame positionally uncertain,
  and the measured registration residual becomes the pose uncertainty instead of being compared to a
  threshold and discarded. Measured: the rejection is the tail of one continuous residual distribution
  (accepted median 0.21 m, rejected 0.36 m against a 0.30 m gate; the overlap gate never fires), and it
  cost a 91-frame stretch of `doubleT_platform` in which the train travelled 25 m.
- **The trailing rail-pair anchor is guarded** against contradicting the prefix it is fitted from,
  because the continuation past the last anchor is fitted in that anchor's frame and one inconsistent
  trailing anchor decides the sign of the extrapolated curvature.

Measured on recorded frames, no tuning to any recording. Reproduction: run the tracked gate recipe
(`configs/real-gate-6tunnels.json`) and the extended recipe, then
`python -m tunnel_guard.corridor_report --before-six build/notes-review/real6 --after-six <run>
--before-extended build/notes-review/extended --after-extended <run> --output
results/corridor-claim-20260922.json --markdown`. Artifact: `results/corridor-claim-20260922.json`.

| | before | after |
|---|---|---|
| six tunnels, frames | 2488 | 2488 |
| `obstacle` / `unresolved_obstacle` | 305 / 2181 | 202 / 54 |
| `candidate` / `no_obstacle_observed` | 0 / 0 | 14 / 2216 |
| hazard observations | 62738 | 515 |
| nearest hazard support, median | 1253 voxels | 17 voxels |
| `nearest_obstacle_m`, median | 2.002 m | 55.914 m |
| extended corpus, `obstacle` / `unresolved` | 33 / 1241 | 7 / 32 |
| extended corpus, `candidate` / clean | 0 / 0 | 5 / 1230 |
| extended corpus, hazard observations | 26683 | 121 |
| extended corpus, `nearest_obstacle_m`, median | 2.004 m | 46.617 m |

Costs, stated rather than absorbed:

- **Three frames of the labelled recording lose `obstacle` status** (169 -> 166). Frame 75 is the one
  the trailing-anchor guard costs: its detection rests on an anchor with support 4 that contradicts six
  better-supported anchors, and it is the only frame in 201 where the guard changes the answer. Frames
  141 and 177 are frames where the certified intrusion is three voxels, so instant confirmation - which
  now reads interior geometry - does not fire; both frames still report the object at 56.4 m, one as
  `candidate` and one as `unresolved_obstacle`.
- **The tunnel's own cross-section is measured per scan, not from a map.** Its resolution limit is
  stated in `docs/DETECTOR.md` section 13: an object within about 0.2 m of the tunnel's own surface at
  the same height is indistinguishable from it by returns alone, and where the tunnel's cross-section
  *changes* along a recording (a recess, a section transition) a cell that holds for metres elsewhere
  does not hold there, which leaves a few returns unexplained. Route memory - a per-station profile
  accumulated over passes - is the honest next step and is not implemented.
- **The reference contour is still GOST M, not the organizers' draft 2.1 m x 3 m.** The draft profile
  is narrower than any metro car body (2.7 m), so narrowing the safety criterion on an ambiguous
  statement would call in-envelope things clear. It is kept as a sensitivity arm
  (`configs/detector-qa-provisional.json`), identical to the shipped recipe in every other key, and it
  is measured: on the same 2488 frames it reports `obstacle` 112 instead of 202, `unresolved_obstacle`
  25 instead of 54, and 2344 clean frames instead of 2216. On the labelled recording it reports 102
  `obstacle` frames instead of 166. **Attribution corrected 2026-09-24:** these are frame-level
  hazard counts, not detections of the labelled person. The current person-specific review finds
  that person's component `adjacent` on all 36 labelled frames. The earlier claim that these counts
  demonstrated loss of the labelled person is withdrawn. Choosing the physical reference contour
  still requires vehicle dimensions and calibration, not optimization against total alarm counts.
- **Runtime is unchanged.** On an idle machine, 30 frames of `doubleT_platform`: processing p50 128.3 ms
  against 131.6 ms for the recorded revision, of which the cross-section test is 8.6 ms per frame and
  the odometry 27.1 ms. The 500 ms readings taken while the six-tunnel and extended-corpus runs were
  executing are contention, not cost.

## Two shipped behaviours measured on 2026-09-19

Both are in `configs/detector.json`, the recipe the ROS container defaults to, and both were enabled only after
their own gate.

**The tunnel's own structures leave the hazard list, but never a measured intrusion.** A duct, cable tray or walkway
edge reaches one scan as a chain of small fragments at one cross-section position, repeated along the whole scan, and a
fallen object is one cluster at its own position. The chain is built from the candidate objects themselves, each placed
by its own support, so nothing inherits a neighbouring structure's span; a candidate whose position recurs beyond it
both fore and aft is the tunnel, not an object. Measured: unresolved objects 2757 -> 1323 on `roundT_doubleT` and
3231 -> 1754 on `doubleT_obstacle`, and the full 1460-frame measured-pattern panel net-identical
(709/251/176, event recall 0.8021, matched IoU 0.8223, zero empty-scene alarms). Recipe key `infrastructure_continuity`.

**SUPERSEDED 2026-09-22.** This object-chain rule is gone. Measured afterwards, it could not demote the
nearest member of any chain at all - the rule demanded the chain continue `reach_m` beyond the candidate
on *both* sides, so the head of a chain, which is the member that sets `nearest_obstacle_m`, was exempt by
construction - and it was the wrong layer anyway: the question "is this return part of the tunnel" is a
question about a return, not about a candidate object. It is replaced by the per-return cross-section test
described under [the corridor claim](#the-corridor-claim-2026-09-22). The chain rule's own history above is
kept because its two failures are what motivated the replacement.

That gate looked safe because 52 of 52 labelled obstacle frame statuses survived it, and that check was too weak:
in 2026-09-20 the chain was measured to be *erasing* judgements rather than structures. On the labelled obstacle
recording 30 objects on 21 frames, and on the tunnel recording 21 on 13 frames, held at least `weak_min_voxels` core
voxels **inside** the swept contour - a measured corridor intrusion - and were relabelled `adjacent` and
`longitudinally_continuous_structure` only because a long chain happened to share their cross-section position. A frame
status survives that: the frame was already `obstacle` for another object. A candidate whose own support lies inside
the contour is therefore no longer demoted whatever repeats beside it; `unresolved` and `adjacent` candidates are still
demoted, because for them repetition is the evidence that distinguishes structure from an object.
Result: demoted-while-inside 30 -> 0 and 21 -> 0, object counts unchanged on every frame of both recordings, one frame
(`doubleT_obstacle` 171) recovered from `unresolved_obstacle` to `obstacle`, the 1460-frame panel byte-identical on
every metric, and no measurable cost: in a back-to-back A/B with the module swapped and restored on one machine state,
processing p50 was 146.03 -> 145.47 ms on `doubleT_obstacle` and 120.17 -> 119.92 ms on `roundT_doubleT`.
The restored objects sit at the contour edge (lateral 1.29-1.31 m against a 1.32 m
half-width) and are plausibly the walkway itself; they are reported because a reference contour that is not validated
as a vehicle swept volume cannot justify discarding measured interior support. Details and artifacts:
`results/infrastructure-continuity-guard-20260920.json`.

The panel figures quoted above come from the recipe in use on 2026-09-19 (350 m input ceiling, histogram rail centre,
`hazard_requires_certified_path` on); the current production recipe on the same 1460 cases and labels scores
699/261/153, event recall 0.78125, matched IoU 0.8205, and `results/metrics-blockers-20260920.json` audits the
difference. Comparing a change against the older artifact therefore compares two recipes, not one.

**The background model CAN refit by distance travelled instead of every frame - opt-in, and off by default.** It claims points from the tunnel's own
longitudinal surfaces — lining, walls, ducts, bed — and those run parallel to travel, so their plane equations in the
sensor frame barely change between frames a fraction of a metre apart. Building it was the largest single cost in a
frame (40.3 ms of 123 ms, about 22 overlapping windows of three RANSAC planes each), so it is now reused until the
pose has advanced `background_refit_travel_m` (1 m), which also means a stopped train refits nothing. Measured on the
shipped recipe: `roundT_doubleT` 130.7 -> 114.4 ms and `doubleT_obstacle` 143.4 -> 105.9 ms, i.e. 12 and 26 per cent,
with identical frame statuses and the labelled obstacle preserved. **CORRECTED:** measured against the same configuration with
the key off, the effect is 114.6 -> 114.4 and 110.7 -> 105.9, i.e. nothing and 4 per cent; the 12 and 26 figures compared
against a configuration that also differed in thread count and in the object-chain filter. With a few per cent at stake the
case for keeping it off while it voids the backend-equivalence guarantee is stronger, not weaker. Its gate held every metric on an identical-case
panel run (tp 182, fn 106, fp 40, precision 0.8198, event recall 0.7917, matched IoU 0.8114, zero negative episodes).
Stated trade-off: staleness mis-places the bed on a grade by about 5 mm per metre of travel against the 25 mm fit
distance, which is the margin the 1 m limit keeps.

Confirmed end to end on 2026-09-19: the full 1460-frame measured-pattern panel run with the shipped changes returns
**every metric byte-identical** to the pre-change baseline (tp 709, fn 251, fp 176, precision 0.8011, event recall
0.8021, matched mean IoU 0.8223, distance MAE 0.00111 m, zero empty-scene alarms, identical recall at every range from
10 to 300 m) while the panel's wall time falls from 430.4 s to 316.5 s - a quarter of the runtime for no change in what
the detector decides. The panel's own declared criterion, event recall >= 0.95, remains unmet at 0.802 before and after,
and is reported here as the standing gap it is.

Both rest on measured limits rather than assumptions: the far-field lateral frame is a sensor property of this route
(rails vanish by 90 m, the bed band is empty beyond 70 m, partial-arc cross-section fits are ill-conditioned beyond
80 m), so objects beyond the corridor horizon are reported as unresolved candidates with their distance and their own
lateral uncertainty, never as a certified clear path. Details, including six approaches measured and rejected for that
class, are in `results/empty-tunnel-alarm-load-20260919.json`,
`results/empty-tunnel-alarm-load-attempt2-20260919.json` and `results/background-refit-cadence-20260919.json`.

## Curves: the full investigation

Two defects, not one: the corridor continued straight past the measured rails, and the detector cropped its
input to a fixed sensor-frame window that on a curve discards the track and its anchors before
classification. Both are fixed and measured — on curved scenes the shipped configuration reports the object
in **16 of 24 cases against 12 before**, with **82% fewer spurious objects**, while the straight-rail panel
stays byte-identical. The far field is an information limit, with six strategies tested and five rejected on
their own numbers. Everything, including what is *not* established, is in
[docs/CURVED_CORRIDOR_AND_RANGE.md](docs/CURVED_CORRIDOR_AND_RANGE.md).

### The alarm decision this leaves

Whether unmeasurable far-field evidence should read as an alarm is a decision, not a defect: six approaches failed to
separate the tunnel's own far surfaces from objects on this data. Both sides are measured - 211 hazard-class objects over
1460 panel frames under an event-scored convention, `unresolved_obstacle` on 59 of 60 frames under a status-scored one - and
the two ways to demote it each cost the 100 m detection tier. The three options, their measured costs and a recommendation
are in [docs/FAR_FIELD_ALARM_DECISION.md](docs/FAR_FIELD_ALARM_DECISION.md).

## What the corridor can and cannot reach (2026-09-19)

Measured, not assumed. The **bed** is sampled to 65-105 m (13-18 anchors per frame - the floor is wide), so
heights above the running surface are known far out; inside its 15 m gate the linear bed extrapolation errs by
<=0.02 m even where the vertical curvature is R_v ~ 7 km. The **lateral** track centre is the binding unknown:
rail returns collapse 1995 -> 105 -> 17 -> 0 per 20 m bin from 10 m to 90 m, and the tunnel bore is a biased
proxy - robust circle fits to perpendicular slabs (16-23 slabs to 105-165 m, conditioned centre sigma
0.003-0.011 m) sit about a metre off the track centre, and calibrating that bias on the rails still predicts
only 1.32 m at 100 m.

The corridor's reach is therefore set by an uncertainty budget, `path_max_uncertainty_m` (0.4 m), which the
existing heuristic sigma `0.06 + 0.008 r + 0.0003 r^2` reaches at **22.9 m** past the last anchor - a ~63 m
horizon at the measured median last anchor of 40.0 m, which is the median `supported_range_m` of 62.5 m
reported on the six recordings. (This section said 33.7 m and ~74 m until 2026-09-22; that was arithmetic
from an older sigma and the code never had such a horizon. Nothing downstream changed: the horizon is
where the budget stops, not a tunable.) `path_max_extrapolation_m` does not bind: raising it 25 -> 45 m
changed no classification at all. Raising the *budget* to 0.7 m would reach 86 m but was measured and rejected:
it turns two frames of `roundT_doubleT` into certified obstacles (intersecting 12 -> 20) in a band where the
centre is uncertain by ~1.0 m, for no measured gain. Fitted-curvature sigma propagated from the anchor window is
over-confident by 1.6x at 50-60 m and 4-8x at 80-150 m, so the heuristic term is the calibrated model.
Every object record now carries `far_field_lateral_bound_m`: the path uncertainty the classifier itself used at that object's distance - `sqrt(base^2 + extension^2)` with `base = 0.06 + 0.008r + 0.0003r^2`, calibrated to 0.46-1.34x the measured centre error over 10-110 m of extrapolation - or `null` beyond the modelled horizon, where no bounded claim exists. On 200 real frames this changed no status, no nearest distance and no object identity: 11,904 of 15,041 observations on `roundT_doubleT` carry a finite bound (max 0.45 m) and 3,137 state that their lateral track relation is unknown. A far-field detection therefore states what it does not know instead of implying that an unmeasured corridor is clear. Full evidence: `results/alignment-long-lever-20260919.json`, `build/uncertainty-calibration.json`.

## Reference corridor follows the curve

The reference contour used to continue past the last measured rail anchor along a straight tangent
whose slope was clipped at `rail_max_heading`, modelled for `path_max_extrapolation_m` beyond the
nearest anchor. Rails here are supported to a median 40 m, so the contour had a modelled horizon near
65 m — and inside it the true track leaves a straight line quadratically. Measured against what later
frames of `roundT_doubleT` see over the same ground, that continuation was 0.20 m off at 40 m, 0.47 m
at 50 m and 1.02 m at 60 m (p90 1.41 m), against a corridor half-width of 1.535 m.

`TrackGeometry._continuation` and its native mirror now continue along a local quadratic fitted to the
anchors inside `path_curve_window_m`, expressed in the edge anchor's frame so the centre-line stays
continuous there, with the curvature shrunk to zero unless it exceeds `path_curvature_significance`
standard errors (default 4), and the extrapolation uncertainty taken from the fit covariance in
quadrature with the previous base term. Inside the anchor span the corridor is bit-identical to before;
the model only acts beyond it. Forward-prediction error becomes **0.045 / 0.136 / 0.203 m** at 40 / 50 / 60 m.
Both backends stay identical (20/20 geometry arrays, 29/29 kernel checks), and 60-frame prefixes of
`roundT_doubleT` and `doubleT_obstacle` keep their statuses; the object set moves slightly
(14,305 → 14,014 on the straight recording, intersecting observations 64 → 70), which no available
label can adjudicate. `path_curve_window_m: 0` reproduces the previous continuation exactly.

The gate is not cosmetic. On the measured-pattern panel, whose rails are straight by construction, a
weaker 2-sigma gate bends the corridor off a geometry the old model already had exactly right:
precision 0.8011 -> 0.7647, tp 709 -> 702, fp 176 -> 216, and empty scenes start alarming
(negative-episode rate 0 -> 0.08). At 4 sigma the same panel is byte-identical to the baseline
(709/251/176, precision 0.8011, zero empty-scene alarms) while the real-curve prediction gain above is
kept, so 4 is the shipped default and `path_curvature_significance` is the knob to loosen deliberately.
Both panel runs are retained as evidence. Straight recordings are still not bit-identical — the fitted
slope replaces the noisy two-point tangent even where curvature is shrunk, which moves one frame of 60
from `unresolved_obstacle` to `candidate` on `doubleT_obstacle`.

This does **not** extend the modelled horizon: beyond anchors + 25 m the corridor is still `unknown`,
which is why the scored 100–300 m band needs a long-lever estimate. Walls and ceiling do return to
120–207 m on the curved recording, but per-bin medians of those returns are not an axis — they jump
5–9 m with platform edges — so that estimator has to be built on surface strips and validated with the
same forward-prediction test before it is allowed to widen the corridor. Evidence:
`results/curve-continuation-20260919.json`.

## Reader overlap in the offline runner

The bag reader (decompression plus PointCloud2 decode) ran between frames: 38.6 ms p50 on
`doubleT_obstacle`, 12.8 ms on `doubleT_platform`. One bounded producer thread now reads ahead by one
scan while inference runs. On the same 30-frame protocol with the same recipe,
`read_and_process` p50 falls 164.7 → **126.7 ms** and 118.8 → **105.2 ms**, and p95 185.2 → 142.0 ms,
with every compared field identical (status, nearest distance, each object's track id and distance, 60
frames). Inference is untouched (125.6 → 126.5 / 105.5 → 104.7 ms).

This removes the reader from the critical path; it does **not** shorten the age of a decision, and
inference at ~126 ms per frame still exceeds the 100 ms input period, so 10 Hz per frame is not met.
Recorded result: `results/reader-overlap-20260919.json`.

## Team work

See [next iteration assignments](docs/TEAM_TASKS.md): reviewed episodes, sensor/time evidence, Ubuntu/RViz validation, and oriented evaluation. Use separate branches from `experiments/morev`.

## Quick start

Python 3.10+; this iteration used native macOS Python 3.12.8 and container Python 3.10 (Humble). Historical audits used other versions. Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked
uv run python -m tunnel_guard.run --experiment configs/evaluation-audit.json
```
Agent policy lives in `AGENTS.md`: no subagents or automated tests. Verify changes through actual detector runs and configured evaluations; the repository intentionally has no test suite.

The audit recipe requires the two real bags described below. It processes the first 30 consecutive frames of each and writes JSONL, configuration, source snapshots and RViz result bags to `build/audit-reviewed/`. Set `visualization` to `false` in a copied experiment JSON for headless processing; detector decisions do not depend on the display consumer. No model download is needed at runtime.

**Output directories must not already exist.** To repeat a run, copy its experiment JSON and change `output`; do not delete evidence merely to rerun. Nix users can optionally use `nix develop`; everyone else can ignore `flake.nix`, `flake.lock`, and `.envrc`.

## Run the real bags

Obtain the organizer's dataset separately and put the extracted directories here:

```text
data/sourcecraft_subset/for_hackathon/
  doubleT_obstacle/
  doubleT_platform/
  roundT_doubleT/
  roundT_pressureGate_roundT/
  roundT_squareT_pressureGate_squareT/
  squareT_platform_squareT_switch/
```

Each directory must contain its `metadata.yaml` and SQLite `.db3` files. Recordings and dataset archives are deliberately not included in Git.

For the organizer archive supplied as `archive/for_hackathon.zst`, the two audit bags can be extracted without unpacking the whole dataset:

```sh
mkdir -p data/sourcecraft_subset
tar --zstd -xf archive/for_hackathon.zst -C data/sourcecraft_subset \
  for_hackathon/doubleT_obstacle for_hackathon/doubleT_platform
uv run python -m tunnel_guard.inspect_bag \
  data/sourcecraft_subset/for_hackathon/doubleT_obstacle \
  --max-frames 30 --output build/input-inspection.json
```

The original full evaluation below requires all six bags:

```sh
uv run python -m tunnel_guard.run --experiment configs/evaluation-quality.json
uv run python -m tunnel_guard.evaluate \
  --run build/tunnel-guard-real-quality \
  --annotations annotations/sourcecraft-provisional.json \
  --output build/tunnel-guard-real-quality/localization.json
```

The five supplied annotation boxes describe **one provisional upright structure**, not independent verified hazards. They are nonexhaustive: real precision cannot be computed from them.

## Label the recordings in SUSTechPOINTS

Reviewed person labels already exist for `doubleT_obstacle` frames 165–200; a complete independently reviewed panel across all six recordings does not. Use the existing labels rather than starting them again. The upstream [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS) annotation tool supports further work: its source is vendored under `SUSTechPOINTS/` at upstream revision `50fa188`, with this project's taxonomy patch already applied (the delta is kept in `patches/`). What is not tracked, matching the tool's own rules and this project's data rule, is its `data/` directory, its virtualenv and the 14 MB model release.

```sh
cd SUSTechPOINTS
python3 -m venv .venv && .venv/bin/pip install -r requirement.txt
wget https://github.com/naurril/SUSTechPOINTS/releases/download/0.1/deep_annotation_inference.h5 -P algos/models
cd .. && uv run python -m tunnel_guard.sustech_import --experiment configs/evaluation-quality.json
```

`tunnel_guard.sustech_import` writes one `.pcd` per bag message into `SUSTechPOINTS/data/<bag>/lidar/`, keeping `tunnel_guard_local` coordinates, intensity and full density, and an empty `label/` beside it. Frame `00000N.pcd` is bag message index `N` — the same index the run JSONL uses, so detector output and labels refer to the same frame. It reproduces the current scene files byte for byte (checked on three frames), and it refuses to overwrite an existing scene. Start the tool with `python main.py` inside `SUSTechPOINTS` and open <http://127.0.0.1:8081>; `server.conf` listens on `0.0.0.0`, so it is also reachable over a private network such as Tailscale (the tool has no authentication).

Boxes are authored by hand: the detector's candidates are not good enough to seed a panel, and labels a detector supplies for its own scoring cannot measure that detector. After a labelling pass, convert the tool's label files into the schema the evaluator validates:

```sh
uv run python -m tunnel_guard.sustech --config configs/annotation-export.json
```

`configs/annotation-export.json` requires the reviewer to declare which frames are exhaustively labelled, including frames with no object: those become the scored negative frames, and `tunnel_guard.evaluate` counts nothing else as a false alarm. Rotated boxes are exported as their axis-aligned envelope, which is what the IoU matcher consumes.

### What is labelled so far

`annotations/doubleT-obstacle-person.json` holds the one object we have: `doubleT_obstacle` frames 165–200, one person-sized box, `event_id 7`, `class Person`. The box was authored by hand at frame 181 and propagated over the other 35 frames by a constant-velocity fit of the detector's own track of that object — it recedes at 0.31 m/frame along +x while the tunnel itself stays fixed in the sensor frame (a tracked ceiling fixture moves 0.001 m/frame), so the motion is the object's, not the train's. Every frame is `exhaustive: false`: this records where one object is, not that the frames contain nothing else, and "person" is the reviewer's judgement.

`annotations/sustech-raw/doubleT_obstacle/*.json` is the authoritative form of the same 36 frames, copied verbatim from the annotation tool: `position`, `rotation` and `scale` per box. The evaluator's schema has no rotation field, so the export stores the axis-aligned envelope of the rotated box and inflates the footprint — for this box, yaw 0.406 rad widens x by 43% and y by 22%. Comparing the label with the detector's own box for the same object over those frames: centre offset median 0.48 m, label/detector extent ratio 1.73 × 1.63 × 1.29 at true scale, median IoU 0.196 (6 of 36 frames at or above the 0.25 gate), falling to 0.147 (0 of 36) through the envelope conversion. The label is a full-person volume; the detector's box is the support of its returns. Those are different measurement conventions, and the mismatch is not evidence about either method's correctness.

### Objects inside the clearance envelope

`tunnel_guard/on_track.py` finds intrusions the way the detector should: only returns that actually fall inside the GOST contour are clustered, and a cluster is dropped when it is a face of a large surface, or when its lateral/vertical profile runs continuously or repeats along the tunnel — cable runs, linings, trays and posts. **Evidence status - the aggregate formerly quoted here is NOT reproducible.** It read *11,115 in-envelope clusters → 381 events →
15 candidates* over the six recordings, with no artifact in this repository, and a re-measurement with the same command does not
reproduce it. Measured so far, four of six recordings (201 to 877 scans each):

| recording | clusters | events | candidates |
|---|---:|---:|---:|
| `roundT_doubleT` | 215 | 17 | 0 |
| `roundT_pressureGate_roundT` | 177 | 18 | 1 |
| `squareT_platform_squareT_switch` | 391 | 9 | 0 |
| `doubleT_platform` | 1030 | 16 | 0 |

That is 1,813 clusters and 60 events across four recordings, so the quoted 11,115 clusters and 381 events are roughly six times
what the probe produces on this data, and the 15 candidates are not reproduced either (1 measured). The shape of the result holds -
tens of events and a handful of candidates out of hundreds to thousands of clusters - but the numbers are treated as stale. The two
remaining recordings are unmeasured; the re-measurement is `results/on-track-counts-20260919.json`.

```sh
uv run python -m tunnel_guard.on_track \
  --bag data/sourcecraft_subset/for_hackathon/doubleT_obstacle \
  --config configs/on-track-probe.json --output build/on-track-events.json
```

Thresholds are in `configs/on-track-probe.json`. The strongest candidate is `doubleT_obstacle` around x ≈ 55.7 m, y ≈ −0.5 m: a compact 0.45 × 1.02 × 1.35 m mass standing on the bed, isolated from any surface, present in frames 2–75 and then gone. Two limits matter. Geometry cannot separate a permanent fixture that pokes into the contour from a foreign object — the ~34 m hit in the same recording is a post running through the whole recording. And on the `roundT_*`/`squareT_*` curves the track-centre estimate drifts laterally, putting 139 of 169 events in one recording at |y| = 1.6–3.2 m while the GOST half-width never exceeds 1.535 m; `report.max_lateral_m` drops those, which is why the count is 15 and not 381. That same drift is a large part of the detector's alarm load.

The surviving candidates are also written into the tool as separate scenes (`<bag>_candidates`, with `lidar` symlinked to the real clouds) so they can be stepped through without touching the human labels.

The tool's own auto-annotation is not usable for this data: `GET /auto_annotate` returns HTTP 500 because `annotate_file` is defined inside an `if False:` block and its clustering binary and discrimination model are absent, and `/predict_rotation` feeds uncentered coordinates to a model trained on centred object crops, so its angle barely depends on the selected object.

### Classes

The patch in `patches/` makes `public/js/obj_cfg.js` carry a metro taxonomy in place of the upstream driving classes: `Person`, `ForeignObject`, `Equipment` (things that must not be on the track), then `PlatformEdge`, `PressureGate`, `TrackSwitch`, `TrackFixture`, `Cable`, `WallLining` (content the recordings actually contain), then `Unknown` and `DontCare`. The same names are repeated in `tools/check_labels.py` (server-side `/checkscene`) and `tools/visualize-camera.py`; keep the three in sync. `Unknown` is the fallback of `get_obj_cfg_by_type`, so it must exist.

The organizers do not require a class and the detector is class-agnostic, so this field is not a supervised target. It exists so a reviewer can attribute alerts: the measured failure mode is nuisance alarms on tunnel infrastructure, and without a label for "this was the platform edge" those alarms cannot be explained. Labels written with the removed driving names still load and render through the `Unknown` configuration, but `/checkscene` reports them as unrecognisable.

Nothing except scene directories may live under `SUSTechPOINTS/data/`: `scene_reader` treats every entry as a scene and fails on any other file.

## Code map

| Path | Purpose |
|---|---|
| `tunnel_guard/io.py` | PointCloud2 decoding, invalid returns, acquisition timestamps |
| `tunnel_guard/geometry.py` | Track bed, paired rails, reference clearance envelope |
| `tunnel_guard/accelerator.py` and `cpp/kernels.cpp` | Required C++ decision kernels and their Python bindings |
| `tunnel_guard/segmentation.py` | Density-core clustering; optional published backends |
| `tunnel_guard/background.py` | Open3D-supported tunnel surfaces and protrusion protection |
| `tunnel_guard/detector.py` | KISS-ICP motion, candidates, tracking and temporal evidence |
| `tunnel_guard/run.py` | Reproducible bag runner |
| `tunnel_guard/inspect_bag.py` | Bounded layout, acquisition-clock and density inspection |
| `tunnel_guard/visualization.py` | Actual PointCloud2 / MarkerArray / status export for RViz2 replay |
| `tunnel_guard/evaluate.py` | One-to-one IoU matching and annotation validity |
| `tunnel_guard/corridor_report.py` | Before/after summary of a recorded run pair; generates `results/*.json` and the table a document quotes |
| `tunnel_guard/alarm_anatomy.py` | What the nearest alarm on each recorded frame is made of, as scalars |
| `tunnel_guard/stress.py` | Occlusion-aware synthetic ray-cast evaluation |
| `tunnel_guard/annotate.py` | Extract raw frames for annotation review |
| `tunnel_guard/sustech.py` | SUSTechPOINTS human labels into the `annotations/*.json` schema |
| `tunnel_guard/on_track.py` | Objects with real point support inside the clearance envelope |
| `tunnel_guard/sustech_import.py` | Recordings into SUSTechPOINTS scenes |
| `patches/` | Modifications applied to the pinned SUSTechPOINTS revision |
| `configs/detector.json` | Default detector recipe; no bag-specific branches |
| `results/` | Small recorded result summaries; full artifacts remain local |

Pipeline: validated points → KISS-ICP pose (optional deskew) → local bed and rails → supported tunnel-surface rejection → density-core segmentation → rail-relative clearance classification → temporal state and spatial evidence.

**Deskew is disabled in the current recipes.** The observed point-time span differs from the frame period, especially in cropped clouds; KISS-ICP normalizes that span to a full previous motion increment. `deskew_enabled=true` restores the experimental mode, but needs verified timing and prior-deskew provenance. Turning it off leaves motion distortion unresolved. This change is not a claim of higher detection accuracy.

`configs/detector.json` is the sole detector recipe. Backend-selection keys are invalid. `local_3d` is unsupported and rejected at configuration load because the native classifier implements the shipped `bed` frame only.

The required `tunnel_guard._native` extension handles radial selection, voxel representatives, mutual-radius connectivity, ground profile and reference, envelope classification, background masks, component assembly, normal statistics and evidence counts. Python retains the single-implementation work for rail-anchor search, robust plane fitting, tunnel-surface proposals, path continuation, range support, object association and mounting observation. Build with `python setup.py build_ext --inplace`. Earlier equivalence panels and Python/native timing results were produced before this revision and do not establish output identity for the consolidated source. Open3D plane proposals remain serial for repeatability. See [integration evidence and limitations](docs/NATIVE_INTEGRATION.md).

Segmentation preserves object portions outside the clearance gate. Dense instances cannot merge through a thin chain of border points. Temporal matching uses velocity, heuristic covariance, shape, and distinct-frame evidence. Missing path support must not turn rail removal into an infinite-width exclusion zone.

Tunnel-background rejection uses Open3D plane fitting and local normals. Only observed longitudinal surface strips are removed; cross-track panels, supported clearance intersections, protruding faces and their attachment edges are protected. No generic sparse-outlier deletion is applied. It is a local planar approximation, not a complete curved-tunnel model.

Focused verification: the original tilted empty-tunnel wall alert disappears; narrow-tunnel wall retention drops from 96.2% to about 1.6%, ceiling retention from 100% to about 8%. Edge protection intentionally leaves some lining near surface intersections. The 30-frame obstacle-preservation panel retained 100% of target points entering the original ROI, with no confirmed-localization regressions. The intermediate real sequence retained 5/5 provisional matches, but median processing increased to 741 ms; the later removal of redundant queries has not had a full-sequence timing run. These checks do not establish general safety or solve all infrastructure alerts. The broader stress rerun was stopped at user request.

## Output semantics

- `obstacle`: intersection with the configured reference envelope has its own confirmation evidence; not a validated collision claim.
- `unresolved_obstacle`: a confirmed intrusion claim whose interval crosses the contour edge - measured interior evidence the frame cannot certify.
- `candidate`: measured edge-uncertain evidence that is neither the tunnel's own cross-section nor attached to it, or an intrusion claim not yet confirmed.
- `no_obstacle_observed`: no claim and no doubt reported; **does not mean the route is clear**.
- `unknown`: insufficient geometry or returns.

Boxes describe observed support, not inferred full object volume. `distance_m` is the minimum forward x of the
evidence that **supports the object's claim** (`distance_method` names the definition), because the train meets an
object where its support first lies inside the contour; `cluster_nearest_x_m` keeps the nearest point of the whole
cluster. Neither is bumper distance or curve-integrated track distance. Uncertainty is heuristic, not a calibrated
safety probability.

- Per-object rail-relative coordinates are reported: `lateral_m` and `height_above_railhead_m` (min/max of the
  object's own support in the frame the decision is made in), `interior_voxels`, `interior_structural_voxels`,
  `interior_unmeasured_voxels`, `claim_voxels`, `boundary_uncertain_voxels`, `boundary_unexplained_voxels`,
  `structure_distance_m`, and `path_relation_reason`. A hazard decision can therefore be audited from the record
  without re-running the detector.
- `path_relation_reason` is one of `inside_heuristic_path_and_ground_interval` (certified interior support that is
  separable from the tunnel), `certified_interior_shared_with_structure` (the same, but every certified voxel sits in a
  cell the tunnel's own cross-section occupies elsewhere along the scan - the object and the tunnel are not separable
  by returns alone; this fired on 411 of 570 certified observations in the earlier experiment, so it
  names an ambiguity rather than discriminating), `envelope_boundary_uncertainty` (measured interior support whose
  interval crosses the edge), `structure_crossing_envelope` (interior support that is the tunnel's own cross-section),
  `unmeasured_corridor` (interior support only where the lateral reference is beyond its budget),
  `outside_envelope_evidence`.
- `certified_unexplained_voxels` splits an object's certified interior support into the part that is not the tunnel's
  own cross-section. The certified channel is deliberately **not** gated by that split: measured, gating it cost the
  labelled recording 108 of its 172 hazard-status frames. Those are scene-level alarms, not verified
  detections of the separately annotated nearby person. A compact cluster can share cells with the
  bed and the walkway. The weak `unresolved` channel is gated, because it rests on the uncertainty interval rather
  than on a measured intrusion.
- `nearest_candidate_m` is the nearest thing that is not a confirmed hazard; `unresolved_range_objects` and
  `nearest_unresolved_range_m` report objects whose lateral relation to the corridor cannot be measured at all,
  alongside `supported_range_m`.
- `health` (`normal` / `degraded` / `unavailable`) is decided by this frame's degradations only
  (`health_degradations`). Permanent calibration caveats are listed separately in `calibration_caveats` and no longer
  force `degraded` on every frame. `position_uncertain` marks a frame whose registration was rejected: its tracks
  persist but nothing accumulates through it.
- `timestamp_s` uses acquisition header time. `measurement_timestamp_ns` and `record_timestamp_ns` preserve both exact clocks; do not interpret their difference as latency.
- `source_scan_id`, `last_observed_s`, `hits`, and `evidence_timestamps_s` expose the source and temporal evidence. Duplicate acquisition timestamps are skipped by the reader; backwards time or a changed sensor frame stops the run explicitly. A new bag creates a new detector.
- `presence_confirmed` / `presence_confirmation` describe measured component presence, using dense current geometry or repeated supported observations. They do not identify a semantic class or authorize a hazard alarm. `confirmed` / `confirmation` retain the corridor-hazard confirmation policy; `intersection_confirmed` separately certifies current interior evidence. `near_track_objects` evaluation uses presence; `collision_hazards` uses the hazard policy. Historical outputs without presence fields must use their frozen evaluator. See [the real-object evidence diagnosis](docs/DETECTOR.md#разрыв-между-присутствием-и-оценкой--2026-09-22). RViz still uses red for confirmed intrusion and orange for confirmed hazards with unresolved/pending intrusion.
- Default detector recipes use one KISS-ICP registration thread for reproducible arithmetic. Object-relative evidence is centered before rotation, so cancellation of world coordinates cannot manufacture voxel support. The [issue #9 audit](docs/DETECTOR.md#численная-поддержка-у-границы-вокселя--issue-9) records exact six-recording replay results and the registration-speed tradeoff. Historical multi-thread recipes retain their explicit settings; neither mode is a field-safety guarantee.
- `coordinate_frame=tunnel_guard_local` identifies the transformed current-scan coordinates. `sensor_frame` is source metadata. No global TF or verified vehicle extrinsics are implied.
- `processing_s`, `read_and_process_s` and optional `visualization_s` use monotonic timing; summary includes ingestion/drop counts and visualization time. `range_observability` reports support, not free-space coverage.
- The runner reads and decodes the next scan on one producer thread while the detector processes the
  current one: `prefetch_depth` in the experiment config, default 1, bounded to keep one unprocessed
  scan in memory. With overlap, `read_and_process_s` is the cost of one loop iteration and
  `ingestion_s` is the consumer's block on that thread, **not** the age of a decision — a scan still
  waits for the scan ahead of it. `prefetch_depth: 0` restores inline reading. No measurement and no
  decision changes either way; measured evidence and the equivalence check are in
  `results/reader-overlap-20260919.json` (`configs/perf-prefetch-on.json`).

## View actual results in RViz2

The audit recipe records `build/audit-reviewed/doubleT_obstacle_rviz/` and `doubleT_platform_rviz/`. Export and CDR readback were run on macOS; the commands below require a machine with **ROS 2 Humble and RViz2** and have not been executed on that runtime here.

Run from the repository root in two terminals after sourcing ROS:

```sh
source /opt/ros/humble/setup.bash
rviz2 -d "$PWD/rviz/tunnel_guard.rviz" --ros-args -p use_sim_time:=true
```

```sh
source /opt/ros/humble/setup.bash
ros2 bag play "$PWD/build/audit-reviewed/doubleT_obstacle_rviz" --clock
```

The result bag is **recorded inference replay**, not live inference. Play it by itself: it supplies the measurement timeline through `/clock`; do not run a second clock publisher or mix it with the original bag's different record-time epoch. Restart playback and reset RViz when switching recordings or seeking backwards. Replay never continues a detector with future state.

Displays: grey measured scene, cyan supported reference contour, yellow tentative candidates, red confirmed nominal intersections, grey adjacent objects, and explicit health/nearest text. `candidate_measurements` markers show the actual voxel representatives even when the background display copy is sampled. Each frame starts with DELETEALL; markers expire after 0.3 simulation seconds. Pausing replay pauses simulation time too; the scene is explicitly labelled replay.

Fixed Frame is `tunnel_guard_local`: all messages already share this local frame, so no invented `map` or identity TF is needed. Do not accumulate clouds across frames. For a top view choose RViz's top-down view; for candidate inspection set the Orbit focal point to the object's reported `center` and reduce view distance. This changes camera framing, not physical coordinates. Marker namespaces can be toggled to inspect points without boxes.

`display_max_points` limits only the background visual copy; candidate representatives remain separate. GUI-off/headless comparison preserved status, boxes, distances, IDs and confirmations on all 60 inspected frames. Full bitwise pose equality is not claimed. A static real-frame overview is saved locally as `build/audit-preview.png`; it is not RViz GUI validation.

The envelope follows the [GOST 23961-80 M reference contour](https://engenegr.ru/gost-23961-80), with a conservatively filled lower contour above 50 mm. Actual vehicle dynamics, curves, mirror/current-collector extensions and organizer-certified extrinsics remain unverified. Anisotropic clustering helps unequal beam spacing but can merge vertically adjacent structures; temporal confirmation delays weak detections. These are explicit tradeoffs, not solved guarantees.

## Recorded results

See `results/acceptance.json` and `results/panel-summary.csv`. **Acceptance: not promoted.** Original IoU thresholds and provisional annotations were retained.

| Panel | Event recall | Object-frame precision | Notes |
|---|---:|---:|---|
| Original synthetic panel | 65/72 (90.3%) | 88.5% | 0/38 negative alarm episodes |
| Untouched random seed | 64/72 (88.9%) | 86.4% | Same scenario families, not a domain holdout |
| Measured beam pattern, 10–300 m | 77/96 (80.2%) | 80.1% | 1,460 frames; 0/50 negative episodes |

All synthetic panels fail the declared 95% event-recall target. Measured-pattern event recall: 100% at 10–100 m, 66.7% at 150 m, 58.3% at 200 m, 16.7% at 300 m. Ideal raycasting is **not hardware range or reflectivity validation**.

Final metro localization: 5/5 unchanged provisional boxes at IoU ≥0.25, mean IoU 0.282; earlier baseline was 2/5. One object sampled five times does not establish generalization.

Full real run: 2,488 frames; 1,579 `obstacle`, 819 `unresolved_obstacle`. These are **not false-positive counts** without exhaustive labels. Per-bag median processing was 294–486 ms on Apple M4 before the latency work; the three re-run recordings measure 99–123 ms with the C++ backend and native kernels afterwards (see `results/performance-20260917.json`): still not real-time at the ~10 Hz recording rate, and not target Intel performance.

Published backend comparisons and height/tilt ground audits are summarized in `results/`. The saved segmentation comparison predates the last support-preservation correction; its original full source/config snapshots are local under `build/`. Running the current comparison recipe evaluates the current code, not that historical snapshot. Paths inside result summaries refer to these intentionally untracked original artifacts.

## Optional research evaluations

```sh
uv sync --locked --extra comparison
uv run python -m tunnel_guard.compare --experiment configs/comparison.json
uv run python -m tunnel_guard.ground_audit --experiment configs/ground-audit.json
uv run python -m tunnel_guard.stress --experiment configs/stress-quality-heldout.json
uv run python -m tunnel_guard.stress --experiment configs/stress-measured-pattern-final.json
```

The ground audit also needs the six organizer bags. TRAVEL and HDBSCAN comparisons share geometry/tracking to isolate segmentation; they are not complete neural SOTA benchmarks.

Target-metro exhaustive positives, negatives, and held-out recordings are still needed.

## Dependencies and data licenses

- [KISS-ICP](https://github.com/PRBonn/kiss-icp), MIT: used directly for motion/deskew.
- [Open3D](https://www.open3d.org/), MIT, pinned to 0.19.0: plane segmentation, voxel sampling, normal and covariance estimation for background rejection. Its standard distribution adds substantial transitive dependencies.
- [TRAVEL](https://github.com/url-kaist/TRAVEL), **GPL-3.0-or-later**: optional comparison dependency only. Review license obligations before distributing an integrated derivative.
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan), BSD; [Patchwork++](https://github.com/url-kaist/patchwork-plusplus), BSD-2-Clause: optional published comparisons.

No project-wide redistribution license is granted here. Keep organizer data, credentials, local agent configuration and generated artifacts out of commits.
