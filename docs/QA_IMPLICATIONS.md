# Q&A: implications and unresolved geometry

Source received 2026-09-16: `Расшифровка.md`, SHA256
`da22e083a14e35bd4eeeca361ca78fc738afc1776a67acf8db8e42dab41e0fac`.
The file calls itself a **draft supplement for organizer agreement**, synthesized
from two transcripts. It is not the raw recording, a signed specification, a
calibration file, or permission to change evaluation labels. The source itself
stays local; the following interpretation is auditable by its hash.

## Decisions

- **10 Hz input is not achieved throughput.** Current Python detector latency must
  be measured and reported. Slow offline replay is useful but cannot demonstrate
  operation at train speed. A live adapter must bound its input queue and report
  gaps/unknown state, rather than deliver an ever older alarm.
- **Reported 2.1 m total width × 3 m height conflicts with our GOST reference.**
  Keep the existing comparison configuration unchanged. Add a clearly provisional
  rectangular sensitivity profile; any alarm reduction from narrowing the
  contour is a specification effect, not an algorithm accuracy improvement.
  Neither profile establishes an actual swept envelope on curves. The vertical
  origin, margins, cant, protrusions, and installation remain unverified.
- **300×300×100 mm and hanging cables are distinct detection regimes.** A person
  track does not validate either. Do not discard sparse returns to clean plots.
  Existing ray-insertion results are controlled diagnostics, not field recall.
  Optical reflectance, cable diameter, sensor beam pattern and the actual
  available returns constrain further claims.
- **Some recordings may use a different sensor mounting.** Do not infer one
  shared extrinsic transform from the filenames. Configured axes are processing
  conventions; local rail fitting is not an extrinsic calibration measurement.
  Keep `sensor_profile_verified=false` until supported by evidence.
- **Flush rails at pressure gates challenge the raised-rail model.** Current
  `TrackGeometry._rail_profile` requires returns 0.10–0.30 m above local bed.
  Missing anchors must remain unavailable geometry. Relaxing the height gate
  would also accept floor seams and is not justified without examining the
  full gate recordings. The selected rail path at switches is also unresolved.
- **No odometry supplied.** Internal scan registration is permissible, but its
  residual and observability checks do not prove longitudinal motion accuracy
  in repetitive tunnels. Do not use future scans to validate current hazards.
- **An object reportedly lies on rails near a person.** This is a search lead,
  not an annotation. No frame identity, support box or negative interval is
  established by the statement. Exhaustive manual ground truth is still absent.
- **Output distance needs a reference definition.** Presently reported distance
  is minimum forward x from the configured processing-frame origin to current
  supporting measurements. It is not braking distance, arc length, or distance
  from the train bumper. Preserve that distinction in the viewer and JSON.

## Actual rail-object search — 2026-09-22

The original draft was located and its SHA256 matched the source above. Question
15, line 56, says there is an item on the rails near a person, but supplies neither
a recording/frame identity nor coordinates. Existing annotations identify the
near person and a different provisional upright structure, not this item.

Raw-return review covered all 201 `doubleT_obstacle` frames in explicit near and
far windows. The near window (forward 2–25 m, estimated track offset ±1.05 m,
bed-relative height −0.15–0.8 m) contained no returns more than 0.05 m above the
estimated railhead; maximum height in that view was 0.229316 m above the estimated
bed. This is **not a negative label**: low objects, missing returns and inaccurate
geometry remain possible. The review bounds are not detector thresholds.

The far review distinguishes a moving upright group near 56 m from a persistent
lower group beside it. The latter is a **candidate for inspection**, not an
identified foreign object. At frame 0 its review ROI contains 12 return slots,
9 unique XYZ positions and 8 detector representatives, assigned to track 179.
Its observed component extends only about 0.077 × 0.356 × 0.511 m; these are
sampled extents, not amodal dimensions or the organizer's 300×300×100 mm object.

| Initial track 179 event | Frame | Time since first available scan | Forward distance |
|---|---:|---:|---:|
| Candidate already present | 0 | 0 s | 56.3262 m |
| Presence confirmed | 2 | 0.200009 s | 56.3300 m |
| `confirmed=true`, intersection still pending | 32 | 3.200177 s | 56.3460 m |
| `intersection_confirmed=true` | Not observed on track 179, frames 0–38 | — | — |

**The 3.20 s value is not a verified obstacle-detection latency.** At frame 32,
the scene-level `obstacle` is supported by upright track 176; track 179 has
`intersection_confirmed=false`. Visibility before the recording is unknown,
and a first observation at 56 m establishes neither maximum range nor 100 m recall.

Saved real-support reclassification isolates both sampling and estimated geometry:
frame-32 support has 2 certified interior voxels in frame-32 geometry, but 0 in
frame-31 geometry after estimated static-world reprojection. Frame-31 support
has 0 certified interior voxels under either geometry. Thus the transition is
not explained by elapsed confirmation time alone. This offline counterfactual
assumes static support and estimated registration; it is not ground truth.

Actual membership also reveals instance merging: the ROI's 7 representatives
at frame 39 and 10 at frame 40 belong to upright track 176, not a separate low
track. At frame 60 its 8 representatives belong to track 1073, a component with
18 representatives in total. Disappearance of track 179 therefore does not prove
that the low support disappeared. The ROI is a review window, not an annotation,
and cross-section sharing does not prove infrastructure identity.

Evidence: [search scope](../results/rail-object-search-20260922.json),
[exact support, timing and counterfactual report](../results/rail-object-evidence-20260922.json),
[raw far-region views](../results/rail-object-raw-views-20260922.png),
[component merge/split views](../results/rail-object-membership-20260922.png).
Recipes: `configs/rail-obstacle-{review,far-review,focus-review,trace}.json`
and `configs/rail-object-report.json`. All executed on real measurements;
no detector threshold, annotation or box was adjusted.

**Still missing:** an independent frame/point-cloud anchor or organizer image
identifying the claimed item. Without it, object-specific recall, physical
collision status and visibility-to-alarm latency for that item remain unmeasured.

## Delivery

Use the same decoder/detector/display builder for offline processing and ROS2
Humble. Preserve acquisition timestamps, sensor frame, configuration, stage
quality, candidate confirmation and intersection confirmation separately.
A stopped input must clear the live display and publish `unknown`, never an
all-clear indication. Browser review of recorded output is explicitly labelled
as recorded review; it must select the exact source measurement by both record
and acquisition timestamp.

## What needs outside evidence

Exact sensor and per-recording installation; vehicle contour/vertical origin;
selected route at switches; independently reviewed object events and exhaustive
negative intervals. None can be established by tuning detector alarms. These
are limits on validation, not reasons to stop implementing a usable runner and
review tool.
