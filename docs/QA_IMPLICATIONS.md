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
