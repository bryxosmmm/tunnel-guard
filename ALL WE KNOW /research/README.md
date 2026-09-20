# Research intake — 2026-09-19

This directory is an isolated intake of primary sources used for the current engineering
decision. `SOURCES.json` records exact URLs, hashes, licenses and the pinned source-code
revision. No downloaded package or repository is installed globally.

## What transfers to Tunnel Guard

- Zhou et al. (2017) build clearance inspection in a dynamic rail-aligned coordinate system.
  That supports the project's decision to express every accepted cluster relative to the
  locally observed rail path rather than only in the sensor frame. Their reported centimetre
  result is from one mapped tunnel and is not evidence for this online detector.
- Nan et al. (2024) use distance-adaptive Euclidean clustering, PCA and local comparison with
  background data. The adaptive-neighbour principle is already present in Tunnel Guard. Their
  scanner is a fixed-position mechanical system with a controlled ±25 m test area, so their
  thresholds and stable-detection rates cannot be transferred to Pandar recordings.
- Pereira et al. (2025) combine a pre-existing track route, GNSS and a search volume. This is
  strong evidence that prior track geometry can reduce work on curves, but GNSS is not a viable
  tunnel dependency and this dataset has no surveyed route to validate such a branch.
- FreeDOM (Li et al., 2025) conservatively declares free space only after repeated ray evidence,
  then treats occupancy inside that space as motion. It requires stable poses and detects
  dynamic change; a static foreign object can remain in its static map. Its MIT implementation
  is retained for study, not copied into the detector.
- The 2026 UAV alignment paper grows rails from manually selected seed points and refines
  alignment offline. It is relevant to building a surveyed prior, not a drop-in online rail
  estimator for a moving tunnel-mounted sensor.

## Decision

Adopted now:

1. Candidate JSON contains `rail_relative_support`: measured lateral support, running-height
   interval and local path uncertainty. It explicitly says this is observed point support, not
   an inferred full object shape.
2. The frame-level geometry/background classification is reused for observability bins instead
   of being computed twice. Profiling the removed call on the fixed 18-frame real panel measured
   4.486 ms p50 (5.197 ms on `doubleT_obstacle`, 2.822 ms on `doubleT_platform`).

Deferred deliberately:

- Free-space motion filtering until pose and point timing have independent evidence.
- Background local-ICP rejection until a clean background acquisition and exhaustive negative
  intervals exist; otherwise the method can learn a real obstacle as background.
- Prior-route fusion until the route, extrinsics and coordinate transform have provenance.
- Multi-hypothesis switch geometry remains necessary; none of these sources supplies a verified
  online solution for the present sensor/data combination.

## Verification boundary

The actual detector processed the same fixed 18 real frames before and after the change. After
removing timing fields and the newly added diagnostic object, every pre-existing JSON field was
identical on all 18 frames. The run produced 3,462 enriched object records; 2,514 had finite
locally supported path uncertainty. This proves deterministic preservation on that panel, not
accuracy, precision, recall, safety or transfer to unseen tunnels.
