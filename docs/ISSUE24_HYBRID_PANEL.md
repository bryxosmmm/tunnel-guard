# Issue #24 hybrid contour panel

The existing `tunnel_guard.realistic_stress` generator is extended rather than
replaced. It ray-casts a synthetic box through measured directions and real
background returns from a recorded scan. The panel is **HYBRID SYNTHETIC**: it
can expose a detector/contour regression, but cannot establish hardware
detection range, field recall, a calibrated Pandar128 profile, vehicle
clearance, or collision avoidance.

The former panel labelled an injected box as a hazard when its centre was within
half the rail gauge. That is not the detector's reference contour: the selected
contour is wider and varies with height. This made contour-edge injections look
like adjacent negatives. The generator now records three distinct facts for
each source frame:

| Field | Meaning |
|---|---|
| `object_present` | a synthetic full box was placed in the world frame |
| `full_shape_reference_contour_intersects` | the full box intersects the configured piecewise-linear reference contour at its placement frame |
| `observed_support_reference_contour_intersects` | the actual returns generated from that box intersect the current frame's same reference contour |

Only the last two together create a collision-hazard evaluation label. A full
shape that has no returns, or whose available returns remain outside the
contour, is retained as an observability outcome rather than silently becoming a
false negative. Adjacent boxes remain inserted scenes, not injection-free
background negatives.

## Frozen smoke recipe

The recipes use one real scan from `doubleT_obstacle`, frozen by a prior
actual detector run. They intentionally cover low, thin/raised and
longitudinally long boxes at two ranges, two heights and signed inside/edge/
adjacent placements. `edge` overlaps the selected contour by 5 cm; `adjacent`
stays 5 cm outside it. These distances are scenario definitions, not tuned
thresholds.

```sh
python -m tunnel_guard.run --experiment configs/issue24-source-poses-smoke-20260924.json
python -m tunnel_guard.realistic_stress --experiment configs/issue24-contour-smoke-20260924.json
python -m tunnel_guard.realistic_report \
  --run build/issue24-contour-smoke-retry2-20260924 \
  --output results/issue24-contour-smoke-20260924.json
```

Inspect `unsupported_cases.json`, `sensor_profile.json`, `cases.json`,
`inserted.jsonl`, the generated annotations and the report together. The
report's `synthetic_provenance` counts full-form intersections, observed-support
intersections, visibility failures and any adjacent scene that produced a
detector hazard. Do not promote the smoke result to a full #24 acceptance.

`empty_slots_return_object` is deliberately `false`: only a measured return slot
that the nearer box occludes produces an inserted return. It therefore brackets
one pessimistic visibility regime, not a calibrated sensor or multiple-return
simulation. The current recipe does not establish physical contact between a
box and a measured wall/lotok; that required wall-attached case remains open and
must not be inferred from an adjacent-contour placement.

## Actual label-smoke result

The final one-source-frame smoke executed the real detector on 73 cases (72
insertions plus the unchanged source background). Its saved report is
`results/issue24-contour-smoke-retry2-20260924.json`. Of the inserted scenes,
48 full boxes intersected the selected contour, 43 had observed support that
intersected it, and 11 full boxes had support only outside it. The resulting
conditional score contains 37 observed-support contour labels: 5 were confirmed
hazards, 21 overlapped but were not confirmed, and 10 did not overlap a reported
object. The 13 scored edge labels yielded **zero** confirmed hazards.

This is a useful exposed failure, not a recall result or a promotion. The source
recording itself has pre-existing reported hazards and is not exhaustively
negative; the 24 adjacent inserted scenes that also produced hazard statuses
cannot be attributed to the insertion. The result demonstrates that the edge
denominator is no longer empty and that full form, visibility and contour
relation are no longer conflated. A multi-frame, independently split panel with
measured wall-attached placement remains required for issue #24 acceptance.
