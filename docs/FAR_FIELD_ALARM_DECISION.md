# The far-field alarm decision, with the numbers on both sides

The empty-tunnel alarm load is the last detection-side question this project has, and it is a **decision**, not a bug to fix.
Six approaches failed to separate the tunnel's own far-field surfaces from objects on this data (see
`results/empty-tunnel-alarm-load-20260919.json` and `results/empty-tunnel-alarm-load-attempt2-20260919.json`), and the
measured sensor limits say why: rails vanish by 90 m, the bed band is empty beyond 70 m, a carried bore offset is about 1 m
wrong, partial-arc cross-section fits are ill-conditioned beyond 80 m, and multi-scan shape evidence is smeared by sparse
returns and imperfect registration. What remains is what the output should *mean*.

## What the two plausible scoring conventions see

| convention | our false-alarm load on an empty tunnel | evidence |
|---|---|---|
| events or objects | **211 hazard-class objects over 1460 panel frames**, about 0.14 per frame; empty-scene alarm **episodes** 0.0 | panel dissection, `results/panel-false-positive-anatomy-20260919.json` |
| frame status | **`unresolved_obstacle` on 59 of 60 frames** of `roundT_doubleT` | recorded runs |

Same system, same data, and the cost differs by orders of magnitude depending on which one the evaluator uses. That is the
whole decision: our output is honest, and how honest output scores depends on the scorer.

## Why the honest form is what it is

`unresolved_obstacle` currently means **"a confirmed potential hazard whose relation to the swept path is not established"**.
Beyond the supported range the corridor centre is extrapolated and wrong by metres, so the tunnel's own surfaces appear to be
inside the swept path. That range is reported per frame as `supported_range_m`: the **contiguous** interval from the first
station whose path uncertainty is within `path_max_uncertainty_m`, whose bed uncertainty is within `ground_max_uncertainty_m`
and which lies inside a measured frame segment in the historical `local_3d` mode, up to the last station before the first gap. It replaces a
maximum over all supported stations, which let an island of support beyond a gap claim a horizon the corridor did not have.

The size of that defect is mode-dependent and was measured on 2026-09-20. In the shipped `bed` mode the contiguity and bed
conditions shortened **one frame of 402** by 2 m, and the reported value moved by half a grid step (the grid is now 0.5 m
instead of 1.0 m, so the field reads 61–62 m rather than 61 m). In the historical `local_3d` mode, where the basis needs measured frame
segments, the old rule was wrong by a wide margin on every sampled frame: over 20 real frames of `roundT_doubleT` it reported
a median 63.5 m while exceeding the end of the last available 3D section on **all 20**, at 67 m against 20 m on the worst
frame. The new rule reports a median 45.0 m and tracks the real section end (48.0 against 48.1, 20.0 against 20.0, 35.0
against 35.0). Reporting an unsupported range as a hazard horizon is what makes an empty tunnel alarming on nearly every frame.

Demoting it is what three measured attempts did, and each cost something real:

| attempt | result |
|---|---|
| demote `unsupported_nominal_envelope` (uncertified path) | an injected object at **100 m stopped being reported**: the frame said `no_obstacle_observed` while the object was there |
| require the contour to contain the object (`intersecting` only) | same failure at 100 m, on every frame, while clearing the empty tunnel completely (59 -> 0 alarms) |
| cross-section buckets, boundary anisotropy, sparsity admission, planar shape | rejected for removing detections, for cost without effect, or for changing nothing |

So the two ends are: **keep the alarms** (an empty tunnel looks alarming under convention B) or **demote them** (an object at
100 m, and any boundary-ambiguous real object, stops being reported - under both conventions).

## The three options, stated plainly

1. **Keep as it is (recommended).** Every object is reported with its distance, its own lateral uncertainty, and now the
   frame's supported range. An evaluator reading objects or events sees a low false-alarm load; an evaluator reading frame
   statuses sees honesty about an unmeasurable path. No real detection is lost, and the case weights *missed obstacles* and
   *detection distance* alongside false alarms.
2. **Demote uncertified-path objects to candidates** (`hazard_requires_certified_path: true`). Clears part of the wall; costs
   the 100 m tier, measured.
3. **Report `obstacle` only for a certified contour intrusion.** Clears the wall entirely; costs the 100 m tier *and* any
   boundary-ambiguous real object, measured (task 100 m: `no_obstacle_observed` on all frames with the object present).

Both 2 and 3 are one recipe key, both measured, both recorded with the check that killed them. If the team decides the
scorer reads statuses, option 2 is the smaller loss of the two; if it reads objects and events - which the case's own
"препятствие обнаружено / не обнаружено **и расстояние до ближайшего**" wording suggests - option 1 costs nothing and keeps
the 100 m detection that the other two give up.

## What would change the answer

A prior map from another pass over the route, denser far-field returns, or a vehicle swept-volume specification. Any of the
three turns the far field from an information limit into a measurement, at which point this decision disappears rather than
being managed.
