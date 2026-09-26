# HYBRID SYNTHETIC panel on `autoresearch/session-20260925`

This panel uses the detector at `065cf37` (`autoresearch/session-20260925`). It ports the contour-label layer from PR #34 (`a8149ab`) into the existing `realistic_stress.py` and `realistic_report.py`; it does not add a second scene generator or change the detector configuration or the numeric reference contour. The panel is the generator/label handoff requested for issue #24 and the surface-support risk in #28. Common detector acceptance runs remain outside its scope.

From a checkout with the supplied ROS bags and Python dependencies, build the required C++ kernel and run the single frozen recipe from the repository root:

```sh
python setup.py build_ext --inplace
python -m tunnel_guard.issue24_panel --plan configs/issue24-session-panel.json --output build/issue24-panel-repeat-01
```

The output directory must be new. The runner freezes effective recipes, processes three consecutive source scans per recording with the actual detector, inserts ten declared scenarios plus an unchanged background on each source, invokes `realistic_report`, and writes a log for every stage. The reviewed run is archived in `results/issue24-session-panel-final-20260926/`, including effective recipes, source and run manifests, raw labels, inserted-point provenance, compressed detector predictions, reports, logs, and SHA-256 sums. Native binary, detector config, source bag files, source poses, and code identity are recorded. The raw multi-gigabyte bags are not committed.

## Frozen source and scene design

The organizer-declared empty `doubleT_platform` recording supplies tuning backgrounds; `roundT_doubleT` supplies synthetic evaluation backgrounds. Both are previously inspected development recordings under the issue #21 contract, not a blind real holdout. The person bag is excluded. The split seeds and concrete dimensions differ; semantic groups are the same: low, thin suspended, contour edge, longitudinally long, and measured-surface attached. Each group has one nominal crossing case and one nominal noncrossing case. The box is fixed in baseline-odometry world coordinates through the three-frame sequence. Poses and geometry come from the chosen session detector.

The PR #34 rule separates object presence, full-form intersection with the selected reference contour, and visible inserted support intersection. We place the box above the estimated **rail running surface**, which is the contour's datum. Full-form intersection is recomputed independently from the return rays in each moving frame on a fixed 4 cm volume lattice; the nominal lateral mode only places the box. An adjacent world-fixed box that crosses a later frame's curved contour is shifted outward using source geometry only, and the shift is saved. The declared first-frame edge overlap and clearance remain 5 cm. These are labels against the reference contour, not verified vehicle collision truth.

Measured-surface cases choose real side-surface points within the box's own longitudinal footprint and vertical span. Their wall/trough anchor is iterated with the local ground estimate; the outside box face overlaps that measured surface by a declared 2 cm, with a predeclared 3 cm maximum contact gap. The final first-frame nearest measured-return gaps are below 6 mm in both splits. The tuning negative uses the farther wall and the protruding case uses the nearer trough. This placement is a local geometric attachment, not proof of a continuous rigid wall behind every part of the object.

The renderer replaces only recorded return slots where the opaque inserted box is nearer. It carries the selected source return's acquisition time, records the deduplicated and raw source point indices, the full world box, and the observed support. Empty emissions stay empty. The assumed 1 cm range jitter and fixed intensity are uncalibrated; multiple-return behavior, material reflectivity, weather, multipath, and Pandar128 detection probability are not modeled. The measured ray pattern does not establish hardware detection range.

## Actual replay and label audit

The one-command final run used the actual session detector for source preparation and all 66 scenario frames. There were no unsupported placements. Every group in **each** split has three full-form crossing frames and three noncrossing frames, representing one scenario of each kind over three correlated scans. The same three positive frames also have observed-support contour intersection. The thinnest evaluation case has only 7–8 inserted returns per frame; zero-return cases would remain explicit in `cases.json` and the report.

The provenance audit found no observed-support intersection without full-form intersection, no changed world box within a case, and no replacement farther than the source return. First-frame measured-surface contact gaps are 0.006 m or less. I visually inspected the final evaluation thin, edge, long, flush, and protruding scenes in `issue24-session-scenes-final-20260926.png`. The thin object is sparse, the long object's returns are discontinuous in x, and the protruding object remains attached to measured side support.

The detector confirmed 5/15 conditionally scored crossing frames on tuning and 3/15 on synthetic evaluation; background-only scenes claimed no obstacles in these three-frame clips. These are frames from five physical scenarios, not 15 independent objects. The attached report separates box overlap, point coverage, confirmation, and path relation. The poor confirmation result is retained, not tuned away. Organizer-declared emptiness is not exhaustive annotation, so the short background result cannot establish precision or a false-alarm rate.

Earlier runs `v1`–`v8` remain under `build/` with their failed or revised label/contact assumptions named in the final configs. They were not used to promote the result. This focused 20 m panel does **not** complete the wider distance, movement, or field-safety acceptance in issue #24, and none of its scores is field or private-set recall.
