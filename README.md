# Tunnel Guard

Class-agnostic LiDAR obstacle-detection baseline for metro tunnels. Reads ROS 2 PointCloud2 bags directly; no ROS installation, Docker, or pretrained weights required for the default pipeline.

**Research baseline, not a validated collision-warning system.** Recall, infrastructure alarms, generalization, and runtime remain unresolved. `CASE.md` contains the original requirements; deployment and demo requirements are not implemented.

## Quick start

Python 3.11+; Python 3.12 was used for the recorded results. Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked
uv run python -m tunnel_guard.stress --experiment configs/stress-quality.json
```
Agent policy lives in `AGENTS.md`: no subagents or automated tests. Verify changes through actual detector runs and configured evaluations; the repository intentionally has no test suite.


The synthetic run needs no external data. It exercises 110 scenarios / 330 frames, and writes predictions, annotations, metrics, exact configuration, and a source snapshot to `build/tunnel-guard-stress-quality-final/`.

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

```sh
uv run python -m tunnel_guard.run --experiment configs/evaluation-quality.json
uv run python -m tunnel_guard.evaluate \
  --run build/tunnel-guard-real-quality \
  --annotations annotations/sourcecraft-provisional.json \
  --output build/tunnel-guard-real-quality/localization.json
```

The five supplied annotation boxes describe **one provisional upright structure**, not independent verified hazards. They are nonexhaustive: real precision cannot be computed from them.

## Code map

| Path | Purpose |
|---|---|
| `tunnel_guard/io.py` | PointCloud2 decoding, invalid returns, acquisition timestamps |
| `tunnel_guard/geometry.py` | Track bed, paired rails, reference clearance envelope |
| `tunnel_guard/segmentation.py` | Density-core clustering; optional published backends |
| `tunnel_guard/detector.py` | KISS-ICP motion, candidates, tracking and temporal evidence |
| `tunnel_guard/run.py` | Reproducible bag runner |
| `tunnel_guard/evaluate.py` | One-to-one IoU matching and annotation validity |
| `tunnel_guard/stress.py` | Occlusion-aware synthetic ray-cast evaluation |
| `tunnel_guard/annotate.py` | Extract raw frames for annotation review |
| `configs/detector.json` | Default detector recipe; no bag-specific branches |
| `results/` | Small recorded result summaries; full artifacts remain local |

Pipeline: validated points → KISS-ICP deskew/pose → local bed and rails → density-core segmentation → rail-relative clearance classification → temporal state and spatial evidence.

Segmentation preserves object portions outside the clearance gate. Dense instances cannot merge through a thin chain of border points. Temporal matching uses velocity, heuristic covariance, shape, and distinct-frame evidence. Missing path support must not turn rail removal into an infinite-width exclusion zone.

## Output semantics

- `obstacle`: confirmed structure intersects the configured reference envelope.
- `unresolved_obstacle`: confirmed nominal intersection with insufficient geometry support; not a proven collision.
- `candidate`: insufficient confirmation evidence.
- `no_obstacle_observed`: no hazard reported; **does not mean the route is clear**.
- `unknown`: insufficient geometry or returns.

Boxes describe observed support, not inferred full object volume. Distance is forward sensor-frame distance, not bumper distance or curve-integrated track distance. Uncertainty is heuristic, not a calibrated safety probability.

The envelope follows the [GOST 23961-80 M reference contour](https://engenegr.ru/gost-23961-80), with a conservatively filled lower contour above 50 mm. Actual vehicle dynamics, curves, mirror/current-collector extensions and organizer-certified extrinsics remain unverified. Anisotropic clustering helps unequal beam spacing but can merge vertically adjacent structures; temporal confirmation delays weak detections. These are explicit tradeoffs, not solved guarantees.

## Recorded results

See `results/acceptance.json` and `results/panel-summary.csv`. **Acceptance: not promoted.** Original IoU thresholds and provisional annotations were retained.

| Panel | Event recall | Object-frame precision | Notes |
|---|---:|---:|---|
| Original synthetic panel | 65/72 (90.3%) | 88.5% | 0/38 negative alarm episodes |
| Untouched random seed | 64/72 (88.9%) | 86.4% | Same scenario families, not a domain holdout |
| Measured beam pattern, 10–300 m | 77/96 (80.2%) | 80.1% | 1,460 frames; 0/50 negative episodes |
| Independent OSDaR23 subset | 1/5 (20%) | Not available | 1/37 independently authored boxes matched |

All synthetic panels fail the declared 95% event-recall target. Measured-pattern event recall: 100% at 10–100 m, 66.7% at 150 m, 58.3% at 200 m, 16.7% at 300 m. Ideal raycasting is **not hardware range or reflectivity validation**.

Final metro localization: 5/5 unchanged provisional boxes at IoU ≥0.25, mean IoU 0.282; earlier baseline was 2/5. One object sampled five times does not establish generalization.

Full real run: 2,488 frames; 1,579 `obstacle`, 819 `unresolved_obstacle`. These are **not false-positive counts** without exhaustive labels. Per-bag median processing was 294–486 ms on Apple M4: not real-time at the recording rate, and not target Intel performance.

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

For the independent outdoor-railway check, download `1_calibration_1.1.zip` from [OSDaR23](https://data.fid-move.de/dataset/osdar23) into `data/real/osdar23/`, then run:

```sh
uv run python -m tunnel_guard.osdar --experiment configs/osdar-evaluation.json
```

The fixed adapter uses standard gauge and translates the railhead-origin coordinates. It does not tune against individual objects. Outdoor transfer was poor, including objects with substantial returns. Target-metro exhaustive positives, negatives, and held-out recordings are still needed.

## Dependencies and data licenses

- [KISS-ICP](https://github.com/PRBonn/kiss-icp), MIT: used directly for motion/deskew.
- [TRAVEL](https://github.com/url-kaist/TRAVEL), **GPL-3.0-or-later**: optional comparison dependency only. Review license obligations before distributing an integrated derivative.
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan), BSD; [Patchwork++](https://github.com/url-kaist/patchwork-plusplus), BSD-2-Clause: optional published comparisons.
- OSDaR23 annotations: CC0-1.0; sensor data: CC BY-SA 3.0 de. Neither the raw dataset nor its archive is committed.

No project-wide redistribution license is granted here. Keep organizer data, credentials, local agent configuration and generated artifacts out of commits.
