# Review: experiments/gerasimov, bb8c20c

Reviewed 2026-09-17 against common ancestor `59f90f8`; current branch is
`experiments/morev` at `48977ad`. Verdict: useful, substantial acceleration;
request changes before integration. No merge or detector changes performed.

## Findings

### P1 — Invalid geometry crashes Detector.process

`tunnel_guard/detector.py:484–503` in the reviewed revision:
`cluster_candidates` populates `carried` only when `geometry.valid`, but
`carried[0]` is read unconditionally. Missing rail/ground support should produce
`unknown`; instead both native and Python paths reach an IndexError once geometry
is invalid after the initial point-count check. Restore classification for the
invalid branch, or explicitly construct its unsupported range-observability data.

Reproduced by running the actual CLI on the first real doubleT_obstacle scan with
`rail_min_longitudinal_bins=1000000`, deliberately preventing rail acceptance.
This is an altered configuration, not a naturally occurring failure claimed on
the default 798-frame panel. Teammate CLI exits with IndexError at line 503;
our current CLI processes the identical scan/configuration and emits `unknown`.
Recipes and both logs are retained under `build/review-gerasimov-*`.

### P1 — Native header omitted from distribution and Docker context

`setup.py:4–7` now builds two translation units requiring `cpp/native.h`
(`cpp/kernels.cpp:21`, `cpp/voxel.cpp:2`). The header is absent from the generated
source distribution, and `.dockerignore:17–18` admits only `cpp/*.cpp`.
Consequently a source-distribution installation or Docker build cannot compile
the accelerator. Since the extension is optional, the build command can return
success while omitting `_native`; selecting `detector.json` subsequently
cannot import the required module.

Reproduced with `setup.py sdist`, extraction, and `setup.py build_ext --inplace`
from that archive: compiler reports `fatal error: 'native.h' file not found`,
but command exit code is 0. Direct checkout build succeeds. Include the header
in source packaging and Docker allowlist, declare it as a build dependency, and
make a deployment selecting native acceleration verify module availability.
Docker itself was not built or run.

## Actual replay results

Compiled the untouched archived revision in `build/review-gerasimov-source` and
ran `detector.json` (three background plane proposals), every frame,
seed 20260915, visualization off. Compared against saved common-ancestor outputs
in `build/native-full-real` using the existing `tunnel_guard.panel_report` CLI.

| Recording | Frames | Saved ancestor p50 ms | Reviewed p50 ms | Reviewed p95 ms |
|---|---:|---:|---:|---:|
| doubleT_obstacle | 201 | 517 | 155 | 209 |
| doubleT_platform | 345 | 324 | 116 | 206 |
| roundT_doubleT | 252 | 407 | 131 | 198 |

All compared non-timing fields match (status, distances, objects including IDs
and evidence, geometry, motion, pose, range observability, health, resets).
Maximum compared float difference: 4.98e-13; tolerance 1e-10. No missing/extra
frames or measurement identity errors. All 44,246 candidate observations at
60 m or farther retained. No future or duplicate timestamps found in compared
object evidence histories. Counts are observations, not independent events.

Times are descriptive, not a paired speedup measurement: the ancestor was run
earlier, and this review also performed short compilation/CLI work during replay.
The results support substantial acceleration but do not reproduce exactly the
README's 116 ms benchmark or establish sustained 10 Hz. No target Intel hardware
or live ROS latency measured; no exhaustive labels or field-safety claim.
`detector-native-fast.json` reduces plane proposals to two and changes behavior;
it was not run or approved in this review.

## Integration and evidence notes

- Preserve our subsequent envelope-interval uncertainty correction and viewer.
  Native `classify_geometry` implements the common ancestor's policy; a textual
  merge alone does not port the newer interval policy into C++. Re-evaluate the
  integrated detector against `build/runtime-context-real`, not just the ancestor.
- Refresh `results/performance-20260917.json`: it names revision `310a8b5` and its
  native equivalence section records changed objects/geometry despite the claim
  of unchanged decisions. Our replay of final `bb8c20c` matches the ancestor;
  the old report must not be used as proof for that final revision.
- Extend run/benchmark source capture to include kernels.cpp and native.h;
  existing capture stores only voxel.cpp. Review provenance separately records
  all native source hashes and the compiled binary hash.
- The source export lives below our checkout, so the runner's automatic Git
  revision identifies the parent checkout incorrectly. Actual reviewed revision
  and hashes are explicitly recorded in
  `build/review-gerasimov-real/review-provenance.json`.

## Commands executed

Commands below use the repository root unless another working directory is given.
`PY` denotes the absolute `.venv-iteration/bin/python` path.

```sh
git fetch origin
git merge-base HEAD origin/experiments/gerasimov
git diff 59f90f8..bb8c20c -- tunnel_guard cpp configs setup.py README.md
git archive bb8c20c | tar -x -C build/review-gerasimov-source
# cwd build/review-gerasimov-source:
$PY setup.py build_ext --inplace
$PY -m tunnel_guard.run --experiment /Users/mors1337/Code/DL/tunnel-guard/build/review-gerasimov-experiment.json
$PY -m tunnel_guard.run --experiment /Users/mors1337/Code/DL/tunnel-guard/build/review-gerasimov-no-rails-experiment.json
$PY setup.py sdist --dist-dir /Users/mors1337/Code/DL/tunnel-guard/build/review-gerasimov-dist
# Extracted the generated tar.gz with Python tarfile; cwd extracted package:
$PY setup.py build_ext --inplace
# cwd repository root:
$PY -m tunnel_guard.run --experiment build/review-gerasimov-no-rails-current.json
$PY -m tunnel_guard.panel_report --panel build/review-gerasimov-panel.json --output build/review-gerasimov-comparison.json
```

Automated tests/suites were neither created nor run. Verification used actual
builds, detector CLI replay and evaluation of saved real outputs. No images were
downloaded/built and no datasets extracted. Review artifacts occupy about 320 MB.
