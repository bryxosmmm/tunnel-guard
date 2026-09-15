---
name: experiment-workflow
description: Run a reproducible local or remote ML experiment from configuration through evaluation and artifact collection.
---

# Experiment workflow

Use this workflow for training, evaluation, ablation, benchmarking, or reproduction.

1. State one hypothesis and one measurable pass/fail criterion.
2. Locate the experiment configuration and confirm the seed, dataset split, model/checkpoint, device, and output directory.
3. Run a small smoke test first. Check that data loading, one forward/backward step, checkpoint writing, and evaluation all work.
4. Run the configured experiment through the project's runner. Do not hide hyperparameters in a skill or invocation.
5. Preserve the command, environment/image, git revision, metrics, logs, checkpoints, and failure diagnostics under the run directory.
6. Evaluate on a fixed holdout or test panel and report validity failures separately from the score.
7. If a remote GPU was rented, fetch required artifacts locally and issue exactly one `scripts/destroy_vast.py` command after collection. Do not poll after teardown.

## Project adapter

This repository currently provides the workflow contract and Vast helpers, but does not yet provide a lidar-specific `scripts/run_experiment.py`. Implement that adapter around the project's actual training/evaluation entry points before using this skill for production runs.
