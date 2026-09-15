---
name: run-experiment
description: Execute one configured ML experiment with smoke tests, reproducible artifacts, and evaluation evidence.
---

# Run an experiment

Read `experiment-workflow` first. Then use the project's experiment runner once it exists:

```bash
scripts/run_experiment.py --name EXPERIMENT_NAME
```

Before a long run, use the runner's dry-run or build-only mode when available. Keep dataset paths, model settings, seeds, and evaluation parameters in explicit project configuration. Keep domain assumptions in the project adapter, not in this skill.

For the current lidar project, the runner is still a required adapter: connect it to the point-cloud dataset loader, training entry point, checkpoint format, and fixed tunnel-scene evaluation protocol before using the command above.
