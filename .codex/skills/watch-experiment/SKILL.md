---
name: watch-experiment
description: Inspect GPU utilization, remote process state, and recent or streaming logs for a remote experiment.
---

# Watch a remote experiment

Run from the project root:

```bash
scripts/watch_experiment.py EXPERIMENT_NAME --lines 80
```

Add `--follow` only when continuous streaming is needed. The watcher reports a detached run's PID and exit code when available and does not modify training state.
