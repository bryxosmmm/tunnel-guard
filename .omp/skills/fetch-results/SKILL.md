---
name: fetch-results
description: Download logs and optional checkpoint weights for a named remote experiment.
---

# Fetch experiment results

Run from the project root:

```bash
scripts/fetch_results.py EXPERIMENT_NAME
```

Use `--checkpoints` only when local checkpoint weights are required because a full progression can be large. Results are written to `build/runs/EXPERIMENT_NAME/` by default, and a detached run's remote exit code is fetched when present.
