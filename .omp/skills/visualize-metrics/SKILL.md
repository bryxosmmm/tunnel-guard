---
name: visualize-metrics
description: Render standalone HTML charts and CSV summaries from one or more metric runs.
---

# Visualize training metrics

Render one run:

```bash
scripts/visualize_metrics.py build/runs/EXPERIMENT_NAME
```

Compare runs by passing multiple directories and an explicit output path:

```bash
scripts/visualize_metrics.py build/runs/RUN_A build/runs/RUN_B --output build/metrics-comparison.html
```

The renderer requires no plotting dependencies, ignores malformed non-metric JSON, flags non-finite values, and never mutates source metric files.
