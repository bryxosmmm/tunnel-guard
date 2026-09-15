---
name: destroy-vast-gpu
description: Destroy the project's tracked Vast.ai instance with one API command and record the local teardown state.
---

# Destroy the Vast instance

Run exactly once after required artifacts and validation results are local:

```bash
scripts/destroy_vast.py
```

Do not poll after the destroy request. Use `--instance-id ID` only to clean up an untracked instance, and use `--dry-run` when only command inspection is intended.
