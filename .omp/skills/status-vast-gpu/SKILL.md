---
name: status-vast-gpu
description: Report the saved and live status, price, GPU, and SSH endpoint of the project's Vast.ai rental.
---

# Inspect Vast status

Run from the project root:

```bash
scripts/status_vast.py
```

Use `--state PATH` only when the experiment intentionally uses a non-default state file. This command is read-only and must not replace `scripts/destroy_vast.py` for teardown.
