---
name: rent-vast-gpu
description: Rent a full verified CUDA-compatible Vast.ai GPU and persist project connection state.
---

# Rent a Vast GPU

Use `scripts/rent_vast.py` from the project root.

```bash
scripts/rent_vast.py --gpu "RTX 5090" --min-driver 580 --max-price 0.65 --disk 100 --timeout 0
```

- Use `--dry-run` to inspect the selected offer without spending money.
- Keep the full-GPU, reliability, PCIe, network, driver, and all-in hourly-price gates enabled.
- `--timeout 0` waits indefinitely for SSH readiness; a finite value triggers automatic teardown on startup failure.
- Connection and cost metadata is written to `build/vast/instance.json`.
- Never bypass an existing active state file or rent a second instance for the same experiment.
