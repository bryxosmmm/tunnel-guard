# Generic experiment workflow

- Keep experiment configuration in explicit project config files; scripts and skills must not hide recipes or silently override them.
- State the hypothesis, success criterion, seed, and artifact paths before spending remote compute.
- Preserve logs and metadata locally before teardown. Every rented instance must end with exactly one `scripts/destroy_vast.py` command, with no polling after it.
- Treat training metrics as diagnostic; make promotion decisions from a fixed, reproducible evaluation panel with validity checks.
