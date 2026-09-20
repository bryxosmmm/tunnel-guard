#!/usr/bin/env python3
"""Project adapter for reproducible Tunnel Guard detector experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tunnel_guard.run import main as run_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="experiment JSON consumed by tunnel_guard.run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="run an explicitly bounded recipe")
    mode.add_argument("--full", action="store_true", help="run the recipe exactly as declared")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        help="new immutable output path; materializes the effective recipe beside it")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    recipe = json.loads(args.config.read_text())
    missing = [name for name in ("seed", "detector_config", "output", "bags", "hypothesis", "criterion")
               if name not in recipe]
    if missing:
        raise ValueError(f"Experiment recipe is missing required provenance: {missing}")
    if args.resume:
        raise ValueError("Runs are immutable and cannot be resumed; use a new output path")
    if args.smoke and recipe.get("max_frames") is None:
        raise ValueError("A smoke recipe must declare max_frames explicitly")
    config_path = args.config
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Evidence directory already exists: {args.output}")
        recipe["output"] = str(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        config_path = args.output.parent / f".{args.output.name}.experiment.json"
        config_path.write_text(json.dumps(recipe, indent=2) + "\n")
    output = Path(recipe["output"])
    if output.exists():
        raise FileExistsError(f"Evidence directory already exists: {output}")

    sys.argv = ["tunnel_guard.run", "--experiment", str(config_path),
                "--workers", str(max(1, args.workers))]
    run_main()


if __name__ == "__main__":
    main()
