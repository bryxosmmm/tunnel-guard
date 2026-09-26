"""Run the frozen issue #24 HYBRID SYNTHETIC panel with one CLI command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .run import digest, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new immutable evidence directory")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["label"] != "HYBRID SYNTHETIC":
        raise ValueError("The issue #24 panel must carry its synthetic label")
    configs = {name: json.loads(Path(path).read_text()) for name, path in plan["configs"].items()}
    if len({configs[name]["background_bag"] for name in ("tuning", "evaluation")}) != 2:
        raise ValueError("Tuning and evaluation must use disjoint background bags")
    if len({configs[name]["scenario_seed"] for name in ("tuning", "evaluation")}) != 2:
        raise ValueError("Tuning and evaluation must use disjoint scenario seeds")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    recipes = output / "recipes"
    recipes.mkdir()
    poses = output / "poses"
    configs["poses"]["output"] = str(poses)
    for name in ("tuning", "evaluation"):
        configs[name]["output"] = str(output / name)
        configs[name]["pose_run"] = str(poses)
    for name, config in configs.items():
        write_json(recipes / f"{name}.json", config)
    write_json(output / "plan.json", plan | {"input_plan_sha256": digest(args.plan),
                                             "input_config_sha256": {name: digest(Path(path))
                                                                     for name, path in plan["configs"].items()}})
    completed = []
    commands = [
        ("poses", [sys.executable, "-m", "tunnel_guard.run", "--experiment", str(recipes / "poses.json")]),
        ("tuning", [sys.executable, "-m", "tunnel_guard.realistic_stress", "--experiment", str(recipes / "tuning.json")]),
        ("tuning-report", [sys.executable, "-m", "tunnel_guard.realistic_report", "--run", str(output / "tuning"),
                           "--output", str(output / "tuning" / "report.json")]),
        ("evaluation", [sys.executable, "-m", "tunnel_guard.realistic_stress", "--experiment", str(recipes / "evaluation.json")]),
        ("evaluation-report", [sys.executable, "-m", "tunnel_guard.realistic_report", "--run", str(output / "evaluation"),
                               "--output", str(output / "evaluation" / "report.json")]),
    ]
    for name, command in commands:
        print(f"HYBRID SYNTHETIC {name}: {' '.join(command)}", flush=True)
        with (output / f"{name}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        completed.append({"name": name, "command": command, "exit_code": result.returncode})
        write_json(output / "run_state.json", {"label": "HYBRID SYNTHETIC", "completed": completed})
        if result.returncode:
            raise SystemExit(f"{name} failed with exit code {result.returncode}; see {output / f'{name}.log'}")
    print(f"HYBRID SYNTHETIC evidence: {output}")


if __name__ == "__main__":
    main()
