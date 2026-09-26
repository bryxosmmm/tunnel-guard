#!/usr/bin/env bash
# Fixed issue #23 evaluation. Sequential actual detector runs; no network or tests.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
exec .venv/bin/python -c '
import json, os, subprocess, sys
from pathlib import Path
recipe = "configs/autoresearch-issue23.json"
plan = json.loads(Path(recipe).read_text())
os.environ.update(plan["thread_environment"])
subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], check=True)
os.execv(sys.executable, [sys.executable, "scripts/benchmark_issue23.py", recipe])
'
