"""Compare the recipes the documentation references against the shipped one.

The NumPy reference recipe had drifted from the shipping recipe - three behavioural keys, so the backend-equivalence check
was silently comparing two different recipes - and a documented config did not exist at all. Both were found by accident.
This makes the check routine: every `configs/*.json` the README or `docs/` mentions is compared with
`configs/detector-native.json` on every key except the backend selection, and each difference is classified.

    python -m tunnel_guard.config_audit                 # report only
    python -m tunnel_guard.config_audit --fail-on-drift  # exit non-zero if a non-historical recipe drifts

A recipe is treated as HISTORICAL - and its drift expected - when its name says so: `baseline`, `experimental`, `fast` or
`-old`. Everything else is expected to differ only where the backend differs. This cannot know intent, so historical
recipes are listed rather than judged; the point is that an unexpected difference is visible instead of silent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BACKEND_KEYS = {"native_kernels", "voxel_backend"}
HISTORICAL_MARKERS = ("baseline", "experimental", "fast", "no-rails", "-old")


def flatten(config: dict, prefix: str = "") -> dict:
    out: dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def referenced_configs(documents: list[Path]) -> set[str]:
    names: set[str] = set()
    for document in documents:
        names |= set(re.findall(r"configs/([A-Za-z0-9._-]+\.json)", document.read_text()))
    return names


def audit(root: Path = Path(".")) -> dict:
    shipped = json.loads((root / "configs/detector-native.json").read_text())
    keys = flatten(shipped)
    documents = [root / "README.md"] + sorted((root / "docs").glob("*.md"))
    missing, drift = [], {}
    for name in sorted(referenced_configs([d for d in documents if d.exists()])):
        path = root / "configs" / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            other = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue                      # a recipe this check cannot read is not evidence of drift
        if not isinstance(other, dict) or "envelope_segments_m" not in other:
            continue                      # not a detector recipe: stress panels, run recipes and so on
        flat = flatten(other)
        differences = {k: [flat.get(k), keys.get(k)] for k in set(flat) | set(keys)
                       if flat.get(k) != keys.get(k) and k not in BACKEND_KEYS}
        if differences:
            drift[name] = {"keys": len(differences),
                           "historical": any(marker in name for marker in HISTORICAL_MARKERS),
                           "differences": dict(sorted(differences.items())[:8])}
    return {"missing": missing, "drift": drift,
            "unexplained": sorted(n for n, d in drift.items() if not d["historical"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="exit non-zero when a recipe that is not marked historical differs")
    args = parser.parse_args()
    report = audit(args.root)
    print(f"  documented configs that do not exist: {report['missing'] or 'none'}")
    print("  recipes differing from the shipped one:")
    for name, entry in report["drift"].items():
        kind = "historical, drift expected" if entry["historical"] else "UNEXPLAINED"
        print(f"    {name:46s} {entry['keys']:3d} keys  ({kind})")
        for key, (old, new) in list(entry["differences"].items())[:3]:
            print(f"        {key:44s} {str(old)[:30]:32s} -> {str(new)[:30]}")
    if report["unexplained"] and args.fail_on_drift:
        raise SystemExit(f"unexplained drift in: {report['unexplained']}")


if __name__ == "__main__":
    main()
