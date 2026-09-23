"""Audit detector recipe references and removed backend selectors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

REMOVED_KEYS = {"native_kernels", "voxel_backend"}
def find_removed_keys(value, prefix=""):
    found=[]
    if isinstance(value, dict):
        for key, child in value.items():
            path=f"{prefix}.{key}" if prefix else key
            if key in REMOVED_KEYS: found.append(path)
            found.extend(find_removed_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value): found.extend(find_removed_keys(child, f"{prefix}[{index}]"))
    return found
def audit(root: Path = Path(".")) -> dict:
    references=set(); removed=[]
    for document in [root / "README.md", *sorted((root / "docs").glob("*.md"))]:
        if document.exists():
            import re
            references.update(re.findall(r"configs/(detector[A-Za-z0-9._-]*\.json)", document.read_text()))
    for recipe in sorted((root / "configs").glob("*.json")):
        try: data=json.loads(recipe.read_text())
        except json.JSONDecodeError: continue
        removed.extend(f"{recipe.name}:{key}" for key in find_removed_keys(data))
        if isinstance(data, dict) and isinstance(data.get("detector_config"), str):
            references.add(Path(data["detector_config"]).name)
    missing=sorted(name for name in references if not (root / "configs" / name).exists())
    return {"references":sorted(references), "missing":missing, "removed_backend_keys":sorted(removed)}
def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=Path(".")); parser.add_argument("--fail-on-drift", action="store_true")
    args=parser.parse_args(); report=audit(args.root)
    print(f"  detector configs that do not exist: {report['missing'] or 'none'}")
    print(f"  removed backend keys: {report['removed_backend_keys'] or 'none'}")
    if args.fail_on_drift and (report["missing"] or report["removed_backend_keys"]): raise SystemExit("configuration drift found")
if __name__ == "__main__": main()
