"""Build one-split subsets of a rosbag2 recording, for sampling it without copying it.

The extended recording is one continuous 20-minute run stored as 221 SQLite splits. Sampling places
spread along it means reading single splits, which the reader will not do while metadata.yaml lists all
221. This writes a subset directory per requested split, symlinked rather than copied, so sampling an
84 GiB recording costs kilobytes per sample.

    python -m tunnel_guard.extended_subset --bag data/new_data --indices 0 9 18 --out data/extended_subset

Editing is line-level, so the source's timestamp blocks and QoS strings are preserved verbatim; only the
global count, the duration and the file list are recomputed for the subset.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_entries(lines: list[str]) -> dict[str, tuple[int, int, int]]:
    """Each `files:` path mapped to (starting_time, duration, message_count)."""
    try:
        start = next(i for i, line in enumerate(lines) if line == "  files:\n")
    except StopIteration:
        raise SystemExit("metadata has no `files:` block; cannot map splits")
    entries: dict[str, tuple[int, int, int]] = {}
    name: str | None = None
    numbers: list[int] = []
    for line in lines[start + 1:]:
        if line.startswith("  relative_file_paths:"):
            break
        match = re.match(r"- path: (\S+)", line.strip())
        if match:
            if name and len(numbers) == 3:
                entries[name] = (numbers[0], numbers[1], numbers[2])
            name, numbers = match.group(1), []
            continue
        number = re.match(r"\s*(?:nanoseconds(?:_since_epoch)?|message_count): (\d+)", line)
        if name and number:
            numbers.append(int(number.group(1)))
    if name and len(numbers) == 3:
        entries[name] = (numbers[0], numbers[1], numbers[2])
    return entries


def block_bounds(lines: list[str], header: str) -> tuple[int, int]:
    """The `header` line and the deeper-indented lines that follow it."""
    start = next(i for i, line in enumerate(lines) if line == header)
    end = start + 1
    while end < len(lines) and lines[end].startswith((" ", "\t")) and lines[end].strip():
        end += 1
    return start, end


def set_scalar(lines: list[str], header: str, value: int) -> list[str]:
    """Set the numeric field directly under `header` (which is followed by its key line)."""
    at = next(i for i, line in enumerate(lines) if line == header)
    for i in (at + 1, at + 2):
        if i < len(lines) and re.match(r"\s+\w+: \d+$", lines[i].rstrip("\n")):
            lines[i] = re.sub(r"\d+$", str(value), lines[i].rstrip("\n")) + "\n"
            return lines
    raise SystemExit(f"no scalar under {header!r}")


def build(bag: Path, indices: list[int], out: Path, group: int = 1) -> list[Path]:
    """One subset per start index, containing `group` consecutive splits.

    A group longer than one split is what makes the subset a stretch of travel rather than a snapshot:
    adjacent frames inside one split advance about a metre, which is too little to re-measure ground a
    continuation predicted 20-80 m ahead. Consecutive splits are the same run continued, so the reader
    sees one stream and the motion estimator carries across the boundary.
    """
    source = (bag / "metadata.yaml").read_text().splitlines(keepends=True)
    entries = parse_entries(source)
    wanted_all = [f"{bag.name}_{i}.db3" for i, in ((i,) for i in indices)]
    missing = [name for name in wanted_all if name not in entries]
    if missing:
        raise SystemExit(f"no such splits: {missing[:3]}")

    built = []
    for first in indices:
        names = [f"{bag.name}_{first + offset}.db3" for offset in range(group)]
        absent = [name for name in names if name not in entries]
        if absent:
            raise SystemExit(f"no such splits: {absent[:3]}")
        name = names[0]
        start = min(entries[n][0] for n in names)
        duration = max(entries[n][0] + entries[n][1] for n in names) - start
        count = sum(entries[n][2] for n in names)
        lines = list(source)
        entry_lines = []
        for entry in names:
            entry_start, entry_duration, entry_count = entries[entry]
            entry_lines += [f"    - path: {entry}\n",
                            "      starting_time:\n",
                            f"        nanoseconds_since_epoch: {entry_start}\n",
                            "      duration:\n",
                            f"        nanoseconds: {entry_duration}\n",
                            f"      message_count: {entry_count}\n"]
        files = block_bounds(lines, "  files:\n")
        lines = lines[:files[0] + 1] + entry_lines + lines[files[1]:]
        paths = block_bounds(lines, "  relative_file_paths:\n")
        lines = lines[:paths[0] + 1] + [f"    - {entry}\n" for entry in names] + lines[paths[1]:]
        lines = set_scalar(lines, "  duration:\n", duration)
        lines = set_scalar(lines, "  starting_time:\n", start)
        # The two message counts carry their value inline (global at two spaces, the topic's at
        # six), so they are patched as scalars rather than through a header-then-child helper.
        text = "".join(lines)
        text = re.sub(r"(?m)^  message_count: \d+$", f"  message_count: {count}", text, count=1)
        text = re.sub(r"(?m)^      message_count: \d+$", f"      message_count: {count}", text, count=1)
        directory = out / name[:-4]
        directory.mkdir(parents=True, exist_ok=True)
        for entry in names:
            link = directory / entry
            if not link.exists():
                link.symlink_to((bag / entry).resolve())
        (directory / "metadata.yaml").write_text(text)
        built.append(directory)
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("data/extended_subset"))
    parser.add_argument("--group", type=int, default=1,
                        help="consecutive splits per subset; 2 gives ~10 s of travel for pair scoring")
    args = parser.parse_args()
    built = build(args.bag, args.indices, args.out, args.group)
    print(f"{len(built)} subsets under {args.out}: {', '.join(p.name for p in built[:6])} ...")


if __name__ == "__main__":
    main()
