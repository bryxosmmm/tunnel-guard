#!/usr/bin/env python3
"""Render self-contained HTML and CSV from immutable Tunnel Guard run folders."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


STATUS_COLORS = {
    "obstacle": "#ef4444",
    "unresolved_obstacle": "#f59e0b",
    "candidate": "#eab308",
    "no_obstacle_observed": "#22c55e",
    "unknown": "#64748b",
}


def load_run(path: Path, label: str) -> dict:
    summaries = json.loads((path / "summary.json").read_text())
    bags = []
    for summary in summaries:
        rows = [json.loads(line) for line in (path / f"{summary['bag']}.jsonl").open()]
        processing = [row["processing_s"] * 1000 for row in rows]
        bags.append({
            "name": summary["bag"],
            "frames": len(rows),
            "processing": processing,
            "p50": summary["processing_ms"]["p50"],
            "p95": summary["processing_ms"]["p95"],
            "p99": summary["processing_ms"]["p99"],
            "statuses": summary["status_frames"],
            "objects": sum(len(row["objects"]) for row in rows),
            "confirmed": sum(sum(obj["confirmed"] for obj in row["objects"]) for row in rows),
            "geometry_valid": summary["geometry_valid_frames"],
        })
    return {"path": str(path), "label": label, "bags": bags}


def sparkline(values: list[float], color: str) -> str:
    width, height, pad = 460, 92, 8
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = pad + index * (width - 2 * pad) / max(1, len(values) - 1)
        y = height - pad - (value - low) * (height - 2 * pad) / span
        points.append(f"{x:.1f},{y:.1f}")
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Processing time by frame"><polyline points="{" ".join(points)}" '
            f'fill="none" stroke="{color}" stroke-width="3"/></svg>')


def bar(value: float, maximum: float, color: str) -> str:
    width = 100 * value / maximum if maximum else 0
    return f'<span class="bar"><i style="width:{width:.2f}%;background:{color}"></i></span>'


def render(runs: list[dict], output: Path) -> None:
    colors = ["#38bdf8", "#a78bfa", "#34d399", "#fb7185"]
    all_bags = [bag for run in runs for bag in run["bags"]]
    max_p95 = max((bag["p95"] for bag in all_bags), default=1)
    sections = []
    for run_index, run in enumerate(runs):
        color = colors[run_index % len(colors)]
        bag_rows = []
        charts = []
        for bag in run["bags"]:
            statuses = " ".join(
                f'<span class="chip" style="border-color:{STATUS_COLORS.get(name, "#64748b")}">'
                f'{html.escape(name)} {count}</span>' for name, count in sorted(bag["statuses"].items()))
            bag_rows.append(
                f'<tr><td>{html.escape(bag["name"])}</td><td>{bag["frames"]}</td>'
                f'<td>{bag["p50"]:.1f}</td><td>{bag["p95"]:.1f} '
                f'{bar(bag["p95"], max_p95, color)}</td><td>{bag["objects"]}</td>'
                f'<td>{bag["confirmed"]}</td><td>{statuses}</td></tr>')
            charts.append(
                f'<article><h3>{html.escape(bag["name"])}</h3>{sparkline(bag["processing"], color)}'
                f'<p>Кадры по оси X; время ядра {min(bag["processing"]):.0f}–{max(bag["processing"]):.0f} мс.</p></article>')
        sections.append(
            f'<section><h2>{html.escape(run["label"])}</h2><p class="path">{html.escape(run["path"])}</p>'
            '<table><thead><tr><th>Запись</th><th>Кадры</th><th>p50, мс</th><th>p95, мс</th>'
            f'<th>Объекты</th><th>Подтверждены</th><th>Статусы</th></tr></thead><tbody>{"".join(bag_rows)}</tbody></table>'
            f'<div class="charts">{"".join(charts)}</div></section>')

    comparison = ""
    if len(runs) >= 2:
        base = {bag["name"]: bag for bag in runs[0]["bags"]}
        rows = []
        for bag in runs[1]["bags"]:
            if bag["name"] not in base:
                continue
            before = base[bag["name"]]["p50"]
            gain = 100 * (before - bag["p50"]) / before
            rows.append(f'<tr><td>{html.escape(bag["name"])}</td><td>{before:.1f}</td>'
                        f'<td>{bag["p50"]:.1f}</td><td class="gain">{gain:+.1f}%</td></tr>')
        comparison = ('<section><h2>Сравнение p50</h2><table><thead><tr><th>Запись</th>'
                      '<th>До, мс</th><th>После, мс</th><th>Ускорение</th></tr></thead>'
                      f'<tbody>{"".join(rows)}</tbody></table></section>')

    document = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Tunnel Guard — метрики реальных прогонов</title>
<style>
:root{{--bg:#07111f;--card:#0f1d2e;--text:#e5eef8;--muted:#93a4b8;--line:#263a50}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#07111f,#0b1527 60%,#10243a);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:36px 22px 70px}} h1{{font-size:clamp(30px,5vw,58px);line-height:1.02;margin:0 0 12px}} h2{{margin-top:0}} h3{{margin:0 0 4px}}
.lead,.path,article p{{color:var(--muted)}} section,article{{background:rgba(15,29,46,.88);border:1px solid var(--line);border-radius:16px;padding:22px;margin-top:22px;box-shadow:0 16px 50px #0005}}
table{{width:100%;border-collapse:collapse;overflow:hidden}} th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:#a9c4df;font-size:12px;text-transform:uppercase;letter-spacing:.08em}}
.bar{{display:inline-block;width:110px;height:8px;background:#1c2c40;border-radius:9px;overflow:hidden;vertical-align:middle;margin-left:8px}} .bar i{{display:block;height:100%}}
.chip{{display:inline-block;border:1px solid;border-radius:999px;padding:2px 7px;margin:1px;font-size:12px}} .charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}} article{{margin-top:16px;padding:16px}} svg{{width:100%;height:92px;background:#091524;border-radius:10px}} .gain{{color:#5ee7a8;font-weight:750}}
@media(max-width:760px){{section{{overflow-x:auto}}.bar{{display:none}}}}
</style></head><body><main><h1>Tunnel Guard: проверяемое ускорение</h1>
<p class="lead">Реальные кадры ROS bag. Графики показывают время вычислительного ядра, а статусы — выход алгоритма, не ground truth.</p>
{comparison}{''.join(sections)}
</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


def write_summary(runs: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "bag", "frames", "p50_ms", "p95_ms", "p99_ms",
                                                        "objects", "confirmed", "geometry_valid_frames", "statuses"])
        writer.writeheader()
        for run in runs:
            for bag in run["bags"]:
                writer.writerow({"run": run["label"], "bag": bag["name"], "frames": bag["frames"],
                                 "p50_ms": f"{bag['p50']:.6f}", "p95_ms": f"{bag['p95']:.6f}",
                                 "p99_ms": f"{bag['p99']:.6f}", "objects": bag["objects"],
                                 "confirmed": bag["confirmed"], "geometry_valid_frames": bag["geometry_valid"],
                                 "statuses": json.dumps(bag["statuses"], sort_keys=True)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--labels", help="comma-separated labels, in --run order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels.split(",") if args.labels else [path.name for path in args.run]
    if len(labels) != len(args.run):
        raise ValueError("--labels must contain one comma-separated label per --run")
    runs = [load_run(path, label.strip()) for path, label in zip(args.run, labels)]
    render(runs, args.output)
    write_summary(runs, args.summary)
    print(f"Visualization: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
