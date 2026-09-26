"""Render measured regions and nominal/path-interval evidence, not ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .detector import load_config
from .geometry import TrackGeometry
from .run import write_json
from .visualization import corridor_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.before / 'detector.json')
    plan = json.loads(args.cases.read_text())
    bag, cases = plan['bag'], plan['cases']
    rows = {r['frame']: r for r in map(json.loads, (args.before / f'{bag}.jsonl').read_text().splitlines())}
    fig, axes = plt.subplots(len(cases), 3, figsize=(17, 4.3 * len(cases)), squeeze=False)
    report = []
    for row, case in enumerate(cases):
        frame, low, high = case['frame'], np.array(case['low']), np.array(case['high'])
        description = rows[frame]['geometry']
        # Reuse the recorded estimate: changing the criterion must not refit the path.
        geometry = object.__new__(TrackGeometry)
        geometry.config = config
        geometry.background = None
        geometry.plane = np.asarray(description['ground_plane'])
        geometry.ground_anchors = np.asarray(description['ground_anchors'])
        geometry.rail_anchors = np.asarray(description['rail_anchors'])
        geometry.rail_head_height_m = description['rail_head_height_m']
        with np.load(args.before / 'diagnostics' / f'{bag}_{frame:06d}.npz') as data:
            points = data['decoded_points']
        points = points[np.all((points >= low) & (points <= high), axis=1)]
        core, _, height, observed, nominal, boundary = geometry.classify(
            points, remove_background=False, include_boundary=True)
        _, uncertainty = geometry.ground(points)
        running_height = (height - geometry.rail_head_height_m) / np.sqrt(1 + np.sum(geometry.plane[:2] ** 2))
        old = observed & nominal & (running_height >= config['envelope_segments_m'][0][0] + uncertainty)
        unresolved = (~observed & nominal) | boundary
        report.append({'frame': frame, 'region': case['name'], 'raw_returns': len(points),
                       'nominal_interior_before': int(old.sum()), 'interval_interior_after': int(core.sum()),
                       'boundary_uncertain': int(boundary.sum()),
                       'unsupported_nominal': int((~observed & nominal).sum())})
        edges = corridor_edges(description, config).reshape(-1, 2, 3)
        # Only this longitudinal slice belongs in the cross-section projection.
        edges = edges[(edges[:, :, 0].min(axis=1) >= low[0]) & (edges[:, :, 0].max(axis=1) <= high[0])]
        views = [((0, 1), old, 'BEFORE: nominal crossing'),
                 ((0, 1), core, 'AFTER: path interval'),
                 ((1, 2), core, 'Cross section YZ (after)')]
        for col, ((x, y), mask, title) in enumerate(views):
            ax = axes[row, col]
            ax.scatter(points[:, x], points[:, y], s=1, c='.65', rasterized=True)
            if col > 0:
                ax.scatter(points[unresolved, x], points[unresolved, y], s=2, c='orange', rasterized=True)
            ax.scatter(points[mask, x], points[mask, y], s=2, c='red', rasterized=True)
            ax.add_collection(LineCollection(edges[:, :, [x, y]], colors='cyan', linewidths=.6, alpha=.65))
            ax.set_xlim(low[x], high[x])
            ax.set_ylim(low[y], high[y])
            ax.set_aspect('equal')
            ax.set_xlabel('xyz'[x] + ' m')
            ax.set_ylabel('xyz'[y] + ' m')
            ax.set_title(f"{case['name']}, frame {frame}\n{title}")
    fig.suptitle('Actual LiDAR returns, fixed regions | red: supported interior | orange: unresolved | cyan: nominal contour\n'
                 'No verified negatives; path uncertainty is a heuristic, not calibrated confidence')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.output / 'cases.png', dpi=150)
    plt.close(fig)
    write_json(args.output / 'cases.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
