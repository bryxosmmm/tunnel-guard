"""Render real raw-cloud evidence for human review; this does not certify labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()
    annotation = json.loads(args.annotations.read_text())
    bags = {f['bag'] for f in annotation['frames']}
    if len(bags) != 1:
        raise ValueError('Render one bag at a time')
    bag = next(iter(bags))
    rows = sorted(annotation['frames'], key=lambda f: f['frame'])
    predictions = {r['frame']: r for r in map(json.loads, args.predictions.read_text().splitlines())
                   if r['bag'] == bag}
    # One panel row per authored object; never choose the best-looking frames.
    entries = [(row['frame'], obj) for row in rows for obj in row['objects']]
    summary = []
    for offset in range(0, len(entries), 6):
        batch = entries[offset:offset + 6]
        fig, axes = plt.subplots(6, 2, figsize=(13, 19))
        for i, (frame, obj) in enumerate(batch):
            with np.load(args.evidence / f'{bag}_{frame:06d}.npz') as evidence:
                points = evidence['points']
            low, high = np.array(obj['bbox_min']), np.array(obj['bbox_max'])
            nearby = np.all((points >= low - [1, 1, .5]) & (points <= high + [1, 1, .5]), axis=1)
            inside = np.all((points >= low) & (points <= high), axis=1)
            summary.append({'frame': frame, 'event_id': obj['event_id'],
                            'points_in_box': int(inside.sum())})
            for j, (x, y) in enumerate(((0, 1), (0, 2))):
                ax = axes[i, j]
                ax.scatter(points[nearby, x], points[nearby, y], s=.2, c='gray')
                ax.scatter(points[inside, x], points[inside, y], s=.8, c='black')
                ax.add_patch(Rectangle((low[x], low[y]), high[x] - low[x], high[y] - low[y],
                                       fill=False, color='blue', lw=1.5))
                for candidate in predictions.get(frame, {}).get('objects', []):
                    cmin, cmax = np.array(candidate['bbox_min']), np.array(candidate['bbox_max'])
                    if np.all(cmax >= low) and np.all(cmin <= high):
                        ax.add_patch(Rectangle((cmin[x], cmin[y]), cmax[x] - cmin[x], cmax[y] - cmin[y],
                                               fill=False, color='orange', lw=.6))
                ax.set_xlim(low[x] - 1, high[x] + 1)
                ax.set_ylim(low[y] - .5, high[y] + .5)
                ax.set_aspect('equal')
                ax.set_xlabel('xyz'[x] + ' m')
                ax.set_ylabel('xyz'[y] + ' m')
                ax.set_title(f'frame {frame}, points {inside.sum()}\n'
                             f'{obj.get("annotation_origin", "unspecified")}')
        for ax in axes[len(batch):].flat:
            ax.set_visible(False)
        fig.suptitle('REAL measured points | blue: provisional label | orange: detector candidates\n'
                     'Review required; projections do not establish identity or full object bounds')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(args.evidence / f'review_{batch[0][0]}.png', dpi=130)
        plt.close(fig)
    (args.evidence / 'support-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Rendered {len(entries)} object observations; point counts include background inside boxes')


if __name__ == '__main__':
    main()
