"""Summarize real coverage and modeled-insertion observations separately."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .run import digest, write_json
from .visualization import corridor_edges


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def render_real(run, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    config = json.loads((run / 'detector.json').read_text())
    for file in sorted(run.glob('*.jsonl')):
        rows = [r for r in read_rows(file) if 'diagnostic_points' in r]
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 2, figsize=(13, len(rows)*3), squeeze=False)
        for i, row in enumerate(rows):
            with np.load(run / row['diagnostic_points']) as data:
                scene = data['registered_points' if 'registered_points' in data else 'decoded_points'][::10]
                points = data['cluster_points'] if 'cluster_points' in data else np.empty((0, 3))
                labels = data['cluster_labels'] if 'cluster_labels' in data else np.empty(0, dtype=int)
                for j, dim in enumerate([1, 2]):
                    ax = axes[i, j]
                    ax.scatter(scene[:, 0], scene[:, dim], s=.3, c='.7', rasterized=True)
                    for obj in row['objects']:
                        if obj['path_relation'] == 'adjacent':
                            continue
                        q = points[labels == obj['component_id']]
                        color = 'red' if obj['intersection_confirmed'] else 'orange' if obj['confirmed'] else 'gold'
                        ax.scatter(q[:, 0], q[:, dim], s=2, c=color, rasterized=True)
                    edges = corridor_edges(row['geometry'], config).reshape(-1, 2, 3)
                    ax.add_collection(LineCollection(edges[:, :, [0, dim]], colors='teal', linewidths=.5))
                    ax.set(xlim=(0, 80), ylim=(-5, 5), xlabel='forward x [m]',
                           ylabel=('lateral y' if dim == 1 else 'z')+' [m]')
                    ax.set_title(f"frame {row['frame']}: {row['status']}; health={row['health']}\nnearest potential hazard={row['nearest_obstacle_m']} m", fontsize=8)
        fig.suptitle(f'{file.stem}: REAL measured scans (display background 1/10), full candidate representatives\nRed: confirmed intersection; orange: confirmed potential hazard; yellow: tentative; teal: reference contour')
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(destination / f'{file.stem}_review.png', dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--injection', type=Path, required=True)
    parser.add_argument('--background-followup', type=Path)
    parser.add_argument('--control-baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    real = []
    for path in sorted(args.real.glob('*.jsonl')):
        rows = read_rows(path)
        real.append({'bag': path.stem, 'frames': len(rows), 'prediction_sha256': digest(path),
            'statuses': dict(Counter(r['status'] for r in rows)),
            'health': dict(Counter(r['health'] for r in rows)),
            'health_reasons': dict(Counter(k for r in rows for k in r.get('health_reasons', []))),
            'geometry_valid_frames': sum(r.get('geometry', {}).get('valid', False) for r in rows),
            'motion_valid_frames': sum(r.get('motion', {}).get('valid', False) for r in rows),
            'candidate_observations': sum(len(r['objects']) for r in rows),
            'intersection_confirmed_observations': sum(o['intersection_confirmed'] for r in rows for o in r['objects']),
            'precision': None, 'recall': None})
    observations = read_rows(args.injection / 'observations.jsonl')
    sampling = json.loads((args.injection / 'sampling.json').read_text())['cases']
    injection = json.loads((args.injection / 'summary.json').read_text())
    for entry in injection['cases']:
        rows = [r for r in observations if r['case'] == entry['case']]
        entry['case_definition'] = rows[0]['case_definition']
        entry['unique_inserted_positions_1um'] = [r['insertion'].get('unique_inserted_positions_1um', 0) for r in rows]
        entry['median_inserted_returns'] = float(np.median(entry['inserted_returns']))
        entry['median_stage_inserted_representatives'] = {
            k: float(np.median([r['stage_inserted_representatives'][k] for r in rows]))
            if all(r['stage_inserted_representatives'][k] is not None for r in rows) else None
            for k in rows[0]['stage_inserted_representatives']}
        qualifying = [r for r in rows if any(o['inserted_fraction'] >= .5 and o['intersection_confirmed']
                                            for o in r['candidates_with_inserted_support'])]
        entry['first_target_dominated_intersection_frame'] = qualifying[0]['frame'] if qualifying else None
        entry['acquisition_seconds_until_first_intersection'] = ((qualifying[0]['measurement_timestamp_ns'] - rows[0]['measurement_timestamp_ns']) / 1e9
                                                                 if qualifying else None)
        entry['sampling'] = sampling.get(entry['case'])
    baseline = {r['frame']: r for r in read_rows(args.control_baseline)}
    controls = [r for r in read_rows(args.injection / 'predictions.jsonl') if r['case'] == 'recorded_control']
    keys = ['track_id', 'bbox_min', 'bbox_max', 'confirmed', 'intersection_confirmed', 'path_relation',
            'support_voxels', 'in_envelope_voxels', 'distance_m']
    changed = [r['frame'] for r in controls if r['frame'] not in baseline or
               r['status'] != baseline[r['frame']]['status'] or
               [[o[k] for k in keys] for o in r['objects']] != [[o[k] for k in keys] for o in baseline[r['frame']]['objects']]]
    prefix = [r for r in read_rows(args.real / args.control_baseline.name) if r['frame'] in baseline]
    prefix_changed = [r['frame'] for r in prefix if r['status'] != baseline[r['frame']]['status'] or
                      [[o[k] for k in keys] for o in r['objects']] != [[o[k] for k in keys] for o in baseline[r['frame']]['objects']]]
    report = {'real_recordings': real, 'modeled_insertions': injection,
        'report_source_sha256': digest(Path(__file__)),
        'real_prefix_comparison': {'frames': len(prefix), 'changed_frames': prefix_changed},
        'recorded_control_comparison': {'frames': len(controls), 'changed_frames': changed,
                                        'baseline_sha256': digest(args.control_baseline)},
        'sources': {str(p): digest(p) for p in [args.real/'manifest.json', args.injection/'manifest.json',
                                               args.injection/'observations.jsonl', args.injection/'sampling.json']}}
    if args.background_followup is not None:
        followup = args.background_followup
        experiment = json.loads((followup / 'experiment.json').read_text())
        source_bag = Path(experiment['bag']).name
        baseline_file = args.control_baseline.parent / f'{source_bag}.jsonl'
        old = {r['frame']: r for r in read_rows(baseline_file)}
        control = [r for r in read_rows(followup / 'predictions.jsonl') if r['case'] == 'recorded_control']
        different = [r['frame'] for r in control if r['frame'] not in old or r['status'] != old[r['frame']]['status'] or
                     [[o[k] for k in keys] for o in r['objects']] != [[o[k] for k in keys] for o in old[r['frame']]['objects']]]
        report['adaptive_background_followup'] = json.loads((followup / 'summary.json').read_text()) | {
            'hypothesis': experiment['hypothesis'], 'source_bag': source_bag,
            'sampling': json.loads((followup / 'sampling.json').read_text())['cases'],
            'control_comparison': {'frames': len(control), 'changed_frames': different, 'baseline_sha256': digest(baseline_file)}}
        report['sources'].update({str(p): digest(p) for p in [followup/'manifest.json', followup/'observations.jsonl', followup/'sampling.json']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries = [e for e in injection['cases'] if e['case'] != 'recorded_control']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y = np.arange(len(entries))
    for offset, key, label in [(-.18, 'median_inserted_returns', 'Inserted return slots'),
                               (.18, 'median_stage_inserted_representatives', 'Input to clustering')]:
        values = [e[key]['cluster_points'] if isinstance(e[key], dict) else e[key] for e in entries]
        axes[0].barh(y+offset, values, height=.32, label=label)
    axes[0].set_xscale('symlog', linthresh=1)
    axes[0].set_yticks(y, [e['case'] for e in entries]); axes[0].invert_yaxis()
    axes[0].set_xlabel('Median points per scan (linear near zero, log above 1)'); axes[0].legend()
    for offset, key, label in [(-.18, 'frames_with_confirmed_target_dominated_object', 'Object confirmed'),
                               (.18, 'frames_with_confirmed_target_dominated_intersection', 'Intersection confirmed')]:
        axes[1].barh(y+offset, [e[key] for e in entries], height=.32, label=label)
    axes[1].set_yticks(y, [e['case'] for e in entries]); axes[1].invert_yaxis()
    axes[1].set_xlabel('Frames of 10; target attribution >=50% inserted support'); axes[1].legend()
    fig.suptitle('HYBRID SYNTHETIC experiment on recorded directions — not field recall or hardware range')
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=140); plt.close(fig)
    render_real(args.real, args.plot.parent)
    print(json.dumps({'real_recordings': real, 'recorded_control_comparison': report['recorded_control_comparison']}, indent=2))


if __name__ == '__main__':
    main()
