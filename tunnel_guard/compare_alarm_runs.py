"""Compare real detector runs without interpreting unlabelled alarms as accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .run import digest, write_json


def describe(rows):
    return {
        'frames': len(rows),
        'status_frames': dict(Counter(r['status'] for r in rows)),
        'confirmed_observations_by_relation': dict(Counter(
            o['path_relation'] for r in rows for o in r['objects'] if o['confirmed'])),
        'candidate_observations': sum(len(r['objects']) for r in rows),
        'support_voxel_observations': sum(o['support_voxels'] for r in rows for o in r['objects']),
        'processing_ms': {name: float(np.quantile([r['processing_s'] for r in rows], q) * 1000)
                          for name, q in [('p50', .5), ('p95', .95)]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'before': str(args.before), 'after': str(args.after), 'bags': {},
              'warning': 'Counts are algorithm outputs, not accuracy or false-alarm rates. '
                         'Timing includes concurrent local work and different visualization settings.',
              'diagnostic_support': []}
    for before_file in sorted(args.before.glob('*.jsonl')):
        after_file = args.after / before_file.name
        before = [json.loads(line) for line in before_file.read_text().splitlines()]
        after = [json.loads(line) for line in after_file.read_text().splitlines()]
        old = {r['frame']: r for r in before}
        new = {r['frame']: r for r in after}
        common = sorted(old.keys() & new.keys())
        report['bags'][before_file.stem] = {
            'before': describe(before), 'after': describe(after),
            'missing_after': sorted(old.keys() - new.keys()),
            'extra_after': sorted(new.keys() - old.keys()),
            'measurement_timestamp_mismatches': sum(old[f]['measurement_timestamp_ns'] !=
                                                     new[f]['measurement_timestamp_ns'] for f in common),
            'status_transitions': dict(Counter(old[f]['status'] + ' -> ' + new[f]['status'] for f in common)),
            'before_sha256': digest(before_file), 'after_sha256': digest(after_file),
        }
    for before_file in sorted((args.before / 'diagnostics').glob('*.npz')):
        after_file = args.after / 'diagnostics' / before_file.name
        if not after_file.exists():
            report['diagnostic_support'].append({'file': before_file.name, 'missing_after': True})
            continue
        with np.load(before_file) as old, np.load(after_file) as new:
            stages = {}
            for name in ['geometry_voxel_points', 'context_after_background', 'cluster_points']:
                a, b = old[name], new[name]
                distances = cKDTree(b).query(a)[0] if len(b) else np.full(len(a), np.inf)
                stages[name] = {'before': len(a), 'after': len(b),
                                'before_points_absent_after': int(np.count_nonzero(distances > 1e-8))}
            report['diagnostic_support'].append({'file': before_file.name, 'stages': stages})
    write_json(args.output, report)
    print(json.dumps(report['bags'], indent=2))


if __name__ == '__main__':
    main()
