"""Summarize saved detector observations and paired runs without inventing labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .diagnostics import distribution, summarize_observations
from .run import digest, write_json


def load_rows(folder):
    rows = {}
    for path in sorted(folder.glob('*.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                rows[(row['bag'], row['frame'])] = row
    return rows


def panel_summary(rows):
    values = list(rows.values())
    groups = {}
    for (bag, _), row in rows.items():
        groups.setdefault(bag, []).append(row)
    periods = [b['timestamp_s'] - a['timestamp_s'] for group in groups.values() for a, b in zip(group, group[1:])]
    return {'frames': len(rows), 'sequences': len(groups),
            'status_frames': dict(Counter(r['status'] for r in values)),
            'geometry_valid_frames': sum(r.get('geometry', {}).get('valid', False) for r in values),
            'geometry_reasons': dict(Counter(r.get('geometry', {}).get('reason', 'not_run') for r in values)),
            'processing_ms': distribution(r['processing_s'] * 1000 for r in values),
            'acquisition_delta_s': distribution(periods),
            'duplicate_acquisition_deltas': sum(t == 0 for t in periods),
            'backward_acquisition_deltas': sum(t < 0 for t in periods),
            'source_messages_reported_in_ingestion': None,
            'observations': summarize_observations(values)}


def compare_rows(before, after):
    keys = sorted(before.keys() & after.keys())
    status, decisions, distances = [], [], []
    support_differences, hazard_differences = [], []
    pose_deltas = []
    def decision(row):
        return [(o['track_id'], o['confirmed'], o['path_relation'], o['hits'], o['support_voxels']) for o in row['objects']]
    def support(row, hazards_only=False):
        return Counter((tuple(o['bbox_min'] + o['bbox_max']), o['path_relation'], o['confirmed'])
                       for o in row['objects'] if not hazards_only or
                       (o['confirmed'] and o['path_relation'] in ('intersecting', 'unresolved')))
    for key in keys:
        a, b = before[key], after[key]
        if a['status'] != b['status']: status.append(key)
        if decision(a) != decision(b): decisions.append(key)
        if a['nearest_obstacle_m'] != b['nearest_obstacle_m']: distances.append(key)
        if support(a) != support(b): support_differences.append(key)
        if support(a, True) != support(b, True): hazard_differences.append(key)
        if 'pose' in a and 'pose' in b:
            pose_deltas.append(float(np.max(np.abs(np.asarray(a['pose']) - np.asarray(b['pose'])))))
    return {'shared_frames': len(keys), 'before_only_frames': len(before.keys()-after.keys()),
            'after_only_frames': len(after.keys()-before.keys()),
            'status_different_frames': len(status), 'object_decision_different_frames': len(decisions),
            'nearest_distance_exactly_different_frames': len(distances),
            'support_and_confirmation_different_frames_ignoring_ids': len(support_differences),
            'confirmed_hazard_geometry_different_frames_ignoring_ids': len(hazard_differences),
            'max_pose_matrix_element_abs_delta': max(pose_deltas, default=None),
            'first_status_differences': status[:10], 'first_object_differences': decisions[:10],
            'first_support_differences': support_differences[:10],
            'before_shared_processing_ms': distribution(before[k]['processing_s']*1000 for k in keys),
            'after_shared_processing_ms': distribution(after[k]['processing_s']*1000 for k in keys),
            'note': 'Exact output comparison on shared (bag, frame), not a test suite or physical identity score. Support multisets compare bounds, relation and confirmation without IDs; numeric track IDs across runs do not establish correspondence. Timing is an uncontrolled single-run measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--compare-to', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output path')
    rows = load_rows(args.run)
    if not rows: parser.error('No prediction rows')
    report = {'run': str(args.run), 'predictions_sha256': {p.name: digest(p) for p in sorted(args.run.glob('*.jsonl'))},
              'panel': panel_summary(rows), 'sequences': {}}
    for bag in sorted({key[0] for key in rows}):
        report['sequences'][bag] = panel_summary({k:v for k,v in rows.items() if k[0] == bag})
    summary = args.run/'summary.json'
    if summary.exists():
        report['ingestion'] = {s['bag']:s.get('ingestion') for s in json.loads(summary.read_text())}
        report['panel']['source_messages_reported_in_ingestion'] = sum(s.get('source_messages',0) for s in report['ingestion'].values() if s)
    manifest = args.run/'manifest.json'
    report['manifest_sha256'] = digest(manifest) if manifest.exists() else None
    if args.compare_to:
        report['compare_to'] = str(args.compare_to)
        report['comparison'] = compare_rows(load_rows(args.compare_to), rows)
    report['accuracy_validity'] = 'No precision/recall from these summaries. Synthetic label metrics belong to their separate evaluator. Real alarm counts are unlabeled observations.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'output':str(args.output), 'frames':len(rows), 'comparison':report.get('comparison')}, indent=2))


if __name__ == '__main__':
    main()
