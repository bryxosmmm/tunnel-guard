"""Describe real recorded intrusions and compare retained measured support.

Size bins are descriptive only: they neither filter detections nor label false
alarms. Repeated object-frame observations are not independent physical objects.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .panel_report import Statistics
from .run import digest, write_json


def box_key(obj):
    return tuple(np.round(obj['bbox_min'] + obj['bbox_max'], 7))


def summarize(path):
    stats = Statistics()
    sizes, supports, interior = [], [], []
    confirmation = Counter()
    examples = []
    for line in path.open():
        row = json.loads(line)
        stats.add(row)
        for obj in row['objects']:
            if not obj['intersection_confirmed']:
                continue
            size = max(obj['extent_m'])
            sizes.append(size)
            supports.append(obj['support_voxels'])
            interior.append(obj['in_envelope_voxels'])
            confirmation[obj['intersection_confirmation']] += 1
            examples.append({'frame': row['frame'], 'track_id': obj['track_id'], 'distance_m': obj['distance_m'],
                             'extent_m': obj['extent_m'], 'support_voxels': obj['support_voxels'],
                             'interior_voxels': obj['in_envelope_voxels'], 'confirmation': obj['intersection_confirmation']})
            examples = sorted(examples, key=lambda o: max(o['extent_m']))[:10]
    return stats.result() | {'confirmed_max_extent_quantiles_m': np.quantile(sizes, [0, .25, .5, .75, 1]).tolist() if sizes else None,
                            'confirmed_size_bins_cumulative': {str(limit): sum(s < limit for s in sizes) for limit in [.1, .3, .5, 1.]},
                            'confirmed_with_at_most_three_interior_voxels': sum(n <= 3 for n in interior),
                            'confirmed_total_support_quantiles': np.quantile(supports, [0, .5, 1]).tolist() if supports else None,
                            'confirmation_methods': dict(confirmation), 'smallest_confirmed_observations': examples}


def compare_rows(before, after):
    matched = Counter()
    examples, identity_errors, expanded_far_boxes = [], [], []
    for old_line, new_line in zip_longest(before.open(), after.open()):
        if old_line is None or new_line is None:
            matched['unpaired_frames'] += 1
            continue
        old, new = json.loads(old_line), json.loads(new_line)
        if any(old[k] != new[k] for k in ['frame', 'measurement_timestamp_ns', 'record_timestamp_ns', 'sensor_frame', 'topic']):
            identity_errors.append([old['frame'], new['frame']])
            continue
        table = {box_key(obj): obj for obj in new['objects']}
        for obj in old['objects']:
            other = table.get(box_key(obj))
            if other is None:
                matched['old_boxes_without_identical_after_bbox'] += 1
                if obj['cluster_nearest_x_m'] >= 60:
                    containing = [o for o in new['objects']
                                  if np.all(np.asarray(o['bbox_min']) <= np.asarray(obj['bbox_min']) + 1e-7)
                                  and np.all(np.asarray(o['bbox_max']) >= np.asarray(obj['bbox_max']) - 1e-7)]
                    matched['far_old_boxes_contained_in_after_bbox'] += bool(containing)
                    matched['far_old_boxes_without_containing_after_bbox'] += not containing
                    expanded_far_boxes.append({'frame': old['frame'], 'old_id': obj['track_id'],
                                               'old_support': obj['support_voxels'],
                                               'containing_after': [{'id': o['track_id'], 'support': o['support_voxels']}
                                                                    for o in containing]})
                continue
            matched['same_bbox'] += 1
            matched['same_bbox_and_support_count'] += other['support_voxels'] == obj['support_voxels']
            if obj['cluster_nearest_x_m'] >= 60:
                matched['far_old_boxes_retained_identically'] += 1
            if obj['intersection_confirmed'] and not other['intersection_confirmed']:
                matched['confirmed_to_' + other['path_relation']] += 1
                if len(examples) < 20:
                    examples.append({'frame': old['frame'], 'old_id': obj['track_id'], 'new_id': other['track_id'],
                                     'extent_m': obj['extent_m'], 'distance_m': obj['distance_m'],
                                     'before_interior': obj['in_envelope_voxels'], 'after_interior': other['in_envelope_voxels'],
                                     'after_boundary': other['boundary_uncertain_voxels'], 'after_relation': other['path_relation']})
        matched['new_boxes_without_identical_before_bbox'] += len(set(table) - {box_key(obj) for obj in old['objects']})
    return {'matched_geometry': dict(matched), 'changed_confirmation_examples': examples,
            'expanded_far_boxes': expanded_far_boxes, 'identity_errors': identity_errors}


def diagnostic_support(before, after):
    reports = []
    for old in sorted((before / 'diagnostics').glob('*.npz')):
        new = after / 'diagnostics' / old.name
        if not new.exists():
            continue
        record = {'file': old.name, 'stages': {}}
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            for stage in ['geometry_voxel_points', 'context_after_background', 'cluster_points']:
                distance, _ = cKDTree(b[stage]).query(a[stage])
                record['stages'][stage] = {'before_points': len(a[stage]), 'after_points': len(b[stage]),
                                            'old_points_missing_after_at_1e_8_m': int(np.count_nonzero(distance > 1e-8))}
        reports.append(record)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    report = {'before': str(args.before), 'after': str(args.after), 'bags': [],
              'limitations': ['No exhaustive independent labels; no precision, recall or false-alarm rate.',
                              'BBox matching uses rounded coordinates (1e-7m); it does not establish physical identity.',
                              'Point preservation is evaluated on shared saved diagnostic frames only.',
                              'A containing bbox is evidence of expanded/merged geometry, not a per-return preservation proof.',
                              'Size bins describe measured support, not amodal objects or collision danger.']}
    for before in sorted(args.before.glob('*.jsonl')):
        after = args.after / before.name
        if not after.exists():
            raise FileNotFoundError(after)
        row = {'bag': before.stem, 'before_sha256': digest(before), 'after_sha256': digest(after),
               'before': summarize(before), 'after': summarize(after)} | compare_rows(before, after)
        report['bags'].append(row)
        print(json.dumps({'bag': before.stem, 'before_confirmed': row['before']['confirmed_intersection_observations'],
                          'after_confirmed': row['after']['confirmed_intersection_observations'],
                          'geometry': row['matched_geometry']}), flush=True)
    report['diagnostic_support'] = diagnostic_support(args.before, args.after)
    write_json(args.output, report)


if __name__ == '__main__':
    main()
