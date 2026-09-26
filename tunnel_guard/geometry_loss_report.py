"""Measure real geometry-loss continuity; never substitute replay agreement for ground truth."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import shutil

import numpy as np
from rosbags.rosbag2 import Reader

from .evaluate import evaluate_frames
from .object_evidence_report import rows
from .run import digest, write_json
from .visualization import ResultBag, corridor_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    before, after, output = (Path(plan[k]) for k in ('before', 'after', 'output'))
    manifests = [json.loads((root / 'manifest.json').read_text()) for root in (before, after)]
    if any('finished_unix_s' not in m for m in manifests):
        raise ValueError('Both actual replays must finish before evaluation')
    configs = [json.loads((root / 'detector.json').read_text()) for root in (before, after)]
    if configs[0] != configs[1] or configs[1]['seed'] != plan['seed']:
        raise ValueError('Detector recipe or seed changed')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'experiment.json', plan)
    shutil.copy2(__file__, output / 'reporter.py')
    annotations = {name: json.loads(Path(path).read_text()) for name, path in plan['annotations'].items()}
    needed = {(f['bag'], f['frame']) for a in annotations.values() for f in a['frames']}
    selected = {(c['bag'], c['frame'] + offset) for c in plan['cases'] for offset in (-1, 0, 1)}
    captured, predictions = {}, [{}, {}]
    differences, status_changes, bag_counts = Counter(), [], {}
    temporal_changes, temporal_examples = Counter(), {}
    totals = Counter()
    for bag, expected in plan['bags'].items():
        count = 0
        for old, new in zip_longest(rows(before, bag), rows(after, bag)):
            if old is None or new is None or old['source_scan_id'] != new['source_scan_id'] or old['frame'] != new['frame']:
                raise ValueError(f'Unpaired source scan: {bag}:{count}')
            count += 1
            totals['frames'] += 1
            key = (bag, new['frame'])
            for field in plan['preserved_frame_fields']:
                if old[field] != new[field]:
                    differences['frame.' + field] += 1
            if new['geometry']['valid']:
                totals['valid_geometry_frames'] += 1
                local = [[{k: o[k] for k in plan['local_object_fields']} for o in r['objects']]
                         for r in (old, new)]
                if local[0] != local[1]:
                    differences['valid_frame_local_objects'] += 1
                else:
                    for previous, current in zip(old['objects'], new['objects']):
                        for field in plan['temporal_outcome_fields']:
                            if previous[field] != current[field]:
                                temporal_changes[field] += 1
                                examples = temporal_examples.setdefault(field, [])
                                if len(examples) < 20:
                                    examples.append({'bag': bag, 'frame': new['frame'],
                                                     'component_id': current['component_id'],
                                                     'before': previous[field], 'after': current[field]})
            else:
                totals['invalid_geometry_frames'] += 1
            if old['status'] != new['status']:
                status_changes.append({'bag': bag, 'frame': new['frame'], 'before': old['status'], 'after': new['status']})
            if key in selected:
                captured[key] = (old, new)
            if key in needed:
                predictions[0][key], predictions[1][key] = old, new
        if count != expected:
            raise ValueError(f'Incomplete panel {bag}: {count} != {expected}')
        bag_counts[bag] = count
    cases = []
    for case in plan['cases']:
        bag, frame = case['bag'], case['frame']
        old, new = captured[(bag, frame)]
        previous = captured[(bag, frame - 1)][1]
        following = captured[(bag, frame + 1)][1]
        previous_ids = {o['track_id'] for o in previous['objects']}
        next_ids = {o['track_id'] for o in following['objects']}
        objects = new['objects']
        unsafe = [o['track_id'] for o in objects if o['path_relation'] != 'unknown'
                  or o['confirmed'] or o['intersection_confirmed']
                  or o['in_envelope_voxels'] or o['uncertain_voxels'] or o['boundary_uncertain_voxels']
                  or o['lateral_m'] != [None, None] or o['height_above_railhead_m'] != [None, None]]
        path = after / new['diagnostic_points']
        with np.load(path, allow_pickle=False) as arrays:
            points = arrays['geometry_voxel_points']
            support = {o['track_id']: arrays[f"support_{o['track_id']}"] for o in objects}
            with ResultBag(output / f'{bag}_{frame:06d}_rviz', configs[1]) as consumer:
                consumer.write(new, points, new['measurement_timestamp_ns'], support)
            decoded_ids = arrays['decoded_source_indices']
            positions = np.searchsorted(decoded_ids, arrays['cluster_source_indices'])
            source_ids_match = bool(np.array_equal(decoded_ids[positions], arrays['cluster_source_indices']))
            attributes_match = source_ids_match
            for name in ('intensity', 'intensity_valid', 'ring', 'ring_valid', 'raw_time', 'raw_time_valid'):
                attributes_match &= arrays[f'decoded_{name}'][positions].tobytes() == arrays[f'cluster_{name}'].tobytes()
        marker_counts = Counter()
        unknown_colors = set()
        with Reader(output / f'{bag}_{frame:06d}_rviz') as reader:
            for connection, _, raw in reader.messages():
                message = consumer.store.deserialize_cdr(raw, connection.msgtype)
                if connection.topic.endswith('debug_markers'):
                    marker_counts.update(m.ns for m in message.markers)
                    unknown_colors.update((m.color.r, m.color.g, m.color.b) for m in message.markers
                                          if m.ns == 'observed_support')
        evidence = {'bag': bag, 'frame': frame, 'before_objects': len(old['objects']),
                    'after_objects': len(objects), 'previous_objects': len(previous['objects']),
                    'previous_tracks_observed': sum(o['track_id'] in previous_ids for o in objects),
                    'tracks_observed_on_all_three_frames': sum(o['track_id'] in previous_ids & next_ids for o in objects),
                    'singleton_proposals': sum(o['support_voxels'] == 1 for o in objects),
                    'unsafe_relation_records': unsafe, 'status': new['status'], 'health': new['health'],
                    'segmentation_state': new['pipeline']['segmentation']['state'],
                    'drawn_reference_vertices': len(corridor_edges(new['geometry'], configs[1])),
                    'source_attributes_exact': bool(attributes_match), 'diagnostic_sha256': digest(path),
                    'recorded_marker_counts': dict(marker_counts), 'unknown_marker_colors': sorted(unknown_colors),
                    'largest_components': sorted([{'track_id': o['track_id'], 'support_voxels': o['support_voxels'],
                        'extent_m': o['extent_m']} for o in objects], key=lambda o: o['support_voxels'], reverse=True)[:5]}
        cases.append(evidence)
        print(json.dumps(evidence), flush=True)
    labels = {name: {'before': evaluate_frames(predictions[0], a), 'after': evaluate_frames(predictions[1], a),
                     'source_sha256': digest(Path(plan['annotations'][name]))} for name, a in annotations.items()}
    ok = not differences and len(cases) == totals['invalid_geometry_frames'] and all(
        c['after_objects'] > 0 and c['previous_tracks_observed'] > 0 and not c['unsafe_relation_records']
        and c['status'] == 'unknown' and c['health'] == 'unavailable' and c['segmentation_state'] == 'ran'
        and not c['drawn_reference_vertices'] and c['source_attributes_exact'] for c in cases)
    report = {'pipeline_contract_passed': ok, 'totals': dict(totals), 'bags': bag_counts,
              'preserved_field_differences': dict(differences), 'status_changes': status_changes,
              'temporal_outcome_changes': dict(temporal_changes), 'temporal_examples': temporal_examples,
              'cases': cases, 'labels': labels, 'manifests': manifests,
              'manifest_sha256': [digest(root / 'manifest.json') for root in (before, after)],
              'reporter_sha256': digest(Path(__file__)), 'limits': [
                  'Association continuity is not independent object identity or recall.',
                  'Large infrastructure and singleton proposals remain; no operator-policy approval.',
                  'Observed invalid frames have insufficient longitudinal rail support, not every possible failure.',
                  'Provisional/nonexhaustive and detector-propagated labels cannot establish field precision or holdout recall.',
                  'No surveyed rail contour, actual swept envelope, target-hardware or live ROS/DDS validation.']}
    write_json(output / 'summary.json', report)
    print(json.dumps({'pipeline_contract_passed': ok, 'totals': dict(totals),
                      'differences': dict(differences), 'status_changes': len(status_changes)}), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
