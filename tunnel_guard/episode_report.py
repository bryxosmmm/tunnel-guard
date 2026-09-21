"""Stream real detector outputs into review episodes; no inferred ground truth."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .run import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    path = args.run / (args.bag + '.jsonl')
    cfg = json.loads((args.run / 'detector.json').read_text())
    episodes = []
    active = {}
    states = Counter()
    reasons = Counter()
    health = Counter()
    health_reasons = Counter()
    count = 0
    previous = None
    violations = []
    gaps = []
    totals = Counter()
    processing = []
    heights = []
    first_stamp = None
    last_stamp = None
    evidence_floor_s = None
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            frame = row['frame']
            stamp = row['measurement_timestamp_ns']
            if previous is not None and stamp <= previous['measurement_timestamp_ns']:
                violations.append({'frame': frame, 'kind': 'nonincreasing_acquisition'})
            gap = previous is not None and (stamp - previous['measurement_timestamp_ns']) / 1000000000.0 > cfg['frame_max_gap_s']
            if gap:
                evidence_floor_s = row['timestamp_s']
            first_stamp = stamp if first_stamp is None else first_stamp
            last_stamp = stamp
            geometry = row.get('geometry', {})
            flags = {}
            if not geometry.get('valid', False):
                flags['geometry_unavailable'] = geometry.get('reason', row.get('reason', 'unavailable'))
            if not row.get('motion', {}).get('valid', False):
                flags['motion_unavailable'] = row.get('motion', {}).get('reason', 'unspecified')
            if row['status'] == 'obstacle':
                flags['confirmed_reference_intersection'] = 'algorithmic_confirmation_not_ground_truth'
            anchors = geometry.get('rail_anchors', [])
            mounting = row.get('mounting', {})
            if mounting.get('state') == 'observed':
                heights.append(mounting['height_above_support_plane_m'])
                if mounting.get('reference_comparison') == 'outside_diagnostic_band':
                    flags['mounting_reference_disagreement'] = 'recording_applicability_unverified'
            for obj in row['objects']:
                totals['candidate_observations'] += 1
                if obj['cluster_nearest_x_m'] >= 60:
                    totals['far_candidate_observations'] += 1
                    if obj['support_voxels'] <= cfg['weak_min_voxels']:
                        totals['far_minimum_support_observations'] += 1
                totals['confirmed_intersection_observations'] += int(obj.get('intersection_confirmed', False))
                if (obj.get('intersection_confirmed', False) and anchors
                        and obj['bbox_min'][0] > anchors[-1][0]):
                    totals['confirmed_clusters_entirely_beyond_last_rail_anchor'] += 1
                    flags['confirmed_beyond_last_rail_anchor'] = 'extrapolated_path_not_ground_truth'
                for key in ('evidence_timestamps_s', 'intersection_evidence_timestamps_s'):
                    times = obj.get(key, [])
                    if len(times) != len(set(times)) or any((t > row['timestamp_s'] for t in times)):
                        violations.append({'frame': frame, 'track_id': obj['track_id'], 'kind': key})
                    if evidence_floor_s is not None and any(t < evidence_floor_s for t in times):
                        violations.append({
                            'frame': frame, 'track_id': obj['track_id'],
                            'kind': 'evidence_before_acquisition_gap', 'field': key,
                        })
            if gap:
                totals['acquisition_gaps'] += 1
                gaps.append({
                    'frame': frame,
                    'previous_frame': previous['frame'],
                    'duration_s': (stamp - previous['measurement_timestamp_ns']) / 1e9,
                    'reset': bool(row.get('gap_reset', False)),
                })
                if not row.get('gap_reset', False):
                    violations.append({'frame': frame, 'kind': 'gap_without_reset'})
            for kind in list(active):
                if kind not in flags or gap or active[kind]['reason'] != flags.get(kind):
                    episodes.append(active.pop(kind))
            for kind, reason in flags.items():
                if kind not in active:
                    active[kind] = {'kind': kind, 'reason': reason, 'start_frame': frame, 'end_frame': frame, 'start_measurement_ns': stamp, 'end_measurement_ns': stamp, 'frames': 0}
                active[kind].update(end_frame=frame, end_measurement_ns=stamp)
                active[kind]['frames'] += 1
            states[row['status']] += 1
            health[row.get('health', 'unspecified')] += 1
            health_reasons.update(row.get('health_reasons', []))
            reasons[geometry.get('reason', row.get('reason', 'unavailable'))] += 1
            totals['geometry_valid_frames'] += int(geometry.get('valid', False))
            totals['gap_reset_frames'] += int(row.get('gap_reset', False))
            processing.append(row['processing_s'] * 1000)
            count += 1
            previous = row
    episodes.extend(active.values())
    chosen = []
    for kind in sorted({e['kind'] for e in episodes}):
        for e in sorted([e for e in episodes if e['kind'] == kind], key=lambda e: (-e['frames'], e['start_frame']))[:3]:
            chosen.append({'kind': kind, 'start_frame': e['start_frame'], 'end_frame': e['end_frame'], 'representative_frame': (e['start_frame'] + e['end_frame']) // 2, 'frames': e['frames']})
    report = {
        'frames': count,
        'source_sha256': digest(path),
        'detector_sha256': digest(args.run / 'detector.json'),
        'report_sha256': digest(Path(__file__)),
        'measurement_first_ns': first_stamp,
        'measurement_last_ns': last_stamp,
        'status_frames': dict(states),
        'geometry_reasons': dict(reasons),
        'health_frames': dict(health),
        'health_reason_frames': dict(health_reasons),
        'totals': dict(totals),
        'evidence_violations': violations,
        'acquisition_gaps': gaps,
        'processing_ms': {
            key: float(np.quantile(processing, q))
            for key, q in [('p50', 0.5), ('p95', 0.95), ('p99', 0.99)]
        } if processing else {},
        'height_proxy_m': {
            key: float(np.quantile(heights, q))
            for key, q in [('p05', 0.05), ('p50', 0.5), ('p95', 0.95)]
        } if heights else None,
        'episodes': episodes,
        'review_selection': chosen,
        'limitations': [
            'Algorithmic observations and review intervals, not labelled obstacle events or false-alarm rates.',
            'Episodes split at acquisition gaps; adjacent frames remain correlated.',
            'Mounting height is a support proxy with unverified reference applicability.',
            'Evidence checks inspect saved timestamp lists; they do not independently reproduce tracking.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in (
        'episodes', 'review_selection', 'evidence_violations', 'acquisition_gaps'
    )}))


if __name__ == '__main__':
    main()
