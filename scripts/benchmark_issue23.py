"""Fixed real-detector evaluation; no generated scenes or automated tests."""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from tunnel_guard import _native
from tunnel_guard.detector import Detector, load_config
from tunnel_guard.io import decode_cloud, iter_bag
from tunnel_guard.person_evidence_report import box_geometry, in_oriented_box, oriented_iou
from tunnel_guard.run import capture_native_sources, digest, environment, git_revision, write_json
from tunnel_guard.sustech import read_box_files


def load(path):
    return json.loads(Path(path).read_text())


def support_probe(plan, config, out):
    """Real PointCloud2 decode and fresh-state detector; never score temporal alarms here."""
    report = load(plan['support_report'])
    records = []
    store = get_typestore(Stores.ROS2_HUMBLE)
    with zipfile.ZipFile(plan['support_archive']) as archive:
        reference = {r['frame']: r for r in map(json.loads, archive.read(
            'build/issue23-point-support-20260924/reference.jsonl').splitlines())}
        with Reader(Path(report['recipe']['bag'])) as reader:
            for entry in report['frames']:
                saved = reference[entry['frame']]
                stamp = saved['record_timestamp_ns']
                conn, _, raw = next(reader.messages(
                    connections=[c for c in reader.connections if c.topic == saved['topic']],
                    start=stamp, stop=stamp + 1))
                message = store.deserialize_cdr(raw, conn.msgtype)
                measured = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                if measured != saved['measurement_timestamp_ns']:
                    raise ValueError('Support probe acquisition mismatch')
                points, times, _, _, attrs = decode_cloud(message, np.asarray(config['sensor_rotation']),
                                                         np.asarray(config['sensor_translation']))
                detector = Detector(config)
                row = detector.process(points, measured * 1e-9, times, capture_diagnostics=True,
                                       point_attributes=attrs)
                keys = report['compared_fields']
                arrays = detector.diagnostic_arrays
                with np.load(io.BytesIO(archive.read(entry['artifact']))) as old:
                    membership_equal = all(np.array_equal(arrays[k], old[k]) for k in (
                        'cluster_points', 'cluster_labels', 'cluster_source_indices', 'cluster_core'))
                records.append({'frame': entry['frame'], 'geometry_equal': row.get('geometry') == saved.get('geometry'),
                    'measured_fields_equal': [[o.get(k) for k in keys] for o in row['objects']] ==
                                             [[o.get(k) for k in keys] for o in saved['objects']],
                    'membership_equal': membership_equal})
                write_json(out / f"probe-{entry['frame']}.json", row)
    write_json(out / 'support-probe.json', records)
    return records


def measure(sequence, max_gap):
    counts = Counter(r['status'] for r in sequence)
    episodes, observed, excluded = 0, 0, 0
    durations = Counter()
    prior = None
    for row in sequence:
        delta = row['measurement_timestamp_ns'] - prior['measurement_timestamp_ns'] if prior else 0
        if prior and delta <= 0:
            raise ValueError('Non-increasing acquisition timestamps')
        gap = delta > max_gap
        if prior:
            if gap:
                excluded += delta
            else:
                observed += delta
                durations[prior['status']] += delta
        if row['alarm'] and (prior is None or gap or not prior['alarm']):
            episodes += 1
        prior = row
    return {'frames': len(sequence), 'alarm_frames': sum(r['alarm'] for r in sequence),
            'alarm_episodes': episodes, 'status_frames': dict(counts),
            'observed_duration_s': observed / 1e9, 'excluded_gap_duration_s': excluded / 1e9,
            'status_duration_s': {k: v / 1e9 for k, v in durations.items()}}


def main():
    plan = load(sys.argv[1])
    config = load_config(plan['detector_config'])
    if config['seed'] != plan['seed'] or any(os.environ.get(k) != v for k, v in plan['thread_environment'].items()):
        raise ValueError('Seed or thread environment differs from fixed recipe')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f'run-{index:04d}').exists():
        index += 1
    out = root / f'run-{index:04d}'
    out.mkdir()
    print(f'Artifacts: {out}', flush=True)
    write_json(out / 'experiment.json', plan)
    write_json(out / 'detector.json', config)
    source = out / 'source'
    shutil.copytree('tunnel_guard', source / 'tunnel_guard', ignore=shutil.ignore_patterns('__pycache__', '*.so'))
    shutil.copy2(__file__, source / Path(__file__).name)
    capture_native_sources(source)
    fixed_paths = [v for k, v in plan.items() if k.endswith(('_report', '_reference', '_plan', '_manifest', '_archive'))]
    manifest = environment() | {'git_revision': git_revision(), 'command': sys.argv,
        'recipe_sha256': digest(Path(sys.argv[1])), 'config_sha256': digest(Path(plan['detector_config'])),
        'fixed_input_sha256': {p: digest(Path(p)) for p in fixed_paths},
        'native_binary_sha256': digest(Path(_native.__file__)), 'thread_environment': plan['thread_environment']}
    write_json(out / 'manifest.json', manifest)
    probes = support_probe(plan, config, out)
    print(f"Support probe: {sum(all(r[k] for k in ('geometry_equal', 'measured_fields_equal', 'membership_equal')) for r in probes)}/{len(probes)} identical to historical evidence", flush=True)
    panel = load(plan['panel_manifest'])
    person_plan = load(plan['person_plan'])
    labels = read_box_files(Path(person_plan['source_labels']))
    if sorted(labels) != person_plan['label_frames']:
        raise ValueError('Person label panel changed')
    person = []
    synthetic = defaultdict(list)
    with zipfile.ZipFile(plan['synthetic_archive']) as archive:
        prefix = plan['synthetic_archive_prefix']
        observations = json.loads(archive.read(prefix + 'input-observations.json'))
        tracks = json.loads(archive.read(prefix + 'source-track-map.json'))
        input_summary = json.loads(archive.read(prefix + 'input-summary.json'))
        with np.load(io.BytesIO(archive.read(prefix + 'inferred-suffix-support.npz'))) as data:
            offsets = data['offsets'].copy()
    by_frame = defaultdict(list)
    for event, track in enumerate(tracks, 1):
        for frame, cluster in track:
            by_frame[frame].append((event, cluster))
    segments = sorted(panel['continuous_negative_recording']['segments'], key=lambda s: s['record_start_ns'])
    starts = [s['record_start_ns'] for s in segments]
    workloads = panel['real_recordings'] + [dict(panel['continuous_negative_recording'], role='continuous_negative')]
    workloads.append({'path': plan['synthetic_bag'], 'role': 'synthetic',
                      'metadata_message_count': plan['synthetic_frames']})
    summaries, partitions = {}, defaultdict(list)
    segment_counts = Counter()
    max_gap = round(plan['frame_max_gap_s'] * 1e9)
    for entry in workloads:
        bag = Path(entry['path'])
        metadata_hash = digest(bag / 'metadata.yaml')
        expected_hash = entry.get('metadata_sha256')
        if entry['role'] == 'synthetic':
            expected_hash = next(p['sha256'] for p in input_summary['source_identity'] if p['path'].endswith('/metadata.yaml'))
        if metadata_hash != expected_hash:
            raise ValueError(f'Changed bag metadata: {bag}')
        for file in entry.get('files', entry.get('segments', [])):
            if (bag / file.get('name', file.get('file'))).stat().st_size != file['bytes']:
                raise ValueError(f'Changed bag file size: {bag}')
        detector = Detector(config)
        compact, latencies, person_alarm_frames = [], [], []
        seen = set()
        started = time.perf_counter()
        with gzip.open(out / f'{bag.name}.jsonl.gz', 'wt', compresslevel=1) as stream:
            for scan in iter_bag(bag, config, every=1, max_frames=None):
                if scan.index != len(compact) or scan.measurement_timestamp_ns in seen:
                    raise ValueError('Missing, reordered or duplicate acquisition')
                seen.add(scan.measurement_timestamp_ns)
                diagnostic = (entry['role'] == 'real_person_event' and scan.index in labels) or (
                    entry['role'] == 'synthetic' and scan.index in by_frame)
                before = time.perf_counter()
                row = detector.process(scan.points, scan.timestamp_s, scan.point_times,
                                       capture_diagnostics=diagnostic, point_attributes=scan.attributes)
                latencies.append(time.perf_counter() - before)
                row.update(frame=scan.index, measurement_timestamp_ns=scan.measurement_timestamp_ns,
                           record_timestamp_ns=scan.record_timestamp_ns,
                           source_scan_id=f'{scan.topic}:{scan.frame_id}:{scan.measurement_timestamp_ns}')
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                item = {k: row[k] for k in ('frame', 'measurement_timestamp_ns', 'record_timestamp_ns', 'status')}
                item['alarm'] = any(o['intersection_confirmed'] for o in row['objects'])
                compact.append(item)
                if entry['role'] == 'continuous_negative':
                    part_index = bisect.bisect_right(starts, scan.record_timestamp_ns) - 1
                    if part_index < 0:
                        raise ValueError('Record precedes negative corpus')
                    segment = segments[part_index]
                    if scan.record_timestamp_ns > segment['record_start_ns'] + round(segment['record_duration_s'] * 1e9):
                        raise ValueError('Record outside frozen segment')
                    segment_counts[segment['segment']] += 1
                    partitions[segment['split']].append(item)
                arrays = detector.diagnostic_arrays if diagnostic else {}
                if entry['role'] == 'real_person_event' and diagnostic:
                    box = labels[scan.index][0]
                    center, scale, rotation, _ = box_geometry(box)
                    mask = in_oriented_box(arrays['cluster_points'], center, scale, rotation)
                    counts = Counter(int(v) for v in arrays['cluster_labels'][mask] if v >= 0)
                    ranked = sorted((o for o in row['objects'] if counts[o['component_id']]),
                                    key=lambda o: (-counts[o['component_id']], o['component_id']))
                    best = ranked[0] if ranked else None
                    person.append({'frame': scan.index, 'component': best,
                        'component_points_inside_obb': counts[best['component_id']] if best else 0,
                        'oriented_iou': oriented_iou(best, center, scale, rotation, person_plan['numerical_epsilon']) if best else 0.0,
                        'center_error_m': float(np.linalg.norm(np.asarray(best['center']) - center)) if best else None})
                    person_alarm_frames.append(item)
                if entry['role'] == 'synthetic':
                    if (scan.measurement_timestamp_ns != observations[scan.index]['stamp_ns']
                            or scan.record_timestamp_ns != observations[scan.index]['recorded_ns']
                            or scan.raw_points != observations[scan.index]['raw_n']):
                        raise ValueError('Synthetic provenance acquisition mismatch')
                    for event, j in by_frame[scan.index]:
                        cluster = observations[scan.index]['clusters'][j]
                        indices = observations[scan.index]['suffix_start'] + np.asarray(cluster['point_indices']) - offsets[scan.index]
                        member = np.isin(arrays.get('cluster_source_indices', np.array([], dtype=int)), indices)
                        labels_array = arrays.get('cluster_labels', np.array([], dtype=int))
                        components = []
                        for obj in row['objects']:
                            selected = labels_array == obj['component_id']
                            total = int(selected.sum())
                            n = int(np.count_nonzero(member & selected))
                            if n and n / total >= plan['minimum_inserted_fraction_for_attribution']:
                                components.append(obj)
                        synthetic[event].append({'frame': scan.index, 'stamp_ns': scan.measurement_timestamp_ns,
                            'source_forward_m': cluster['center'][0], 'components': components})
                if len(compact) % 100 == 0:
                    stream.flush()
                    print(f'{bag.name}: {len(compact)}/{entry["metadata_message_count"]}', flush=True)
        if len(compact) != entry['metadata_message_count']:
            raise ValueError(f'Incomplete corpus: {bag}: {len(compact)}')
        summaries[bag.name] = measure(compact, max_gap) | {'role': entry['role'],
            'metadata_sha256': metadata_hash, 'inference_ms_p50': float(np.quantile(latencies, .5) * 1000),
            'inference_ms_p95': float(np.quantile(latencies, .95) * 1000), 'wall_s': time.perf_counter() - started}
        if entry['role'] == 'continuous_negative':
            partitions['full'] = compact
        if person_alarm_frames:
            summaries[bag.name]['labelled_interval'] = measure(person_alarm_frames, max_gap)
        write_json(out / 'recordings.json', summaries)
    if any(segment_counts[s['segment']] != s['messages'] for s in segments):
        raise ValueError('Negative segment coverage differs from frozen panel')
    measured_partitions = {k: measure(v, max_gap) for k, v in partitions.items()}
    write_json(out / 'negative-partitions.json', measured_partitions)
    write_json(out / 'person.json', person)
    write_json(out / 'synthetic.json', dict(synthetic))
    score(plan, out, manifest, summaries, measured_partitions, person, synthetic, probes)


def score(plan, out, manifest, summaries, measured_partitions, person, synthetic, probes):
    violations = []
    write_json(out / 'evaluation-plan.json', plan)
    if manifest['config_sha256'] != digest(Path(plan['detector_config'])):
        raise ValueError('Scoring recipe does not identify the replayed detector')
    old_original = load(plan['original_recordings_reference'])['real']
    for name, result in summaries.items():
        if result['role'] != 'organizer_declared_empty':
            continue
        old = old_original[name]['before']
        for key in ('alarm_frames', 'alarm_episodes'):
            if result[key] > old[key]:
                violations.append(f'{name}: increased {key}')
        for status in ('unknown', 'unresolved_obstacle'):
            if result['status_frames'].get(status, 0) > old['status_frames'].get(status, 0):
                violations.append(f'{name}: increased {status} frames')
    person_plan = load(plan['person_plan'])
    old_person_summary = load(plan['person_summary_reference'])
    label_hashes = {p.name: digest(p) for p in sorted(Path(person_plan['source_labels']).glob('*.json'))}
    if label_hashes != old_person_summary['provenance']['original_labels_sha256']:
        raise ValueError('Original person label contents changed')
    if [r['frame'] for r in person] != person_plan['label_frames']:
        raise ValueError('Incomplete person evidence panel')
    with zipfile.ZipFile(plan['synthetic_archive']) as archive:
        observations = json.loads(archive.read(plan['synthetic_archive_prefix'] + 'input-observations.json'))
    with gzip.open(out / 'cloud_with_fake_obj.jsonl.gz', 'rt') as stream:
        acquisition_count = 0
        for row in map(json.loads, stream):
            expected = observations[acquisition_count]
            if (row['frame'] != acquisition_count or row['measurement_timestamp_ns'] != expected['stamp_ns']
                    or row['record_timestamp_ns'] != expected['recorded_ns']):
                raise ValueError('Synthetic replay does not match frozen provenance')
            acquisition_count += 1
    if acquisition_count != plan['synthetic_frames']:
        raise ValueError('Incomplete synthetic acquisition coverage')
    old_negative = load(plan['negative_reference'])['partitions']
    for name, result in measured_partitions.items():
        old = old_negative[name]['before']
        for key in ('alarm_frames', 'alarm_episodes'):
            if result[key] > old[key]:
                violations.append(f'{name}: increased {key}')
        if result['status_frames'].get('unknown', 0) > old['status_frames'].get('unknown', 0):
            violations.append(f'{name}: increased unknown frames')
        if result['status_frames'].get('unresolved_obstacle', 0) > old['status_frames'].get('unresolved_obstacle', 0):
            violations.append(f'{name}: increased unresolved frames')
    old_person = {r['frame']: r for r in load(plan['person_reference'])}
    for row in person:
        old = old_person[row['frame']]
        obj = row['component']
        if not obj or (old['component']['presence_confirmed'] and not obj['presence_confirmed']):
            violations.append(f'Person {row["frame"]}: lost presence')
        if row['oriented_iou'] + 1e-9 < old['oriented_iou']:
            violations.append(f'Person {row["frame"]}: lower oriented IoU')
        if obj and row['center_error_m'] > old['center_error_m'] + 1e-9:
            violations.append(f'Person {row["frame"]}: larger localization error')
        if obj and obj['intersection_confirmed'] and not old['component']['intersection_confirmed']:
            violations.append(f'Person {row["frame"]}: new unverified intersection')
    switches = sum(a['component']['track_id'] != b['component']['track_id'] for a, b in zip(person, person[1:]) if a['component'] and b['component'])
    if switches:
        violations.append('Person: new track switches')
    positive_metrics = {}
    for event in load(plan['synthetic_reference'])['events']:
        number = event['event']
        current = {r['frame']: r for r in synthetic[number]}
        for field, predicate in [('reported', lambda o: True), ('presence', lambda o: o['presence_confirmed']),
                                 ('intersection', lambda o: o['intersection_confirmed'])]:
            old_frames = {r['frame'] for r in event['diagnostic_observations'] for c in r['components']
                          if c['attribution_majority'] for o in c['objects'] if predicate(o)}
            new_frames = {f for f, r in current.items() if any(predicate(o) for o in r['components'])}
            if not old_frames <= new_frames:
                violations.append(f'Synthetic event {number}: lost {field} frames {sorted(old_frames - new_frames)}')
            positive_metrics[f'synthetic_event_{number}_{field}_frames'] = len(new_frames)
        qualifying = [r for r in current.values() if any(o['intersection_confirmed'] for o in r['components'])]
        positive_metrics[f'synthetic_event_{number}_first_confirmation_frame'] = min(
            (r['frame'] for r in qualifying), default=-1)
        positive_metrics[f'synthetic_event_{number}_first_confirmation_range_m'] = (
            min(qualifying, key=lambda r: r['frame'])['source_forward_m'] if qualifying else -1)
    metrics = {'negative_alarm_episodes': sum(r['alarm_episodes'] for r in summaries.values()
                 if r['role'] in ('organizer_declared_empty', 'continuous_negative')),
        'continuous_alarm_frames': measured_partitions['full']['alarm_frames'],
        'continuous_alarm_episodes': measured_partitions['full']['alarm_episodes'],
        'continuous_unknown_frames': measured_partitions['full']['status_frames'].get('unknown', 0),
        'continuous_unresolved_frames': measured_partitions['full']['status_frames'].get('unresolved_obstacle', 0),
        'person_presence_frames': sum(bool(r['component'] and r['component']['presence_confirmed']) for r in person),
        'person_track_switches': switches, 'evaluated_frames': sum(r['frames'] for r in summaries.values()),
        'historical_support_identical_frames': sum(all(r[k] for k in ('geometry_equal', 'measured_fields_equal', 'membership_equal')) for r in probes),
        **positive_metrics}
    for name, result in summaries.items():
        for key in ('frames', 'alarm_frames', 'alarm_episodes', 'inference_ms_p50', 'inference_ms_p95'):
            metrics[f'{name}_{key}'] = result[key]
        for status in ('unknown', 'unresolved_obstacle', 'candidate', 'no_obstacle_observed'):
            metrics[f'{name}_{status}_frames'] = result['status_frames'].get(status, 0)
    evaluation_path = out / f'evaluation-{digest(Path(__file__))[:12]}.json'
    write_json(evaluation_path, {'metrics': metrics, 'violations': violations,
        'valid': not violations, 'limitations': plan['limits'], 'recordings': summaries})
    manifest['completed'] = True
    manifest['evaluation_sha256'] = digest(evaluation_path)
    manifest['evaluation_path'] = str(evaluation_path)
    manifest['evaluator_sha256'] = digest(Path(__file__))
    write_json(out / 'manifest.json', manifest)
    print(f'Evaluation: {evaluation_path}', flush=True)
    if violations:
        raise SystemExit('Acceptance failed: ' + '; '.join(violations))
    for name, value in metrics.items():
        print(f'METRIC {name}={value}')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[2] == '--score':
        plan = load(sys.argv[1])
        out = Path(sys.argv[3])
        manifest = load(out / 'manifest.json')
        if not manifest.get('completed'):
            raise ValueError('Separate scoring requires a complete replay')
        score(plan, out, manifest, load(out / 'recordings.json'),
              load(out / 'negative-partitions.json'), load(out / 'person.json'),
              {int(k): v for k, v in load(out / 'synthetic.json').items()},
              load(out / 'support-probe.json'))
    else:
        main()
