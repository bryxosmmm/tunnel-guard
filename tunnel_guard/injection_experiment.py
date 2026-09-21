"""Controlled ray-conditioned obstacle insertion into recorded scans; no field accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree

from .detector import Detector, load_config
from .io import iter_bag
from .run import digest, environment, write_json
from .stress import ray_box
from .visualization import corridor_edges


STAGES = ('range_points', 'cropped_points', 'geometry_voxel_points',
          'context_after_background', 'cluster_points')


def nearest_return_ranges(points, point_times, origin, direction_quantum):
    """Conservative return grouping by scan-relative acquisition time and direction.

    Quantized direction is a declared proxy, not calibrated hardware beam identity.
    Without point times this experiment cannot distinguish emissions and refuses input.
    """
    if len(point_times) != len(points):
        raise ValueError('Insertion requires per-point times for return grouping.')
    delta = points - origin
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance == 0):
        raise ValueError('Return at sensor origin has no direction.')
    direction = delta / distance[:, None]
    keys = np.column_stack((np.round(direction / direction_quantum), point_times))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    nearest = np.full(inverse.max()+1, np.inf)
    np.minimum.at(nearest, inverse, distance)
    return nearest[inverse]


def insert(points, point_times, origin, case, rng, nearest_ranges):
    """Opaque first-surface replacement along existing valid return directions.

    Dropout removes intercepted returns, including their hidden background. Invalid
    source slots supply no directions; duplicate return directions remain duplicates.
    """
    if 'bbox_min' not in case:
        return points, point_times, np.empty((0, 3)), {'intercepted_returns': 0, 'inserted_returns': 0}
    delta = points - origin
    measured_range = np.linalg.norm(delta, axis=1)
    directions = delta / measured_range[:, None]
    hit_range = ray_box(directions, origin, np.asarray(case['bbox_min']), np.asarray(case['bbox_max']))
    intercepted = np.isfinite(hit_range) & (hit_range < nearest_ranges)
    retained = intercepted & (rng.random(len(points)) >= case.get('return_drop_probability', 0.))
    keep = ~intercepted | retained
    cloud = points.copy()
    cloud[retained] = origin + directions[retained] * hit_range[retained, None]
    return (cloud[keep], point_times[keep] if len(point_times) else point_times,
            cloud[retained], {'intercepted_returns': int(intercepted.sum()),
                              'inserted_returns': int(retained.sum()),
                              'blocked_by_nearer_group_return': int(np.count_nonzero((hit_range < measured_range) & ~intercepted)),
                              'unique_inserted_positions_1um': len(np.unique(np.round(cloud[retained], 6), axis=0))})


def membership(points, tree, tolerance):
    return tree.query(points)[0] <= tolerance if tree is not None and len(points) else np.zeros(len(points), dtype=bool)


def inspect_sampling(experiment, config, output):
    """Separate missing measured directions from occlusion by nearer returns."""
    records = {c['name']: [] for c in experiment['cases'] if 'bbox_min' in c}
    origin = np.asarray(config['sensor_translation'])
    for scan in iter_bag(Path(experiment['bag']), config, max_frames=experiment['frames']):
        delta = scan.points - origin
        distance = np.linalg.norm(delta, axis=1)
        rays = delta / distance[:, None]
        nearest = nearest_return_ranges(scan.points, scan.point_times, origin, experiment['return_direction_quantum'])
        for case in experiment['cases']:
            if 'bbox_min' not in case:
                continue
            hit = ray_box(rays, origin, np.array(case['bbox_min']), np.array(case['bbox_max']))
            aimed = np.isfinite(hit)
            records[case['name']].append({'frame': scan.index,
                'valid_return_rays_through_box': int(aimed.sum()),
                'unoccluded_intercepts': int((aimed & (hit < nearest)).sum()),
                'nearer_measured_returns': int((aimed & (hit >= nearest)).sum())})
    path = output / 'sampling.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, {'cases': records, 'source_sha256': digest(Path(__file__)),
        'note': 'No measured direction does not establish no emitted beam. Counts retain source return multiplicity.'})
    print(path)


def render(output, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    rows = [json.loads(s) for s in (output / 'observations.jsonl').read_text().splitlines()]
    cases = list(dict.fromkeys(r['case'] for r in rows if r['case'] != 'recorded_control'))
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 2.7 * len(cases)), squeeze=False)
    for i, name in enumerate(cases):
        row = next(r for r in reversed(rows) if r['case'] == name)
        with np.load(output / f'{name}_preview.npz') as data:
            scene, injected = data['scene'], data['inserted']
            for j, dim in enumerate([1, 2]):
                ax = axes[i, j]
                ax.scatter(scene[:, 0], scene[:, dim], s=1, c='.65', rasterized=True)
                ax.scatter(injected[:, 0], injected[:, dim], s=8, c='#bd30bd', rasterized=True)
                for key in data.files:
                    if key.startswith('support_'):
                        q = data[key]
                        ax.scatter(q[:, 0], q[:, dim], s=12, facecolors='none', edgecolors='#ed7800', linewidths=.5)
                edges = data['corridor'].reshape(-1, 2, 3)
                ax.add_collection(LineCollection(edges[:, :, [0, dim]], colors='teal', linewidths=.5))
                lo, hi = np.array(row['case_definition']['bbox_min']), np.array(row['case_definition']['bbox_max'])
                ax.set(xlim=(lo[0]-2, hi[0]+2), ylim=(lo[dim]-1, hi[dim]+1),
                       xlabel='forward x [m]', ylabel=('lateral y' if dim == 1 else 'z')+' [m]')
                ax.set_title(f"{name}: inserted={row['insertion']['inserted_returns']}, cluster={row['stage_inserted_representatives'].get('cluster_points')}\nframe {row['frame']}; whole scene: {row['status']}; health={row['health']}", fontsize=8)
    fig.suptitle('MODELED INSERTIONS into real scans — not real obstacles or sensor range validation\nGray: input scene; magenta: inserted returns; orange rings: candidate support; teal: estimated contour')
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(output / 'insertion-review.png', dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--render-only', action='store_true')
    modes.add_argument('--sampling-only', action='store_true')
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    output = Path(experiment['output'])
    config = load_config(Path(experiment['detector_config']))
    if args.sampling_only:
        if experiment != json.loads((output / 'experiment.json').read_text()) or config != json.loads((output / 'detector.json').read_text()):
            raise ValueError('Sampling review must use the recorded experiment and detector configuration.')
        inspect_sampling(experiment, config, output)
        return
    if args.render_only:
        render(output, json.loads((output / 'detector.json').read_text()))
        return
    if config.get('deskew_enabled', False):
        raise ValueError('Point provenance currently requires deskew disabled.')
    if experiment['seed'] != config['seed']:
        raise ValueError('Experiment and detector seeds differ.')
    if not 0 < experiment['return_direction_quantum'] < 1:
        raise ValueError('Return grouping requires a finite direction quantum in (0,1).')
    if output.exists():
        raise FileExistsError(output)
    names = [c['name'] for c in experiment['cases']]
    if len(set(names)) != len(names) or any(not n.replace('_', '').isalnum() for n in names):
        raise ValueError('Case names must be unique safe filenames.')
    for case in experiment['cases']:
        if 'bbox_min' in case:
            bounds = np.array([case['bbox_min'], case['bbox_max']], dtype=float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
                raise ValueError('Cases require finite positive boxes.')
        if not 0 <= case.get('return_drop_probability', 0) <= 1:
            raise ValueError('Return drop probability outside [0,1].')
    scans = list(iter_bag(Path(experiment['bag']), config, max_frames=experiment['frames']))
    if len(scans) != experiment['frames']:
        raise ValueError('Requested recording prefix unavailable.')
    nearest_ranges = [nearest_return_ranges(s.points, s.point_times, np.asarray(config['sensor_translation']),
                                            experiment['return_direction_quantum']) for s in scans]
    output.mkdir(parents=True)
    write_json(output / 'experiment.json', experiment)
    write_json(output / 'detector.json', config)
    source = output / 'source' / 'tunnel_guard'
    source.mkdir(parents=True)
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(p, source / p.name)
    manifest = environment() | {'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'experiment_sha256': digest(args.experiment), 'config_sha256': digest(Path(experiment['detector_config'])),
        'metadata_sha256': digest(Path(experiment['bag']) / 'metadata.yaml'),
        'source_measurement_timestamps_ns': [s.measurement_timestamp_ns for s in scans],
        'source_files': [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                         for p in sorted(Path(experiment['bag']).glob('*.db3'))],
        'started_unix_s': time.time(), 'warning': experiment['limitations']}
    write_json(output / 'manifest.json', manifest)
    summary = []
    with (output / 'predictions.jsonl').open('x') as predictions, (output / 'observations.jsonl').open('x') as observations:
        for index, case in enumerate(experiment['cases']):
            detector = Detector(config)
            records = []
            for frame, scan in enumerate(scans):
                # Stable per-frame random stream shares dropout draws across cases.
                rng = np.random.default_rng(np.random.SeedSequence([experiment['seed'], frame]))
                cloud, times, inserted, stats = insert(scan.points, scan.point_times,
                    np.asarray(config['sensor_translation']), case, rng, nearest_ranges[frame])
                row = detector.process(cloud, scan.timestamp_s, times, capture_diagnostics=True)
                row.update(frame=scan.index, case=case['name'], measurement_timestamp_ns=scan.measurement_timestamp_ns)
                predictions.write(json.dumps(row, allow_nan=False)+'\n')
                tree = cKDTree(inserted) if len(inserted) else None
                stages = {k: int(membership(detector.diagnostic_arrays[k], tree, experiment['membership_tolerance_m']).sum())
                          if k in detector.diagnostic_arrays else None for k in STAGES}
                objects, matched_support = [], {}
                for obj in row['objects']:
                    q = detector.display_support[obj['track_id']]
                    count = int(membership(q, tree, experiment['membership_tolerance_m']).sum())
                    if count:
                        objects.append({k: obj[k] for k in ['track_id', 'support_voxels', 'in_envelope_voxels',
                            'path_relation', 'confirmed', 'intersection_confirmed']} | {'inserted_support': count,
                            'inserted_fraction': count / len(q)})
                        matched_support[f"support_{obj['track_id']}"] = q
                record = {'case': case['name'], 'case_definition': case, 'frame': scan.index,
                    'measurement_timestamp_ns': scan.measurement_timestamp_ns, 'status': row['status'],
                    'health': row['health'], 'insertion': stats, 'stage_inserted_representatives': stages,
                    'candidates_with_inserted_support': objects}
                observations.write(json.dumps(record, allow_nan=False)+'\n')
                records.append(record)
                if frame == len(scans)-1 and 'bbox_min' in case:
                    lo, hi = np.array(case['bbox_min'])-[3, 2, 2], np.array(case['bbox_max'])+[3, 2, 2]
                    roi = np.all((cloud >= lo) & (cloud <= hi), axis=1)
                    np.savez_compressed(output / f"{case['name']}_preview.npz", scene=cloud[roi], inserted=inserted,
                                        corridor=corridor_edges(row.get('geometry', {}), config), **matched_support)
            entry = {'case': case['name'], 'frames': len(records),
                'inserted_returns': [r['insertion']['inserted_returns'] for r in records],
                'cluster_inserted_representatives': [r['stage_inserted_representatives']['cluster_points'] for r in records],
                'frames_with_target_dominated_candidate': sum(any(o['inserted_fraction'] >= .5 for o in r['candidates_with_inserted_support']) for r in records),
                'frames_with_confirmed_target_dominated_object': sum(any(o['inserted_fraction'] >= .5 and o['confirmed'] for o in r['candidates_with_inserted_support']) for r in records),
                'frames_with_confirmed_target_dominated_intersection': sum(any(o['inserted_fraction'] >= .5 and o['intersection_confirmed'] for o in r['candidates_with_inserted_support']) for r in records),
                'whole_scene_statuses': dict(Counter(r['status'] for r in records))}
            summary.append(entry)
            print(json.dumps(entry), flush=True)
    write_json(output / 'summary.json', {'cases': summary, 'accuracy': None,
        'interpretation': 'Target-dominated means >=50% inserted support, a diagnostic attribution convention, not semantic detection ground truth. Whole-scene status includes pre-existing unlabelled objects.',
        'limitations': experiment['limitations']})
    manifest['finished_unix_s'] = time.time()
    write_json(output / 'manifest.json', manifest)
    render(output, config)


if __name__ == '__main__':
    main()
