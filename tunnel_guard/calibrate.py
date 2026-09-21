"""Estimate provisional track-relative orientation; never infer vehicle extrinsics.

Fit a chronological prefix, freeze its rotation, then re-estimate geometry on
later measurements. Ground and rail assumptions are inherited from the detector.
A stable result is not surveyed calibration and cannot identify vehicle offsets.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag
from .run import digest, environment, git_revision, write_json


def estimate(points, config, recipe):
    radius = np.linalg.norm(points, axis=1)
    mask = ((radius >= config['min_range_m']) & (radius <= config['max_range_m'])
            & (points[:, 0] >= config['min_forward_m'])
            & (np.abs(points[:, 1]) <= config['context_half_width_m']))
    reduced = voxel_representatives(points[mask], config['geometry_voxel_m'],
                                     config.get('voxel_backend', 'numpy'))
    geometry = TrackGeometry(reduced, config)
    result = {'accepted': False, 'reason': geometry.reason,
              'ground_quality': geometry.ground_quality,
              'rail_anchors': geometry.rail_anchors.tolist()}
    if not geometry.valid:
        return result
    lo, hi = recipe['fit_range_m']
    rail = geometry.rail_anchors
    rail = rail[(rail[:, 0] >= lo) & (rail[:, 0] <= hi)]
    if len(rail) < recipe['min_rail_anchors'] or np.ptp(rail[:, 0]) < recipe['min_span_m']:
        return result | {'reason': 'insufficient_near_rail_span'}
    # Use the local bed profile (including its longitudinal shift), not the
    # global plane alone. Track bed is not automatically a railhead/cant plane.
    ground, uncertainty = geometry.ground(np.column_stack((rail[:, :2], np.zeros(len(rail)))))
    if not np.isfinite(uncertainty).all() or np.max(uncertainty) > config['ground_max_uncertainty_m']:
        return result | {'reason': 'unsupported_local_ground'}
    design = np.column_stack((rail[:, 0], np.ones(len(rail))))
    lateral = np.linalg.lstsq(design, rail[:, 1], rcond=None)[0]
    vertical = np.linalg.lstsq(design, ground, rcond=None)[0]
    lateral_residual = float(np.max(np.abs(rail[:, 1] - design @ lateral)))
    vertical_residual = float(np.max(np.abs(ground - design @ vertical)))
    result |= {'max_lateral_residual_m': lateral_residual,
               'max_vertical_residual_m': vertical_residual}
    if max(lateral_residual, vertical_residual) > recipe['max_line_residual_m']:
        return result | {'reason': 'local_track_not_straight_or_planar'}
    forward = np.array([1., lateral[0], vertical[0]])
    forward /= np.linalg.norm(forward)
    # Project the bed normal perpendicular to the fitted longitudinal tangent.
    up = np.array([-geometry.plane[0], -geometry.plane[1], 1.])
    up -= forward * np.dot(up, forward)
    up /= np.linalg.norm(up)
    left = np.cross(up, forward)
    correction = np.stack((forward, left, up))
    height = float(-vertical[1])
    return result | {'accepted': True, 'reason': 'supported_local_track_orientation',
                     'correction_rotation': correction.tolist(),
                     'correction_xyz_deg': Rotation.from_matrix(correction).as_euler('xyz', degrees=True).tolist(),
                     'bed_height_at_origin_extrapolated_m': height,
                     'track_center_at_origin_extrapolated_m': float(lateral[1])}


def stability(records, rotation):
    valid = [r for r in records if r['accepted']]
    if not valid:
        return {'accepted_frames': 0, 'fraction': 0., 'max_rotation_deviation_deg': None,
                'height_span_m': None, 'center_span_m': None}
    rotations = Rotation.from_matrix([r['correction_rotation'] for r in valid])
    deviations = (rotations * rotation.inv()).magnitude() * 180 / np.pi
    return {'accepted_frames': len(valid), 'fraction': len(valid) / len(records),
            'max_rotation_deviation_deg': float(np.max(deviations)),
            'height_span_m': float(np.ptp([r['bed_height_at_origin_extrapolated_m'] for r in valid])),
            'center_span_m': float(np.ptp([r['track_center_at_origin_extrapolated_m'] for r in valid]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    cfg = load_config(recipe['detector_config'])
    if recipe['seed'] != cfg['seed']:
        raise ValueError('Seed mismatch')
    fit_count, validation_count = recipe['fit_frames'], recipe['validation_frames']
    if min(fit_count, validation_count) < 3:
        raise ValueError('Need at least three frames in each chronological partition')
    output = Path(recipe['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'experiment.json', recipe)
    write_json(output / 'detector.json', cfg)
    shutil.copytree(Path(__file__).parent, output / 'source' / 'tunnel_guard',
                    ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.pyd', '*.dylib'))
    write_json(output / 'manifest.json', environment() | {'command': sys.argv, 'git_revision': git_revision(),
                                                        'config_sha256': digest(Path(recipe['detector_config']))})
    geometry_cfg = copy.deepcopy(cfg)
    geometry_cfg['background']['enabled'] = False
    summaries = []
    for entry in recipe['bags']:
        bag = Path(entry['path'])
        folder = output / bag.name
        folder.mkdir()
        provenance = {'bag': str(bag), 'metadata_sha256': digest(bag / 'metadata.yaml'),
                      'files': [{'name': p.name, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                for p in sorted(bag.glob('*.db3'))]}
        fit, heldout, corrected = [], [], []
        rotation = None
        cutoff = None
        ingestion = {}
        with (folder / 'frames.jsonl').open('x') as stream:
            for number, scan in enumerate(iter_bag(bag, cfg, max_frames=fit_count + validation_count,
                                                   topic=entry.get('topic'), diagnostics=ingestion)):
                raw = estimate(scan.points, geometry_cfg, recipe)
                identity = {'frame': scan.index, 'measurement_timestamp_ns': scan.measurement_timestamp_ns,
                            'record_timestamp_ns': scan.record_timestamp_ns, 'sensor_frame': scan.frame_id,
                            'topic': scan.topic}
                raw |= identity
                if number < fit_count:
                    fit.append(raw)
                    if number == fit_count - 1:
                        valid = [r['correction_rotation'] for r in fit if r['accepted']]
                        if valid:
                            rotation = Rotation.from_matrix(valid).mean()
                        cutoff = scan.measurement_timestamp_ns
                    row = {'partition': 'fit', 'before': raw}
                else:
                    if cutoff is None or scan.measurement_timestamp_ns <= cutoff:
                        raise ValueError('Validation measurement must be strictly later than fitting measurements')
                    heldout.append(raw)
                    aligned = None
                    if rotation is not None:
                        aligned = estimate(scan.points @ rotation.as_matrix().T, geometry_cfg, recipe) | identity
                        corrected.append(aligned)
                    row = {'partition': 'heldout', 'before': raw, 'after': aligned}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        rejection = []
        if len(fit) != fit_count or len(heldout) != validation_count:
            rejection.append('incomplete_chronological_partitions')
        profile = None
        metrics = {}
        if rotation is None:
            rejection.append('no_supported_fit_rotation')
        else:
            metrics = {'fit': stability(fit, rotation), 'heldout_before': stability(heldout, rotation),
                       'heldout_after': stability(corrected, Rotation.identity())}
            for partition, values in metrics.items():
                if values['fraction'] < recipe['min_accepted_fraction']:
                    rejection.append(partition + ':insufficient_supported_frames')
                for metric, threshold in [('max_rotation_deviation_deg', 'max_rotation_deviation_deg'),
                                          ('height_span_m', 'max_height_span_m'),
                                          ('center_span_m', 'max_center_span_m')]:
                    if values[metric] is None or values[metric] > recipe[threshold]:
                        rejection.append(partition + ':' + metric)
            profile = {'kind': 'provisional_track_relative_orientation', 'vehicle_extrinsics_verified': False,
                       'source_frame': fit[0]['sensor_frame'], 'fit_cutoff_measurement_ns': cutoff,
                       'correction_rotation': rotation.as_matrix().tolist(),
                       'correction_xyz_deg': rotation.as_euler('xyz', degrees=True).tolist(),
                       'raw_to_aligned_rotation': (rotation.as_matrix() @ np.asarray(cfg['sensor_rotation'])).tolist(),
                       'raw_to_aligned_translation': (rotation.as_matrix() @ np.asarray(cfg['sensor_translation'])).tolist(),
                       'origin': 'unchanged sensor-processing origin; not railhead or vehicle front',
                       'numerical_stability_accepted': not rejection, 'provenance': provenance}
            write_json(folder / 'orientation-candidate.json', profile)
        summary = {'bag': bag.name, 'numerical_stability_accepted': not rejection, 'rejection_reasons': rejection,
                   'metrics': metrics, 'candidate_xyz_deg': profile['correction_xyz_deg'] if profile else None,
                   'fit_frames': len(fit), 'heldout_frames': len(heldout), 'ingestion': ingestion,
                   'provenance': provenance, 'vehicle_extrinsics_verified': False,
                   'limitations': ['Track-bed normal may differ from railhead/cant plane.',
                                   'Track heading and grade cannot identify mounting angles without vehicle pose reference.',
                                   'No measured vehicle lever arm, front offset, installation stationarity or time calibration.',
                                   'Same geometry estimator is used for fitting and consistency evaluation; no independent accuracy claim.',
                                   'Neighbouring frames are correlated; spread is not statistical confidence.',
                                   'Candidate is not installed into the detector; rejected candidates are retained for inspection.']}
        write_json(folder / 'summary.json', summary)
        summaries.append(summary)
        print(json.dumps({k: summary[k] for k in ('bag', 'numerical_stability_accepted', 'rejection_reasons', 'candidate_xyz_deg')}), flush=True)
    write_json(output / 'summary.json', summaries)


if __name__ == '__main__':
    main()
