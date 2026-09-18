"""Run explicit moving/stopped/static ray-cast scenarios through the actual detector.

Scenario semantics adapted from Gerasimov's hmm_mos_probe.object_at (525e276).
The ray caster is the existing stress.scene_scan. No HMM-MOS binary is used.
Results describe ideal authored scenes, never field accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .detector import Detector, load_config
from .evaluate import box_iou
from .run import digest, environment, write_json
from .stress import beam_directions, scene_scan


def object_at(case, frame, period, scene):
    if case['kind'] == 'none' or (case['kind'] == 'appearing' and frame < case['appear_frame']):
        return None
    x = case['start_m']
    if case['kind'] == 'moving':
        x += case['velocity_mps'] * min(frame, case.get('stop_frame', frame)) * period
    low = np.asarray([x, case.get('lateral_m', 0) - case['dimensions_m'][1] / 2,
                      scene['ground_z_m'] + scene['rail_height_m']])
    return {'bbox_min': low.tolist(), 'bbox_max': (low + case['dimensions_m']).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    cfg = load_config(recipe['detector_config'])
    scene = json.loads(Path(recipe['scene_config']).read_text())
    output = Path(recipe['output'])
    output.mkdir(parents=True, exist_ok=False)
    for name, content in [('experiment', recipe), ('detector', cfg), ('scene', scene)]:
        write_json(output / (name + '.json'), content)
    write_json(output / 'manifest.json', environment() | {
        'runner_sha256': digest(Path(__file__)), 'scene_sha256': digest(Path(recipe['scene_config']))})
    snapshot = output / 'source'
    snapshot.mkdir()
    # Includes the actual imported detector package, including frozen baselines.
    import inspect
    package = Path(inspect.getfile(Detector)).parent
    for path in package.glob('*.py'):
        shutil.copyfile(path, snapshot / path.name)
    shutil.copyfile(Path(__file__), snapshot / Path(__file__).name)
    records = []
    for case in recipe['cases']:
        case_scene = scene | case.get('scene_overrides', {})
        rays = beam_directions(case_scene, np.random.default_rng(recipe['seed']))
        detector = Detector(cfg)
        cloud_hash = hashlib.sha256()
        observed = []
        with (output / (case['id'] + '.jsonl')).open('x') as stream:
            for frame in range(recipe['frames']):
                origin_x = case['sensor_step_m'] * frame
                obj = object_at(case, frame, recipe['frame_period_s'], case_scene)
                # Independent frame noise, shared deterministic draws between
                # recipes/cases. No future geometry or scan enters the detector.
                rng = np.random.default_rng(np.random.SeedSequence([recipe['seed'], frame]))
                cloud, visible = scene_scan(case_scene, rays, origin_x, obj, rng)
                cloud_hash.update(np.ascontiguousarray(cloud).tobytes())
                row = detector.process(cloud, frame * recipe['frame_period_s'])
                truth = None
                if visible.any():
                    support = cloud[visible]
                    low = support.min(axis=0)
                    truth = {'bbox_min': low.tolist(),
                             'bbox_max': np.maximum(support.max(axis=0), low + 1e-3).tolist()}
                candidates = row['objects']
                confirmed = [o for o in candidates if o.get('intersection_confirmed', False)]
                best = lambda objects: max((box_iou(o, truth) for o in objects), default=0) if truth else 0
                measurement = {
                    'frame': frame, 'object_present': obj is not None,
                    'object_world_box': obj, 'visible_object_points': int(visible.sum()),
                    'best_candidate_iou': best(candidates), 'best_confirmed_intersection_iou': best(confirmed),
                    'status': row['status'], 'geometry_valid': row.get('geometry', {}).get('valid', False),
                    'confirmed_intersections': len(confirmed),
                }
                observed.append(measurement)
                stream.write(json.dumps(row | {'frame': frame, 'scenario': measurement}, allow_nan=False) + '\n')
        detected = [r['frame'] for r in observed if r['best_confirmed_intersection_iou'] >= recipe['minimum_iou']]
        after_stop = [r for r in observed if 'stop_frame' in case and r['frame'] >= case['stop_frame']]
        result = {
            'case': case, 'cloud_sequence_sha256': cloud_hash.hexdigest(), 'frames': observed,
            'detected_frames': detected,
            'first_detection_frame': min(detected) if detected else None,
            'after_stop_visible_frames': sum(r['visible_object_points'] > 0 for r in after_stop),
            'after_stop_detected_frames': sum(r['best_confirmed_intersection_iou'] >= recipe['minimum_iou'] for r in after_stop),
            'empty_scene_alarm_frames': sum(r['status'] in ('obstacle', 'unresolved_obstacle') for r in observed) if case['kind'] == 'none' else None,
        }
        records.append(result)
        print(json.dumps({k:v for k,v in result.items() if k != 'frames'}), flush=True)
    write_json(output / 'summary.json', {
        'cases': records, 'scope': 'Ray-cast procedural diagnostic, not field safety/recall',
        'limitations': ['Ideal straight tunnel, fixed beams, simplified independent range noise/dropout.',
                       'No material reflectance, multipath, rolling acquisition or real calibration error.',
                       'Box IoU uses visible ray support; no invisible-object detection is implied.']})


if __name__ == '__main__':
    main()
