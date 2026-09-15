"""Measure background rejection versus visible obstacle preservation."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .detector import Detector, load_config
from .geometry import TrackGeometry, voxel_representatives
from .evaluate import box_iou
from .run import environment, write_json
from .stress import beam_directions, scene_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(plan['detector_config'])
    scene = json.loads(Path(plan['scene_config']).read_text())
    scene.update(tunnel_half_width_m=plan['half_width_m'],
                 tunnel_ceiling_z_m=scene['ground_z_m'] + scene['rail_height_m'] + plan['ceiling_above_rail_m'],
                 range_noise_sigma_m=plan['noise_sigma_m'], random_drop_probability=plan['drop_probability'])
    write_json(out/'experiment.json', plan)
    write_json(out/'detector.json', cfg)
    write_json(out/'manifest.json', environment())
    sources = out/'source'/'tunnel_guard'
    sources.mkdir(parents=True)
    for file in Path(__file__).parent.glob('*.py'):
        (sources/file.name).write_bytes(file.read_bytes())
    records = []
    for tilt in plan['tilts_deg']:
        rotation = Rotation.from_euler('xy', tilt, degrees=True).as_matrix()
        for index, target in enumerate(plan['objects']):
            rng = np.random.default_rng(np.random.SeedSequence([plan['seed'], index, *tilt]))
            rays = beam_directions(scene, rng)
            off = copy.deepcopy(cfg)
            off['background']['enabled'] = False
            detectors = [Detector(off), Detector(cfg)]
            for frame in range(plan['frames']):
                points, visible = scene_scan(scene, rays, 0., {'bbox_min':target['minimum'], 'bbox_max':target['maximum']}, rng)
                points = points @ rotation.T
                valid = (np.linalg.norm(points,axis=1) >= cfg['min_range_m']) & (np.linalg.norm(points,axis=1) <= cfg['max_range_m'])
                crop = valid & (points[:,0] >= cfg['min_forward_m']) & (np.abs(points[:,1]) < cfg['context_half_width_m'])
                geometry = TrackGeometry(voxel_representatives(points[crop],cfg['geometry_voxel_m']),cfg)
                after = geometry.classify(points)[1] & crop
                before = geometry.classify(points, remove_background=False)[1] & crop
                denominator = np.count_nonzero(before & visible)
                kept = np.count_nonzero(after & visible)
                target_points = points[visible]
                truth = {'bbox_min':target_points.min(axis=0).tolist(), 'bbox_max':np.maximum(target_points.max(axis=0),target_points.min(axis=0)+1e-3).tolist()} if len(target_points) else None
                runs = [d.process(points,frame*.1) for d in detectors]
                ious = [max((box_iou(o,truth) for o in row['objects'] if o['confirmed']),default=0.) if truth else 0. for row in runs]
                row = {'target':target['name'],'tilt_deg':tilt,'frame':frame,'visible_points':int(visible.sum()),
                       'target_roi_before':int(denominator),'target_roi_after':int(kept),
                       'retained_fraction':kept/denominator if denominator else None,
                       'background_removed_points':int(np.count_nonzero(before & ~after & ~visible)),
                       'confirmed_iou_before':ious[0],'confirmed_iou_after':ious[1],
                       'status_before':runs[0]['status'],'status_after':runs[1]['status'],
                       'processing_s_before':runs[0]['processing_s'],'processing_s_after':runs[1]['processing_s']}
                records.append(row)
                print(json.dumps(row),flush=True)
    write_json(out/'frames.json',records)
    summary={'frames':len(records),'minimum_target_retention':min(r['retained_fraction'] for r in records if r['retained_fraction'] is not None),
             'localization_regressions':sum(int(r['confirmed_iou_before']>=plan['minimum_iou'] and r['confirmed_iou_after']<plan['minimum_iou']) for r in records),
             'median_processing_s_before':float(np.median([r['processing_s_before'] for r in records])),
             'median_processing_s_after':float(np.median([r['processing_s_after'] for r in records]))}
    summary['pass']=bool(summary['minimum_target_retention']>=plan['min_retained_fraction'] and summary['localization_regressions']==0)
    write_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
