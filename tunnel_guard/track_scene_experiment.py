"""Ray-cast authored curved/graded/canted tracks through the actual detector.

Ray casting uses the existing Open3D dependency without changing detector dispatch.
Piecewise planar swept surfaces are authored geometry, not sensor/field validation.
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
from .run import environment, write_json
from .stress import beam_directions
from .visualization import corridor_edges


def sections(stations, scene, case):
    x = np.asarray(stations)
    k, v, grade = case['horizontal_curvature'], case['vertical_curvature'], case['grade']
    center = np.column_stack((x, .5 * k * x*x,
                              scene['ground_z_m'] + grade*x + .5*v*x*x))
    tangent = np.column_stack((np.ones(len(x)), k*x, grade+v*x))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    roll = case['cant_gradient_rad_per_m'] * x
    lateral = np.column_stack((np.zeros(len(x)), np.cos(roll), np.sin(roll)))
    lateral -= np.sum(lateral*tangent, axis=1)[:, None] * tangent
    lateral /= np.linalg.norm(lateral, axis=1)[:, None]
    normal = np.cross(tangent, lateral)
    return center, tangent, lateral, normal


def mesh_scene(scene, case, recipe):
    import open3d as o3d
    vertices, triangles = [], []
    def strip(a, b):
        start = len(vertices)
        vertices.extend(np.stack((a, b), axis=1).reshape(-1, 3).tolist())
        for i in range(len(a)-1):
            j = start + 2*i
            triangles.extend([[j, j+1, j+2], [j+1, j+3, j+2]])
    s = np.arange(-5., scene['scene_end_m'] + recipe['mesh_step_m'], recipe['mesh_step_m'])
    center, _, lateral, normal = sections(s, scene, case)
    def point(y, h): return center + y*lateral + h*normal
    width = scene['tunnel_half_width_m']
    roof = scene['tunnel_ceiling_z_m'] - scene['ground_z_m']
    strip(point(-width, 0), point(width, 0))
    strip(point(-width, 0), point(-width, roof))
    strip(point(width, 0), point(width, roof))
    strip(point(-width, roof), point(width, roof))
    for y in (-scene['rail_gauge_m']/2, scene['rail_gauge_m']/2):
        w, h = scene['rail_width_m']/2, scene['rail_height_m']
        strip(point(y-w, h), point(y+w, h))
        strip(point(y-w, 0), point(y-w, h))
        strip(point(y+w, 0), point(y+w, h))
    caster = o3d.t.geometry.RaycastingScene()
    def add(v, t):
        return caster.add_triangles(o3d.core.Tensor(np.asarray(v, dtype=np.float32)),
                                    o3d.core.Tensor(np.asarray(t, dtype=np.uint32)))
    add(vertices, triangles)
    obj = case.get('object')
    target_id, corners = None, None
    if obj:
        c, t, b, n = sections([obj['station_m']], scene, case)
        c = c[0] + scene['rail_height_m'] * n[0] + obj['lateral_m']*b[0]
        dx, dy, dz = obj['dimensions_m']
        corners = np.asarray([c + i*dx*t[0] + (j-.5)*dy*b[0] + k*dz*n[0]
                              for i in (0,1) for j in (0,1) for k in (0,1)])
        faces = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                 [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
        target_id = add(corners, faces)
    return caster, target_id, corners


def main():
    import open3d as o3d
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    recipe = json.loads(args.experiment.read_text())
    scene = json.loads(Path(recipe['scene_config']).read_text())
    configs = {name:load_config(path) for name,path in recipe['detectors'].items()}
    output = Path(recipe['output']); output.mkdir(parents=True, exist_ok=False)
    for name, data in [('experiment',recipe),('scene',scene),('detectors',configs),('manifest',environment())]:
        write_json(output/(name+'.json'), data)
    snapshot = output/'source'; snapshot.mkdir()
    for file in Path(__file__).parent.glob('*.py'): shutil.copyfile(file, snapshot/file.name)
    rays = beam_directions(scene, np.random.default_rng(recipe['seed'])).astype(np.float32)
    ray_query = o3d.core.Tensor(np.column_stack((np.zeros_like(rays), rays)))
    summary = []
    for case in recipe['cases']:
        caster, target_id, corners = mesh_scene(scene, case, recipe)
        hits = caster.cast_rays(ray_query)
        distances = hits['t_hit'].numpy()
        visible = hits['geometry_ids'].numpy() == target_id if target_id is not None else np.zeros(len(rays),bool)
        detectors = {k:Detector(c) for k,c in configs.items()}
        streams = {k:(output/(case['id']+'-'+k+'.jsonl')).open('x') for k in configs}
        records, checksum = [], hashlib.sha256()
        try:
            for frame in range(recipe['frames']):
                rng = np.random.default_rng(np.random.SeedSequence([recipe['seed'],frame]))
                keep = np.isfinite(distances) & (rng.random(len(rays)) >= scene['random_drop_probability'])
                cloud = rays[keep] * (distances[keep] + rng.normal(0, scene['range_noise_sigma_m'], keep.sum()))[:,None]
                support = cloud[visible[keep]]
                truth = {'bbox_min':support.min(axis=0).tolist(), 'bbox_max':np.maximum(support.max(axis=0),support.min(axis=0)+.001).tolist()} if len(support) else None
                checksum.update(cloud.tobytes())
                for name, detector in detectors.items():
                    row = detector.process(cloud, frame*recipe['frame_period_s'])
                    objects = row['objects']
                    record = {'frame':frame,'detector':name,'visible_object_points':len(support),
                              'candidate_iou':max((box_iou(o,truth) for o in objects),default=0) if truth else None,
                              'intersection_iou':max((box_iou(o,truth) for o in objects if o.get('intersection_confirmed')),default=0) if truth else None,
                              'geometry_valid':row['geometry']['valid'],'status':row['status'],
                              }
                    records.append(record)
                    streams[name].write(json.dumps(row|{'frame':frame,'scenario':record},allow_nan=False)+'\n')
                    if frame == 0:
                        np.savez_compressed(output/(case['id']+'-'+name+'.npz'),points=cloud,
                            corridor=corridor_edges(row['geometry'],configs[name]),
                            target_corners=corners if corners is not None else np.empty((0,3)))
        finally:
            for f in streams.values(): f.close()
        result = {'case':case,'shared_cloud_sequence_sha256':checksum.hexdigest(),'records':records}
        summary.append(result)
        print(json.dumps({'case':case['id'],'last_frame':records[-len(configs):]}),flush=True)
    write_json(output/'summary.json',{'cases':summary,'scope':'Stationary-sensor ray casting of authored piecewise planar tracks; not real drives',
        'limitations':['Simplified beam/noise/dropout; no reflectance, multipath, rolling scan or real calibration.',
                       'No moving ego platform; does not validate odometry.',
                       'Visible-support IoU is a diagnostic, not field recall.']})

if __name__ == '__main__': main()
