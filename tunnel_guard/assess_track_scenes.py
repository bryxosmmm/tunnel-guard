"""Report authored-scene geometric errors separately from object matches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import TrackGeometry
from .run import digest, write_json
from .track_scene_experiment import sections


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('Output exists')
    recipe=json.loads((args.run/'experiment.json').read_text())
    scene=json.loads((args.run/'scene.json').read_text())
    configs=json.loads((args.run/'detectors.json').read_text())
    report=[]
    for case in recipe['cases']:
        stations=np.arange(-5.,scene['scene_end_m'],.05)
        center,_,_,normal=sections(stations,scene,case)
        truth=center+scene['rail_height_m']*normal
        rows={name:[json.loads(l) for l in (args.run/(case['id']+'-'+name+'.jsonl')).open()] for name in configs}
        metrics={name:{'height_errors_m':[],'center_errors_m':[],'normal_errors_deg':[],
                       'matched_candidate_frames':0,'matched_intersection_frames':0} for name in configs}
        support=[]
        for frame,local_row in enumerate(rows['local3d']):
            g=TrackGeometry.__new__(TrackGeometry)
            g.rail_frames=local_row['geometry']['rail_frames'];g.config=configs['local3d']
            g.rail_frame_version=local_row['geometry'].get('rail_frame_version',1)
            g.rail_support_diagnostics=local_row['geometry']['rail_support_diagnostics']
            segments=g.frame_segments()
            support.append(sum(np.linalg.norm(b-a) for a,b,*_ in segments))
            for name in configs:
                row=rows[name][frame];m=metrics[name];d=row['geometry']
                m['matched_candidate_frames']+=int((row['scenario']['candidate_iou'] or 0)>=.1)
                m['matched_intersection_frames']+=int((row['scenario']['intersection_iou'] or 0)>=.1)
                old=TrackGeometry.__new__(TrackGeometry);old.config=configs[name]
                old.rail_anchors=np.asarray(d['rail_anchors']);old.plane=np.asarray(d['ground_plane']);old.ground_anchors=np.asarray(d['ground_anchors'])
                for a,b,_,_,n,*_ in segments:
                    c=(a+b)/2;x=c[0]
                    actual=np.array([np.interp(x,truth[:,0],truth[:,i]) for i in range(3)])
                    actual_n=np.array([np.interp(x,truth[:,0],normal[:,i]) for i in range(3)])
                    actual_n/=np.linalg.norm(actual_n)
                    if name=='local3d': predicted,predicted_n=c,n
                    else:
                        y,_,u=old.path(np.array([x]));predicted=np.array([x,y[0],0.])
                        z,gu=old.ground(predicted[None,:])
                        if not np.isfinite(u[0]) or not np.isfinite(gu[0]):continue
                        predicted[2]=z[0]+d['rail_head_height_m']
                        predicted_n=np.array([-old.plane[0],-old.plane[1],1.]);predicted_n/=np.linalg.norm(predicted_n)
                    m['height_errors_m'].append(abs(float(predicted[2]-actual[2])))
                    m['center_errors_m'].append(float(np.linalg.norm(predicted-actual)))
                    m['normal_errors_deg'].append(float(np.degrees(np.arccos(np.clip(predicted_n@actual_n,-1,1)))))
        for m in metrics.values():
            for k in ['height_errors_m','center_errors_m','normal_errors_deg']:
                values=m[k];m[k]={'n':len(values),'median':float(np.median(values)) if values else None,'max':max(values) if values else None}
        report.append({'case':case['id'],'frames':len(rows['local3d']),'local_supported_length_m':support,'metrics':metrics})
    write_json(args.output,{'run':str(args.run),'input_sha256':digest(args.run/'summary.json'),'cases':report,
        'scope':'Geometric residuals at local-3D supported segment midpoints in authored scenes only.',
        'limitations':['Coverage selected by local-3D estimator, not unbiased route-wide accuracy.',
                       'Other variants skip unavailable geometry; compare sample counts.',
                       'Normals and head centres reference the analytic scene; meshes approximate it with finite segments.']})
    for r in report: print(json.dumps(r),flush=True)

if __name__=='__main__':main()
