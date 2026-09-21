"""Extract measured alert support and neighbouring scans for offline cause diagnosis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree

from .geometry import TrackGeometry
from .io import iter_bag
from .run import digest, write_json


def geometry(row, config):
    d = row['geometry']
    g = object.__new__(TrackGeometry)
    g.config, g.background = config, None
    g.plane = np.asarray(d['ground_plane']) if d['ground_plane'] is not None else None
    g.ground_anchors = np.asarray(d['ground_anchors']).reshape(-1, 3)
    g.rail_anchors = np.asarray(d['rail_anchors']).reshape(-1, 4)
    g.rail_head_height_m = d['rail_head_height_m']
    return g


def transform(points, source, target):
    source, target = np.asarray(source['pose']), np.asarray(target['pose'])
    world = points @ source[:3, :3].T + source[:3, 3]
    return (world - target[:3, 3]) @ target[:3, :3]


def extract(plan):
    root, run = Path(plan['output']), Path(plan['run'])
    config = json.loads((run / 'detector.json').read_text())
    if config.get('deskew_enabled', False):
        raise ValueError('Raw-scan review requires deskew-disabled results; otherwise save and use the registered cloud.')
    root.mkdir(parents=True, exist_ok=False)
    store = get_typestore(Stores.ROS2_HUMBLE)
    evidence = {'config_sha256': digest(run / 'detector.json'), 'bags': {},
                'warning': 'Offline measurement review, not new ground truth or detection accuracy. '
                           'Neighbour counterfactuals assume stationary support and estimated poses.'}
    for bag in [plan['platform_bag'], plan['obstacle_bag']]:
        rows = {r['frame']: r for r in map(json.loads, (run / f'{bag}.jsonl').read_text().splitlines())}
        target = ([f for f, r in rows.items() if r['status'] == 'obstacle']
                  if bag == plan['platform_bag'] else plan['obstacle_frames'])
        selected = sorted({n for f in target for n in [f - 1, f, f + 1] if n in rows})
        timestamps = {rows[f]['measurement_timestamp_ns']: f for f in selected}
        folder = root / bag
        folder.mkdir()
        support = {f: {} for f in selected}
        with Reader(run / f'{bag}_rviz') as reader:
            connections = [c for c in reader.connections if c.topic == '/perception/debug_markers']
            for connection, stamp, raw in reader.messages(connections=connections):
                if stamp not in timestamps:
                    continue
                msg = store.deserialize_cdr(raw, connection.msgtype)
                frame = timestamps[stamp]
                for marker in msg.markers:
                    if marker.ns == 'candidate_measurements':
                        support[frame][marker.id] = np.asarray([[p.x, p.y, p.z] for p in marker.points]).reshape(-1, 3)
        clouds = {}
        for scan in iter_bag(Path(plan['bag_root']) / bag, config):
            if scan.index not in selected:
                continue
            clouds[scan.index] = scan.points
            if len(clouds) == len(selected):
                break
        records = []
        for f in selected:
            arrays = {'points': clouds[f]}
            arrays.update({f'track_{tid}': q for tid, q in support[f].items()})
            np.savez_compressed(folder / f'{f:06d}.npz', **arrays)
        for f in target:
            r = rows[f]
            g = geometry(r, config)
            objects = ([o for o in r['objects'] if o['confirmed'] and o['path_relation'] == 'intersecting']
                       if bag == plan['platform_bag'] else
                       [o for o in r['objects'] if np.all(np.asarray(o['bbox_max']) >= plan['obstacle_region']['low'])
                        and np.all(np.asarray(o['bbox_min']) <= plan['obstacle_region']['high'])])
            for obj in objects:
                q = support[f][obj['track_id']]
                core = g.classify(q, remove_background=False)[0]
                neighbours = []
                for n in [f - 1, f + 1]:
                    if n not in rows or n not in clouds:
                        continue
                    projected = transform(q, r, rows[n])
                    distances = cKDTree(clouds[n]).query(projected)[0]
                    ng = geometry(rows[n], config)
                    neighbour_core = ng.classify(projected, remove_background=False)[0]
                    prev = next((o for o in rows[n]['objects'] if o['track_id'] == obj['track_id']), None)
                    neighbours.append({'frame': n, 'same_support_core_under_neighbour_geometry': int(neighbour_core.sum()),
                                       'surface_match_median_m': float(np.median(distances)),
                                       'surface_match_p95_m': float(np.quantile(distances, .95)),
                                       'core_surface_match_median_m': float(np.median(distances[core])) if core.any() else None,
                                       'same_id_relation': prev['path_relation'] if prev else None,
                                       'same_id_core': prev['in_envelope_voxels'] if prev else None})
                records.append({'frame': f, 'object': {k: obj[k] for k in [
                    'track_id', 'bbox_min', 'bbox_max', 'support_voxels', 'in_envelope_voxels',
                    'boundary_uncertain_voxels', 'confirmation', 'confirmed', 'path_relation', 'hits']},
                    'recomputed_core': int(core.sum()), 'neighbours': neighbours})
        evidence['bags'][bag] = {'target_frames': target, 'extracted_frames': selected,
                                'prediction_sha256': digest(run / f'{bag}.jsonl'), 'records': records}
        print(bag, 'extracted', len(selected), 'frames; reviewed', len(records), 'candidate observations', flush=True)
    write_json(root / 'evidence.json', evidence)
    write_json(root / 'plan.json', plan)


def render(plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from .visualization import corridor_edges

    root, run = Path(plan['output']), Path(plan['run'])
    config = json.loads((run / 'detector.json').read_text())
    evidence = json.loads((root / 'evidence.json').read_text())
    for bag in [plan['platform_bag'], plan['obstacle_bag']]:
        rows = {r['frame']: r for r in map(json.loads, (run / f'{bag}.jsonl').read_text().splitlines())}
        targets = evidence['bags'][bag]['target_frames']
        for projection in [(0, 1), (0, 2)]:
            x, y = projection
            batches = [targets[i:i+5] for i in range(0, len(targets), 5)] if bag == plan['platform_bag'] else [targets]
            for page, batch in enumerate(batches):
                platform = bag == plan['platform_bag']
                nr = len(batch) if platform else int(np.ceil(len(batch)/3))
                fig, axes = plt.subplots(nr, 3, figsize=(16, nr*2.7), squeeze=False)
                for k, anchor in enumerate(batch):
                    if platform:
                        objs = [o for o in rows[anchor]['objects'] if o['confirmed'] and o['path_relation']=='intersecting']
                        low = np.min([o['bbox_min'] for o in objs],axis=0)-[1,.6,.5]
                        high = np.max([o['bbox_max'] for o in objs],axis=0)+[1,.6,.5]
                        neighbours = [anchor-1,anchor,anchor+1]
                    else:
                        low,high = np.array(plan['obstacle_region']['low']),np.array(plan['obstacle_region']['high'])
                        neighbours = [anchor]
                    for j,n in enumerate(neighbours):
                        ax = axes[k,j] if platform else axes[k//3,k%3]
                        if n not in rows:
                            ax.set_visible(False)
                            continue
                        with np.load(root/bag/f'{n:06d}.npz') as data:
                            scene = transform(data['points'],rows[n],rows[anchor])
                            visible = np.all((scene>=low)&(scene<=high),axis=1)
                            scene=scene[visible]
                            ax.scatter(scene[:,x],scene[:,y],s=.35,c='.65',rasterized=True)
                            g=geometry(rows[n],config)
                            for obj in rows[n]['objects']:
                                key=f"track_{obj['track_id']}"
                                if key not in data:continue
                                q=data[key]; core,_,_,obs,nom,bound=g.classify(q,remove_background=False,include_boundary=True)
                                shown=transform(q,rows[n],rows[anchor])
                                roi=np.all((shown>=low)&(shown<=high),axis=1)
                                uncertain=((~obs&nom)|bound)&roi
                                ax.scatter(shown[uncertain,x],shown[uncertain,y],s=3,c='orange',rasterized=True)
                                use=core&roi
                                ax.scatter(shown[use,x],shown[use,y],s=5,c='red' if obj['confirmed'] else 'gold',rasterized=True)
                        edges=corridor_edges(rows[n]['geometry'],config).reshape(-1,2,3)
                        edges=transform(edges.reshape(-1,3),rows[n],rows[anchor]).reshape(-1,2,3)
                        edges=edges[(edges[:,:,0].max(axis=1)>=low[0])&(edges[:,:,0].min(axis=1)<=high[0])]
                        ax.add_collection(LineCollection(edges[:,:,[x,y]],colors='teal',linewidths=.5,alpha=.6))
                        ax.set_xlim(low[x],high[x]);ax.set_ylim(low[y],high[y])
                        ax.set_xlabel('xyz'[x]+' m');ax.set_ylabel('xyz'[y]+' m')
                        ax.set_title(f"frame {n} ({rows[n]['status']})\n"+(f'pose-aligned to frame {anchor}' if platform else 'fixed 56m review region'),fontsize=9)
                fig.suptitle(f'{bag}: actual measurements and algorithmic evidence ({"xy" if y==1 else "xz"})\nRed: confirmed object interior points; orange: uncertain; teal: estimated contour. Not verified hazard labels.')
                fig.tight_layout(rect=(0,0,1,.96))
                path=root/f'{bag}_{"xy" if y==1 else "xz"}_{page}.png'
                fig.savefig(path,dpi=135);plt.close(fig)
                print(path,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    plan=json.loads(args.config.read_text())
    if not args.render_only:
        extract(plan)
    render(plan)


if __name__ == '__main__':
    main()
