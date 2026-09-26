"""Explain frozen measured support through the actual native classifier, without refitting history."""
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from tunnel_guard import accelerator
from tunnel_guard.detector import load_config
from tunnel_guard.geometry import TrackGeometry, envelope_width_bounds
from tunnel_guard.run import digest, write_json

plan = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
config = load_config(plan['detector_config'])
report = json.loads(Path(plan['support_report']).read_text())
records = []
with zipfile.ZipFile(plan['support_archive']) as archive:
    references = {r['frame']: r for r in map(json.loads, archive.read(
        'build/issue23-point-support-20260924/reference.jsonl').splitlines())}
    for entry in report['frames']:
        row = references[entry['frame']]
        saved = row['geometry']
        geometry = SimpleNamespace(config=config, plane=np.array(saved['ground_plane']),
            ground_anchors=np.array(saved['ground_anchors']), rail_anchors=np.array(saved['rail_anchors']),
            rail_head_height_m=saved['rail_head_height_m'])
        geometry._continuation = lambda edge: TrackGeometry._continuation(geometry, edge)
        raw = archive.read(entry['artifact'])
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Frozen support digest mismatch')
        with np.load(io.BytesIO(raw)) as arrays:
            points = arrays['cluster_points']
            native = accelerator.classify_geometry(points, geometry, accelerator.native(config))
            core, _, _, observed, _, _, lateral, running, _ = native
            center, _, path_error = TrackGeometry.path(geometry, points[:, 0])
            _, bed_error = TrackGeometry.ground(geometry, points)
            normal_scale = np.sqrt(1 + np.sum(geometry.plane[:2] ** 2))
            lateral_scale = np.sqrt(1 + geometry.plane[1] ** 2)
            width, _ = envelope_width_bounds(running - bed_error / normal_scale,
                running + bed_error / normal_scale, np.array(config['envelope_segments_m']), config['envelope_margin_m'])
            lateral_error = path_error * lateral_scale + abs(geometry.plane[1]) * bed_error / lateral_scale
            margin = np.full(len(points), np.nan)
            finite = np.isfinite(width) & np.isfinite(lateral_error)
            margin[finite] = width[finite] - np.abs(lateral[finite]) - lateral_error[finite]
            continuation = geometry._continuation(-1)
            for obj in row['objects']:
                if not obj['intersection_confirmed']:
                    continue
                mask = (arrays['cluster_labels'] == obj['component_id']) & arrays['cluster_core']
                indices = np.flatnonzero(mask)
                after = points[indices, 0] > geometry.rail_anchors[-1, 0]
                records.append({'frame': entry['frame'], 'component_id': obj['component_id'],
                    'original_continuous_confirmation': obj['intersection_confirmation'],
                    'last_anchor_m': float(geometry.rail_anchors[-1, 0]),
                    'continuation_x_y_slope_curvature_slope_sigma_curvature_sigma_covariance': list(continuation),
                    'certified_points': len(indices), 'after_last_anchor': int(after.sum()),
                    'native_recognizes_all_frozen_core': bool(np.all(core[indices])),
                    'points': [{'source_index': int(arrays['cluster_source_indices'][i]),
                        'xyz_m': points[i].tolist(), 'path_center_m': float(center[i]),
                        'path_error_m': float(path_error[i]), 'bed_error_m': float(bed_error[i]),
                        'lateral_m': float(lateral[i]), 'running_height_m': float(running[i]),
                        'lateral_error_m': float(lateral_error[i]), 'minimum_contour_half_width_m': float(width[i]),
                        'certified_lateral_margin_m': float(margin[i]), 'observed': bool(observed[i]),
                        'native_core': bool(core[i])} for i in indices]})
write_json(out / 'certification.json', {'records': records,
    'config_sha256': digest(Path(plan['detector_config'])),
    'native_module': str(accelerator.native(config).__file__),
    'all_core_reproduced': all(r['native_recognizes_all_frozen_core'] for r in records),
    'limits': ['Frozen rail and ground estimates, not refitted scan history.',
               'A positive margin proves the implemented heuristic inequality, not correct physical clearance.',
               'Extrapolation alone is not evidence of a false alarm; anchor association and uncertainty calibration remain separate questions.']})
print(json.dumps({'components': len(records), 'all_core_reproduced': all(r['native_recognizes_all_frozen_core'] for r in records),
    'all_support_after_anchor': sum(r['certified_points'] > 0 and r['after_last_anchor'] == r['certified_points'] for r in records),
    'output': str(out / 'certification.json')}))
