"""Replay real acquisitions and compare geometry to an immutable detector run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import accelerator
from .detector import load_config
from .geometry import TrackGeometry
from .io import iter_bag
from .run import digest, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    output = Path(recipe['output'])
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(recipe['baseline_run'])
    bag = Path(recipe['bag'])
    cfg = load_config(baseline / 'detector.json')
    # One explicit experimental policy; all other baseline settings stay fixed.
    policy = recipe.get('rail_initial_heading', 'zero')
    if policy not in ('zero', 'fitted'):
        raise ValueError('Unknown rail_initial_heading')
    cfg = cfg | {'rail_initial_heading': policy}
    for name, allowed in [('rail_pair_continuity', ('window', 'relocated')),
                          ('rail_frame_mode', ('bed',))]:
        value = recipe.get(name, allowed[0])
        if value not in allowed:
            raise ValueError('Unknown ' + name)
        cfg[name] = value
    if cfg.get('deskew_enabled', False):
        raise ValueError('Source-only geometry replay requires deskew disabled')
    manifest = json.loads((baseline / 'manifest.json').read_text())
    source = next(b for b in manifest['bags'] if Path(b['path']).name == bag.name)
    if source['metadata_sha256'] != digest(bag / 'metadata.yaml'):
        raise ValueError('Source metadata differs from baseline')
    write_json(output / 'experiment.json', recipe)
    write_json(output / 'detector.json', cfg)
    write_json(output / 'manifest.json', environment() | {
        'baseline_manifest_sha256': digest(baseline / 'manifest.json'),
        'scope': 'Geometry replay only; no tracking, obstacle accuracy or timing comparison',
    })
    snapshot = output / 'source'
    snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, snapshot / path.name)
    cfg = cfg | {'background': cfg['background'] | {'enabled': False}}
    native = accelerator.native(cfg)
    voxel_native = native
    probes = np.asarray(recipe['probe_ranges_m'], dtype=float)
    counts = {'frames': 0, 'before_valid': 0, 'after_valid': 0, 'unbracketed_anchors': 0}
    recovered, regressed, changes = [], [], []
    reference = baseline / (bag.name + '.jsonl')
    with reference.open() as stream, (output / 'geometry.jsonl').open('x') as records:
        for scan in iter_bag(bag, cfg, max_frames=recipe.get('max_frames')):
            line = stream.readline()
            if not line:
                raise ValueError('Baseline ended before source')
            before = json.loads(line)
            if (before['frame'] != scan.index or
                    before['measurement_timestamp_ns'] != scan.measurement_timestamp_ns):
                raise ValueError('Source acquisition differs from baseline')
            points = scan.points[accelerator.range_indices(
                scan.points, cfg['min_range_m'], cfg['max_range_m'], native)]
            reduced = points[accelerator.crop_voxels(
                points, cfg['min_forward_m'], cfg['context_half_width_m'],
                cfg['geometry_voxel_m'], voxel_native)]
            after = TrackGeometry(reduced, cfg)
            old = before.get('geometry', {})
            old_valid = old.get('valid', False)
            counts['frames'] += 1
            counts['before_valid'] += int(old_valid)
            counts['after_valid'] += int(after.valid)
            counts['unbracketed_anchors'] += sum(
                not a['bracketed'] for a in after.rail_support_diagnostics)
            if after.valid and not old_valid:
                recovered.append(scan.index)
            if old_valid and not after.valid:
                regressed.append(scan.index)
            delta = None
            if old_valid and after.valid:
                previous = TrackGeometry.__new__(TrackGeometry)
                previous.config = cfg
                previous.rail_anchors = np.asarray(old['rail_anchors'])
                y0, _, u0 = previous.path(probes)
                y1, _, u1 = after.path(probes)
                valid = np.isfinite(u0) & np.isfinite(u1)
                if valid.any():
                    delta = float(np.abs(y1[valid] - y0[valid]).max())
                    changes.append(delta)
            records.write(json.dumps({
                'frame': scan.index, 'measurement_timestamp_ns': scan.measurement_timestamp_ns,
                'before_valid': old_valid, 'after_valid': after.valid,
                'max_common_probe_center_change_m': delta, 'geometry': after.describe(),
            }, allow_nan=False) + '\n')
            if counts['frames'] % 500 == 0:
                records.flush()
                print(counts, 'recovered', len(recovered), 'regressed', len(regressed), flush=True)
        if recipe.get('max_frames') is None and stream.readline():
            raise ValueError('Source ended before baseline')
    result = counts | {
        'recovered_frames': recovered, 'regressed_frames': regressed,
        'common_probe_max_change_m': {
            k: float(np.quantile(changes, q)) for k, q in [('p50', .5), ('p95', .95), ('max', 1)]
        } if changes else {},
        'frames_with_changed_common_probe': sum(v > 1e-9 for v in changes),
        'baseline_result_sha256': digest(reference),
        'limitations': ['Geometry availability and disagreement are not independent rail truth.',
                       'No obstacle detection or temporal association replay in this command.'],
    }
    write_json(output / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
