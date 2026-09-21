"""Evaluate two rail recipes on withheld real spatial bins, without obstacle labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .detector import load_config
from .geometry import TrackGeometry
from .run import digest, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=False)
    configs = [load_config(plan[k]) for k in ('before_config', 'after_config')]
    for c in configs:
        c['background']['enabled'] = False
    records = []
    for source in plan['clouds']:
        points = np.load(source)['geometry_voxel_points']
        # Whole longitudinal bins, not adjacent returns, are withheld from fitting.
        held = (np.floor(points[:, 0] / plan['split_bin_m']).astype(np.int64) % 2) == 1
        fit, query = points[~held], points[held]
        models = [TrackGeometry(fit, c) for c in configs]
        if not all(g.valid for g in models):
            records.append({'source': source, 'valid': False, 'reasons': [g.reason for g in models]})
            continue
        paths = [g.path(query[:, 0]) for g in models]
        ground, uncertainty = models[0].ground(query)
        height = query[:, 2] - ground
        lo, hi = configs[0]['rail_height_bounds_m']
        shared = ((query[:, 0] >= plan['range_m'][0]) & (query[:, 0] < plan['range_m'][1])
                  & (height > lo) & (height < hi)
                  & (uncertainty < configs[0]['ground_max_uncertainty_m']))
        residuals = [np.abs(np.abs(query[:, 1] - c) - gauge / 2) for c, gauge, _ in paths]
        for _, _, u in paths:
            shared &= u <= configs[0]['path_max_uncertainty_m']
        # Shared union avoids evaluating each method only on its own easy points.
        shared &= np.minimum(*residuals) < plan['selection_half_width_m']
        record = {'source': source, 'source_sha256': digest(Path(source)), 'valid': True,
                  'fit_points': len(fit), 'withheld_points': len(query), 'shared_evaluation_points': int(shared.sum())}
        for label, residual in zip(('before', 'after'), residuals):
            values = residual[shared]
            record[label] = {'median_m': float(np.median(values)) if len(values) else None,
                             'p90_m': float(np.quantile(values, .9)) if len(values) else None,
                             'within_half_width': int(np.count_nonzero(values < configs[0]['rail_half_width_m']/2))}
        records.append(record)
        print(json.dumps(record), flush=True)
    write_json(output/'summary.json', {'records': records, 'limitations': [
        'Real spatially withheld returns, not independent surveyed rail labels.',
        'Shared selection uses both fitted paths and baseline bed; measures geometric fit only.',
        'Existing recordings are development data; no precision or collision-safety claim.']})
    write_json(output/'experiment.json', plan)
    write_json(output/'manifest.json', environment() | {'recipe_sha256': digest(args.experiment),
        'config_sha256': [digest(Path(plan[k])) for k in ('before_config','after_config')],
        'geometry_sha256': digest(Path(__file__).with_name('geometry.py')), 'report_sha256': digest(Path(__file__))})

if __name__ == '__main__':
    main()
