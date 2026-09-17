"""Paired end-to-end detector timing on cached real scans, with retained outputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from . import _native
from .detector import Detector, load_config
from .io import iter_bag
from .panel_report import compare
from .run import digest, environment, git_revision, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    config = load_config(recipe['detector_config'])
    if recipe['seed'] != config['seed']:
        raise ValueError('Seed mismatch')
    output = Path(recipe['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'experiment.json', recipe)
    write_json(output / 'detector.json', config)
    shutil.copytree(Path(__file__).parent, output / 'source' / 'tunnel_guard',
                    ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.dylib', '*.pyd'))
    source = Path(__file__).parent.parent / 'cpp' / 'voxel.cpp'
    shutil.copyfile(source, output / 'source' / 'voxel.cpp')
    bag = Path(recipe['bag'])
    manifest = environment() | {'command': sys.argv, 'git_revision': git_revision(),
                               'native_binary_sha256': digest(Path(_native.__file__)),
                               'native_source_sha256': digest(source),
                               'config_sha256': digest(Path(recipe['detector_config'])),
                               'bag_metadata_sha256': digest(bag / 'metadata.yaml'),
                               'bag_files': [{'name': p.name, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                             for p in sorted(bag.glob('*.db3'))]}
    scans = list(iter_bag(bag, config, max_frames=recipe['frames'], topic=recipe.get('topic')))
    if len(scans) != recipe['frames']:
        raise ValueError('Not enough real scans')
    manifest['measurement_timestamps_ns'] = [s.measurement_timestamp_ns for s in scans]
    write_json(output / 'manifest.json', manifest)
    # Load libraries and exercise each implementation on the same real first
    # frame before timing. Fresh detectors below retain identical cold tracking.
    for backend in ('numpy', 'cpp'):
        cfg = copy.deepcopy(config) | {'voxel_backend': backend}
        Detector(cfg).process(scans[0].points, scans[0].timestamp_s, scans[0].point_times)
    runs, comparisons = [], []
    for repeat in range(recipe['repetitions']):
        order = ('numpy', 'cpp') if repeat % 2 == 0 else ('cpp', 'numpy')
        paths = {}
        for backend in order:
            cfg = copy.deepcopy(config) | {'voxel_backend': backend}
            detector = Detector(cfg)
            path = output / f'{repeat}-{backend}.jsonl'
            paths[backend] = path
            elapsed = []
            with path.open('x') as stream:
                for scan in scans:
                    started = time.perf_counter()
                    row = detector.process(scan.points, scan.timestamp_s, scan.point_times)
                    elapsed.append(time.perf_counter() - started)
                    row.update(frame=scan.index, measurement_timestamp_ns=scan.measurement_timestamp_ns,
                               record_timestamp_ns=scan.record_timestamp_ns, sensor_frame=scan.frame_id,
                               topic=scan.topic, benchmark_elapsed_s=elapsed[-1])
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            record = {'repeat': repeat, 'backend': backend, 'order': list(order),
                      'frames': len(scans), 'elapsed_s': sum(elapsed),
                      'processing_ms': {name: float(np.quantile(elapsed, q)*1000)
                                        for name, q in [('p50', .5), ('p95', .95)]}}
            runs.append(record)
            print(json.dumps(record), flush=True)
        comparisons.append(compare({'bag': bag.name, 'before': str(paths['numpy']), 'after': str(paths['cpp'])}))
    medians = {backend: float(np.median([r['processing_ms']['p50'] for r in runs if r['backend'] == backend]))
               for backend in ('numpy', 'cpp')}
    write_json(output / 'report.json', {'runs': runs, 'comparisons': comparisons,
                                      'median_of_run_p50_ms': medians,
                                      'p50_speedup': medians['numpy'] / medians['cpp'],
                                      'limitations': ['Cached prefix of one development recording, not full field latency.',
                                                      'No bag decoding, rendering or ROS transport in timed region.',
                                                      'Sequential alternating runs reduce but cannot eliminate machine-load and thermal effects.',
                                                      'No accuracy conclusion; only output equivalence and cost.']})


if __name__ == '__main__':
    main()
