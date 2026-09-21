"""Paired end-to-end detector timing on cached real scans, with retained outputs."""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
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
from .run import digest, environment, git_revision, write_json, capture_native_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.experiment.read_text())
    config = load_config(recipe['detector_config'])
    if recipe['seed'] != config['seed']:
        raise ValueError('Seed mismatch')
    variants = [('numpy', Detector, 'numpy'), ('cpp', Detector, 'cpp')]
    baseline_hashes = None
    baseline_native_hash = None
    if recipe.get('baseline_source'):
        # Compare a saved Python source snapshot with this revision, optionally loading its
        # own compiled native module. Never rewrite the historical source snapshot.
        baseline = Path(recipe['baseline_source']).resolve()
        spec = importlib.util.spec_from_file_location('_runtime_before', baseline / '__init__.py',
                                                     submodule_search_locations=[str(baseline)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        before_native = _native
        if recipe.get('baseline_native'):
            binary = Path(recipe['baseline_native']).resolve()
            native_spec = importlib.util.spec_from_file_location(spec.name + '._native', binary)
            before_native = importlib.util.module_from_spec(native_spec)
            native_spec.loader.exec_module(before_native)
            baseline_native_hash = digest(binary)
        sys.modules[spec.name + '._native'] = before_native
        package._native = before_native
        before_detector = importlib.import_module(spec.name + '.detector').Detector
        backend = config.get('voxel_backend', 'numpy')
        variants = [('before', before_detector, backend), ('after', Detector, backend)]
        baseline_hashes = {p.name: digest(p) for p in sorted(baseline.glob('*.py'))}
    output = Path(recipe['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'experiment.json', recipe)
    write_json(output / 'detector.json', config)
    shutil.copytree(Path(__file__).parent, output / 'source' / 'tunnel_guard',
                    ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.dylib', '*.pyd'))
    native_sources = capture_native_sources(output / 'source')
    bag = Path(recipe['bag'])
    manifest = environment() | {'command': sys.argv, 'git_revision': git_revision(),
                               'native_binary_sha256': digest(Path(_native.__file__)),
                               'native_sources_sha256': native_sources,
                               'baseline_native_sha256': baseline_native_hash,
                               'config_sha256': digest(Path(recipe['detector_config'])),
                               'baseline_source_sha256': baseline_hashes,
                               'variants': [{'name': name, 'voxel_backend': backend} for name, _, backend in variants],
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
    for _, constructor, backend in variants:
        cfg = copy.deepcopy(config) | {'voxel_backend': backend}
        constructor(cfg).process(scans[0].points, scans[0].timestamp_s, scans[0].point_times)
    runs, comparisons = [], []
    for repeat in range(recipe['repetitions']):
        order = variants if repeat % 2 == 0 else list(reversed(variants))
        paths = {}
        for name, constructor, backend in order:
            cfg = copy.deepcopy(config) | {'voxel_backend': backend}
            detector = constructor(cfg)
            path = output / f'{repeat}-{name}.jsonl'
            paths[name] = path
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
            record = {'repeat': repeat, 'backend': name, 'voxel_backend': backend,
                      'order': [v[0] for v in order],
                      'frames': len(scans), 'elapsed_s': sum(elapsed),
                      'processing_ms': {name: float(np.quantile(elapsed, q)*1000)
                                        for name, q in [('p50', .5), ('p95', .95)]}}
            runs.append(record)
            print(json.dumps(record), flush=True)
        comparisons.append(compare({'bag': bag.name, 'before': str(paths[variants[0][0]]), 'after': str(paths[variants[1][0]])}))
    medians = {backend: float(np.median([r['processing_ms']['p50'] for r in runs if r['backend'] == backend]))
               for backend, _, _ in variants}
    write_json(output / 'report.json', {'runs': runs, 'comparisons': comparisons,
                                      'median_of_run_p50_ms': medians,
                                      'p50_speedup': medians[variants[0][0]] / medians[variants[1][0]],
                                      'limitations': ['Cached prefix of one development recording, not full field latency.',
                                                      'No bag decoding, rendering or ROS transport in timed region.',
                                                      'Sequential alternating runs reduce but cannot eliminate machine-load and thermal effects.',
                                                      'No accuracy conclusion; only output equivalence and cost.']})


if __name__ == '__main__':
    main()
