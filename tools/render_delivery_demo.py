"""Render recorded measurements and decisions; no inference or invented scene states."""
import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from tunnel_guard.visualization import replay_frame

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--annotations', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--live', type=Path, help='Actual DDS recording directory, including received.jsonl and output bag')
a = p.parse_args()
labels = {r['frame']: r['objects'] for r in json.loads(a.annotations.read_text())['frames']}
frames = []


def pack(title, row, cloud, markers, annotation=None):
    objects = row['objects']
    target = None
    if annotation:
        low, high = np.array(annotation['bbox_min']), np.array(annotation['bbox_max'])
        def overlap(o):
            lo, hi = np.array(o['bbox_min']), np.array(o['bbox_max'])
            intersection = np.prod(np.maximum(0, np.minimum(high, hi) - np.maximum(low, lo)))
            return float(intersection / max(1e-12, np.prod(high-low)+np.prod(hi-lo)-intersection))
        target = max(objects, key=overlap, default=None)
        if target and overlap(target) == 0:
            target = None
    visible = []
    for m in markers:
        if m.ns == 'reference_envelope' or (target and m.id == target['track_id'] and
                m.ns in ('candidate_measurements', 'observed_support')):
            visible.append(dict(ns=m.ns, type=m.type, xyz=[[p.x, p.y, p.z] for p in m.points],
                                color=[m.color.r, m.color.g, m.color.b]))
    return dict(title=title, stamp=str(row.get('measurement_timestamp_ns')), status=row['status'],
                health=row['health'], reasons=row['health_reasons'], target=target,
                cloud=cloud.tolist(), markers=visible, annotation=annotation,
                all_objects=len(objects), nearest=row.get('nearest_obstacle_m'))


for index in (0, 6, 10, 30, 60, 70, 80, 104):
    row, cloud, markers = replay_frame(a.run, 'doubleT_obstacle', index)
    frames.append(pack(f'Запись · кадр {index}', row, cloud, markers, labels[index][0]))

if a.live:
    records = [json.loads(line) for line in (a.live/'received.jsonl').read_text().splitlines()]
    statuses = [r for r in records if r.get('topic') == 'status']
    selected = []
    for phase, unavailable in [('sequential', False), ('duplicate_stream', True), ('recovery', False), ('startup_silence', True)]:
        matches = [(i, r) for i, r in enumerate(statuses)
                   if r['phase'] == phase and (r['result']['measurement_timestamp_ns'] is None) == unavailable]
        if matches:
            selected.append(matches[-1])
    store = get_typestore(Stores.ROS2_HUMBLE)
    topic_messages = {}
    with Reader(a.live/'output') as reader:
        for c, _, raw in reader.messages():
            topic_messages.setdefault(c.topic.split('/')[-1], []).append(store.deserialize_cdr(raw, c.msgtype))
    # An ordinal join is allowed only when all five recorded counts agree.
    if len({len(v) for v in topic_messages.values()}) != 1 or len(topic_messages) != 5:
        raise ValueError('DDS topic counts differ; cannot join freshness scenes by ordinal')
    for i, r in selected:
        cloud = topic_messages['points_display'][i]
        xyz = np.frombuffer(cloud.data, dtype='<f4').reshape(-1, 3)
        row = r['result']
        markers = topic_messages['debug_markers'][i].markers
        if row.get('measurement_timestamp_ns') is not None:
            stamp = cloud.header.stamp.sec*10**9+cloud.header.stamp.nanosec
            if stamp != row['measurement_timestamp_ns']:
                raise ValueError('DDS cloud/status acquisition mismatch')
        title = {'sequential': 'DDS · свежие измерения', 'duplicate_stream': 'DDS · unknown, сцена очищена',
                 'recovery': 'DDS · новые timestamps, восстановление', 'startup_silence': 'DDS · нет входа: unknown, пустая сцена'}[r['phase']]
        frames.append(pack(title, row, xyz, markers))

html = Path(__file__).with_name('delivery_demo.html').read_text()
html = html.replace('__LIVE_SCOPE__', 'DDS-состояния взяты из указанной реальной записи ROS Humble; startup_silence означает отдельный запуск без входа, не проверку потери после inference и восстановления' if a.live else 'DDS-состояния не включены в этот файл')
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(html.replace('__RECORDED_FRAMES__', json.dumps(frames, allow_nan=False).replace('</','<\\/')))
print(a.output)
