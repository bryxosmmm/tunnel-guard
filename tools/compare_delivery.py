"""Compare actual recorded detector runs, rejecting incomplete delivery evidence."""
import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('before', type=Path)
p.add_argument('after', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
runtime = {'processing_s', 'motion_s', 'mounting_observation_s', 'deserialize_s',
           'decode_s', 'ingestion_s', 'inference_s', 'read_and_process_s',
           'visualization_s', 'diagnostic_write_s'}


def differing_fields(left, right):
    return sorted(k for k in left.keys() | right.keys()
                  if k not in runtime and (k not in left or k not in right or left[k] != right[k]))


def timings(root):
    with (root / 'doubleT_obstacle-timing.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    if not records:
        return {}
    return {key: {name: float(np.quantile([r[key] for r in records], q) * 1000)
                  for name, q in [('p50', .5), ('p95', .95), ('p99', .99)]}
            for key in records[0] if key.endswith('_s')}


frame_counts = [0, 0]
differences = []
with (a.before / 'doubleT_obstacle.jsonl').open() as before, \
        (a.after / 'doubleT_obstacle.jsonl').open() as after:
    for ordinal, (left, right) in enumerate(zip_longest(before, after)):
        frame_counts[0] += left is not None
        frame_counts[1] += right is not None
        if left is None or right is None:
            differences.append({'ordinal': ordinal, 'fields': ['missing_before' if left is None else 'missing_after']})
            continue
        x, y = json.loads(left), json.loads(right)
        fields = differing_fields(x, y)
        if fields:
            differences.append({'frame': x['frame'], 'fields': fields})

store = get_typestore(Stores.ROS2_HUMBLE)
counts, bad = Counter(), Counter()
order_mismatches = 0
status_differences = []
with Reader(a.before / 'doubleT_obstacle_rviz') as x, Reader(a.after / 'doubleT_obstacle_rviz') as y:
    for ordinal, (left, right) in enumerate(zip_longest(x.messages(), y.messages())):
        if left is None or right is None:
            order_mismatches += 1
            continue
        cx, tx, rx = left
        cy, ty, ry = right
        if (cx.topic, cx.msgtype, tx) != (cy.topic, cy.msgtype, ty):
            order_mismatches += 1
            continue
        if cx.topic.endswith('/status'):
            fields = differing_fields(json.loads(store.deserialize_cdr(rx, cx.msgtype).data),
                                      json.loads(store.deserialize_cdr(ry, cy.msgtype).data))
            if fields:
                status_differences.append({'ordinal': ordinal, 'fields': fields})
        else:
            counts[cx.topic] += 1
            if bytes(rx) != bytes(ry):
                bad[cx.topic] += 1
    expected_topics = {'/perception/' + name for name in
                       ('points_display', 'debug_markers', 'status', 'attention_required', 'nearest_obstacle_m')}
    complete = all(reader.message_count == frames * 5
                   and {c.topic for c in reader.connections} == expected_topics
                   and all(c.msgcount == frames for c in reader.connections)
                   for reader, frames in ((x, frame_counts[0]), (y, frame_counts[1])))
    report = {'before_frames': frame_counts[0], 'after_frames': frame_counts[1],
              'ignored_runtime_fields': sorted(runtime), 'decision_differences': differences,
              'message_order_or_timestamp_mismatches': order_mismatches,
              'display_messages_compared': dict(counts), 'display_cdr_differences': dict(bad),
              'status_semantic_differences': status_differences,
              'before_message_count': x.message_count, 'after_message_count': y.message_count,
              'complete_five_topic_delivery': complete,
              'before_ms': timings(a.before), 'after_ms': timings(a.after),
              'scope': 'Actual recorded outputs, not DDS or GUI. All non-runtime row/status fields and four non-status CDR topics.'}
report['equivalent'] = bool(frame_counts[0] and complete and not differences
                            and not order_mismatches and not bad and not status_differences)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['equivalent'] else 1)
