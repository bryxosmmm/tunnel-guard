"""Configured real PointCloud2 delivery experiment; no substitute node or simulated outputs."""
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import hashlib
import platform
from collections import Counter

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import MarkerArray
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

recipe = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
(out / 'experiment.json').write_text(json.dumps(recipe, indent=2) + '\n')
from tunnel_guard import _native
root = Path(_native.__file__).parent
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in list(root.glob('*.py')) + [Path(_native.__file__), Path(recipe['detector_config'])]}
(out / 'manifest.json').write_text(json.dumps(dict(platform=platform.platform(), python=sys.version,
    command=sys.argv, sha256=hashes), indent=2) + '\n')
phase = 'startup'
counts = Counter()
sent = Counter()
statuses = []
start = time.monotonic()
rclpy.init()
recorder = Node('delivery_evidence_recorder')
publisher = recorder.create_publisher(PointCloud2, recipe['input_topic'], 1)
store = get_typestore(Stores.ROS2_HUMBLE)
writer = Writer(out / 'output', version=8)
writer.open()
stream = (out / 'received.jsonl').open('w')

def received(topic, message):
    writer.write(connections[topic], time.time_ns(), bytes(serialize_message(message)))
    counts[topic] += 1
    row = {'elapsed_s': time.monotonic() - start, 'phase': phase, 'topic': topic}
    if topic == 'status':
        row['result'] = json.loads(message.data)
        statuses.append(row)
    elif topic == 'attention_required':
        row['value'] = bool(message.data)
    elif topic == 'nearest_obstacle_m':
        row['value'] = None if math.isnan(message.data) else message.data
    elif topic == 'points_display':
        row.update(width=message.width, height=message.height, frame_id=message.header.frame_id)
    else:
        row['actions'] = [marker.action for marker in message.markers]
        row['frames'] = sorted({marker.header.frame_id for marker in message.markers})
    stream.write(json.dumps(row, allow_nan=False) + '\n')
    stream.flush()

kinds = {'status': ('std_msgs/msg/String', String), 'attention_required': ('std_msgs/msg/Bool', Bool),
         'nearest_obstacle_m': ('std_msgs/msg/Float32', Float32),
         'points_display': ('sensor_msgs/msg/PointCloud2', PointCloud2),
         'debug_markers': ('visualization_msgs/msg/MarkerArray', MarkerArray)}
connections = {topic: writer.add_connection('/perception/' + topic, kind, typestore=store)
               for topic, (kind, _) in kinds.items()}
subscriptions = [recorder.create_subscription(native, '/perception/' + topic,
                 lambda message, topic=topic: received(topic, message), 100)
                 for topic, (_, native) in kinds.items()]

def spin_for(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(recorder, timeout_sec=min(0.1, max(0, deadline - time.monotonic())))

def publish(message):
    publisher.publish(message)
    sent[phase] += 1
    stamp = message.header.stamp
    stream.write(json.dumps({'phase': phase, 'direction': 'sent', 'stamp_ns': stamp.sec * 10**9 + stamp.nanosec,
                            'frame_id': message.header.frame_id, 'elapsed_s': time.monotonic() - start}) + '\n')
    stream.flush()

node_log = (out / 'node.log').open('w')
process = subprocess.Popen([sys.executable, '-m', 'tunnel_guard.ros_node', '--ros-args',
    '-p', 'config:=' + recipe['detector_config'], '-p', 'input_topic:=' + recipe['input_topic'],
    '-p', 'input_timeout_s:=' + str(recipe['input_timeout_s']),
    '-p', 'use_sim_time:=' + str(recipe['use_sim_time']).lower()], stdout=node_log, stderr=subprocess.STDOUT)
try:
    deadline = time.monotonic() + 30
    while (publisher.get_subscription_count() == 0
           or any(recorder.count_publishers('/perception/' + topic) == 0 for topic in kinds)) and time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Actual ROS detector exited; inspect node.log')
        spin_for(0.1)
    if publisher.get_subscription_count() == 0 or any(
            recorder.count_publishers('/perception/' + topic) == 0 for topic in kinds):
        raise RuntimeError('DDS input/output endpoints were not discovered')
    spin_for(recipe['dds_settle_s'])
    (out / 'qos.txt').write_text(str(recorder.get_publishers_info_by_topic('/perception/points_display'))) 
    print('delivery_recorder_ready', flush=True)
    with Reader(Path(recipe['bag'])) as reader:
        inputs = [c for c in reader.connections if c.topic == recipe['input_topic']]
        source = iter(reader.messages(connections=inputs))
        phase = 'sequential'
        if recipe['mode'] == 'startup_silence':
            phase = 'startup_silence'
            spin_for(recipe['input_timeout_s'] + 2.0)
        elif recipe['mode'] == 'rate1':
            failures = []
            def produce():
                try:
                    first_record = None
                    start_playback = time.monotonic()
                    for _, record_ns, raw in source:
                        if first_record is None:
                            first_record = record_ns
                        due = start_playback + (record_ns - first_record) * 1e-9
                        time.sleep(max(0, due - time.monotonic()))
                        publish(deserialize_message(bytes(raw), PointCloud2))
                except Exception as error:
                    failures.append(repr(error))
            worker = threading.Thread(target=produce)
            worker.start()
            while worker.is_alive():
                spin_for(.1)
            worker.join()
            if failures:
                raise RuntimeError(failures)
            spin_for(recipe['drain_s'])
            if process.poll() is not None:
                raise RuntimeError(f'Detector exited during rate1: {process.returncode}')
        else:
            def publish_and_observe(message):
                stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                publish(message)
                deadline = time.monotonic() + recipe['result_timeout_s']
                while not any(r['result'].get('measurement_timestamp_ns') == stamp for r in statuses):
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f'No actual result for {stamp}; detector exit={process.poll()}; inspect retained logs')
                    spin_for(.05)
                spin_for(.1)
            for index in range(recipe['sequential_frames']):
                _, _, raw = next(source)
                message = deserialize_message(bytes(raw), PointCloud2)
                publish_and_observe(message)
            phase = 'duplicate_stream'
            deadline = time.monotonic() + recipe['duplicate_duration_s']
            while time.monotonic() < deadline:
                publish(message)
                spin_for(recipe['duplicate_interval_s'])
            phase = 'silence'
            spin_for(recipe['silence_duration_s'])
            phase = 'recovery'
            for _ in range(recipe['recovery_frames']):
                _, _, raw = next(source)
                publish_and_observe(deserialize_message(bytes(raw), PointCloud2))
    phase = 'final_observation'
    spin_for(1.0)
    summary = {'sent_by_phase': dict(sent), 'received_by_topic': dict(counts),
               'unknown_phases': [r['phase'] for r in statuses if r['result']['status'] == 'unknown'],
               'status_frames': dict(Counter(r['result']['status'] for r in statuses)),
               'discovered_topics': recorder.get_topic_names_and_types(),
               'final_status': statuses[-1]['result'] if statuses else None,
               'limits': recipe['limits']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('final_status','discovered_topics')}), flush=True)
finally:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    (out / 'process-exit.json').write_text(json.dumps({'returncode': process.returncode}) + '\n')
    node_log.close()
    writer.close()
    stream.close()
    recorder.destroy_node()
    rclpy.shutdown()
