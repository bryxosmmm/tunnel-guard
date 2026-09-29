"""Actual DDS replay of recorded detector messages; isolates delivery, does not run inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import MarkerArray
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['roundtrip','raw'],required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rclpy.init();node=Node('recorded_display_delivery');store=get_typestore(Stores.ROS2_HUMBLE)
native={'sensor_msgs/msg/PointCloud2':PointCloud2,'std_msgs/msg/String':String,
        'std_msgs/msg/Bool':Bool,'std_msgs/msg/Float32':Float32,'visualization_msgs/msg/MarkerArray':MarkerArray}
counts=Counter();pending={};mismatch=[];received_times={}
stream=(a.output/'timing.jsonl').open('w')
with Reader(a.input) as reader,Writer(a.output/'output',version=8) as writer:
    connections={c.topic:writer.add_connection(c.topic,c.msgtype,typestore=store) for c in reader.connections}
    publishers={c.topic:node.create_publisher(native[c.msgtype],c.topic,1) for c in reader.connections}
    def receive(topic,raw):
        received_times[topic]=time.monotonic()
        writer.write(connections[topic],time.time_ns(),raw)
        counts[topic]+=1
        expected=pending.get(topic)
        # Raw subscriptions expose the actual received CDR without native conversion.
        # Canonicalize padding/encapsulation differences through the same typestore.
        kind=connections[topic].msgtype
        canonical=bytes(store.serialize_cdr(store.deserialize_cdr(raw,kind),kind))
        if expected!=canonical:mismatch.append({'topic':topic,'ordinal':counts[topic]})
    subscriptions=[node.create_subscription(native[c.msgtype],c.topic,
        lambda raw,topic=c.topic:receive(topic,raw),100,raw=True) for c in reader.connections]
    deadline=time.monotonic()+30
    while any(pub.get_subscription_count()==0 for pub in publishers.values()):
        if time.monotonic()>deadline:raise RuntimeError('DDS discovery timeout')
        rclpy.spin_once(node,timeout_sec=.1)
    for index,(c,stamp,raw) in enumerate(reader.messages()):
        canonical=bytes(store.serialize_cdr(store.deserialize_cdr(raw,c.msgtype),c.msgtype))
        pending[c.topic]=canonical
        previous=counts[c.topic]
        started=time.monotonic()
        message=deserialize_message(bytes(raw),native[c.msgtype]) if a.mode=='roundtrip' else bytes(raw)
        prepared=time.monotonic()
        publishers[c.topic].publish(message)
        published=time.monotonic()
        while counts[c.topic]==previous:
            if time.monotonic()-started>30:raise RuntimeError('DDS receipt timeout')
            rclpy.spin_once(node,timeout_sec=.1)
        stream.write(json.dumps(dict(index=index,topic=c.topic,record_stamp_ns=stamp,
            prepare_s=prepared-started,publication_s=published-prepared,
            prepare_and_publish_s=published-started,
            offered_to_receipt_s=received_times[c.topic]-started))+'\n')
    report=dict(mode=a.mode,input=str(a.input),input_messages=reader.message_count,
                received=dict(counts),semantic_cdr_mismatches=mismatch,
                scope='Recorded real detector display messages through actual DDS; no inference, input overload or watchdog acceptance.')
stream.close();node.destroy_node();rclpy.shutdown()
rows=[json.loads(line) for line in (a.output/'timing.jsonl').read_text().splitlines()]
report['message_ms']={k:{n:float(np.quantile([r[k] for r in rows],q)*1000)
 for n,q in [('p50',.5),('p95',.95),('p99',.99)]} for k in rows[0] if k.endswith('_s')}
# Each source scan has exactly five topic messages, verified by source counts.
report['scan_ms']={k:{n:float(np.quantile([sum(r[k] for r in rows[i:i+5]) for i in range(0,len(rows),5)],q)*1000)
 for n,q in [('p50',.5),('p95',.95),('p99',.99)]} for k in ('prepare_s','publication_s','prepare_and_publish_s')}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
