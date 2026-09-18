"""Inspect real ROS2 SQLite acquisition headers without loading full point arrays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import struct
import yaml
from rosbags.typesys import Stores, get_typestore
from .run import digest, write_json


def read_header(prefix):
    if len(prefix) < 16 or prefix[:2] not in (b'\x00\x00', b'\x00\x01'):
        raise ValueError('Expected plain CDR PointCloud2 serialization')
    endian = '<' if prefix[1] == 1 else '>'
    seconds, nanoseconds, length = struct.unpack_from(endian+'iII', prefix, 4)
    if nanoseconds >= 1000000000 or length < 1 or 16+length > len(prefix) or prefix[15+length] != 0:
        raise ValueError('Unsupported or truncated PointCloud2 header')
    return seconds*1000000000+nanoseconds, prefix[16:15+length].decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():parser.error('Use a new output path')
    info=yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    expected={entry['path']:entry for entry in info['files']}
    store=get_typestore(Stores.ROS2_HUMBLE)
    seen=set(); previous={}; segments=[]; gaps=[]; repeats=[]; backwards=[]; total=0
    for name in info['relative_file_paths']:
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe metadata path')
        db=(args.bag/rel).resolve()
        with sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True) as connection:
            all_count,start,end=connection.execute('SELECT count(*),min(timestamp),max(timestamp) FROM messages').fetchone()
            if all_count != expected[name]['message_count'] or start != expected[name]['starting_time']['nanoseconds_since_epoch']:
                raise ValueError(f'Metadata mismatch: {name}')
            topics=list(connection.execute("SELECT id,name,type,serialization_format FROM topics WHERE type='sensor_msgs/msg/PointCloud2'"))
            count=0; first=None; last=None
            for topic_id,topic,kind,serialization in topics:
                if serialization!='cdr':raise ValueError('Unsupported serialization')
                first_payload=connection.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id LIMIT 1',(topic_id,)).fetchone()
                if first_payload is None:continue
                message=store.deserialize_cdr(first_payload[0],kind)
                expected_header=(message.header.stamp.sec*1000000000+message.header.stamp.nanosec,message.header.frame_id)
                if read_header(first_payload[0][:4096]) != expected_header:
                    raise ValueError('CDR prefix and standard deserializer disagree')
                # Incremental blob reads avoid materializing 8 MB to inspect a header.
                for rowid,record_ns in connection.execute('SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id',(topic_id,)):
                    with connection.blobopen('messages','data',rowid,readonly=True) as blob:
                        stamp,frame=read_header(blob.read(4096))
                    identity=(topic,frame,stamp); clock=(topic,frame)
                    event={'segment':name,'segment_cloud_index':count,'global_cloud_index':total,'measurement_ns':stamp,'record_ns':record_ns}
                    if identity in seen:repeats.append(event)
                    seen.add(identity)
                    if clock in previous:
                        delta=stamp-previous[clock][0]
                        if delta<0:backwards.append(event | {'delta_s':delta/1e9})
                        if delta>500000000:gaps.append(event | {'measurement_gap_s':delta/1e9,'record_gap_s':(record_ns-previous[clock][1])/1e9})
                    previous[clock]=(stamp,record_ns)
                    first=stamp if first is None else first;last=stamp;count+=1;total+=1
            segments.append({'file':name,'messages':all_count,'clouds':count,'measurement_first_ns':first,'measurement_last_ns':last,'record_first_ns':start,'record_last_ns':end})
    report={'bag':str(args.bag),'metadata_sha256':digest(args.bag/'metadata.yaml'),'source_sha256':digest(Path(__file__)),'clouds':total,'segments':segments,'duplicate_acquisitions':repeats,'backward_acquisitions':backwards,'gaps_over_half_second':gaps,'scope':'All PointCloud2 headers; first payload per segment checked with standard ROS deserializer. No full point decoding, detector inference, semantic labels or calibration validation implied.'}
    write_json(args.output,report)
    print(json.dumps({'clouds':total,'segments':len(segments),'duplicates':len(repeats),'backwards':len(backwards),'gaps_over_half_second':len(gaps)}))

if __name__=='__main__':
    main()
