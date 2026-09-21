"""Make an explicitly partial ROS bag view of one unchanged SQLite segment."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segment', type=Path)
    parser.add_argument('--source-metadata', type=Path, required=True)
    args = parser.parse_args()
    db = args.segment.resolve()
    output = db.parent / 'metadata.yaml'
    provenance = db.parent / 'subset-provenance.json'
    if output.exists() or provenance.exists():
        parser.error('Use a segment directory without generated metadata/provenance')
    original = yaml.safe_load(args.source_metadata.read_text())
    info = original['rosbag2_bagfile_information']
    paths = info['relative_file_paths']
    index = paths.index(db.name)
    files = {f['path']: f for f in info['files']}
    expected = files[db.name]
    with sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1', uri=True) as connection:
        topics = list(connection.execute('SELECT id,name,type,serialization_format FROM topics'))
        counts = dict(connection.execute('SELECT topic_id,count(*) FROM messages GROUP BY topic_id'))
        count, start, end = connection.execute('SELECT count(*),min(timestamp),max(timestamp) FROM messages').fetchone()
    if count != expected['message_count'] or start != expected['starting_time']['nanoseconds_since_epoch']:
        raise ValueError('Extracted segment does not match original metadata count/start')
    selected = copy.deepcopy(original)
    target = selected['rosbag2_bagfile_information']
    target.update(relative_file_paths=[db.name], message_count=count,
                  starting_time={'nanoseconds_since_epoch': start}, duration={'nanoseconds': end-start})
    target['files'] = [dict(path=db.name, starting_time=target['starting_time'], duration=target['duration'], message_count=count)]
    by_name = {t['topic_metadata']['name']:t for t in target['topics_with_message_count']}
    target['topics_with_message_count'] = []
    for identity,name,kind,serialization in topics:
        topic = by_name[name]
        if topic['topic_metadata']['type'] != kind or topic['topic_metadata']['serialization_format'] != serialization:
            raise ValueError('Source and segment topic schemas disagree')
        topic['message_count'] = counts.get(identity,0)
        target['topics_with_message_count'].append(topic)
    record = {'source_metadata':str(args.source_metadata),
              'source_metadata_sha256':hashlib.sha256(args.source_metadata.read_bytes()).hexdigest(),
              'original_relative_file':db.name,'original_segment_index':index,
              'preceding_record_messages':sum(files[name]['message_count'] for name in paths[:index]),
              'subset_messages':count,'source_messages':info['message_count'],
              'scope':'Unmodified real SQLite segment; generated metadata covers this segment only. Start tracker anew; not a complete recording.'}
    output.write_text(yaml.safe_dump(selected,sort_keys=False))
    provenance.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__ == '__main__':
    main()
