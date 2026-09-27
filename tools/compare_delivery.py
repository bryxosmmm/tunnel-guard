"""Summarize real paired runs, including decision and serialized display differences."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from rosbags.rosbag2 import Reader

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
runtime={'processing_s','motion_s','mounting_observation_s','deserialize_s','decode_s','ingestion_s',
         'inference_s','read_and_process_s','visualization_s'}
def rows(root):
 return [json.loads(line) for line in (root/'doubleT_obstacle.jsonl').read_text().splitlines()]
def timings(root):
 rows=[json.loads(line) for line in (root/'doubleT_obstacle-timing.jsonl').read_text().splitlines()]
 return {k:{n:float(np.quantile([r[k] for r in rows],q)*1000) for n,q in [('p50',.5),('p95',.95),('p99',.99)]}
         for k in rows[0] if k.endswith('_s')}
before,after=rows(a.before),rows(a.after)
differences=[]
for x,y in zip(before,after):
 fields=[k for k in x.keys()|y.keys() if k not in runtime and x.get(k)!=y.get(k)]
 if fields: differences.append({'frame':x['frame'],'fields':fields})
counts=Counter(); bad=Counter(); order_mismatches=0
with Reader(a.before/'doubleT_obstacle_rviz') as x, Reader(a.after/'doubleT_obstacle_rviz') as y:
 for (cx,tx,rx),(cy,ty,ry) in zip(x.messages(),y.messages()):
  if cx.topic != cy.topic or tx != ty: order_mismatches+=1
  if not cx.topic.endswith('/status'):
   counts[cx.topic]+=1
   if bytes(rx)!=bytes(ry):bad[cx.topic]+=1
 report={'before_frames':len(before),'after_frames':len(after),'ignored_runtime_fields':sorted(runtime),
         'decision_differences':differences,'message_order_or_timestamp_mismatches':order_mismatches,
         'display_messages_compared':dict(counts),'display_cdr_differences':dict(bad),
         'before_message_count':x.message_count,'after_message_count':y.message_count,
         'before_ms':timings(a.before),'after_ms':timings(a.after),
         'scope':'Actual full sequential CLI replay; exported messages, not DDS or GUI. Exact equality for every non-runtime row field and four non-status CDR topics.'}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
