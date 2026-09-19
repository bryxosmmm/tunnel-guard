"""Plot recorded real and explicitly synthetic local-frame corridor evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .io import iter_bag
from .visualization import corridor_edges


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',type=Path,required=True)
    parser.add_argument('--real',type=Path,required=True)
    parser.add_argument('--bag',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((args.real/'detector.json').read_text())
    row=json.loads((args.real/(args.bag.name+'.jsonl')).open().readline())
    scan=next(iter_bag(args.bag,cfg,max_frames=1))
    if scan.measurement_timestamp_ns != row['measurement_timestamp_ns']:
        raise ValueError('Real source does not match recorded measurement')
    examples=[('REAL '+args.bag.name+' / frame 0',scan.points,corridor_edges(row['geometry'],cfg))]
    for name in ['left_uphill','right_downhill','cant_transition']:
        with np.load(args.scenes/(name+'-local3d.npz')) as d:
            examples.append(('SYNTHETIC '+name,d['points'],d['corridor']))
    fig,axes=plt.subplots(len(examples),2,figsize=(14,12))
    for index,(label,points,edges) in enumerate(examples):
        selected=points[(points[:,0]>0)&(points[:,0]<65)&(np.abs(points[:,1])<4)&(points[:,2]<.5)]
        selected=selected[::max(1,len(selected)//30000)]
        for col,(a,b) in enumerate([(0,1),(0,2)]):
            ax=axes[index,col]
            ax.scatter(selected[:,a],selected[:,b],s=.3,c='.55',rasterized=True)
            if len(edges):ax.add_collection(LineCollection(edges.reshape(-1,2,3)[:,:,[a,b]],colors='#007faa',linewidths=.55))
            ax.set_xlim(0,65);ax.set_ylim((-4,4) if col==0 else (-3,4))
            ax.set_title(label+(' / plan XY' if col==0 else ' / elevation XZ'))
            ax.set_xlabel('forward x (m)');ax.set_ylabel('yz'[col]+' (m)');ax.grid(alpha=.15)
    fig.suptitle('Measured local rail frames: cyan = nominal reference contour, grey = returns\n'
                 'Plot crops are for display only. Real rails unlabelled; synthetic scenes are authored, not field evidence.')
    fig.tight_layout(rect=(0,0,1,.95));args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150);plt.close(fig)

if __name__=='__main__':main()
