#!/usr/bin/env python3
"""Segmentation metrics for native RIS-LAD masks."""
import argparse,json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

def decode(r):
    x=dict(r)
    if isinstance(x['counts'],str):x['counts']=x['counts'].encode()
    return mu.decode(x).astype(bool)

def align(mask,shape):
    if mask.shape==shape:return mask
    h,w=mask.shape;th,tw=shape
    yi=np.minimum(np.arange(th)*h//th,h-1);xi=np.minimum(np.arange(tw)*w//tw,w-1)
    return mask[np.ix_(yi,xi)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gt',required=True);ap.add_argument('--pred',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    gd=json.load(open(args.gt));pd=json.load(open(args.pred));anns={a['id']:a for a in gd['annotations']}
    ids=[r['annotation_id'] for r in pd]
    assert len(ids)==len(set(ids)) and set(ids)==set(anns),'Missing, duplicate, or unexpected annotations'
    intersections=unions=nonempty=0;ious=[]
    for r in pd:
        gt=decode(anns[r['annotation_id']]['segmentation']);pred=np.zeros(gt.shape,bool)
        for p in r.get('predictions',[]):pred|=align(decode(p['rle']),gt.shape)
        nonempty+=bool(pred.any());inter=int(np.logical_and(gt,pred).sum());union=int(np.logical_or(gt,pred).sum())
        intersections+=inter;unions+=union;ious.append(inter/union if union else 0.)
    x=np.asarray(ious);out={'protocol':'RIS-LAD native-mask evaluation','samples':len(ids),'pred_nonempty':nonempty,'mIoU':100*float(x.mean()),'oIoU':100*intersections/unions if unions else 0.,'IoU=0':100*float((x==0).mean()),'intersection_sum':intersections,'union_sum':unions}
    out.update({f'P@{t:.1f}':100*float((x>=t).mean()) for t in [.5,.6,.7,.8,.9]})
    Path(args.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
