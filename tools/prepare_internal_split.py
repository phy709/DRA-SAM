#!/usr/bin/env python3
"""Split a SAM3-I train JSON by original RGB image, seed 123, 90/10."""
import argparse,json,random,math
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--ann',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
d=json.load(open(args.ann)); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
def group(im):
 return im['file_name']

keys=sorted({group(im) for im in d['images']});random.Random(123).shuffle(keys);nval=math.ceil(len(keys)*0.1);vg=set(keys[:nval]);ids={im['id'] for im in d['images'] if group(im) in vg}
counts={}
for sp in ('train','val'):
 ims=[im for im in d['images'] if (im['id'] in ids)==(sp=='val')]; sel={x['id'] for x in ims}
 anns=[a for a in d['annotations'] if a['image_id'] in sel];sub={**d,'images':ims,'annotations':anns};counts[sp]=len(anns)
 (out/f'sam3i_internal_{sp}.json').write_text(json.dumps(sub,ensure_ascii=False),encoding='utf-8')
meta={'seed':123,'val_fraction':0.1,'split_unit':'original RGB image/frame','unique_files':len(keys),'train_files':len(keys)-nval,'val_files':nval,'train_records':counts['train'],'val_records':counts['val'],'source':args.ann,'official_val_used_for_rai_training':False}
(out/'split_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta))
