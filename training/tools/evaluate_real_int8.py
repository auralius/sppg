#!/usr/bin/env python3
"""Noodle2 INT8 comparison with v17 80ms guard on heldout recordings only.

Metrics are exploratory: S references are mostly algorithmic, and D events are
human-reviewed by one reviewer only; no independent second annotation.
"""
from pathlib import Path
import csv,json,subprocess,sys,collections
import numpy as np
from peaknet_quant_support import to_int8_input
from finetune_real import holdout_records
ROOT=Path(__file__).resolve().parents[1]
STARTS=(0,128,256,384,512,640,745)

def load_data():
    rows=list(csv.DictReader(open(ROOT/'data/real_256_windows_reviewed.csv')))
    auto=json.loads((ROOT/'data/auto_session.json').read_text())
    dec=json.loads((ROOT/'data/manual_review.json').read_text())
    return rows,auto,dec

def candidates(logits,start,guard=0):
    out=[]
    for c in (0,1):
        last=-1000
        for i in range(96,224):
            a=logits[c];y=int(a[i])
            if y<0 or y<int(a[i-1]) or y<=int(a[i+1]):continue
            if out and last>=0 and i-last<35:
                if y>out[-1][2]:
                    out[-1]=(start+i,c,y);last=i
                continue
            out.append((start+i,c,y));last=i
    return out

def dedup(src,guard=8):
    # Across windows starts 128, center spans don't overlap unless terminal extra
    # window is included (we exclude last 745 for exact 128-hop streaming comparison).
    raw=sorted(src)
    kept=[(p,c,v) for p,c,v in raw if c==0 or not any(x==0 and abs(sp-p)<=guard for sp,x,_ in raw)]
    return raw,kept

def report(pred,truth,rec_name):
    # Only scores [96, 863] (six windows) matching streaming firmware decode.
    events=[p for p,c,_ in pred if c==1 and 96<=p<864]
    rr=truth['decisions'][rec_name]['beats']
    beats=next(x['beats'] for x in auto['recordings'] if x['file']==rec_name)
    # True D is a manually confirmed DISTINCT secondary maximum.
    truths=[b['d_confirmed'] for base in beats if (b:=rr.get(base['beat_id'],{})).get('d_status')=='distinct'
            and b.get('d_confirmed') is not None and 96<=b['d_confirmed']<864]
    matched=set();tp=0;fp=0;ignored=0
    for p in events:
        j=min(range(len(truths)),key=lambda j:abs(p-truths[j])) if truths else None
        if j is not None and abs(p-truths[j])<=10 and j not in matched:
            tp+=1;matched.add(j);continue
        owning=next((base for base in beats if base['s']<=p<(base['next_s'] if base['next_s'] is not None else 1001)),None)
        status=rr.get(owning['beat_id'],{}).get('d_status') if owning else None
        if status in ('distinct','shoulder','absent'):fp+=1
        else:ignored+=1
    return {'record':rec_name,'tp':tp,'fp':fp,'fn':len(truths)-tp,'ignore':ignored,'n_truth':len(truths),'d_events':len(events)}

if __name__=='__main__':
    rows,auto,truth=load_data()
    held=holdout_records([rec['file'] for rec in auto['recordings']]);sel=[]
    for rec in held:
        rr=[row for row in rows if row['record_id']+'.CSV'==rec and int(row['start_sample']) in STARTS[:6]]
        rr.sort(key=lambda r:int(r['start_sample']))
        sel.extend(rr)
    from peaknet_quant_support import to_int8_input
    reports=[]
    for model in ('baseline_lambda050','real_finetuned'):
        manifest=json.loads((ROOT/'models'/model/'peaknet_manifest.json').read_text())
        scale=manifest['input_scale']; inp=np.array([[float(r[f'x_{i:03d}']) for i in range(256)] for r in sel],np.float32)
        q=to_int8_input(inp,scale); in_file=ROOT/'results'/f'{model}_input.int8';out_file=ROOT/'results'/f'{model}_logits.int8'
        q.tofile(in_file)
        exec_name='batch_baseline' if model=='baseline_lambda050' else 'batch_finetuned'
        subprocess.run([str(ROOT/'cpp/build'/exec_name),str(in_file),str(out_file)],check=True,capture_output=True)
        pred=np.fromfile(out_file,np.int8).reshape(-1,2,256)
        pairs=[]
        for rec in held:
            rs=[(r,p) for r,p in zip(sel,pred) if r['record_id']+'.CSV'==rec]
            all_c=[]
            for row,log in rs:all_c+=candidates(log,int(row['start_sample']))
            raw,kept=dedup(all_c)
            for stage,ev in [('raw',raw),('guard80ms',kept)]:
                item=report(ev,truth,rec);item.update(model=model,decoder=stage)
                item['cross_overlap_d']=sum(c==1 and any(c2==0 and abs(pos-p)<=8 for pos,c2,_ in ev) for p,c,_ in ev)
                reports.append(item)
    with (ROOT/'results/heldout_int8_events.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(reports[0]));wr.writeheader();wr.writerows(reports)
    summary=[]
    for (model,decoder),rs in __import__('itertools').groupby(sorted(reports,key=lambda a:(a['model'],a['decoder'])),key=lambda a:(a['model'],a['decoder'])):
        rs=list(rs); tp=sum(r['tp'] for r in rs);fp=sum(r['fp'] for r in rs);fn=sum(r['fn'] for r in rs)
        pre=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
        summary.append(dict(model=model,decoder=decoder,TP=tp,FP=fp,FN=fn,precision=round(pre,4),recall=round(rec,4),F1=round(2*pre*rec/(pre+rec),4) if pre+rec else 0,ignored=sum(r['ignore'] for r in rs),overlap=sum(r['cross_overlap_d'] for r in rs)))
    with (ROOT/'results/heldout_summary.json').open('w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
