#!/usr/bin/env python3
"""Experimental within-person MAX30102 adaptation of v15 lambda=.50 PeakNet.

Takes reviewed 100 Hz REAL data and the exact old v15 negative-aware
synthetic generator. Applies loss masks, never converts unknown-D to absence.
Split is by full recording ID, never by overlapping windows. This is not a
clinical or multi-subject validation.
"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from negative_training_data import make_negative_windows
from peaknet_quant_support import PeakNet, save_artifacts, parse_testvector, integer_forward, export_layers

ROOT=Path(__file__).resolve().parents[1]
FS=100; W=256

def data_from_csv(path):
    rows=list(csv.DictReader(open(path,newline='')))
    x=np.array([[float(row[f'x_{i:03d}']) for i in range(W)] for row in rows],dtype=np.float32)
    y=np.array([[[float(row[f'{c}_{i:03d}']) for i in range(W)] for c in ('s','d')] for row in rows],dtype=np.float32)
    mask=np.array([[float(row[c]) for c in ('s_supervised','d_supervised')] for row in rows],dtype=np.float32)
    rec=np.array([row['record_id']+'.CSV' for row in rows])
    return x,y,mask,rec,rows

def holdout_records(recs,n=5):
    # Fixed before training, independent of model's outputs.
    return sorted(set(recs),key=lambda f:hashlib.sha256(('sppg-real-holdout-v1:'+f).encode()).hexdigest())[:n]

def masked_parts(o,y,mask,guard):
    # Average each channel across 256 positions, then average over supervised
    # sample-channel pairs. Unknown classes contribute ZERO to numerator/denominator.
    by_ch=F.binary_cross_entropy_with_logits(o,y,weight=1+9*y,reduction='none').mean(dim=2)
    bce=(by_ch*mask).sum()/mask.sum().clamp_min(1.)
    ps=torch.sigmoid(o[:,0:1,:]);pd=torch.sigmoid(o[:,1:2,:])
    near=F.max_pool1d(ps,2*guard+1,stride=1,padding=guard)
    per_sample=(pd*near*(1-y[:,1:2,:])).mean(dim=(1,2))
    dmask=mask[:,1]
    exclusion=(per_sample*dmask).sum()/dmask.sum().clamp_min(1.)
    return bce,exclusion

def do_eval(model,x,y,mask,batch=48,guard=8):
    if not len(x):return {'n':0}
    model.eval(); bs=0.; ov=0.; ch=np.zeros(2); denom=np.zeros(2); n=0
    with torch.no_grad():
        for i in range(0,len(x),batch):
            xx=torch.from_numpy(x[i:i+batch,None,:]); yy=torch.from_numpy(y[i:i+batch]); mm=torch.from_numpy(mask[i:i+batch]);o=model(xx)
            b,e=masked_parts(o,yy,mm,guard)
            cnt=int(len(xx));bs+=float(b)*cnt;ov+=float(e)*cnt;n+=cnt
            loss_ch=F.binary_cross_entropy_with_logits(o,yy,weight=1+9*yy,reduction='none').mean(dim=2)
            ch+=(loss_ch*mm).sum(dim=0).numpy();denom+=mm.sum(dim=0).numpy()
    return {'windows':n,'weighted_bce':bs/n,'overlap':ov/n,'S_masked_BCE':ch[0]/max(denom[0],1.),'D_masked_BCE':ch[1]/max(denom[1],1.),'S_supervised_windows':int(denom[0]),'D_supervised_windows':int(denom[1])}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--real',type=Path,default=ROOT/'data/real_256_windows_reviewed.csv')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/baseline_lambda050/peaknet_negative_qat.pt')
    p.add_argument('--output',type=Path,default=ROOT/'models/real_finetuned')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--overlap-lambda',type=float,default=.5)
    p.add_argument('--guard',type=int,default=8)
    p.add_argument('--synthetic-records',type=int,default=120)
    p.add_argument('--real-repeat',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261116)
    args=p.parse_args()
    if args.epochs<1:raise ValueError('epochs must be >=1')
    args.output.mkdir(parents=True,exist_ok=True)
    args.real_data_dir = None  # export helper optional real-vector source; explicit none
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    x,y,m,rec,rows=data_from_csv(args.real)
    held=holdout_records(rec)
    is_hold=np.isin(rec,held)
    rx,ry,rm=x[~is_hold],y[~is_hold],m[~is_hold]
    vx,vy,vm=x[is_hold],y[is_hold],m[is_hold]
    sx,sy,_=make_negative_windows(n_records=args.synthetic_records,dur=12,seed=1007)
    tx,ty,_=make_negative_windows(n_records=40,dur=12,seed=92926)
    sm=np.ones((len(sx),2),dtype=np.float32);tm=np.ones((len(tx),2),dtype=np.float32)
    train_x=np.concatenate([sx,np.tile(rx,(args.real_repeat,1))],axis=0)
    train_y=np.concatenate([sy,np.tile(ry,(args.real_repeat,1,1))],axis=0)
    train_m=np.concatenate([sm,np.tile(rm,(args.real_repeat,1))],axis=0)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=PeakNet();model.load_state_dict(ck['state_dict']);model.scales=ck['activation_scales'];model.qat=True
    real_initial=do_eval(model,vx,vy,vm,guard=args.guard)
    synth_initial=do_eval(model,tx,ty,tm,guard=args.guard)
    print('REAL full-recording heldout=',held,flush=True)
    print('Data: synthetic=',len(sx),'real train=',len(rx),'real holdout=',len(vx),'total repeated=',len(train_x),flush=True)
    print('BASELINE real',real_initial,'synthetic',synth_initial,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1.e-4)
    hist=[]
    for epoch in range(args.epochs):
        model.train();gen=torch.Generator().manual_seed(args.seed+epoch)
        order=torch.randperm(len(train_x),generator=gen);trainloss=0.;t0=time.monotonic()
        for i in range(0,len(order),32):
            inds=order[i:i+32].numpy()
            xx=torch.from_numpy(train_x[inds,None,:]);yy=torch.from_numpy(train_y[inds]);mm=torch.from_numpy(train_m[inds])
            xx=xx+torch.randn_like(xx)*.015
            out=model(xx);b,e=masked_parts(out,yy,mm,args.guard)
            loss=b+args.overlap_lambda*e
            opt.zero_grad();loss.backward();opt.step();trainloss+=float(loss.detach())*len(inds)
        re=do_eval(model,vx,vy,vm,guard=args.guard)
        se=do_eval(model,tx,ty,tm,guard=args.guard)
        hist.append({'epoch':epoch+1,'train_loss':trainloss/len(train_x),'heldout_real':re,'synthetic_val':se})
        print('EPOCH',epoch+1,'train',round(trainloss/len(train_x),5),'real D BCE',round(re['D_masked_BCE'],5),'real S BCE',round(re['S_masked_BCE'],5),'synth BCE',round(se['weighted_bce'],5),'sec',round(time.monotonic()-t0,1),flush=True)
    # Keep ORIGINAL lambda=.5 calibration scales fixed to isolate the effect
    # of new annotations; scales themselves can be retested later as an ablation.
    scales=list(model.scales)
    stats=json.loads((ROOT/'models/baseline_lambda050/peaknet_manifest.json').read_text())['calibration']
    orig_vec,_=parse_testvector(ROOT/'tools/peaknet_ppg_testvector.h')
    metrics={'experiment':'lambda050_real_sensor_masked_finetuning_v1',
      'source_review_sha256':hashlib.sha256((ROOT/'data/manual_review.json').read_bytes()).hexdigest(),
      'heldout_selection':'SHA256(sppg-real-holdout-v1:<filename>) first 5',
      'heldout_recordings':held,'train_recordings':sorted(set(rec)-set(held)),
      'synthetic_windows_train':len(sx),'synthetic_windows_val':len(tx),
      'real_windows_train':len(rx),'real_windows_holdout':len(vx),'real_repeat':args.real_repeat,
      'real_training_S_supervision':int(rm[:,0].sum()),'real_training_D_supervision':int(rm[:,1].sum()),
      'epochs':args.epochs,'lr':args.learning_rate,'lambda':args.overlap_lambda,'guard':args.guard,
      'baseline_real_holdout':real_initial,'baseline_synthetic_val':synth_initial,
      'final_real_holdout':re,'final_synthetic_val':se,'history':hist,
      'caution':'S pseudo-labeled except explicitly reviewed; D human reviewed but not second-annotator validated; heldout is within-person by recording, not independent subjects.'}
    man=save_artifacts(args,model,scales,stats,metrics,orig_vec)
    man.update(source='v15 lambda=0.5 QAT checkpoint fine-tuned on old v15 negative-aware synthetic + manually reviewed MAX30102', status='EXPERIMENTAL within-person fine-tuning; validate event detection before deployment', data_provenance='manual_review.json + supervision masks from real_256_windows_reviewed.csv; grouped recording split')
    (args.output/'peaknet_manifest.json').write_text(json.dumps(man,indent=2)+'\n')
    torch.save({'state_dict':model.state_dict(),'activation_scales':scales,'metrics':metrics},args.output/'peaknet_negative_qat.pt')
    (args.output/'training_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('DONE parameter_bytes',man['parameter_bytes'],flush=True)
if __name__=='__main__':main()
