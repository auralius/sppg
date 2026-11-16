#!/usr/bin/env python3
"""PeakNet-PPG: preserve FP32 architecture, calibrate, QAT fine-tune, export
integer-exact INT8 artifacts for Noodle2 v0.1.18 (CHW/1xK Conv2D).

Produces a concrete *synthetic-QAT starter*, not validated clinical peak detection.
Real SPPG PPG CSVs may be used for scale CALIBRATION only, never unlabelled training.
"""
import argparse, json, math, re, struct, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from numpy.lib.stride_tricks import sliding_window_view
from ppg_synth import make_windows,real_windows

CFG=((1,8,15,True),(8,16,15,True),(16,16,15,True),
     (16,16,15,True),(16,16,1,True),(16,2,1,False))
NAMES=tuple(f'conv{i+1}' for i in range(len(CFG)))

class RoundAwaySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return torch.sign(x)*torch.floor(torch.abs(x)+.5)
    @staticmethod
    def backward(ctx,d):return d

def fakequant(x,s,low=-128,high=127):
    q=(RoundAwaySTE.apply(x/s).clamp(low,high))*s
    return x+(q-x).detach()

def w_scale(w):
    return max(float(w.detach().abs().max().item())/127.,1.e-8)

class PeakNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(*sum(([nn.Conv1d(ci,co,k,padding=k//2),nn.ReLU()] if act else [nn.Conv1d(ci,co,k,padding=k//2)] for ci,co,k,act in CFG),[]))
        self.qat=False
        self.scales=None
    def convs(self):return [m for m in self.net if isinstance(m,nn.Conv1d)]
    def forward(self,x,collect=False):
        if self.qat:
            x=fakequant(x,self.scales[0])
        outputs=[]
        for i,(layer,(_,_,_,act)) in enumerate(zip(self.convs(),CFG)):
            w=fakequant(layer.weight,w_scale(layer.weight),-127,127) if self.qat else layer.weight
            x=F.conv1d(x,w,layer.bias,stride=1,padding=layer.padding[0])
            if act:x=F.relu(x)
            if self.qat:
                x=fakequant(x,self.scales[i+1],0 if act else -128,127)
            if collect:outputs.append(x.detach())
        return (x,outputs) if collect else x

def round_away(x):
    v=np.asarray(x,dtype=np.float64)
    return np.copysign(np.floor(np.abs(v)+0.5),v)

def quant_q31(ratio):
    frac,exp=math.frexp(float(ratio))
    mult=int(math.floor(frac*2**31+.5))
    if mult==2**31:mult//=2;exp+=1
    if not -30<=exp<=30:raise ValueError(f'out-of-range Q31 shift {exp}')
    return mult,exp

def requantize(acc,mult,shift):
    product=np.asarray(acc,dtype=np.int64)*np.int64(mult)
    r=31-shift
    mag=(np.abs(product)+(np.int64(1)<<(r-1)))>>r
    return np.clip(np.where(product<0,-mag,mag),-128,127).astype(np.int8)

def integer_layer(x,l,act):
    w=l['weight'].astype(np.int64) # [out,in,k]
    pad=w.shape[-1]//2
    xp=np.pad(x.astype(np.int64),((0,0),(pad,pad)))
    windows=sliding_window_view(xp,w.shape[-1],axis=1) # [in,length,k]
    accum=np.tensordot(w,windows,axes=([1,2],[0,2])).astype(np.int64)
    accum+=l['bias'].astype(np.int64)[:,None]
    y=requantize(accum,l['mult'],l['shift'])
    return np.maximum(y,0).astype(np.int8) if act else y

def integer_forward(x,layers):
    for i,l in enumerate(layers):x=integer_layer(x,l,CFG[i][-1])
    return x

def to_int8_input(x,scale):
    return np.clip(round_away(np.asarray(x,dtype=np.float64)/scale),-128,127).astype(np.int8)

def calibrate(model,cal_data,percentile=99.97,margin=1.2):
    model.qat=False;model.eval()
    sample=cal_data[:min(len(cal_data),96)]
    values=[[] for _ in range(7)];values[0].append(np.abs(sample).reshape(-1))
    with torch.no_grad():
        for start in range(0,len(sample),16):
            _,outs=model(torch.from_numpy(sample[start:start+16,None,:]),collect=True)
            for i,a in enumerate(outs):values[i+1].append(a.cpu().numpy().reshape(-1))
    scales=[];stats=[]
    for i,arrays in enumerate(values):
        v=np.concatenate(arrays)
        # Quantization remains symmetric INT8; hidden ReLU uses nonnegative 0..127.
        limit=float(np.percentile(np.abs(v),percentile))
        peak=float(np.max(np.abs(v)))
        # Protect short samples from unrepresentative calibration maxima.
        scale=max(limit*margin,peak*1.02,1.e-4)/127.
        scales.append(scale)
        stats.append(dict(name='input' if i==0 else NAMES[i-1],p_abs=float(limit),
                          max_abs=peak,scale=scale))
    return scales,stats

def export_layers(model,scales):
    layers=[]
    for i,layer in enumerate(model.convs()):
        wi=w_scale(layer.weight)
        w=np.clip(round_away(layer.weight.detach().cpu().numpy()/wi),-127,127).astype(np.int8)
        bias_float=layer.bias.detach().cpu().numpy()
        b64=round_away(bias_float/(scales[i]*wi))
        if not np.all(np.abs(b64)<=2**31-1):raise ValueError('bias int32 overflow')
        b=b64.astype(np.int32)
        mult,shift=quant_q31(scales[i]*wi/scales[i+1])
        layers.append(dict(weight=w,bias=b,mult=mult,shift=shift,
                           weight_scale=wi,input_scale=scales[i],output_scale=scales[i+1]))
    return layers

def c_bytes(data,n=20):
    return '\n'.join('    '+', '.join(str(v) for v in data[i:i+n])+',' for i in range(0,len(data),n))

def c_int8s(arr,n=22):
    d=[str(int(v)) for v in np.asarray(arr).reshape(-1)]
    return '\n'.join('    '+', '.join(d[i:i+n])+',' for i in range(0,len(d),n))

def export_header(out,layers):
    buf=bytearray();offsets=[]
    for l in layers:
        offsets.append(len(buf));buf.extend(l['weight'].tobytes(order='C'))
        buf.extend(l['bias'].astype('<i4').tobytes())
    lines=['#pragma once','// Generated by tools/peaknet_qat_export.py; do not edit.','// INT8 weights followed by little-endian INT32 biases, per layer.',
           '#include <Noodle2.h>','#include <stdint.h>','namespace peaknet_ppg {',
           'static const uint8_t parameter_data[] = {',c_bytes(buf),'};',
           'static const uint32_t parameter_bytes = sizeof(parameter_data);',
           'struct Model {','    noodle2::NoodleBuffer storage;']
    for n in NAMES:lines.append('    noodle2::NoodleParams '+n+';')
    lines.append('    Model() :')
    for i,(n,l) in enumerate(zip(NAMES,layers)):
        wc=l['weight'].size
        lines.append('        '+n+'(0u, '+str(wc)+'u, '+str(wc)+'u, '+str(l['bias'].size)+'u, '+
                     'noodle2::Quant('+str(l['mult'])+'L, '+str(l['shift'])+'))'+
                     (',' if i<len(NAMES)-1 else ' {}'))
    lines.extend(['    noodle2::Status bind_flash() {',
                  '        noodle2::Status s=storage.bind_flash(parameter_data,sizeof(parameter_data));',
                  '        if(s != noodle2::OK)return s;'])
    for n,off in zip(NAMES,offsets):
        lines.extend(['        s='+n+'.bind(storage,'+str(off)+'u);',
                      '        if(s != noodle2::OK)return s;'])
    lines.extend(['        return noodle2::OK;','    }','};',
                  'inline noodle2::Status build_graph(noodle2::NoodleGraph& g, const Model& m) {',
                  '    auto x=g.input(1,1,256);'])
    for i,n in enumerate(NAMES):
        lines.append('    x=g.conv2d(x,m.'+n+',noodle2::Conv2D::same(1,'+str(CFG[i][2])+'),'+
                     ('noodle2::RELU' if CFG[i][-1] else 'noodle2::LOGITS')+');')
    lines.extend(['    return g.compile();','}',
                  'static const float input_scale='+format(layers[0]['input_scale'],'.10g')+'f;',
                  'static const float output_scale='+format(layers[-1]['output_scale'],'.10g')+'f;',
                  '} // namespace peaknet_ppg',''])
    (out/'peaknet_model_flash.h').write_text('\n'.join(lines))
    return len(buf)

def parse_testvector(header):
    src=Path(header).read_text()
    a=re.search(r'const float test_in\[256\]\s*=\s*\{([^}]+)\}',src,re.S)
    b=re.search(r'const float test_out_sigmoid\[512\]\s*=\s*\{([^}]+)\}',src,re.S)
    if not a or not b:raise ValueError('test vector header format missing')
    get=lambda s:np.array([float(v.strip().rstrip('f')) for v in s.split(',') if v.strip()],dtype=np.float32)
    return get(a.group(1)),get(b.group(1)).reshape(2,256)

def checkpoint_vector_test(model,header):
    x,sig=parse_testvector(header)
    model.qat=False;model.eval()
    with torch.no_grad():
        o=torch.sigmoid(model(torch.from_numpy(x[None,None,:]))).cpu().numpy()[0]
    err=float(np.max(np.abs(o-sig)))
    print('FP32 checkpoint vs notebook test vector max sigmoid difference:',err)
    if err>2.e-5:raise AssertionError('Incorrect architecture or FP32 checkpoint')
    return x,err

def save_artifacts(args,model,scales,stats,metrics,vector):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    layers=export_layers(model,scales)
    test_x=[vector];names=['notebook_fp32_golden_input']
    real=real_windows(args.real_data_dir,limit=12) if args.real_data_dir else np.empty((0,256),np.float32)
    for i in range(min(2,len(real))):test_x.append(real[i]);names.append('real_SPPG_unlabelled_'+str(i))
    synthetic,_=make_windows(n_records=1,dur=10,seed=999)
    for i in range(min(2,len(synthetic))):test_x.append(synthetic[i]);names.append('synthetic_'+str(i))
    qinputs=np.stack([to_int8_input(x,scales[0]) for x in test_x])
    qouts=np.stack([integer_forward(x[None,:],layers) for x in qinputs])
    # Verify quantized activation-level results are reproducible in Python.
    for j in range(len(qinputs)):
        assert np.array_equal(qouts[j],integer_forward(qinputs[j][None,:],layers))
    flash_bytes=export_header(out,layers)
    lines=['#pragma once','// Golden inputs and integer-exact INT8 EXPECTED logits.','// Shape [COUNT][1][256] and [COUNT][2][256].',
           '#include <stdint.h>','namespace peaknet_golden {','static const int COUNT='+str(len(qinputs))+';',
           'static const int8_t input[COUNT][256] = {']
    for v in qinputs:lines.extend(['{',c_int8s(v),'},'])
    lines.extend(['};','static const int8_t output[COUNT][512] = {'])
    for v in qouts:lines.extend(['{',c_int8s(v),'},'])
    lines.extend(['};','} // namespace peaknet_golden',''])
    (out/'peaknet_golden.h').write_text('\n'.join(lines))
    npzd={}
    for i,l in enumerate(layers):
        for field in ('weight','bias','mult','shift'):
            npzd[f'layer{i}_{field}']=l[field]
    npzd.update(inputs=qinputs,outputs=qouts)
    np.savez_compressed(out/'peaknet_quantized.npz',**npzd)
    man=dict(source='peaknet_ppg.pt from user and ppg_peaknet notebook',
      status='synthetic-QAT starter; NOT validated on annotated real pulse peaks',
      model='six Conv1D, implemented as 1xK Conv2D, CHW',
      input_shape=[1,1,256],output_shape=[2,1,256],input_preprocess='-raw IR; z score within each 256-sample window',
      output_semantics='raw INT8 logits for systolic,diastolic; NO SIGMOID',
      input_scale=scales[0],output_scale=scales[-1],
      calibration=stats,metrics=metrics,parameter_bytes=flash_bytes,
      activation_estimate='measure actual arena_bytes() using C++ host test',
      golden_names=names,
      layers=[dict(name=NAMES[i],weight_shape=list(l['weight'].shape),
                   weights=int(l['weight'].size),outputs=int(l['bias'].size),
                   multiplier=l['mult'],shift=l['shift'],
                   weight_scale=l['weight_scale'],input_scale=l['input_scale'],output_scale=l['output_scale'])
              for i,l in enumerate(layers)])
    (out/'peaknet_manifest.json').write_text(json.dumps(man,indent=2)+'\n')
    return man

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent.parent
    ap.add_argument('--checkpoint',type=Path,default=root/'assets/peaknet_ppg.pt')
    ap.add_argument('--testvector',type=Path,default=root/'assets/peaknet_ppg_testvector.h')
    ap.add_argument('--real-data-dir',type=Path,default=None,
                    help='Optional unlabeled MAX30102 CSVs for activation calibration only')
    ap.add_argument('--output',type=Path,default=root/'include')
    ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--train-records',type=int,default=10)
    ap.add_argument('--val-records',type=int,default=3)
    ap.add_argument('--record-seconds',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1.e-4)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=214)
    args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=PeakNet()
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(weights)
    orig_test,checkpoint_error=checkpoint_vector_test(model,args.testvector)
    tr_x,tr_y=make_windows(n_records=args.train_records,dur=args.record_seconds,seed=args.seed)
    va_x,va_y=make_windows(n_records=args.val_records,dur=args.record_seconds,seed=args.seed+1)
    real=real_windows(args.real_data_dir,limit=96) if args.real_data_dir else np.empty((0,256),np.float32)
    calib=np.concatenate([tr_x[:48],real],axis=0)
    scales,stats=calibrate(model,calib)
    model.scales=scales
    model.eval();model.qat=False
    def loss_on(x,y):
        total=0.;denom=0
        with torch.no_grad():
            for j in range(0,len(x),32):
                xb=torch.from_numpy(x[j:j+32,None,:]);yb=torch.from_numpy(y[j:j+32])
                loss=F.binary_cross_entropy_with_logits(model(xb),yb,weight=1+9.*yb)
                total+=float(loss)*len(xb);denom+=len(xb)
        return total/max(denom,1)
    fp32_before=loss_on(va_x,va_y)
    model.qat=True
    qat_before=loss_on(va_x,va_y)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1.e-4)
    train_hist=[]
    for ep in range(args.epochs):
        model.train();order=torch.randperm(len(tr_x));running=0.;count=0
        start=time.monotonic()
        for a in range(0,len(tr_x),args.batch_size):
            idx=order[a:a+args.batch_size].numpy()
            xb=torch.from_numpy(tr_x[idx,None,:]);yb=torch.from_numpy(tr_y[idx])
            xb=xb+torch.randn_like(xb)*0.03
            y=model(xb)
            loss=F.binary_cross_entropy_with_logits(y,yb,weight=1+9.*yb)
            opt.zero_grad();loss.backward();opt.step()
            running+=float(loss.detach())*len(idx);count+=len(idx)
        model.eval(); val=loss_on(va_x,va_y)
        train_hist.append(dict(epoch=ep+1,train_loss=running/count,val_loss=val))
        print('QAT epoch',ep+1,'train',round(running/count,6),'val',round(val,6),
              'elapsed',round(time.monotonic()-start,1),'s',flush=True)
    # do not change QAT-frozen scales after training; they were calibrated from
    # synthetic+unlabelled real examples and trained with those same scales.
    model.eval()
    qat_after=loss_on(va_x,va_y)
    metrics=dict(fp32_checkpoint_testvector_max_error=checkpoint_error,
                 fp32_synthetic_val_loss=fp32_before,
                 qat_initial_synthetic_val_loss=qat_before,
                 qat_final_synthetic_val_loss=qat_after,
                 synthetic_train_windows=len(tr_x),synthetic_validation_windows=len(va_x),
                 real_unlabelled_calibration_windows=len(real),qat_epochs=args.epochs,
                 training_history=train_hist)
    desc=save_artifacts(args,model,scales,stats,metrics,orig_test)
    torch.save({'state_dict':model.state_dict(),'activation_scales':scales,'source_checkpoint':args.checkpoint.name,
                'metrics':metrics,'synthetic_train':True},args.output/'peaknet_qat_checkpoint.pt')
    print('EXPORTED INT8 flash bytes=',desc['parameter_bytes'],'golden vectors=',len(desc['golden_names']),
          ' QAT validation loss=',round(qat_after,5))
    print('NEXT: make test (integer-exact against Noodle2 C++ graph)')

# QAT/export helpers. Use train_negative_aware.py as the supported entry point.
