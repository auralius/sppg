"""Synthetic PPG generator, adapted from the user-supplied ppg_peaknet.ipynb.

CAUTION: diastolic targets are GENERATOR centers, not adjudicated real landmarks.
Signal orientation and input normalization match the notebook.
"""
import numpy as np
from scipy import signal
FS=100
WIN=256
P_DEFAULT=np.array([
 [0.724,0.168,0.077, 0.345,0.362,0.096, 0.947,0.5,0.316],
 [0.793,0.164,0.075, 0.319,0.353,0.104, 0.927,0.5,0.342],
 [0.621,0.162,0.075, 0.392,0.332,0.123, 1.006,0.5,0.372],
 [0.683,0.155,0.071, 0.407,0.328,0.116, 0.986,0.5,0.369]
], dtype=np.float64)
NOISE0_DEFAULT=11.5

def wg(phi,a,c,w):
    d=((phi-c+0.5)%1.)-0.5
    return a*np.exp(-0.5*(d/w)**2)

def template(phi,p):
    return sum(wg(phi,*p[3*k:3*k+3]) for k in range(3))

def synth_record(P=P_DEFAULT,noise0=NOISE0_DEFAULT,dur=20,rng=None,ac0=1000.,hr=None,clean=False):
    rng=rng or np.random.default_rng()
    n=int(dur*FS); t=np.arange(n)/FS; x=np.zeros(n)
    hr=hr or rng.uniform(45,120); rr0=60/hr
    resp_f=rng.uniform(.15,.35);resp_ph=rng.uniform(0,2*np.pi);resp_a=rng.uniform(0,.12)
    T=-rng.uniform(0,rr0);sysI=[];diaI=[];shoulder=[];pf=np.linspace(0,1,4001)
    while T<dur:
        rr=rr0*(1+0.04*np.sin(2*np.pi*resp_f*T+resp_ph)+rng.normal(0,.02))
        p=P[rng.integers(len(P))].copy()
        p[0::3]*=np.exp(rng.normal(0,.10,3))
        p[2::3]*=np.exp(rng.normal(0,.10,3))
        p[1::3]+=rng.normal(0,.008,3)
        b0=template(pf,p)-template(np.array([0.]),p)[0]
        ac=ac0*np.exp(rng.normal(0,.15))*(1+resp_a*np.sin(2*np.pi*resp_f*T+resp_ph+1))
        i0=int(np.ceil(T*FS));i1=int(np.ceil((T+rr)*FS))
        idx=np.arange(max(i0,0),min(i1,n))
        if len(idx):
            phi=(idx/FS-T)/rr
            x[idx]=ac*np.interp(phi,pf,b0)/b0.max()
        w1=p[2]; m=(pf>p[1]-2*w1)&(pf<p[1]+2*w1)
        ps=pf[m][np.argmax(b0[m])]
        sysI.append((T+ps*rr)*FS);diaI.append((T+p[4]*rr)*FS)
        d=np.diff(b0);region=(pf[1:-1]>ps+.05)&(pf[1:-1]<.8)
        shoulder.append(not np.any(region&(d[:-1]>0)&(d[1:]<=0)))
        T+=rr
    ok=lambda a:np.array([int(round(v)) for v in a if 0<=round(v)<n])
    if clean:return x,ok(sysI),ok(diaI),np.array(shoulder)
    for _ in range(rng.integers(1,4)):
        x+=rng.uniform(.05,.8)*ac0*np.sin(2*np.pi*rng.uniform(.03,.3)*t+rng.uniform(0,6.28))
    x+=rng.normal(0,noise0*np.exp(rng.uniform(np.log(.5),np.log(3))),n)
    for _ in range(rng.poisson(1.0)):
        if n<160:break
        s=int(rng.integers(0,n-150));L=int(rng.integers(30,150))
        sos=signal.butter(2,[.5,5],btype='bandpass',fs=FS,output='sos')
        art=signal.sosfilt(sos,rng.normal(0,1,L+100))[100:]
        x[s:s+L]+=art/max(art.std(),1.e-6)*ac0*rng.uniform(.5,2)*np.hanning(L)
    return x,ok(sysI),ok(diaI),np.array(shoulder)

def make_windows(n_records=12,dur=20,stride=64,seed=1):
    rng=np.random.default_rng(seed);X=[];Y=[]
    for _ in range(n_records):
        x,s,d,_=synth_record(dur=dur,rng=rng)
        n=len(x);hs=np.zeros(n);hd=np.zeros(n);k=np.arange(n)
        for i in s:hs=np.maximum(hs,np.exp(-.5*((k-i)/3.)**2))
        for i in d:hd=np.maximum(hd,np.exp(-.5*((k-i)/4.)**2))
        for a in range(0,n-WIN,stride):
            w=x[a:a+WIN];X.append(((w-w.mean())/(w.std()+1.e-9)).astype(np.float32))
            Y.append(np.stack([hs[a:a+WIN],hd[a:a+WIN]]).astype(np.float32))
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float32)

def real_windows(csv_dir,limit=128):
    import pandas as pd
    from pathlib import Path
    xs=[]
    for f in sorted(Path(csv_dir).glob('PPG*.CSV')):
        try:
            df=pd.read_csv(f,comment='#');x=-df.ir.to_numpy(dtype=np.float64)
        except Exception:continue
        for a in range(0,len(x)-WIN+1,128):
            w=x[a:a+WIN];xs.append(((w-w.mean())/(w.std()+1.e-9)).astype(np.float32))
            if len(xs)>=limit:return np.asarray(xs)
    return np.asarray(xs,dtype=np.float32).reshape(-1,WIN)
