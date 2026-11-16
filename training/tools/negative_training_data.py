"""Negative-aware PPG training data (strictly synthetic, record-level splits).

Four kinds: dual peak, weak secondary component, systolic only, noise/motion only.
The D label is the *generator's second component center*, NOT a clinical landmark.
Never use this file to infer prevalence or make medical claims.
"""
from pathlib import Path
import numpy as np
from scipy import signal as sg
from ppg_synth import P_DEFAULT, FS, WIN, synth_record

CASES = ('dual', 'weak', 'systolic_only', 'noise_only')
PROBS = np.array([.40, .25, .25, .10])

def generate_record(case, dur, rng, hr=None):
    n=int(FS*dur)
    if case == 'noise_only':
        t=np.arange(n,dtype=np.float64)/FS
        # sensor baseline and random drift, no pulsatile component
        x=(rng.uniform(50,550)*np.sin(2*np.pi*rng.uniform(.02,.15)*t + rng.uniform(0,6.28))
           +rng.normal(0,rng.uniform(8,100),n))
        for _ in range(int(rng.integers(0,4))):
            L=int(rng.integers(25,120))
            if L>=n:continue
            a=int(rng.integers(0,n-L))
            artifact=sg.sosfilt(sg.butter(2,[.7,8],fs=FS,btype='bandpass',output='sos'),rng.standard_normal(L+60))[60:]
            x[a:a+L]+=rng.uniform(100,1600)*np.hanning(L)*artifact/(artifact.std()+1e-6)
        return x, np.array([],np.int32), np.array([],np.int32)
    p=P_DEFAULT.copy()
    if case == 'weak':
        p[:,3] *= rng.uniform(.35,.70)  # second Gaussian remains present but reduced
        p[:,4] += rng.uniform(.01,.055)
    elif case == 'systolic_only':
        p[:,3] = 0.0
        p[:,6] = 0.0   # remove second + late broad components, like held-out control
    elif case != 'dual':
        raise ValueError(case)
    # Broader amplitude/width/phase perturbations; independently seeded by split.
    p[:,0] *= rng.uniform(.75,1.35)
    p[:,2] *= rng.uniform(.85,1.20)
    p[:,5] *= rng.uniform(.83,1.20)
    x,s,d,_ = synth_record(P=p, dur=dur, rng=rng, hr=hr,noise0=rng.uniform(6,75))
    if case=='systolic_only':d=np.array([],dtype=np.int32)
    return x,s,d

def make_negative_windows(n_records=120,dur=12,seed=541,hop=64):
    rng=np.random.default_rng(seed)
    cases=[CASES[i] for i in rng.choice(len(CASES),size=n_records,p=PROBS)]
    # Stratification: ensure every case is represented for moderately small data sets.
    if n_records>=4:
        for i,c in enumerate(CASES):cases[i]=c
    rng.shuffle(cases)
    X=[];Y=[];meta=[]
    for rid,case in enumerate(cases):
        x,s,d=generate_record(case,dur,rng)
        idx=np.arange(len(x));hs=np.zeros(len(x));hd=np.zeros(len(x))
        for k in s:hs=np.maximum(hs,np.exp(-.5*((idx-k)/3.)**2))
        for k in d:hd=np.maximum(hd,np.exp(-.5*((idx-k)/4.)**2))
        for start in range(0,len(x)-WIN+1,hop):
            w=x[start:start+WIN]
            z=((w-w.mean())/(w.std()+1e-9)).astype(np.float32)
            X.append(z)
            Y.append(np.stack((hs[start:start+WIN],hd[start:start+WIN])).astype(np.float32))
            meta.append(case)
    return np.asarray(X,dtype=np.float32),np.asarray(Y,dtype=np.float32),np.asarray(meta)
