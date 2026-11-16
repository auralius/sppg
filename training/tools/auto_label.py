#!/usr/bin/env python3
"""Conservative, model-independent PPG landmark PROPOSAL generator.

This is NOT a validated physiological ground truth annotation system.
Only systolic candidates are treated as pseudo-labels; D/DN require review.
Requires: numpy, scipy, matplotlib.
"""
import argparse, csv, io, json, zipfile, hashlib
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, savgol_filter, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 100
SIGMA = 3.0
WINDOW = 256
STARTS = (0, 128, 256, 384, 512, 640, 745)

def read_input(path):
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.csv') and '/' not in n.strip('/'))
        out=[]
        for name in names:
            text=z.read(name).decode('utf-8-sig')
            comments=[line for line in text.splitlines() if line.startswith('#')]
            rows=list(csv.DictReader(io.StringIO('\n'.join(t for t in text.splitlines() if t and not t.startswith('#')))))
            required={'sample_idx','nominal_time_ms','read_elapsed_us','red','ir'}
            if not rows or not required.issubset(rows[0]):
                raise ValueError(f'{name}: unexpected CSV columns')
            d={k:np.asarray([int(r[k]) for r in rows],dtype=np.int64) for k in required}
            if len(rows)!=1001 or not np.array_equal(d['sample_idx'],np.arange(len(rows))):
                raise ValueError(f'{name}: expect contiguous 1001 samples')
            if not np.array_equal(d['nominal_time_ms'],d['sample_idx']*10):
                raise ValueError(f'{name}: nominal time not 100 Hz')
            if np.any(np.diff(d['read_elapsed_us'])<0):
                raise ValueError(f'{name}: host timestamps non-monotonic')
            if not any('fs_hz=100 ' in c for c in comments):
                raise ValueError(f'{name}: fs_hz=100 metadata missing')
            out.append((name,d,comments,z.read(name)))
        if len(out)==0: raise ValueError('No CSVs found')
        return out

def detect_s(y):
    prominence = max(75.0, 0.30*float(np.std(y)))
    s, props = find_peaks(y, distance=52, prominence=prominence)
    # The signal amplitude is typically hundreds of ADC units.
    return s, props, prominence

def extract(input_zip, outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for part in ['raw','tables','plots']:(outdir/part).mkdir(exist_ok=True)
    data=read_input(input_zip)
    sos=butter(2,(0.45,12.),btype='bandpass',fs=FS,output='sos')
    store=[]; event_rows=[]; stats=[]; all_s=all_beat=all_dcand=0
    for name, cols, meta, original in data:
        (outdir/'raw'/name).write_bytes(original)
        ir=cols['ir'].astype(float); red=cols['red'].astype(float)
        y=-sosfiltfilt(sos,ir)
        y=savgol_filter(y,9,3,mode='interp')
        ss,props,pro=detect_s(y)
        rr=np.diff(ss)
        bpm=round(60*FS/np.median(rr),1) if rr.size else None
        med_rr=float(np.median(rr)) if rr.size else 0.0
        beats=[]
        for i,s in enumerate(ss):
            next_s = int(ss[i+1]) if i+1<len(ss) else None
            prev_s = int(ss[i-1]) if i>0 else None
            this_rr=next_s-s if next_s is not None else (s-prev_s if prev_s is not None else 0)
            period_regular=bool(med_rr and (0.73*med_rr<=this_rr<=1.30*med_rr))
            s_auto_ok=bool(period_regular and s>=18 and s<=len(y)-22)
            d = None; dn = None; strength=0.0; d_rel=0.0
            if next_s is not None and next_s-s>=44:
                stop=min(next_s-12,s+int(0.60*(next_s-s)))
                start=s+9
                if stop-start>9:
                    local_p,_=find_peaks(y[start:stop],prominence=0.05,distance=7)
                    local_p=local_p+start
                    candidates=[]
                    for p in local_p:
                        if p-s<9: continue
                        # Require a local trough between S and potential D.
                        pre = y[s+3:p]
                        if len(pre)<4: continue
                        j=s+3+int(np.argmin(pre))
                        rise=float(y[p]-y[j]);
                        # A beat amplitude proxy from S to next onset trough.
                        amp=max(1.0, float(y[s]-np.min(y[s:next_s])))
                        frac=rise/amp
                        if rise>0:
                            candidates.append((rise,p,j,frac))
                    if candidates:
                        rise,p,j,frac=max(candidates)
                        if rise>=max(8.0,0.010*float(y[s]-np.min(y[s:next_s]))):
                            d=int(p);dn=int(j);strength=round(rise,2);d_rel=round(frac,4)
            status='candidate_unverified' if d is not None else 'unreviewed_shoulder_or_absent'
            beat={
                'beat_id':f'{name[:-4]}_b{i+1:02d}', 'file':name,
                's':int(s),'s_sec':round(int(s)/FS,3), 's_source':'algorithm_provisional',
                's_auto_ok':s_auto_ok,'next_s':next_s,
                'dn_candidate':dn,'d_candidate':d,'d_rise_adc':strength,'d_rise_fraction':d_rel,
                'd_status':status,'d_confirmed':None,'dn_confirmed':None,
                'reviewed':False,'review_note':'',
            }
            beats.append(beat)
            event_rows.append(beat.copy())
        n_candidate=sum(b['d_candidate'] is not None for b in beats)
        n_good=sum(b['s_auto_ok'] for b in beats)
        all_s+=len(ss);all_beat+=len(beats);all_dcand+=n_candidate
        rr_std=round(float(np.std(rr)/np.mean(rr)*100),1) if len(rr)>1 else None
        jitter_count=int(sum('fifo_bursts_ge4=0' not in c for c in meta if 'capture_seconds=' in c))
        stats.append(dict(file=name,s_samples=len(ss),s_pseudo_usable=n_good,d_tentative=n_candidate,
                          median_bpm=bpm,rr_cv_pct=rr_std,ir_std_adc=round(float(ir.std()),1),
                          red_std_adc=round(float(red.std()),1),host_span_sec=round(float(cols['read_elapsed_us'][-1]/1e6),3),
                          fifo_bursts_ge4=int(next((c.split('fifo_bursts_ge4=')[1].strip() for c in meta if 'fifo_bursts_ge4=' in c),'0')),
                          hand='',finger='',recording_notes=''))
        store.append({'file':name,'red':red.astype(int).tolist(),'ir':ir.astype(int).tolist(),
                      'waveform':np.round(y,3).tolist(),'beats':beats,'median_bpm':bpm})
        # Raw and smoothed inverted IR are both visible, but only filtered plot is marked.
        t=np.arange(len(y))/FS
        fig,ax=plt.subplots(figsize=(15,3.75))
        ax.plot(t,y,color='#1872b6',lw=1.35,label='Inverted IR, zero-phase offline 0.45–12 Hz')
        for beat in beats:
            s=beat['s'];ax.plot(s/FS,y[s],marker='o',ms=4,color='#cf951e')
            if beat['d_candidate'] is not None:
                d=beat['d_candidate'];dn=beat['dn_candidate'];
                ax.plot(d/FS,y[d],marker='x',ms=5,color='#d34c82')
                ax.plot(dn/FS,y[dn],marker='v',ms=3,color='#479973')
        ax.set(title=f'{name}   |   S candidates={len(ss)}   |   tentative D={n_candidate}   |   median pulse rate={bpm} BPM',
               xlabel='Time (seconds)',ylabel='Relative IR waveform (ADC units)')
        ax.grid(alpha=.18);ax.set_xlim(0,10)
        ax.text(.01,.02,'Gold circles: provisional S   ·   Pink crosses: UNVERIFIED D candidates   ·   Green triangles: UNVERIFIED notch',
                transform=ax.transAxes,fontsize=8,color='#666')
        fig.tight_layout();fig.savefig(outdir/'plots'/name.replace('.CSV','.png'),dpi=125);plt.close(fig)

    with (outdir/'tables'/'auto_landmark_candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(event_rows[0])); writer.writeheader();writer.writerows(event_rows)
    with (outdir/'tables'/'recording_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    with (outdir/'tables'/'auto_session.json').open('w') as f:
        json.dump({'schema':'SPPG_real_auto_candidates_v1','sample_rate_hz':FS,'recordings':store,
                   'input_sha256':hashlib.sha256(Path(input_zip).read_bytes()).hexdigest(),
                   'note':'All S are provisional. D/DN are UNVERIFIED candidates only. Review before D training.'},f,separators=(',',':'))
    # CSV whose headers resemble the frozen synthetic training CSVs, plus explicit
    # masks to prevent unknown D labels from silently becoming negative examples.
    build_windows(store,outdir/'tables'/'real_256_windows_S_pseudo_D_masked.csv')
    with (outdir/'tables'/'summary.json').open('w') as f:
        json.dump({'recordings':len(store),'total_samples':sum(len(x['ir']) for x in store),
                   'systolic_proposals':all_s,'systolic_pseudo_usable':sum(s['s_pseudo_usable'] for s in stats),
                   'd_candidates_unverified':all_dcand,'window_count':len(store)*len(STARTS),
                   'note':'Not physiological reference labels; D candidate count is not D true positive count.'},f,indent=2)
    print(json.dumps(json.loads((outdir/'tables'/'summary.json').read_text()),indent=2))

def peaks_to_heatmap(locals,window=WINDOW):
    a=np.zeros(window,dtype=np.float32)
    xx=np.arange(window,dtype=np.float32)
    for i in locals:
        a=np.maximum(a,np.exp(-0.5*((xx-i)/SIGMA)**2))
    return a

def build_windows(store,target):
    fields=['sample_id','split','seed','record_id','window_in_record','start_sample','case','augmented','d_negative_weight','s_label_source','d_label_source','s_supervised','d_supervised','dn_supervised']
    fields += [f'{kind}_{i:03d}' for kind in 'xsdn'.split() for i in range(WINDOW)] if False else []
    fields += [f'x_{i:03d}' for i in range(WINDOW)]
    fields += [f's_{i:03d}' for i in range(WINDOW)]
    fields += [f'd_{i:03d}' for i in range(WINDOW)]
    fields += [f'dn_{i:03d}' for i in range(WINDOW)]
    target=Path(target)
    with target.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for rec in store:
            ir=np.asarray(rec['ir'],dtype=float)
            for j,start in enumerate(STARTS):
                vals=-(ir[start:start+WINDOW]);vals=(vals-vals.mean())/max(float(vals.std()),1e-6)
                ss=[b['s']-start for b in rec['beats'] if b['s_auto_ok'] and start<=b['s']<start+WINDOW]
                # Do not treat a questionable S proposal as a confirmed negative.
                questionable=any((not b['s_auto_ok']) and start<=b['s']<start+WINDOW for b in rec['beats'])
                s_supervised=int(not questionable)
                s=peaks_to_heatmap(ss)
                zero=np.zeros(WINDOW,dtype=np.float32)
                header=[f"{rec['file'][:-4]}_w{j:02d}",'real_unassigned','',rec['file'][:-4],j,start,
                        'real_unknown_D','no','', 'algorithm_provisional','unknown_masked',s_supervised,0,0]
                writer.writerow(header + [f'{v:.8g}' for v in vals] + [f'{v:.7g}' for v in s] + [0]*WINDOW + [0]*WINDOW)

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--zip',required=True);pa.add_argument('--out',required=True)
    args=pa.parse_args(); extract(args.zip,args.out)
