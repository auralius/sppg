#!/usr/bin/env python3
"""Convert annotated review.json to a 256-window CSV with safe D labels.

S is algorithmic pseudo-label unless manually changed/reviewed.
D is supervised *only* in windows whose entire beat coverage is explicitly
reviewed as distinct, shoulder, or absent. Unknown D is never negative.
"""
import argparse,csv,json,copy,sys
from pathlib import Path
import numpy as np
from auto_label import WINDOW, STARTS, peaks_to_heatmap

def export(folder,review_file,target):
    folder=Path(folder)
    original=json.loads((folder/'tables'/'auto_session.json').read_text())
    review=json.loads(Path(review_file).read_text())
    if review.get('schema')!='SPPG_real_manual_review_v1':raise ValueError('Review schema mismatch')
    if review.get('source_input_sha256')!=original.get('input_sha256'):raise ValueError('Wrong recording set: SHA256 mismatch')
    headers=['sample_id','split','seed','record_id','window_in_record','start_sample','case','augmented',
             'd_negative_weight','s_label_source','d_label_source','s_supervised','d_supervised','dn_supervised']
    for typ in ('x','s','d','dn'):
        headers += [f'{typ}_{i:03d}' for i in range(WINDOW)]
    finished=0;total=0; reviewed_beats=0
    with Path(target).open('w',newline='') as out:
        wr=csv.writer(out);wr.writerow(headers)
        for rec in original['recordings']:
            notes=review.get('decisions',{}).get(rec['file'],{})
            overlays=notes.get('beats',{})
            beats=[]
            for base in rec['beats']:
                b={**base,**overlays.get(base['beat_id'],{})}
                status=b.get('d_status','unreviewed')
                if status=='distinct':
                    if b.get('d_confirmed') is None:
                        raise ValueError(f"{base['beat_id']}: distinct D but no reviewed D position")
                    if not (b['s'] < b['d_confirmed'] < (b['next_s'] if b['next_s'] is not None else len(rec['ir']))):
                        raise ValueError(f"{base['beat_id']}: D must fall after S before next S")
                if status in ('distinct','shoulder','absent'): reviewed_beats+=1
                beats.append(b)
            ir=np.asarray(rec['ir'],dtype=float)
            for j,start in enumerate(STARTS):
                total+=1
                end=start+WINDOW
                # Ensure all source beats overlapping this window are reviewed,
                # including the beat whose S precedes the window.
                covering=[b for b in beats if b['s']<end and (b['next_s'] if b['next_s'] is not None else len(ir))>start]
                fully_covered=bool(covering and covering[0]['s']<=start and (covering[-1]['next_s'] if covering[-1]['next_s'] is not None else len(ir))>=end)
                d_ok=int(fully_covered and all(b.get('d_status') in ('distinct','shoulder','absent') for b in covering))
                if d_ok: finished+=1
                s_ok=int(not any(not b.get('s_auto_ok',False) and start<=b['s']<end and not b.get('s_reviewed',False) for b in beats))
                s_l=[b['s']-start for b in beats if start<=b['s']<end and (b.get('s_auto_ok') or b.get('s_reviewed'))]
                d_l=[b['d_confirmed']-start for b in beats if b.get('d_status')=='distinct' and b.get('d_confirmed') is not None and start<=b['d_confirmed']<end]
                vals=-(ir[start:end]);vals=(vals-vals.mean())/max(float(vals.std()),1e-6)
                head=[f"{rec['file'][:-4]}_w{j:02d}",'real_reviewed','',rec['file'][:-4],j,start,'real_reviewed','no','',
                      'manual_or_algorithm_provisional','manual' if d_ok else 'unknown_masked',s_ok,d_ok,0]
                wr.writerow(head+[f'{v:.8g}' for v in vals] +
                       [f'{v:.7g}' for v in peaks_to_heatmap(s_l)] +
                       [f'{v:.7g}' for v in peaks_to_heatmap(d_l)] + [0]*WINDOW)
    print(f'{target}: {total} windows, D supervised on {finished}, manually reviewed secondary classifications {reviewed_beats}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',default='.');p.add_argument('--review',required=True);p.add_argument('--out',default='real_256_windows_reviewed.csv');a=p.parse_args()
    export(a.package,a.review,a.out)
