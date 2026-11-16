#!/usr/bin/env python3
"""Audit reviewed D statuses and supervised-window masks without mutating labels."""
from pathlib import Path
import csv,json,collections,hashlib
r=Path(__file__).resolve().parents[1]
original=json.loads((r/'data/manual_review.json').read_text());auto=json.loads((r/'tables/auto_session.json').read_text())
assert original['source_input_sha256']==auto['input_sha256']==hashlib.sha256((r/'data/PPG00_original.zip').read_bytes()).hexdigest()
assert len(original['decisions'])==len(auto['recordings'])==23
b=collections.Counter(x.get('d_status','unset') for v in original['decisions'].values() for x in v['beats'].values())
assert b=={'distinct':72,'shoulder':184,'absent':3,'unset':3},b
rows=list(csv.DictReader((r/'data/real_256_windows_reviewed.csv').open()))
assert len(rows)==161
assert sum(int(x['s_supervised']) for x in rows)==151
assert sum(int(x['d_supervised']) for x in rows)==135
for row in rows:
    assert row['d_supervised'] in ('0','1')
    if row['d_supervised']=='0':assert row['d_label_source']=='unknown_masked'
    if row['d_supervised']=='1':assert row['d_label_source']=='manual'
    assert row['dn_supervised']=='0'
print('PASS: 23 recordings, 262 beats, correct masks, raw-input provenance, all D/unknown decisions preserved')
