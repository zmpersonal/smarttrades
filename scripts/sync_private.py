#!/usr/bin/env python3
import os,json,requests
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=os.getenv('SMARTTRADES_API_BASE','').rstrip('/'); token=os.getenv('SMARTTRADES_SYNC_TOKEN','')
if not base or not token: print('Private sync skipped: SMARTTRADES_API_BASE / SMARTTRADES_SYNC_TOKEN not configured'); raise SystemExit(0)
h={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
r=json.loads((root/'.private/rankings.json').read_text())
for slug,items in r['rankings'].items():
    x=requests.post(base+'/api/admin/sync-ranking',headers=h,json={'slug':slug,'updated_at':r['updated_at'],'items':items},timeout=60);x.raise_for_status();print('synced ranking',slug)
s=json.loads((root/'.private/stocks.json').read_text())['stocks']
for i in range(0,len(s),50):
    x=requests.post(base+'/api/admin/sync-stocks',headers=h,json={'stocks':s[i:i+50]},timeout=60);x.raise_for_status();print('synced stocks',i,'-',i+50)
