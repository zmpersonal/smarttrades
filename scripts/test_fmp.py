#!/usr/bin/env python3
"""Minimal FMP entitlement test for SmartTrades.ai.
Uses only two API calls so a plan/access problem does not burn through the daily quota.
"""
import os, sys, requests

KEY = os.getenv('FMP_API_KEY', '')
BASE = 'https://financialmodelingprep.com/stable'

if not KEY:
    raise SystemExit('FMP_API_KEY is missing from GitHub Secrets.')

s = requests.Session()
s.headers.update({'User-Agent': 'SmartTrades.ai FMP entitlement test/0.2'})

def check(label, path, **params):
    params['apikey'] = KEY
    r = s.get(f'{BASE}/{path}', params=params, timeout=30)
    print(f'{label}: HTTP {r.status_code}')
    if r.status_code == 402:
        print('RESULT: Your FMP subscription does not permit this endpoint/data entitlement. Check the endpoint Limited Access plan requirements in FMP or upgrade/contact FMP.')
        return False
    if r.status_code == 429:
        print('RESULT: FMP rate/daily usage limit reached. Wait for the quota reset before testing again.')
        return False
    if r.status_code in (401, 403):
        print('RESULT: API key is missing, invalid, or unauthorized.')
        return False
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print('Response:', r.text[:500])
        return False
    data = r.json()
    ok = isinstance(data, list) and len(data) > 0 or isinstance(data, dict) and len(data) > 0
    print('RESULT:', 'OK' if ok else 'Endpoint returned no usable data')
    return ok

quote_ok = check('1/2 quote', 'quote', symbol='AAPL')
statement_ok = check('2/2 income statement', 'income-statement', symbol='AAPL', period='annual', limit=6)

if quote_ok and statement_ok:
    print('\nPASS: The two core FMP endpoint types required by SmartTrades are accessible.')
    print('NOTE: A full refresh still requires a plan/rate limit capable of the production request volume, or a lower-call/bulk data architecture.')
    sys.exit(0)

print('\nFAIL: Do not run the full SmartTrades refresh yet.')
sys.exit(1)
