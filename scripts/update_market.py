#!/usr/bin/env python3
"""SmartTrades market updater.
Default provider: FMP. Public display requires appropriate data licensing.
Writes limited public data to data/public.json and full private payloads to .private/.
"""
from __future__ import annotations
import os, csv, json, math, statistics, time
from pathlib import Path
from datetime import datetime, timezone
import requests
ROOT=Path(__file__).resolve().parents[1]
API_KEY=os.getenv('FMP_API_KEY','')
BASE=os.getenv('FMP_BASE_URL','https://financialmodelingprep.com/stable')
PROVIDER=os.getenv('DATA_PROVIDER','fmp')
DEMO=os.getenv('DEMO_MODE','false').lower()=='true'
MIN_MARKET_CAP=float(os.getenv('MIN_MARKET_CAP','2000000000'))
PUBLIC_N=int(os.getenv('PUBLIC_TOP_N','3'))
RANK_N=int(os.getenv('FULL_RANK_N','25'))

S=requests.Session(); S.headers.update({'User-Agent':'SmartTrades.ai research updater/0.1'})

def get(path, **params):
    if not API_KEY:
        raise RuntimeError('FMP_API_KEY is required unless DEMO_MODE=true')
    params['apikey']=API_KEY
    r=S.get(BASE+'/'+path.lstrip('/'),params=params,timeout=30)
    if r.status_code == 402:
        raise RuntimeError(f'FMP_PLAN_ACCESS: HTTP 402 for {path}. Your current FMP subscription does not permit this endpoint/data entitlement. Stop the full refresh and run the FMP entitlement test workflow.')
    if r.status_code == 429:
        raise RuntimeError(f'FMP_RATE_LIMIT: HTTP 429 for {path}. Your FMP request/daily usage limit has been reached.')
    r.raise_for_status()
    return r.json()

def first(x): return x[0] if isinstance(x,list) and x else (x if isinstance(x,dict) else {})
def val(d,*keys,default=None):
    for k in keys:
        if k in d and d[k] not in (None,''): return d[k]
    return default

def safe_div(a,b):
    try:return float(a)/float(b) if b not in (0,None) else None
    except:return None

def cagr(a,b,years):
    try:
        if a<=0 or b<=0:return None
        return (float(a)/float(b))**(1/years)-1
    except:return None

def pct_rank(vals, higher=True):
    good=[v for v in vals if v is not None and math.isfinite(v)]
    if not good:return [50 if v is not None else None for v in vals]
    s=sorted(good); n=len(s)
    out=[]
    for v in vals:
        if v is None or not math.isfinite(v): out.append(None);continue
        import bisect
        p=(bisect.bisect_left(s,v)+bisect.bisect_right(s,v)-1)/2
        q=100*(p/(n-1) if n>1 else .5); out.append(q if higher else 100-q)
    return out

def avg(*xs):
    ys=[x for x in xs if x is not None and math.isfinite(x)]
    return sum(ys)/len(ys) if ys else None

def load_universe():
    with open(ROOT/'config/universe.csv',newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def fetch_ticker(u):
    t=u['ticker']
    prof=first(get('profile',symbol=t)); quote=first(get('quote',symbol=t))
    inc=get('income-statement',symbol=t,period='annual',limit=6)
    bal=get('balance-sheet-statement',symbol=t,period='annual',limit=6)
    cf=get('cash-flow-statement',symbol=t,period='annual',limit=6)
    ratios=get('ratios',symbol=t,period='annual',limit=6)
    metrics=get('key-metrics',symbol=t,period='annual',limit=6)
    divs=get('dividends',symbol=t,limit=80)
    if not inc:return None
    i0=inc[0]; b0=bal[0] if bal else {}; c0=cf[0] if cf else {}; r0=ratios[0] if ratios else {}; m0=metrics[0] if metrics else {}
    revenue=[val(x,'revenue') for x in inc]; eps=[val(x,'eps','epsDiluted') for x in inc]; fcf=[val(x,'freeCashFlow') for x in cf]
    rev_g=cagr(revenue[0],revenue[3],3) if len(revenue)>3 else None
    eps_g=cagr(eps[0],eps[3],3) if len(eps)>3 and eps[0] and eps[3] and eps[0]>0 and eps[3]>0 else None
    fcf_g=cagr(fcf[0],fcf[3],3) if len(fcf)>3 and fcf[0] and fcf[3] and fcf[0]>0 and fcf[3]>0 else None
    price=float(val(quote,'price',default=val(prof,'price',default=0)) or 0)
    mcap=float(val(quote,'marketCap',default=val(prof,'marketCap',default=0)) or 0)
    ni=val(i0,'netIncome'); rev=val(i0,'revenue'); op=val(i0,'operatingIncome'); equity=val(b0,'totalStockholdersEquity','totalEquity'); debt=val(b0,'totalDebt'); cash=val(b0,'cashAndCashEquivalents','cashAndShortTermInvestments'); ebitda=val(i0,'ebitda'); interest=abs(float(val(i0,'interestExpense',default=0) or 0));
    fcf0=val(c0,'freeCashFlow'); capex=val(c0,'capitalExpenditure')
    div_paid=abs(float(val(c0,'dividendsPaid',default=0) or 0))
    pe=val(r0,'priceToEarningsRatio','priceEarningsRatio'); pfcf=val(r0,'priceToFreeCashFlowsRatio','priceToFreeCashFlowRatio'); evebitda=val(m0,'enterpriseValueOverEBITDA','evToEBITDA')
    roe=safe_div(ni,equity); opm=safe_div(op,rev); fcfm=safe_div(fcf0,rev); fcfconv=safe_div(fcf0,ni) if ni and ni>0 else None
    netdebt=(float(debt or 0)-float(cash or 0)); nd_ebitda=safe_div(netdebt,ebitda) if ebitda and ebitda>0 else None; coverage=safe_div(op,interest) if interest else 50
    payout=safe_div(div_paid,fcf0) if fcf0 and fcf0>0 else None
    dy=100*safe_div(div_paid,mcap) if mcap else 0
    # annual dividend history from payment dates/amounts
    annual={}
    for d in divs if isinstance(divs,list) else []:
        date=val(d,'date','paymentDate'); amt=val(d,'dividend','adjDividend','amount')
        if date and amt is not None:
            annual[str(date)[:4]]=annual.get(str(date)[:4],0)+float(amt)
    yrs=sorted(annual,reverse=True); div_g=None
    if len(yrs)>=4 and annual[yrs[0]]>0 and annual[yrs[3]]>0: div_g=cagr(annual[yrs[0]],annual[yrs[3]],3)
    avg200=val(quote,'priceAvg200')
    price_vs_200d=(price/float(avg200)-1) if avg200 not in (None,0) and price else None
    return {'ticker':t,'name':val(prof,'companyName',default=u['name']),'sector':val(prof,'sector',default=u['sector']),'price':price,'market_cap':mcap,'revenue_growth_3y':rev_g,'eps_growth_3y':eps_g,'fcf_growth_3y':fcf_g,'roe':roe,'operating_margin':opm,'fcf_margin':fcfm,'fcf_conversion':fcfconv,'net_debt_ebitda':nd_ebitda,'interest_coverage':coverage,'pe':float(pe) if pe is not None else None,'p_fcf':float(pfcf) if pfcf is not None else None,'ev_ebitda':float(evebitda) if evebitda is not None else None,'dividend_yield':dy,'fcf_payout':payout,'dividend_growth_3y':div_g,'price_vs_200d':price_vs_200d,'source':'Financial Modeling Prep','source_timestamp':datetime.now(timezone.utc).isoformat()}

def score(stocks):
    factors={
      'roe':True,'operating_margin':True,'fcf_margin':True,'fcf_conversion':True,
      'revenue_growth_3y':True,'eps_growth_3y':True,'fcf_growth_3y':True,
      'pe':False,'p_fcf':False,'ev_ebitda':False,'net_debt_ebitda':False,'interest_coverage':True,'price_vs_200d':True,
      'dividend_growth_3y':True,'fcf_payout':False,'dividend_yield':True}
    ranks={}
    for f,hi in factors.items():ranks[f]=pct_rank([x.get(f) for x in stocks],hi)
    for i,x in enumerate(stocks):
        q=avg(ranks['roe'][i],ranks['operating_margin'][i],ranks['fcf_margin'][i],ranks['fcf_conversion'][i])
        g=avg(ranks['revenue_growth_3y'][i],ranks['eps_growth_3y'][i],ranks['fcf_growth_3y'][i])
        v=avg(ranks['pe'][i],ranks['p_fcf'][i],ranks['ev_ebitda'][i])
        fs=avg(ranks['net_debt_ebitda'][i],ranks['interest_coverage'][i])
        mom=ranks['price_vs_200d'][i]
        def comb(parts):
            vals=[(a,b) for a,b in parts if a is not None];return sum(a*b for a,b in vals)/sum(b for a,b in vals) if vals else 0
        smart=comb([(q,.30),(g,.20),(v,.20),(fs,.20),(mom,.10)])
        divsafe=avg(ranks['fcf_payout'][i],ranks['net_debt_ebitda'][i],ranks['interest_coverage'][i])
        divgrow=ranks['dividend_growth_3y'][i]
        x.update({'quality_score':q or 0,'growth_score':g or 0,'valuation_score':v or 0,'financial_strength_score':fs or 0,'momentum_score':mom or 0,'smart_score':smart})
        x['scores']={
          'dividend-growth':comb([(divsafe,.25),(divgrow,.20),(q,.25),(g,.15),(v,.15)]),
          'quality-growth':comb([(q,.30),(g,.25),(v,.20),(fs,.15),(mom,.10)]),
          'garp':comb([(g,.30),(v,.25),(q,.20),(fs,.15),(mom,.10)]),
          'quality-on-sale':comb([(q,.40),(v,.35),(fs,.15),(mom,.10)]),
          'compounders':comb([(q,.35),(ranks['fcf_growth_3y'][i],.25),(g,.20),(fs,.10),(v,.10)]),
          'cash-machines':comb([(ranks['fcf_margin'][i],.35),(ranks['fcf_conversion'][i],.25),(fs,.20),(g,.10),(v,.10)]),
          'fallen-angels':comb([(q,.30),((100-(mom or 50)),.25),(v,.20),(fs,.15),(g,.10)]),
          'low-debt-growth':comb([(g,.30),(fs,.30),(q,.20),(v,.10),(mom,.10)])}
        x['valuation_label']='Attractive' if (v or 0)>=75 else ('Fair' if (v or 0)>=45 else 'Premium')
        tags=[]
        if x.get('dividend_yield',0)>.25:tags.append('dividend')
        if (g or 0)>=65:tags.append('growth')
        if (q or 0)>=70:tags.append('quality')
        if (v or 0)>=70:tags.append('value')
        x['tags']=tags
    return stocks

def eligible(x,slug):
    if x.get('market_cap',0)<MIN_MARKET_CAP:return False
    if slug=='dividend-growth' and (x.get('dividend_yield',0)<=0 or x.get('dividend_growth_3y') is None):return False
    return True

def main():
    if DEMO: raise RuntimeError('DEMO_MODE does not fabricate market rankings. Provide a licensed provider key for live data.')
    if PROVIDER!='fmp':raise RuntimeError('Only DATA_PROVIDER=fmp is implemented in v0.1')
    stocks=[]
    access_failure = None
    for n,u in enumerate(load_universe(),1):
        try:
            x=fetch_ticker(u)
            if x:stocks.append(x)
            print(f'[{n}] {u["ticker"]}: ok')
        except Exception as e:
            print(f'[{n}] {u["ticker"]}: FAIL {e}')
            if 'FMP_PLAN_ACCESS:' in str(e) or 'FMP_RATE_LIMIT:' in str(e):
                access_failure = str(e)
                break
        time.sleep(float(os.getenv('API_DELAY','0.15')))
    if access_failure:
        raise RuntimeError(access_failure)
    if len(stocks)<25:raise RuntimeError(f'Only {len(stocks)} stocks fetched; refusing to publish materially incomplete rankings.')
    stocks=score(stocks)
    rankings={}
    for slug in ['dividend-growth','quality-growth','garp','quality-on-sale','compounders','cash-machines','fallen-angels','low-debt-growth']:
        arr=sorted([x for x in stocks if eligible(x,slug)],key=lambda x:x['scores'][slug],reverse=True)[:RANK_N]
        rankings[slug]=[{**{k:x.get(k) for k in ['ticker','name','sector','price','market_cap','valuation_label','dividend_yield','tags']},'score':x['scores'][slug]} for x in arr]
    stamp=datetime.now(timezone.utc).isoformat()
    pub={'is_demo':False,'updated_at':stamp,'methodology_version':'0.1','coverage_count':len(stocks),'rankings':{k:v[:PUBLIC_N] for k,v in rankings.items()},'public_universe':[{k:x.get(k) for k in ['ticker','name','sector','smart_score','valuation_label','dividend_yield','tags']} for x in sorted(stocks,key=lambda z:z['smart_score'],reverse=True)[:60]]}
    (ROOT/'data/public.json').write_text(json.dumps(pub,indent=2),encoding='utf-8')
    priv=ROOT/'.private';priv.mkdir(exist_ok=True)
    (priv/'rankings.json').write_text(json.dumps({'updated_at':stamp,'rankings':rankings},separators=(',',':')),encoding='utf-8')
    (priv/'stocks.json').write_text(json.dumps({'updated_at':stamp,'stocks':stocks},separators=(',',':')),encoding='utf-8')
    print('Wrote public + private datasets')
if __name__=='__main__': main()
