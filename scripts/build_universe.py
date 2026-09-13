#!/usr/bin/env python3
"""
Build data/universe.json — the ticker list every screener sees.

Ranking: DOLLAR ADV over the full FINRA/SEC intersection.

The earlier version ranked candidates by FINRA off-exchange SHARE volume and
priced only the top slice. That biases the list toward low-priced, high-turnover
names: the first 300 came back with a $21.90 median close and put penny stocks
ahead of megacaps. FINRA has no price, so share volume is all it can offer.

So price the whole intersection instead and rank on dollars. yfinance batches
~50 symbols per request, so ~6,900 candidates is ~140 requests, a few minutes,
cached to disk. Market caps are fetched only for names that clear the ADV floor,
which is where the per-symbol cost would otherwise dominate.

  python scripts/build_universe.py --target 1500
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engines import finra_darkpool as fd          # noqa: E402
from engines import free_sources as fs            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "universe.json"
CACHE = ROOT / "data" / ".universe_prices.pkl"

MIN_DOLLAR_ADV = 10_000_000
MIN_MARKET_CAP = 2_000_000_000


def finra_candidates(days: int = 5) -> pd.Series:
    s, frames, d = requests.Session(), [], date.today()
    while len(frames) < days and (date.today() - d).days < 15:
        day = fd.fetch_finra_day(d, s)
        if not day.empty:
            frames.append(day)
        d -= timedelta(days=1)
    if not frames:
        raise RuntimeError("no FINRA data in the last 15 calendar days")
    print(f"  FINRA sessions: {len(frames)}")
    panel = pd.concat(frames)
    return panel.groupby("symbol")["oe_total_vol"].mean().sort_values(ascending=False)


def price_all(symbols: list[str], chunk: int = 50, use_cache: bool = True) -> pd.DataFrame:
    import yfinance as yf

    cached = {}
    if use_cache and CACHE.exists():
        try:
            cached = pickle.loads(CACHE.read_bytes())
            print(f"  cache hit: {len(cached)} symbols")
        except Exception:
            cached = {}

    todo = [s for s in symbols if s not in cached]
    print(f"  pricing {len(todo)} of {len(symbols)} ({len(todo)//chunk + 1} requests)")
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        try:
            df = yf.download(batch, period="3mo", interval="1d", auto_adjust=False,
                             progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"    [warn] batch {i}: {e}")
            continue
        for sym in batch:
            try:
                sub = df[sym].dropna(subset=["Close"]) if len(batch) > 1 else df.dropna(subset=["Close"])
            except KeyError:
                cached[sym] = None
                continue
            if sub.empty or "Volume" not in sub:
                cached[sym] = None
                continue
            dv = (sub["Close"] * sub["Volume"]).tail(20)
            cached[sym] = {"close": float(sub["Close"].iloc[-1]),
                           "dollar_adv": float(dv.mean()),
                           "px_days": int(len(sub))}
        if (i // chunk) % 20 == 0:
            print(f"    {min(i + chunk, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
            CACHE.write_bytes(pickle.dumps(cached))
    CACHE.write_bytes(pickle.dumps(cached))
    rows = {k: v for k, v in cached.items() if v}
    return pd.DataFrame.from_dict(rows, orient="index")


def market_caps(symbols: list[str]) -> dict:
    import yfinance as yf

    out = {}
    for n, s in enumerate(symbols, 1):
        try:
            out[s] = float(yf.Ticker(s).fast_info["market_cap"])
        except Exception:
            out[s] = None
        if n % 250 == 0:
            print(f"    caps {n}/{len(symbols)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print("[1] FINRA candidate pool")
    adv = finra_candidates()
    print(f"  FINRA symbols: {len(adv)}")

    print("[2] SEC CIK join")
    cik = fs.ticker_cik_map()
    resolved = [s for s in adv.index if s.upper() in cik]
    print(f"  intersection: {len(resolved)}")

    print("[3] dollar ADV for the FULL intersection")
    px = price_all(resolved, use_cache=not args.no_cache)
    print(f"  priced: {len(px)}/{len(resolved)}")

    liquid = px[px["dollar_adv"] >= MIN_DOLLAR_ADV].sort_values("dollar_adv", ascending=False)
    print(f"  clear the ${MIN_DOLLAR_ADV/1e6:.0f}M dollar-ADV floor: {len(liquid)}")

    print("[4] market caps for the liquid set only")
    caps = market_caps(list(liquid.index))

    rows = []
    for sym in liquid.index:
        rows.append({"symbol": sym, "cik": cik[sym.upper()],
                     "close": round(float(liquid.at[sym, "close"]), 4),
                     "dollar_adv": round(float(liquid.at[sym, "dollar_adv"]), 0),
                     "market_cap": caps.get(sym),
                     "finra_oe_adv": round(float(adv.loc[sym]), 0),
                     "px_days": int(liquid.at[sym, "px_days"])})
    df = pd.DataFrame(rows)
    df["pass_cap"] = df["market_cap"].fillna(0) >= MIN_MARKET_CAP
    kept = df[df["pass_cap"]].sort_values("dollar_adv", ascending=False).head(args.target)

    OUT.write_text(json.dumps({
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "ranking": "dollar ADV (20d mean of close x volume) over the full FINRA/SEC intersection",
        "source": "FINRA daily off-exchange candidates, SEC CIK join, yfinance prices",
        "floors": {"dollar_adv": MIN_DOLLAR_ADV, "market_cap": MIN_MARKET_CAP},
        "finra_symbols": int(len(adv)), "cik_resolved": int(len(resolved)),
        "priced": int(len(px)), "cleared_adv": int(len(liquid)),
        "cleared_both": int(df["pass_cap"].sum()), "target": args.target,
        "count": int(len(kept)),
        "tickers": kept["symbol"].tolist(),
        "rows": kept.drop(columns=["pass_cap"]).to_dict("records"),
    }, indent=1, default=str))

    print("\n[5] SHAPE")
    print(f"  FINRA symbols              : {len(adv)}")
    print(f"  resolved to a CIK          : {len(resolved)}")
    print(f"  priced successfully        : {len(px)}")
    print(f"  cleared dollar-ADV floor   : {len(liquid)}")
    print(f"  cleared market-cap floor   : {int(df['pass_cap'].sum())}")
    print(f"  missing market cap         : {int(df['market_cap'].isna().sum())}")
    print(f"  WRITTEN (target {args.target})      : {len(kept)}")
    q = kept["dollar_adv"].quantile([0, .25, .5, .75, 1])
    print("\n  dollar ADV of the written set:")
    for k, v in q.items():
        print(f"    p{int(k*100):3d}  ${v/1e6:,.1f}M")
    q2 = kept["market_cap"].quantile([0, .25, .5, .75, 1])
    print("  market cap:")
    for k, v in q2.items():
        print(f"    p{int(k*100):3d}  ${v/1e9:,.1f}B")
    print(f"  median close: ${kept['close'].median():.2f}   (share-volume ranking gave $21.90)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
