"""
SmartTrades.AI — Engine 1: Dark Pool Radar
==========================================

Ingests FINRA off-exchange volume data and scores stocks for stealth
institutional accumulation.

Data sources and their real cadence
-----------------------------------
1. FINRA Daily Short Sale Volume, consolidated TRF/ADF ("CNMS")
   https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
   Free. Posted by 6:00pm ET on the trade date. Pipe-delimited:
       Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
   TotalVolume here = OFF-EXCHANGE volume only, not consolidated tape volume.
   This is the daily signal.

2. FINRA ATS Transparency (true per-dark-pool volume)
   Weekly, published 2 weeks late for Tier 1 NMS and 4 weeks for the rest.
   Far too stale to trigger on. Used only as a slow confirming overlay for
   the block-size trend component.

3. Consolidated tape volume + OHLC — from your market data provider
   (Polygon, Tiingo, EODHD, or yfinance for a prototype). Needed because
   FINRA alone cannot tell you what fraction of total volume went off-exchange.

The central metric
------------------
DPI = off-exchange short volume / off-exchange total volume

A high DPI reads BULLISH, which surprises people. Most off-exchange prints
marked "short" are market makers taking the other side of a buy order they
must then hedge. High DPI therefore measures absorbed buying demand, not
bearish positioning.

The main false positive: heavily shorted names, where genuine short
initiation inflates DPI. `short_interest_pct` is used to damp the signal.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

FINRA_DAILY = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"

# ---------------------------------------------------------------- config

@dataclass
class Config:
    lookback_days: int = 90          # trailing window for z-scores
    dpi_window: int = 5              # smoothing window for DPI
    min_dollar_adv: float = 5e6      # liquidity gate
    min_price: float = 3.0           # avoid sub-$3 noise
    min_score: int = 70              # publish threshold

    # Component weights — must sum to 1.0
    weights: dict = field(default_factory=lambda: {
        "dpi_persistence":  0.28,    # sustained buy-side absorption
        "off_exch_share":   0.22,    # institutions routing away from lit venues
        "block_trend":      0.15,    # weekly ATS: prints getting bigger
        "rel_volume":       0.12,    # engagement
        "compression":      0.13,    # coiled range
        "price_stealth":    0.10,    # absorbed without markup — the key tell
    })


# ---------------------------------------------------------------- ingest

def fetch_finra_day(d: date, session: requests.Session | None = None) -> pd.DataFrame:
    """One trading day of consolidated off-exchange volume. Empty on holidays."""
    s = session or requests.Session()
    url = FINRA_DAILY.format(d=d.strftime("%Y%m%d"))
    r = s.get(url, timeout=30)
    if r.status_code != 200 or "Symbol" not in r.text[:200]:
        return pd.DataFrame()

    # keep_default_na=False is load-bearing: "NA" is a real NYSE ticker, and
    # pandas' default NA list coerces it to NaN, after which the notna() filter
    # below silently drops it. Every day, in every file.
    df = pd.read_csv(io.StringIO(r.text), sep="|", keep_default_na=False)
    # The file ends with a bare record-count trailer ("Records: 12326") which a
    # naive read_csv turns into a junk row with a NaN symbol. Verified live.
    df = df[df["Symbol"].notna() & (df["Symbol"] != "")]
    df = df[df["Date"].astype(str).str.fullmatch(r"\d{8}")]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    # Volumes come back as floats, not ints, despite being share counts.
    for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TotalVolume"])

    df = df.rename(columns={
        "Symbol": "symbol",
        "ShortVolume": "oe_short_vol",
        "ShortExemptVolume": "oe_short_exempt",
        "TotalVolume": "oe_total_vol",
    })
    return df[["Date", "symbol", "oe_short_vol", "oe_short_exempt", "oe_total_vol"]]


def fetch_finra_range(start: date, end: date) -> pd.DataFrame:
    """Walk calendar days; FINRA simply 404s on non-trading days."""
    s = requests.Session()
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            day = fetch_finra_day(d, s)
            if not day.empty:
                out.append(day)
        d += timedelta(days=1)
    if not out:
        raise RuntimeError("No FINRA data returned for the requested range.")
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------- transform

def build_panel(finra: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    """
    Join FINRA off-exchange volume to consolidated tape data.

    `tape` must carry: Date, symbol, close, volume (consolidated), high, low.
    """
    df = finra.merge(tape, on=["Date", "symbol"], how="inner")
    df = df.sort_values(["symbol", "Date"])

    # DPI: share of off-exchange volume printed short
    df["dpi"] = df["oe_short_vol"] / df["oe_total_vol"].replace(0, np.nan)

    # Off-exchange participation in total market volume
    df["oe_share"] = df["oe_total_vol"] / df["volume"].replace(0, np.nan)

    g = df.groupby("symbol", group_keys=False)

    df["dpi_5d"]      = g["dpi"].transform(lambda s: s.rolling(5).mean())
    df["oe_share_5d"] = g["oe_share"].transform(lambda s: s.rolling(5).mean())
    df["vol_20d"]     = g["volume"].transform(lambda s: s.rolling(20).mean())
    df["rvol"]        = df["volume"] / df["vol_20d"]
    df["dollar_adv"]  = df["vol_20d"] * df["close"]

    # Range compression: recent true range vs its longer-run baseline.
    # High value = coiling, which is what accumulation without markup looks like.
    tr = (df["high"] - df["low"]) / df["close"]
    df["atr20"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(20).mean())
    df["atr60"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(60).mean())
    df["compression"] = 1 - (df["atr20"] / df["atr60"])

    df["ret_20d"] = g["close"].transform(lambda s: s.pct_change(20))
    return df


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def add_zscores(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = df.groupby("symbol", group_keys=False)
    w = cfg.lookback_days
    df["dpi_z"]      = g["dpi_5d"].transform(lambda s: zscore(s, w))
    df["oe_share_z"] = g["oe_share_5d"].transform(lambda s: zscore(s, w))
    df["rvol_z"]     = g["rvol"].transform(lambda s: zscore(s, w))
    return df


# ----------------------------------------------------------------- score

def _squash(z: float, k: float = 1.6) -> float:
    """Map a z-score to 0-100, saturating so one wild input can't dominate."""
    if not np.isfinite(z):
        return 50.0
    return float(100 / (1 + np.exp(-z / k)))


def score_symbol(row: pd.Series, cfg: Config, block_trend_z: float = 0.0) -> dict:
    """
    Score one symbol on its most recent session.

    `block_trend_z` comes from the weekly FINRA ATS overlay: the 4-week slope
    of average print size. Pass 0.0 if the ATS join isn't wired up yet — the
    component simply contributes a neutral 50.
    """
    c = {
        "dpi_persistence": _squash(row["dpi_z"]),
        "off_exch_share":  _squash(row["oe_share_z"]),
        "block_trend":     _squash(block_trend_z),
        "rel_volume":      _squash(row["rvol_z"]),
        "compression":     float(np.clip(row["compression"], 0, 1) * 100),
        # Stealth: accumulation that hasn't been paid for yet. A big move
        # already made means the information is in the price.
        "price_stealth":   float(np.clip(100 - abs(row["ret_20d"]) * 100 * 6, 0, 100)),
    }

    raw = sum(c[k] * w for k, w in cfg.weights.items())

    # Damp the DPI reading on heavily shorted names, where short initiation
    # inflates DPI for reasons that have nothing to do with accumulation.
    si = row.get("short_interest_pct", 0.0) or 0.0
    if si > 15:
        raw *= 1 - min((si - 15) / 100, 0.20)

    # Directional read
    if row["dpi_5d"] > 0.50 and row["dpi_z"] > 1.0:
        state = "Accumulation"
    elif row["dpi_5d"] < 0.42 and row["dpi_z"] < -1.0:
        state = "Distribution"
    else:
        state = "Neutral"

    return {
        "symbol": row["symbol"],
        "score": round(raw),
        "state": state,
        "dpi_5d": round(row["dpi_5d"] * 100, 1),
        "dpi_z": round(row["dpi_z"], 2),
        "oe_share": round(row["oe_share_5d"] * 100, 1),
        "rvol": round(row["rvol"], 2),
        "compression": round(float(np.clip(row["compression"], 0, 1)) * 100),
        "ret_20d": round(row["ret_20d"] * 100, 1),
        "components": {k: round(v) for k, v in c.items()},
    }


def run(finra: pd.DataFrame, tape: pd.DataFrame,
        short_interest: pd.Series | None = None,
        block_trend: pd.Series | None = None,
        cfg: Config = Config()) -> pd.DataFrame:
    """Full pipeline. Returns the ranked board for the latest session."""
    panel = add_zscores(build_panel(finra, tape), cfg)

    latest = panel.groupby("symbol").tail(1).copy()
    latest = latest[
        (latest["dollar_adv"] >= cfg.min_dollar_adv)
        & (latest["close"] >= cfg.min_price)
        & latest["dpi_z"].notna()
    ]

    if short_interest is not None:
        latest["short_interest_pct"] = latest["symbol"].map(short_interest).fillna(0.0)
    else:
        latest["short_interest_pct"] = 0.0

    bt = block_trend if block_trend is not None else pd.Series(dtype=float)
    rows = [score_symbol(r, cfg, float(bt.get(r["symbol"], 0.0)))
            for _, r in latest.iterrows()]

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out[out["score"] >= cfg.min_score].reset_index(drop=True)


if __name__ == "__main__":
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=150)

    print(f"Pulling FINRA off-exchange volume {start} → {end} …")
    finra = fetch_finra_range(start, end)
    print(f"  {len(finra):,} symbol-days, {finra['symbol'].nunique():,} symbols")

    # Supply `tape` from your market data provider with columns:
    #   Date, symbol, close, volume, high, low
    # e.g. yfinance for a prototype, Polygon or Tiingo for production.
    print("\nNext: join consolidated tape volume, then call run(finra, tape).")
