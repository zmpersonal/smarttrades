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

# The rolling windows build_panel uses. Named rather than inline because the
# panel length every component needs is DERIVED from them — see
# `panel_sessions_required`. Inline, the requirement was a magic 109 that
# nothing connected to the 150-day fetch that had to satisfy it.
VOL_WINDOW = 20                  # vol_20d, the rvol denominator
ATR_SHORT, ATR_LONG = 20, 60     # compression
RET_WINDOW = 20                  # price_stealth drift


@dataclass
class Config:
    lookback_days: int = 90          # trailing window for z-scores
    dpi_window: int = 5              # smoothing window for DPI
    min_dollar_adv: float = 5e6      # liquidity gate
    min_price: float = 3.0           # avoid sub-$3 noise
    # Publish threshold, set against the OBSERVED distribution. The old 70 was
    # chosen against sample rows invented in session one and never audited —
    # the seventh threshold found in that state. Measured on the 11 Sep 2026
    # session (newest FINRA file), ETFs excluded, all 8,121 non-ETF FINRA
    # symbols fetched, 2,860 liquid and scored: max 71, p99 61, p95 57,
    # p90 55, median 48. At 70 the board held two names, a rare alert rather
    # than a board. Counts at or above each cut: 63 -> 12, 62 -> 21, 61 -> 30,
    # 60 -> 57. 62 is the cut that yields ~20. It sits one point above p99, so
    # re-audit whenever weights change or block_trend is wired — that
    # component is pinned at a neutral 50 today and caps every score.
    min_score: int = 62

    # Component weights — must sum to 1.0
    weights: dict = field(default_factory=lambda: {
        "dpi_persistence":  0.28,    # sustained buy-side absorption
        "off_exch_share":   0.22,    # institutions routing away from lit venues
        "block_trend":      0.15,    # weekly ATS: prints getting bigger
        "rel_volume":       0.12,    # engagement
        "compression":      0.13,    # coiled range
        "price_stealth":    0.10,    # absorbed without markup — the key tell
    })


# --------------------------------------------------- what the panel must hold

# Every component is a rolling chain over the JOINED panel, so the panel — not
# the tape fetch — is what has to be long enough. rel_volume is the binding
# one, and it was short by six sessions: a 150-calendar-day FINRA pull yields
# ~103 trading days against the 109 that rvol_z needs, so `rvol_z` was NaN for
# EVERY symbol in every run and `_squash` mapped it to a neutral 50. A 0.12
# weight pinned at neutral compresses every score and lowers the ceiling the
# publish cut is set against.
#
# The chains, and why the longest one wins:
#   rel_volume    volume -> rolling(20) -> rvol -> zscore(90)   = 109
#   dpi / oe      dpi    -> rolling(5)  ->      -> zscore(90)   =  94
#   compression   range  -> rolling(60)                         =  60
#   price_stealth close  -> pct_change(20)                      =  21
def panel_sessions_required(cfg: "Config | None" = None) -> dict:
    """Trading sessions each component needs in the panel to produce a value."""
    cfg = cfg or Config()
    z = cfg.lookback_days
    return {
        "rel_volume":      VOL_WINDOW - 1 + z,
        "dpi_persistence": cfg.dpi_window - 1 + z,
        "off_exch_share":  cfg.dpi_window - 1 + z,
        "compression":     ATR_LONG,
        "price_stealth":   RET_WINDOW + 1,
    }


# Calendar days to request so the panel clears the longest chain with room.
#
# Measured 4 Oct 2026: a weekday FINRA call costs 0.20s and 260 calendar days
# returned 178 trading days in 36 seconds, so the window is nearly free — and
# the TAPE side is free outright, because `equity_ohlcv` pulls full history per
# symbol (`period="max"`) and `load_tape` only SLICES it. Widening the window
# does not add a single per-symbol request.
#
# The margin is the point, not the minimum. At exactly 109 the component works
# until one holiday-heavy stretch, and when it stops nothing fails — it reverts
# to a silent neutral 50, which is the state this is fixing. 200 calendar days
# yielded 138 trading days, 29 clear of the requirement, which survives the
# Thanksgiving-to-MLK run where the weekday-to-session yield is worst.
_SESSION_MARGIN = 30
_WEEKDAY_YIELD = 0.69            # trading days per calendar day, measured


def fetch_calendar_days(cfg: "Config | None" = None) -> int:
    need = max(panel_sessions_required(cfg).values()) + _SESSION_MARGIN
    return int(round(need / _WEEKDAY_YIELD / 10.0) * 10)


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

def build_panel(finra: pd.DataFrame, tape: pd.DataFrame,
                cfg: "Config | None" = None) -> pd.DataFrame:
    """
    Join FINRA off-exchange volume to consolidated tape data.

    `tape` must carry: Date, symbol, close, volume (consolidated), high, low.
    """
    cfg = cfg or Config()
    df = finra.merge(tape, on=["Date", "symbol"], how="inner")
    df = df.sort_values(["symbol", "Date"])

    # DPI: share of off-exchange volume printed short
    df["dpi"] = df["oe_short_vol"] / df["oe_total_vol"].replace(0, np.nan)

    # Off-exchange participation in total market volume
    df["oe_share"] = df["oe_total_vol"] / df["volume"].replace(0, np.nan)

    g = df.groupby("symbol", group_keys=False)

    df["dpi_5d"]      = g["dpi"].transform(lambda s: s.rolling(cfg.dpi_window).mean())
    df["oe_share_5d"] = g["oe_share"].transform(lambda s: s.rolling(cfg.dpi_window).mean())
    df["vol_20d"]     = g["volume"].transform(lambda s: s.rolling(VOL_WINDOW).mean())
    df["rvol"]        = df["volume"] / df["vol_20d"]
    df["dollar_adv"]  = df["vol_20d"] * df["close"]

    # Range compression: recent true range vs its longer-run baseline.
    # High value = coiling, which is what accumulation without markup looks like.
    tr = (df["high"] - df["low"]) / df["close"]
    df["atr20"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(ATR_SHORT).mean())
    df["atr60"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(ATR_LONG).mean())
    df["compression"] = 1 - (df["atr20"] / df["atr60"])

    df["ret_20d"] = g["close"].transform(lambda s: s.pct_change(RET_WINDOW))
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


def score_symbol(row: pd.Series, cfg: Config,
                 block_trend_z: float | None = None) -> dict:
    """
    Score one symbol on its most recent session.

    `block_trend_z` comes from the weekly FINRA ATS overlay: the 4-week slope
    of average print size, which has no loader on the free path. None means
    "not measured" and scores a neutral 50 — distinct from a measured 0.0,
    which also scores 50 and means something entirely different. The old
    signature defaulted to 0.0 and so could not tell the two apart, which is
    why the UI had to guess from the value.
    """
    c = {
        "dpi_persistence": _squash(row["dpi_z"]),
        "off_exch_share":  _squash(row["oe_share_z"]),
        "block_trend":     50.0 if block_trend_z is None else _squash(block_trend_z),
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

    # Which components are a neutral 50 because their INPUT was unavailable,
    # rather than because the reading is genuinely mid. The UI used to infer
    # this from `== 50`, which cannot distinguish the two.
    neutral = [k for k, ok in (
        ("dpi_persistence", np.isfinite(row["dpi_z"])),
        ("off_exch_share",  np.isfinite(row["oe_share_z"])),
        ("block_trend",     block_trend_z is not None),
        ("rel_volume",      np.isfinite(row["rvol_z"])),
    ) if not ok]

    return {
        "symbol": row["symbol"],
        "score": round(raw),
        "state": state,
        "neutral_components": neutral,
        "dpi_5d": round(row["dpi_5d"] * 100, 1),
        "dpi_z": round(row["dpi_z"], 2),
        "oe_share": round(row["oe_share_5d"] * 100, 1),
        "rvol": round(row["rvol"], 2),
        "compression": round(float(np.clip(row["compression"], 0, 1)) * 100),
        "ret_20d": round(row["ret_20d"] * 100, 1),
        "dollar_adv": (float(row["dollar_adv"]) if pd.notna(row.get("dollar_adv"))
                       else None),
        "components": {k: round(v) for k, v in c.items()},
    }


def exclude_etfs(finra: pd.DataFrame, etf_symbols: frozenset) -> tuple[pd.DataFrame, list]:
    """
    Drop ETFs before anything else touches the data — before the per-symbol
    tape fetch, which they would otherwise cost thousands of calls for.

    Off-exchange flow in an ETF is creation/redemption mechanics, not
    accumulation, so a high DPI there is not the signal this engine reads.
    """
    norm = finra["symbol"].map(lambda s: "".join(ch for ch in str(s).upper() if ch.isalnum()))
    is_etf = norm.isin(etf_symbols)
    return finra[~is_etf].copy(), sorted(finra.loc[is_etf, "symbol"].unique())


def run(finra: pd.DataFrame, tape: pd.DataFrame,
        short_interest: pd.Series | None = None,
        block_trend: pd.Series | None = None,
        cfg: Config = Config(),
        report: dict | None = None) -> pd.DataFrame:
    """
    Full pipeline. Returns the ranked board for the latest session.

    `report`, when given, is filled with the funnel and the distribution of
    EVERY scored name before the publish threshold. An empty board is a
    legitimate result only if something was scored: zero rows from 1,500
    names is a quiet market or a mis-set cut, zero rows from 12 names is a
    tape source that died, and the bare row count cannot tell them apart.
    """
    panel = add_zscores(build_panel(finra, tape, cfg), cfg)

    # A component whose chain is longer than the panel does not fail — it
    # returns NaN, `_squash` maps that to a neutral 50, and the score publishes
    # with a silently compressed ceiling. `rvol_z` was in that state for every
    # symbol of every run. The panel length is checkable arithmetic, so check
    # it and say so rather than letting a weight go quietly dead.
    sessions = int(panel["Date"].nunique()) if len(panel) else 0
    short = {k: n for k, n in panel_sessions_required(cfg).items() if n > sessions}
    for k, n in sorted(short.items(), key=lambda kv: -kv[1]):
        print(f"  [warn] darkpool: {k} needs {n} sessions, panel holds "
              f"{sessions} — it will score a NEUTRAL 50 for every symbol and "
              f"compress the ceiling the {cfg.min_score} cut is set against. "
              f"Raise the fetch to {fetch_calendar_days(cfg)} calendar days.")

    latest = panel.groupby("symbol").tail(1).copy()
    liquid = latest[
        (latest["dollar_adv"] >= cfg.min_dollar_adv)
        & (latest["close"] >= cfg.min_price)
        & latest["dpi_z"].notna()
    ]
    # score_symbol feeds `compression` and `ret_20d` through raw np.clip, which
    # passes NaN straight through, where the four z-score components go via
    # _squash and map a non-finite input to neutral. A single symbol with a
    # halted day, a zero close or a flat 60-day range made `round(raw)` raise
    # "cannot convert float NaN to integer" and took down the whole board on the
    # first live run. Require the inputs it consumes raw, the same way dpi_z is
    # already required — an exclusion for missing data, not a substitute value.
    finite = np.isfinite(liquid["compression"]) & np.isfinite(liquid["ret_20d"])
    dropped = int((~finite).sum())
    if dropped:
        print(f"  [warn] darkpool: {dropped}/{len(liquid)} symbols excluded — "
              f"compression or 20d return not computable from the tape "
              f"(e.g. {', '.join(liquid.loc[~finite, 'symbol'].head(3))})")
    latest = liquid[finite].copy()

    if short_interest is not None:
        latest["short_interest_pct"] = latest["symbol"].map(short_interest).fillna(0.0)
    else:
        latest["short_interest_pct"] = 0.0

    bt = block_trend if block_trend is not None else pd.Series(dtype=float)
    rows = [score_symbol(r, cfg, float(bt.get(r["symbol"], 0.0)))
            for _, r in latest.iterrows()]

    out = (pd.DataFrame(rows).sort_values("score", ascending=False)
           if rows else pd.DataFrame(columns=["symbol", "score"]))
    if report is not None:
        sc_ = out["score"].astype(float) if len(out) else pd.Series(dtype=float)
        q = (lambda p: None if sc_.empty else float(sc_.quantile(p)))
        report.update({
            "finra_symbols": int(finra["symbol"].nunique()),
            "tape_symbols": int(tape["symbol"].nunique()) if len(tape) else 0,
            "joined_symbols": int(panel["symbol"].nunique()),
            "latest_session": str(panel["Date"].max())[:10] if len(panel) else None,
            "liquid": int(len(liquid)),
            "excluded_uncomputable": dropped,
            "scored": int(len(out)),
            "min_score": cfg.min_score,
            "passed": int((sc_ >= cfg.min_score).sum()),
            "max": None if sc_.empty else float(sc_.max()),
            "p99": q(.99), "p95": q(.95), "p90": q(.90), "median": q(.50),
            "top": out.head(10)[["symbol", "score"]].to_dict("records") if len(out) else [],
            # The components that cannot move today, so a ceiling is visible
            # as a ceiling rather than read as a quiet market.
            "rvol_z_available": int(latest["rvol_z"].notna().sum()) if len(latest) else 0,
            "block_trend_wired": block_trend is not None,
            "panel_sessions": sessions,
            "components_short_of_data": short,
        })
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
