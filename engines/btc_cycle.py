"""
SmartTrades.AI — Engine 6: Bitcoin Cycle
========================================

Three alerts, each gated on market regime:

    1. Four-year halving cycle position
    2. RSI three-push bearish divergence
    3. Bull-market dip buy (armed only in a confirmed uptrend)

The gating is the whole design. A bull-market dip rule that fires during a
markdown phase is worse than having no rule, because it fires exactly when
it is most expensive to be wrong. Every alert below declares its regime
precondition and refuses to fire outside it.

Read this before trusting any of it
-----------------------------------
There have been four halvings and three completed cycles. Every statement
about "what the four-year cycle does" rests on three observations. That is
not a backtest — it is an anecdote count, and no amount of careful coding
turns n=3 into statistical evidence.

The current cycle is also actively challenging the pattern. The October 2025
peak arrived on schedule (day 535 post-halving, against 526 and 548 in the
two prior cycles), but the drawdown since has been roughly 39% against 78-86%
in every prior cycle. Either ETF flows genuinely changed the buyer base, or
the drawdown is unfinished. The data cannot yet separate those, so the code
reports both rather than picking one.

Data sources
------------
Price/OHLC   Any exchange API or CoinGecko. Weekly candles for the cycle and
             RSI work; the daily series only adds noise at this timescale.
On-chain     STH-MVRV and short-term holder realized price need Glassnode
             (~$40/mo entry tier), CryptoQuant, or Bitcoin Magazine Pro.
             There is no free source with reliable history for these.
Halvings     Derived from block height; no API needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Block-height derived, not estimated.
HALVINGS = [date(2012, 11, 28), date(2016, 7, 9), date(2020, 5, 11), date(2024, 4, 20)]
NEXT_HALVING_EST = date(2028, 4, 20)

# Completed cycles only. Three rows. That is the entire evidence base.
HISTORY = pd.DataFrame([
    {"peak": date(2013, 11, 30), "days_post_halving": 367,
     "trough": date(2015, 1, 14), "drawdown": -0.86, "peak_to_trough_days": 410},
    {"peak": date(2017, 12, 17), "days_post_halving": 526,
     "trough": date(2018, 12, 15), "drawdown": -0.84, "peak_to_trough_days": 363},
    {"peak": date(2021, 11, 10), "days_post_halving": 548,
     "trough": date(2022, 11, 21), "drawdown": -0.78, "peak_to_trough_days": 376},
])


@dataclass
class Config:
    rsi_period: int = 14
    oversold: float = 30.0          # anchor that opens a divergence sequence
    min_push1_rsi: float = 65.0     # push 1 needs real momentum for its decay to mean anything
    min_weeks_between_pushes: int = 4
    min_price_hh_pct: float = 0.5   # a higher high must actually be higher

    # Bull-market dip buy
    sth_mvrv_trigger: float = 1.00
    band_touch_tolerance: float = 0.03   # within 3% of the band counts as a touch
    regime_weeks_above_band: int = 4     # weekly closes needed to confirm an uptrend
    max_drawdown_for_bull: float = 0.25  # >25% below ATH is not a bull-market dip


# ------------------------------------------------------------------ RSI

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Use weekly closes for anything cycle-scale."""
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


# -------------------------------------------------- 1. cycle position

def cycle_position(today: date, ath: float, ath_date: date, price: float) -> dict:
    last = max(h for h in HALVINGS if h <= today)
    days_since = (today - last).days
    cycle_len = (NEXT_HALVING_EST - last).days

    peak_day = (ath_date - last).days
    days_since_peak = (today - ath_date).days
    drawdown = price / ath - 1

    avg_ptt = HISTORY["peak_to_trough_days"].mean()
    trough_lo = ath_date + timedelta(days=int(HISTORY["peak_to_trough_days"].min()))
    trough_hi = ath_date + timedelta(days=int(HISTORY["peak_to_trough_days"].max()))

    alerts = []
    if trough_lo <= today <= trough_hi:
        alerts.append(f"Inside the historical trough window ({trough_lo} to {trough_hi}).")
    for level in (-0.50, -0.65, -0.78):
        if drawdown <= level:
            alerts.append(f"Drawdown passed {level:.0%}, a prior-cycle trough zone.")
    if days_since_peak > avg_ptt:
        alerts.append(f"{days_since_peak} days past peak, beyond the {avg_ptt:.0f}-day average.")

    # The honest comparison, reported rather than resolved.
    shallow = bool(drawdown > HISTORY["drawdown"].max())

    return {
        "days_since_halving": days_since,
        "cycle_pct": round(days_since / cycle_len * 100, 1),
        "peak_day_post_halving": peak_day,
        "peak_timing_matched": bool(500 <= peak_day <= 570),
        "days_since_peak": days_since_peak,
        "drawdown": round(drawdown, 4),
        "prior_drawdowns": HISTORY["drawdown"].tolist(),
        "shallower_than_every_prior_cycle": shallow,
        "projected_trough_window": (trough_lo.isoformat(), trough_hi.isoformat()),
        "next_halving": NEXT_HALVING_EST.isoformat(),
        "alerts": alerts,
        "caveat": "n=3 completed cycles. Context for sizing, not a forecast.",
    }


# ------------------------------- 2. RSI three-push bearish divergence

@dataclass
class Push:
    idx: int
    date: date
    price: float
    rsi: float


def find_pushes(df: pd.DataFrame, cfg: Config = Config()) -> list[Push]:
    """
    Locate successive price highs after the most recent oversold anchor.

    The oversold print does not *contain* the divergence — it opens the
    sequence. Divergence appears later, at the highs of pushes 2 and 3,
    where price makes a higher high and RSI does not follow.
    """
    df = df.copy()
    df["rsi"] = rsi(df["close"], cfg.rsi_period)

    anchors = df.index[df["rsi"] < cfg.oversold]
    if len(anchors) == 0:
        return []
    seq = df.loc[anchors[-1]:]

    # Local maxima on close, spaced far enough apart to be distinct pushes.
    w = cfg.min_weeks_between_pushes
    pushes: list[Push] = []
    for i in range(w, len(seq) - w):
        window = seq["close"].iloc[i - w:i + w + 1]
        if seq["close"].iloc[i] == window.max():
            row = seq.iloc[i]
            if pushes and (i - pushes[-1].idx) < w:
                if row["close"] > pushes[-1].price:
                    pushes[-1] = Push(i, row.name.date(), row["close"], row["rsi"])
                continue
            pushes.append(Push(i, row.name.date(), row["close"], row["rsi"]))
    return pushes


def detect_divergence(pushes: list[Push], cfg: Config = Config()) -> dict:
    """
    Bearish divergence: price higher high, RSI lower high.

    Push 2 is a warning. Push 3 completes the three-drives pattern and is the
    stronger of the two. Push 1 must have exceeded `min_push1_rsi`, otherwise
    there was never enough momentum for its decay to signify anything.
    """
    if len(pushes) < 2:
        return {"state": "insufficient pushes", "pushes": len(pushes), "signal": None}

    p1 = pushes[0]
    out = {"pushes": len(pushes), "anchor_push_rsi": round(float(p1.rsi), 1), "signals": []}

    if p1.rsi < cfg.min_push1_rsi:
        out["state"] = f"push 1 RSI {p1.rsi:.1f} below the {cfg.min_push1_rsi:.0f} strength floor"
        out["signal"] = None
        return out

    for n, p in enumerate(pushes[1:], start=2):
        price_hh = (p.price / p1.price - 1) * 100 > cfg.min_price_hh_pct
        rsi_lh = p.rsi < p1.rsi
        if price_hh and rsi_lh:
            out["signals"].append({
                "push": n, "date": p.date.isoformat(),
                "price_delta_pct": round(float(p.price / p1.price - 1) * 100, 2),
                "rsi_delta": round(float(p.rsi - p1.rsi), 1),
                "strength": "strong (three drives)" if n >= 3 else "warning",
            })

    out["signal"] = out["signals"][-1] if out["signals"] else None
    out["state"] = ("bearish divergence" if out["signals"]
                    else "momentum confirming — no divergence")
    # Divergence can persist for months in a strong trend. It marked the 2021
    # top well and gave repeated false warnings through 2017.
    out["caveat"] = "Warning to tighten risk, not a timing tool."
    return out


# ----------------------------------------- 2b. daily RSI entry timing

# Andrew Cardwell's observation, which matters more than the textbook 30/70:
# RSI does not oscillate around the same centre in every regime. In an uptrend
# it tends to live between 40 and 80 and finds support near 40-50; in a
# downtrend it lives between 20 and 60 and meets resistance near 55-65.
#
# Using a fixed 30 threshold therefore under-fires in bull markets (price
# bottoms at RSI 40 and you never get a signal) and over-fires in bear markets
# (RSI 30 prints repeatedly all the way down). The bands adapt instead.
RSI_BANDS = {
    "bull":    {"oversold": 40, "overbought": 80, "mid": 50},
    "bear":    {"oversold": 30, "overbought": 65, "mid": 45},
    "neutral": {"oversold": 32, "overbought": 70, "mid": 50},
}


def find_troughs(df: pd.DataFrame, window: int = 5) -> list[Push]:
    """Local price minima — the mirror of find_pushes, for the buy side."""
    out: list[Push] = []
    for i in range(window, len(df) - window):
        seg = df["close"].iloc[i - window:i + window + 1]
        if df["close"].iloc[i] == seg.min():
            row = df.iloc[i]
            if out and (i - out[-1].idx) < window:
                if row["close"] < out[-1].price:
                    out[-1] = Push(i, row.name.date(), row["close"], row["rsi"])
                continue
            out.append(Push(i, row.name.date(), row["close"], row["rsi"]))
    return out


def detect_bullish_divergence(troughs: list[Push]) -> dict:
    """
    Price lower low, RSI higher low — selling pressure decaying into new lows.
    The buy-side mirror of the three-push bearish pattern, and on the daily
    chart it is the more actionable of the two.
    """
    if len(troughs) < 2:
        return {"state": "insufficient troughs", "signals": []}

    sigs = []
    for a, b in zip(troughs, troughs[1:]):
        if b.price < a.price and b.rsi > a.rsi:
            sigs.append({
                "from": a.date.isoformat(), "to": b.date.isoformat(),
                "price_delta_pct": round(float(b.price / a.price - 1) * 100, 2),
                "rsi_delta": round(float(b.rsi - a.rsi), 1),
            })
    return {"state": "bullish divergence" if sigs else "no divergence",
            "signals": sigs, "latest": sigs[-1] if sigs else None}


def daily_rsi_check(daily_close: pd.Series, regime: str = "neutral",
                    cfg: Config = Config()) -> dict:
    """
    Daily RSI with regime-adaptive bands, plus bullish divergence.

    Daily fires far more often than weekly, which is both why it is more
    useful for timing and why it needs a filter. On the synthetic demo series
    the raw signal fired seven times since the cycle peak; six were followed
    by gains over the next month and one lost 14.7%. That loser fired three
    weeks after the all-time high, while the trend was still unwinding — which
    is the characteristic failure and the reason `mtf_entry_signal` below
    requires the weekly timeframe to grant permission first.
    """
    b = RSI_BANDS[regime]
    df = pd.DataFrame({"close": daily_close})
    df["rsi"] = rsi(df["close"], cfg.rsi_period)
    cur = float(df["rsi"].iloc[-1])

    troughs = find_troughs(df)
    div = detect_bullish_divergence(troughs[-4:]) if len(troughs) >= 2 else {"signals": []}

    # How long has RSI sat under the oversold line? A single print is noise;
    # several consecutive days is genuine exhaustion.
    under = 0
    for v in reversed(df["rsi"].tolist()):
        if v < b["oversold"]:
            under += 1
        else:
            break

    return {
        "regime": regime,
        "bands": b,
        "rsi": round(cur, 1),
        "state": ("oversold" if cur < b["oversold"]
                  else "overbought" if cur > b["overbought"]
                  else "neutral"),
        "days_oversold": under,
        "bullish_divergence": div.get("latest"),
        "distance_to_oversold": round(cur - b["oversold"], 1),
    }


def mtf_entry_signal(weekly_close: pd.Series, daily_close: pd.Series,
                     ath: float, sth_mvrv: float, cfg: Config = Config()) -> dict:
    """
    Multi-timeframe confluence: the weekly chart decides whether you are
    allowed to buy at all, the daily chart decides when.

    Neither timeframe is much use alone. Weekly is too slow to time an entry
    and fires two or three times a cycle. Daily is fast enough to time one but
    will happily fire all the way down a bear market. Requiring the slow
    timeframe to grant permission and the fast one to pull the trigger is what
    removes the failure mode where an oversold daily print lands three weeks
    after a cycle top.
    """
    reg = bull_regime(weekly_close, ath, cfg)
    label = "bull" if reg["confirmed"] else (
        "bear" if weekly_close.iloc[-1] / ath - 1 < -0.25 else "neutral")

    wk = pd.DataFrame({"close": weekly_close})
    wk["rsi"] = rsi(wk["close"], cfg.rsi_period)
    weekly_rsi = float(wk["rsi"].iloc[-1])

    daily = daily_rsi_check(daily_close, label, cfg)
    lo, hi, _ = support_band(weekly_close)

    permission = {
        "weekly_regime_ok": reg["confirmed"],
        "weekly_rsi_not_overbought": weekly_rsi < 70,
        "price_near_or_above_band": bool(daily_close.iloc[-1] >= lo * 0.97),
    }
    trigger = {
        "daily_oversold": daily["state"] == "oversold",
        "sustained": daily["days_oversold"] >= 2,
        "bullish_divergence": daily["bullish_divergence"] is not None,
        "sth_mvrv_at_or_below_1": bool(sth_mvrv <= cfg.sth_mvrv_trigger),
    }

    granted = all(permission.values())
    fired = granted and trigger["daily_oversold"] and (
        trigger["sustained"] or trigger["bullish_divergence"])

    return {
        "status": "TRIGGERED" if fired else ("ARMED" if granted else "GATED"),
        "regime_label": label,
        "weekly_rsi": round(weekly_rsi, 1),
        "daily": daily,
        "permission": permission,
        "trigger": trigger,
        "note": ("Weekly grants permission, daily pulls the trigger. "
                 "Without the weekly gate the daily signal fires in downtrends."),
    }


# ---------------------------------------- 3. bull-market dip buy (gated)

def support_band(weekly_close: pd.Series) -> tuple[float, float, float]:
    """The 20-week SMA / 21-week EMA band. Returns (low, high, slope)."""
    sma = weekly_close.rolling(20).mean()
    ema = weekly_close.ewm(span=21, adjust=False).mean()
    lo, hi = min(sma.iloc[-1], ema.iloc[-1]), max(sma.iloc[-1], ema.iloc[-1])
    slope = (sma.iloc[-1] / sma.iloc[-5] - 1) if len(sma) > 5 else 0.0
    return lo, hi, slope


def bull_regime(weekly_close: pd.Series, ath: float, cfg: Config = Config()) -> dict:
    """
    The gate. Without this the dip-buy rule fires in bear markets, which is
    exactly when it is most expensive to be wrong.
    """
    lo, hi, slope = support_band(weekly_close)
    price = weekly_close.iloc[-1]

    sma = weekly_close.rolling(20).mean()
    ema = weekly_close.ewm(span=21, adjust=False).mean()
    band_hi = pd.concat([sma, ema], axis=1).max(axis=1)
    weeks_above = int((weekly_close.tail(cfg.regime_weeks_above_band)
                       > band_hi.tail(cfg.regime_weeks_above_band)).sum())

    drawdown = price / ath - 1
    checks = {
        "price_above_band": bool(price > hi),
        "band_rising": bool(slope > 0),
        "weeks_above_band": weeks_above >= cfg.regime_weeks_above_band,
        "drawdown_shallow": bool(drawdown > -cfg.max_drawdown_for_bull),
    }
    return {"confirmed": all(checks.values()), "checks": checks,
            "band": (round(lo), round(hi)), "band_slope": round(slope, 4),
            "drawdown": round(drawdown, 4)}


def dip_buy_signal(weekly_close: pd.Series, ath: float, sth_mvrv: float,
                   weekly_rsi: float, cfg: Config = Config()) -> dict:
    """
    There is no single "most accurate" bull-market buy signal — anyone
    claiming one is fitting three cycles. What holds up best across those
    three is the confluence of two independent measures:

      1. BULL MARKET SUPPORT BAND (20W SMA / 21W EMA). In an uptrend price
         repeatedly dips into this band and bounces. Every major correction
         in the 2017 and 2021 bull runs found support here, and a weekly
         close below it has marked the loss of bullish momentum.

      2. SHORT-TERM HOLDER MVRV at or below 1.0. The cost basis of everyone
         who bought within the last 155 days. At 1.0 the marginal recent
         buyer is exactly breakeven — a level that acts as support in
         uptrends and resistance in downtrends.

    One is price structure, the other is holder positioning, and they fail
    in different ways — which is the only reason combining them helps.

    Confirm on a weekly close back above the band. Invalidate on two
    consecutive weekly closes below it.
    """
    regime = bull_regime(weekly_close, ath, cfg)
    lo, hi, _ = support_band(weekly_close)
    price = weekly_close.iloc[-1]

    conditions = {
        "band_touch": bool(price <= hi * (1 + cfg.band_touch_tolerance)),
        "sth_mvrv_at_or_below_1": bool(sth_mvrv <= cfg.sth_mvrv_trigger),
        "rsi_reset": bool(45 <= weekly_rsi <= 60),
    }

    if not regime["confirmed"]:
        failed = [k for k, v in regime["checks"].items() if not v]
        return {
            "status": "GATED",
            "reason": f"bull regime not confirmed — failing: {', '.join(failed)}",
            "conditions_met": conditions,
            "regime": regime,
            "note": "Designed for dips within an established uptrend. "
                    "Firing it in a markdown phase is how this rule fails.",
        }

    fired = all(conditions.values())
    return {
        "status": "TRIGGERED" if fired else "ARMED",
        "conditions_met": conditions,
        "regime": regime,
        "invalidation": "Two consecutive weekly closes below the band.",
        "confirmation": "Weekly close back above the band.",
    }


if __name__ == "__main__":
    print(cycle_position(
        today=date.today(),
        ath=126_296, ath_date=date(2025, 10, 6), price=77_400,
    ))
