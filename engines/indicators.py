"""
SmartTrades.AI — Indicators and entry ladders
=============================================

Correct implementations of the per-ticker indicators, plus the entry-ladder
logic behind the good / great / fantastic tiers.

Read this before adding indicators
----------------------------------
The temptation with a detail page is to stack gauges until it looks
authoritative. That manufactures false confidence, because most technical
indicators are transformations of the same price series and agreeing with each
other is what they do.

Of the five stock indicators here:

    MACD, RSI, StochRSI   all momentum derivatives of close. When they agree
                          that is ONE observation, not three. StochRSI is RSI
                          run through a stochastic — it is the most redundant
                          of the set and the fastest, so it mostly adds noise
                          and lead time in equal measure.
    OBV                   volume flow. Genuinely orthogonal: it can diverge
                          from price, which is the whole reason to carry it.
    Z-Score               distance from the mean in standard deviations.
                          Mean-reversion, not momentum. Also orthogonal.

So the panel groups them as momentum (one vote, three readings), flow, and
position. `independence_report()` measures this on the actual data rather than
asserting it, because correlation between them varies by regime.

The same applies to the Bitcoin bear set. MVRV Z-Score and the Realized Price
Oscillator are both built on (price - realized price). They will agree nearly
always, and that agreement carries no extra information.

An ambiguity worth resolving
----------------------------
"Stock RSI" is implemented here as Stochastic RSI, since it was listed
separately from RSI. "Z-Score" is implemented as a price z-score against a
rolling mean, not the Altman Z-Score — Altman already lives in `screeners.py`
as a solvency gate. If either reading is wrong, they are one-line swaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------- momentum set

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    line = f - s
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def stoch_rsi(close: pd.Series, period: int = 14,
              k: int = 3, d: int = 3) -> pd.DataFrame:
    """
    RSI passed through a stochastic. Ranges 0-1 and whipsaws far more than RSI.
    Fast, and mostly redundant against RSI — carry it for the crossover, not as
    independent evidence.
    """
    r = rsi(close, period)
    lo = r.rolling(period).min()
    hi = r.rolling(period).max()
    raw = (r - lo) / (hi - lo).replace(0, np.nan)
    kline = raw.rolling(k).mean()
    return pd.DataFrame({"k": kline, "d": kline.rolling(d).mean()})


# --------------------------------------------------- flow and position set

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-balance volume. The level is arbitrary and meaningless on its own; only
    its trend and its divergence from price carry information.
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def obv_divergence(close: pd.Series, ob: pd.Series, window: int = 60) -> dict:
    """
    The reason OBV earns its place: price making a new high while OBV does not
    means the advance is running on thinner participation.
    """
    c, o = close.tail(window), ob.tail(window)
    price_high = c.iloc[-1] >= c.max() * 0.995
    obv_high = o.iloc[-1] >= o.max() * 0.995
    price_low = c.iloc[-1] <= c.min() * 1.005
    obv_low = o.iloc[-1] <= o.min() * 1.005

    if price_high and not obv_high:
        state = "bearish divergence"
    elif price_low and not obv_low:
        state = "bullish divergence"
    else:
        state = "confirming"

    slope = float(np.polyfit(range(len(o)), o.values, 1)[0])
    return {"state": state, "obv_slope": slope,
            "trend": "accumulation" if slope > 0 else "distribution"}


def zscore(close: pd.Series, window: int = 60) -> pd.Series:
    """Standard deviations from the rolling mean. Mean reversion, not momentum."""
    mu = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    return (close - mu) / sd.replace(0, np.nan)


def money_flow_index(high, low, close, volume, period: int = 14) -> pd.Series:
    """Volume-weighted RSI. Needs volume, so it is crypto/equity only."""
    tp = (high + low + close) / 3
    flow = tp * volume
    up = flow.where(tp > tp.shift(), 0.0)
    dn = flow.where(tp < tp.shift(), 0.0)
    pos = up.rolling(period).sum()
    neg = dn.rolling(period).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def atr(high, low, close, period: int = 14) -> pd.Series:
    """
    Average true range. Directionless — it measures how much, never which way.
    In a bull market its two real uses are trailing stops (commonly 2.5-3x ATR)
    and spotting volatility expansion, which often accompanies a blowoff rather
    than a healthy trend.
    """
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ------------------------------------------------------------- BTC onchain

def mvrv_zscore(market_cap: pd.Series, realized_cap: pd.Series) -> pd.Series:
    """
    (Market cap - realized cap) / stdev(market cap). The classic cycle
    top/bottom gauge: historically above ~7 marks distribution zones and below
    ~0 marks accumulation zones, on four cycles.
    """
    diff = market_cap - realized_cap
    return diff / market_cap.expanding(min_periods=100).std()


def realized_price_oscillator(price: pd.Series, realized_price: pd.Series) -> pd.Series:
    """
    Spot as a percentage deviation from realized price — the aggregate on-chain
    cost basis. Equivalent to (MVRV - 1) * 100.

    Shares its numerator with `mvrv_zscore`, so the two agree by construction.
    Carry both only if you want the raw and normalised views; do not count them
    as two confirmations.
    """
    return (price / realized_price - 1) * 100


# ------------------------------------------------------------ entry ladder

@dataclass
class LadderConfig:
    good_hurdle: float = 0.12        # required forward return to bother
    great_discount: float = 0.22     # vs fair value
    fantastic_discount: float = 0.38
    # A price this far below fair value usually means the thesis broke rather
    # than the market being wrong. Confirmation is mandatory at that tier.
    require_confirmation_at_tier3: bool = True
    # How far BELOW a tier's computed price a support may sit and still be
    # treated as that tier's anchor. Without a floor, "nearest level at or
    # below" snaps every tier onto the same distant low. 10% is one ordinary
    # pullback; past that the level belongs to a different tier or to none.
    snap_window: float = 0.10


def entry_ladder(price: float, fair_value: float,
                 support_levels: list[float] | None = None,
                 # Defaults are UNKNOWN, not safe. They were True/False/0.0,
                 # and 0.0 > -5 is True, so a caller that passed nothing got
                 # "confirmation met" on three checks it had never supplied.
                 solvency_ok: bool | None = None,
                 insider_buying: bool | None = None,
                 estimate_revision_3m: float | None = None,
                 cfg: LadderConfig = LadderConfig()) -> dict:
    """
    Three tiers, each with what it assumes and what would make it a trap.

    The third tier is where capital gets destroyed, and the user's instinct
    about it is right: a price that good usually means something is breaking.
    Price alone cannot separate "cheap" from "impaired", so tier 3 carries an
    explicit confirmation gate rather than just a lower number.
    """
    sup = sorted(support_levels or [], reverse=True)

    tiers = []
    for name, disc, note in [
        ("good", cfg.good_hurdle,
         "Fair-value discount clears the hurdle rate. Start a position; "
         "expect to average down."),
        ("great", cfg.great_discount,
         "Meaningful discount, usually at a technical level or the low end of "
         "the historical multiple range. Add here."),
        ("fantastic", cfg.fantastic_discount,
         "Deep value or a broken thesis, and price cannot tell you which. "
         "Only act with confirmation."),
    ]:
        target = fair_value * (1 - disc)
        # Snap to the nearest support below the computed level when one exists;
        # round numbers and prior bases are where fills actually happen.
        #
        # But only when the support is NEAR the target. "The first level at or
        # below target" degenerates the moment fair value sits well above
        # spot: Kinsale's tiers compute to $1,674 / $1,484 / $1,180 against
        # supports at $326 / $300 / $290, and every tier snapped to $326 —
        # three identical prices labelled good, great and fantastic. A level
        # 80% below the target is not that target's anchor, it is a different
        # idea. Found by wiring real prices; invisible on generated ones,
        # where fair value always sat a tidy 15-70% above spot.
        anchor = next((s for s in sup
                       if target * (1 - cfg.snap_window) <= s <= target * 1.03),
                      None)
        lvl = anchor if anchor else target
        # Price may already be below a tier. That is a materially different
        # situation from waiting for it and must not read the same on screen.
        reached = price <= lvl
        tiers.append({
            "tier": name,
            "price": round(lvl, 2),
            "pct_from_spot": round((lvl / price - 1) * 100, 1),
            "discount_to_fv": round(disc * 100, 1),
            "anchored_on": "support level" if anchor else "fair-value discount",
            "status": "reached" if reached else "awaiting",
            "note": note,
        })

    # Tier 3 gate, in THREE values: met, not met, or not answerable.
    #
    # The two-valued version read `insider_buying or estimate_revision_3m > -5`
    # against defaults of False and 0.0 — and `0 > -5` is True, so the whole
    # clause collapsed to solvency alone and the page printed "Confirmation
    # met" on a name where two of the three checks had no data behind them.
    # On the tier that exists because a price this good usually means the
    # thesis broke, told to someone deciding whether to buy, that is the worst
    # place in the project for "unknown reads as safe".
    #
    # None means unknown, and unknown propagates: the OR is True if either arm
    # is True, False only if BOTH are known False, and unknown otherwise.
    def _or(a, b):
        if a is True or b is True:
            return True
        if a is None or b is None:
            return None
        return False

    est = None if estimate_revision_3m is None else estimate_revision_3m > -5
    either = _or(insider_buying, est)
    if solvency_ok is False:
        confirmed = False
    elif solvency_ok is None or either is None:
        confirmed = None
    else:
        confirmed = bool(either)

    missing = [n for n, v in (("solvency", solvency_ok),
                              ("insider buying", insider_buying),
                              ("estimate revisions", est)) if v is None]
    tiers[2]["confirmation"] = {
        "required": cfg.require_confirmation_at_tier3,
        "met": confirmed,                      # True / False / None
        "unconfirmable": confirmed is None,
        "missing_inputs": missing,
        "checks": {
            "solvency_intact": solvency_ok,
            "insider_buying": insider_buying,
            "estimates_not_collapsing": est,
        },
        "if_unmet": "Treat as a falling knife, not a discount. The tier exists "
                    "to be waited for, not automatically bought.",
        "if_unconfirmable": (
            "Not confirmed and not refuted: "
            + ", ".join(missing) + " has no source, so the check cannot be "
            "run. Treat it as unconfirmed — the tier's whole purpose is that "
            "price alone cannot tell cheap from impaired."),
    }
    return {"spot": price, "fair_value": fair_value, "tiers": tiers}


# --------------------------------------------------------- independence

def independence_report(frame: pd.DataFrame, window: int = 250) -> dict:
    """
    Measure, do not assume. Returns pairwise correlation of the indicator
    series and flags pairs above 0.8 as effectively one signal.

    Run this before adding an indicator to a panel. If the new series
    correlates above 0.8 with something already there, it is decoration.
    """
    tail = frame.tail(window).dropna()
    if len(tail) < 30:
        return {"error": "not enough overlapping history"}

    corr = tail.corr()
    redundant = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = float(corr.loc[a, b])
            if abs(c) >= 0.8:
                redundant.append({"pair": [a, b], "corr": round(c, 2)})

    return {
        "correlations": {f"{a}|{b}": round(float(corr.loc[a, b]), 2)
                         for i, a in enumerate(cols) for b in cols[i + 1:]},
        "redundant_pairs": redundant,
        "effective_signals": len(cols) - len(redundant),
        "note": "Pairs above 0.8 should count as one signal, not two.",
    }


# ------------------------------------------------------------ panel build

@dataclass
class Panel:
    momentum: dict = field(default_factory=dict)
    flow: dict = field(default_factory=dict)
    position: dict = field(default_factory=dict)


def stock_panel(df: pd.DataFrame) -> dict:
    """
    `df` needs close, high, low, volume on a daily index.

    Grouped by what is actually independent: momentum is one vote with three
    readings; flow and position are separate votes.
    """
    close, vol = df["close"], df["volume"]
    m = macd(close)
    sr = stoch_rsi(close)
    ob = obv(close, vol)
    z = zscore(close)
    r = rsi(close)

    ind = independence_report(pd.DataFrame({
        "rsi": r, "macd_hist": m["hist"], "stochrsi": sr["k"],
        "obv": ob.diff(), "zscore": z,
    }))

    return {
        "momentum": {
            "weight": "one signal, three readings",
            "rsi": round(float(r.iloc[-1]), 1),
            "macd_hist": round(float(m["hist"].iloc[-1]), 3),
            "macd_cross": "bullish" if m["macd"].iloc[-1] > m["signal"].iloc[-1] else "bearish",
            "stoch_rsi_k": round(float(sr["k"].iloc[-1]), 2),
            "stoch_rsi_state": ("overbought" if sr["k"].iloc[-1] > 0.8
                                else "oversold" if sr["k"].iloc[-1] < 0.2 else "neutral"),
        },
        "flow": {"weight": "independent", **obv_divergence(close, ob)},
        "position": {"weight": "independent",
                     "zscore": round(float(z.iloc[-1]), 2),
                     "state": ("extended" if z.iloc[-1] > 2
                               else "depressed" if z.iloc[-1] < -2 else "normal")},
        "independence": ind,
    }


def btc_panel(df: pd.DataFrame, regime: str,
              market_cap: pd.Series | None = None,
              realized_cap: pd.Series | None = None,
              realized_price: pd.Series | None = None) -> dict:
    """
    Regime-conditional, matching how the indicators actually behave.

    BEAR: MFI, MVRV Z-Score, realized price oscillator, RSI — across monthly,
          weekly and daily. Bear markets end on valuation and capitulation,
          which is what the on-chain pair measures; the higher timeframes are
          where those turn.

    BULL: RSI and ATR, daily. Bull markets are trend-following problems, not
          valuation ones. ATR is for trailing stops and for spotting the
          volatility expansion that tends to accompany a blowoff.
    """
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    if regime == "bull":
        a = atr(high, low, close)
        atr_pct = float(a.iloc[-1] / close.iloc[-1] * 100)
        a_med = float((a / close * 100).tail(250).median())
        return {
            "regime": "bull", "timeframe": "daily",
            "rsi": round(float(rsi(close).iloc[-1]), 1),
            "atr_pct": round(atr_pct, 2),
            "atr_vs_median": round(atr_pct - a_med, 2),
            "trailing_stop_3atr": round(float(close.iloc[-1] - 3 * a.iloc[-1])),
            "volatility_expansion": atr_pct > a_med * 1.5,
            "note": "ATR is directionless. Use it for stop distance and to spot "
                    "volatility expansion, never for direction.",
        }

    out = {"regime": "bear", "timeframes": {}}
    for label, rule in (("daily", "D"), ("weekly", "W"), ("monthly", "ME")):
        r_ = df.resample(rule).agg({"close": "last", "high": "max",
                                    "low": "min", "volume": "sum"}).dropna()
        if len(r_) < 20:
            continue
        out["timeframes"][label] = {
            "rsi": round(float(rsi(r_["close"]).iloc[-1]), 1),
            "mfi": round(float(money_flow_index(r_["high"], r_["low"],
                                                r_["close"], r_["volume"]).iloc[-1]), 1),
        }

    if market_cap is not None and realized_cap is not None:
        out["mvrv_z"] = round(float(mvrv_zscore(market_cap, realized_cap).iloc[-1]), 2)
    if realized_price is not None:
        out["realized_price_osc"] = round(
            float(realized_price_oscillator(close, realized_price).iloc[-1]), 1)
        out["shared_numerator_warning"] = (
            "MVRV Z-Score and the realized price oscillator are both built on "
            "(price - realized price). They agree by construction; count them "
            "as one signal.")
    return out
