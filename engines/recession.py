"""
SmartTrades.AI — Engine 7: Recession Radar
==========================================

Why this is split into two scores instead of one
------------------------------------------------
Recession indicators do not share a time horizon, and averaging them into a
single number destroys the only information that matters — whether trouble is
*coming* or *already here*.

    LEAD    yield curve, un-inversion clock, LEI, claims trend
            Fires 6-18 months ahead. Noisy. Produces false positives.

    STRESS  HY OAS, CCC-BB dispersion, financial conditions, Sahm
            Fires at or just before the event. Rarely wrong, rarely early.

The gap between them is the read. High lead with low stress is late-cycle:
the conditions are in place but the market has not repriced. High stress with
low lead means something broke that the curve never saw coming (2020).

A note on the indicator this engine was built around
----------------------------------------------------
HY OAS is a superb confirmer and a weak predictor. Spreads stay tight until
trouble is visible, then widen violently — they tell you a recession is
arriving, not that one is coming. Used as a forecast it produces the classic
error: spreads are tight, therefore nothing is wrong. In 2007 HY OAS sat near
250bp in June and passed 900bp within eighteen months.

So OAS carries most of the STRESS score and almost none of the LEAD score.
That is deliberate and should not be "fixed."

What moves first inside credit
------------------------------
CCC-BB dispersion turns before headline HY OAS. When the index is dominated by
BB paper that stays well bid, an index-level number can look calm while the
low-quality tail is already repricing. Dispersion is the early-warning line
within credit.

Momentum beats level. A 100bp widening in a month from tight levels is a bigger
signal than a stable 500bp, because the level tells you where you are and the
rate of change tells you where you are going.

Data sources
------------
FRED, all free, no key required for the series themselves:

    BAMLH0A0HYM2   HY OAS                      daily
    BAMLH0A1HYBB   BB OAS                      daily
    BAMLH0A3HYC    CCC & lower OAS             daily
    T10Y3M         10y-3m spread (NY Fed uses this one)   daily
    T10Y2Y         10y-2y spread               daily
    SAHMREALTIME   real-time Sahm rule         monthly
    NFCI           Chicago Fed financial conditions       weekly
    ICSA           initial claims              weekly
    USREC          NBER recession dates        monthly

IMPORTANT — as of April 2026 FRED serves only a rolling THREE-YEAR window for
every ICE BofA series, and directs users to ICE Data Indices for older history.
Percentile ranking against a 3-year window is close to meaningless when the
question is "how does this compare to a credit cycle." Two options: license the
history from ICE, or start caching the daily prints yourself now and build the
history forward. `data/oas_history.csv` is the cache for the second option.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

FRED = {
    "hy": "BAMLH0A0HYM2", "bb": "BAMLH0A1HYBB", "ccc": "BAMLH0A3HYC",
    "t10y3m": "T10Y3M", "t10y2y": "T10Y2Y",
    "sahm": "SAHMREALTIME", "nfci": "NFCI", "claims": "ICSA",
}

# Level buckets in percentage points, not basis points — FRED publishes pp.
OAS_BANDS = [
    (0.00, 3.00, "complacent", 0),
    (3.00, 4.00, "normal",     20),
    (4.00, 5.00, "caution",    45),
    (5.00, 7.00, "stress",     70),
    (7.00, 10.0, "severe",     88),
    (10.0, 99.0, "crisis",     100),
]


@dataclass
class Config:
    # Momentum thresholds — the part that actually carries signal.
    widening_1m_alert: float = 1.00      # +100bp in a month = regime change
    widening_3m_alert: float = 1.50
    dispersion_pctile_alert: float = 75.0
    sahm_trigger: float = 0.50

    # Dispersion sits in LEAD, not STRESS. It was in stress until a live run
    # showed the structural problem: dispersion at the 99.9th percentile
    # produced a stress score of 24.2, because the four coincident components
    # beside it were all near zero. That is not mixed evidence — it is
    # guaranteed. If dispersion genuinely leads headline HY OAS, then whenever
    # it is extreme the coincident credit measures MUST still be calm, so
    # averaging them buries the early signal every time by construction.
    # Same error the lead/stress split exists to prevent, one level down.
    lead_weights: dict = field(default_factory=lambda: {
        "curve_probit":       0.32,   # Estrella-Mishkin, 6-18m lead
        "uninversion_clock":  0.20,   # the re-steepening trigger
        "quality_dispersion": 0.20,   # the early line INSIDE credit
        "claims_trend":       0.16,
        "lei_trend":          0.12,
    })
    # Purely coincident. Every one of these fires at or just before the event.
    stress_weights: dict = field(default_factory=lambda: {
        "oas_level":            0.34,
        "oas_momentum":         0.30,   # rate of change beats level
        "sahm":                 0.24,
        "financial_conditions": 0.12,
    })
    # Dispersion is also surfaced on its own, so a weighted average can never
    # be the only place it appears.
    dispersion_warning_pctile: float = 85.0


def _lookback(s: pd.Series, days: int) -> float:
    """Value `days` calendar days ago, by date rather than bar count."""
    target = s.index[-1] - pd.Timedelta(days=days)
    prior = s.loc[:target]
    return float(prior.iloc[-1]) if len(prior) else float(s.iloc[0])


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ------------------------------------------------------------- indicators

def curve_probit(spread_10y3m: float) -> dict:
    """
    Estrella-Mishkin (1998) probit, the model behind the NY Fed's published
    recession probability:

        P(recession within 12m) = Phi(-0.5333 - 0.6629 * spread_10y3m)

    The 10y-3m version is the academically preferred one; 10y-2y is more
    widely quoted but forecasts slightly worse. The model knows nothing about
    QE, term premium compression, or a Fed balance sheet, all of which have
    been argued to distort the spread since 2010. Treat the number as one
    input, not an answer.
    """
    p = _norm_cdf(-0.5333 - 0.6629 * spread_10y3m)
    return {"spread": spread_10y3m, "probability": round(p * 100, 1),
            "inverted": spread_10y3m < 0,
            "note": "Probit output, not a forecast. n=8 recessions since 1976."}


def uninversion_clock(inversion_ended: date | None, today: date) -> dict:
    """
    The part most coverage gets backwards: recessions historically begin AFTER
    the curve re-steepens, not while it is inverted. Inversion says a recession
    is being priced; un-inversion — driven by the short end falling as the Fed
    starts cutting — says it is arriving.

    Typical lead from un-inversion to recession start is 6-12 months.
    """
    if inversion_ended is None:
        return {"state": "still inverted or never inverted", "score": 30}

    months = (today - inversion_ended).days / 30.44
    in_window = 6 <= months <= 12
    # Risk peaks inside the historical window and decays after it.
    if months < 6:
        score = 35 + months * 5
    elif months <= 12:
        score = 65 + (months - 6) * 4
    else:
        score = max(20.0, 89 - (months - 12) * 4)

    return {"months_since": round(months, 1), "in_historical_window": in_window,
            "score": round(min(score, 100.0), 1),
            "window": "6-12 months post un-inversion"}


def oas_state(hy: pd.Series, cfg: Config = Config()) -> dict:
    """Level, own-history percentile, momentum, and position vs the 200-day."""
    cur = float(hy.iloc[-1])
    band = next((n, s) for lo, hi, n, s in OAS_BANDS if lo <= cur < hi)

    # Date offsets, not positional — the engine must give the same answer on
    # daily or weekly input, and iloc[-22] silently means 22 weeks on the latter.
    ma200 = float(hy.rolling(200, min_periods=20).mean().iloc[-1])
    chg_1m = cur - _lookback(hy, 30)
    chg_3m = cur - _lookback(hy, 91)
    pctile = float((hy < cur).mean() * 100)

    # Momentum score. Widening from a tight base is the informative case.
    mom = np.clip(chg_1m / cfg.widening_1m_alert * 60
                  + chg_3m / cfg.widening_3m_alert * 40, 0, 100)

    return {
        "level": round(cur, 2), "band": band[0], "level_score": band[1],
        "percentile_in_window": round(pctile, 1),
        "vs_200d": round(cur - ma200, 2),
        "above_200d": bool(cur > ma200),
        "change_1m": round(chg_1m, 2), "change_3m": round(chg_3m, 2),
        "momentum_score": round(float(mom), 1),
        "caveat": "Percentile is against a 3-year FRED window, not a full cycle.",
    }


def quality_dispersion(ccc: pd.Series, bb: pd.Series) -> dict:
    """
    CCC minus BB. Turns before headline HY OAS, because an index dominated by
    well-bid BB paper can look calm while the low-quality tail reprices.
    """
    disp = (ccc - bb).dropna()
    cur = float(disp.iloc[-1])
    pctile = float((disp < cur).mean() * 100)
    chg_3m = cur - _lookback(disp, 91)
    return {"dispersion": round(cur, 2),
            "percentile_in_window": round(pctile, 1),
            "change_3m": round(chg_3m, 2),
            "score": round(float(np.clip(pctile, 0, 100)), 1),
            "widening": chg_3m > 0.5}


def sahm_state(sahm: float, cfg: Config = Config()) -> dict:
    """
    Triggers when the 3-month average unemployment rate rises 0.50pp above its
    trailing 12-month low. Historically coincident-to-slightly-lagging: it
    confirms a recession that has usually already started.

    Sahm herself has cautioned that post-pandemic labour supply swings can push
    the rule up without the demand collapse it was built to detect, so a
    trigger is evidence rather than proof.
    """
    return {"value": round(sahm, 2), "triggered": sahm >= cfg.sahm_trigger,
            "distance": round(cfg.sahm_trigger - sahm, 2),
            "score": round(float(np.clip(sahm / cfg.sahm_trigger * 100, 0, 100)), 1)}


# -------------------------------------------------------------- composite

def composite(lead_parts: dict, stress_parts: dict,
              dispersion: dict | None = None,
              cfg: Config = Config()) -> dict:
    """
    Two scores, never one. The divergence between them is the actual output.

    `dispersion` is passed separately as well as being weighted into lead, so
    that an extreme reading is reported in its own right rather than only as
    one fifth of an average. A weighted component can be diluted; a flag
    cannot.
    """
    lead = sum(lead_parts[k] * w for k, w in cfg.lead_weights.items())
    stress = sum(stress_parts[k] * w for k, w in cfg.stress_weights.items())
    gap = lead - stress

    # Standalone escalation. Requires BOTH a high percentile and active
    # widening — a high level that is compressing is a late signal, not an
    # early one, and firing on level alone would make this noisy.
    warn = None
    if dispersion:
        pct = dispersion.get("percentile_in_window", 0)
        if pct >= cfg.dispersion_warning_pctile and dispersion.get("widening"):
            warn = {
                "flag": "CREDIT EARLY WARNING",
                "dispersion": dispersion.get("dispersion"),
                "percentile": pct,
                "change_3m": dispersion.get("change_3m"),
                "read": ("The low-quality tail is repricing while the index "
                         "still looks calm. CCC-BB turns before headline HY "
                         "OAS, so a calm headline here is expected rather "
                         "than reassuring."),
            }

    if gap >= 22 and stress < 25:
        read = ("Late cycle, unpriced. Conditions for a recession are in place "
                "and credit has not repriced. Historically the most dangerous "
                "configuration to be complacent in, and the least dramatic to "
                "look at.")
        posture = "Reduce risk gradually. Nothing is broken yet."
    elif lead >= 45 and stress >= 45:
        read = "Both timeframes agree. Repricing is underway."
        posture = "Defensive. The move has started."
    elif gap <= -20 and stress >= 45:
        read = ("Stress without a lead signal — a shock the curve never saw, "
                "as in 2020. Fast onset, no warning period.")
        posture = "React, do not forecast."
    elif lead < 35 and stress < 25 and abs(gap) < 22:
        read = "Mid-cycle. Neither timeframe is flagging."
        posture = "Normal risk."
    else:
        read = "Mixed. No clean configuration."
        posture = "Watch the gap."

    if warn:
        posture = f"{posture} Credit early warning is active — see the flag."

    return {
        "lead_score": round(lead, 1), "stress_score": round(stress, 1),
        "divergence": round(gap, 1), "read": read, "posture": posture,
        "credit_early_warning": warn,
        "lead_components": {k: round(v, 1) for k, v in lead_parts.items()},
        "stress_components": {k: round(v, 1) for k, v in stress_parts.items()},
    }


# ------------------------------------------------------------------ notes
#
# Base rates, which every recession dashboard should state and almost none do:
#
#   The US economy is in recession roughly 15% of months. A model that always
#   says "no recession" is right about 85% of the time. Any indicator must beat
#   that, and most published hit rates quietly do not.
#
#   There have been 8 recessions since 1976. Every claim about lead times rests
#   on 8 observations, several of which had idiosyncratic causes (1980 credit
#   controls, 2020 pandemic). This is a small-sample problem that no amount of
#   indicator stacking solves.
#
#   Precision matters more than recall here. Calling 10 of the last 3 recessions
#   is expensive: you sit in cash through the expansion that pays for
#   everything. Prefer a late, high-confidence signal to an early, noisy one.
