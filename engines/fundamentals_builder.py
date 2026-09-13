"""
SmartTrades.AI — building Fundamentals from EDGAR plus a price series
=====================================================================

`free_sources.load_fundamentals` was a sketch that filled one of 44 fields, so
every screen returned zero rows in strict mode. This fills the rest.

The split, measured rather than assumed:

    ~20 fields   EDGAR alone. All 11 quality gates derive from XBRL.
    ~16 fields   need a price series — every "vs its own 5-10y history"
                 measure, which is the core of all three screens.
    ~3 fields    no free source. Listed at the bottom with their defaults and
                 why the default is safe.

Prices come from yfinance here rather than Polygon. Polygon grouped-daily is
the right tool for the dark pool tape denominator (one call returns every US
ticker for a date) but the wrong shape for this: the screens need 10 years of
history for ONE ticker at a time, which is a per-symbol call either way, and
yfinance needs no key.

The one trap worth repeating
----------------------------
`fp == "FY"` does not mean annual. A 10-K tags its quarterly comparatives
fp="FY", form="10-K" as well, so filtering on it alone mixes 90-day and 365-day
periods in the same series. `free_sources._is_annual` checks period LENGTH via
the `start` field. Any new extraction path must go through it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from engines import free_sources as fs
from engines.screeners import Fundamentals

# A FLAT WACC is not a minor input to a reverse DCF — it is the thing being
# inverted. A two-point error moves implied growth by ~4.5 points, which is the
# same size as the signal, since delivered growth for most names sits in the
# mid single to low double digits.
#
# Worse, the error has a direction. A flat 9% overcharges a low-beta utility
# (true ~5.5%), so the model reads 8.6% embedded growth where the price really
# embeds about -1%, the gap collapses and the utility screens EXPENSIVE. The
# same 9% undercharges high-beta software (true ~11.5%), the gap widens and it
# screens CHEAP. That is backwards for a value screen, and it tilts toward
# exactly the names whose valuations depend most on the discount rate.
#
# Sector buckets are not CAPM, but they remove most of the systematic tilt for
# nothing, off the SIC sector already computed. Floored at 4% so the
# `r <= terminal_growth` guard never fires on a low-beta name.
# Cost of EQUITY, not WACC. For a financial, WACC is close to meaningless —
# deposits and policy float are raw material, not financing, which is the same
# argument that rules ROIC out. The hurdle a bank's ROE must clear is what
# equity holders require, and 8.5% (the financial WACC bucket) is a point or
# two light for that.
SECTOR_COST_OF_EQUITY = {
    "depository": 10.5,
    "broker": 11.0,
    "insurer": 9.5,
    "manager": 11.5,
    "fee_based": 10.0,
    "financial": 10.5,          # fallback when the subtype is unresolved
}

# The explicit SIC set. `financial` as a sector label spans 6000-6499 and
# 6700-6770, which sweeps in 6199 "Finance services" — MicroStrategy, IREN,
# Coinbase, Circle, Nubank, 24 of the 176 — and 6770 blank-check SPACs.
# Membership here is necessary but NOT sufficient; a witness must agree.
FINANCIAL_SIC = {
    "depository": {6021, 6022, 6029, 6035, 6036, 6099, 6141, 6163},
    "broker":     {6211, 6221},
    "manager":    {6282},
    "insurer":    {6311, 6321, 6324, 6331, 6351, 6361, 6399},
    # Fee businesses wearing someone else's SIC. 6200 is "security and
    # commodity brokers" but holds the EXCHANGES — CME, ICE, CBOE, Nasdaq —
    # which earn transaction and data fees and carry no interest income, so a
    # dealer's witness rejected them. 6411 is "insurance agents and brokers":
    # Aon, Marsh and Willis earn commissions and never underwrite, so an
    # underwriter's witness rejected them too. In both cases the witness was
    # right and the sub-bucket assignment was wrong.
    "fee_based":  {6200, 6411},
}

# The corroborating witness. A depository with no deposits and no interest
# income is not a depository whatever its SIC says. Measured: real banks carry
# both; every 6199 name carries neither. The witness is PER SUBTYPE — a single
# global "deposits and interest" rule would also reject Interactive Brokers,
# every asset manager and every insurer, all of which are legitimately in
# scope. Coinbase and Circle are the case that proves the pairing matters:
# they carry interest income and revenue, so a standalone broker witness would
# admit them, and only SIC membership excludes them.
FINANCIAL_WITNESS = {
    "depository": lambda w: w["deposits"] and w["interest_income"],
    "broker":     lambda w: w["interest_income"] and w["revenue"],
    "manager":    lambda w: w["revenue"] and not w["deposits"],
    "insurer":    lambda w: w["premiums_earned"],
    # Fees only. Requiring neither deposits nor premiums is the point: these
    # businesses have neither by construction.
    "fee_based":  lambda w: w["revenue"] and not w["deposits"],
}


# Every SIC in scope. 6199 and 6770 are deliberately absent.
IN_SCOPE_SIC = set().union(*FINANCIAL_SIC.values())
MANAGER_SIC = FINANCIAL_SIC["manager"]
# Used only to label a name that is in scope by SIC but whose filings
# corroborate nothing — it is reported, not screened.
SIC_HINT = {c: b for b, codes in FINANCIAL_SIC.items() for c in codes}


def financial_scope(sic: int, facts: dict) -> tuple[str, bool]:
    """
    SIC decides SCOPE. The filer's own reporting decides the SUB-BUCKET.

    Three rounds of SIC-led assignment produced three rounds of the same error:
    6200 "security and commodity brokers" holds the EXCHANGES, 6411 "insurance
    agents and brokers" holds commission brokers who never underwrite, and 6211
    holds BlackRock and SEI, who are asset managers. Each time the witness was
    right and the mapping was wrong, so the mapping is now derived from the
    witness rather than checked against it.

    SIC is still what decides whether a name is in scope at all — it is the
    only thing that excludes 6199 "Finance services", which holds MicroStrategy,
    IREN, Coinbase and Circle alongside genuine lenders, and 6770 blank cheques.

    Within the fee businesses the witness cannot distinguish an asset manager
    from an exchange: both earn fees and carry neither deposits nor premiums.
    SIC breaks that tie, which is the one place it still assigns.
    """
    if sic not in IN_SCOPE_SIC:
        return "", False

    ug = facts.get("facts", {}).get("us-gaap", {})
    present = {c: any(t in ug for t in fs.TAG_CHAINS[c])
               for c in ("deposits", "interest_income", "premiums_earned", "revenue")}

    # MATERIALITY, not presence. Franklin Resources carries the deposits tag
    # with a value of zero, and a presence test read it as a bank. Ameriprise
    # genuinely holds $34bn of deposits and $1.4bn of premiums against $191bn
    # of assets and $19bn of revenue — it HAS a bank and an insurer without
    # BEING either. Real depositories run deposits above half of assets
    # (JPMorgan 57.8%, Schwab 52.1%) and real underwriters run premiums at most
    # of revenue (Travelers 89.9%), so a 20% floor separates them cleanly with
    # room to spare.
    _dep = _last(_annual(facts, "deposits"))
    _ass = _last(_annual(facts, "assets"))
    _pre = _last(_annual(facts, "premiums_earned"))
    _rev = _last(_annual(facts, "revenue"))
    dep_share = (_dep / _ass) if _ass else 0.0
    prem_share = (_pre / _rev) if _rev else 0.0

    if dep_share >= 0.20 and present["interest_income"]:
        return "depository", True
    if prem_share >= 0.20:
        return "insurer", True
    if present["interest_income"] and present["revenue"]:
        return "broker", True
    if present["revenue"]:
        return ("manager" if sic in MANAGER_SIC else "fee_based"), True

    # In the SIC set but nothing in the filings corroborates any sub-bucket.
    return SIC_HINT.get(sic, ""), False



SECTOR_WACC = {
    "utility":   6.0,
    "reit":      6.5,
    "financial": 8.5,
    "general":   9.0,
}
ASSUMED_WACC = 9.0          # fallback only; prefer SECTOR_WACC
MIN_WACC = 4.0
ASSUMED_TAX = 0.21


def _annual(facts: dict, field: str, as_of: date | None = None,
            sink: dict | None = None) -> pd.Series:
    """
    One value per fiscal year end, first-reported, annual periods only.

    `sink` collects each field's extraction attrs. Converting the DataFrame to
    a Series drops `df.attrs`, which is how `multi_unit` came to be set and
    never read — the same dead-code shape as `sector` being read and never set.
    """
    df = fs.extract_series(facts, field, as_of=as_of, annual_only=True)
    if sink is not None:
        sink[field] = dict(df.attrs)
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["val"].values, index=pd.to_datetime(df["end"])).sort_index()


def _cagr(s: pd.Series, years: int) -> float:
    if len(s) < 2:
        return 0.0
    span = min(years, len(s) - 1)
    first, last = float(s.iloc[-1 - span]), float(s.iloc[-1])
    if first <= 0 or last <= 0:
        return 0.0
    return ((last / first) ** (1 / span) - 1) * 100


def _safe(n, d, pct: bool = True, default: float = 0.0) -> float:
    try:
        if d in (0, None) or (isinstance(d, float) and not np.isfinite(d)):
            return default
        v = float(n) / float(d)
        return v * 100 if pct else v
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def _last(s: pd.Series, default: float = 0.0) -> float:
    return float(s.iloc[-1]) if len(s) else default


def reverse_dcf_growth(enterprise_value: float, fcf: float, wacc: float,
                       years: int = 10, terminal_growth: float = 2.5,
                       lo: float = -20.0, hi: float = 60.0) -> float | None:
    """
    Solve for the growth rate the current price already embeds.

    This was a placeholder reading 0.0 for every name, and it is a SCORED
    component — so the value screen's expectations gap was a constant, and the
    detail panel read "Reverse-DCF implies 0.0% growth against 16.5%
    delivered" on all 37 rows. The gap and upside columns were both derived
    from it.

    It needs no data the builder does not already have: enterprise value,
    trailing free cash flow and an assumed cost of capital. Ten explicit years
    growing at g, then a terminal value at `terminal_growth`, discounted at
    `wacc`. Bisect for the g that reproduces the observed EV.

    Returns None rather than a number when the inputs cannot support one —
    negative FCF has no growth rate that justifies any positive value.
    """
    if not enterprise_value or enterprise_value <= 0 or not fcf or fcf <= 0:
        return None
    r, tg = wacc / 100.0, terminal_growth / 100.0
    if r <= tg:
        return None

    def pv(g):
        g = g / 100.0
        total, cf = 0.0, fcf
        for t in range(1, years + 1):
            cf *= (1 + g)
            total += cf / (1 + r) ** t
        terminal = cf * (1 + tg) / (r - tg)
        return total + terminal / (1 + r) ** years

    if pv(lo) > enterprise_value or pv(hi) < enterprise_value:
        return None                                 # price outside the solvable band
    for _ in range(60):
        mid = (lo + hi) / 2
        if pv(mid) < enterprise_value:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def _near_value(series: pd.Series, when, tol: int = 10):
    """
    Value at the period nearest `when`, within `tol` days.

    Fiscal year-ends drift by days across series, so an exact `.get(when)`
    misses and the caller silently falls back — that is how roic_5y became a
    one-year figure labelled as a five-year mean for 75 of 190 names.
    """
    if series is None or series.empty:
        return None
    hit = series.index[np.abs((series.index - when).days) <= tol]
    return float(series.loc[hit[-1]]) if len(hit) else None


def denominator_reliability(value: float, terms: dict | None = None,
                            residual_of: tuple | None = None,
                            cancellation_floor: float = 0.25,
                            materiality_floor: float = 0.05) -> dict:
    """
    Flag a denominator at CONSTRUCTION, rather than bounding the ratio after it
    blows up.

    Four bounds were added one at a time, each after a different metric blew up
    individually: ROIC at 100%, EV/EBIT at 150x, revenue at 2% of assets,
    min_den at $10m. Testing whether one rule unified them showed it does not —
    but two do, and they are distinguishable by HOW the denominator is built.

    CANCELLATION — a signed sum whose result is small relative to its terms.
        Invested capital is equity + debt - cash. Otis nets -$5.35bn of equity
        against $7.12bn of debt for $0.66bn of capital: c = 0.10.
        This form is necessary but NOT sufficient, and that was the surprise.
        Trex scores 1.08 — higher than Microsoft's 1.04 — and still published
        a 116.9% ROIC. Cancellation explains the negative-equity cases and says
        nothing about the rest.

    MATERIALITY — a residual of larger quantities, approaching zero.
        EBIT is revenue minus costs, so EV/EBIT explodes as the operating
        margin approaches zero. CoStar at a 2.2% margin published 2,998x;
        Accenture at 14.7% publishes 10.3x. The separation here IS clean.

    The second is a CAUSE-level check where a 150x cap is symptom-level:
    CoStar's 2,998x is not an expensive stock, it is a 2.2% margin, and an
    output bound cannot tell a real 150x from an artifact one while a margin
    floor can — and can say so where it is read.

    A new ratio inherits the check from how its denominator is built, rather
    than needing someone to notice it blew up first.
    """
    out = {"value": value, "reliable": True, "reasons": []}

    if terms:
        largest = max((abs(v) for v in terms.values()), default=0.0)
        c = abs(value) / largest if largest else 0.0
        out["cancellation_ratio"] = round(c, 3)
        out["terms"] = {k: round(v, 2) for k, v in terms.items()}
        if largest and c < cancellation_floor:
            out["reliable"] = False
            out["reasons"].append(
                f"cancellation: {value:,.0f} survives from terms up to "
                f"{largest:,.0f} (ratio {c:.2f})")

    if residual_of:
        base, label = residual_of
        m = abs(value) / abs(base) if base else 0.0
        out["materiality"] = round(m, 4)
        if base and m < materiality_floor:
            out["reliable"] = False
            out["reasons"].append(
                f"materiality: {value:,.0f} is {m:.1%} of {label} — a residual "
                "near zero, so any ratio on it describes the denominator")

    return out


def _ratio(num: pd.Series, den: pd.Series, pct: bool = True,
           default: float = 0.0, as_of: date | None = None,
           max_stale_days: int = 550,
           min_den: float = 0.0) -> tuple[float, bool]:
    """
    Ratio at the latest period both series cover — and only if that period is
    still current. Returns (value, stale).

    Two failures, in order. Taking _last() of each series independently gave
    Apple's gross margin as FY2025 gross profit over FY2017 revenue: 85.15%.
    Intersecting the indices fixed the mismatch but introduced the subtler
    version: Amazon's gross profit series ends in 2009, so the intersection is
    2009 and the margin came back 22.57% — internally consistent, sixteen years
    old, with nothing in the value saying so. It would have rendered.

    This is the same shape as `_align`'s unlimited forward-fill and the
    count-only EV guard. Correctness at an arbitrary historical period is not
    correctness. Stale returns the default and the flag.
    """
    if num is None or den is None or num.empty or den.empty:
        return default, True
    common = num.index.intersection(den.index)
    if common.empty:
        return default, True
    i = common.max()
    ref = pd.Timestamp(as_of or date.today())
    if (ref - i).days > max_stale_days:
        return default, True
    d = float(den.loc[i])
    if d == 0 or not np.isfinite(d):
        return default, True
    # A ratio against a near-zero denominator is arithmetic, not information.
    # Archer Aviation published a 33.33% gross margin on essentially no revenue
    # and an FCF margin of -170,566%, with nothing flagged.
    if min_den and abs(d) < min_den:
        return default, True
    v = float(num.loc[i]) / d
    return (v * 100 if pct else v), False


def _fill_integrity(name: str, result: pd.Series, filled: pd.Series,
                    filled_label: str, max_filled_share: float = 0.5) -> list:
    """
    THIRD invariant: no derived quantity may be computed from a zero-filled
    input without declaring itself unavailable.

    Freshness sees a concept that STOPPED. `_derivation_integrity` sees a join
    that collapsed. Neither sees a concept that NEVER STARTED whose absence is
    filled with a value that flatters the result — and the fill is silent
    precisely because the arithmetic succeeds.

    NextEra tags no capex concept at all, so `ocf - capex.fillna(0)` set free
    cash flow equal to operating cash flow: a +48.4% FCF margin and 10 of 10
    positive years, for a utility that spends ~$25bn a year on capital and is
    persistently free-cash-flow negative. The quality screen scored the
    fabrication as a strength. Five more names did the same, including BTE at
    100.3%, which would mean free cash flow equals revenue.

    The rule the debt intersection already learned, applied to subtraction.
    """
    if result is None or result.empty:
        return []
    n = len(result)
    missing = int(filled.isna().sum()) if filled is not None else n
    if missing == 0:
        return []
    share = missing / max(n, 1)
    if share >= max_filled_share:
        return [f"{name} computed with {filled_label} absent for {missing} of "
                f"{n} periods ({share:.0%}) — the result is fabricated, not "
                "merely incomplete"]
    return [f"{name}: {filled_label} absent for {missing} of {n} periods — "
            "those periods overstate the result"]


def _derivation_integrity(derived: dict, stale_days: int = 450) -> list:
    """
    Assert each derived quantity is no more absent, and no more stale, than the
    freshest input it was built from.

    Freshness checks a concept against revenue. This checks the JOIN. NVIDIA's
    debt concepts every read lag 0 while total debt came out empty, because an
    optional component that died in 2017 vetoed the intersection — a failure
    invisible to any per-concept measure.
    """
    out = []
    for name, (result, inputs) in derived.items():
        live = {k: v for k, v in inputs.items()
                if v is not None and not v.empty}
        if not live:
            continue
        freshest_in = max(v.index.max() for v in live.values())

        if result is None or result.empty:
            out.append(f"{name} is empty though {', '.join(live)} have data "
                       f"through {freshest_in.date()} — composition failure, "
                       "not a data gap")
            continue

        lag = (freshest_in - result.index.max()).days
        if lag > stale_days:
            out.append(f"{name} ends {result.index.max().date()} but its inputs "
                       f"run to {freshest_in.date()} ({lag}d) — a component is "
                       "truncating the join")
    return out


def _period_mismatch_days(*series: pd.Series) -> int:
    """Gap between the newest period any input covers and the newest they share."""
    live = [s for s in series if s is not None and not s.empty]
    if len(live) < 2:
        return 0
    newest_any = max(s.index.max() for s in live)
    common = live[0].index
    for s in live[1:]:
        common = common.intersection(s.index)
    if common.empty:
        # A sentinel in a field the gate treats as a day count and the UI
        # prints. MPT rendered "99999". Use a real, large-but-plausible span.
        return int((newest_any - min(s.index.min() for s in live)).days)
    return int((newest_any - common.max()).days)


def _altman(p: dict, market_equity: float | None = None) -> float:
    """
    Altman Z.

    With a market cap, the original 1968 manufacturing model:
        1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5,  X4 = market equity / total liabilities
    Distress below 1.81, safe above 2.99.

    Without one, Altman's Z' private-firm revision, which substitutes book
    equity AND re-fits every coefficient:
        0.717 X1 + 0.847 X2 + 3.107 X3 + 0.420 X4 + 0.998 X5
    Distress below 1.23, safe above 2.90.

    Mixing the two — original coefficients with book equity — is the common
    error and understates Z for anything trading above book.
    """
    ta = p.get("ta", 0)
    if not ta:
        return 99.0
    x1, x2 = p["wc"] / ta, p["re"] / ta
    x3, x5 = p["ebit"] / ta, p["sales"] / ta
    tl = p["tl"] or 1.0
    if market_equity:
        return float(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * (market_equity / tl) + 1.0 * x5)
    return float(0.717 * x1 + 0.847 * x2 + 3.107 * x3
                 + 0.420 * (p["book_equity"] / tl) + 0.998 * x5)


# ------------------------------------------------------- dividend record

def split_adjust(dps: pd.Series, splits: pd.Series | None = None,
                 drop_threshold: float = 0.60, kind: str = "per_share") -> pd.Series:
    """
    Put a dividends-per-share series on a single split-adjusted basis.

    XBRL reports DPS exactly as filed and never restates it for splits. Apple
    runs $3.00 in 2019 then $0.795 in 2020 — the 4:1 split, not a 73% cut.
    Unadjusted, `dividend_record` reads it as a cut, and a cut inside ten years
    is gate-disqualifying.

    `splits` must come from an authoritative source (yfinance's split history).
    There is deliberately NO ratio heuristic here. It was tried and it failed
    in the worst possible direction: 3M's real 2024 halving ($6.00 -> $2.86)
    was inferred as a 2:1 split and adjusted away, turning a disqualifying cut
    into an unbroken record. A 2:1 split and a 50% cut are numerically
    identical in a DPS series — the information needed to separate them is not
    in the series. Guessing is worse than declaring ignorance.

    With no split data, suspicious drops are FLAGGED by `dividend_record`
    rather than adjusted.

    `kind` sets the DIRECTION, and getting it backwards is silent:

        "per_share"  dividends, EPS — a split DIVIDES the per-share figure, so
                     prior values are divided to reach today's basis.
        "count"      share counts — a split MULTIPLIES the count, so prior
                     values are multiplied.

    Share counts were never adjusted at all. NVIDIA steps 628m -> 2.535bn
    across the 2021 4:1 and 2.494bn -> 24.804bn across the 2024 10:1, so
    `share_count_cagr_5y` read +108.1% a year and the gate
    `share_count_cagr_5y > 0.5` excluded the most aggressive buyback names in
    the market as serial diluters — 126 of 190 on quality, the single largest
    business gate. It also poisoned `shares_hist`, so NVIDIA's implied market
    cap read $0.01T in 2021 against ~$5.3T today, making EV/EBIT and its
    percentile incomparable across any split date.
    """
    if dps is None or dps.empty or splits is None or splits.empty:
        return dps
    if kind not in ("per_share", "count"):
        raise ValueError(f"kind must be per_share or count, got {kind!r}")

    out = dps.astype(float).copy()
    for when, ratio in splits.items():
        when = pd.Timestamp(when)
        if when.tzinfo is not None:
            when = when.tz_localize(None)
        if ratio and ratio > 0:
            prior = out.index < when
            out.loc[prior] = (out.loc[prior] / ratio if kind == "per_share"
                              else out.loc[prior] * ratio)
    return out


def suspicious_drops(dps: pd.Series, drop_threshold: float = 0.60) -> list:
    """
    Year-on-year falls large enough to be either a split or a real cut.

    Reported, never silently resolved. Which one it is decides whether the
    company passes or fails the dividend screen outright, so an unverified
    guess is not an acceptable input to that decision.
    """
    out = []
    for i in range(1, len(dps)):
        prev, cur = float(dps.iloc[i - 1]), float(dps.iloc[i])
        if prev > 0 and cur > 0 and cur / prev < drop_threshold:
            out.append({"at": dps.index[i].date().isoformat(),
                        "from": round(prev, 4), "to": round(cur, 4),
                        "ratio": round(prev / cur, 2)})
    return out


def drop_dividend_outliers(s: pd.Series, low: float = 0.45,
                           high: float = 2.5) -> pd.Series:
    """
    Remove values roughly a quarter the size of their neighbours.

    Third instance of one failure, on a different date offset each time: J&J
    one day off, Eaton two months off, Cognex in early October against a
    December year end. The calendar filter caught the first two and cannot
    catch the third, because an October date on a December fiscal year is not
    obviously off-cycle.

    The invariant that holds across all three is MAGNITUDE, not date: a DPS
    value roughly a quarter of its neighbours is a quarterly figure whatever
    date it carries. Cognex interleaves $0.065 on 2022-10-02 beside $0.265 on
    2022-12-31 and reads a false cut from it.

    Compared against the running median so a genuine halving — 3M in 2024 — is
    not mistaken for a quarterly straggler: a real cut lands near 0.5, a
    quarterly figure near 0.25.

    SPECIALS are the third category and the filter is symmetric for them.
    Cognex shows $2.2250 in 2020 against $0.2450 in 2021, a ratio of 37 — a
    regular dividend bundled with a special, which is neither a cut nor a
    split and which magnitude-below alone cannot separate from a cut. A value
    far ABOVE the running median is a bundled special; the following year is
    not a reduction.
    """
    if len(s) < 4:
        return s

    # Iterate. A single pass catches only the isolated stragglers, because a
    # centred rolling median includes the neighbouring stragglers themselves
    # and is dragged down at exactly the positions that matter — on Cognex,
    # where they alternate, one pass caught two of four.
    cur, dropped, _was_high = s, [], {}
    for _ in range(4):
        if len(cur) < 4:
            break
        med = cur.rolling(5, center=True, min_periods=3).median()
        rel = cur / med.replace(0, np.nan)
        too_low, too_high = (rel < low), (rel > high)
        keep = (~(too_low | too_high)).fillna(True)
        for d, v in cur[~keep].items():
            _was_high[f"{d.date()}={v:.4g}"] = bool(too_high.get(d, False))
        if keep.all():
            break
        dropped += [f"{d.date()}={v:.4g}" for d, v in cur[~keep].items()]
        cur = cur[keep]

    if dropped:
        print(f"  [info] dropped {len(dropped)} dividend outlier(s): "
              f"{', '.join(dropped[:4])}")
    # Attribute the drops, so "how many were bundled specials" is answerable
    # without a re-run. There was no field, so the question could not be asked.
    cur.attrs["outliers_dropped"] = len(dropped)
    cur.attrs["specials_dropped"] = sum(1 for d in dropped if _was_high.get(d, False))
    return cur


def drop_offcycle_periods(s: pd.Series, tol_days: int = 45) -> pd.Series:
    """
    Remove facts whose period end sits nowhere near the filer's own fiscal
    year end.

    Eaton reads "cut dividend 9y ago" and has never cut: its annual series
    rises monotonically from $2.40 to $4.16. The damage is two stray points,
    $0.49 dated 2014-02-26 and $0.60 dated 2017-02-22 — quarterly declared
    amounts carrying an ANNUAL-LENGTH period context, so `_is_annual` admits
    them. The +/-7 day dedup tolerance cannot merge them because they sit two
    MONTHS off the December year end rather than a few days.

    The signature is exactly that: a period end far from where this filer ends
    its year. Take the modal month-day across the series and drop the outliers.
    """
    if len(s) < 4:
        return s
    doy = pd.Series([d.month * 31 + d.day for d in s.index], index=s.index)
    modal = int(doy.mode().iloc[0])
    # Wrap-aware distance, so a January year end is not "far" from December.
    dist = (doy - modal).abs()
    dist = dist.where(dist <= 186, 372 - dist)
    keep = dist <= tol_days
    if (~keep).any():
        dropped = [d.date().isoformat() for d in s.index[~keep]]
        print(f"  [info] dropped {len(dropped)} off-cycle period(s) "
              f"{', '.join(dropped[:3])} — not near this filer's year end")
    return s[keep]


def dividend_record(dps: pd.Series) -> dict:
    """
    Consecutive years of increases, and years since the last cut.

    A cut inside ten years is disqualifying regardless of how healthy the
    company currently looks — see 3M, which raised for 64 years and then cut in
    2024. Rounding to the cent avoids counting float noise as a raise.
    """
    if len(dps) < 2:
        # Same key set as the main path — a partial dict here raised KeyError
        # downstream for every non-dividend-payer.
        return {"streak": 0, "years_since_cut": 99, "cagr_5y": 0.0,
                "cagr_3y": 0.0, "split_ambiguous": False, "suspicious_drops": []}

    v = dps.round(4)
    streak, cut_idx = 0, None
    for i in range(len(v) - 1, 0, -1):
        if v.iloc[i] > v.iloc[i - 1]:
            streak += 1
        else:
            if v.iloc[i] < v.iloc[i - 1] and cut_idx is None:
                cut_idx = i
            break
    if cut_idx is None:
        for i in range(len(v) - 1, 0, -1):
            if v.iloc[i] < v.iloc[i - 1]:
                cut_idx = i
                break
    since = 99 if cut_idx is None else len(v) - 1 - cut_idx
    drops = suspicious_drops(dps)
    return {"streak": streak, "years_since_cut": since,
            "cagr_5y": _cagr(dps, 5), "cagr_3y": _cagr(dps, 3),
            # True when a large drop was seen but no split data was supplied to
            # explain it. The record cannot be trusted either way.
            "split_ambiguous": bool(drops), "suspicious_drops": drops}


# ------------------------------------------------------- price-derived

def _align(annual: pd.Series, index: pd.DatetimeIndex,
           lag_days: int = 75, max_stale_days: int = 450) -> pd.Series:
    """
    Step an annual series onto a daily index, applied only from the date the
    figure would actually have been public, and only while it is still current.

    The lag prevents lookahead: a FY ending 31 December is not filed until late
    February, so using it for a January price is using information nobody had.
    75 days approximates the large-filer 10-K deadline.

    `max_stale_days` is the part that was missing, and its absence silently
    undid the whole EV/EBIT fix. J&J's operating income series has six points
    ending 2014; an unlimited forward-fill carried that single value across the
    entire ten-year window, making the denominator constant. A constant
    denominator turns EV/EBIT back into a constant times price, and the
    percentile collapsed to the raw close percentile — to two decimal places.
    The degraded case looked exactly like the fixed one.

    Past the limit the value goes NaN, so it drops out of the percentile rather
    than quietly poisoning it.
    """
    if annual is None or annual.empty:
        return pd.Series(index=index, dtype=float)
    shifted = annual.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)

    aligned = shifted.reindex(index, method="ffill")
    # Age of the fact underlying each aligned point.
    src = pd.Series(shifted.index, index=shifted.index).reindex(index, method="ffill")
    age = (pd.Series(index, index=index) - src).dt.days
    return aligned.where(age <= max_stale_days)


def price_context(px: pd.DataFrame, shares: float, dps_annual: pd.Series,
                  ebit: float, ebitda: float, fcf: float,
                  revenue: float,
                  ebit_hist: pd.Series | None = None,
                  shares_hist: pd.Series | None = None,
                  net_debt_hist: pd.Series | None = None,
                  revenue_hist: pd.Series | None = None,
                  splits: pd.Series | None = None) -> dict:
    """
    Everything that needs price history: market cap, liquidity, and the
    "vs its own history" measures that all three screens rank on.

    `px` needs close and volume on a daily index, ideally 10+ years.
    """
    if px is None or px.empty:
        return {}

    close = px["close"].dropna()
    spot = float(close.iloc[-1])
    mcap = spot * shares if shares else 0.0
    adv = float((close * px["volume"]).tail(20).mean()) if "volume" in px else 0.0

    ath = float(close.max())
    dd = (spot / ath - 1) * 100 if ath else 0.0
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else spot
    above200 = (spot / ma200 - 1) * 100 if ma200 else 0.0

    # Trailing yield history. DPS must be split-adjusted onto the same basis
    # as the price series, or a 4:1 split reads as a 75% dividend cut.
    yields = pd.Series(dtype=float)
    if len(dps_annual):
        d = _align(split_adjust(dps_annual, splits), close.index)
        yields = (d / close * 100).dropna()
    y5 = yields.tail(1260)

    # EV/EBIT against the company's own history.
    #
    # This was previously (close * shares_today) / ebit_today — a constant
    # times the price series, whose percentile equalled the raw price
    # percentile to machine precision. It was reporting "near its high" while
    # labelled "expensive versus its own history", and it carries ~30% of the
    # quality score. Historical EBIT, share count and net debt are now stepped
    # onto the price index, each lagged to its filing date.
    ev_hist = evs_hist = pd.Series(dtype=float)
    sh = _align(shares_hist, close.index) if shares_hist is not None else None
    if sh is None or sh.dropna().empty:
        sh = pd.Series(shares, index=close.index)
    nd = _align(net_debt_hist, close.index) if net_debt_hist is not None else None
    if nd is None or nd.dropna().empty:
        nd = pd.Series(0.0, index=close.index)

    ev = close * sh + nd
    # e.where(e > 0) drops every year of negative EBIT, so a loss-maker gets a
    # short ev_hist and was reported as "history too short or stale". Excluding
    # an unprofitable company from a VALUE screen is right; calling it a data
    # problem is not, and it hid 14 of the 75 flagged names.
    negative_ebit_years, aligned_points = 0, 0
    if ebit_hist is not None and not ebit_hist.empty:
        negative_ebit_years = int((ebit_hist <= 0).sum())
        e = _align(ebit_hist, close.index)
        aligned_points = int(e.notna().sum())
        ev_hist = (ev / e.where(e > 0)).replace([np.inf, -np.inf], np.nan).dropna()
    if revenue_hist is not None and not revenue_hist.empty:
        rv = _align(revenue_hist, close.index)
        evs_hist = (ev / rv.where(rv > 0)).replace([np.inf, -np.inf], np.nan).dropna()

    # A long series that stopped years ago is not a current ranking.
    ev_stale = (len(ev_hist) == 0
                or (close.index.max() - ev_hist.index.max()).days > 450)

    def pct_rank(s, window):
        s = s.tail(window).dropna()
        return float((s < s.iloc[-1]).mean() * 100) if len(s) > 30 else 50.0

    # Months the price has spent inside a 25% band — a rough base-length proxy.
    base = 0.0
    if len(close) > 60:
        recent = close.tail(400)
        lo, hi = recent.min(), recent.max()
        if lo > 0 and hi / lo < 1.25:
            base = len(recent) / 21

    return {
        "price": spot, "market_cap": mcap, "dollar_adv": adv,
        "drawdown_from_ath": dd, "pct_above_200dma": above200,
        "months_in_base": base,
        "dividend_yield": float(yields.iloc[-1]) if len(yields) else 0.0,
        "yield_median_5y": float(y5.median()) if len(y5) else 0.0,
        "yield_std_5y": float(y5.std()) if len(y5) else 0.0,
        # CoStar published 2,998x, PowerIntegrations 283x; 50 names exceeded
        # 200x and the maximum was 26,625x. That is a near-zero EBIT
        # denominator, the same shape as the ROIC artifact and equally
        # ungated. Above ~150x the multiple describes the denominator, not
        # the valuation.
        # Bounding by ZEROING was the inverted trap: CoStar's 2,998x artifact
        # became 0.0x, which renders as the CHEAPEST possible stock rather than
        # as unknown. 286 names carried an unvoided 0.0 and 33 a negative, and
        # three were published in the recovery table. A bound must void, never
        # substitute — the same rule the rest of the builder already follows.
        "ev_ebit": (float(ev_hist.iloc[-1])
                    if (len(ev_hist) and not ev_stale
                        and 0 < float(ev_hist.iloc[-1]) <= 150) else None),
        "ev_ebit_implausible": bool(
            len(ev_hist) and not (0 < float(ev_hist.iloc[-1]) <= 150)),
        "_ebit_margin": round(ebit / revenue * 100, 2) if revenue else None,
        "_ev_hist_points": len(ev_hist),
        "_ev_last_valid": ev_hist.index.max().date().isoformat() if len(ev_hist) else None,
        # Count AND recency. Counting alone passed J&J, whose aligned EBIT had
        # 1,564 valid points every one of which fell before June 2016 — so
        # pct_rank's .tail() ranked against a series that had already ended and
        # reported a mid-2016 multiple as current.
        "ev_history_degraded": bool(len(ev_hist) < 500 or ev_stale),
        # Why it is short, so an unprofitable company is not filed under a
        # data-quality reason.
        # Plenty of history aligned but little survived e.where(e > 0) means
        # the company was unprofitable, not that the history is short. The old
        # test required negative_ebit_years >= 2 on the ANNUAL series, which
        # missed names whose losses clustered — MRVL aligned 1,610 days and
        # only 358 were usable.
        "ev_short_because_unprofitable": bool(
            len(ev_hist) < 500 and not ev_stale
            and aligned_points >= 500 and len(ev_hist) < aligned_points * 0.8),
        "_aligned_ebit_points": aligned_points,
        "_negative_ebit_years": negative_ebit_years,
        "ev_ebit_median_10y": float(ev_hist.tail(2520).median()) if len(ev_hist) else 0.0,
        # Neutral 50 when there is not enough history to rank against, rather
        # than a confident number derived from a constant.
        "ev_ebit_percentile_10y": (pct_rank(ev_hist, 2520)
                                   if len(ev_hist) >= 500 and not ev_stale else 50.0),
        "ev_sales_percentile_5y": pct_rank(evs_hist, 1260) if len(evs_hist) else 50.0,
        "fcf_yield": _safe(fcf, mcap) if mcap else 0.0,
    }


# ---------------------------------------------------------------- build

def build(ticker: str, facts: dict, px: pd.DataFrame | None = None,
          as_of: date | None = None, sector: str = "general",
          splits: pd.Series | None = None, sic: int = 0) -> Fundamentals:
    """EDGAR facts plus an optional price frame -> a populated Fundamentals."""
    _attrs = {}
    g = lambda f: _annual(facts, f, as_of, _attrs)

    rev, ni, gp, oi = g("revenue"), g("net_income"), g("gross_profit"), g("operating_income")
    ocf, capex, da = g("ocf"), g("capex"), g("dep_amort")
    assets, equity, cash = g("assets"), g("equity"), g("cash")

    # Total debt = long-term + current portion + finance leases, summed on
    # matched periods. Any component may be absent; absent means zero for that
    # component but NOT for the total, so a name with no component at all is
    # flagged rather than recorded as debt-free.
    # Combine debt components without either failure mode.
    #
    # Zero-filled union was wrong: a component whose series ENDED became "zero
    # debt" rather than "unknown debt", and $35.5bn of Coca-Cola's long-term
    # debt vanished from invested capital.
    #
    # Strict intersection was wrong in the opposite direction: NVIDIA stopped
    # tagging finance leases in 2017, so intersecting all three components gave
    # ZERO dates and dropped a company whose two core components are perfectly
    # current. Every freshness number read 0 while the derived quantity was
    # empty — which is the composition failure freshness cannot see.
    #
    # Resolution: drop components that are STALE relative to the freshest one,
    # then intersect what remains. A dead optional component stops vetoing; a
    # live one still contributes.
    _combined = g("debt_total")
    _debt_notes = []
    # Prefer the combined tag ONLY if it is not stale relative to the
    # components. EnerSys tags debt_total with 4 points ending 2014 ($0.288bn)
    # while debt_noncurrent runs to 2026 ($1.080bn) — taking the combined tag
    # unconditionally produced net debt of -$0.151bn and a leverage ratio of
    # -0.30x, reading NET CASH for a company carrying +$0.64bn of net debt.
    # A wrong-signed leverage ratio lets a levered company clear a leverage
    # gate, and net_debt_ebitda gates all three screens.
    _comp_newest = max((v.index.max() for v in
                        (g("debt_noncurrent"), g("debt_current")) if not v.empty),
                       default=None)
    if (not _combined.empty and _comp_newest is not None
            and (_comp_newest - _combined.index.max()).days > 450):
        _debt_notes.append(
            f"debt_total stale (ends {_combined.index.max().date()}); "
            "using components")
        _combined = pd.Series(dtype=float)

    if not _combined.empty:
        debt = _combined                            # filer tags the total directly
    else:
        _nc, _cur, _fl = (g("debt_noncurrent"), g("debt_current"),
                          g("finance_leases"))

        # Some tags already BUNDLE leases. Adding finance_leases on top of
        # LongTermDebtAndCapitalLeaseObligations double-counts them.
        _nc_tags = fs.tags_used(facts, "debt_noncurrent", as_of)
        if any("CapitalLease" in t or "FinanceLease" in t for t in _nc_tags):
            if not _fl.empty:
                _debt_notes.append("finance leases already bundled in "
                                   + _nc_tags[0])
            _fl = pd.Series(dtype=float)

        # CORE vs OPTIONAL is the distinction that matters, not staleness.
        #
        # Dropping a stale component is right for finance leases (NVIDIA
        # stopped tagging them in 2017; the balance was immaterial) and
        # catastrophic for long-term debt (Coca-Cola's ends 2023 and is
        # $35.5bn — dropping it reproduces the zero-fill bug exactly).
        #
        # A stale CORE component means the TOTAL is stale: intersect and let
        # the series end honestly, so data_stale_days and concept_freshness
        # report it. A stale OPTIONAL component is simply left out.
        _core = {k: v for k, v in
                 {"debt_noncurrent": _nc, "debt_current": _cur}.items()
                 if not v.empty}
        _opt = {"finance_leases": _fl} if not _fl.empty else {}

        if _core:
            _freshest = max(v.index.max() for v in
                            list(_core.values()) + list(_opt.values()))
            for k, v in list(_opt.items()):
                lag = (_freshest - v.index.max()).days
                if lag > 450:
                    _debt_notes.append(f"{k} dropped, {lag}d stale (optional)")
                    _opt.pop(k)
            _fresh = {**_core, **_opt}
            if _fresh:
                idx = None
                for v in _fresh.values():
                    idx = v.index if idx is None else idx.intersection(v.index)
                debt = (sum((v.reindex(idx) for v in _fresh.values()),
                            start=pd.Series(0.0, index=idx))
                        if len(idx) else pd.Series(dtype=float))
            else:
                debt = pd.Series(dtype=float)
        else:
            debt = pd.Series(dtype=float)

    if _debt_notes:
        print(f"  [info] {ticker}: debt — {'; '.join(_debt_notes)}")

    shares_raw = g("shares")
    # Counts go UP at a split; per-share figures go down. Adjusting shares was
    # simply missing, though splits were already being fetched.
    shares = split_adjust(shares_raw, splits, kind="count")
    dps, eps = g("dividends_per_share"), g("eps_diluted")
    interest, div_paid = g("interest_expense"), g("dividends_paid")
    ca, cl, re_ = g("current_assets"), g("current_liabilities"), g("retained_earnings")

    if rev.empty:
        raise ValueError(f"{ticker}: no annual revenue facts")
    _rb = _attrs.get("revenue", {})
    if _rb.get("revenue_basis"):
        _rep = _rb.get("revenue_chain_replaced")
        print(f"  [info] {ticker}: revenue is bank-format total net revenue "
              f"({' + '.join(_rb.get('tags_used', []))})"
              + (f"; chain read {_rep['chain_latest']/1e6:,.0f}m from "
                 f"{', '.join(_rep['chain_tags'])} against a total of "
                 f"{_rep['total_latest']/1e6:,.0f}m" if _rep else ""))

    # A missing capex YEAR is treated as zero capex, which overstates free cash
    # flow rather than omitting it. Capex is absent entirely for QCOM and
    # Verizon and has three points for NVIDIA, so the overstatement is not rare.
    _capex_aligned = capex.reindex(ocf.index) if len(ocf) else pd.Series(dtype=float)
    fcf = (ocf - _capex_aligned.fillna(0)) if len(ocf) else pd.Series(dtype=float)
    _fill_warnings = _fill_integrity("free cash flow", fcf, _capex_aligned, "capex")
    r, n = _last(rev), _last(ni)

    # Oil majors and many financials report neither GrossProfit nor
    # OperatingIncomeLoss. EBIT = pretax income + interest expense is the
    # standard derivation and keeps ROIC, EV/EBIT and coverage from cascading
    # to zero for an entire sector.
    pretax = g("pretax_income")
    ref = pd.Timestamp(as_of or date.today())

    def _stale(series, limit=550):
        return series.empty or (ref - series.index.max()).days > limit

    # Fall back when the primary series is EMPTY *or* STALE. Testing emptiness
    # alone let J&J keep six OperatingIncomeLoss facts ending 2014 while its
    # pretax income ran to 2025 — so ROIC and EV/EBIT were computed on
    # eleven-year-old earnings and reported as current, making the name look
    # ~25% cheaper than it was, with no flag raised.
    oi_usable = not _stale(oi)
    if oi_usable:
        ebit_series = oi
    elif not _stale(pretax):
        ebit_series = pretax + g("interest_expense").abs().reindex(pretax.index).fillna(0)
        why = "absent" if oi.empty else f"stale (ends {oi.index.max().date()})"
        print(f"  [info] {ticker}: OperatingIncomeLoss {why}; "
              "EBIT derived from pretax + interest")
    else:
        ebit_series = oi if not oi.empty else pretax
    ebit = _last(ebit_series)
    if not ebit:
        print(f"  [warn] {ticker}: no derivable EBIT — ROIC, EV/EBIT and "
              "interest coverage will be unavailable, not zero")

    # Oil majors, financials and REITs frequently report no GrossProfit tag.
    # Deriving it from revenue minus cost of revenue covers most; where even
    # that fails, the absence must be labelled rather than presented as a 0%
    # margin, which reads as catastrophic. Mirror of the EBIT trap: unknown
    # must read as unknown in BOTH directions.
    if gp.empty:
        cost = g("cost_of_revenue")
        if not cost.empty:
            common = rev.index.intersection(cost.index)
            if len(common):
                derived = (rev.loc[common] - cost.loc[common]).dropna()
                implied = (derived / rev.loc[derived.index] * 100).dropna()
                # Some filers tag a partial cost line, which inflates the
                # derived margin. Verizon came back at 82.4%, implausible for a
                # telco on its face. Reject rather than publish a number that
                # looks authoritative and is not.
                if len(implied) and implied.tail(3).median() > 80:
                    print(f"  [warn] {ticker}: derived gross margin "
                          f"{implied.iloc[-1]:.1f}% implausible — cost line is "
                          "likely partial; leaving gross profit unavailable")
                else:
                    gp = derived
                    print(f"  [info] {ticker}: no GrossProfit tag; "
                          "derived from revenue - cost of revenue")
    ebitda = ebit + _last(da)
    net_debt = _last(debt) - _last(cash)
    # Period-matched, not three independent _last() calls — the same
    # cross-period bug _ratio() exists to prevent, never applied here.
    _bs = [s_ for s_ in (equity, debt, cash) if not s_.empty]
    if _bs:
        _common = _bs[0].index
        for s_ in _bs[1:]:
            _common = _common.intersection(s_.index)
    else:
        _common = pd.DatetimeIndex([])
    if len(_common):
        _i = _common.max()
        inv_cap = (float(equity.get(_i, 0.0)) + float(debt.get(_i, 0.0))
                   - float(cash.get(_i, 0.0)))
    else:
        inv_cap = _last(equity) + _last(debt) - _last(cash)

    f = Fundamentals(symbol=ticker, name=facts.get("entityName", ticker), sector=sector)
    f.wacc = max(SECTOR_WACC.get(sector, ASSUMED_WACC), MIN_WACC)
    f.wacc_is_assumed = True        # a bucket, not a measurement

    # XOM's predecessor CIK stopped filing after the restructuring, so its
    # series ends in 2021. The 54.95% "growth" it produced was a 2020 COVID
    # trough against 2021, presented as current. Age is now measured and gated.
    newest = rev.index.max()
    f.data_stale_days = int((pd.Timestamp(as_of or date.today()) - newest).days)
    # Measure the spread across the series the builder ACTUALLY USES, not the
    # ones it inspected and discarded. Including raw `oi` meant any company
    # that stopped tagging OperatingIncomeLoss got a false data-quality
    # failure: J&J's used series all end 2025-12-28 — a true mismatch of 0 —
    # but it read 4,018 and failed the gate. That inflated exactly the
    # data-quality rate this is supposed to measure honestly.
    _used = [rev, ebit_series]
    if not gp.empty:
        _used.append(gp)
    if not ocf.empty:
        _used.append(ocf)
    f.period_mismatch_days = _period_mismatch_days(*_used)

    # Systemic check: every concept against revenue, in one pass. Four separate
    # universe runs each found one instance of this; measuring it catches the
    # next one without a fifth.
    # Composition invariant, complementing freshness.
    #
    # NVIDIA violated this while every freshness number read 0: total debt was
    # EMPTY though both core components were current. A derived quantity must
    # not be more absent, or more stale, than its freshest input.
    f.derivation_warnings = _derivation_integrity({
        "debt": (debt, {"debt_noncurrent": g("debt_noncurrent"),
                        "debt_current": g("debt_current"),
                        "finance_leases": g("finance_leases")}),
        "fcf": (fcf, {"ocf": ocf, "capex": capex}),
    })
    f.derivation_warnings += _fill_warnings
    for w in f.derivation_warnings:
        print(f"  [warn] {ticker}: {w}")

    # Concepts reported in more than one currency. Pinning keeps the series
    # clean, but a filer that mixes units is telling you something about its
    # whole submission — Nebius mixes RUB and USD across nine concepts.
    f.mixed_unit_concepts = sorted(
        k for k, a in _attrs.items() if a.get("multi_unit"))
    # Where choosing USD cost coverage, say so — the streak and every "own
    # history" percentile are computed on the shorter series.
    f.unit_coverage_cost = {k: a["unit_coverage_cost"]
                            for k, a in _attrs.items()
                            if a.get("unit_coverage_cost")}
    for k, c in f.unit_coverage_cost.items():
        print(f"  [info] {ticker}: {k} pinned to {c['chosen']} "
              f"({c['chosen_facts']} facts) over {c['richest']} "
              f"({c['richest_facts']}) — history truncated")
    if f.mixed_unit_concepts:
        print(f"  [warn] {ticker}: {len(f.mixed_unit_concepts)} concept(s) "
              f"reported in multiple units — {', '.join(f.mixed_unit_concepts[:5])}")

    fresh = fs.concept_freshness(facts, as_of=as_of)
    f.stale_concepts = fresh.get("stale", [])
    # The yardstick, measured against the calendar rather than itself.
    if fresh.get("reference_stale"):
        f.data_stale_days = max(f.data_stale_days,
                                fresh.get("reference_lag_days", 0))
    f.concept_lags = fresh.get("lags", {})
    if f.stale_concepts:
        worst = max((f.concept_lags.get(c, 0) for c in f.stale_concepts), default=0)
        print(f"  [warn] {ticker}: concepts lagging revenue by up to {worst}d — "
              f"{', '.join(f.stale_concepts[:6])}")
    if f.period_mismatch_days > 400:
        print(f"  [warn] {ticker}: statement series end up to "
              f"{f.period_mismatch_days} days apart — ratios are cross-period")
    if f.data_stale_days > 550:
        print(f"  [warn] {ticker}: newest annual period is {newest.date()} "
              f"({f.data_stale_days} days old) — figures are not current")

    # --- quality, all from EDGAR -----------------------------------------
    # Set after the ratio, so a stale series counts as unavailable too — a
    # 2009 gross margin is not more usable than a missing one.
    # Third instance of the same pattern. Absent operating cash flow was
    # surfacing as "FCF margin 0.0% under 8%" and "FCF positive 0/10 years" —
    # two business labels on one missing tag.
    # Pre-revenue companies cannot have meaningful margins. $10m is low enough
    # to admit any real operating business and high enough to exclude a shell.
    # An ABSOLUTE floor leaves a band unguarded. $10m catches Archer at $200k,
    # but a $2B-market-cap biotech with $12m of revenue clears it and publishes
    # a 33% gross margin that is still arithmetic noise. Scale the floor to the
    # size of the business.
    #
    # Financials are exempt from the relative test: revenue is structurally a
    # low percentage of assets for a bank, and that is not pre-commercial.
    MIN_REVENUE = 1e7
    _rev_now, _assets_now = abs(_last(rev)), abs(_last(assets))
    _relative_floor = (0.0 if sector == "financial" or not _assets_now
                       else 0.02 * _assets_now)
    f.pre_revenue = bool(_rev_now < max(MIN_REVENUE, _relative_floor))
    if f.pre_revenue and _rev_now >= MIN_REVENUE:
        print(f"  [info] {ticker}: revenue ${_rev_now/1e6:,.0f}m is "
              f"{_rev_now/_assets_now:.1%} of assets — pre-commercial scale")
    _den_floor = max(MIN_REVENUE, _relative_floor)
    f.gross_margin, _gm_stale = _ratio(gp, rev, as_of=as_of, min_den=_den_floor)
    f.gross_profit_unavailable = gp.empty or _gm_stale
    f.fcf_margin, _fcf_stale = _ratio(fcf, rev, as_of=as_of, min_den=_den_floor)
    # Zero-filled capex makes FCF fabricated, not merely incomplete. Half the
    # periods missing is enough to void the series.
    _capex_missing = int(_capex_aligned.isna().sum()) if len(ocf) else 0
    _capex_void = bool(len(ocf)) and _capex_missing >= 0.5 * len(ocf)
    f.fcf_unavailable = ocf.empty or _fcf_stale or _capex_void
    f.capex_voided_fcf = _capex_void
    if _capex_void:
        print(f"  [warn] {ticker}: capex absent for {_capex_missing} of "
              f"{len(ocf)} periods — free cash flow voided, not published")
    f.revenue_cagr_5y = _cagr(rev, 5)
    # Annual year-over-year, NOT trailing twelve months — the builder reads
    # annual periods only. Misleading where the newest period is stale, which
    # is why data_stale_days is gated above.
    f.revenue_growth_ttm = _cagr(rev, 1)
    f.share_count_cagr_5y = _cagr(shares, 5)
    # Separate "not profitable enough" from "not listed long enough".
    #
    # The gate read `fcf_positive_years_of_10 < 8`, so a company with six years
    # of history — every one positive — scored 6 and failed. 56 of 102 failures
    # were this. Record the history length and the negative-year count so the
    # gate can test a RATE rather than an absolute count.
    _fcf10 = fcf.tail(10)
    f.fcf_positive_years_of_10 = int((_fcf10 > 0).sum())
    f.fcf_history_years = int(len(_fcf10))
    f.fcf_negative_years = int((_fcf10 <= 0).sum())
    f.capex_years_missing = int(_capex_aligned.isna().sum()) if len(ocf) else 0
    if _capex_void:
        # Must come after the counts above, or the reset is silently
        # overwritten — which left fcf_positive_years_of_10 at 10/10.
        f.fcf_margin = 0.0
        f.fcf_positive_years_of_10 = 0
        f.fcf_negative_years = 0
    if f.capex_years_missing:
        print(f"  [warn] {ticker}: capex missing for {f.capex_years_missing} of "
              f"{len(ocf)} years — free cash flow is overstated in those years")

    nopat = ebit * (1 - ASSUMED_TAX)

    # Invested capital at or below zero produces nonsense that GATES ON: SLS
    # read +2,422%, ASAN +341%, PTON -313%. The positive ones pass the ROIC
    # gate spuriously, and gates run before scores, so a false pass is worse
    # than a false fail. Refuse rather than publish.
    # Distinguish "no tag because no debt" from "no tag because migration".
    #
    # concept_freshness cannot help here by construction: it sees a concept
    # that STOPPED, never one that never started. Interest expense is the
    # corroborating witness — a company paying material interest has debt
    # whether or not it tagged the balance, and one paying none plausibly does
    # not. A genuinely debt-free company should get a real ROIC rather than
    # being excluded alongside the migrations.
    _interest_now = abs(_last(interest))
    if debt.empty and not equity.empty:
        if _interest_now > 0.005 * max(abs(r), 1.0):
            f.debt_unavailable = True               # pays interest, no balance
            print(f"  [warn] {ticker}: no debt tag but interest expense is "
                  f"{_interest_now / 1e6:,.0f}m — balance is missing, not zero")
        else:
            # Treated as debt-free, recorded so the assumption is visible.
            debt = pd.Series(0.0, index=equity.index)
            f.debt_assumed_zero = True
            f.debt_unavailable = False
    else:
        f.debt_unavailable = debt.empty
    # Near-zero invested capital explodes ROIC just as surely as negative does.
    # Deckers: equity $2.50bn, no debt, cash $1.91bn leaves $0.59bn of invested
    # capital against $1.26bn of EBIT, and it published 281.1% against a real
    # 35-40%. It ranked FIRST on the value screen. ADP showed 105.0% the same
    # way. A capital base smaller than half of EBIT is not a denominator.
    # Three ways the denominator fails, not one.
    #
    # Otis carries NEGATIVE book equity of -$5.35bn from its spinoff; netting
    # that against $7.12bn of debt and $1.10bn of cash leaves $0.66bn of
    # invested capital and a published ROIC of 204.2%. Deckers' was $0.59bn —
    # the old ratio guard sat between them. Buyback-heavy and spun-off
    # companies break an equity-plus-debt-minus-cash definition structurally,
    # however clean the inputs, so negative equity voids it outright.
    #
    # And any ROIC above 100% on a business of real scale is a denominator
    # artifact rather than a finding. Ten names published above it.
    # Construction-level check, carrying the term breakdown for the caveat.
    f.invested_capital_check = denominator_reliability(
        inv_cap, terms={"equity": _last(equity), "debt": _last(debt),
                        "cash": _last(cash)})
    f.ebit_margin_check = denominator_reliability(
        ebit, residual_of=(r, "revenue")) if r else {"reliable": True}

    _neg_equity = bool(_last(equity) < 0)
    _capital_too_thin = bool(inv_cap > 0 and ebit and inv_cap < abs(ebit) * 0.5)
    _implausible = bool(inv_cap > 0 and ebit
                        and (ebit * (1 - ASSUMED_TAX)) / inv_cap > 1.0)
    f.roic_unavailable = (inv_cap <= 0 or _neg_equity or _capital_too_thin
                          or _implausible or f.ebit_unavailable or debt.empty
                          or not f.invested_capital_check["reliable"])
    if f.roic_unavailable:
        reason = ("negative book equity — the equity+debt-cash definition "
                  "does not describe this balance sheet" if _neg_equity
                  else "implied ROIC over 100% — denominator artifact"
                  if _implausible
                  else "invested capital <= 0" if inv_cap <= 0
                  else f"invested capital ${inv_cap/1e9:.2f}bn is under half of "
                       f"EBIT ${ebit/1e9:.2f}bn — denominator too thin"
                  if _capital_too_thin
                  else "no debt tag" if debt.empty else "no EBIT")
        print(f"  [warn] {ticker}: ROIC unavailable ({reason}) — not published")
        f.roic_ttm = f.roic_5y = 0.0
    else:
        f.roic_ttm = _safe(nopat, inv_cap)

        # Nearest-period lookup, not exact. Fiscal year-ends drift by days
        # across series, so `equity.get(i)` missed and the list emptied —
        # roic_5y then silently fell back to roic_ttm for 75 of 190 names,
        # a one-year figure labelled as a five-year mean.
        def _near(series, when, tol=10):
            if series.empty:
                return np.nan
            j = series.index[np.abs((series.index - when).days) <= tol]
            return float(series.loc[j[-1]]) if len(j) else np.nan

        # The SAME zero-fill removed from inv_cap, surviving here.
        #
        # `(0 if np.isnan(d_) else d_)` substituted zero for missing debt, so
        # Coca-Cola's 2024 and 2025 — where its debt series has ended — took a
        # capital base of $14.0bn and $21.9bn instead of ~$52bn and read 56.3%
        # and 49.6%. The five-year mean jumped from 16.1% to 30.9%.
        #
        # Equity, debt and cash are all CORE to invested capital. A period
        # missing any of them is a period where invested capital cannot be
        # computed, so it is skipped rather than filled.
        roic_hist, _skipped_years = [], 0
        for i, o in ebit_series.tail(5).items():      # the USED series, not `oi`
            e_, d_, c_ = _near(equity, i), _near(debt, i), _near(cash, i)
            if np.isnan(e_) or np.isnan(d_) or np.isnan(c_):
                _skipped_years += 1
                continue
            base = e_ + d_ - c_
            if base <= 0:
                continue
            roic_hist.append(_safe(o * (1 - ASSUMED_TAX), base))
        if _skipped_years:
            print(f"  [info] {ticker}: {_skipped_years} ROIC year(s) skipped — "
                  "a balance-sheet input was absent, not zero")

        roic_hist = [x for x in roic_hist if x]
        # Flag rather than silently substitute the one-year number.
        f.roic_5y_is_ttm_fallback = len(roic_hist) == 0
        f.roic_5y = float(np.mean(roic_hist)) if roic_hist else f.roic_ttm
        f.roic_declining_years = sum(
            1 for a, b in zip(roic_hist, roic_hist[1:]) if b < a
        ) if len(roic_hist) > 1 else 0

    common_gm = gp.index.intersection(rev.index)
    if len(common_gm) >= 4:
        i = common_gm[-4]
        old = _safe(float(gp.loc[i]), float(rev.loc[i]))
        f.gross_margin_delta_3y = f.gross_margin - old

    f.ebit_unavailable = not bool(ebit)
    f.net_debt_ebitda = _safe(net_debt, ebitda, pct=False) if ebitda > 0 else 99.0

    # `or 99.0` turned a zero EBIT into coverage of 99 — maximum safety from
    # missing data. A company with no derivable EBIT has UNKNOWN coverage, and
    # unknown must not read as safe.
    if not ebit:
        f.interest_coverage = 0.0
    else:
        ie = abs(_last(interest))
        f.interest_coverage = 99.0 if ie == 0 else _safe(ebit, ie, pct=False)

    # Altman Z. The original 1968 coefficients require the MARKET value of
    # equity in the X4 term; using book value there silently understates Z for
    # any company trading above book, which is most of them. Market cap is only
    # available once prices are joined, so X4 is deferred to _finish_altman
    # below and the private-firm Z' variant is used when there is no price.
    # Altman fitted Z on manufacturers. Regulated utilities, banks and REITs
    # run leverage that reads as distress under those coefficients without
    # being distressed — NextEra scores 0.54. A domain mismatch, not a signal.
    # Archer scored 6.62 — comfortably "safe" — while burning $430m a year
    # pre-revenue, because the market-equity term dominates every other.
    # Use the raw SIC, not the coarse sector label: a crypto miner filed under
    # 6199 is not a bank, and Altman describes its balance sheet fine.
    f.altman_not_applicable = (fs.altman_exempt(sic) if sic
                               else sector in ("utility", "reit"))
    if abs(_last(rev)) < 1e7:
        f.altman_not_applicable = True
    f._altman_parts = {
        "ta": _last(assets), "tl": _last(assets) - _last(equity),
        "wc": _last(ca) - _last(cl), "re": _last(re_), "ebit": ebit,
        "sales": r, "book_equity": _last(equity),
    }

    burn = -_last(fcf)
    f.cash_runway_quarters = 99.0 if burn <= 0 else max(_last(cash) / (burn / 4), 0.0)

    # --- dividends --------------------------------------------------------
    # Split-adjust BEFORE reading the record, or every split reads as a cut.
    _clean = drop_dividend_outliers(drop_offcycle_periods(split_adjust(dps, splits)))
    f.specials_dropped = int(_clean.attrs.get("specials_dropped", 0))
    f.dividend_outliers_dropped = int(_clean.attrs.get("outliers_dropped", 0))
    rec = dividend_record(_clean)
    f.increase_streak_years = rec["streak"]
    f.years_since_cut = rec["years_since_cut"]
    f.dps_cagr_5y, f.dps_cagr_3y = rec["cagr_5y"], rec["cagr_3y"]
    # Unexplained large drop: the record is unverifiable, so the dividend
    # screen must not pass OR fail this name on it silently.
    f.dividend_record_ambiguous = rec["split_ambiguous"]
    f.eps_payout, _ = _ratio(dps, eps, as_of=as_of)
    f.fcf_payout, _ = _ratio(div_paid.abs() if len(div_paid) else div_paid, fcf, as_of=as_of)

    # An ADR trades at a MULTIPLE of the underlying share, and XBRL never says
    # what that multiple is. Dividing a per-ordinary-share dividend by an ADR
    # price therefore understates the yield by an unknown integer factor —
    # HDFC Bank came out at 0.56% against a real 1.11%, out by roughly two,
    # which is its ADR ratio and not a rounding error.
    #
    # Unit pinning fixed the currency and could not fix this, because the
    # information is not in the filing. A yield that is wrong by an unknown
    # factor should not be published at all.
    f.foreign_private_issuer = fs.is_foreign_private_issuer(facts)
    f.adr_ratio_unknown = f.foreign_private_issuer and not dps.empty
    # A filer with NO USD facts has nothing for unit pinning to prefer, so its
    # statements arrive in the home currency and get divided by a USD price.
    # Unreachable until taxonomy_of stopped reading one stray us-gaap concept
    # as a US filer: Telus (CAD) then scored valuation_gap 100 and would have
    # published on the recovery screen, and Ericsson (SEK) is out by ~10x.
    # There is no FX series on the free path to convert with, and converting
    # would still mix a spot rate into ten years of history. Gate and void.
    f.statement_currency = str(_attrs.get("revenue", {}).get("unit_used")
                               or "USD").split("/")[0]

    # --- price-derived ----------------------------------------------------
    if px is not None and not px.empty:
        ebit_hist = ebit_series
        net_debt_hist = (debt.reindex(cash.index).fillna(0) - cash
                         if len(cash) else pd.Series(dtype=float))
        for k, v in price_context(px, _last(shares), dps, ebit, ebitda,
                                  _last(fcf), r,
                                  ebit_hist=ebit_hist, shares_hist=shares,
                                  net_debt_hist=net_debt_hist, revenue_hist=rev,
                                  splits=splits).items():
            if not k.startswith("_"):
                # None means unavailable and must stay None, not become 0.0.
                setattr(f, k, v)
        f.buyback_yield = max(-f.share_count_cagr_5y, 0.0)

        # Compute it rather than leave a placeholder feeding a scored component.
        _ev_now = f.market_cap + (_last(debt) - _last(cash))
        _implied = reverse_dcf_growth(_ev_now, _last(fcf), f.wacc)
        if _implied is None:
            f.reverse_dcf_unavailable = True
        else:
            f.reverse_dcf_implied_growth = _implied
        # Assigned explicitly rather than only through the setattr loop above.
        # A field that reaches the dataclass by indirection is invisible to any
        # check that the field is written at all — which is the same dead-code
        # shape as `sector` (read, never set) and `multi_unit` (set, never read).
        _pc = price_context(px, _last(shares), dps, ebit, ebitda, _last(fcf), r,
                            ebit_hist=ebit_series, shares_hist=shares,
                            net_debt_hist=net_debt_hist, revenue_hist=rev,
                            splits=splits)
        f.ev_history_degraded = bool(_pc.get("ev_history_degraded"))
        f.ev_short_because_unprofitable = bool(
            _pc.get("ev_short_because_unprofitable"))
    f.altman_z = _altman(f._altman_parts,
                         market_equity=f.market_cap if px is not None else None)
    # --- financials -------------------------------------------------------
    # Only populated for names the explicit SIC set AND the witness agree on.
    # Everything here carries its own unavailable flag; nothing substitutes a
    # neutral value for an absent input.
    if sector == "financial":
        f.financial_subtype, f.financial_in_scope = financial_scope(sic, facts)
        f.cost_of_equity = SECTOR_COST_OF_EQUITY.get(
            f.financial_subtype or "financial", 10.5)

        _gw, _intan = g("goodwill"), g("intangibles")
        _eq_s, _ni_s, _as_s = equity, g("net_income"), assets

        # ROE. Equity is the denominator, so its materiality is checked against
        # assets rather than against its own terms — it is a primary reported
        # quantity, not a signed sum.
        f.roe_ttm, _roe_stale = _ratio(_ni_s, _eq_s, as_of=as_of)
        _roe_hist = []
        for i_, ni_ in _ni_s.tail(5).items():
            e_ = _near_value(_eq_s, i_)
            if e_ and e_ > 0:
                _roe_hist.append(float(ni_) / e_ * 100)
        f.roe_5y = float(np.mean(_roe_hist)) if _roe_hist else f.roe_ttm
        f.roe_declining_years = sum(1 for a_, b_ in zip(_roe_hist, _roe_hist[1:])
                                    if b_ < a_) if len(_roe_hist) > 1 else 0
        f.roe_unavailable = bool(_eq_s.empty or _ni_s.empty or _roe_stale
                                 or _last(_eq_s) <= 0)
        if f.roe_unavailable:
            f.roe_5y = f.roe_ttm = None

        f.equity_to_assets = (_ratio(_eq_s, _as_s, as_of=as_of)[0]
                              if not _as_s.empty else None)

        # Tangible book is equity MINUS goodwill MINUS intangibles: a signed
        # sum, so it gets the cancellation check. For an acquisitive bank it is
        # a small residual of large terms, which is the exact shape that made
        # Deckers read 281% and Otis 204%.
        if not _eq_s.empty:
            _tb = _last(_eq_s) - _last(_gw) - _last(_intan)
            f.tangible_book_check = denominator_reliability(
                _tb, {"equity": _last(_eq_s), "goodwill": -_last(_gw),
                      "intangibles": -_last(_intan)})
            f.tangible_book = _tb if f.tangible_book_check["reliable"] else None
        f.tangible_book_unavailable = f.tangible_book is None

        f.rotce = (None if (f.tangible_book is None or _ni_s.empty
                            or f.tangible_book <= 0)
                   else float(_last(_ni_s)) / f.tangible_book * 100)
        f.rotce_unavailable = f.rotce is None

        _sh = _last(shares)
        if f.tangible_book and _sh and px is not None and not px.empty:
            _tbvps = f.tangible_book / _sh
            f.price_to_tangible_book = (f.price / _tbvps) if _tbvps > 0 else None
            _tb_hist = (_eq_s - _gw.reindex(_eq_s.index).fillna(0)
                        - _intan.reindex(_eq_s.index).fillna(0))
            _shs = shares.reindex(_eq_s.index).ffill()
            _tbvps_hist = (_tb_hist / _shs).dropna()
            if len(_tbvps_hist) >= 2:
                f.tbvps_cagr_5y = _cagr(_tbvps_hist, 5)
            _close = px["close"].dropna()
            _al = _align(_tbvps_hist, _close.index)
            _series = (_close / _al.where(_al > 0)).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(_series) >= 500:
                f.ptbv_percentile_10y = float(
                    (_series.tail(2520) < _series.iloc[-1]).mean() * 100)
            else:
                f.ptbv_history_degraded = True

    del f._altman_parts

    # Enforce the invariant before returning: no value may survive on a field
    # whose input was declared unavailable.
    if f.adr_ratio_unknown and f.dividend_yield:
        print(f"  [warn] {ticker}: foreign issuer — per-share dividend cannot "
              "be matched to an ADR price without the ratio; yield voided")
        f.dividend_yield = f.yield_median_5y = f.yield_std_5y = 0.0

    f.voided_fields = void_derived_fields(f)
    if f.voided_fields:
        print(f"  [info] {ticker}: voided {len(f.voided_fields)} field(s) "
              f"derived from unavailable inputs — {', '.join(f.voided_fields[:5])}")

    # --- no free source ---------------------------------------------------
    # eps_revision_3m / eps_revision_6m need analyst estimates. Default 0 is
    # SAFE rather than convenient: the dividend yield-trap gate only fires
    # below -20%, so zero passes without silently admitting a trap. It does
    # mean the trap filter is inactive on the free tier — say so in the UI.
    # insider_net_6m is derivable from SEC Form 4 but needs its own parser.
    return f


def load_fundamentals_report(tickers: list[str], as_of: date | None = None,
                             with_prices: bool = True) -> dict:
    """
    Records PLUS a categorised account of everything that did not become one.

    A ticker that fails before a Fundamentals exists never reaches a gate, so
    it vanishes from every pass/fail statistic. Foreign private issuers were
    doing exactly that — a data-quality rate computed over surviving records
    silently excludes the worst-covered names and flatters itself.
    """
    out, skipped = [], []
    for t in tickers:
        try:
            facts = fs.company_facts(t)
        except Exception as e:
            skipped.append((t, f"facts: no CIK or fetch failed — {e}"))
            continue
        tax = fs.taxonomy_of(facts)
        if tax == "unknown":
            skipped.append((t, "unknown taxonomy — neither us-gaap nor ifrs-full"))
            continue
        sector, sic = fs.company_sector(t), fs.company_sic(t)
        px, splits = None, None
        if with_prices:
            try:
                px = fs.equity_ohlcv(t)
                splits = fs.equity_splits(t)     # authoritative, never inferred
            except Exception as e:
                skipped.append((t, f"prices: {e} (built without price context)"))
        try:
            out.append(build(t, facts, px, as_of, sector=sector,
                             splits=splits, sic=sic))
        except Exception as e:
            skipped.append((t, str(e)))

    reasons = {}
    for t, why in skipped:
        # "facts: no CIK or fetch failed" prefixes EVERY fetch failure, so a
        # substring test on "no CIK" filed Bank OZK's 404 — a bank that files
        # with the FDIC, not the SEC, and has no XBRL facts at all — as a
        # ticker-map miss. Match the KeyError's own text instead.
        key = ("no CIK" if "no CIK for" in why else
               "facts fetch failed" if why.startswith("facts:") else
               "unknown taxonomy" if "taxonomy" in why else
               "no annual revenue" if "revenue" in why else
               "price fetch failed" if "prices:" in why else "other")
        reasons.setdefault(key, []).append(t)

    if skipped:
        print(f"  [info] {len(skipped)}/{len(tickers)} never became a record:")
        for k, v in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            print(f"    {k}: {len(v)}  e.g. {', '.join(v[:5])}")

    return {"records": out, "excluded": skipped, "by_reason": reasons,
            "attempted": len(tickers), "built": len(out),
            "never_built_pct": round(len(skipped) / max(len(tickers), 1) * 100, 1)}


def load_fundamentals(tickers: list[str], as_of: date | None = None,
                      with_prices: bool = True) -> list[Fundamentals]:
    return load_fundamentals_report(tickers, as_of, with_prices)["records"]


def void_derived_fields(f: Fundamentals) -> list:
    """
    A field derived from an unavailable input must not HOLD a value.

    The gates exclude these names, so nothing downstream of a gate is wrong.
    But PayPal stores an EV/EBIT of 7.25 and an Altman Z of 1.95 computed on an
    EMPTY debt series, and anything reading a Fundamentals outside the gate
    path — a dashboard detail page, an export, a notebook — renders them as
    fact. The meta-test cannot see this: both fields are written, both are
    read, and the read is reachable.

    Containment by gate is not the same as correctness by construction.
    """
    voided = []
    rules = [
        (f.debt_unavailable, ["net_debt_ebitda", "altman_z", "ev_ebit",
                              "ev_ebit_median_10y", "ev_ebit_percentile_10y"]),
        (f.ebit_unavailable, ["ev_ebit", "ev_ebit_median_10y",
                              "ev_ebit_percentile_10y", "interest_coverage",
                              "net_debt_ebitda"]),
        (f.roic_unavailable, ["roic_5y", "roic_ttm", "roic_declining_years"]),
        (f.fcf_unavailable, ["fcf_margin", "fcf_payout",
                             "fcf_positive_years_of_10"]),
        (f.gross_profit_unavailable, ["gross_margin", "gross_margin_delta_3y"]),
        (f.pre_revenue, ["gross_margin", "fcf_margin", "ev_sales_percentile_5y"]),
        (f.ev_history_degraded, ["ev_ebit_percentile_10y", "ev_ebit_median_10y"]),
        (f.statement_currency != "USD",
         ["ev_ebit", "ev_ebit_median_10y", "ev_ebit_percentile_10y",
          "fcf_yield", "ev_sales_percentile_5y", "altman_z",
          "price_to_tangible_book", "ptbv_percentile_10y"]),
    ]
    for unavailable, fields_ in rules:
        if not unavailable:
            continue
        for name in fields_:
            cur = getattr(f, name, None)
            if cur not in (None,) and cur != 0:
                # None, not 0.0. Setting zero made the field say "unavailable"
                # while the VALUE said "zero" — 1,284 fields across the
                # universe, and Trade Desk listed ev_ebit, altman_z and
                # net_debt_ebitda as voided while all three held 0.0. The
                # invariant was enforced in the adapter and not in the builder,
                # one layer below where it was fixed.
                setattr(f, name, None)
                voided.append(name)
    return sorted(set(voided))


def coverage_report(f: Fundamentals) -> dict:
    """
    How much of this record is real versus dataclass default.

    Worth checking before trusting a screen: a Fundamentals that is 90%
    defaults will fail gates for reasons that have nothing to do with the
    company.
    """
    from dataclasses import fields as dc_fields

    # Data-quality flags are metadata about the record, not fields of it. A
    # healthy company correctly leaves every one at its default, so counting
    # them drags reported coverage down as more flags are added.
    META = {"symbol", "name", "sector", "dividend_record_ambiguous",
            "data_stale_days", "ebit_unavailable", "period_mismatch_days",
            "ev_history_degraded", "gross_profit_unavailable", "fcf_unavailable",
            "roic_unavailable", "roic_5y_is_ttm_fallback", "debt_unavailable",
            "fcf_history_years", "fcf_negative_years", "capex_years_missing",
            "stale_concepts", "concept_lags", "derivation_warnings",
            "mixed_unit_concepts", "unit_coverage_cost", "voided_fields",
            "foreign_private_issuer", "adr_ratio_unknown", "pre_revenue",
            "statement_currency",
            "debt_assumed_zero", "capex_voided_fcf", "altman_not_applicable",
            "ev_short_because_unprofitable",
            # Financial flags, same reasoning.
            "roe_unavailable", "rotce_unavailable", "tangible_book_unavailable",
            "ptbv_history_degraded", "financial_in_scope", "tangible_book_check"}

    # Sector-specific fields are NOT missing data when the sector does not
    # apply — a software company has no ROE by construction and an insurer has
    # no EV/EBIT. Counting them as absent is the same unavailable-versus-not-
    # applicable confusion the gates already learned to separate.
    FINANCIAL_ONLY = {"financial_subtype", "roe_5y", "roe_ttm",
                      "roe_declining_years", "rotce", "tangible_book",
                      "equity_to_assets", "price_to_tangible_book",
                      "ptbv_percentile_10y", "tbvps_cagr_5y", "cost_of_equity"}
    GENERAL_ONLY = {"roic_5y", "roic_ttm", "roic_declining_years", "ev_ebit",
                    "ev_ebit_median_10y", "ev_ebit_percentile_10y",
                    "reverse_dcf_implied_growth", "wacc"}
    skip = set(META) | (FINANCIAL_ONLY if f.sector != "financial" else GENERAL_ONLY)

    blank = Fundamentals(symbol="x", name="x")
    filled, empty = [], []
    for fld in dc_fields(Fundamentals):
        if fld.name in skip:
            continue
        (filled if getattr(f, fld.name) != getattr(blank, fld.name) else empty
         ).append(fld.name)
    total = len(filled) + len(empty)
    return {"filled": len(filled), "total": total,
            "pct": round(len(filled) / total * 100, 1), "missing": empty}
