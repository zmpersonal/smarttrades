"""
Does a screen's ranking carry information?

That is the only question here. NOT "would this have made money" — a
1,449-name universe over five usable years of XBRL history cannot answer that,
and a backtest that emits a return number will be believed far past what it
can support. So this emits hit rates and signs, and states its own effective
sample size beside every one of them.

WHAT IT MEASURES

For each name that cleared a screen on an as-of date, its forward total return
at 1m/3m/6m/12m, pooled, against two nulls drawn from the SAME universe on the
SAME date. No position sizing, no rebalancing, no transaction costs: each is a
free parameter that would dominate a five-observation sample and say nothing
about whether the gates work.

THE TWO NULLS, because one cannot test both halves

  gate test   gate-clean names        vs the liquid universe
              -> do the GATES select?
  score test  published (score >= cut) vs gate-clean-but-below-cut
              -> does the SCORE rank within the gated set?

The score test is the sharper comparison and the harder to argue with, but
both of its arms are gate-clean, so it cannot evaluate the gates at all — and
the gates are the part this project has invested most in. Its null set is
already computed every run as `near_miss`.

WHY A PAIRED SIGN TEST

Forward windows from quarterly dates overlap 75% at the 12m horizon, so 23
as-of dates carry about FIVE independent 12m observations. Names are not
independent either: 20 names on one date share market and sector factors. The
test that survives this is paired and same-date — on how many dates did the
screen beat its matched null? — which cancels the date and market factor
entirely. It needs 17 of 23 to reject at 5%, and 6 of 6 at the 12m horizon.

It can detect a consistent sign. It cannot estimate a magnitude.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import backtest_cache as cache

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}

# Trading days a forward window needs before two observations stop sharing it.
# A 12m return measured quarterly overlaps the next three observations, so the
# independent count is the SPAN divided by the horizon, not the date count.
QUARTER_DAYS = 63

HEADLINE = ("This can detect a consistent sign, not estimate a magnitude. "
            "Every figure below carries its effective sample size; where that "
            "is single digits, the figure is an observation and not a result.")


# --------------------------------------------------------------- the universe

def universe_at(as_of: date, *, min_price: float = 3.0) -> list[str]:
    """
    Every symbol FINRA saw trading on or just before `as_of`.

    This is the survivorship fix, and it is a real one: the archived daily
    off-exchange file is a free source of historical constituents, back to at
    least January 2019. The 2021-06-15 file still carries ATVI, TWTR, VMW,
    SIVB, CERN, XLNX, ZNGA and BBBY — every name a current snapshot has
    already dropped. A traded-symbol list is also the right shape here, since
    the universe this project screens is "liquid US equities", not an index.
    """
    from . import finra_darkpool as fd

    d, tries = as_of, 0
    while tries < 7:
        day = fd.fetch_finra_day(d)
        if len(day):
            return sorted(day["symbol"].unique())
        d -= timedelta(days=1)
        tries += 1
    raise RuntimeError(f"no FINRA file within a week of {as_of}")


def price_covers(px: pd.DataFrame, as_of: date,
                 *, need_days: int = 60, max_gap_days: int = 10) -> bool:
    """
    Does this price series actually describe the company at `as_of`?

    The ticker-reuse guard, and it is the one that stops a silent wrong
    answer. yfinance returns NOTHING for most delisted tickers — 8 of 10
    tested — which is a clean failure. The dangerous case is the other two:
    BBBY returns 55 rows beginning 2026-07-17 and SBNY 535 rows from
    2024-08-15, because both tickers were REUSED after the original company
    died. Joining 2021 fundamentals to those prices produces a plausible
    number about two different companies, with no error raised anywhere.

    So presence is not the test — coverage is. A series must have real history
    running UP TO `as_of`, or the symbol is treated as unreachable.

    `need_days` is one quarter, deliberately short: this is a reuse guard, not
    a history requirement. How much history a MEASURE needs is already decided
    where that measure is built — `ev_history_degraded` for the percentiles,
    `panel_sessions_required` for the dark pool — and duplicating it here
    would silently exclude a recent listing for the wrong reason.
    """
    if px is None or px.empty:
        return False
    ts = pd.Timestamp(as_of)
    before = px.loc[:ts]
    if len(before) < need_days:
        return False
    # The last observation before as_of must be CLOSE to as_of. A series that
    # ends in 2019 and resumes in 2026 covers neither side of a 2021 date.
    gap = (ts - before.index.max()).days
    return gap <= max_gap_days


# ------------------------------------------------------------ exit classifier

# A company leaves the tape for reasons that point in OPPOSITE directions, and
# the survivorship bias cannot be signed until they are told apart. An
# acquisition pays a premium, so missing it biases a screen's measured return
# DOWN; a bankruptcy biases it UP. For screens selecting solvent, high-ROIC,
# low-leverage names the acquisition case is the more likely exit, which is why
# "survivorship inflates everything" is not safe to assume here.
#
# EDGAR resolves 39 of the 3,480 departed symbols, so most exits cannot be
# classified from filings at all. What IS cheap is the shape of the exit in the
# data we already have: the last price before disappearance, relative to the
# price a quarter earlier.
EXIT_ACQUIRED = "acquired or merged"
EXIT_DISTRESS = "delisted distressed"
EXIT_UNKNOWN = "unknown"

# Measured against the symbol's own TRAILING-YEAR HIGH, not against the price
# a fixed quarter earlier. The first version used a 63-day lookback and
# mis-sorted both test shapes: a collapse spread over eighteen months shows a
# mild final quarter and read as "unknown", while a slow -35% fade read as
# "acquired". The distinguishing feature is not the recent slope, it is where
# the tape ENDS relative to where the company had been trading — an
# acquisition closes at or near the deal price, which is usually the high,
# and 0.90 rather than 0.80 because a STEADILY FADING series also ends close
# to its own (falling) year high — a linear -35% drift read as "acquired" at
# 0.80, since its trailing-year max is only ~20% above where it stops.
# and a distressed delisting ends far below it.
_ACQUIRED_OF_HIGH = 0.90      # ends within 10% of its own year high
_DISTRESS_OF_HIGH = 0.40      # ends at or below 40% of it


def classify_exit(px: pd.DataFrame | None, last_seen: date | None) -> str:
    """
    Rough three-way split of why a symbol left the tape.

    Deliberately rough. A bounded, counted split into acquired / distressed /
    unknown says more than a generic caveat, because it bounds the bias
    instead of noting it. It is NOT evidence about any individual name.
    """
    if px is None or px.empty or last_seen is None:
        return EXIT_UNKNOWN
    close = px["close"].dropna()
    tail = close.loc[:pd.Timestamp(last_seen)]
    if len(tail) < QUARTER_DAYS:
        return EXIT_UNKNOWN
    final = float(tail.iloc[-1])
    high = float(tail.iloc[-HORIZONS["12m"]:].max())
    if high <= 0:
        return EXIT_UNKNOWN
    ratio = final / high
    if ratio <= _DISTRESS_OF_HIGH:
        return EXIT_DISTRESS
    if ratio >= _ACQUIRED_OF_HIGH:
        return EXIT_ACQUIRED
    return EXIT_UNKNOWN


# ------------------------------------------------------------ forward returns

def forward_return(px: pd.DataFrame, as_of: date, days: int,
                   dividends: pd.Series | None = None) -> float | None:
    """
    TOTAL return over `days` trading sessions after `as_of`.

    Total, not price: a dividend screen measured on price return is measured
    on everything EXCEPT the component it selects for.
    """
    if px is None or px.empty:
        return None
    close = px["close"].dropna()
    ts = pd.Timestamp(as_of)
    before, after = close.loc[:ts], close.loc[ts:]
    if before.empty or len(after) < 2:
        return None
    start = float(before.iloc[-1])
    end_idx = min(days, len(after) - 1)
    end = float(after.iloc[end_idx])
    if start <= 0:
        return None
    # A window that could not run its full length is NOT a short return, it is
    # an absent one. Treating a truncated window as a result would read a
    # delisting as a flat quarter.
    if end_idx < days:
        return None
    div = 0.0
    if dividends is not None and len(dividends):
        window = dividends.loc[ts:after.index[end_idx]]
        div = float(window.sum())
    return (end + div) / start - 1.0


# ------------------------------------------------------------------ the stats

@dataclass
class Paired:
    """One screen, one horizon: the paired per-date comparison."""
    screen: str
    horizon: str
    test: str
    dates: list = field(default_factory=list)
    screen_mean: list = field(default_factory=list)
    null_mean: list = field(default_factory=list)
    n_screen: list = field(default_factory=list)
    n_null: list = field(default_factory=list)

    def add(self, d: date, screen_rets: list, null_rets: list) -> None:
        s = [r for r in screen_rets if r is not None]
        n = [r for r in null_rets if r is not None]
        if not s or not n:
            return
        self.dates.append(d)
        self.screen_mean.append(float(np.mean(s)))
        self.null_mean.append(float(np.mean(n)))
        self.n_screen.append(len(s))
        self.n_null.append(len(n))

    # ---- the honest sample size ------------------------------------------
    @property
    def span_days(self) -> int:
        if len(self.dates) < 2:
            return 0
        return (max(self.dates) - min(self.dates)).days

    @property
    def effective_n(self) -> int:
        """
        Non-overlapping forward windows, NOT the number of as-of dates.

        Quarterly dates with a 12m horizon overlap 75%, so 23 dates carry
        about five independent observations. Reporting 23 there would imply
        four and a half times the precision the data has.
        """
        h = HORIZONS[self.horizon]
        if not self.dates:
            return 0
        trading_span = self.span_days * 252 / 365.25
        return max(1, min(len(self.dates), int(trading_span // h)))

    @property
    def wins(self) -> int:
        return sum(1 for a, b in zip(self.screen_mean, self.null_mean) if a > b)

    def sign_test_p(self) -> float | None:
        """
        Two-sided binomial p for `wins` of `len(dates)` under p=0.5.

        Assumption-light on purpose. A t-test here would assume normal,
        independent date effects; the sign test only assumes each date's
        comparison is an independent coin flip, which the pairing buys.
        """
        n = len(self.dates)
        if n == 0:
            return None
        k = self.wins
        c = lambda a, b: math.comb(a, b)
        tail = sum(c(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
        return min(1.0, 2 * tail)

    @property
    def too_thin(self) -> bool:
        """Under 8 independent observations nothing may be claimed."""
        return self.effective_n < 8

    def summary(self) -> dict:
        if not self.dates:
            return {"screen": self.screen, "horizon": self.horizon,
                    "test": self.test, "dates": 0, "effective_n": 0,
                    "verdict": "no dates produced a comparable pair"}
        diff = [a - b for a, b in zip(self.screen_mean, self.null_mean)]
        return {
            "screen": self.screen, "horizon": self.horizon, "test": self.test,
            "dates": len(self.dates),
            "effective_n": self.effective_n,
            "wins": self.wins,
            "hit_rate": round(self.wins / len(self.dates), 3),
            "median_excess": round(float(np.median(diff)), 4),
            "mean_excess": round(float(np.mean(diff)), 4),
            "worst_date_excess": round(float(np.min(diff)), 4),
            # Withheld, not just unmentioned: printing p=0.001 next to "five
            # observations, not enough to support a claim" lets the reader
            # take the number and discard the sentence.
            "sign_test_p": (None if self.too_thin
                            else round(self.sign_test_p(), 4)),
            "median_excess_withheld": self.too_thin,
            "names_per_date": round(float(np.mean(self.n_screen)), 1),
            "verdict": self.verdict(),
        }

    def verdict(self) -> str:
        n, eff = len(self.dates), self.effective_n
        p = self.sign_test_p()
        if eff < 8:
            return (f"{eff} independent observations — not enough to support a "
                    f"claim in either direction")
        if p is not None and p < 0.05:
            # "beat the null on 4 of 19 dates (below)" was self-contradictory
            # — it read as a win while the parenthesis said the opposite, on
            # the one line where the direction is the whole result.
            if self.wins * 2 > n:
                return (f"consistent sign: beat the null on {self.wins} of "
                        f"{n} dates, p={p:.3f}")
            return (f"consistent sign, WRONG WAY: lost to the null on "
                    f"{n - self.wins} of {n} dates, p={p:.3f}")
        return f"no consistent sign: {self.wins} of {n} dates, p={p:.3f}"


# ------------------------------------------------------------------ the arms

# Each screen's scorer, its cut, and the Fundamentals attribute that scopes it.
# Reusing the scorers rather than reimplementing the gates is the point: a
# second copy of a gate is a second thing to keep in step, and this project has
# already paid for that with the rule tables drifting from the gates they
# describe.
SCREENS = {
    "dividend":  {"engine": "dividend",  "min_score": 60},
    "value":     {"engine": "quality",   "min_score": 60},
    "recovery":  {"engine": "recovery",  "min_score": 50},
}


def arms(records: list, screen: str) -> dict:
    """
    The three populations a screen defines on one date.

    published    cleared every gate AND scored at or above the cut
    near_miss    cleared every gate and scored BELOW the cut  <- score null
    universe     everything scored, gate failures included    <- gate null
    """
    from . import screeners as sc

    spec = SCREENS[screen]
    scorer = {"dividend": sc.score_dividend,
              "quality": sc.score_quality_value,
              "recovery": lambda f: sc.score_recovery(f, 40.0, 1.2, 1.6)}[spec["engine"]]

    results = []
    for f in records:
        try:
            results.append(scorer(f))
        except Exception:                            # noqa: BLE001
            continue
    clean = [r for r in results if not r["gates_failed"]]
    cut = spec["min_score"]
    return {
        "published":  [r["symbol"] for r in clean if r["score"] >= cut],
        "near_miss":  [r["symbol"] for r in clean if r["score"] < cut],
        "gate_clean": [r["symbol"] for r in clean],
        "universe":   [r["symbol"] for r in results],
    }


# ------------------------------------------------- the empty-result preflight

class EmptyBoardError(RuntimeError):
    """A screen produced no rows at a historical date."""


# Measured 4 Oct 2026 across three dates: `ev_ebit` is present for 69-72% of
# records at 2020-12-31, 2021-06-30 and 2022-06-30 alike. When the as-of path
# was broken it was present for ZERO. So the ranking input's availability
# separates the two cases cleanly, and a floor of 40% sits far from both.
_INPUT_FLOOR = 0.40


def input_health(records: list) -> dict:
    """
    Are the RANKING INPUTS present, independent of what the screens published?

    This is the evidence that separates "the market offered nothing" from "the
    as-of path is broken", and it has to be measured rather than guessed from
    universe size. `ev_ebit` is the right witness because it is what both the
    value and recovery screens rank on, and because its absence was the
    original failure: with the price frame unsliced it was None for EVERY
    record at every historical date.
    """
    n = len(records) or 1
    have = sum(1 for f in records if getattr(f, "ev_ebit", None) is not None)
    deg = sum(1 for f in records if getattr(f, "ev_history_degraded", False))
    return {"records": len(records), "ev_ebit_present": have,
            "ev_ebit_rate": round(have / n, 3),
            "ev_history_degraded": deg, "sound": have / n >= _INPUT_FLOOR}


def preflight(as_of: date, records: list, *, require: int = 3) -> dict:
    """
    Decide whether an empty board is a FINDING or a FAILURE, from evidence.

    A backtest that returns an empty board reads exactly like one that returns
    no signal, and the second is a finding while the first is a bug. But the
    first version of this raised on ANY empty board and guessed the cause from
    universe size — which discarded all four 2021 dates and told them the
    as-of path had failed. It had not: `ev_ebit` was present for 69% of
    records, identically to 2020 and 2022, while the value screen's
    `discount_to_own_history` averaged 11 against 50 a year later. Mid-2021
    was the most expensive market in the sample and a value screen SHOULD
    publish nothing at a top. The screen was working; my diagnosis was not.

    So: raise only when the ranking inputs have actually collapsed. Otherwise
    record the empty board with the numbers behind it and let the sweep carry
    on — a date that publishes nothing contributes nothing to a paired test
    either way, and `Paired.add` already skips it. What must never happen is
    an empty board becoming "no signal" SILENTLY; a measured reason satisfies
    that, and raising unconditionally throws away real observations.
    """
    found, short = {}, []
    for screen in SCREENS:
        a = arms(records, screen)
        found[screen] = {k: len(v) for k, v in a.items()}
        if len(a["published"]) < require:
            short.append(f"{screen}: {len(a['published'])} published "
                         f"({len(a['gate_clean'])} gate-clean of "
                         f"{len(a['universe'])} scored)")

    health = input_health(records)
    found["_input_health"] = health
    found["_empty"] = short

    if short and not health["sound"]:
        raise EmptyBoardError(
            f"at as_of={as_of} these screens produced no usable board AND the "
            f"ranking inputs have collapsed, so a null result here would be a "
            f"silent failure rather than a finding: {'; '.join(short)}. "
            f"ev_ebit is present for only {health['ev_ebit_present']} of "
            f"{health['records']} records ({health['ev_ebit_rate']:.0%}, floor "
            f"{_INPUT_FLOOR:.0%}) — check that the price frame is sliced and "
            f"that ev_ebit survives the as-of build")
    if short:
        print(f"  [note] {as_of}: {'; '.join(short)} — but the inputs are "
              f"sound (ev_ebit present for {health['ev_ebit_rate']:.0%} of "
              f"{health['records']}), so this is a market that offered these "
              f"screens nothing, not a failure. Recorded as a real zero.")
    return found


# ------------------------------------------------------ survivorship, counted

def last_seen_map(start: date, end: date, *, step_days: int = 30) -> dict:
    """
    The last date each symbol appeared in a sampled FINRA file.

    Sampled monthly rather than daily: this exists to bound the survivorship
    bias, and a month's resolution on an exit date is ample for that. Daily
    would be ~1,500 fetches to refine a number whose own confidence is far
    coarser than its precision.
    """
    from . import finra_darkpool as fd

    seen: dict = {}
    d = start
    while d <= end:
        day = fd.fetch_finra_day(d)
        if len(day):
            for s in day["symbol"].unique():
                seen[s] = d
        d += timedelta(days=step_days)
    return seen


def survivorship_report(symbols: list[str], last_seen: dict,
                        as_of: date, *, tail: date,
                        sample: int | None = 400, seed: int = 0,
                        live: bool = True) -> dict:
    """
    How many of the names a screen could have picked are now unreachable, and
    which way that cuts.

    A generic caveat cannot be acted on. A count, split by the SHAPE of the
    exit, bounds the bias instead.

    TWO things this has to keep separate, and the first version did not:

      * NOT CACHED is not a survivorship fact. The cache holds today's
        universe, so reading it alone would count every 2021 symbol we simply
        never fetched as delisted — turning a cache boundary into a finding.
        So an uncached symbol is fetched LIVE, and a symbol that cannot be
        resolved either way is counted as `unresolved` rather than as an exit.
      * The as-of FINRA universe is ~9,700 names and fetching all of them
        costs hours, so this SAMPLES and reports the sample size. A bounded
        estimate that says it is an estimate beats an exact number nobody will
        wait for.
    """
    from . import free_sources as free

    pool = list(symbols)
    if sample is not None and len(pool) > sample:
        rng = np.random.default_rng(seed)
        pool = [pool[i] for i in rng.choice(len(pool), sample, replace=False)]

    out = {"as_of_universe": len(symbols), "sampled": len(pool),
           "reachable": 0, "unreachable": 0, "unresolved": 0,
           "exits": {EXIT_ACQUIRED: 0, EXIT_DISTRESS: 0, EXIT_UNKNOWN: 0},
           "unreachable_symbols": []}

    for s in pool:
        px = None
        try:
            px = cache.prices(s)
        except Exception:                            # noqa: BLE001
            if live:
                try:
                    px = free.equity_ohlcv(s)
                except Exception:                    # noqa: BLE001
                    px = None
            else:
                out["unresolved"] += 1
                continue

        if px is not None and price_covers(px, as_of) and \
                forward_return(px, as_of, HORIZONS["12m"]) is not None:
            out["reachable"] += 1
            continue
        out["unreachable"] += 1
        if len(out["unreachable_symbols"]) < 60:
            out["unreachable_symbols"].append(s)
        out["exits"][classify_exit(px, last_seen.get(s, tail))] += 1
    return out


# ----------------------------------------------------------------- the driver

# Bounded at three ends, and all three are real:
#   lower  XBRL facts start ~FY2010, so a full ten-year EV/EBIT percentile is
#          only available from ~2020. Earlier dates would compute the SAME
#          named measure over a shorter window, which changes what it means
#          between dates rather than just adding noise.
#   lower  the archived FINRA daily file reaches back to at least Jan 2019.
#   upper  a 12m forward return needs twelve months of tape after the date.
WINDOW_START = date(2020, 1, 1)


def as_of_dates(*, today: date | None = None, quarters: bool = True) -> list:
    """Quarter-ends from WINDOW_START to the last date with a full 12m ahead."""
    today = today or date.today()
    end = today - timedelta(days=366)
    freq = "QE" if quarters else "ME"
    idx = pd.date_range(WINDOW_START, pd.Timestamp(end), freq=freq)
    return [d.date() for d in idx]


def build_at(as_of: date, symbols: list[str], *, verbose: bool = False) -> list:
    """
    Build Fundamentals as of `as_of` from the CACHE only.

    Cache-only, and it has to be ACTUALLY cache-only. A live fetch mixed into
    one date's build would give that date inputs pulled weeks apart from every
    other date's, and nothing downstream could see it. This function claimed
    that and was not: sector and SIC came from the SEC submissions endpoint,
    one request per symbol, lru_cached only within a process — so a sweep
    across four parallel processes made ~1,446 live requests per process on
    its first date and stalled on the rate limiter before any date finished.
    `cache.warm_meta()` now stores them beside the facts.
    """
    from . import fundamentals_builder as fb

    # Loaded ONCE, not per symbol: sector and SIC come from a live SEC request
    # otherwise, which is what made this function's cache-only claim false.
    meta = cache.load_meta()
    out, skipped = [], {"no_cache": 0, "no_price_coverage": 0,
                        "no_meta": 0, "build": 0}
    for sym in symbols:
        if not cache.has(sym):
            skipped["no_cache"] += 1
            continue
        try:
            px = cache.prices(sym)
        except Exception:                            # noqa: BLE001
            skipped["no_cache"] += 1
            continue
        if not price_covers(px, as_of):
            skipped["no_price_coverage"] += 1
            continue
        m = meta.get(sym)
        if m is None:
            skipped["no_meta"] += 1
            continue
        try:
            out.append(fb.build(sym, cache.facts(sym), px, as_of=as_of,
                                sector=m.get("sector", "general"),
                                splits=cache.splits(sym),
                                sic=int(m.get("sic") or 0)))
        except Exception:                            # noqa: BLE001
            skipped["build"] += 1
    if verbose:
        print(f"  {as_of}: {len(out)} records, skipped {skipped}")
    return out, skipped


def run(dates: list | None = None, *, universe: list[str] | None = None,
        verbose: bool = True) -> dict:
    """
    The whole measurement. Returns a report; writes nothing.

    Order matters: the preflight runs FIRST, on the earliest date, and raises
    if any screen cannot produce a board. Computing results and then noticing
    the boards were empty is how a bug becomes a published null result.
    """
    dates = dates or as_of_dates()
    if not dates:
        raise RuntimeError("no as-of date has a full 12m of forward tape")

    paired: dict = {}
    for screen in SCREENS:
        for test, null in (("gate", "universe"), ("score", "near_miss")):
            for h in HORIZONS:
                paired[(screen, test, h)] = Paired(screen, h, test)

    per_date, checked = [], False
    for as_of in dates:
        recs, skipped = build_at(as_of, universe or universe_at(as_of),
                                 verbose=verbose)
        if not recs:
            per_date.append({"as_of": str(as_of), "records": 0,
                             "note": "no records built"})
            continue
        if not checked:
            preflight(as_of, recs)          # raises on an empty board
            checked = True

        row = {"as_of": str(as_of), "records": len(recs), "skipped": skipped}
        for screen in SCREENS:
            a = arms(recs, screen)
            row[screen] = {k: len(v) for k, v in a.items()}
            rets = {}
            for group in ("published", "near_miss", "universe"):
                rets[group] = {
                    h: [forward_return(cache.prices(s), as_of, d)
                        for s in a[group] if cache.has(s)]
                    for h, d in HORIZONS.items()}
            for test, null in (("gate", "universe"), ("score", "near_miss")):
                arm = "gate_clean" if test == "gate" else "published"
                src = ("published" if arm == "published" else "universe")
                # The gate test's own arm is every gate-clean name, which is
                # published + near_miss, so it is assembled rather than reused.
                if test == "gate":
                    pos = {h: [forward_return(cache.prices(s), as_of, d)
                               for s in a["gate_clean"] if cache.has(s)]
                           for h, d in HORIZONS.items()}
                else:
                    pos = rets["published"]
                for h in HORIZONS:
                    paired[(screen, test, h)].add(as_of, pos[h], rets[null][h])
        row["hole"] = obs.get("hole")
        per_date.append(row)

    holes = [r["hole"] for r in per_date if r.get("hole")]
    hole = None
    if holes:
        ex = {EXIT_ACQUIRED: 0, EXIT_DISTRESS: 0, EXIT_UNKNOWN: 0}
        for h in holes:
            for k, v in h["exits"].items():
                ex[k] = ex.get(k, 0) + v
        hole = {
            "dates": len(holes),
            "universe_mean": round(sum(h["universe"] for h in holes) / len(holes)),
            "measured_mean": round(sum(h["measured"] for h in holes) / len(holes)),
            "unmeasured_rate_mean": round(
                sum(h["unmeasured_rate"] for h in holes) / len(holes), 3),
            "exits": ex,
        }

    return {
        "headline": HEADLINE,
        "hole": hole,
        "window": {"from": str(dates[0]), "to": str(dates[-1]),
                   "as_of_dates": len(dates)},
        "cache_built_at": cache.load_manifest().get("built_at"),
        "results": [paired[k].summary() for k in sorted(paired)],
        "per_date": per_date,
    }


# ------------------------------------- measurement, separated from statistics

OBS_DIR = Path("data/backtest/obs")

# The expensive half is building 1,446 records at each of 23 dates — measured
# 0.56s per name, so ~13 minutes per date and ~5 hours for the sweep. The
# cheap half is the statistics over the result.
#
# Keeping them together would mean re-running five hours of builds to change a
# null, add a horizon, or correct a threshold — which is how an analysis
# quietly stops being re-run and starts being trusted. So each date writes its
# per-name forward returns and arm membership ONCE, and the statistics are
# assembled from those files every time.
#
# It also makes the sweep resumable: a crash at date 19 costs one date.


def measure_date(as_of: date, universe: list[str] | None = None,
                 *, candidates: list[str] | None = None,
                 force: bool = False, verbose: bool = True) -> dict:
    """
    Build at `as_of`, record arm membership and forward returns, save.

    `candidates` is the FULL as-of universe before any filtering, and it is
    what the hole must be measured against. Passing only the already-filtered
    `universe` understated the hole badly: the archive sweep pre-filters 3,000
    candidates to the ~1,590 that can be priced, and a hole computed against
    the 1,500 survivors of that read 12% where the true attrition was 56%. A
    denominator that already excludes the problem is the same shape as a
    p-value printed beside five observations — it reads settled because what
    would overturn it was removed before counting.
    """
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OBS_DIR / f"{as_of}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    syms = universe if universe is not None else universe_at(as_of)
    recs, skipped = build_at(as_of, syms, verbose=verbose)
    if not recs:
        raise RuntimeError(f"{as_of}: no records built from {len(syms)} symbols")

    preflight(as_of, recs)

    # ---- the reuse guard, ASSERTED rather than trusted ---------------------
    #
    # `price_covers` is applied in build_at, but a filter that is merely
    # called is not a filter that worked — and this is the one place where
    # being wrong is SILENT. BBBY and SBNY both return a live series for a
    # DIFFERENT company (their tickers were reused after delisting), so a
    # symbol that slipped through would join 2021 fundamentals to 2026 prices
    # and produce a plausible number with nothing raised anywhere. A backtest
    # is precisely where that survives to be believed.
    #
    # So the invariant is re-checked on the records that were ADMITTED, not
    # on the candidates that were filtered: every one must have real tape on
    # both sides of as_of. This cannot be satisfied by the filter being
    # present; only by it having worked.
    ts = pd.Timestamp(as_of)
    bad = []
    for f in recs:
        try:
            px = cache.prices(f.symbol)
        except Exception:                            # noqa: BLE001
            bad.append((f.symbol, "admitted with no price series"))
            continue
        before = px.loc[:ts]
        if before.empty:
            bad.append((f.symbol, f"series starts {px.index.min().date()}, "
                                  f"after as_of"))
        elif (ts - before.index.max()).days > 10:
            bad.append((f.symbol, f"last print {before.index.max().date()}, "
                                  f"{(ts - before.index.max()).days}d before as_of"))
    if bad:
        raise RuntimeError(
            f"{as_of}: {len(bad)} admitted record(s) have a price series that "
            f"does not bracket the as-of date — a reused or stale ticker "
            f"reached the measurement: " +
            "; ".join(f"{s_} ({why})" for s_, why in bad[:6]))

    rets: dict = {}
    for f in recs:
        s = f.symbol
        try:
            px = cache.prices(s)
        except Exception:                            # noqa: BLE001
            continue
        rets[s] = {h: forward_return(px, as_of, d) for h, d in HORIZONS.items()}

    # The hole is recorded HERE, in the same pass that produces the result,
    # so a figure and its bound cannot drift apart. `requested` is the as-of
    # FINRA universe; `records` is what survived to be measured. The gap is
    # the names a screen could have picked and we cannot price — and the
    # classification says which way each exit cuts, because acquired and
    # delisted-distressed point in OPPOSITE directions.
    admitted = {f.symbol for f in recs}
    # Against the FULL as-of universe, not the filtered one handed in.
    denom = list(candidates) if candidates else list(syms)
    missing = [t for t in denom if t not in admitted]
    exits = {EXIT_ACQUIRED: 0, EXIT_DISTRESS: 0, EXIT_UNKNOWN: 0}
    for t in missing:
        px = None
        try:
            px = cache.prices(t)
        except Exception:                            # noqa: BLE001
            px = None
        exits[classify_exit(px, as_of)] += 1

    obs = {"as_of": str(as_of), "requested": len(syms), "records": len(recs),
           "skipped": skipped, "returns": rets,
           "hole": {"universe": len(denom), "measured": len(recs),
                    "handed_in": len(syms),
                    "unmeasured": len(missing),
                    "unmeasured_rate": round(len(missing) / max(len(denom), 1), 3),
                    "exits": exits},
           "arms": {screen: arms(recs, screen) for screen in SCREENS}}
    out_path.write_text(json.dumps(obs))
    if verbose:
        pub = {k: len(v["published"]) for k, v in obs["arms"].items()}
        print(f"  {as_of}: {len(recs)} records, published {pub}")
    return obs


def saved_dates() -> list:
    if not OBS_DIR.exists():
        return []
    return sorted(date.fromisoformat(p.stem) for p in OBS_DIR.glob("*.json"))


def combine(dates: list | None = None) -> dict:
    """Assemble the report from saved observations. Cheap; re-runnable."""
    dates = dates or saved_dates()
    if not dates:
        raise RuntimeError("no saved observations — run measure_date first")

    paired: dict = {}
    for screen in SCREENS:
        for test in ("gate", "score"):
            for h in HORIZONS:
                paired[(screen, test, h)] = Paired(screen, h, test)

    per_date = []
    for d in dates:
        obs = json.loads((OBS_DIR / f"{d}.json").read_text())
        rets = obs["returns"]
        get = lambda names, h: [rets[s][h] for s in names if s in rets]
        row = {"as_of": obs["as_of"], "records": obs["records"],
               "requested": obs["requested"], "skipped": obs["skipped"]}
        for screen, a in obs["arms"].items():
            row[screen] = {k: len(v) for k, v in a.items()}
            for test, pos_arm, null_arm in (
                    ("gate", "gate_clean", "universe"),
                    ("score", "published", "near_miss")):
                for h in HORIZONS:
                    paired[(screen, test, h)].add(
                        d, get(a[pos_arm], h), get(a[null_arm], h))
        row["hole"] = obs.get("hole")
        per_date.append(row)

    holes = [r["hole"] for r in per_date if r.get("hole")]
    hole = None
    if holes:
        ex = {EXIT_ACQUIRED: 0, EXIT_DISTRESS: 0, EXIT_UNKNOWN: 0}
        for h in holes:
            for k, v in h["exits"].items():
                ex[k] = ex.get(k, 0) + v
        hole = {
            "dates": len(holes),
            "universe_mean": round(sum(h["universe"] for h in holes) / len(holes)),
            "measured_mean": round(sum(h["measured"] for h in holes) / len(holes)),
            "unmeasured_rate_mean": round(
                sum(h["unmeasured_rate"] for h in holes) / len(holes), 3),
            "exits": ex,
        }

    return {
        "headline": HEADLINE,
        "hole": hole,
        "window": {"from": str(dates[0]), "to": str(dates[-1]),
                   "as_of_dates": len(dates)},
        "cache_built_at": cache.load_manifest().get("built_at"),
        "results": [paired[k].summary() for k in sorted(paired)],
        "per_date": per_date,
    }


# ---------------------------------------------------------------- the report

def format_report(rep: dict) -> str:
    """
    Text report. The headline is the FIRST thing, not a footnote.

    "This can detect a consistent sign, not estimate a magnitude" is the
    result's own error bar, so it is printed before any number — and the
    effective sample size sits in the same row as every figure rather than in
    a note at the bottom, because a reader who sees `median_excess +2.1%`
    without `effective_n 5` beside it has been told something untrue.
    """
    L = ["=" * 78, "DOES THE RANKING CARRY INFORMATION?", "=" * 78, ""]
    for line in _wrap(rep["headline"], 78):
        L.append(line)
    L.append("")
    w = rep["window"]
    L.append(f"window      {w['from']} to {w['to']}  ({w['as_of_dates']} as-of dates)")
    L.append(f"inputs      cached {rep.get('cache_built_at') or 'unknown'}")
    L.append("")

    # The hole goes BESIDE the result, not after it. A gate figure with an
    # unquantified 35% of its universe unpriceable is the same shape as a
    # p-value printed next to five observations: a number that reads as
    # settled while the thing that could overturn it sits out of frame.
    hole = rep.get("hole")
    L.append("-" * 78)
    L.append("THE HOLE — what the gate figures below cannot see")
    L.append("-" * 78)
    if not hole:
        L.append("  NOT MEASURED. The gate test compares against the universe,")
        L.append("  so an unquantified hole in that universe is an unbounded")
        L.append("  error on every gate row. Treat them as unreported.")
    else:
        u, m = hole["universe_mean"], hole["measured_mean"]
        L.append(f"  as-of universe, mean per date   {u:>6,}")
        L.append(f"  measured                        {m:>6,}")
        L.append(f"  UNMEASURED                      {u - m:>6,}  "
                 f"({hole['unmeasured_rate_mean']:.1%} of the universe)")
        L.append("")
        # Per date, not summed. Summed across 23 dates these read as 9,209
        # companies when they are 9,209 date-symbol observations of roughly
        # 400 names — a name missing at every date is counted 23 times.
        nd = hole.get("dates") or 1
        L.append(f"  and which way each exit cuts, mean per date "
                 f"(over {nd} dates):")
        ex = {k: v / nd for k, v in hole["exits"].items()}
        tot = sum(ex.values()) or 1
        for k, bias in ((EXIT_ACQUIRED, "biases the screen DOWN (premium missed)"),
                        (EXIT_DISTRESS, "biases the screen UP (zero missed)"),
                        (EXIT_UNKNOWN, "direction unknown")):
            v = ex.get(k, 0)
            L.append(f"    {k:<22s}{v:>7,.0f}  {100*v/tot:>5.1f}%   {bias}")
        net = ex.get(EXIT_ACQUIRED, 0) - ex.get(EXIT_DISTRESS, 0)
        unk = ex.get(EXIT_UNKNOWN, 0)
        L.append("")
        L.append(f"  net of the two signed exits: {net:+,.0f} per date toward "
                 f"{'UNDERSTATING' if net > 0 else 'OVERSTATING'} the screens")
        L.append(f"  BUT only {100*(1 - unk/tot):.0f}% of the hole can be "
                 f"signed at all — {unk:,.0f} per date are unknown, so this "
                 f"bounds the direction weakly, not the magnitude")
    L.append("")
    L.append("-" * 78)
    L.append("GATE TEST — gate-clean names vs the liquid universe, same date")
    L.append("SCORE TEST — published vs gate-clean-but-below-cut, same date")
    L.append("-" * 78)
    hdr = (f"{'screen':10s} {'test':6s} {'hz':4s} {'dates':>5} {'effN':>5} "
           f"{'wins':>5} {'hit':>5} {'medXS':>8} {'p':>6}  verdict")
    L.append(hdr)
    for r in rep["results"]:
        if not r.get("dates"):
            L.append(f"{r['screen']:10s} {r['test']:6s} {r['horizon']:4s} "
                     f"{'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>8} {'-':>6}  "
                     f"{r['verdict']}")
            continue
        thin = r.get("sign_test_p") is None
        pcol = f"{'  --':>6}" if thin else f"{r['sign_test_p']:>6.3f}"
        xs = f"{'  --':>8}" if thin else f"{r['median_excess']:>8.4f}"
        L.append(f"{r['screen']:10s} {r['test']:6s} {r['horizon']:4s} "
                 f"{r['dates']:>5} {r['effective_n']:>5} {r['wins']:>5} "
                 f"{r['hit_rate']:>5.2f} {xs} {pcol}  {r['verdict']}")

    surv = rep.get("survivorship")
    if surv:
        L += ["", "-" * 78,
              "SURVIVORSHIP — counted and split, because the two exits point "
              "opposite ways", "-" * 78]
        for line in _wrap(
                "An acquisition pays a premium, so missing it biases a "
                "screen's measured return DOWN. A bankruptcy biases it UP. "
                "For screens selecting solvent, high-ROIC, low-leverage "
                "names the acquisition case is the more likely exit, so "
                "'survivorship inflates everything' is not safe to assume "
                "here. These are counts, not attributions of any one name.",
                78):
            L.append(line)
        L.append("")
        L.append(f"  as-of universe      {surv['total']:>6,}")
        L.append(f"  reachable today     {surv['reachable']:>6,} "
                 f"({100*surv['reachable']/max(surv['total'],1):.1f}%)")
        L.append(f"  unreachable         {surv['unreachable']:>6,} "
                 f"({100*surv['unreachable']/max(surv['total'],1):.1f}%)")
        for k, v in surv["exits"].items():
            L.append(f"    {k:<22s}{v:>6,}")
    return "\n".join(L)


def _wrap(text: str, width: int) -> list:
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def run_survivorship(as_of: date, *, tail: date | None = None,
                     step_days: int = 30, **kw) -> dict:
    """
    Count what a screen could have picked at `as_of` and can no longer reach.

    Uses the ARCHIVED FINRA universe, which is the survivorship fix and a real
    one: the daily off-exchange file reaches back to at least January 2019,
    and the 2021-06-15 file still carries ATVI, TWTR, VMW, SIVB, CERN, XLNX
    and BBBY — every name a current snapshot has already dropped.
    """
    tail = tail or date.today()
    syms = universe_at(as_of)
    seen = last_seen_map(as_of, tail, step_days=step_days)
    rep = survivorship_report(syms, seen, as_of, tail=tail, **kw)
    rep["as_of"] = str(as_of)
    rep["last_seen_sampled_every_days"] = step_days
    return rep


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="backtest harness")
    ap.add_argument("--dates", help="comma-separated as-of dates, or 'all'")
    ap.add_argument("--combine", action="store_true",
                    help="assemble the report from saved observations")
    ap.add_argument("--survivorship", metavar="AS_OF",
                    help="count unreachable names at one as-of date")
    ap.add_argument("--universe", default="data/universe.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.survivorship:
        print(json.dumps(run_survivorship(date.fromisoformat(a.survivorship)),
                         indent=1)[:4000])
    elif a.combine:
        rep = combine()
        Path("data/backtest").mkdir(parents=True, exist_ok=True)
        Path("data/backtest/report.json").write_text(json.dumps(rep, indent=1))
        print(format_report(rep))
    elif a.dates:
        univ = json.loads(Path(a.universe).read_text())["tickers"] \
            if a.universe else None
        want = (as_of_dates() if a.dates == "all"
                else [date.fromisoformat(d) for d in a.dates.split(",")])
        for d in want:
            measure_date(d, univ, force=a.force)
    else:
        ap.print_help()
