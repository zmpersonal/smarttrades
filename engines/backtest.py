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
            "sign_test_p": round(self.sign_test_p(), 4),
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
            d = "above" if self.wins * 2 > n else "below"
            return f"consistent sign: beat the null on {self.wins} of {n} dates ({d}), p={p:.3f}"
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


def preflight(as_of: date, records: list, *, require: int = 3) -> dict:
    """
    Assert every screen produces a NON-EMPTY board before any result is run.

    A backtest that returns an empty board reads exactly like one that returns
    no signal, and the second is a finding while the first is a bug. This
    project has already shipped that failure twice in other forms: `ev_ebit`
    goes None at a past as_of if the price frame is not sliced, which empties
    the value board completely, and `min_score` once silently dropped every
    name that passed every gate.

    So a null result has to be EARNED. Raises rather than returning a verdict,
    because the alternative is a report of "no signal" computed over nothing.
    """
    found, short = {}, []
    for screen in SCREENS:
        a = arms(records, screen)
        found[screen] = {k: len(v) for k, v in a.items()}
        if len(a["published"]) < require:
            short.append(f"{screen}: {len(a['published'])} published "
                         f"({len(a['gate_clean'])} gate-clean of "
                         f"{len(a['universe'])} scored)")
    if short:
        # Say WHICH of the two it is. An empty board on 40 names is a universe
        # too small to publish from; an empty board on 1,400 is the as-of path
        # broken. The production screens publish ~22 of 1,446, so a slate of a
        # few dozen legitimately publishes nothing and that is not a bug —
        # reporting it as one would train the reader to ignore this exception.
        scored = max((f["universe"] for f in found.values()), default=0)
        cause = ("the universe is far below production scale "
                 f"({scored} scored against ~1,400 in a real run), so these "
                 "screens would publish nothing even working perfectly — "
                 "widen the universe or lower `require` deliberately"
                 if scored < 400 else
                 "the universe is at production scale, so this is the as-of "
                 "path failing, not a quiet market — check that the price "
                 "frame is sliced and ev_ebit survives")
        raise EmptyBoardError(
            f"at as_of={as_of} these screens produced no usable board, so a "
            f"null result here would be a silent failure rather than a "
            f"finding: {'; '.join(short)}. Likely cause: {cause}")
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
                        as_of: date, *, tail: date) -> dict:
    """
    How many of the names a screen could have picked are now unreachable, and
    which way that cuts.

    A generic caveat cannot be acted on. A count, split by the SHAPE of the
    exit, bounds the bias instead: an acquisition pays a premium so missing it
    biases the measured return DOWN, while a bankruptcy biases it UP. For
    screens selecting solvent, high-ROIC, low-leverage names the acquisition
    case is the more likely exit, which is exactly why "survivorship inflates
    everything" is not safe to assume here.
    """
    out = {"total": len(symbols), "reachable": 0, "unreachable": 0,
           "exits": {EXIT_ACQUIRED: 0, EXIT_DISTRESS: 0, EXIT_UNKNOWN: 0},
           "unreachable_symbols": []}
    for s in symbols:
        px = None
        try:
            px = cache.prices(s)
        except Exception:                            # noqa: BLE001
            px = None
        if px is not None and price_covers(px, as_of) and \
                forward_return(px, as_of, HORIZONS["12m"]) is not None:
            out["reachable"] += 1
            continue
        out["unreachable"] += 1
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

    Cache-only on purpose. A live fetch mixed into one date's build would give
    that date inputs pulled weeks apart from every other date's, and nothing
    downstream could see it.
    """
    from . import fundamentals_builder as fb
    from . import free_sources as free

    out, skipped = [], {"no_cache": 0, "no_price_coverage": 0, "build": 0}
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
        try:
            facts = cache.facts(sym)
            out.append(fb.build(sym, facts, px, as_of=as_of,
                                sector=free.company_sector(sym),
                                splits=cache.splits(sym),
                                sic=free.company_sic(sym)))
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
        per_date.append(row)

    return {
        "headline": HEADLINE,
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
                 *, force: bool = False, verbose: bool = True) -> dict:
    """Build at `as_of`, record arm membership and forward returns, save."""
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OBS_DIR / f"{as_of}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    syms = universe if universe is not None else universe_at(as_of)
    recs, skipped = build_at(as_of, syms, verbose=verbose)
    if not recs:
        raise RuntimeError(f"{as_of}: no records built from {len(syms)} symbols")

    preflight(as_of, recs)

    rets: dict = {}
    for f in recs:
        s = f.symbol
        try:
            px = cache.prices(s)
        except Exception:                            # noqa: BLE001
            continue
        rets[s] = {h: forward_return(px, as_of, d) for h, d in HORIZONS.items()}

    obs = {"as_of": str(as_of), "requested": len(syms), "records": len(recs),
           "skipped": skipped, "returns": rets,
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
        per_date.append(row)

    return {
        "headline": HEADLINE,
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
        L.append(f"{r['screen']:10s} {r['test']:6s} {r['horizon']:4s} "
                 f"{r['dates']:>5} {r['effective_n']:>5} {r['wins']:>5} "
                 f"{r['hit_rate']:>5.2f} {r['median_excess']:>8.4f} "
                 f"{r['sign_test_p']:>6.3f}  {r['verdict']}")

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
                     step_days: int = 30) -> dict:
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
    rep = survivorship_report(syms, seen, as_of, tail=tail)
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
