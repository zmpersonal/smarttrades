#!/usr/bin/env python3
"""
SmartTrades.AI — daily runner
=============================

Executes every engine and writes one JSON file per engine into data/.
The dashboard fetches those files; if they are absent it falls back to the
embedded sample data, so the page always renders.

Run locally:      python run_all.py
Run one engine:   python run_all.py --only darkpool
CI:               .github/workflows/daily.yml

Design rule: one engine failing must never take down the others. Each runs
inside its own try/except, writes its own file, and records its own status in
data/status.json. A stale tab is better than a blank dashboard, and the UI
shows each engine's age so stale data is visible rather than silent.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

UTC = timezone.utc
# "details" runs last — it reads the other engines' output to decide which
# symbols are worth computing indicator panels for.
ENGINES = ["darkpool", "dividend", "recovery", "value", "financial", "reit",
           "politicians", "bitcoin", "recession", "details"]

# Not every engine is worth running every day. Fundamentals barely move
# day to day; FINRA and congressional filings do.
CADENCE = {
    "darkpool":    "daily",     # FINRA posts by 6pm ET each trading day
    "politicians": "daily",     # new PTRs land continuously
    "bitcoin":     "daily",
    "dividend":    "weekly",    # Sunday
    "recovery":    "weekly",
    "value":       "weekly",
    "financial":   "weekly",
    "reit":        "weekly",
    "recession":   "daily",     # OAS and curve are daily; the read can turn fast
    "details":     "daily",
}


def read_prior_status() -> dict:
    """
    The last recorded outcome per engine, each stamped with when it happened
    and marked as carried. Unreadable means none, not a crash.

    Carrying an entry forward without saying so was its own bug: Monday's run
    skipped the weekly screeners, kept Sunday's pre-fix traceback with no
    timestamp, and it read as a fresh crash at Monday 00:53 — against code
    that had already been fixed and had not yet run.
    """
    try:
        doc = json.loads((DATA / "status.json").read_text())
    except (OSError, ValueError):
        return {}
    when = doc.get("updated_at")
    out = {}
    for name, entry in (doc.get("engines") or {}).items():
        e = dict(entry)
        if not e.get("at") and e.get("state") != "skipped":
            # Entries written before per-entry timestamps have no time of
            # their own. The file's updated_at is only an UPPER bound — using
            # it as `at` dated a Sunday 19:37 crash to Monday 00:53. Record
            # the bound as a bound, never as the time.
            e.setdefault("at_or_before", when)
        e["carried"] = True
        out[name] = e
    return out


def should_run(engine: str, force: bool) -> bool:
    if force:
        return True
    if CADENCE[engine] == "daily":
        return True
    return datetime.now(UTC).weekday() == 6  # Sunday


def _json_safe(o):
    """NaN and infinity become null — unknown, which is what they mean."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):   # numpy scalar
        try:
            return _json_safe(o.item())
        except (TypeError, ValueError):
            return o
    return o


def dump_json(payload) -> str:
    # Python's json writes bare NaN, which no browser will parse. bitcoin.json
    # carried 14 of them; the dashboard's fetch threw, boot() fell into its
    # catch, and every tab on the live site rendered invented sample rows.
    # allow_nan=False makes a missed case raise here, in the run, not silently
    # in a reader's browser.
    return json.dumps(_json_safe(payload), indent=1, default=str, allow_nan=False)


# Boards whose previous run is kept. The delta is what is worth reading
# weekly — the board itself barely moves — so the rotation happens on EVERY
# run rather than inside the Slack branch, which is where bitcoin's and
# recession's snapshots live and why they only exist when --notify ran.
SNAPSHOT = ("dividend", "recovery", "value", "financial", "reit")


def write(name: str, payload: dict) -> None:
    payload["generated_at"] = datetime.now(UTC).isoformat()
    cur = DATA / f"{name}.json"
    # Rotate BEFORE overwriting. The full board, not {ticker, score}: the
    # moment the question is "its yield moved but its score did not", the slim
    # form is a rebuild, and five files of 20-40 rows is trivial beside
    # universe.json.
    if name in SNAPSHOT and cur.exists():
        (DATA / f"{name}.prev.json").write_text(cur.read_text())
    cur.write_text(dump_json(payload))
    print(f"  wrote data/{name}.json  ({len(payload.get('rows', []))} rows)")


# --------------------------------------------------------------- engines

def run_darkpool() -> dict:
    from engines import finra_darkpool as fd

    end = date.today()
    # Derived from what the components need, not a round number. 150 days
    # yielded ~103 trading days against the 109 rvol_z needs, so relative
    # volume scored a neutral 50 for every symbol in every run.
    start = end - timedelta(days=fd.fetch_calendar_days())
    finra = fd.fetch_finra_range(start, end)
    all_symbols = finra["symbol"].nunique()
    finra, etfs = fd.exclude_etfs(finra, free.load_etf_symbols())

    tape = load_tape(sorted(finra["symbol"].unique()), start, end)
    report: dict = {}
    board = fd.run(finra, tape, report=report)
    report["finra_symbols_before_etf_exclusion"] = int(all_symbols)
    report["etfs_excluded"] = len(etfs)
    print(f"  darkpool: {report['passed']} passed of {report['scored']} scored "
          f"({report['liquid']} liquid, tape for {report['tape_symbols']} of "
          f"{report['finra_symbols']} FINRA symbols); max {report['max']}, "
          f"p90 {report['p90']}, cut {report['min_score']}")
    from engines import dashboard_adapter as da
    titles = free.ticker_titles()
    return {"engine": "darkpool",
            "rows": [da.darkpool_row(r, titles) for r in board.to_dict("records")],
            "funnel": report}


# Set against each screen's OBSERVED distribution, not one number for all
# three. A 1,500-name run showed value reaching 85 and dividend 91, so a 60 cut
# sits mid-distribution and separates a top tier from a long tail. Recovery's
# highest achievable score in the entire universe is 68 — a 60 cut there is at
# 88% of the observed ceiling, admitting seven names while thirty sit in the
# fifties. That is not a quality judgment, it is a threshold set against an
# imagined distribution.
#
# Recovery scores cooler by construction: its components cap lower, and a
# genuine 3x candidate with a clean balance sheet still cannot score like a
# compounder. Re-audit these whenever the component weights change.
#
# Financial, measured 13 Sep 2026 over 152 in-scope names: 55 gate-clean,
# median 53, 75th percentile 62, 90th 68, ceiling 82. A 60 cut sits at 73% of
# the ceiling — the same place recovery's 50 sits against its 68 — and admits
# 19 rows, roughly the upper quartile of what clears the gates.
MIN_SCORE = {"value": 60, "dividend": 60, "recovery": 50, "financial": 60,
             # Measured on the 62-trust cohort, 2 Oct 2026: 15 gate-clean,
             # scores 25 to 90, median 62. A 60 cut sits at 67% of the ceiling
             # — where value and dividend sit against theirs — and publishes 9
             # rows. Re-audit when component weights change.
             "reit": 60}


def _prev_board(which: str):
    """
    The board as it stood on the previous run, and when that was.

    Reads the CURRENT file, because `write` rotates it to .prev.json only at
    the moment it is replaced — so during a run the live file still holds the
    last run's rows. Returns ([], None) on the first ever run, which renders as
    "no previous run to compare" rather than as everything being new.
    """
    f = DATA / f"{which}.json"
    if not f.exists():
        return [], None
    try:
        d = json.loads(f.read_text())
    except Exception:
        return [], None
    return d.get("rows", []) or [], d.get("generated_at")


def _board_delta(which, prev_rows, prev_at, scored, outcome) -> dict:
    """
    What changed since the previous run: joined, dropped WITH A REASON, moved.

    The reason is the point. A dropped name is no longer in the board it left,
    so nothing downstream can explain it — the explanation has to be retained
    while the name is being scored, which is what `outcome` carries.
    """
    now = {f.symbol: res["score"] for f, res in scored}
    prev = {r["ticker"]: r for r in prev_rows}
    if not prev:
        return {"first_run": True, "since": None,
                "joined": [], "dropped": [], "moved": []}

    joined = [{"ticker": t, "score": now[t]}
              for t in now if t not in prev]
    dropped = []
    for t in prev:
        if t in now:
            continue
        o = outcome.get(t)
        if o is None:
            why = ("no longer in this screen's universe — it did not become a "
                   "record this run")
        elif o["gates_failed"]:
            why = o["gates_failed"][0]
        else:
            # Passed every gate and still left: it fell under the cut, which is
            # a different fact from failing a rule and reads differently.
            why = (f"scored {o['score']}, under the cut of "
                   f"{MIN_SCORE.get(which, '?')} — it failed no gate")
        dropped.append({"ticker": t, "prev_score": prev[t].get("score"),
                        "reason": why})
    moved = []
    for t, sc in now.items():
        if t in prev and prev[t].get("score") is not None:
            d = sc - prev[t]["score"]
            if d:
                moved.append({"ticker": t, "from": prev[t]["score"],
                              "to": sc, "delta": d})
    moved.sort(key=lambda m: -abs(m["delta"]))
    return {"first_run": False, "since": prev_at,
            "joined": sorted(joined, key=lambda j: -j["score"]),
            "dropped": dropped, "moved": moved[:12]}


def run_screener(which: str) -> dict:
    """
    Emits rows in the shape index.html renders, not the raw scorer output.

    Those two shapes have never matched: the scorer returns symbol/components/
    gates_failed and the dashboard asks for ticker/roic/fcfy/evp. Writing the
    raw output would render a table of correctly-ranked blanks.
    """
    from engines import screeners as sc
    from engines import dashboard_adapter as da

    universe = load_fundamentals()
    if which == "reit":
        # SIC 6798 and the filings agreeing. The 6500-6599 real-estate block is
        # deliberately absent: those are services businesses on general gates.
        universe = [f for f in universe if f.sector == "reit"]
    if which == "financial":
        # SIC decides scope and the filer's own reporting corroborates it.
        # Names outside scope are not "gated" by this screen — they were never
        # in it — so they are excluded from its funnel rather than counted as
        # failures. Capital strength ranks within sub-bucket, which needs the
        # whole in-scope population in hand before any single name is scored.
        universe = [f for f in universe
                    if f.sector == "financial" and f.financial_in_scope]
        dist = sc.financial_distributions(universe)
    scorer = {"dividend": sc.score_dividend,
              "recovery": lambda f: sc.score_recovery(f, 40.0, 1.2, 1.6),
              "value": sc.score_quality_value,
              "financial": lambda f: sc.score_financial(f, dist),
              "reit": sc.score_reit}[which]

    # Names that were on the board LAST run, so this run can say why any of
    # them left. "ADM dropped" is a fact; "ADM dropped: FCF payout 94% over the
    # 70% cap" is a reason to look or not look. Retaining the outcome for ~22
    # names costs nothing; retaining it for all 1,369 gated ones would.
    prev_rows, prev_at = _prev_board(which)
    prev_syms = {r["ticker"]: r for r in prev_rows}
    outcome = {}

    scored, gated, near, data_gated = [], 0, 0, 0
    never_built = 0                     # set by the loader when it reports
    for f in universe:
        res = scorer(f)
        if f.symbol in prev_syms:
            outcome[f.symbol] = {"score": res["score"],
                                 "gates_failed": res["gates_failed"]}
        if res["gates_failed"]:
            gated += 1
            # Count the reason the SCREEN gave, not a reason it never consulted.
            # financial_gates does not call data_quality_gates at all, so 52 of
            # its 97 exclusions were labelled data-quality while the actual
            # cause was a financial gate.
            if set(res["gates_failed"]) & set(sc.data_quality_gates(f)):
                data_gated += 1
            continue
        if res["score"] < MIN_SCORE[which]:
            near += 1
            continue
        scored.append((f, res))

    scored.sort(key=lambda pair: pair[1]["score"], reverse=True)
    print(f"  {which}: {len(scored)} passed, {near} near-miss, {gated} gated "
          f"of {len(universe)}")
    # The funnel travels with the rows. A row count means nothing without it,
    # and it was written to JSON but never shown on the page.
    # Over ALL passers, not the 40 rows that get emitted. Computing a mean
    # client-side from `rows` is right until a screen passes 41 names and
    # silently wrong after, with nothing to mark the transition — the same
    # shape as every other quiet threshold this project has found.
    all_scores = sorted(r["score"] for _, r in scored)
    stats = {"n": len(all_scores)}
    if all_scores:
        mid = len(all_scores) // 2
        stats.update({
            "mean": round(sum(all_scores) / len(all_scores), 1),
            "median": (all_scores[mid] if len(all_scores) % 2
                       else round((all_scores[mid - 1] + all_scores[mid]) / 2, 1)),
            "min": all_scores[0], "max": all_scores[-1],
            "rows_emitted": min(len(scored), 40),
        })

    return {"engine": which, "rows": da.to_rows(which, scored[:40]),
            "stats": stats,
            "changes": _board_delta(which, prev_rows, prev_at, scored, outcome),
            "funnel": {"universe": len(universe) + never_built,
                       "built": len(universe), "data_gated": data_gated,
                       "business_gated": gated - data_gated,
                       "near_miss": near, "passed": len(scored)},
            "counts": {"universe": len(universe), "passed": len(scored),
                       "near_miss": near, "gated": gated,
                       "min_score": MIN_SCORE[which]}}


def run_politicians() -> dict:
    from engines import capitol_flow as cf

    trades = load_ptrs()
    prices = load_tape_wide()
    board = cf.score_member_alpha(trades, prices)
    return {
        "engine": "politicians",
        "leaderboard": board.reset_index().to_dict("records"),
        "rows": [],  # populate from recent filings via cf.score_disclosure
    }


def run_recession() -> dict:
    from engines import recession as rc

    fred = load_fred(list(rc.FRED.values()))
    hy, bb, ccc = fred["BAMLH0A0HYM2"], fred["BAMLH0A1HYBB"], fred["BAMLH0A3HYC"]

    oas = rc.oas_state(hy)
    disp = rc.quality_dispersion(ccc, bb)
    probit = rc.curve_probit(float(fred["T10Y3M"].iloc[-1]))
    sahm = rc.sahm_state(float(fred["SAHMREALTIME"].iloc[-1]))
    clock = rc.uninversion_clock(load_uninversion_date(), date.today())

    return {
        "engine": "recession",
        "oas": oas, "dispersion": disp, "probit": probit,
        "sahm": sahm, "uninversion": clock,
        # Dispersion is weighted into LEAD (it leads headline HY OAS) and also
        # passed whole, so an extreme reading surfaces as its own flag rather
        # than only as one fifth of an average.
        "composite": rc.composite(
            {"curve_probit": probit["probability"], "uninversion_clock": clock["score"],
             "quality_dispersion": disp["score"],
             "claims_trend": claims_trend_score(fred["ICSA"]),
             "lei_trend": load_lei_score()},
            {"oas_level": oas["level_score"], "oas_momentum": oas["momentum_score"],
             "sahm": sahm["score"], "financial_conditions": nfci_score(fred["NFCI"])},
            dispersion=disp,
        ),
        # The WHOLE window FRED serves (three years for ICE BofA series), on
        # dates all three share. It was the last 160 points, while the tab's
        # sample chart drew three years — the live series was never the one
        # on screen. Dispersion is derived from these same points.
        "series": _aligned_series({"hy": hy, "ccc": ccc, "bb": bb}),
        "curve": {"t10y3m": _last_obs(fred["T10Y3M"]), "t10y2y": _last_obs(fred["T10Y2Y"])},
        "as_of": {k: _last_obs(fred[v])["date"] for k, v in rc.FRED.items() if v in fred},
    }


def _last_obs(s: pd.Series) -> dict:
    s = s.dropna()
    return {"value": float(s.iloc[-1]), "date": s.index[-1].strftime("%Y-%m-%d")} if len(s) \
        else {"value": None, "date": None}


def _aligned_series(cols: dict, digits: int = 2) -> dict:
    df = pd.DataFrame(cols).dropna()
    out = {"t": [d.strftime("%Y-%m-%d") for d in df.index]}
    for k in cols:
        out[k] = [round(float(v), digits) for v in df[k]]
    return out


def run_details() -> dict:
    """
    Per-ticker detail pages: indicator panel and entry ladder for every symbol
    currently surfaced by any engine. Runs last so it can read the other
    engines' output and only compute details for names that actually rank.
    """
    from engines import indicators as ind

    symbols = set()
    for name in ("darkpool", "dividend", "recovery", "value", "politicians"):
        f = DATA / f"{name}.json"
        if f.exists():
            for r in json.loads(f.read_text()).get("rows", [])[:25]:
                sym = r.get("symbol") or r.get("ticker")
                if sym:
                    symbols.add(sym)

    from engines import fundamentals_builder as fbuild

    ohlcv = load_ohlcv(sorted(symbols))
    fv = load_fair_values(sorted(symbols))

    out, news_state = {}, {"reachable": 0, "refused": 0, "reason": None}
    for sym in sorted(symbols):
        df = ohlcv.get(sym)
        if df is None or len(df) < 60:
            continue
        spot = float(df["close"].iloc[-1])
        meta = fv.get(sym, {})

        # NO DEFAULT FAIR VALUE. The previous shape passed `spot * 1.3` when a
        # name had none, which is a fabricated 30% upside presented as a
        # valuation — on the page that tells someone where to buy. A name
        # without a fair value gets no ladder and says why.
        fair = meta.get("fair_value")
        ladder = None
        if fair:
            ladder = ind.entry_ladder(
                spot, fair,
                support_levels=fbuild.support_levels(df["close"]),
                solvency_ok=meta.get("solvency_ok"),
                insider_buying=meta.get("insider_buying"),
                estimate_revision_3m=meta.get("estimate_revision_3m"),
            )

        # A refusal is not "no news". load_news raises so the two cannot be
        # confused, and the page shows which of them it is.
        try:
            news, news_err = load_news(sym), None
            news_state["reachable"] += 1
        except free.NewsUnavailable as e:
            news, news_err = [], str(e)
            news_state["refused"] += 1
            news_state["reason"] = news_state["reason"] or str(e)

        out[sym] = {
            "spot": spot,
            "as_of": df.index[-1].date().isoformat(),
            "panel": ind.stock_panel(df),
            "fair_value": fair,
            "fair_value_basis": meta.get("fair_value_basis"),
            "solvency_basis": meta.get("solvency_basis"),
            "ladder": ladder,
            "news": news,
            "news_unavailable": news_err,
        }
    return {"engine": "details", "rows": [], "details": out,
            "news_source": news_state,
            "unavailable": {
                "insider_buying": "SEC Form 4 is free but needs its own "
                                  "parser; not wired",
                "estimate_revision_3m": "analyst estimates have no free source",
            }}


def run_bitcoin() -> dict:
    from engines import btc_cycle as bc

    weekly = load_btc_weekly()
    ath, ath_date = weekly.max(), weekly.idxmax().date()
    price = float(weekly.iloc[-1])

    df = pd.DataFrame({"close": weekly})
    df["rsi"] = bc.rsi(df["close"])
    pushes = bc.find_pushes(df)

    sth = load_sth_mvrv()
    daily = load_btc_daily()
    sma20 = weekly.rolling(20).mean()
    ema21 = weekly.ewm(span=21, adjust=False).mean()
    d365 = daily.iloc[-365:]
    d_rsi = bc.rsi(daily).iloc[-365:]
    return {
        "price": {"close": price, "ath": float(ath), "ath_date": ath_date.isoformat(),
                  "as_of": weekly.index[-1].strftime("%Y-%m-%d"),
                  "daily_close": float(daily.iloc[-1]),
                  "daily_as_of": daily.index[-1].strftime("%Y-%m-%d")},
        # Coin Metrics community tier has no short-term-holder cost basis, so
        # this is AGGREGATE MVRV. The tab labels it as such; it is not STH-MVRV.
        "mvrv": {"value": float(sth), "kind": "aggregate MVRV (Coin Metrics)",
                 "not": "STH-MVRV"},
        "engine": "bitcoin",
        "cycle": bc.cycle_position(date.today(), float(ath), ath_date, price),
        "divergence": bc.detect_divergence(pushes),
        "dip_buy": bc.dip_buy_signal(weekly, float(ath), sth, float(df["rsi"].iloc[-1])),
        # Daily is the timing chart; the weekly gate is what keeps it honest.
        "entry": bc.mtf_entry_signal(weekly, daily, float(ath), sth),
        "series": {
            "t": [d.strftime("%Y-%m-%d") for d in df.index],
            "p": [int(v) for v in df["close"]],
            "rsi": [round(float(v), 1) for v in df["rsi"]],
            "sma20": [round(float(v)) if pd.notna(v) else None for v in sma20],
            "ema21": [round(float(v)) if pd.notna(v) else None for v in ema21],
        },
        # Daily is the timing chart. It was drawn from a constant embedded in
        # index.html and never emitted here at all.
        "daily": {"t": [d.strftime("%Y-%m-%d") for d in d365.index],
                  "p": [round(float(v)) for v in d365],
                  "rsi": [round(float(v), 1) if pd.notna(v) else None for v in d_rsi]},
    }


# ------------------------------------------------------------ data loaders
#
# Default to the free stack. Every loader below resolves to engines/free_sources
# unless you set a paid provider key, so `python run_all.py` works at $0 with no
# accounts beyond a Slack webhook.
#
# The only genuine downgrade is STH-MVRV: short-term holder cost basis is
# Glassnode proprietary, so the free path substitutes AGGREGATE MVRV from Coin
# Metrics. Slower, less sensitive mid-cycle, excellent at extremes. The Bitcoin
# tab flags which one it is running on rather than implying parity.

from engines import free_sources as free

load_fred = free.load_fred
load_ohlcv = free.load_ohlcv
load_tape = free.load_tape
load_ptrs = free.load_ptrs
load_news = free.load_news
load_btc_daily = free.load_btc_daily
load_btc_weekly = free.load_btc_weekly
load_sth_mvrv = free.load_sth_mvrv
load_uninversion_date = free.load_uninversion_date
claims_trend_score = free.claims_trend_score
nfci_score = free.nfci_score
load_lei_score = free.load_lei_score


def load_fundamentals():
    """
    EDGAR statements joined to a price series, via fundamentals_builder.

    Coverage is ~32% of fields from EDGAR alone and ~78% once prices are
    joined. Every quality gate is satisfiable from EDGAR; the price series is
    what enables the "vs its own 5-10y history" measures all three screens
    rank on.
    """
    from engines import fundamentals_builder as fbuild

    if not (DATA / "universe.json").exists():
        raise NotImplementedError(
            "Create data/universe.json with a ticker list. EDGAR has no "
            "screener, so the universe has to be supplied.")
    doc = json.loads((DATA / "universe.json").read_text())
    # The file is a metadata document, not a bare list. Passing it straight
    # through iterated its KEYS, so the builder tried to resolve
    # "generated_at", "ranking", "source" and "floors" as tickers and all
    # three screeners reported "0 of 0".
    tickers = doc.get("tickers") if isinstance(doc, dict) else doc
    if not tickers:
        raise NotImplementedError(
            "data/universe.json has no 'tickers' list — rebuild it with "
            "scripts/build_universe.py")
    if not all(isinstance(t, str) for t in tickers):
        raise NotImplementedError(
            f"data/universe.json 'tickers' is not a list of strings: "
            f"{type(tickers[0]).__name__}")
    return _build_universe(tuple(tickers))


# Every weekly screener reads the same universe, and each call rebuilt it from
# EDGAR and yfinance — a measured 34.6 minutes per build locally, repeated once
# per screener inside a 90-minute job. The lru_cache note in CLAUDE.md is about
# a cache whose keys are all distinct; this one has one key read four times.
# Keyed on the ticker list, not on nothing, so a changed universe is rebuilt.
# A tuple so no screener can mutate another's input.
@functools.lru_cache(maxsize=1)
def _build_universe(tickers: tuple[str, ...]) -> tuple:
    from engines import fundamentals_builder as fbuild
    return tuple(fbuild.load_fundamentals(list(tickers), with_prices=True))


def load_tape_wide():
    """Wide price frame for politician alpha scoring, plus SPY as benchmark."""
    syms = sorted({t.ticker for t in load_ptrs()} | {"SPY"})
    data = free.load_ohlcv(syms)
    return pd.DataFrame({k: v["close"] for k, v in data.items()})


def load_fair_values(symbols: list[str]) -> dict:
    """
    Per symbol: fair_value, support levels, and the tier-3 confirmation inputs.

    Fair value is the FORWARD half of the reverse DCF the value screen already
    runs backwards, so the ladder and the screen cannot disagree about what a
    company is worth. Nothing new is fetched: the universe build has already
    produced enterprise value, free cash flow, the sector cost of capital and
    delivered growth for every name.

    What this returns honestly is as much the Nones as the numbers. Fair value
    is absent for a loss-maker rather than modelled anyway; solvency is None
    when its inputs are voided rather than True; and insider buying and
    estimate revisions are None ALWAYS, because neither has a free source —
    they are not False, and the difference decides whether tier 3 reads
    "confirmed" or "cannot be confirmed".
    """
    from engines import fundamentals_builder as fbuild

    want = {s.upper() for s in symbols}
    recs = {r.symbol.upper(): r for r in load_fundamentals()
            if r.symbol.upper() in want}

    out = {}
    for sym in sorted(want):
        f = recs.get(sym)
        if f is None:
            out[sym] = {"fair_value": None,
                        "fair_value_basis": "no fundamentals record was built "
                                            "for this symbol in the current run",
                        "support": [], "solvency_ok": None,
                        "insider_buying": None, "estimate_revision_3m": None}
            continue
        fv = fbuild.fair_value_per_share(f)
        # Solvency is a real reading where its inputs survived, and None where
        # they did not — never True by default. Altman does not apply to banks,
        # insurers, REITs or utilities, so leverage and coverage carry it there.
        if f.altman_not_applicable or f.debt_unavailable or f.ebit_unavailable:
            solvency = None
            if not f.debt_unavailable and f.net_debt_ebitda is not None:
                solvency = f.net_debt_ebitda < 4.0
        else:
            solvency = bool(f.altman_z is not None and f.altman_z >= 1.8)
        out[sym] = {
            "fair_value": fv["value"],
            "fair_value_basis": fv["basis"],
            "growth_used": fv["growth_used"],
            "support": [],                    # filled from prices by the caller
            "solvency_ok": solvency,
            "solvency_basis": ("Altman Z" if not f.altman_not_applicable
                               else "net debt/EBITDA (Altman does not apply "
                                    "to this sector)"),
            # No free source for either. None, not False — see the docstring.
            "insider_buying": None,
            "estimate_revision_3m": None,
        }
    return out


RUNNERS = {
    "darkpool":    run_darkpool,
    "dividend":    lambda: run_screener("dividend"),
    "recovery":    lambda: run_screener("recovery"),
    "value":       lambda: run_screener("value"),
    "financial":   lambda: run_screener("financial"),
    "reit":        lambda: run_screener("reit"),
    "politicians": run_politicians,
    "bitcoin":     run_bitcoin,
    "recession":   run_recession,
    "details":     run_details,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=ENGINES, help="run a single engine")
    ap.add_argument("--force", action="store_true", help="ignore the weekly cadence")
    ap.add_argument("--notify", action="store_true", help="send Slack alerts")
    ap.add_argument("--digest", action="store_true", help="force the biweekly digest")
    ap.add_argument("--dry-run", action="store_true", help="print Slack payloads instead of sending")
    args = ap.parse_args()

    targets = [args.only] if args.only else ENGINES
    status, failures = {}, 0
    prior = read_prior_status()

    for name in targets:
        if not should_run(name, args.force or bool(args.only)):
            print(f"[skip] {name} — {CADENCE[name]} cadence, not due today")
            # Not running is not a new result. Recording "skipped" here
            # overwrote Sunday's "ok" on every weekday run, so the dashboard
            # lost the weekly screeners six days in seven. Keep the last
            # real outcome; only a name that has never run reads as skipped.
            status[name] = prior.get(name) or {"state": "skipped",
                                                "cadence": CADENCE[name]}
            continue

        print(f"[run ] {name}")
        try:
            write(name, RUNNERS[name]())
            status[name] = {"state": "ok", "at": datetime.now(UTC).isoformat()}
        except NotImplementedError as e:
            print(f"[stub] {name} — {e}")
            status[name] = {"state": "not_wired", "detail": str(e),
                            "at": datetime.now(UTC).isoformat()}
        except Exception:
            failures += 1
            traceback.print_exc()
            status[name] = {"state": "error", "detail": traceback.format_exc(limit=2),
                            "at": datetime.now(UTC).isoformat()}

    # --- Slack -----------------------------------------------------------
    # BTC alerts fire only on state changes; most days send nothing, which is
    # the intended behaviour rather than a failure.
    if args.notify:
        import alerts

        if (DATA / "bitcoin.json").exists() and status.get("bitcoin", {}).get("state") == "ok":
            cur = json.loads((DATA / "bitcoin.json").read_text())
            prev_path = DATA / "bitcoin.prev.json"
            prev = json.loads(prev_path.read_text()) if prev_path.exists() else None
            print("\n[slack] BTC check")
            alerts.send_btc(cur, prev, dry_run=args.dry_run)
            prev_path.write_text(json.dumps(cur, default=str))

        # Recession alerts use the same transition-only path.
        if (DATA / "recession.json").exists() and status.get("recession", {}).get("state") == "ok":
            cur = json.loads((DATA / "recession.json").read_text())
            prev_path = DATA / "recession.prev.json"
            prev = json.loads(prev_path.read_text()) if prev_path.exists() else None
            print("[slack] recession check")
            alerts.send_generic(alerts.recession_alerts(cur, prev), dry_run=args.dry_run)
            prev_path.write_text(json.dumps(cur, default=str))

        if args.digest or alerts.digest_due():
            payloads = {}
            for k in ("darkpool", "dividend", "recovery", "value", "politicians"):
                f = DATA / f"{k}.json"
                if f.exists():
                    payloads[k] = json.loads(f.read_text())
            if payloads:
                print("[slack] biweekly digest")
                alerts.send_digest(payloads, dry_run=args.dry_run)
            else:
                print("[slack] digest due but no engine data yet")

    # MERGE, never replace. `--only recession` rewrote status.json with one
    # engine in it, every other tab lost its status, and the dashboard fell
    # back to invented sample rows on the live site. Engines this run did not
    # touch keep their last recorded outcome.
    merged = {**prior, **status}
    (DATA / "status.json").write_text(dump_json({
        "updated_at": datetime.now(UTC).isoformat(),
        "engines": {k: merged[k] for k in ENGINES if k in merged},
    }))

    # Stubs are an expected state during buildout and must not fail the job.
    # Genuine exceptions should, so a broken feed is noisy rather than silent.
    print(f"\ndone — {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
