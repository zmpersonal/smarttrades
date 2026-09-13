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
import json
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
ENGINES = ["darkpool", "dividend", "recovery", "value", "politicians", "bitcoin",
           "recession", "details"]

# Not every engine is worth running every day. Fundamentals barely move
# day to day; FINRA and congressional filings do.
CADENCE = {
    "darkpool":    "daily",     # FINRA posts by 6pm ET each trading day
    "politicians": "daily",     # new PTRs land continuously
    "bitcoin":     "daily",
    "dividend":    "weekly",    # Sunday
    "recovery":    "weekly",
    "value":       "weekly",
    "recession":   "daily",     # OAS and curve are daily; the read can turn fast
    "details":     "daily",
}


def should_run(engine: str, force: bool) -> bool:
    if force:
        return True
    if CADENCE[engine] == "daily":
        return True
    return datetime.now(UTC).weekday() == 6  # Sunday


def write(name: str, payload: dict) -> None:
    payload["generated_at"] = datetime.now(UTC).isoformat()
    (DATA / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"  wrote data/{name}.json  ({len(payload.get('rows', []))} rows)")


# --------------------------------------------------------------- engines

def run_darkpool() -> dict:
    from engines import finra_darkpool as fd

    end = date.today()
    start = end - timedelta(days=150)
    finra = fd.fetch_finra_range(start, end)

    tape = load_tape(sorted(finra["symbol"].unique()), start, end)
    board = fd.run(finra, tape)
    return {"engine": "darkpool", "rows": board.to_dict("records")}


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
MIN_SCORE = {"value": 60, "dividend": 60, "recovery": 50}


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
    scorer = {"dividend": sc.score_dividend,
              "recovery": lambda f: sc.score_recovery(f, 40.0, 1.2, 1.6),
              "value": sc.score_quality_value}[which]

    scored, gated, near, data_gated = [], 0, 0, 0
    never_built = 0                     # set by the loader when it reports
    for f in universe:
        res = scorer(f)
        if res["gates_failed"]:
            gated += 1
            if sc.data_quality_gates(f):
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
    return {"engine": which, "rows": da.to_rows(which, scored[:40]),
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
        "series": {"t": [d.strftime("%Y-%m-%d") for d in hy.index[-160:]],
                   "hy": [float(v) for v in hy.iloc[-160:]],
                   "ccc": [float(v) for v in ccc.iloc[-160:]],
                   "bb": [float(v) for v in bb.iloc[-160:]]},
    }


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

    ohlcv = load_ohlcv(sorted(symbols))
    fv = load_fair_values(sorted(symbols))

    out = {}
    for sym in sorted(symbols):
        df = ohlcv.get(sym)
        if df is None or len(df) < 60:
            continue
        panel = ind.stock_panel(df)
        spot = float(df["close"].iloc[-1])
        meta = fv.get(sym, {})
        out[sym] = {
            "spot": spot,
            "panel": panel,
            "ladder": ind.entry_ladder(
                spot, meta.get("fair_value", spot * 1.3),
                support_levels=meta.get("support", []),
                solvency_ok=meta.get("solvency_ok", True),
                insider_buying=meta.get("insider_buying", False),
                estimate_revision_3m=meta.get("estimate_revision_3m", 0.0),
            ),
            "news": load_news(sym),
        }
    return {"engine": "details", "rows": [], "details": out}


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
    return {
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
        },
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
    universe = json.loads((DATA / "universe.json").read_text())
    if not universe:
        raise NotImplementedError("data/universe.json is empty")
    return fbuild.load_fundamentals(universe, with_prices=True)


def load_tape_wide():
    """Wide price frame for politician alpha scoring, plus SPY as benchmark."""
    syms = sorted({t.ticker for t in load_ptrs()} | {"SPY"})
    data = free.load_ohlcv(syms)
    return pd.DataFrame({k: v["close"] for k, v in data.items()})


def load_fair_values(symbols: list[str]) -> dict:
    """
    Per symbol: fair_value, support levels, and the tier-3 confirmation inputs.
    Derive fair value from the same reverse-DCF the quality engine uses so the
    ladder and the screen cannot disagree with each other.
    """
    raise NotImplementedError(
        "Join screeners.py output to prices. The last piece of the free path.")


RUNNERS = {
    "darkpool":    run_darkpool,
    "dividend":    lambda: run_screener("dividend"),
    "recovery":    lambda: run_screener("recovery"),
    "value":       lambda: run_screener("value"),
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

    for name in targets:
        if not should_run(name, args.force or bool(args.only)):
            print(f"[skip] {name} — {CADENCE[name]} cadence, not due today")
            status[name] = {"state": "skipped", "cadence": CADENCE[name]}
            continue

        print(f"[run ] {name}")
        try:
            write(name, RUNNERS[name]())
            status[name] = {"state": "ok", "at": datetime.now(UTC).isoformat()}
        except NotImplementedError as e:
            print(f"[stub] {name} — {e}")
            status[name] = {"state": "not_wired", "detail": str(e)}
        except Exception:
            failures += 1
            traceback.print_exc()
            status[name] = {"state": "error", "detail": traceback.format_exc(limit=2)}

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

    (DATA / "status.json").write_text(json.dumps({
        "updated_at": datetime.now(UTC).isoformat(),
        "engines": status,
    }, indent=1))

    # Stubs are an expected state during buildout and must not fail the job.
    # Genuine exceptions should, so a broken feed is noisy rather than silent.
    print(f"\ndone — {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
