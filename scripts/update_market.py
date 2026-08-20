#!/usr/bin/env python3
"""SmartTrades.ai free-beta updater.

Uses SEC EDGAR Company Facts only. No paid market-data API is required.
The free beta intentionally ranks business fundamentals rather than live price/valuation.

Data: https://data.sec.gov/api/xbrl/companyfacts/
Ticker map: https://www.sec.gov/files/company_tickers.json
"""
from __future__ import annotations
import csv, json, math, os, time, bisect
from pathlib import Path
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "config/free_universe.csv"
OUT = ROOT / "data/public.json"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_DELAY = float(os.getenv("SEC_DELAY", "0.18"))  # ~5.5 req/sec max
MIN_COVERAGE = int(os.getenv("MIN_COVERAGE", "25"))
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "SmartTrades.ai research admin@smarttrades.ai")

S = requests.Session()
S.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})


def get_json(url, attempts=4):
    last = None
    for n in range(attempts):
        try:
            r = S.get(url, timeout=45)
            if r.status_code == 429:
                time.sleep(2.0 * (n + 1))
                last = RuntimeError(f"SEC rate limit HTTP 429: {url}")
                continue
            if r.status_code >= 500:
                time.sleep(1.5 * (n + 1))
                last = RuntimeError(f"SEC server HTTP {r.status_code}: {url}")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if n < attempts - 1:
                time.sleep(1.5 * (n + 1))
    raise last


def load_universe():
    with UNIVERSE.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ticker_cik_map():
    raw = get_json(SEC_TICKERS)
    out = {}
    for v in raw.values():
        ticker = str(v.get("ticker", "")).upper()
        if ticker:
            out[ticker] = int(v["cik_str"])
    return out


def _annual_entries(facts, tag, unit="USD"):
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag, {})
    units = node.get("units", {})
    entries = units.get(unit, [])
    rows = {}
    for x in entries:
        form = x.get("form")
        if form not in ("10-K", "10-K/A"):
            continue
        start, end = x.get("start"), x.get("end")
        if not start or not end:
            continue
        try:
            days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
        except Exception:
            continue
        if not 300 <= days <= 450:
            continue
        # Prefer the most recently filed fact for each fiscal period (captures restatements).
        filed = x.get("filed", "")
        prev = rows.get(end)
        if prev is None or filed >= prev.get("filed", ""):
            rows[end] = x
    return [(end, float(x["val"])) for end, x in sorted(rows.items(), reverse=True)]


def annual_series(facts, tags, unit="USD"):
    best = []
    for tag in tags:
        arr = _annual_entries(facts, tag, unit)
        if len(arr) > len(best):
            best = arr
    return best


def annual_per_share_series(facts, tags):
    best = []
    for unit in ("USD/shares", "USD / shares"):
        arr = annual_series(facts, tags, unit)
        if len(arr) > len(best):
            best = arr
    return best


def _instant_entries(facts, tag, unit="USD"):
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag, {})
    rows = {}
    for x in node.get("units", {}).get(unit, []):
        if x.get("form") not in ("10-K", "10-K/A") or not x.get("end"):
            continue
        end = x["end"]
        filed = x.get("filed", "")
        prev = rows.get(end)
        if prev is None or filed >= prev.get("filed", ""):
            rows[end] = x
    return [(end, float(x["val"])) for end, x in sorted(rows.items(), reverse=True)]


def instant_value(facts, tags, unit="USD"):
    best = []
    for tag in tags:
        arr = _instant_entries(facts, tag, unit)
        if len(arr) > len(best):
            best = arr
    return best[0][1] if best else None


def series_values(series, n=4):
    return [v for _, v in series[:n]]


def cagr(new, old, years):
    try:
        new, old = float(new), float(old)
        if new <= 0 or old <= 0 or years <= 0:
            return None
        return (new / old) ** (1 / years) - 1
    except Exception:
        return None


def safe_div(a, b):
    try:
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)
    except Exception:
        return None


def aligned_latest(series_map):
    """Return latest values from independent annual series; simple and robust for beta ranking."""
    out = {}
    for k, s in series_map.items():
        out[k] = s[0][1] if s else None
    return out


def fetch_company(u, cik):
    facts = get_json(SEC_FACTS.format(cik=cik))
    revenue_s = annual_series(facts, [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"
    ])
    ni_s = annual_series(facts, ["NetIncomeLoss", "ProfitLoss"])
    op_s = annual_series(facts, ["OperatingIncomeLoss"])
    cfo_s = annual_series(facts, ["NetCashProvidedByUsedInOperatingActivities"])
    capex_s = annual_series(facts, [
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForAdditionsToPropertyPlantAndEquipment"
    ])
    divpaid_s = annual_series(facts, [
        "PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfDividendsAndDividendEquivalentsOnCommonStock"
    ])
    dps_s = annual_per_share_series(facts, [
        "CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"
    ])
    interest_s = annual_series(facts, ["InterestExpenseNonOperating", "InterestExpense"])

    rv = series_values(revenue_s, 4)
    nv = series_values(ni_s, 4)
    cv = series_values(cfo_s, 4)
    xv = series_values(capex_s, 4)
    dv = series_values(divpaid_s, 4)
    dpsv = series_values(dps_s, 4)

    if len(rv) < 3 or len(cv) < 2:
        return None

    # Calculate FCF series using matching rank positions. Fiscal period ends normally line up.
    fcfv = []
    for i in range(min(len(cv), len(xv))):
        fcfv.append(cv[i] - abs(xv[i]))

    latest = aligned_latest({
        "revenue": revenue_s, "net_income": ni_s, "operating_income": op_s,
        "cfo": cfo_s, "capex": capex_s, "div_paid": divpaid_s, "interest": interest_s,
    })
    latest_fcf = fcfv[0] if fcfv else None

    equity = instant_value(facts, [
        "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    ])
    cash = instant_value(facts, [
        "CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    ])
    debt_current = instant_value(facts, [
        "LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"
    ]) or 0
    debt_noncurrent = instant_value(facts, [
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent", "LongTermDebtNoncurrent"
    ])
    debt_total_fallback = instant_value(facts, [
        "LongTermDebtAndFinanceLeaseObligations", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"
    ])
    debt = (debt_current + debt_noncurrent) if debt_noncurrent is not None else debt_total_fallback

    rev_g = cagr(rv[0], rv[3], 3) if len(rv) >= 4 else (cagr(rv[0], rv[2], 2) if len(rv) >= 3 else None)
    ni_g = cagr(nv[0], nv[3], 3) if len(nv) >= 4 else None
    fcf_g = cagr(fcfv[0], fcfv[3], 3) if len(fcfv) >= 4 else None
    div_g = None
    if len(dpsv) >= 4:
        div_g = cagr(dpsv[0], dpsv[3], 3)
    elif len(dv) >= 4:
        div_g = cagr(abs(dv[0]), abs(dv[3]), 3)

    roe = safe_div(latest["net_income"], equity)
    op_margin = safe_div(latest["operating_income"], latest["revenue"])
    fcf_margin = safe_div(latest_fcf, latest["revenue"])
    fcf_conversion = safe_div(latest_fcf, latest["net_income"]) if (latest["net_income"] or 0) > 0 else None
    net_debt = (float(debt or 0) - float(cash or 0)) if debt is not None else None
    net_debt_fcf = safe_div(net_debt, latest_fcf) if (latest_fcf or 0) > 0 else None
    interest_coverage = safe_div(latest["operating_income"], abs(latest["interest"])) if latest["interest"] not in (None, 0) else 50.0
    payout = safe_div(abs(latest["div_paid"]), latest_fcf) if latest["div_paid"] is not None and (latest_fcf or 0) > 0 else None

    return {
        "ticker": u["ticker"], "name": u["name"], "sector": u["sector"],
        "revenue_growth_3y": rev_g, "earnings_growth_3y": ni_g, "fcf_growth_3y": fcf_g,
        "roe": roe, "operating_margin": op_margin, "fcf_margin": fcf_margin,
        "fcf_conversion": fcf_conversion, "net_debt_fcf": net_debt_fcf,
        "interest_coverage": interest_coverage, "fcf_payout": payout,
        "dividend_growth_3y": div_g, "pays_dividend": bool((latest["div_paid"] or 0) != 0),
        "latest_fiscal_end": revenue_s[0][0] if revenue_s else None,
        "source": "SEC EDGAR Company Facts",
    }


def finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def pct_rank(vals, higher=True):
    good = sorted(v for v in vals if finite(v))
    n = len(good)
    out = []
    for v in vals:
        if not finite(v):
            out.append(None)
            continue
        lo, hi = bisect.bisect_left(good, v), bisect.bisect_right(good, v)
        pos = (lo + hi - 1) / 2
        q = 100 * (pos / (n - 1) if n > 1 else .5)
        out.append(q if higher else 100 - q)
    return out


def mean_neutral(*xs):
    vals = [50 if x is None else x for x in xs]
    return sum(vals) / len(vals) if vals else 50


def weighted(parts):
    return sum((50 if v is None else v) * w for v, w in parts) / sum(w for _, w in parts)


def score(stocks):
    specs = {
        "roe": True, "operating_margin": True, "fcf_margin": True, "fcf_conversion": True,
        "revenue_growth_3y": True, "earnings_growth_3y": True, "fcf_growth_3y": True,
        "net_debt_fcf": False, "interest_coverage": True,
        "fcf_payout": False, "dividend_growth_3y": True,
    }
    ranks = {k: pct_rank([x.get(k) for x in stocks], hi) for k, hi in specs.items()}
    for i, x in enumerate(stocks):
        q = mean_neutral(ranks["roe"][i], ranks["operating_margin"][i], ranks["fcf_margin"][i], ranks["fcf_conversion"][i])
        g = mean_neutral(ranks["revenue_growth_3y"][i], ranks["earnings_growth_3y"][i], ranks["fcf_growth_3y"][i])
        fs = mean_neutral(ranks["net_debt_fcf"][i], ranks["interest_coverage"][i])
        smart = weighted([(q, .40), (g, .30), (fs, .30)])
        divsafe = mean_neutral(ranks["fcf_payout"][i], ranks["net_debt_fcf"][i], ranks["interest_coverage"][i])
        x.update({
            "quality_score": q, "growth_score": g, "financial_strength_score": fs, "smart_score": smart,
            "scores": {
                "quality-growth": weighted([(q, .40), (g, .35), (fs, .25)]),
                "dividend-growth": weighted([(divsafe, .30), (ranks["dividend_growth_3y"][i], .25), (q, .25), (fs, .20)]),
                "compounders": weighted([(q, .40), (ranks["fcf_growth_3y"][i], .30), (g, .20), (fs, .10)]),
                "cash-machines": weighted([(ranks["fcf_margin"][i], .35), (ranks["fcf_conversion"][i], .30), (fs, .25), (g, .10)]),
                "low-debt-growth": weighted([(g, .40), (fs, .35), (q, .25)]),
            }
        })
        tags = []
        if x.get("pays_dividend"):
            tags.append("dividend")
        if g >= 65:
            tags.append("growth")
        if q >= 70:
            tags.append("quality")
        if fs >= 70:
            tags.append("balance-sheet")
        x["tags"] = tags
    return stocks


def eligible(x, slug):
    if slug == "dividend-growth":
        return x.get("pays_dividend") and x.get("dividend_growth_3y") is not None
    if slug == "cash-machines":
        return x.get("fcf_margin") is not None and x.get("fcf_conversion") is not None
    return True


def round_or_none(v, digits=4):
    return round(v, digits) if finite(v) else None


def main():
    universe = load_universe()
    cikmap = ticker_cik_map()
    stocks, failures = [], []
    for n, u in enumerate(universe, 1):
        ticker = u["ticker"].upper()
        cik = cikmap.get(ticker)
        if cik is None:
            failures.append(f"{ticker}: CIK not found")
            print(f"[{n}/{len(universe)}] {ticker}: SKIP CIK not found")
            continue
        try:
            x = fetch_company(u, cik)
            if x:
                stocks.append(x)
                print(f"[{n}/{len(universe)}] {ticker}: ok")
            else:
                failures.append(f"{ticker}: insufficient comparable annual facts")
                print(f"[{n}/{len(universe)}] {ticker}: SKIP insufficient facts")
        except Exception as e:
            failures.append(f"{ticker}: {e}")
            print(f"[{n}/{len(universe)}] {ticker}: FAIL {e}")
        time.sleep(SEC_DELAY)

    if len(stocks) < MIN_COVERAGE:
        raise RuntimeError(f"Only {len(stocks)} companies produced usable SEC fundamentals; refusing to publish. Failures: {failures[:10]}")

    stocks = score(stocks)
    slugs = ["dividend-growth", "quality-growth", "compounders", "cash-machines", "low-debt-growth"]
    rankings = {}
    for slug in slugs:
        arr = sorted([x for x in stocks if eligible(x, slug)], key=lambda x: x["scores"][slug], reverse=True)[:10]
        rankings[slug] = [{
            "ticker": x["ticker"], "name": x["name"], "sector": x["sector"],
            "score": round(x["scores"][slug], 1), "smart_score": round(x["smart_score"], 1),
            "quality_score": round(x["quality_score"], 1), "growth_score": round(x["growth_score"], 1),
            "financial_strength_score": round(x["financial_strength_score"], 1),
        } for x in arr]

    public_universe = []
    for x in sorted(stocks, key=lambda z: z["smart_score"], reverse=True):
        public_universe.append({
            "ticker": x["ticker"], "name": x["name"], "sector": x["sector"],
            "smart_score": round(x["smart_score"], 1), "quality_score": round(x["quality_score"], 1),
            "growth_score": round(x["growth_score"], 1), "financial_strength_score": round(x["financial_strength_score"], 1),
            "revenue_growth_3y": round_or_none(x.get("revenue_growth_3y")),
            "earnings_growth_3y": round_or_none(x.get("earnings_growth_3y")),
            "fcf_growth_3y": round_or_none(x.get("fcf_growth_3y")),
            "fcf_margin": round_or_none(x.get("fcf_margin")),
            "dividend_growth_3y": round_or_none(x.get("dividend_growth_3y")),
            "tags": x["tags"], "latest_fiscal_end": x.get("latest_fiscal_end"),
        })

    stamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "is_demo": False,
        "updated_at": stamp,
        "methodology_version": "0.2-sec-free-beta",
        "coverage_count": len(stocks),
        "coverage_label": f"{len(stocks)} large U.S. operating companies with usable SEC filing data",
        "data_note": "Free beta uses SEC filing fundamentals only. It does not use live prices, valuation multiples, analyst estimates or momentum.",
        "rankings": rankings,
        "public_universe": public_universe,
        "failures": failures,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with {len(stocks)} companies and {len(failures)} skips/failures")

if __name__ == "__main__":
    main()
