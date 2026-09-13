"""
SmartTrades.AI — the $0 data layer
==================================

Every loader in `run_all.py`, implemented against sources that cost nothing and
mostly need no key at all.

    FINRA      cdn.finra.org        off-exchange volume      no key
    FRED       fredgraph CSV        OAS, curve, Sahm, NFCI   no key for CSV
    SEC EDGAR  data.sec.gov         fundamentals, Form 4     no key, User-Agent required
    Stooq      stooq.com CSV        EOD OHLCV                no key
    Coin Metrics community          BTC market cap, MVRV     no key
    Binance    api.binance.com      BTC OHLCV candles        no key
    Stock Watcher                   parsed congressional PTRs no key
    Yahoo RSS                       headlines                no key

The one genuinely good surprise
-------------------------------
SEC EDGAR's companyfacts endpoint is free, needs no key, and every fact carries
the date it was `filed`. That means you can reconstruct what was actually known
on any past date by filtering on filing date rather than period end — which is
the point-in-time property that otherwise costs ~$150/month.

It is arguably better than the budget paid tier for backtesting, because those
providers store one mutable value per period: a restatement silently overwrites
history, so even a filing-date join can leak, and you cannot recover what the
market originally saw. EDGAR keeps every filed version.

The cost is engineering, not money. Companies switch XBRL tags over time, Q4 is
not reported and has to be derived as FY minus the first three quarters, and
fiscal calendars drift. Budget a few weekends.

What is genuinely NOT free
--------------------------
Short-term holder MVRV (the 155-day cohort cost basis) is Glassnode's
proprietary cut and has no free equivalent. `load_sth_mvrv` below substitutes
AGGREGATE MVRV from Coin Metrics, which is a different measure: all holders
rather than recent ones. It is directionally useful and historically reliable
at cycle extremes, but it is slower and less sensitive than STH-MVRV, so the
dip-buy rule's second leg gets duller. That is a real downgrade and is labelled
as such in the UI rather than papered over.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd
import requests

# SEC blocks requests without a descriptive User-Agent carrying a contact
# address. This is not optional and is the most common reason EDGAR 403s.
UA = os.environ.get("SEC_USER_AGENT", "SmartTrades research contact@example.com")
SEC_HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
POLYGON_GROUPED = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d}"
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CM_METRICS = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
HOUSE_SW = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
SENATE_SW = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"

_session = requests.Session()


# Rate-limit windows differ by two orders of magnitude, and one global backoff
# turned a 45-minute run into a 7-hour one. Polygon's free tier is 5 requests
# per MINUTE, so a 429 there needs a wait measured in minutes. SEC allows ~10
# per SECOND and throttles briefly; sleeping 62s there is pure dead time.
_RATE_LIMIT_WAIT = {
    "api.polygon.io": 62,
    "data.sec.gov": 2,
    "www.sec.gov": 2,
    "community-api.coinmetrics.io": 5,
}
_DEFAULT_RATE_WAIT = 10


def _get(url: str, headers: dict | None = None, tries: int = 4,
         rate_limit_wait: int | None = None, **kw):
    """
    Polite retry with a backoff that actually matches the limiter it is facing.

    The previous version slept 1s then 2s and gave up ~3 seconds into a window
    that lasts a minute. That is fine for the nightly single call and fails
    immediately on any backfill loop. Polygon's free tier is 5 calls/minute, so
    a 429 needs a wait measured in the same units as the window — hence
    `rate_limit_wait` defaulting just past 60s, and honouring Retry-After when
    the server sends one.
    """
    if rate_limit_wait is None:
        host = url.split("/")[2] if "//" in url else ""
        rate_limit_wait = _RATE_LIMIT_WAIT.get(host, _DEFAULT_RATE_WAIT)

    for i in range(tries):
        r = _session.get(url, headers=headers or {}, timeout=45, **kw)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", rate_limit_wait))
            if i < tries - 1:
                print(f"  [rate-limit] {wait}s (attempt {i + 1}/{tries})")
                time.sleep(wait)
                continue
        elif r.status_code in (502, 503, 504):
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
    raise RuntimeError(f"{url} failed after {tries} tries")


# --------------------------------------------------------------- FRED

def fred_series(series_id: str) -> pd.Series:
    """
    Free, no key. The CSV endpoint works unauthenticated; a free API key only
    raises rate limits.

    Reminder: ICE BofA series (BAMLH0A0HYM2 etc.) are capped at a rolling
    three-year window since April 2026. Append each pull to
    data/oas_history.csv so the history builds forward.
    """
    r = _get(FRED_CSV.format(sid=series_id))
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # FRED writes "." for gaps
    return df.dropna().set_index("date")["value"]


def load_fred(series_ids: list[str]) -> dict:
    return {sid: fred_series(sid) for sid in series_ids}


# -------------------------------------------------------------- prices

# Stooq moved behind a JavaScript proof-of-work bot challenge (serves a ~800
# byte SHA-256 puzzle page instead of CSV). Not bypassed — that is scraping
# around an explicit anti-bot control. Two replacements below.

def polygon_grouped_daily(day: date, api_key: str | None = None) -> pd.DataFrame:
    """
    OHLCV for EVERY US ticker on one date, in a single request.

    This is the right shape for this project: the dark pool engine needs
    consolidated volume for thousands of symbols, and one call per trading day
    beats one call per symbol by three orders of magnitude. Polygon's free tier
    allows 5 calls/minute with no card required, so a 2-year backfill is about
    100 minutes unattended and the nightly job is a single call.
    """
    key = api_key or os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY unset — free key at polygon.io, no card")
    r = _get(POLYGON_GROUPED.format(d=day.isoformat()), params={"apiKey": key,
                                                               "adjusted": "true"})
    j = r.json()
    if not j.get("results"):
        return pd.DataFrame()                       # holiday or weekend
    df = pd.DataFrame(j["results"]).rename(columns={
        "T": "symbol", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "t": "ts"})
    df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df[["Date", "symbol", "open", "high", "low", "close", "volume"]]


# Splits arrive free with history when actions are requested. Fetching them
# separately cost a second round-trip per ticker — 19.8% of a 41.8-minute run
# for data the first call could already have returned.
_SPLIT_CACHE: dict = {}


def yfinance_ohlcv(symbol: str) -> pd.DataFrame:
    """
    No-key fallback. Unofficial and it breaks periodically, but it needs no
    registration and covers full history. Fine for a personal dashboard;
    do not build a backtest on it without caching to your own store.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("pip install yfinance for the no-key price path") from e
    df = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)
    if df.empty:
        raise ValueError(f"yfinance returned nothing for {symbol}")
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Harvest splits from the same response rather than making a second call.
    if "stock splits" in df.columns:
        sp = df["stock splits"]
        sp = sp[sp > 0]
        _SPLIT_CACHE[symbol.upper()] = (sp if len(sp) else pd.Series(dtype=float))

    return df[["open", "high", "low", "close", "volume"]]


def equity_splits(symbol: str) -> pd.Series:
    """
    Actual split history. Required for split-adjusting XBRL dividends, which
    are reported as filed and never restated.

    Must be authoritative. Inferring splits from the dividend series itself was
    tried and mistook 3M's real 2024 halving for a 2:1 split — a 2:1 split and
    a 50% cut are identical in a DPS series.
    """
    # Served from the history call when equity_ohlcv already ran for this
    # symbol, which is the normal order in load_fundamentals.
    if symbol.upper() in _SPLIT_CACHE:
        return _SPLIT_CACHE[symbol.upper()]
    try:
        import yfinance as yf
        s = yf.Ticker(symbol).splits
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return pd.Series(dtype=float)


def equity_ohlcv(symbol: str) -> pd.DataFrame:
    """Polygon if a key is present, else yfinance."""
    if os.environ.get("POLYGON_API_KEY"):
        end = date.today()
        frames = []
        for i in range(5):                          # last few sessions only
            d = end - timedelta(days=i)
            g = polygon_grouped_daily(d)
            if not g.empty:
                row = g[g["symbol"] == symbol.upper()]
                if not row.empty:
                    frames.append(row.set_index("Date"))
        if frames:
            return pd.concat(frames)[["open", "high", "low", "close", "volume"]]
    return yfinance_ohlcv(symbol)


def _warn_collapsed(label: str, failures: dict, total: int, cap: int = 3) -> None:
    """
    One line per FAILURE MODE, not one per item.

    A per-item warning in a 1,500-symbol loop produced ~12,000 identical
    "pip install yfinance" lines and made the CI log unreadable — the one
    genuine signal in it, that the provider was missing, was buried in its own
    repetition. Group by message, show a few examples, and count the rest.
    """
    if not failures:
        return
    by_msg: dict = {}
    for item, msg in failures.items():
        by_msg.setdefault(str(msg)[:120], []).append(item)
    for msg, items in sorted(by_msg.items(), key=lambda kv: -len(kv[1])):
        shown = ", ".join(items[:cap])
        more = f" and {len(items) - cap} more" if len(items) > cap else ""
        print(f"  [warn] {label}: {len(items)}/{total} failed — {msg} "
              f"({shown}{more})")


def load_ohlcv(symbols: list[str]) -> dict:
    """
    Per-symbol failure warns and continues; TOTAL failure raises.

    Returning {} when every request failed is the exact silent-wrong-answer
    mode the project forbids — downstream cannot distinguish "no data" from
    "source is dead".
    """
    out, failed = {}, {}
    for sym in symbols:
        try:
            out[sym] = equity_ohlcv(sym)
        except Exception as e:
            failed[sym] = e
    _warn_collapsed("prices", failed, len(symbols))
    if symbols and not out:
        raise RuntimeError(
            f"every one of {len(symbols)} symbols failed — source is down, "
            "not a data gap")
    return out


def load_tape(symbols, start, end) -> pd.DataFrame:
    """
    Consolidated tape volume for the dark pool engine. FINRA gives the
    off-exchange numerator; this is the denominator.
    """
    frames, failed = [], {}
    for s in symbols:
        try:
            df = equity_ohlcv(s).loc[str(start):str(end)].reset_index()
        except Exception as e:
            failed[s] = e
            continue
        df.columns = ["Date" if c.lower() in ("date", "index") else c for c in df.columns]
        df["symbol"] = s
        frames.append(df)
    _warn_collapsed("tape", failed, len(symbols))
    if not frames:
        raise RuntimeError("no tape data for any symbol — source is down")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------- SEC point-in-time

@lru_cache(maxsize=1)
def ticker_cik_map() -> dict:
    r = _get(SEC_TICKERS, SEC_HEADERS)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in r.json().values()}


# Companies switch XBRL tags over time — Apple's revenue has lived under
# several concepts. Try each in order and take the first that has data.
TAG_CHAINS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold",
                        "CostOfGoodsSold", "CostOfServices"],
    "operating_income": ["OperatingIncomeLoss",
                         "OperatingIncomeLossIncludingNoncontrollingInterest"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # ASU 2016-18 (effective ~fiscal 2019) moved filers off
    # CashAndCashEquivalentsAtCarryingValue to the restricted-cash bundle. With
    # a single-tag chain there was nothing to stitch, so `cash` died in 2019
    # for most large filers — and net_debt inherits cash's index, which killed
    # the EV history for 84 of 190 names. The successor includes restricted
    # cash, so net debt shifts slightly in definition: a labelled
    # approximation, not an equivalence.
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
             "IncludingDisposalGroupAndDiscontinuedOperations"],

    # Total debt is a SUM of components, not a choice between alternatives.
    # The old chain held long-term only, so Oracle and Comcast read debt of
    # exactly zero against roughly $90bn each — which understates invested
    # capital and INFLATES ROIC. Oracle came out at 138.3%.
    "debt_noncurrent": ["LongTermDebtNoncurrent",
                        "LongTermDebtAndCapitalLeaseObligations"],
    "debt_current": ["LongTermDebtCurrent", "DebtCurrent",
                     "LongTermDebtAndCapitalLeaseObligationsCurrent",
                     "ShortTermBorrowings", "OtherShortTermBorrowings"],
    "finance_leases": ["FinanceLeaseLiabilityNoncurrent",
                       "CapitalLeaseObligationsNoncurrent"],
    # Some filers tag the combined figure directly; prefer it when present.
    # `LongTermDebt` is the TOTAL carrying amount in most filers' usage —
    # current maturities included. Leaving it in debt_noncurrent and then
    # adding debt_current double-counts: Oracle's invested capital read
    # $141.3bn against roughly $100bn actual. It belongs here.
    "debt_total": ["DebtLongtermAndShorttermCombinedAmount", "LongTermDebt"],
    "debt": ["LongTermDebtNoncurrent", "LongTermDebt"],   # legacy single-component
    # Cash-flow statement tags have "ContinuingOperations" variants that filers
    # switch to whenever they report a disposal, and switch back afterwards —
    # the same single-tag failure that killed `cash` at ASU 2016-18. A filer on
    # the variant returned an EMPTY ocf series, which sets fcf_unavailable and
    # hard-fails the data-quality gate for a company that reports the figure
    # perfectly well.
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    # Capex is tagged at least four ways depending on whether intangibles and
    # productive assets are bundled in.
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
              "PaymentsForCapitalImprovements",
              "PaymentsToAcquireOtherPropertyPlantAndEquipment"],
    "shares": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
    "dividends_per_share": ["CommonStockDividendsPerShareDeclared",
                            "CommonStockDividendsPerShareCashPaid"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "dep_amort": ["DepreciationDepletionAndAmortization",
                  "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    # Found by concept_freshness before it cost a universe run: stale for
    # 46.4% of names, median lag a full year. InterestExpenseNonoperating is
    # the fresher tag in 18 of 22 sampled.
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating",
                         "InterestExpenseDebt", "InterestExpenseOperating",
                         "InterestIncomeExpenseNonoperatingNet",
                         "InterestIncomeExpenseNet"],
    "tax_expense": ["IncomeTaxExpenseBenefit",
                    "IncomeTaxExpenseBenefitContinuingOperations"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                      "ExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                      "MinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    # --- financials -------------------------------------------------------
    # Coverage measured across a 60-name sample of the 176: equity, net income,
    # assets and revenue 87%; goodwill 78%; intangibles 77%; noninterest
    # expense 62%; interest income 47%; provisions 42%. The last two are
    # bank-only and are NEVER gated on — insurers and asset managers
    # legitimately do not report them, and gating on them would exclude two
    # thirds of the bucket by construction, which is the mistake the quality
    # screen makes with utilities.
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill",
                    "FiniteLivedIntangibleAssetsNet"],
    # Witness tags. A depository with no deposits and no interest income is not
    # a depository, whatever its SIC says — 6199 holds MicroStrategy, IREN,
    # Coinbase and Circle alongside genuine lenders.
    "deposits": ["Deposits", "InterestBearingDepositLiabilities",
                 "NoninterestBearingDepositLiabilities", "TimeDeposits"],
    "interest_income": ["InterestAndDividendIncomeOperating",
                        "InterestIncomeOperating",
                        "InterestAndFeeIncomeLoansAndLeases"],
    "premiums_earned": ["PremiumsEarnedNet", "PremiumsWrittenNet",
                        "PremiumsEarnedNetPropertyAndCasualty"],
    "noninterest_expense": ["NoninterestExpense", "OperatingExpenses"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
}


# SEC's ticker map points at the CURRENT registrant, which after a corporate
# restructuring can be a shell with no filing history. XOM is the live example:
# the map gives CIK 2115436 ("ExxonMobil Holdings Corp", zero annual facts)
# while the entire history sits under predecessor CIK 34088. Nothing in the map
# resolves predecessors, so they are pinned here.
PREDECESSOR_CIK = {
    "XOM": 34088,
}


def company_facts(ticker: str) -> dict:
    t = ticker.upper()
    cik = ticker_cik_map().get(t)
    if cik is None:
        raise KeyError(f"no CIK for {t}")
    time.sleep(0.12)                                # stay under SEC's ~10 req/s
    facts = _get(SEC_FACTS.format(cik=cik), SEC_HEADERS).json()

    # If the mapped registrant has no usable annual history, try a known
    # predecessor before giving up. Silently skipping the ticker — the old
    # behaviour — drops a mega-cap from the universe with no warning.
    if not _has_annual_facts(facts) and t in PREDECESSOR_CIK:
        print(f"  [info] {t}: mapped CIK {cik} has no annual facts, "
              f"using predecessor {PREDECESSOR_CIK[t]}")
        time.sleep(0.12)
        facts = _get(SEC_FACTS.format(cik=PREDECESSOR_CIK[t]), SEC_HEADERS).json()
    return facts


def _has_annual_facts(facts: dict) -> bool:
    tax = taxonomy_of(facts)
    us_gaap = facts.get("facts", {}).get(tax, {})
    chains = TAG_CHAINS if tax != "ifrs-full" else IFRS_CHAINS
    for chain in chains["revenue"]:
        node = us_gaap.get(chain)
        if not node:
            continue
        for items in node.get("units", {}).values():
            if any(i.get("fp") == "FY" for i in items):
                return True
    return False


# Fiscal-year ends drift by a few days for 52/53-week filers, so two tags can
# describe the same year on different dates.
_PERIOD_TOL_DAYS = 7


# Foreign private issuers (WIT, GGB and every other 20-F filer) report under
# the ifrs-full taxonomy with ZERO us-gaap concepts. Reading only us-gaap made
# them return no revenue and drop out before becoming a Fundamentals record at
# all — so they never appeared in any coverage or failure statistic. An
# invisible exclusion is worse than a counted one.
IFRS_CHAINS = {
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers"],
    "net_income": ["ProfitLoss", "ProfitLossAttributableToOwnersOfParent"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfSales"],
    "operating_income": ["ProfitLossFromOperatingActivities"],
    "pretax_income": ["ProfitLossBeforeTax"],
    "assets": ["Assets"],
    "equity": ["Equity", "EquityAttributableToOwnersOfParent"],
    "cash": ["CashAndCashEquivalents"],
    "debt": ["Borrowings", "NoncurrentPortionOfNoncurrentBorrowings"],
    "ocf": ["CashFlowsFromUsedInOperatingActivities"],
    "capex": ["PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],
    "dep_amort": ["DepreciationAndAmortisationExpense"],
    "interest_expense": ["FinanceCosts"],
    "tax_expense": ["IncomeTaxExpenseContinuingOperations"],
    "current_assets": ["CurrentAssets"],
    "current_liabilities": ["CurrentLiabilities"],
    "retained_earnings": ["RetainedEarnings"],
    "shares": ["NumberOfSharesOutstanding"],
    "eps_diluted": ["DilutedEarningsLossPerShare"],
    "dividends_per_share": ["DividendsPaidOrdinarySharePerShare"],
    "dividends_paid": ["DividendsPaidOrdinaryShares"],
}


def tags_used(facts: dict, field: str, as_of=None) -> list:
    """Which tags actually contributed to a stitched series."""
    df = extract_series(facts, field, as_of=as_of, annual_only=True)
    return list(df.attrs.get("tags_used", []))


# Two DIFFERENT measurements were sharing one threshold, and that created a
# dead band exactly one reporting cycle wide.
#
#   CALENDAR-relative (data_stale_days, _align, _ratio): a healthy annual fact
#   is legitimately up to ~440 days old — 365 between filings plus ~75 of
#   filing lag. 450-550 is correct there.
#
#   REFERENCE-relative (this function): two concepts from the SAME filing share
#   a period end, give or take 52/53-week drift. A lag of 365 days does not
#   mean "recent"; it means the concept was NOT TAGGED in the latest filing —
#   one whole reporting cycle skipped. At 450 that passed unflagged, and a
#   one-year-old balance sheet divided into current EBIT is precisely the
#   cross-period error this check exists to catch.
REFERENCE_LAG_DAYS = 200        # under one reporting cycle, over fiscal drift
CALENDAR_LAG_DAYS = 550         # one cycle plus filing lag, plus headroom


def concept_freshness(facts: dict, as_of=None, reference: str = "revenue",
                      fields: list[str] | None = None,
                      stale_days: int = REFERENCE_LAG_DAYS,
                      calendar_stale_days: int = CALENDAR_LAG_DAYS) -> dict:
    """
    How far each concept's newest annual period lags the reference concept's.

    This exists because the SAME bug has now been found four separate times, in
    four separate universe runs, on four different concepts:

        revenue   ASC 606 tag migration, 2018 — series froze at 2017
        cash      ASU 2016-18 migration, 2019 — killed EV history for 44%
        gross     ASC 606 presentation — Amazon's margin read its 2009 figure
        debt      truncated for 43.7% of names — inflated ROIC 2.4x

    Each time the shape was identical: a concept stops being tagged, the
    truncated series silently substitutes for the full quantity, and the output
    is wrong in a way that looks plausible. Fixing chains reactively finds them
    one universe run at a time.

    Comparing every concept against revenue turns an open-ended bug class into
    a measured one. A filer still reporting revenue through 2025 but whose debt
    stops in 2013 is describing a tag migration, not a company that repaid its
    debt. That is checkable at extraction time, for every concept at once.
    """
    fields = fields or list(TAG_CHAINS.keys())
    newest = {}
    for f in fields:
        df = extract_series(facts, f, as_of=as_of, annual_only=True)
        if not df.empty:
            newest[f] = pd.to_datetime(df["end"]).max()

    if reference not in newest:
        return {"reference": reference, "available": False, "lags": {}, "stale": []}

    ref = newest[reference]
    lags = {f: int((ref - d).days) for f, d in newest.items()}

    # Revenue is the reference, so its own lag is ZERO BY CONSTRUCTION. A filer
    # whose revenue is 500 days old reads lag 0 on every concept and sails
    # through, because nothing measures the yardstick. Check it against the
    # calendar instead.
    ref_lag = int((pd.Timestamp(as_of or date.today()) - ref).days)
    return {
        "reference_lag_days": ref_lag,
        # The yardstick is judged on the CALENDAR, not on the tighter
        # reference-relative bar, because being 400 days into a reporting
        # cycle is normal.
        "reference_stale": ref_lag > calendar_stale_days,
        "reference": reference,
        "reference_period": ref.date().isoformat(),
        "available": True,
        "lags": lags,
        "newest": {f: d.date().isoformat() for f, d in newest.items()},
        # Present in the filing but lagging the reference — the signature of a
        # tag migration rather than a genuine absence.
        "stale": sorted([f for f, lag in lags.items() if lag > stale_days]),
        "missing": sorted([f for f in fields if f not in newest]),
    }


SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SIC ranges -> the sector labels screeners.py already branches on. Those
# branches — the REIT/utility payout, FCF and debt caps in dividend_gates, and
# the Altman skip — have NEVER fired for any name in any run, because `sector`
# defaulted to "general" and nothing ever classified it.
_SIC_SECTORS = [
    ((6000, 6499), "financial"), ((6700, 6770), "financial"),
    ((6798, 6798), "reit"), ((6500, 6599), "reit"),
    ((4900, 4999), "utility"),
]


# The Altman exemption must be NARROWER than the sector label. SIC classifies
# crypto miners and SPAC-descended issuers under 6199 "finance services", and
# eleven of twenty-four `financial` names were exactly that — all of them lost
# the Altman gate. Core Scientific carries negative equity of -$0.96bn, which
# is precisely the case Altman exists for, and it was skipped.
#
# Exempt only the balance sheets the model genuinely does not describe:
# depository institutions, insurers, REITs and regulated utilities.
_ALTMAN_EXEMPT_SIC = [(6020, 6099), (6199, 6199), (6311, 6411),
                      (6500, 6599), (6798, 6798), (4900, 4999)]
# Raymond James read Z 0.77 and was NOT exempt while every commercial bank
# was: broker-dealers sit at 6200-6299 and depositories at 6020-6099. Altman
# describes a broker's balance sheet no better than a bank's.
_ALTMAN_EXEMPT_SIC = [(6020, 6099), (6200, 6299), (6311, 6411),
                      (6500, 6599), (6798, 6798), (4900, 4999)]


@lru_cache(maxsize=4096)
def company_submissions(ticker: str) -> dict:
    """
    One cached fetch of the submissions endpoint.

    `company_sector` and `company_sic` each fetched this separately — 3,000
    requests across a 1,500-name universe where 1,500 would do, on top of a
    global 62s backoff. Together those turned a 45-minute run into seven hours.
    """
    cik = ticker_cik_map().get(ticker.upper())
    if cik is None:
        return {}
    try:
        time.sleep(0.12)                            # stay under SEC's ~10 req/s
        return _get(SEC_SUBMISSIONS.format(cik=cik), SEC_HEADERS).json()
    except Exception:
        return {}


def company_sic(ticker: str) -> int:
    try:
        return int(company_submissions(ticker).get("sic") or 0)
    except (TypeError, ValueError):
        return 0


def altman_exempt(sic: int) -> bool:
    return any(lo <= sic <= hi for lo, hi in _ALTMAN_EXEMPT_SIC)


def company_sector(ticker: str) -> str:
    """Sector from the SEC's own SIC code. Shares one cached request."""
    sic = company_sic(ticker)
    if not sic:
        return "general"
    for (lo, hi), label in _SIC_SECTORS:
        if lo <= sic <= hi:
            return label
    return "general"


def is_foreign_private_issuer(facts: dict) -> bool:
    """
    Detected from the form type, which is the reliable marker — a 20-F or 40-F
    filer may still report under us-gaap, as HDFC Bank does.
    """
    if taxonomy_of(facts) == "ifrs-full":
        return True
    for node in facts.get("facts", {}).get("us-gaap", {}).values():
        for items in node.get("units", {}).values():
            for it in items[:40]:
                if str(it.get("form", "")).startswith(("20-F", "40-F")):
                    return True
    return False


def taxonomy_of(facts: dict) -> str:
    """Which taxonomy a filer actually uses. 20-F filers report ifrs-full."""
    f = facts.get("facts", {})
    if f.get("us-gaap"):
        return "us-gaap"
    if f.get("ifrs-full"):
        return "ifrs-full"
    return "unknown"


def _is_annual(item: dict, lo_days: int = 330, hi_days: int = 400) -> bool:
    """
    fp == "FY" is NOT enough, and this cost a silent 5x error.

    A 10-K carries the year's quarterly comparatives alongside the annual
    figures, and XBRL tags every one of them fp="FY", form="10-K". Filtering on
    fp alone returned Apple 90-day periods of $53bn interleaved with 363-day
    periods of $416bn, so revenue_cagr_5y compared a quarter to a year and read
    45.1% instead of 8.7%.

    Period LENGTH is the only reliable discriminator. Facts with no `start` are
    balance-sheet instants — a point-in-time snapshot with no duration to test
    — so they pass through unchanged.
    """
    if item.get("fp") != "FY":
        return False
    start = item.get("start")
    if not start:
        return True                                 # instant, not a duration
    try:
        days = (date.fromisoformat(item["end"]) - date.fromisoformat(start)).days
    except (ValueError, KeyError, TypeError):
        return True                                 # unparseable: keep, do not guess
    return lo_days <= days <= hi_days


def extract_series(facts: dict, field: str, as_of: date | None = None,
                   annual_only: bool = True) -> pd.DataFrame:
    """
    Pull one normalised field, honouring the tag fallback chain.

    `as_of` is the point-in-time filter and the whole reason to do this from
    EDGAR rather than a cheap API: it keeps only facts whose FILING date is on
    or before `as_of`, so the result is what was actually knowable then. A 10-K
    for FY ending 31 Dec might not be filed until late February; using it for a
    January decision is lookahead.
    """
    tax = taxonomy_of(facts)
    us_gaap = facts.get("facts", {}).get(tax, {})
    chains = TAG_CHAINS if tax != "ifrs-full" else IFRS_CHAINS

    # STITCH across the chain rather than picking one tag.
    #
    # Picking the "richest" tag by period count was wrong in a way that only
    # shows up on real filers: most large companies migrated revenue tags at
    # the ASC 606 adoption in 2018, so the deprecated tag carries a LONGER
    # history that stops dead at the migration. SalesRevenueNet gave Apple 11
    # periods ending 2017-09-30 and beat RevenueFromContract...'s 9 periods
    # ending 2025-09-27. Apple's revenue series silently ended eight years ago.
    #
    # Tags within a chain are by definition the same concept, so the correct
    # answer is neither tag — it is both, stitched. Preference goes to the tag
    # with the most recent coverage where periods overlap, and older tags fill
    # the gaps behind it.
    per_tag, units_seen, coverage_cost = [], {}, {}
    for tag in chains.get(field, [field]):
        node = us_gaap.get(tag)
        if not node:
            continue
        # PIN THE UNIT. A concept can be tagged in several currencies at once
        # and appending them all produces an incommensurable series.
        #
        # HDFC Bank tags dividends per share in INR/shares (35 facts, ending
        # Rs22.00) AND USD/shares (11 facts, ending $0.26). Blended, the rupee
        # figure got divided by a dollar ADR price and the dividend yield
        # published as 47.45% against a real 1.11%. Nebius blends RUB and USD
        # across revenue, net income, assets, equity, cash, every debt
        # component and operating cash flow.
        #
        # Invisible to all three integrity checks by construction: both units
        # are current, nothing is absent, nothing is zero-filled. The numbers
        # are simply not the same kind of thing.
        by_unit = {}
        for unit, items in node.get("units", {}).items():
            keep = []
            for it in items:
                if annual_only and not _is_annual(it):
                    continue
                filed = it.get("filed")
                if as_of and filed and date.fromisoformat(filed) > as_of:
                    continue                        # not yet public on as_of
                keep.append({"end": it.get("end"), "start": it.get("start"),
                             "filed": filed, "fy": it.get("fy"),
                             "form": it.get("form"), "val": it.get("val"),
                             "unit": unit, "tag": tag})
            if keep:
                by_unit[unit] = keep

        if len(by_unit) > 1:
            units_seen.update(by_unit)
        # Prefer the reporting currency of the security we price in, then
        # coverage. Never merge.
        rows_t = []
        if by_unit:
            # USD wins unconditionally: every ratio here divides by a USD price,
            # and mixing bases across concepts is worse than a short series.
            def _rank(u):
                return (u.startswith("USD"), len(by_unit[u]),
                        max(r["end"] or "" for r in by_unit[u]))
            chosen = max(by_unit, key=_rank)
            rows_t = by_unit[chosen]
            # But record the cost. HDFC Bank has 35 INR facts and 11 USD, so
            # preferring USD truncates its dividend history from 35 years to 11
            # — which silently caps the increase streak. The choice is right;
            # making it invisible is not.
            richest = max(by_unit, key=lambda u: len(by_unit[u]))
            if richest != chosen:
                coverage_cost[field] = {
                    "chosen": chosen, "chosen_facts": len(rows_t),
                    "richest": richest, "richest_facts": len(by_unit[richest]),
                }

        if rows_t:
            newest = max(r["end"] for r in rows_t if r["end"])
            per_tag.append((newest, len(rows_t), tag, rows_t))

    # Most recent coverage first; a longer stale series must not outrank a
    # shorter current one.
    per_tag.sort(key=lambda c: (c[0], c[1]), reverse=True)

    # Dedupe with a few days of tolerance, not on the exact end date.
    #
    # J&J files on a 52/53-week calendar, so two tags in the same chain report
    # the SAME fiscal year one day apart (2021-01-03 and 2021-01-04). Exact-key
    # dedup kept both, and the second was a quarterly declared rate ($1.01)
    # sitting beside the annual figure ($3.98). Inserted mid-series it read as
    # a 3.94x drop and recovery, giving a 5-year streak and a 38% DPS CAGR.
    by_period, contributing = {}, []
    for _, _, tag, rows_t in per_tag:
        added = 0
        for r in rows_t:
            if not r.get("end"):
                continue
            end = date.fromisoformat(r["end"])
            near = next((k for k in by_period
                         if abs((end - date.fromisoformat(k)).days) <= _PERIOD_TOL_DAYS),
                        None)
            if near is None:                       # first (newest) tag wins
                by_period[r["end"]] = r
                added += 1
        if added:
            contributing.append(tag)
    rows = list(by_period.values())

    if not rows:
        return pd.DataFrame(columns=["end", "start", "filed", "fy", "form",
                                     "val", "unit", "tag"])

    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    # Keep the FIRST reported value per period, not the latest restatement —
    # that is what the market actually saw at the time.
    df = (df.sort_values("filed")
            .drop_duplicates(subset=["end"], keep="first")
            .sort_values("end")
            .reset_index(drop=True))
    df.attrs["tags_used"] = contributing
    # Record when a concept was reported in more than one unit, so a mixed
    # filer is visible even though the series itself is now clean.
    df.attrs["units_available"] = sorted(units_seen)
    df.attrs["unit_used"] = rows[0]["unit"] if rows else None
    df.attrs["multi_unit"] = len(units_seen) > 1
    df.attrs["unit_coverage_cost"] = coverage_cost.get(field)
    return df


def derive_q4(annual: pd.Series, quarterly: pd.Series) -> pd.Series:
    """
    Q4 is never reported directly. It is FY minus Q1+Q2+Q3 for flow items
    (revenue, income, cash flow). Balance-sheet instants need no derivation —
    the 10-K value already is the year-end snapshot.
    """
    out = {}
    for fy_end, fy_val in annual.items():
        yr = quarterly[(quarterly.index > fy_end - pd.DateOffset(years=1))
                       & (quarterly.index <= fy_end)]
        if len(yr) >= 3:
            out[fy_end] = fy_val - yr.iloc[:3].sum()
    return pd.Series(out)


def load_fundamentals(tickers: list[str], as_of: date | None = None) -> list:
    """
    Build `screeners.Fundamentals` from EDGAR. Deliberately partial: it fills
    the fields XBRL gives cleanly and leaves derived ones (ROIC, multiples vs
    own history) to be computed against price from Stooq.
    """
    from engines.screeners import Fundamentals

    out, skipped, failed = [], [], {}
    for t in tickers:
        try:
            facts = company_facts(t)
        except Exception as e:
            failed[t] = e
            skipped.append((t, str(e)))
            continue

        rev = extract_series(facts, "revenue", as_of)
        if rev.empty:
            # Loud, not silent — a dropped mega-cap changes every screen.
            failed[t] = "no annual revenue facts"
            skipped.append((t, "no annual revenue facts"))
            continue
        f = Fundamentals(symbol=t, name=facts.get("entityName", t))
        if len(rev) >= 6:
            span = min(5, len(rev) - 1)
            first, last = rev["val"].iloc[-1 - span], rev["val"].iloc[-1]
            if first > 0:
                f.revenue_cagr_5y = ((last / first) ** (1 / span) - 1) * 100
        out.append(f)
    _warn_collapsed("fundamentals", failed, len(tickers))
    if skipped:
        print(f"  [info] {len(skipped)} of {len(tickers)} tickers excluded")
    return out


# ------------------------------------------------------------ Bitcoin

# Binance returns HTTP 451 from US IPs, which includes GitHub-hosted runners.
# Coinbase is US-domiciled and does not geo-block, so it is the durable choice
# for OHLCV. For long close-only history, Coin Metrics is better still: it
# already works, needs no key, and reaches back to 2010 rather than 2015.

def coinbase_candles(granularity: int = 86400, days: int = 300) -> pd.DataFrame:
    """
    Daily BTC-USD OHLCV. Free, no key, no geo-block.

    Capped at 300 candles per request, so this returns the recent window only.
    That is enough for ATR and MFI, which are the two indicators that need
    high/low/volume. Use `load_btc_daily` for the long close series.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=min(days, 300))
    r = _get(COINBASE_CANDLES, params={"granularity": granularity,
                                       "start": start.isoformat(),
                                       "end": end.isoformat()})
    df = pd.DataFrame(r.json(), columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s")
    return df.set_index("t").sort_index()[["open", "high", "low", "close", "volume"]]


def load_btc_daily() -> pd.Series:
    """
    Long daily close history from Coin Metrics — verified live, no key, and it
    reaches back further than any exchange API.
    """
    df = coinmetrics(["PriceUSD"])
    return df["PriceUSD"].rename("close")


def load_btc_weekly() -> pd.Series:
    return load_btc_daily().resample("W-SUN").last().dropna()


def load_btc_ohlcv() -> pd.DataFrame:
    """OHLCV for ATR and MFI, which need high/low/volume rather than close."""
    return coinbase_candles()


# Metrics the community tier silently truncates. ReferenceRateUSD returns a
# rolling SEVEN-DAY window: HTTP 200, seven rows, no error field, no page
# token. Nothing in the response says it was cut. PriceUSD is the same series
# on the same tier with full history back to 2013 — verified identical, offset
# one day because one stamps interval-end and the other interval-start.
TRUNCATED_ON_COMMUNITY = {"ReferenceRateUSD": "PriceUSD"}


def coinmetrics(metrics: list[str], asset: str = "btc",
                start: str = "2013-01-01",
                min_expected_rows: int | None = None) -> pd.DataFrame:
    """
    Coin Metrics Community API — free, no key.

    CapMrktCurUSD (market cap) and CapMVRVCur (MVRV ratio) are both in the
    community tier. CapRealUSD sometimes is not, so realized cap is derived as
    market cap / MVRV, which stays auditable because MVRV is Coin Metrics' own
    realized-cap ratio.

    Guards against silent truncation, which is the failure mode that cost a
    debugging session here: a short series propagates as NaN through every
    rolling window and surfaces four layers downstream as an unrelated
    TypeError. Better to fail at the source with a message that names the
    problem.
    """
    swapped = [TRUNCATED_ON_COMMUNITY.get(m, m) for m in metrics]
    for old, new in zip(metrics, swapped):
        if old != new:
            print(f"  [info] {old} is 7-day-capped on the community tier; "
                  f"using {new}")
    metrics = swapped

    rows, page = [], None
    while True:
        params = {"assets": asset, "metrics": ",".join(metrics),
                  "frequency": "1d", "start_time": start, "page_size": 10000}
        if page:
            params["next_page_token"] = page
        j = _get(CM_METRICS, params=params).json()
        rows += j.get("data", [])
        page = j.get("next_page_token")
        if not page:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Coin Metrics returned no rows for {metrics}")
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    for m in metrics:
        if m in df:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # A request starting in 2013 that returns a fortnight was truncated, not
    # answered. HTTP 200 is not evidence of a complete response.
    floor = min_expected_rows if min_expected_rows is not None else 365
    if len(df) < floor:
        span = (df["time"].max() - df["time"].min()).days
        raise RuntimeError(
            f"Coin Metrics returned only {len(df)} rows spanning {span} days "
            f"for {metrics} (expected >={floor} from {start}). Likely a "
            "community-tier window cap — check TRUNCATED_ON_COMMUNITY.")
    return df.set_index("time")


def btc_onchain() -> pd.DataFrame:
    df = coinmetrics(["CapMrktCurUSD", "CapMVRVCur", "SplyCur"])
    df["realized_cap"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
    df["realized_price"] = df["realized_cap"] / df["SplyCur"]
    return df


def load_sth_mvrv() -> float:
    """
    SUBSTITUTE, and the substitution matters.

    Short-term holder MVRV — the 155-day cohort's cost basis — is Glassnode
    proprietary with no free equivalent. This returns AGGREGATE MVRV instead:
    all holders, not recent ones.

    Consequence: aggregate MVRV is slower and less sensitive. It is excellent
    at cycle extremes (below 1.0 has marked generational lows) and much duller
    mid-cycle, which is exactly where the dip-buy rule wants to fire. Treat the
    second leg of that rule as degraded on the free tier, and lean harder on
    the support band. The UI should say so rather than implying parity.
    """
    df = btc_onchain()
    return float(df["CapMVRVCur"].iloc[-1])


# -------------------------------------------------------- politicians

def load_ptrs() -> list:
    """
    DEAD SOURCE as of September 2026.

    Both House and Senate Stock Watcher S3 buckets return 403 at the bucket
    root and both domains now fail DNS. They were volunteer-maintained and have
    been shut down. There is no drop-in free replacement.

    What remains, in descending order of effort:

      1. House Clerk annual ZIP -> XML index. Free, works, and gives member +
         filing date + DocID. The TRANSACTIONS (ticker, amount, side) are only
         in the per-filing PDFs, so this alone cannot feed the engine.
      2. Parse those PDFs with pdfplumber. Recent filings are mostly native
         text; older ones are scans needing OCR. This is the two-weekend job
         the Stock Watcher dependency was avoiding.
      3. Paid: Quiver (~$10-75/mo) or Unusual Whales (~$48/mo).

    Until one of those is done, this raises rather than returning [] — an empty
    list would let the politicians engine report "no trades" when the truth is
    "no data source".
    """
    raise NotImplementedError(
        "Stock Watcher is dead (403 + NXDOMAIN, Sept 2026). Parse House Clerk "
        "PDFs via capitol_flow.fetch_house_ptrs, or buy Quiver. See docstring.")


def _load_ptrs_stockwatcher_ARCHIVED() -> list:
    """Kept for reference if the buckets ever return."""
    from engines.capitol_flow import Trade

    out = []
    for url, chamber in ((HOUSE_SW, "House"), (SENATE_SW, "Senate")):
        try:
            data = _get(url).json()
        except Exception as e:
            print(f"  [warn] {chamber} Stock Watcher: {e}")
            continue
        for r in data:
            tx, fd = r.get("transaction_date"), r.get("disclosure_date")
            if not (tx and fd and r.get("ticker") not in (None, "--", "")):
                continue
            try:
                out.append(Trade(
                    member=r.get("representative") or r.get("senator", "unknown"),
                    chamber=chamber, party=r.get("party", ""), committees=[],
                    ticker=r["ticker"].upper(),
                    side="Buy" if "purchase" in str(r.get("type", "")).lower() else "Sell",
                    amount_range=r.get("amount", ""),
                    transaction_date=date.fromisoformat(tx[:10]),
                    filing_date=date.fromisoformat(fd[:10]),
                ))
            except (ValueError, KeyError):
                continue
    return out


# --------------------------------------------------------------- news

def load_news(symbol: str, limit: int = 6) -> list[dict]:
    """
    Yahoo Finance RSS — free, no key, no registration.

    Tag each item against the stated bull/bear triggers before showing it. An
    untagged feed is noise; the value is the link back to the thesis.
    """
    import xml.etree.ElementTree as ET

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    try:
        root = ET.fromstring(_get(url).text)
    except Exception as e:
        # Per-symbol failure is tolerable; the caller treats [] as "no items".
        # If Yahoo's RSS endpoint is retired, this will quietly return [] for
        # every symbol — check data/status.json rather than trusting silence.
        print(f"  [warn] news {symbol}: {e}")
        return []
    items = []
    for it in root.iter("item"):
        items.append({"headline": (it.findtext("title") or "").strip(),
                      "url": it.findtext("link"),
                      "published": it.findtext("pubDate"),
                      "source": "Yahoo Finance RSS",
                      "thesis_tag": None})         # fill in downstream
        if len(items) >= limit:
            break
    return items


# ------------------------------------------------------------- helpers

def load_uninversion_date() -> date | None:
    """Last date the 10y-2y spread crossed back above zero."""
    s = fred_series("T10Y2Y").dropna()
    pos = s > 0
    # fill_value=False keeps bool dtype. Using .shift(1).fillna(False) upcasts
    # to object, after which ~True is Python's integer bitwise-not (-2, truthy)
    # and almost every day reads as a crossing. This flagged 10,487 of 12,566.
    flips = pos & ~pos.shift(1, fill_value=False)
    return flips[flips].index[-1].date() if flips.any() else None


def claims_trend_score(claims: pd.Series) -> float:
    ma4 = claims.rolling(4).mean()
    chg = (ma4.iloc[-1] / ma4.iloc[-26] - 1) * 100 if len(ma4) > 26 else 0.0
    return float(min(max((chg + 5) / 25 * 100, 0), 100))


def nfci_score(nfci: pd.Series) -> float:
    """NFCI is standardised: 0 is average, positive means tightening."""
    return float(min(max((nfci.iloc[-1] + 1) / 2 * 100, 0), 100))


def load_lei_score() -> float:
    """
    Conference Board LEI is licensed and not on FRED. On the free tier,
    substitute ISM new orders (also not free) or drop the component — it is
    15% of the lead score. Returning a neutral 50 keeps the composite honest
    rather than silently inventing a reading.
    """
    return 50.0
