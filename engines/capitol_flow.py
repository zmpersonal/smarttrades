"""
SmartTrades.AI — Engine 5: Capitol Flow
=======================================

Ingests congressional Periodic Transaction Reports (PTRs), ranks members by
two-year replicable alpha, and scores each new disclosure for how much of the
move is still available.

The timing problem, stated plainly
----------------------------------
The STOCK Act gives members 45 days to report a trade. A filing you see today
describes something that happened, on average, about a month ago. There is no
feed that fixes this — every commercial "congressional trading alert" product
is working from the same PTRs and is bounded by the same window. What you can
compete on is latency *after* filing: polling the portals every few minutes
and parsing immediately buys you hours over daily-batch competitors.

That distinction drives the whole design:

    ALPHA (TRADE)  what the member earned, entering on the transaction date
    ALPHA (DISC.)  what a follower earns, entering on the filing date

Only the second is available to you, so members are ranked on it. A member
with excellent trade-date alpha who always files on day 44 is worthless to
follow, and the ranking must say so rather than flattering them.

Pending change worth building for
---------------------------------
H.R. 7008 (Stop Insider Trading Act) passed the House 232-198 on 22 July 2026
and is with the Senate. If enacted it bans purchases outright and requires
seven days of ADVANCE public notice before a member sells, published by the
Clerk of the House / Secretary of the Senate on receipt. That would turn sell
signals from six weeks late into a week early. `fetch_advance_sale_notices()`
below is the stub for it. It takes effect 180 days after enactment, so there
is time to build it before it matters.

Data sources
------------
House   disclosures-clerk.house.gov/FinancialDisclosure
        Annual ZIP with an XML index of filings + one PDF per PTR.
        Many older PTRs are scanned images and need OCR.
Senate  efdsearch.senate.gov/search/
        Requires accepting terms to obtain a session cookie, then POST search.
        Better structured than the House, still HTML scraping.

Skipping the PDF work is legitimate: House Stock Watcher and Senate Stock
Watcher publish parsed JSON for free, and Quiver Quantitative, Capitol Trades
and Unusual Whales sell cleaner feeds with faster latency. Parse it yourself
only if latency is the product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd
import requests

HOUSE_INDEX = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
SENATE_SEARCH = "https://efdsearch.senate.gov/search/"

Side = Literal["Buy", "Sell", "Exchange"]

# PTRs disclose ranges, never exact amounts. Midpoints are the standard
# convention; the geometric mean is arguably better for the wide upper
# brackets but the difference rarely changes a ranking.
AMOUNT_RANGES = {
    "$1,001 - $15,000":          (1_001, 15_000),
    "$15,001 - $50,000":         (15_001, 50_000),
    "$50,001 - $100,000":        (50_001, 100_000),
    "$100,001 - $250,000":       (100_001, 250_000),
    "$250,001 - $500,000":       (250_001, 500_000),
    "$500,001 - $1,000,000":     (500_001, 1_000_000),
    "$1,000,001 - $5,000,000":   (1_000_001, 5_000_000),
    "$5,000,001 - $25,000,000":  (5_000_001, 25_000_000),
    "$25,000,001 - $50,000,000": (25_000_001, 50_000_000),
    "Over $50,000,000":          (50_000_001, 100_000_000),
}


def range_midpoint(label: str, geometric: bool = True) -> float:
    key = re.sub(r"\s+", " ", label.strip())
    lo, hi = AMOUNT_RANGES.get(key, (0, 0))
    if lo == 0:
        return 0.0
    return float(np.sqrt(lo * hi)) if geometric else (lo + hi) / 2


@dataclass
class Trade:
    member: str
    chamber: Literal["House", "Senate"]
    party: str
    committees: list[str]
    ticker: str
    side: Side
    amount_range: str
    transaction_date: date
    filing_date: date

    @property
    def lag_days(self) -> int:
        return (self.filing_date - self.transaction_date).days

    @property
    def size_usd(self) -> float:
        return range_midpoint(self.amount_range)


@dataclass
class Config:
    min_trades_for_ranking: int = 25    # below this, alpha is noise
    alpha_window_years: int = 2
    holding_period_days: int = 90       # window over which each trade is scored
    max_useful_lag: int = 38            # past this, treat the signal as spent
    min_follow_score: int = 55

    weights: dict = field(default_factory=lambda: {
        "member_record":     0.26,   # replicable alpha, not raw alpha
        "freshness":         0.22,   # days since filing, and lag on filing
        "committee_overlap": 0.18,   # jurisdiction over the traded sector
        "position_size":     0.14,   # size relative to member's own portfolio
        "cluster":           0.12,   # multiple members, same sector, same window
        "availability":      0.08,   # how much of the move is left
    })


# ---------------------------------------------------------------- ingest

def fetch_house_ptrs(year: int) -> pd.DataFrame:
    """
    Download the House annual filing index. Returns metadata only — each PTR's
    line items live in a separate PDF keyed by DocID and need parsing (pdfplumber
    for native PDFs, OCR for the scanned ones, which are still common).
    """
    import io
    import zipfile
    import xml.etree.ElementTree as ET

    r = requests.get(HOUSE_INDEX.format(year=year), timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        xml_name = next(n for n in z.namelist() if n.endswith(".xml"))
        root = ET.fromstring(z.read(xml_name))

    rows = [{c.tag: c.text for c in m} for m in root.findall("Member")]
    df = pd.DataFrame(rows)
    return df[df.get("FilingType") == "P"]  # P = Periodic Transaction Report


def fetch_senate_ptrs(start: date, end: date) -> pd.DataFrame:
    """
    Senate search requires accepting the terms page first to get a CSRF token
    and session cookie, then POSTing the search. Structured better than the
    House but still HTML.
    """
    s = requests.Session()
    home = s.get(SENATE_SEARCH + "home/", timeout=30)
    token = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', home.text)
    if not token:
        raise RuntimeError("Senate CSRF token not found — the page layout changed.")

    s.post(SENATE_SEARCH + "home/", timeout=30, data={
        "csrfmiddlewaretoken": token.group(1),
        "prohibition_agreement": "1",
    }, headers={"Referer": SENATE_SEARCH + "home/"})

    raise NotImplementedError(
        "Paginate the POST to /search/report/data/ with report_types=[11] "
        "(PTR) and the date range. Returns JSON rows of filings."
    )


def fetch_advance_sale_notices() -> pd.DataFrame:
    """
    Stub for H.R. 7008's seven-day advance sale notice, if the Senate passes it
    and it is signed. The bill requires the Clerk of the House and Secretary of
    the Senate to publish each notice on receipt, and takes effect 180 days
    after enactment.

    This would be the highest-value feed in the entire product — a known
    forward-dated sale from a named member, seven days before it happens.
    Poll it on a tight interval when it goes live.
    """
    raise NotImplementedError("Not law yet. Passed House 2026-07-22; with the Senate.")


# ------------------------------------------------------- member alpha

def score_member_alpha(trades: list[Trade], prices: pd.DataFrame,
                       cfg: Config = Config()) -> pd.DataFrame:
    """
    Two alpha figures per member, both versus SPY over matched holding periods.

    `prices` needs a DatetimeIndex and one column per ticker, plus SPY.

    Sells are scored inverted: avoiding a decline is a correct call, so a sale
    followed by a drop counts as positive alpha.
    """
    rows = []
    for t in trades:
        for label, entry in (("trade", t.transaction_date), ("disc", t.filing_date)):
            exit_d = entry + pd.Timedelta(days=cfg.holding_period_days)
            try:
                px = prices[t.ticker].asof(pd.Timestamp(entry))
                px_x = prices[t.ticker].asof(pd.Timestamp(exit_d))
                bm = prices["SPY"].asof(pd.Timestamp(entry))
                bm_x = prices["SPY"].asof(pd.Timestamp(exit_d))
            except (KeyError, IndexError):
                continue
            if not all(np.isfinite([px, px_x, bm, bm_x])) or px == 0 or bm == 0:
                continue

            alpha = (px_x / px - 1) - (bm_x / bm - 1)
            if t.side == "Sell":
                alpha = -alpha

            rows.append({"member": t.member, "chamber": t.chamber,
                         "committees": ", ".join(t.committees[:1]),
                         "basis": label, "alpha": alpha,
                         "lag": t.lag_days, "size": t.size_usd})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    piv = df.pivot_table(index="member", columns="basis", values="alpha", aggfunc="mean")
    meta = df.groupby("member").agg(
        chamber=("chamber", "first"),
        committee=("committees", "first"),
        trades=("alpha", lambda s: len(s) // 2),
        median_lag=("lag", "median"),
    )
    win = (df[df.basis == "disc"].groupby("member")["alpha"]
           .apply(lambda s: (s > 0).mean()).rename("win_rate"))

    out = meta.join(piv).join(win)
    out = out.rename(columns={"trade": "alpha_trade", "disc": "alpha_disc"})
    out = out[out["trades"] >= cfg.min_trades_for_ranking]

    # Rank on what a follower can actually capture, not what the member earned.
    return out.sort_values("alpha_disc", ascending=False)


# ------------------------------------------------------- follow score

SECTOR_JURISDICTION = {
    "Armed Services":       {"Industrials", "Aerospace & Defense"},
    "Energy & Commerce":    {"Energy", "Utilities", "Health Care", "Technology"},
    "Financial Services":   {"Financials", "Real Estate"},
    "Banking":              {"Financials", "Real Estate"},
    "Ways & Means":         {"Health Care", "Financials"},
    "Agriculture":          {"Consumer Staples", "Materials"},
    "Transportation":       {"Industrials"},
}


def committee_relevance(committees: list[str], sector: str) -> float:
    """Direct jurisdiction over the traded sector is the strongest single
    predictor of forward alpha in the published research on this data."""
    for c in committees:
        for key, sectors in SECTOR_JURISDICTION.items():
            if key.lower() in c.lower() and sector in sectors:
                return 100.0
    return 35.0 if committees else 10.0


def score_disclosure(t: Trade, member_alpha: float, sector: str,
                     pct_of_portfolio: float, cluster_count: int,
                     move_since_trade: float, days_since_filing: int,
                     cfg: Config = Config()) -> dict:
    """
    `move_since_trade` is signed and in percent. For a buy, a large positive
    move means the opportunity is largely gone; for a sell, a large negative
    move means the same. Availability normalises that direction.
    """
    consumed = move_since_trade if t.side == "Buy" else -move_since_trade

    comp = {
        "member_record":     float(np.clip((member_alpha + 5) / 25 * 100, 0, 100)),
        "freshness":         float(np.clip(100 - (t.lag_days / cfg.max_useful_lag) * 80
                                           - days_since_filing * 6, 0, 100)),
        "committee_overlap": committee_relevance(t.committees, sector),
        "position_size":     float(np.clip(pct_of_portfolio / 15 * 100, 0, 100)),
        "cluster":           float(np.clip((cluster_count - 1) / 3 * 100, 0, 100)),
        "availability":      float(np.clip(100 - abs(consumed) * 8, 0, 100)),
    }

    score = sum(comp[k] * w for k, w in cfg.weights.items())

    # A signal filed at the deadline is spent regardless of who filed it.
    if t.lag_days >= cfg.max_useful_lag:
        score *= 0.75

    return {
        "member": t.member, "ticker": t.ticker, "side": t.side,
        "size": t.amount_range, "traded": t.transaction_date.isoformat(),
        "filed": t.filing_date.isoformat(), "lag_days": t.lag_days,
        "since_trade_pct": round(move_since_trade, 1),
        "score": round(score),
        "actionable": score >= cfg.min_follow_score and t.lag_days < cfg.max_useful_lag,
        "components": {k: round(v) for k, v in comp.items()},
    }


# ---------------------------------------------------------------- notes
#
# Known limits, none of which are fixable and all of which should be visible
# in the UI rather than hidden:
#
#   * Amounts are ranges, so portfolio weights are estimates, not measurements.
#   * Spousal and dependent trades are disclosed but not always distinguishable,
#     and a spouse's trade may carry no information at all.
#   * Blind trusts and managed accounts produce trades the member never chose.
#   * A disclosed sale does not reveal what fraction of the position was sold.
#   * Options and derivatives appear inconsistently across filings.
#   * Late filings are common and the $200 penalty is routinely waived, so the
#     45-day window is a soft bound in practice.
#
# Survivorship also matters when ranking: screen on members active across the
# entire window, or you will rank whoever happened to trade during a good run.
