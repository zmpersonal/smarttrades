"""
SmartTrades.AI — Engines 2, 3 and 4
===================================

Every engine follows the same two-stage shape:

    GATES   hard pass/fail. A name that fails any gate never ranks,
            however attractive it looks on the score.
    SCORE   weighted components, each returned individually so the UI
            can show why a name ranks where it does.

Fundamentals input is provider-agnostic. Populate the `Fundamentals`
dataclass from whichever source you use — FMP, EODHD, Tiingo, Sharadar
or Koyfin all carry every field below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Sector = Literal["general", "utility", "reit", "financial", "energy"]


@dataclass
class Fundamentals:
    symbol: str
    name: str
    sector: Sector = "general"

    price: float = 0.0
    market_cap: float = 0.0
    dollar_adv: float = 0.0

    # Dividend
    dividend_yield: float = 0.0          # %
    yield_median_5y: float = 0.0         # %
    yield_std_5y: float = 0.0            # %
    dps_cagr_5y: float = 0.0             # %
    dps_cagr_3y: float = 0.0             # %
    increase_streak_years: int = 0
    years_since_cut: int = 99
    eps_payout: float = 0.0              # %
    fcf_payout: float | None = 0.0              # %

    # Quality / returns
    roic_5y: float | None = 0.0                 # %
    roic_ttm: float | None = 0.0                # %
    roic_declining_years: int | None = 0
    wacc: float = 9.0                    # %
    gross_margin: float | None = 0.0            # %
    gross_margin_delta_3y: float | None = 0.0   # pts
    fcf_margin: float | None = 0.0              # %
    fcf_positive_years_of_10: int | None = 0
    share_count_cagr_5y: float = 0.0     # %  negative = buying back

    # Growth
    revenue_cagr_5y: float = 0.0         # %
    revenue_growth_ttm: float = 0.0      # %
    eps_revision_3m: float = 0.0         # %
    eps_revision_6m: float = 0.0         # %

    # Balance sheet
    net_debt_ebitda: float | None = 0.0
    interest_coverage: float | None = 99.0
    altman_z: float | None = 99.0
    cash_runway_quarters: float = 99.0
    debt_maturing_24m_pct: float = 0.0   # % of total debt

    # Valuation
    ev_ebit: float | None = 0.0   # None = unavailable, never 0 as "cheapest"
    ev_ebit_median_10y: float | None = 0.0
    ev_ebit_percentile_10y: float | None = 50.0
    ev_sales_percentile_5y: float | None = 50.0
    fcf_yield: float | None = 0.0               # %
    reverse_dcf_implied_growth: float = 0.0   # %

    # Price context
    drawdown_from_ath: float = 0.0       # negative %
    pct_above_200dma: float = 0.0        # %
    months_in_base: float = 0.0

    # Data quality — set by fundamentals_builder, consumed by the gates
    dividend_record_ambiguous: bool = False   # large DPS drop, no split data
    data_stale_days: int = 0                  # age of the newest annual period
    ebit_unavailable: bool = False            # no GrossProfit/OperatingIncome tag
    period_mismatch_days: int = 0             # spread between statement series
    roic_unavailable: bool = False            # invested capital <= 0, or no debt tag
    roic_5y_is_ttm_fallback: bool = False     # no usable history; 5y == ttm
    debt_unavailable: bool = False            # no debt component tag at all
    capex_voided_fcf: bool = False            # capex absent; FCF would be fabricated
    altman_not_applicable: bool = False       # utility/financial/REIT/pre-revenue
    pre_revenue: bool = False                 # revenue under $10m — ratios meaningless
    specials_dropped: int = 0                 # bundled regular+special figures removed
    dividend_outliers_dropped: int = 0        # specials plus quarterly stragglers
    # --- financials -------------------------------------------------------
    # Debt is a financial's raw material, not its financing, so invested
    # capital is not a denominator and ROIC is not a measure. ROE is, and the
    # valuation anchor is price to tangible book rather than EV/EBIT.
    financial_subtype: str = ""                # depository | broker | insurer | manager
    financial_in_scope: bool = False           # explicit SIC set AND witness agree
    roe_5y: float | None = None
    roe_ttm: float | None = None
    roe_unavailable: bool = False
    roe_declining_years: int = 0
    rotce: float | None = None                 # return on tangible common equity
    rotce_unavailable: bool = False
    tangible_book: float | None = None
    tangible_book_check: dict = field(default_factory=dict)
    tangible_book_unavailable: bool = False
    equity_to_assets: float | None = None
    price_to_tangible_book: float | None = None
    ptbv_percentile_10y: float | None = None
    ptbv_history_degraded: bool = False
    tbvps_cagr_5y: float | None = None
    cost_of_equity: float = 10.5
    reverse_dcf_unavailable: bool = False     # negative FCF, or price outside the band
    wacc_is_assumed: bool = True              # sector bucket, never measured
    invested_capital_check: dict = field(default_factory=dict)  # cancellation
    ebit_margin_check: dict = field(default_factory=dict)       # materiality
    mixed_unit_concepts: list = field(default_factory=list)  # reported in >1 currency
    unit_coverage_cost: dict = field(default_factory=dict)   # USD chosen over a longer series
    voided_fields: list = field(default_factory=list)        # cleared: input unavailable
    foreign_private_issuer: bool = False      # files 20-F/40-F or ifrs-full
    adr_ratio_unknown: bool = False           # per-share figure vs ADR price
    statement_currency: str = "USD"           # revenue's unit; prices are USD
    fcf_history_years: int = 0                # annual FCF periods available
    fcf_negative_years: int = 0               # of those, how many were negative
    capex_years_missing: int = 0              # years where capex was absent
    ev_short_because_unprofitable: bool = False  # loss-maker, not a data gap
    debt_assumed_zero: bool = False              # no tag AND no interest paid
    derivation_warnings: list = field(default_factory=list)
    stale_concepts: list = field(default_factory=list)   # lagging revenue >450d
    concept_lags: dict = field(default_factory=dict)     # per-concept lag in days
    ev_history_degraded: bool = False         # EV/EBIT history too stale to rank
    ev_ebit_implausible: bool = False         # >150x — a denominator, not a valuation
    gross_profit_unavailable: bool = False    # no GrossProfit tag, none derivable
    fcf_unavailable: bool = False             # no operating cash flow tag

    # Flow
    insider_net_6m: float = 0.0          # $
    buyback_yield: float = 0.0           # %


def _below(v, bound) -> bool:
    """
    True only when v is KNOWN and below bound.

    An unknown value cannot fail a business gate — `data_quality_gates` has
    already excluded the name for the reason the value is unknown, and adding
    "net debt/EBITDA None over 3.5x" beside it is the business-label-on-a-data-
    problem failure again. Voided fields hold None, so every comparison against
    one has to say what None means rather than crash or coerce.
    """
    return v is not None and v < bound


def _above(v, bound) -> bool:
    return v is not None and v > bound


def _s(v, lo, hi):
    """_scale, but a None input yields None so _mean_available can omit it."""
    return None if v is None else _scale(v, lo, hi)


def _d(a, b):
    """a - b, None if either side is unavailable."""
    return None if a is None or b is None else a - b


def _mean_available(terms: list) -> float:
    """
    Average over the terms that EXIST, dropping None, and renormalise.

    `_scale(None)` returning 50.0 is a neutral SUBSTITUTION, which is the thing
    that made missing data outscore a real 55% gross margin — the scale mapped
    35 to 0 and 85 to 100, so a neutral 50 beat it. `_scale`'s None handling
    stays as a last-resort net against unexpected values; this is the intended
    mechanism wherever an input is legitimately absent.
    """
    present = [t for t in terms if t is not None]
    return float(np.mean(present)) if present else 50.0


def _clamp(x, lo=0.0, hi=100.0):
    if x is None:
        return 50.0
    return float(np.clip(x, lo, hi))


def voidable_fields() -> frozenset:
    """
    The fields a record may hold as None: exactly those annotated `| None`.

    This is a CONTRACT, not a description. Every scorer, gate and adapter row
    is tested against all of them at once and each one alone, so annotating a
    field is what enrols it in that test. `build()` refuses to return a record
    with None in any field outside this set, which is what stops the next
    voiding site from reaching a bare `100 - f.x` in production — the bug that
    crashed dividend and recovery, then crashed the adapter one layer out.
    """
    import dataclasses
    return frozenset(x.name for x in dataclasses.fields(Fundamentals)
                     if "None" in str(x.type))


def _scale(v, lo, hi):
    """
    Linear map of v from [lo, hi] onto 0-100, clipped.

    None means UNAVAILABLE and scores neutral, never as a bad value. Voids
    arrive as None from the builder and as 0.0 from `void_derived_fields`, and
    the scorers assumed floats throughout — `_scale(None, ...)` raised, which
    is at least loud. The silent version is worse: an unavailable field scored
    at the bottom of its range would rank a name down for missing data.
    """
    if v is None:
        return 50.0
    if hi == lo:
        return 50.0
    try:
        return _clamp((v - lo) / (hi - lo) * 100)
    except TypeError:
        return 50.0


# =====================================================================
# ENGINE 2 — Dividend Growers at Elevated Yield
# =====================================================================

def dividend_gates(f: Fundamentals) -> list[str]:
    """Return the list of failures. Empty list = passes."""
    fails = data_quality_gates(f)
    payout_cap = 85.0 if f.sector in ("reit", "utility") else 65.0
    fcf_cap    = 90.0 if f.sector in ("reit", "utility") else 70.0
    debt_cap   = 6.0  if f.sector in ("reit", "utility") else 3.5

    if f.increase_streak_years < 7:
        fails.append(f"only {f.increase_streak_years}y increase streak (need 7)")
    if f.years_since_cut < 10:
        fails.append(f"cut dividend {f.years_since_cut}y ago")
    if f.dps_cagr_5y < 5:
        fails.append(f"5y DPS CAGR {f.dps_cagr_5y:.1f}% below 5%")
    if f.eps_payout > payout_cap:
        fails.append(f"EPS payout {f.eps_payout:.0f}% over {payout_cap:.0f}%")
    if not f.fcf_unavailable and _above(f.fcf_payout, fcf_cap):
        fails.append(f"FCF payout {f.fcf_payout:.0f}% over {fcf_cap:.0f}%")
    if _above(f.net_debt_ebitda, debt_cap):
        fails.append(f"net debt/EBITDA {f.net_debt_ebitda:.1f}x over {debt_cap:.1f}x")
    if not f.ebit_unavailable and _below(f.interest_coverage, 4):
        fails.append(f"interest coverage {f.interest_coverage:.1f}x under 4x")
    if f.revenue_cagr_5y <= 0:
        fails.append("revenue not growing over 5y")
    if f.market_cap < 2e9 or f.dollar_adv < 5e6:
        fails.append("below size or liquidity floor")

    # Yield-trap filter: the yield is only "elevated" if the price fell.
    # If forward estimates collapsed, the yield is elevated because the
    # denominator is about to be revised, not because the stock is cheap.
    if f.eps_revision_6m < -20:
        fails.append(f"forward EPS cut {f.eps_revision_6m:.0f}% in 6m — yield trap")
    # An unexplained DPS drop is either a split or a cut, and those point in
    # opposite directions. Excluding on "cannot verify" is the safe error:
    # wrongly passing a genuine cut is far more expensive than missing a name.
    if f.dividend_record_ambiguous:
        fails.append("unexplained DPS drop with no split data — record unverifiable")
    if f.adr_ratio_unknown:
        fails.append("foreign issuer — per-share dividend cannot be matched to "
                     "an ADR price without the ratio, which XBRL does not carry")
    return fails


def score_dividend(f: Fundamentals) -> dict:
    yz = ((f.dividend_yield - f.yield_median_5y) / f.yield_std_5y
          if f.yield_std_5y > 0 else 0.0)
    chowder = f.dividend_yield + f.dps_cagr_5y
    chowder_target = 15.0 if f.dividend_yield < 3 else 12.0

    comp = {
        # The differentiator: yield relative to the company's OWN history.
        "yield_vs_own_history": _scale(yz, -0.5, 3.0),

        # Voided inputs are None. run_screen scores EVERY name before filtering,
        # so arithmetic on a voided field crashed dividend and recovery on the
        # first live run. Omit, never substitute.
        "dividend_safety": _clamp(_mean_available([
            _s(_d(100, f.fcf_payout), 20, 70),
            _s(_d(100, f.eps_payout), 25, 70),
            _s(_d(4.0, f.net_debt_ebitda), 0.5, 4.0),
            _s(f.interest_coverage, 4, 20),
        ])),

        "growth_durability": _clamp(_mean_available([
            _scale(f.dps_cagr_5y, 5, 15),
            _scale(chowder / chowder_target * 100, 70, 140),
            _scale(f.revenue_cagr_5y, 0, 10),
            _s(f.roic_5y, 8, 25),
            # Reward acceleration, penalise a fading raise cadence
            _scale(f.dps_cagr_3y - f.dps_cagr_5y, -4, 4),
        ])),

        "valuation": _clamp(_mean_available([
            _s(_d(100, f.ev_ebit_percentile_10y), 20, 90),
            _scale(f.fcf_yield, 3, 10),
        ])),

        "balance_sheet": _clamp(_mean_available([
            _s(_d(3.5, f.net_debt_ebitda), 0, 3.5),
            _s(f.interest_coverage, 4, 25),
        ])),
    }

    weights = {"yield_vs_own_history": 0.30, "dividend_safety": 0.28,
               "growth_durability": 0.24, "valuation": 0.12,
               "balance_sheet": 0.06}

    return {
        "symbol": f.symbol, "name": f.name,
        "score": round(sum(comp[k] * w for k, w in weights.items())),
        "yield_z": round(yz, 2), "chowder": round(chowder, 1),
        "gates_failed": dividend_gates(f),
        "components": {k: round(v) for k, v in comp.items()},
    }


# =====================================================================
# ENGINE 3 — Recovery Compounders (3x in two years)
# =====================================================================

def recovery_gates(f: Fundamentals) -> list[str]:
    fails = data_quality_gates(f)
    if not (-85 <= f.drawdown_from_ath <= -50):
        fails.append(f"drawdown {f.drawdown_from_ath:.0f}% outside the -50% to -85% band")
    if _below(f.fcf_margin, 0) and f.cash_runway_quarters < 8:
        fails.append(f"burning cash with only {f.cash_runway_quarters:.0f}q runway")
    if _above(f.net_debt_ebitda, 4):
        fails.append(f"net debt/EBITDA {f.net_debt_ebitda:.1f}x over 4x")
    if f.debt_maturing_24m_pct > 30:
        fails.append(f"{f.debt_maturing_24m_pct:.0f}% of debt matures inside 24m")
    if not f.gross_profit_unavailable and _below(f.gross_margin, 30):
        fails.append(f"gross margin {f.gross_margin:.0f}% under 30%")
    if not f.gross_profit_unavailable and _above(abs(f.gross_margin_delta_3y or 0), 5):
        fails.append(f"gross margin moved {f.gross_margin_delta_3y:+.1f}pts over 3y — unstable")
    if f.share_count_cagr_5y > 5:
        fails.append(f"share count growing {f.share_count_cagr_5y:.1f}%/yr — dilution")
    # Skip for sectors the model was never fitted on. A regulated utility at
    # Z 0.54 is describing its capital structure, not its solvency.
    if not f.altman_not_applicable and _below(f.altman_z, 1.8):
        fails.append(f"Altman Z {f.altman_z:.1f} under 1.8")
    if f.revenue_cagr_5y < 8 and f.revenue_growth_ttm < 0:
        fails.append("no growth history and no evidence of a trough")
    return fails


def decompose_upside(rev_growth_2y: float, margin_factor: float,
                     multiple_factor: float) -> dict:
    """
    A tripling is arithmetic, not a hope. Show the three legs.

        3.0x  =  (1 + revenue growth)  ×  margin factor  ×  multiple factor

    Returning the share each leg contributes is the honest part: a case
    that leans 85% on multiple re-rating is far more fragile than one
    where revenue does most of the work, even at the same headline number.
    """
    total = (1 + rev_growth_2y / 100) * margin_factor * multiple_factor
    legs = {
        "revenue": np.log1p(rev_growth_2y / 100),
        "margin": np.log(margin_factor),
        "multiple": np.log(multiple_factor),
    }
    denom = sum(abs(v) for v in legs.values()) or 1.0
    return {
        "total_multiple": round(total, 2),
        "annualized_pct": round((total ** 0.5 - 1) * 100, 1),
        "leg_share": {k: round(abs(v) / denom * 100) for k, v in legs.items()},
    }


def score_recovery(f: Fundamentals, rev_growth_2y: float,
                   margin_factor: float, multiple_factor: float) -> dict:
    path = decompose_upside(rev_growth_2y, margin_factor, multiple_factor)

    comp = {
        # Same shape as the value screen's discount_to_own_history, same fix:
        # relative to its own history, plus an absolute anchor so a name that
        # has always been expensive cannot score full marks for being slightly
        # less so.
        "valuation_gap": _clamp(_mean_available([
            None if f.ev_sales_percentile_5y is None
            else _scale(100 - f.ev_sales_percentile_5y, 40, 95),
            None if f.ev_ebit is None else _scale(30 - f.ev_ebit, 5, 22),
        ])),

        "durability": _clamp(_mean_available([
            _s(f.altman_z, 1.8, 6.0),
            _scale(min(f.cash_runway_quarters, 24), 8, 24),
            _s(_d(4, f.net_debt_ebitda), 0, 4),
            _s(f.fcf_margin, -10, 20),
        ])),

        "reacceleration": _clamp(np.mean([
            _scale(f.revenue_growth_ttm, -5, 30),
            _scale(f.eps_revision_3m, -10, 10),
            _scale(f.revenue_cagr_5y, 0, 25),
        ])),

        "insider_and_buyback": _clamp(np.mean([
            _scale(f.insider_net_6m / 1e6, -2, 15),
            _scale(f.buyback_yield, 0, 8),
            _scale(-f.share_count_cagr_5y, -3, 5),
        ])),

        "technical_base": _clamp(np.mean([
            _scale(f.months_in_base, 2, 12),
            _scale(f.pct_above_200dma, -20, 10),
        ])),
    }

    weights = {"valuation_gap": 0.25, "durability": 0.25,
               "reacceleration": 0.20, "insider_and_buyback": 0.15,
               "technical_base": 0.15}
    score = sum(comp[k] * w for k, w in weights.items())

    # A case carried almost entirely by multiple re-rating is the fragile
    # kind. Penalise it rather than letting cheapness alone drive the rank.
    if path["leg_share"]["multiple"] > 70:
        score *= 0.90

    return {
        "symbol": f.symbol, "name": f.name,
        "score": round(score),
        "path": path,
        "gates_failed": recovery_gates(f),
        "qualifies": path["total_multiple"] >= 2.5 and not recovery_gates(f),
        "components": {k: round(v) for k, v in comp.items()},
    }


# =====================================================================
# ENGINE 4 — Undervalued Quality
# =====================================================================

def data_quality_gates(f: Fundamentals) -> list[str]:
    """
    Shared preconditions. These were set by the builder and read by only one
    screen, so quality and recovery went on ranking names whose revenue series
    stopped in 2017 — Coca-Cola failed on "revenue declining -5.9%" purely as
    an artifact of a frozen series.
    """
    fails = []
    if f.data_stale_days > 550:
        fails.append(f"newest annual filing is {f.data_stale_days} days old")
    if f.period_mismatch_days > 400:
        fails.append(f"statement periods differ by {f.period_mismatch_days}d "
                     "— ratios are cross-period")
    if f.ebit_unavailable:
        fails.append("no derivable EBIT — ROIC and EV/EBIT unavailable")
    # _derivation_integrity produced these last session and NOTHING consumed
    # them — the check ran and its output only ever reached stdout. A
    # composition failure means a derived quantity came out empty while its
    # inputs were current, which is not a company fact.
    # Two variants, and only one gated. "composition failure" means the derived
    # quantity came out EMPTY; "truncating the join" means it came out STALE —
    # EnerSys shipped a wrong-signed leverage ratio through the second, with
    # data_quality_gates empty and voided_fields empty. A stale derived value
    # feeding a gate is not better than a missing one.
    _bad = [w for w in f.derivation_warnings
            if "composition failure" in w or "truncating the join" in w]
    if _bad:
        fails.append(_bad[0])
    # Without this the absence reads as "gross margin 0%", a business failure
    # label on a data problem. Exxon reports no GrossProfit tag at all, which
    # is normal for oil majors, and the whole sector would be rejected for
    # looking catastrophically unprofitable.
    if f.pre_revenue:
        fails.append("revenue under $10m — margins and ratios are not meaningful")
    if f.statement_currency != "USD":
        fails.append(f"statements in {f.statement_currency}, price in USD — "
                     "every price ratio is cross-currency")
    # Two different questions, and the >= 3 threshold only answered one.
    #
    # HDFC Bank has exactly ONE mixed concept — dividends_per_share in INR and
    # USD — and that single concept is what produced the 47.45% yield. The
    # field was written, the read was reachable, and the threshold sat ABOVE
    # the case that motivated building the check. Pinning corrects the series,
    # but pinning had to CHOOSE, and a wrong choice on a ratio-critical concept
    # is silent. That choice should be visible.
    _CRITICAL_UNITS = {"revenue", "dividends_per_share", "net_income",
                       "equity", "assets", "ocf", "shares"}
    _crit = sorted(set(f.mixed_unit_concepts) & _CRITICAL_UNITS)
    if _crit and len(f.mixed_unit_concepts) < 3:
        fails.append(f"{', '.join(_crit)} reported in multiple currencies — "
                     "a unit was chosen, verify before trusting the ratio")

    # Three or more is a different problem: the whole submission is not
    # internally comparable, as with Nebius across nine concepts.
    if len(f.mixed_unit_concepts) >= 3:
        fails.append(f"{len(f.mixed_unit_concepts)} concepts reported in multiple "
                     f"currencies ({', '.join(f.mixed_unit_concepts[:4])}) "
                     "— statement is not internally comparable")
    if f.fcf_unavailable:
        # Name the actual cause. "No operating cash flow tag" is wrong when OCF
        # is present and capex is the missing subtrahend.
        fails.append(
            "capex not tagged — free cash flow would be operating cash flow, "
            "which is fabrication not approximation"
            if f.capex_voided_fcf
            else "no operating cash flow tag — FCF unknown, not 0%")
    # A concept present in the filing but years behind revenue is a tag
    # migration, not a company that stopped having debt. Only gate on the ones
    # that actually drive gates.
    _critical = {"debt", "debt_noncurrent", "cash", "equity", "ocf"}
    _bad = sorted(set(f.stale_concepts) & _critical)
    if _bad:
        worst = max(f.concept_lags.get(c, 0) for c in _bad)
        fails.append(f"{', '.join(_bad)} lag revenue by up to {worst}d "
                     "— likely tag migration, not a real change")
    # gross_profit_unavailable is deliberately NOT a hard failure. See
    # data_quality_report() below for why, and for where it is still counted.
    return fails


# Per-sub-bucket gate thresholds.
#
# A shared 6% equity-to-assets floor is thin-but-survivable for a bank and
# alarming for an insurer. If the SCORING scale had to be per-bucket — banks
# live at 7-11% and insurers at 15-30%, so a common range read a bank's
# defining leverage as weakness — then the GATE threshold does too, for the
# same reason and on the same evidence.
#
# ROE floors track how much of the return a balance sheet is doing: a capital-
# light fee business earning only 10% on equity is doing badly, while a bank
# earning 10% on a leveraged book is doing acceptably.
FINANCIAL_FLOORS = {
    #              roe   equity/assets   tangible book
    "depository": (10.0,  6.0,  True),   # 6% ~ a 10-12% Tier 1 ratio
    "broker":     (10.0,  8.0,  True),   # thinner books, less stable funding
    "insurer":    (10.0, 12.0,  True),   # underwriters hold reserves; 12% is lean
    "manager":    (12.0, 15.0,  True),   # capital-light: equity should be most of it
    "fee_based":  (12.0, 15.0,  True),   # same economics as a manager
    "":           (10.0,  6.0,  True),   # unresolved sub-bucket, most permissive
}


def financial_gates(f: Fundamentals) -> list[str]:
    """
    Return the list of failures. Empty list = passes.

    Built on the concepts that all three sub-buckets share. Net interest
    margin, efficiency ratio, provisions and CET1 are deliberately NOT gates:
    coverage across the 176 is 42-65% because they are bank-only, and gating
    on them would exclude every insurer and asset manager by construction —
    the same structural exclusion ROIC imposes on utilities.
    """
    fails = []
    if not f.financial_in_scope:
        fails.append("outside the financial screen's scope — SIC set and "
                     "filing witness do not agree it is a bank, broker, "
                     "insurer or asset manager")
        return fails

    # ROE and equity-to-assets are currency-free, but price to tangible book
    # divides a USD price by book in the filer's currency. Scotiabank reports
    # in CAD, Itau in BRL.
    if f.statement_currency != "USD":
        fails.append(f"statements in {f.statement_currency}, price in USD — "
                     "price to tangible book is cross-currency")
        return fails

    if f.roe_unavailable or f.roe_5y is None:
        fails.append("ROE unavailable — no equity or no net income")
        return fails

    roe_floor, ea_floor, _tb_required = FINANCIAL_FLOORS.get(
        f.financial_subtype, FINANCIAL_FLOORS[""])
    if _below(f.roe_5y, roe_floor):
        fails.append(f"5y ROE {f.roe_5y:.1f}% under {roe_floor:.0f}% "
                     f"for a {f.financial_subtype or 'financial'}")
    if _below(f.roe_5y, f.cost_of_equity):
        fails.append(f"ROE {f.roe_5y:.1f}% under cost of equity "
                     f"{f.cost_of_equity:.1f}%")
    if _below(f.equity_to_assets, ea_floor):
        fails.append(f"equity/assets {f.equity_to_assets:.1f}% under "
                     f"{ea_floor:.0f}% for a {f.financial_subtype or 'financial'} "
                     "— too thinly capitalised")
    if f.tangible_book is not None and f.tangible_book <= 0:
        fails.append("tangible book at or below zero — the book is "
                     "acquisition accounting")

    # Erosion needs a floor, not just a direction. Microsoft's 27% ROIC falling
    # three years running is mean reversion from exceptional to excellent.
    if f.roe_declining_years >= 3 and _below(f.roe_5y, 12.0):
        fails.append(f"ROE fell {f.roe_declining_years}y to {f.roe_5y:.1f}% "
                     "— eroding")
    if _above(f.share_count_cagr_5y, 0.5):
        fails.append(f"share count growing {f.share_count_cagr_5y:.1f}%/yr — "
                     "dilution")
    if f.revenue_cagr_5y is not None and f.revenue_cagr_5y <= 0:
        fails.append("revenue not growing over 5y")
    return fails


def financial_distributions(universe: list[Fundamentals]) -> dict:
    """
    Sub-bucket distributions, computed across the in-scope set in one pass.

    Capital strength cannot share a scale across sub-buckets. Banks live at
    7-11% equity-to-assets and insurers at 15-30%, so a common 6-to-16 range
    read a bank's defining leverage as weakness and capped it near 45 however
    good it was — the same failure ROIC imposes on utilities, one level down.

    A percentile WITHIN the sub-bucket is used rather than per-bucket
    constants, for the reason the rest of this project prefers self-referencing
    measures (yield against own median, EV/EBIT against own history): it
    self-corrects if the population shifts and cannot silently acquire the same
    bias again.
    """
    dist: dict = {}
    for f in universe:
        if not f.financial_in_scope or not f.financial_subtype:
            continue
        b = dist.setdefault(f.financial_subtype, {"equity_to_assets": []})
        if f.equity_to_assets is not None:
            b["equity_to_assets"].append(float(f.equity_to_assets))
    for b in dist.values():
        for k in b:
            b[k].sort()
    return dist


def _pct_within(value, peers: list) -> float | None:
    """Percentile of `value` among its sub-bucket peers. None when unusable."""
    if value is None or not peers or len(peers) < 5:
        return None
    return float(sum(1 for p in peers if p < value) / len(peers) * 100)


def score_financial(f: Fundamentals, dist: dict | None = None) -> dict:
    """
    Five components, each averaging over the terms that EXIST.

    `_mean_available` rather than `_scale(None)`: a neutral 50 substitution let
    missing data outscore a real reading, so an absent ROTCE drops out of the
    mean and the component renormalises over what survives.
    """
    spread = (None if f.roe_5y is None
              else f.roe_5y - f.cost_of_equity)

    comp = {
        "returns": _clamp(_mean_available([
            None if f.roe_5y is None else _scale(f.roe_5y, 8, 22),
            None if f.rotce is None else _scale(f.rotce, 10, 28),
            None if f.roe_ttm is None or f.roe_5y is None
            else _scale(f.roe_ttm - f.roe_5y, -4, 4),
        ])),
        # Capital strength is scored against the name's OWN sub-bucket. When
        # no distribution is supplied the term drops out rather than falling
        # back to a shared scale — a silent fallback is how the bias would
        # return.
        "capital_strength": _clamp(_mean_available([
            _pct_within(f.equity_to_assets,
                        (dist or {}).get(f.financial_subtype, {})
                        .get("equity_to_assets", [])),
            None if spread is None else _scale(spread, 0, 12),
        ])),
        # Price to tangible book against its own history — the anchor that
        # replaces EV/EBIT, which has no meaning where debt is raw material.
        "valuation_vs_own_history": _clamp(_mean_available([
            None if f.ptbv_percentile_10y is None
            else _scale(100 - f.ptbv_percentile_10y, 20, 90),
            None if f.price_to_tangible_book is None
            else _scale(3.0 - f.price_to_tangible_book, -0.5, 2.0),
        ])),
        "growth_durability": _clamp(_mean_available([
            None if f.tbvps_cagr_5y is None else _scale(f.tbvps_cagr_5y, 0, 12),
            None if f.revenue_cagr_5y is None else _scale(f.revenue_cagr_5y, 0, 10),
        ])),
        "shareholder_return": _clamp(_mean_available([
            _scale(f.dividend_yield + f.buyback_yield, 0, 8),
        ])),
    }
    weights = {"returns": 0.30, "capital_strength": 0.25,
               "valuation_vs_own_history": 0.25, "growth_durability": 0.12,
               "shareholder_return": 0.08}
    return {
        "symbol": f.symbol, "name": f.name,
        "score": round(sum(comp[k] * w for k, w in weights.items())),
        "subtype": f.financial_subtype,
        "roe_spread": None if spread is None else round(spread, 2),
        "gates_failed": financial_gates(f),
        "components": {k: round(v) for k, v in comp.items()},
    }


def data_quality_report(f: Fundamentals) -> dict:
    """
    Every data-quality flag in one place, regardless of which screen gates on
    it. Measurement and gating are different jobs.

    `ev_history_degraded` is read only by the value screen and
    `dividend_record_ambiguous` only by the dividend screen — correctly, since
    a stale EV history does not make a company a worse dividend payer. But that
    made the true data-loss rate unmeasurable: a universe run counted 35.3%
    from the shared gates while 46.5% carried a degraded EV history and 23.5%
    an unverifiable dividend record, invisible in the headline.
    """
    flags = {
        "data_stale": f.data_stale_days > 550,
        "period_mismatch": f.period_mismatch_days > 400,
        "ebit_unavailable": f.ebit_unavailable,
        "fcf_unavailable": f.fcf_unavailable,
        "gross_profit_unavailable": f.gross_profit_unavailable,
        "ev_history_degraded": f.ev_history_degraded,
        "dividend_record_ambiguous": f.dividend_record_ambiguous,
        "pre_revenue": f.pre_revenue,
        "mixed_units": bool(f.mixed_unit_concepts),
        "unit_coverage_cost": bool(f.unit_coverage_cost),
        "voided_fields": bool(f.voided_fields),
        "adr_ratio_unknown": f.adr_ratio_unknown,
        "statement_currency_not_usd": f.statement_currency != "USD",
        # Assumptions and degradations that must be visible even where they do
        # not gate. Each of these was set by the builder and read by nothing.
        "debt_unavailable": f.debt_unavailable,
        "debt_assumed_zero": f.debt_assumed_zero,
        "roic_5y_is_ttm_fallback": f.roic_5y_is_ttm_fallback,
        "derivation_warnings": bool(f.derivation_warnings),
        # Financial degradation flags. Read here so the shared report is
        # the one place every screen's data state is visible.
        "rotce_unavailable": f.rotce_unavailable,
        "tangible_book_unavailable": f.tangible_book_unavailable,
        "ptbv_history_degraded": f.ptbv_history_degraded,
    }
    shared = data_quality_gates(f)

    # Per-screen hard exclusions caused by DATA, not business.
    #
    # `ev_history_degraded` is a hard gate inside quality_gates but sits
    # outside data_quality_gates, so it was reported as "degraded only" while
    # hard-excluding 84 of 190 names from the value screen — 44%. A data reason
    # producing a hard exclusion must be counted as one, for the screen it
    # affects. Same for dividend_record_ambiguous on the dividend screen.
    per_screen = {
        "quality": list(shared) + (
            [("too many loss-making years to rank on EV/EBIT"
              if f.ev_short_because_unprofitable
              else "EV/EBIT history too short or stale to rank")]
            if f.ev_history_degraded else []),
        "dividend": list(shared) + (
            ["unexplained DPS drop with no split data"]
            if f.dividend_record_ambiguous else []),
        "recovery": list(shared),
    }

    return {"flags": flags, "any_flag": any(flags.values()),
            "hard_failures": shared, "hard_failed": bool(shared),
            "hard_by_screen": per_screen,
            "hard_failed_any_screen": any(bool(v) for v in per_screen.values()),
            # True only when NO screen is hard-excluded by data.
            "degraded_only": (any(flags.values())
                              and not any(bool(v) for v in per_screen.values()))}


def quality_gates(f: Fundamentals) -> list[str]:
    fails = data_quality_gates(f)
    if f.roic_unavailable:
        fails.append("ROIC unavailable — invested capital <= 0 or no debt tag")
    elif not f.ebit_unavailable:
        if _below(f.roic_5y, 12):
            fails.append(f"5y ROIC {f.roic_5y:.1f}% under 12%")
        if f.roic_5y is not None and f.roic_5y <= f.wacc:
            fails.append(f"ROIC {f.roic_5y:.1f}% does not exceed WACC {f.wacc:.1f}%")
    # Gross margin gates ONLY when the figure is actually available.
    #
    # A 207-name universe run found gross profit missing or stale for a large
    # fraction of US large caps — ASC 606 presentation does not require the
    # tag, so Amazon's series ends 2009, Netflix's 2011, Oracle's 2018. As a
    # hard gate this rejected real companies for a filing-presentation choice,
    # which is measuring the filing rather than the business. ROIC, FCF margin
    # and leverage all remain hard gates and are well covered; gross margin now
    # contributes to the score and the absence is surfaced, not fatal.
    if not f.gross_profit_unavailable and _below(f.gross_margin, 35):
        fails.append(f"gross margin {f.gross_margin:.0f}% under 35%")
    if not f.fcf_unavailable and _below(f.fcf_margin, 8):
        fails.append(f"FCF margin {f.fcf_margin:.1f}% under 8%")
    if _above(f.net_debt_ebitda, 2.5):
        fails.append(f"net debt/EBITDA {f.net_debt_ebitda:.1f}x over 2.5x")
    # Test a RATE, not an absolute count, and require enough history to judge.
    # The absolute form failed any company listed under eight years regardless
    # of profitability — 56 of 102 failures on a 207-name universe.
    if not f.fcf_unavailable and f.fcf_positive_years_of_10 is not None:
        yrs = f.fcf_history_years or f.fcf_positive_years_of_10
        if yrs < 4:
            fails.append(f"only {yrs}y of FCF history — too short to judge")
        elif f.fcf_positive_years_of_10 / max(yrs, 1) < 0.8:
            fails.append(f"FCF positive {f.fcf_positive_years_of_10}/{yrs} years "
                         f"({f.fcf_positive_years_of_10 / yrs:.0%})")
    if f.share_count_cagr_5y > 0.5:
        fails.append(f"share count growing {f.share_count_cagr_5y:.1f}%/yr")
    if f.revenue_cagr_5y < 4:
        fails.append(f"5y revenue CAGR {f.revenue_cagr_5y:.1f}% under 4%")

    # Cannot rank undervaluation without a usable valuation history. This is
    # specific to the value screen rather than shared, because a degraded
    # EV/EBIT history does not make a company lower quality.
    # Cause-level: a near-zero operating margin, said plainly, rather than a
    # symptom-level complaint about a large multiple.
    if f.ebit_margin_check and not f.ebit_margin_check.get("reliable", True):
        fails.append(f.ebit_margin_check["reasons"][0])
    if f.ev_ebit_implausible or f.ev_ebit is None:
        fails.append("EV/EBIT unavailable — near-zero or negative EBIT, "
                     "which is a denominator not a valuation")
    if f.ev_history_degraded:
        fails.append(
            # An unprofitable company excluded from a VALUE screen is correct;
            # filing it under a data-quality reason is not, and it hid 14 of
            # the 75 flagged names behind a label about tags.
            "too many loss-making years to rank on EV/EBIT"
            if f.ev_short_because_unprofitable
            else "EV/EBIT history too short or stale to rank against")

    # Value-trap filter — cheap for a reason.
    #
    # Declining ROIC alone is too blunt. Microsoft went 33.0 -> 27.1 over five
    # years while EBIT grew 17%; equity simply compounded faster. That is
    # mean reversion from exceptional to excellent, not erosion, and the gate
    # was disqualifying it. Now requires a decline that is material AND lands
    # somewhere that actually matters.
    if (f.roic_declining_years is not None and f.roic_declining_years >= 3
            and f.roic_ttm is not None and f.roic_5y is not None and (
            f.roic_ttm < max(15.0, f.wacc * 1.5) or
            (f.roic_5y > 0 and (f.roic_5y - f.roic_ttm) / f.roic_5y > 0.30))):
        fails.append(
            f"ROIC fell {f.roic_declining_years}y to {f.roic_ttm:.1f}% — eroding")
    if not f.gross_profit_unavailable and _below(f.gross_margin_delta_3y, -4):
        fails.append(f"gross margin down {abs(f.gross_margin_delta_3y):.1f}pts over 3y")
    if f.revenue_growth_ttm < 0 and f.revenue_cagr_5y < 4:
        fails.append("revenue declining")
    return fails


def score_quality_value(f: Fundamentals) -> dict:
    # The central edge: what growth does the current price require, versus
    # what the business has actually delivered? The gap is the opportunity.
    # A gap computed against an unpopulated implied growth is a gap against
    # zero — which is what produced +16.5 on every row. Score it neutral and
    # let the caveat say why, rather than rank on a placeholder.
    expectations_gap = (0.0 if f.reverse_dcf_unavailable
                        else f.revenue_cagr_5y - f.reverse_dcf_implied_growth)

    comp = {
        # OMIT an unavailable component rather than substituting for it.
        # Substituting a neutral 50 was worse than it sounds: _scale maps a 35%
        # margin to 0 and 85% to 100, so a real 55% margin scores 40 — and a
        # neutral 50 made MISSING data score better than a good margin. Absent
        # data should neither help nor hurt, which means averaging over the
        # components that exist.
        "quality": _clamp(_mean_available(
            [_s(f.roic_5y, 12, 35),
             _s(_d(f.roic_5y, f.wacc), 0, 20),
             _s(f.fcf_margin, 8, 35),
             _scale(-f.share_count_cagr_5y, -1, 8)]
            + ([] if f.gross_profit_unavailable
               else [_s(f.gross_margin, 35, 85)]))),

        # A purely relative measure cannot tell "cheap" from "less expensive
        # than it has ever been". A decade-expensive name sitting at its own
        # 3rd percentile scored 100 — MercadoLibre at 31.8x, Autodesk at 29.1x
        # and Intuit at 19.1x all did. Blending the percentile with an absolute
        # multiple keeps the own-history signal while refusing to call 30x
        # cheap: MELI moves 98.7 -> 49.3, ADSK 100 -> 50.0, INTU 100 -> 67.4,
        # and a genuine 9x name scores 87.1.
        "discount_to_own_history": _clamp(_mean_available([
            None if f.ev_ebit_percentile_10y is None
            else _scale(100 - f.ev_ebit_percentile_10y, 20, 95),
            None if f.ev_ebit is None else _scale(30 - f.ev_ebit, 5, 22),
        ])),

        "reverse_dcf_gap": (50.0 if f.reverse_dcf_unavailable
                            else _scale(expectations_gap, -2, 14)),

        "fcf_yield": _scale(f.fcf_yield, 2.5, 9),

        "fundamental_momentum": _clamp(_mean_available([
            _scale(f.eps_revision_3m, -12, 8),
            _s(_d(f.roic_ttm, f.roic_5y), -5, 5),
            _scale(f.revenue_growth_ttm, -3, 20),
        ])),
    }

    weights = {"quality": 0.30, "discount_to_own_history": 0.24,
               "reverse_dcf_gap": 0.22, "fcf_yield": 0.12,
               "fundamental_momentum": 0.12}

    fair_value_upside = (
        (f.ev_ebit_median_10y / f.ev_ebit - 1) * 100
        if (f.ev_ebit and f.ev_ebit_median_10y and f.ev_ebit > 0) else None
    )

    return {
        "symbol": f.symbol, "name": f.name,
        "score": round(sum(comp[k] * w for k, w in weights.items())),
        "expectations_gap": round(expectations_gap, 1),
        # None all the way out. Rounding it here would resurrect the zero the
        # builder went to trouble to void.
        "upside_to_own_median": (None if fair_value_upside is None
                                 else round(fair_value_upside)),
        "gates_failed": quality_gates(f),
        "components": {k: round(v) for k, v in comp.items()},
    }


# =====================================================================

def run_screen(universe: list[Fundamentals], engine: str,
               strict: bool = True, min_score: int = 60) -> list[dict]:
    """
    engine: "dividend" | "recovery" | "quality"
    strict: drop gate failures entirely. Set False to keep them visible
            with their failure reasons attached, which is useful when
            tuning thresholds.
    """
    scorer = {
        "dividend": score_dividend,
        "quality": score_quality_value,
        "recovery": lambda f: score_recovery(f, 40.0, 1.2, 1.6),
    }[engine]

    results = [scorer(f) for f in universe]
    gated = [r for r in results if r["gates_failed"]]
    passed = [r for r in results if not r["gates_failed"]]

    # A name that clears every gate and is then dropped by the score threshold
    # is a completely different outcome from one that failed a gate, and the
    # two were indistinguishable. Microsoft passed every dividend gate and
    # scored 51 against a default threshold of 60 — the screen reported zero
    # rows with no explanation.
    below = [r for r in passed if r["score"] < min_score]
    if below:
        top = sorted(below, key=lambda r: r["score"], reverse=True)[:5]
        print(f"  [info] {len(below)} name(s) cleared all gates but scored under "
              f"{min_score}: " + ", ".join(f"{r['symbol']} {r['score']}" for r in top))
    for r in below:
        r["near_miss"] = True

    out = passed if not strict else passed
    out = [r for r in out if r["score"] >= min_score]
    if not strict:
        out = sorted(gated + passed, key=lambda r: r["score"], reverse=True)
    return sorted(out, key=lambda r: r["score"], reverse=True)
