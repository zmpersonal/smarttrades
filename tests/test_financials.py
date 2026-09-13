"""
Financials screen — contract and gate tests.

Written BEFORE the screen, so the dashboard contract breaks at test time rather
than on screen. The value screen taught this: `run_screen` emitted
{symbol, score, components} while index.html asked for {ticker, roic, fcfy...},
and wiring them straight through would have rendered correctly-ranked blanks.
"""

import re
from pathlib import Path

import pytest

from engines import screeners as sc
from engines import dashboard_adapter as da
from engines import fundamentals_builder as fb


def _populated() -> sc.Fundamentals:
    """A financial with every input present — the contract's happy path."""
    f = sc.Fundamentals(symbol="JPM", name="JPMorgan Chase", sector="financial")
    f.roe_5y, f.roe_ttm = 17.0, 18.0
    f.rotce = 21.0
    f.equity_to_assets = 8.4
    f.tangible_book = 280e9
    f.price_to_tangible_book = 2.1
    f.ptbv_percentile_10y = 72.0
    f.tbvps_cagr_5y = 9.0
    f.cost_of_equity = 10.5
    f.revenue_cagr_5y = 7.0
    f.share_count_cagr_5y = -1.5
    f.dividend_yield = 2.2
    f.buyback_yield = 1.5
    f.financial_subtype = "depository"
    f.financial_in_scope = True
    return f


# ----------------------------------------------------------- contract

def test_financial_dashboard_contract_is_satisfied():
    html = (Path(sc.__file__).parent.parent / "index.html").read_text()
    assert "financial:{" in html, "index.html has no financial engine block"

    f = _populated()
    res = sc.score_financial(f)
    row = da.to_rows("financial", [(f, res)])[0]

    block = html[html.index("financial:{"):]
    block = block[:block.index("comps:[")]
    declared = re.findall(r'\{k:"(\w+)"', block)
    assert declared, "no columns parsed for the financial block"

    missing = [k for k in declared if k not in row]
    assert not missing, f"financial would render blanks for {missing}"

    for key in ("ticker", "name", "score", "comp", "note", "facts"):
        assert key in row, f"financial row missing {key}"
    assert len(row["comp"]) == 5, "financial component count mismatch"


def test_financial_comps_labels_match_component_order():
    html = (Path(sc.__file__).parent.parent / "index.html").read_text()
    block = html[html.index("financial:{"):]
    labels = re.findall(r'"([^"]+)"', block[block.index("comps:["):block.index("]", block.index("comps:["))])
    assert len(labels) == len(da.COMPONENT_ORDER["financial"]) == 5


# --------------------------------------------------------------- gates

def test_unavailable_roe_fails_rather_than_defaulting():
    f = _populated()
    f.roe_5y = f.roe_ttm = None
    f.roe_unavailable = True
    assert any("ROE unavailable" in g for g in sc.financial_gates(f))


def test_gates_do_not_fire_on_unknown_values():
    """A None must not read as a business failure — _below/_above discipline."""
    f = _populated()
    f.equity_to_assets = None
    f.rotce = None
    fails = sc.financial_gates(f)
    assert not any("equity/assets" in g for g in fails), fails


def test_thin_capital_is_disqualifying():
    f = _populated()
    f.equity_to_assets = 3.0
    assert any("equity/assets" in g for g in sc.financial_gates(f))


def test_roe_below_cost_of_equity_is_disqualifying():
    f = _populated()
    f.roe_5y, f.cost_of_equity = 8.0, 10.5
    assert any("cost of equity" in g for g in sc.financial_gates(f))


def test_negative_tangible_book_is_disqualifying():
    f = _populated()
    f.tangible_book = -1e9
    assert any("tangible book" in g for g in sc.financial_gates(f))


def test_roe_decline_needs_a_floor_not_just_a_direction():
    """Mean reversion from exceptional to excellent is not erosion — the
    lesson Microsoft's 27% ROIC taught the quality screen."""
    f = _populated()
    f.roe_declining_years = 3
    f.roe_5y = 22.0
    assert not any("eroding" in g for g in sc.financial_gates(f))
    f.roe_5y = 10.5
    assert any("eroding" in g for g in sc.financial_gates(f))


# --------------------------------------------------------------- scoring

def test_missing_input_is_omitted_not_neutral_substituted():
    """
    _scale(None) is 50.0, which let missing data outscore a real reading.
    A component with a missing term must average over what survives.
    """
    full = _populated()
    without = _populated()
    without.rotce = None
    without.rotce_unavailable = True
    a = sc.score_financial(full)["components"]["returns"]
    b = sc.score_financial(without)["components"]["returns"]
    assert b != 50.0, "missing ROTCE fell back to a neutral 50"
    assert abs(a - b) < 40, "dropping one term should not swing the component wildly"


def test_score_is_bounded_and_components_present():
    res = sc.score_financial(_populated())
    assert 0 <= res["score"] <= 100
    assert set(res["components"]) == set(da.COMPONENT_ORDER["financial"])
    assert "gates_failed" in res


# ------------------------------------------- within-bucket capital strength

def _bank(ea):
    f = _populated()
    f.financial_subtype, f.equity_to_assets = "depository", ea
    return f


def _insurer(ea):
    f = _populated()
    f.financial_subtype, f.equity_to_assets = "insurer", ea
    f.cost_of_equity = 9.5
    return f


def test_capital_strength_is_scored_within_sub_bucket():
    """
    Banks live at 7-11% equity/assets and insurers at 15-30%. On a shared
    6-to-16 scale a bank's defining leverage read as weakness and capped it
    near 45 — the ROIC/utilities failure one level down.
    """
    banks = [_bank(x) for x in (7.0, 8.0, 9.0, 10.0, 11.0)]
    insurers = [_insurer(x) for x in (15.0, 20.0, 25.0, 30.0, 35.0)]
    dist = sc.financial_distributions(banks + insurers)

    best_bank = sc.score_financial(_bank(11.0), dist)["components"]["capital_strength"]
    worst_ins = sc.score_financial(_insurer(15.0), dist)["components"]["capital_strength"]
    assert best_bank > worst_ins, (
        f"best bank {best_bank} still below weakest insurer {worst_ins}")


def test_capital_strength_term_drops_out_without_a_distribution():
    """A silent fallback to a shared scale is how the bias would return."""
    f = _bank(8.0)
    with_dist = sc.score_financial(f, sc.financial_distributions([_bank(x) for x in
                                   (7, 8, 9, 10, 11)]))["components"]["capital_strength"]
    without = sc.score_financial(f)["components"]["capital_strength"]
    assert with_dist != without


def test_fee_based_bucket_exists_and_needs_no_deposits_or_premiums():
    from engines import fundamentals_builder as fb
    assert 6200 in fb.FINANCIAL_SIC["fee_based"]
    assert 6411 in fb.FINANCIAL_SIC["fee_based"]
    assert 6200 not in fb.FINANCIAL_SIC["broker"]
    assert 6411 not in fb.FINANCIAL_SIC["insurer"]
    w = {"revenue": True, "deposits": False, "interest_income": False,
         "premiums_earned": False}
    assert fb.FINANCIAL_WITNESS["fee_based"](w)


# ------------------------------------------------- absolute valuation anchor

def _quality(ev, pct):
    f = sc.Fundamentals(symbol="X", name="X")
    f.ev_ebit, f.ev_ebit_percentile_10y = ev, pct
    f.roic_5y, f.wacc, f.gross_margin = 25.0, 9.0, 60.0
    f.fcf_margin, f.revenue_cagr_5y = 25.0, 12.0
    f.fcf_positive_years_of_10, f.fcf_history_years = 10, 10
    f.share_count_cagr_5y = -1.0
    return f


def test_relative_cheapness_alone_cannot_score_full_marks():
    """
    A decade-expensive name at its own 3rd percentile scored 100. Relative
    cheapness cannot distinguish "cheap" from "less expensive than it has
    ever been", so an absolute multiple is blended in.
    """
    expensive = sc.score_quality_value(_quality(31.8, 6))["components"]["discount_to_own_history"]
    genuine = sc.score_quality_value(_quality(9.0, 10))["components"]["discount_to_own_history"]
    assert expensive < 60, f"31.8x still scores {expensive}"
    assert genuine > 85, f"9x only scores {genuine}"
    assert genuine - expensive > 30


def test_absent_multiple_is_omitted_not_substituted():
    f = _quality(None, 5)
    c = sc.score_quality_value(f)["components"]["discount_to_own_history"]
    assert c > 80, "a voided ev_ebit should leave the percentile term intact"


def test_recovery_valuation_gap_has_the_same_anchor():
    import inspect
    src = inspect.getsource(sc.score_recovery)
    assert "_mean_available" in src and "ev_ebit" in src


# ----------------------------------------------------- per-sub-bucket floors

def test_gate_floors_differ_by_sub_bucket():
    """6% equity/assets is survivable for a bank and alarming for an insurer."""
    bank, ins = _populated(), _populated()
    bank.financial_subtype, bank.equity_to_assets = "depository", 7.0
    ins.financial_subtype, ins.equity_to_assets, ins.cost_of_equity = "insurer", 7.0, 9.5
    assert not any("equity/assets" in g for g in sc.financial_gates(bank))
    assert any("equity/assets" in g for g in sc.financial_gates(ins))


def test_fee_business_needs_a_higher_roe_than_a_bank():
    bank, fee = _populated(), _populated()
    bank.financial_subtype, bank.roe_5y = "depository", 11.0
    fee.financial_subtype, fee.roe_5y, fee.cost_of_equity = "fee_based", 11.0, 10.0
    assert not any("under" in g and "ROE" in g for g in sc.financial_gates(bank))
    assert any("ROE" in g and "under 12%" in g for g in sc.financial_gates(fee))
