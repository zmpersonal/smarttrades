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


# --------------------------------------------------- CI failures, first run

def test_universe_loader_receives_tickers_not_dict_keys(tmp_path, monkeypatch):
    """
    `json.loads(universe.json)` was passed straight to the builder, but the
    file is a metadata document — so the loop iterated its KEYS and tried to
    resolve "generated_at", "ranking", "source" and "floors" as tickers. All
    three screeners reported "0 of 0" and nothing in the log said why.
    """
    import json
    import run_all

    seen = {}
    data = tmp_path / "data"
    data.mkdir()
    (data / "universe.json").write_text(json.dumps({
        "generated_at": "2026-09-13T00:00:00", "ranking": "dollar ADV",
        "source": "FINRA/SEC/yfinance", "floors": {"dollar_adv": 1e7},
        "count": 3, "tickers": ["JPM", "KO", "MSFT"],
    }))
    monkeypatch.setattr(run_all, "DATA", data)

    from engines import fundamentals_builder as fbuild
    monkeypatch.setattr(fbuild, "load_fundamentals",
                        lambda t, **k: seen.setdefault("got", list(t)) or [])
    run_all.load_fundamentals()

    assert seen["got"] == ["JPM", "KO", "MSFT"], seen["got"]
    assert not any(k in seen["got"] for k in
                   ("generated_at", "ranking", "source", "floors", "count"))


def test_bare_list_universe_still_works(tmp_path, monkeypatch):
    import json
    import run_all
    from engines import fundamentals_builder as fbuild

    seen = {}
    data = tmp_path / "data"
    data.mkdir()
    (data / "universe.json").write_text(json.dumps(["AAPL", "MSFT"]))
    monkeypatch.setattr(run_all, "DATA", data)
    monkeypatch.setattr(fbuild, "load_fundamentals",
                        lambda t, **k: seen.setdefault("got", list(t)) or [])
    run_all.load_fundamentals()
    assert seen["got"] == ["AAPL", "MSFT"]


def test_non_string_tickers_raise_rather_than_screen_nothing(tmp_path, monkeypatch):
    import json
    import pytest as _pytest
    import run_all

    data = tmp_path / "data"
    data.mkdir()
    (data / "universe.json").write_text(json.dumps({"tickers": [{"symbol": "JPM"}]}))
    monkeypatch.setattr(run_all, "DATA", data)
    with _pytest.raises(NotImplementedError):
        run_all.load_fundamentals()


def test_per_item_warnings_collapse_to_one_line(capsys):
    """~12,000 identical lines buried the one genuine signal in the CI log."""
    from engines import free_sources as fs

    fs._warn_collapsed("tape", {f"S{i}": "pip install yfinance" for i in range(12000)}, 12000)
    out = capsys.readouterr().out
    assert out.count("\n") == 1, f"emitted {out.count(chr(10))} lines"
    assert "12000/12000" in out

    fs._warn_collapsed("prices", {"A": "404", "B": "404", "C": "timeout"}, 50)
    out = capsys.readouterr().out
    assert out.count("\n") == 2, "one line per distinct failure mode"

    fs._warn_collapsed("quiet", {}, 10)
    assert capsys.readouterr().out == ""


def test_deploy_publishes_even_when_an_engine_fails():
    """
    One engine failing must not block the site. The run job still goes red;
    the dashboard still publishes whatever wrote — the same reasoning that
    isolates each engine in its own try/except.
    """
    from pathlib import Path
    wf = (Path(__file__).parent.parent / ".github/workflows/daily.yml").read_text()
    dep = wf[wf.index("deploy:"):]
    cond = dep[dep.index("if:"):dep.index("\n", dep.index("if:"))]
    assert "always()" in cond, cond


def test_yfinance_is_a_declared_dependency():
    """equity_ohlcv resolves to yfinance; commented out, the runner had no
    price provider and darkpool crashed with 'no tape data for any symbol'."""
    from pathlib import Path
    req = (Path(__file__).parent.parent / "requirements.txt").read_text()
    live = [l.strip() for l in req.splitlines()
            if l.strip() and not l.strip().startswith("#")]
    assert any(l.startswith("yfinance") for l in live), live


# ------------------------------------------------ dry run must not mutate

def _fresh_state(tmp_path, monkeypatch):
    import alerts
    st = tmp_path / "alert_state.json"
    st.write_text('{"fired": {}, "last_digest": null, "last_top": {}}')
    monkeypatch.setattr(alerts, "STATE", st)
    monkeypatch.setattr(alerts, "WEBHOOK", "https://hooks.example/test")
    return alerts, st


def _transition():
    prev = {"entry": {"status": "GATED"}, "divergence": {}, "cycle": {}}
    cur = {"entry": {"status": "TRIGGERED", "regime_label": "bull",
                     "weekly_rsi": 54.2,
                     "daily": {"rsi": 38.4, "bands": {"oversold": 40},
                               "days_oversold": 2, "distance_to_oversold": -1.6,
                               "bullish_divergence": None},
                     "permission": {}, "trigger": {}},
           "divergence": {}, "cycle": {}}
    return cur, prev


def test_two_dry_runs_then_a_real_send_still_sends(tmp_path, monkeypatch, capsys):
    """
    post() returns True on a dry run, so both senders were recording alerts
    that were never sent: the preview suppressed the real alert, and a local
    preview diverged from the runner's committed state.
    """
    alerts, st = _fresh_state(tmp_path, monkeypatch)
    cur, prev = _transition()

    assert alerts.send_btc(cur, prev, dry_run=True) > 0, "first preview said nothing"
    assert alerts.send_btc(cur, prev, dry_run=True) > 0, "second preview went quiet"
    assert '"fired": {}' in st.read_text().replace("\n", "").replace(" ", "") or \
           alerts.load_state()["fired"] == {}, "a dry run wrote state"

    posted = []
    monkeypatch.setattr(alerts, "post", lambda p, dry_run=False: posted.append(p) or True)
    assert alerts.send_btc(cur, prev) > 0, "the real send never fired"
    assert posted, "nothing was posted for real"
    assert alerts.load_state()["fired"], "a real send failed to record state"


def test_dry_run_prints_the_dedup_decision(tmp_path, monkeypatch, capsys):
    alerts, _ = _fresh_state(tmp_path, monkeypatch)
    cur, prev = _transition()
    alerts.send_btc(cur, prev, dry_run=True)
    out = capsys.readouterr().out
    assert "would record" in out, out[-400:]
    assert "state NOT written" in out


def test_generic_sender_has_the_same_discipline(tmp_path, monkeypatch):
    alerts, _ = _fresh_state(tmp_path, monkeypatch)
    a = alerts.Alert(key="rec_dispersion", severity="warn",
                     title="Credit quality dispersion widening",
                     body="CCC-BB at 9.15pp, 100th percentile.")
    assert alerts.send_generic([a], dry_run=True) > 0
    assert alerts.send_generic([a], dry_run=True) > 0, "preview suppressed itself"
    assert alerts.load_state()["fired"] == {}
    monkeypatch.setattr(alerts, "post", lambda p, dry_run=False: True)
    assert alerts.send_generic([a]) > 0, "the real send was suppressed by a preview"


def test_digest_dry_run_does_not_consume_the_biweekly_slot(tmp_path, monkeypatch):
    """Writing last_digest on a preview would make the next real digest
    believe it had already run, and overwrite the new/dropped baseline."""
    alerts, _ = _fresh_state(tmp_path, monkeypatch)
    engines = {"value": {"rows": [{"ticker": "AXP", "score": 82}]}}
    assert alerts.send_digest(engines, dry_run=True)
    st = alerts.load_state()
    assert not st.get("last_digest"), "a preview consumed the digest slot"
    assert not st.get("last_top"), "a preview overwrote the delta baseline"
