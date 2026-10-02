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


# --------------------------------------------------------- workflow guards

def _workflow() -> str:
    from pathlib import Path
    return (Path(__file__).parent.parent / ".github/workflows/daily.yml").read_text()


def test_timeout_exceeds_the_measured_run():
    """
    30 minutes was set before the run was ever measured, and cancelled the
    first full run mid-flight. The weekly screen builds 1,500 records at
    33-36 minutes, so the ceiling has to clear that with headroom.
    """
    import re
    m = re.search(r"timeout-minutes:\s*(\d+)", _workflow())
    assert m, "no timeout-minutes in the workflow"
    mins = int(m.group(1))
    assert mins >= 72, f"{mins}m leaves under 2x headroom on a 36m run"
    assert mins <= 360, f"{mins}m exceeds GitHub's 6h ceiling"


def test_data_is_committed_even_when_the_engines_step_fails():
    """
    The deploy job's always() publishes whatever is on main. If the commit step
    is skipped on failure, main never receives the partial run, so always()
    republishes stale data and looks like it worked.
    """
    wf = _workflow()
    commit = wf[wf.index("Commit data and alert state"):]
    head = commit[:commit.index("run:")]
    assert "if: always()" in head, head


def test_a_timeout_still_notifies():
    """timeout-minutes does not reliably report as failure()."""
    wf = _workflow()
    notify = wf[wf.index("Notify Slack on failure"):]
    cond = notify[notify.index("if:"):notify.index("\n", notify.index("if:"))]
    assert "cancelled()" in cond, cond


def test_no_node16_era_action_versions():
    """Node 20 deprecation: checkout@v4 and setup-python@v5 warn on every run."""
    import re
    wf = _workflow()
    stale = {"actions/checkout": 4, "actions/setup-python": 5,
             "actions/upload-pages-artifact": 3, "actions/deploy-pages": 4,
             "actions/configure-pages": 5}
    for action, worst in stale.items():
        for found in re.findall(rf"{re.escape(action)}@v(\d+)", wf):
            assert int(found) > worst, f"{action}@v{found} is at or below v{worst}"


# ------------------------------------------ voided fields must not crash

def _all_voided():
    """A record with every field void_derived_fields can null, set to None."""
    import dataclasses, inspect, re
    from engines import fundamentals_builder as fb
    voidable = set(sc.voidable_fields())
    f = sc.Fundamentals(symbol="VOID", name="All voided")
    for v in voidable:
        setattr(f, v, None)
    return f, voidable


@pytest.mark.parametrize("name,fn", [
    ("score_dividend", lambda f: sc.score_dividend(f)),
    ("score_quality_value", lambda f: sc.score_quality_value(f)),
    ("score_recovery", lambda f: sc.score_recovery(f, 40.0, 1.2, 1.6)),
    ("dividend_gates", lambda f: sc.dividend_gates(f)),
    ("quality_gates", lambda f: sc.quality_gates(f)),
    ("recovery_gates", lambda f: sc.recovery_gates(f)),
])
def test_every_scorer_and_gate_survives_all_voided_fields(name, fn):
    """
    void_derived_fields sets unavailable derived fields to None. run_screen
    scores EVERY name before filtering, so `100 - f.fcf_payout` and
    `4 - f.net_debt_ebitda` crashed dividend and recovery on the first live
    run. Local verification scored only gate-clean names and never reached a
    voided record, which is why it passed while production failed.
    """
    f, voidable = _all_voided()
    assert voidable, "found no voidable fields to test"
    fn(f)


def test_run_screen_survives_a_voided_record_among_clean_ones():
    """The production path, not a pre-filtered subset."""
    f, _ = _all_voided()
    for w in ("dividend", "quality", "recovery"):
        sc.run_screen([f], w, strict=False, min_score=0)


# ------------------------------------------------------------ darkpool NaN

def _dp_panel(bad_field=None):
    import numpy as np, pandas as pd
    from engines import finra_darkpool as fd
    rows = []
    for sym in ("GOOD", "BAD"):
        rows.append(dict(symbol=sym, Date=pd.Timestamp("2026-09-11"),
                         dollar_adv=5e7, close=50.0, dpi_z=1.2, oe_share_z=0.4,
                         rvol_z=np.nan, compression=0.3, ret_20d=0.05,
                         dpi_5d=0.55, oe_share_5d=0.4, rvol=1.1))
    df = pd.DataFrame(rows)
    if bad_field:
        df.loc[df["symbol"] == "BAD", bad_field] = np.nan
    return df


@pytest.mark.parametrize("field", ["compression", "ret_20d"])
def test_darkpool_survives_a_symbol_with_uncomputable_raw_input(field, monkeypatch):
    """
    One symbol with a NaN compression or 20d return raised
    "cannot convert float NaN to integer" and took down the whole board.
    """
    from engines import finra_darkpool as fd
    panel = _dp_panel(field)
    monkeypatch.setattr(fd, "build_panel", lambda finra, tape: panel)
    monkeypatch.setattr(fd, "add_zscores", lambda df, cfg: df)
    cfg = fd.Config()
    cfg.min_score = 0
    board = fd.run(None, None, cfg=cfg)
    assert "GOOD" in set(board["symbol"]), "a clean symbol was lost"
    assert "BAD" not in set(board["symbol"]), "an uncomputable symbol was scored"


def test_nan_zscore_inputs_are_still_safe_via_squash():
    """rvol_z and oe_share_z route through _squash; their NaN must not crash."""
    import numpy as np, pandas as pd
    from engines import finra_darkpool as fd
    r = pd.Series(dict(symbol="X", dpi_z=1.2, oe_share_z=np.nan, rvol_z=np.nan,
                       compression=0.3, ret_20d=0.05, dpi_5d=0.55,
                       oe_share_5d=0.4, rvol=1.1, short_interest_pct=0.0))
    out = fd.score_symbol(r, fd.Config())
    assert np.isfinite(out["score"])


# ------------------------------------------------- bank-format total revenue

def _ann(tag, vals, filed_lag=60):
    """{year: value} -> an annual USD concept as EDGAR serves it."""
    return {tag: {"units": {"USD": [
        {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": v, "fy": y,
         "fp": "FY", "form": "10-K", "filed": f"{y + 1}-02-{min(28, filed_lag % 28 + 1):02d}"}
        for y, v in vals.items()]}}}


def _gaap(*concepts):
    ug = {}
    for c in concepts:
        ug.update(c)
    return {"facts": {"us-gaap": ug}}


YEARS = range(2018, 2026)


def test_bank_fee_slice_is_replaced_by_total_net_revenue():
    """
    Huntington's chain revenue was RevenueFromContractWithCustomer — the ASC 606
    fee slice, $1.56bn — against a total of $8.17bn. Interest income is outside
    ASC 606 scope, so a bank's contract revenue can never be its total.
    """
    from engines import free_sources as fs
    facts = _gaap(
        _ann("RevenueFromContractWithCustomerExcludingAssessedTax", {y: 1.5e9 for y in YEARS}),
        _ann("InterestIncomeExpenseNet", {y: 6.0e9 for y in YEARS}),
        _ann("NoninterestIncome", {y: 2.0e9 for y in YEARS}))
    df = fs.extract_series(facts, "revenue")
    assert list(df["val"]) == [8.0e9] * len(YEARS)
    assert df.attrs["revenue_basis"] == "bank_format_total"
    assert df.attrs["revenue_chain_replaced"]["chain_latest"] == 1.5e9


def test_bank_with_no_chain_revenue_becomes_buildable():
    """Goldman tags only RevenuesNetOfInterestExpense and never became a record."""
    from engines import free_sources as fs
    facts = _gaap(_ann("RevenuesNetOfInterestExpense", {y: 5.0e10 for y in YEARS}))
    df = fs.extract_series(facts, "revenue")
    assert len(df) == len(YEARS) and df["val"].iloc[-1] == 5.0e10
    assert df.attrs["revenue_basis"] == "bank_format_only"


def test_tagged_total_wins_over_derived_parts_on_the_same_period():
    from engines import free_sources as fs
    facts = _gaap(
        _ann("RevenuesNetOfInterestExpense", {y: 9.0e9 for y in YEARS}),
        _ann("InterestIncomeExpenseNet", {y: 6.0e9 for y in YEARS}),
        _ann("NoninterestIncome", {y: 2.0e9 for y in YEARS}))
    df = fs.extract_series(facts, "revenue")
    assert set(df["val"]) == {9.0e9}


def test_parts_sum_on_intersection_never_a_zero_filled_union():
    """A year with NII and no noninterest income must not publish NII as revenue."""
    from engines import free_sources as fs
    facts = _gaap(
        _ann("InterestIncomeExpenseNet", {y: 6.0e9 for y in YEARS}),
        _ann("NoninterestIncome", {y: 2.0e9 for y in YEARS if y != 2021}))
    df = fs.extract_series(facts, "revenue")
    assert 2021 not in set(df["end"].dt.year)
    assert set(df["val"]) == {8.0e9}


@pytest.mark.parametrize("chain_tag,chain_val", [
    ("Revenues", 8.0e9),       # JPM/BAC: the chain already holds the total
    ("Revenues", 1.3e11),      # StoneX: gross revenue far ABOVE the net figure
])
def test_chain_at_or_above_the_bank_total_is_left_alone(chain_tag, chain_val):
    from engines import free_sources as fs
    facts = _gaap(
        _ann(chain_tag, {y: chain_val for y in YEARS}),
        _ann("RevenuesNetOfInterestExpense", {y: 8.0e9 for y in YEARS}),
        _ann("InterestIncomeExpenseNet", {y: 6.0e9 for y in YEARS}),
        _ann("NoninterestIncome", {y: 2.0e9 for y in YEARS}))
    df = fs.extract_series(facts, "revenue")
    assert set(df["val"]) == {chain_val}
    assert "revenue_basis" not in df.attrs


def test_non_bank_revenue_is_untouched():
    """A biotech's interest income on cash must never become revenue."""
    from engines import free_sources as fs
    facts = _gaap(
        _ann("RevenueFromContractWithCustomerExcludingAssessedTax", {y: 3.0e9 for y in YEARS}),
        _ann("InterestIncomeExpenseNet", {y: 4.0e7 for y in YEARS}))
    df = fs.extract_series(facts, "revenue")
    assert set(df["val"]) == {3.0e9} and "revenue_basis" not in df.attrs
    empty = fs.extract_series(_gaap(_ann("InterestIncomeExpenseNet", {y: 4.0e7 for y in YEARS})),
                              "revenue")
    assert empty.empty


def test_stale_chain_is_replaced_by_a_current_bank_total():
    """Regions' fee slice ended 2021 while its income statement runs to 2025."""
    from engines import free_sources as fs
    facts = _gaap(
        _ann("RevenueFromContractWithCustomerIncludingAssessedTax", {y: 1.0e8 for y in range(2018, 2022)}),
        _ann("InterestIncomeExpenseNet", {y: 4.0e9 for y in YEARS}),
        _ann("NoninterestIncome", {y: 2.5e9 for y in YEARS}))
    df = fs.extract_series(facts, "revenue")
    assert df["end"].max().year == 2025 and set(df["val"]) == {6.5e9}


def test_bank_revenue_respects_point_in_time():
    from engines import free_sources as fs
    from datetime import date
    facts = _gaap(_ann("RevenuesNetOfInterestExpense", {y: 5.0e10 for y in YEARS}))
    df = fs.extract_series(facts, "revenue", as_of=date(2023, 1, 15))
    assert df["end"].max().year == 2021


def test_a_stale_bank_total_never_replaces_a_current_chain():
    """T. Rowe Price: bank-format tags end 2015, chain revenue runs to 2025."""
    from engines import free_sources as fs
    facts = _gaap(
        _ann("Revenues", {y: 7.0e9 for y in YEARS}),
        _ann("InterestIncomeExpenseNet", {y: 1.0e8 for y in range(2010, 2016)}),
        _ann("NoninterestIncome", {y: 4.1e9 for y in range(2010, 2016)}))
    df = fs.extract_series(facts, "revenue")
    assert df["end"].max().year == 2025 and set(df["val"]) == {7.0e9}
    assert "revenue_basis" not in df.attrs


def test_universe_is_built_once_per_process_and_rebuilt_when_it_changes(tmp_path, monkeypatch):
    """Four weekly screeners, one 35-minute build — not four."""
    import json
    import run_all
    from engines import fundamentals_builder as fbuild

    calls = []
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(run_all, "DATA", data)
    monkeypatch.setattr(fbuild, "load_fundamentals",
                        lambda t, **k: calls.append(list(t)) or [])
    run_all._build_universe.cache_clear()
    (data / "universe.json").write_text(json.dumps({"tickers": ["ZZA", "ZZB"]}))
    for _ in range(4):
        run_all.load_fundamentals()
    assert calls == [["ZZA", "ZZB"]]
    (data / "universe.json").write_text(json.dumps({"tickers": ["ZZC"]}))
    run_all.load_fundamentals()
    assert calls[-1] == ["ZZC"]
    run_all._build_universe.cache_clear()


def test_financial_engine_is_wired_end_to_end():
    import run_all
    assert "financial" in run_all.ENGINES and "financial" in run_all.RUNNERS
    assert run_all.CADENCE["financial"] == "weekly"
    assert "financial" in run_all.MIN_SCORE
    html = (run_all.ROOT / "index.html").read_text() if hasattr(run_all, "ROOT") \
        else (run_all.DATA.parent / "index.html").read_text()
    assert '"financial"' in html.split("const ORDER=")[1].split(";")[0]


# ------------------------------------------------------------ taxonomy choice

def test_one_stray_us_gaap_concept_does_not_make_an_ifrs_filer_american():
    """BTI: 372 ifrs-full concepts, one us-gaap concept, read as us-gaap."""
    from engines import free_sources as fs
    facts = {"facts": {
        "us-gaap": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2025-12-31", "val": 1, "form": "20-F", "filed": "2026-03-01"}]}}},
        "ifrs-full": _ann("Revenue", {y: 3.0e10 for y in YEARS})}}
    assert fs.taxonomy_of(facts) == "ifrs-full"
    assert fs.extract_series(facts, "revenue")["val"].iloc[-1] == 3.0e10


def test_a_stale_us_gaap_history_loses_to_a_current_ifrs_one():
    """Itau: 339 us-gaap concepts ending 2010 beside 336 ifrs-full to today."""
    from engines import free_sources as fs
    old = _ann("Revenues", {y: 1.0e10 for y in range(2005, 2011)})
    old.update(_ann("NetIncomeLoss", {y: 1.0e9 for y in range(2005, 2011)}))
    facts = {"facts": {"us-gaap": old,
                       "ifrs-full": _ann("Revenue", {y: 3.0e10 for y in YEARS})}}
    assert fs.taxonomy_of(facts) == "ifrs-full"


def test_us_filer_with_no_ifrs_is_unchanged():
    from engines import free_sources as fs
    assert fs.taxonomy_of(_gaap(_ann("Revenues", {y: 1.0 for y in YEARS}))) == "us-gaap"
    assert fs.taxonomy_of({"facts": {}}) == "unknown"


# ------------------------------------------------------- statement currency

def test_non_usd_statements_are_gated_and_price_ratios_voided():
    """Telus reports in CAD and trades in USD; valuation_gap scored 100."""
    from engines import screeners as sc
    from engines import fundamentals_builder as fb
    f = sc.Fundamentals(symbol="TU", name="Telus")
    f.statement_currency = "CAD"
    f.ev_ebit, f.fcf_yield, f.altman_z = 9.0, 7.5, 2.4
    f.price_to_tangible_book = 1.1
    voided = fb.void_derived_fields(f)
    assert {"ev_ebit", "fcf_yield", "altman_z", "price_to_tangible_book"} <= set(voided)
    assert f.ev_ebit is None and f.fcf_yield is None
    assert any("CAD" in g for g in sc.data_quality_gates(f))
    f.sector, f.financial_in_scope = "financial", True
    assert any("CAD" in g for g in sc.financial_gates(f))
    for scorer in (sc.score_dividend, sc.score_quality_value,
                   lambda x: sc.score_recovery(x, 40.0, 1.2, 1.6)):
        assert scorer(f)["gates_failed"]


def test_builder_records_the_statement_currency():
    from engines import fundamentals_builder as fb
    facts = {"facts": {"ifrs-full": {
        **{k: {"units": {"CAD": v["units"]["USD"]}} for k, v in
           _ann("Revenue", {y: 2.0e10 for y in YEARS}).items()},
        **{k: {"units": {"CAD": v["units"]["USD"]}} for k, v in
           _ann("ProfitLoss", {y: 1.0e9 for y in YEARS}).items()}}}}
    f = fb.build("TU", facts)
    assert f.statement_currency == "CAD"
    usd = fb.build("X", _gaap(_ann("Revenues", {y: 2.0e10 for y in YEARS}),
                              _ann("NetIncomeLoss", {y: 1.0e9 for y in YEARS})))
    assert usd.statement_currency == "USD"


# ------------------------------------------------ status.json across runs

def _run_main(monkeypatch, data, argv, runners):
    import sys
    import run_all
    monkeypatch.setattr(run_all, "DATA", data)
    monkeypatch.setattr(run_all, "RUNNERS", {**run_all.RUNNERS, **runners})
    monkeypatch.setattr(run_all, "write", lambda name, payload: None)
    monkeypatch.setattr(sys, "argv", ["run_all.py", *argv])
    run_all.main()
    import json
    return json.loads((data / "status.json").read_text())["engines"]


def test_single_engine_run_keeps_every_other_engines_status(tmp_path, monkeypatch):
    """`--only recession` wiped six tabs' status and the site fell back to sample rows."""
    import json
    data = tmp_path / "data"; data.mkdir()
    (data / "status.json").write_text(json.dumps({"updated_at": "2026-09-13T04:30:00+00:00",
        "engines": {"dividend": {"state": "ok", "at": "2026-09-13T04:30:00+00:00"},
                    "value": {"state": "error", "detail": "boom", "at": "2026-09-13T04:30:00+00:00"}}}))
    eng = _run_main(monkeypatch, data, ["--only", "recession"], {"recession": lambda: {}})
    assert eng["recession"]["state"] == "ok"
    assert eng["dividend"] == {"state": "ok", "at": "2026-09-13T04:30:00+00:00", "carried": True}
    assert eng["value"]["state"] == "error" and eng["value"]["carried"] is True


def test_weekday_skip_does_not_overwrite_the_last_real_outcome(tmp_path, monkeypatch):
    """Every non-Sunday run recorded 'skipped' over Sunday's 'ok'."""
    import json
    import run_all
    data = tmp_path / "data"; data.mkdir()
    (data / "status.json").write_text(json.dumps({"engines": {
        "dividend": {"state": "ok", "at": "2026-09-13T14:00:00+00:00"}}}))
    monkeypatch.setattr(run_all, "should_run", lambda name, force: name == "recession")
    stubs = {k: (lambda: {}) for k in run_all.ENGINES}
    eng = _run_main(monkeypatch, data, [], stubs)
    assert eng["dividend"] == {"state": "ok", "at": "2026-09-13T14:00:00+00:00", "carried": True}
    assert eng["value"]["state"] == "skipped"          # never ran: honest
    assert eng["recession"]["state"] == "ok"


def test_failure_entries_carry_their_own_timestamp(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    def boom(): raise TypeError("int - NoneType")
    eng = _run_main(monkeypatch, data, ["--only", "dividend"], {"dividend": boom})
    assert eng["dividend"]["state"] == "error" and eng["dividend"]["at"]


def test_dashboard_never_falls_back_to_sample_rows_silently():
    """The live boot path must not assign embedded rows to a tab whose run failed."""
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "index.html").read_text()
    boot = html.split("async function loadEngine")[1].split("function srcBanner")[0]
    assert 'e.rows=[]' in boot, "table tabs must start empty in live mode"
    assert "SAMPLE — NOT REAL" in html and "samplerow" in html
    assert "Running on sample data" not in html, "static rail claim replaced by per-tab state"
    for k in ("dividend", "recovery", "darkpool", "value", "financial"):
        assert f'{k}:' in html.split("const CADENCE_H=")[1].split(";")[0]


def test_engine_files_are_strict_json_a_browser_can_parse():
    """14 bare NaN tokens in bitcoin.json put every live tab on sample data."""
    import json
    import numpy as np
    import run_all
    out = run_all.dump_json({"rsi": [float("nan"), 51.2, np.float64("inf")],
                             "n": np.int64(3), "ok": np.bool_(True), "s": "NaN in a string"})
    assert "NaN," not in out and "Infinity" not in out
    back = json.loads(out)
    assert back["rsi"] == [None, 51.2, None] and back["n"] == 3 and back["ok"] is True


# ------------------------------------------- exhaustive None, by contract
#
# Two sessions of the same crash, fixed site by site: `100 - f.fcf_payout` in
# the dividend scorer, then `int(round(f.fcf_payout))` in the adapter one
# layer out. The test below does not sample. It takes the contract —
# every field annotated `| None` — and runs every consumer of a Fundamentals
# against all of them None at once AND each one None alone on records that
# are otherwise populated, across every sector and sub-bucket, with values on
# both sides of the gate thresholds so the failure-message branches that
# format a field are reached too.

import dataclasses as _dc
import itertools as _it


def _contract_record(sector, subtype, bad):
    """Every numeric field set, either comfortably passing or clearly failing."""
    f = sc.Fundamentals(symbol=f"{sector[:3].upper()}{subtype[:3]}", name="contract")
    f.sector, f.financial_subtype, f.financial_in_scope = sector, subtype, sector == "financial"
    for x in _dc.fields(sc.Fundamentals):
        t = str(x.type)
        if "bool" in t or x.name in ("symbol", "name", "sector", "financial_subtype"):
            continue
        if "int" in t:
            setattr(f, x.name, 1 if bad else 12)
        elif "float" in t:
            setattr(f, x.name, -40.0 if bad else 22.0)
    f.drawdown_from_ath = -60.0
    f.statement_currency = "USD"
    return f


_CONTRACT_BASES = [_contract_record(sec, sub, bad)
                   for (sec, sub), bad in _it.product(
                       [("general", ""), ("utility", ""), ("reit", ""),
                        ("financial", "depository"), ("financial", "insurer"),
                        ("financial", "broker"), ("financial", "manager"),
                        ("financial", "fee_based")],
                       (False, True))]


def _consumers():
    from engines import dashboard_adapter as da
    dist = sc.financial_distributions(_CONTRACT_BASES * 3)
    return {
        "score_dividend": sc.score_dividend,
        "score_quality_value": sc.score_quality_value,
        "score_recovery": lambda f: sc.score_recovery(f, 40.0, 1.2, 1.6),
        "score_financial": lambda f: sc.score_financial(f, dist),
        "dividend_gates": sc.dividend_gates, "quality_gates": sc.quality_gates,
        "recovery_gates": sc.recovery_gates, "financial_gates": sc.financial_gates,
        "data_quality_gates": sc.data_quality_gates,
        "data_quality_report": sc.data_quality_report,
        "row:dividend": lambda f: da.to_rows("dividend", [(f, sc.score_dividend(f))]),
        "row:value": lambda f: da.to_rows("value", [(f, sc.score_quality_value(f))]),
        "row:recovery": lambda f: da.to_rows(
            "recovery", [(f, sc.score_recovery(f, 40.0, 1.2, 1.6))]),
        "row:financial": lambda f: da.to_rows("financial", [(f, sc.score_financial(f, dist))]),
        "financial_distributions": lambda f: sc.financial_distributions([f]),
    }


@pytest.mark.parametrize("consumer", sorted(_consumers()))
def test_every_consumer_survives_every_voidable_field_none(consumer):
    import copy, json
    fn = _consumers()[consumer]
    V = sorted(sc.voidable_fields())
    assert len(V) >= 24
    failures = []
    for base in _CONTRACT_BASES:
        variants = [("ALL", V)] + [(v, [v]) for v in V]
        for label, fields in variants:
            f = copy.deepcopy(base)
            for v in fields:
                setattr(f, v, None)
            try:
                out = fn(f)
                json.dumps(out, default=str, allow_nan=False)   # and it must serialise
            except Exception as e:
                failures.append(f"{base.symbol} None[{label}]: {type(e).__name__}: {e}")
    assert not failures, f"{len(failures)} failure(s), first: " + "; ".join(failures[:4])


def test_run_screen_survives_every_voidable_none_among_real_rows():
    import copy
    V = sc.voidable_fields()
    recs = []
    for base in _CONTRACT_BASES:
        g = copy.deepcopy(base)
        for v in V:
            setattr(g, v, None)
        recs += [copy.deepcopy(base), g]
    for w in ("dividend", "quality", "recovery"):
        sc.run_screen(recs, w, strict=False, min_score=0)


def test_everything_that_can_be_voided_is_declared_voidable():
    """Enrolment is automatic only if the annotation cannot be forgotten."""
    import ast, inspect, re
    from engines import fundamentals_builder as fb
    names = {x.name for x in _dc.fields(sc.Fundamentals)}
    V = sc.voidable_fields()
    in_void_rules = set(re.findall(r'"([a-z_0-9]+)"',
                                   inspect.getsource(fb.void_derived_fields))) & names
    none_assigned = set()
    for node in ast.walk(ast.parse(inspect.getsource(fb))):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "f" and t.attr in names):
                    continue
                # `x = 0.0 if y is None else y` never ASSIGNS None — the None
                # is the test, not the value. Walk the value with comparison
                # operands removed so a guard does not read as a void.
                assigned = [n for n in ast.walk(node.value)
                            if not isinstance(n, ast.Compare)]
                in_compare = {id(c) for cmp_ in ast.walk(node.value)
                              if isinstance(cmp_, ast.Compare)
                              for c in ast.walk(cmp_)}
                if any(isinstance(n, ast.Constant) and n.value is None
                       and id(n) not in in_compare for n in assigned):
                    none_assigned.add(t.attr)
    bools = {x.name for x in _dc.fields(sc.Fundamentals) if "bool" in str(x.type)}
    missing = (in_void_rules | (none_assigned - bools)) - V
    assert not missing, f"can be None but not annotated `| None`: {sorted(missing)}"


def test_build_refuses_none_in_an_undeclared_field(monkeypatch):
    from engines import fundamentals_builder as fb
    real = fb.void_derived_fields
    def sneaky(f):
        out = real(f)
        f.revenue_growth_ttm = None         # not declared voidable
        return out
    monkeypatch.setattr(fb, "void_derived_fields", sneaky)
    with pytest.raises(ValueError, match="revenue_growth_ttm"):
        fb.build("X", _gaap(_ann("Revenues", {y: 2.0e10 for y in YEARS}),
                            _ann("NetIncomeLoss", {y: 1.0e9 for y in YEARS})))


def test_a_carried_status_entry_says_when_it_happened_and_that_it_was_carried(tmp_path, monkeypatch):
    """Sunday's pre-fix crash read as a Monday 00:53 crash."""
    import json
    import run_all
    data = tmp_path / "data"; data.mkdir()
    (data / "status.json").write_text(json.dumps({"updated_at": "2026-09-13T19:37:45+00:00",
        "engines": {"dividend": {"state": "error", "detail": "TypeError: int - NoneType"}}}))
    monkeypatch.setattr(run_all, "should_run", lambda name, force: name == "recession")
    eng = _run_main(monkeypatch, data, [], {k: (lambda: {}) for k in run_all.ENGINES})
    assert "at" not in eng["dividend"], "an upper bound must not be recorded as the time"
    assert eng["dividend"]["at_or_before"] == "2026-09-13T19:37:45+00:00"
    assert eng["dividend"]["carried"] is True
    assert "carried" not in eng["recession"]


def test_darkpool_reports_its_funnel_and_distribution(monkeypatch):
    """Zero rows must say whether 1,500 names were scored or 12."""
    from engines import finra_darkpool as fd
    panel = _dp_panel()
    monkeypatch.setattr(fd, "build_panel", lambda finra, tape: panel)
    monkeypatch.setattr(fd, "add_zscores", lambda df, cfg: df)
    import pandas as pd
    finra = pd.DataFrame({"symbol": ["GOOD", "BAD", "NOTAPE"]})
    tape = pd.DataFrame({"symbol": ["GOOD", "BAD"]})
    rep = {}
    board = fd.run(finra, tape, report=rep)
    assert rep["finra_symbols"] == 3 and rep["tape_symbols"] == 2
    assert rep["scored"] == 2 and rep["passed"] == len(board)
    assert rep["max"] is not None and rep["top"][0]["symbol"] in {"GOOD", "BAD"}
    assert rep["block_trend_wired"] is False and rep["rvol_z_available"] == 0


# ------------------------------------------------ ETFs, series, custom tabs

def test_etfs_are_excluded_before_the_tape_is_fetched():
    import pandas as pd
    from engines import finra_darkpool as fd
    finra = pd.DataFrame({"symbol": ["SPYI", "DV", "BF/B", "SPYG", "BRK/B"]})
    kept, removed = fd.exclude_etfs(finra, frozenset({"SPYI", "SPYG"}))
    assert set(kept["symbol"]) == {"DV", "BF/B", "BRK/B"}
    assert removed == ["SPYG", "SPYI"]


def test_etf_list_refuses_to_screen_with_a_partial_list(monkeypatch):
    """An empty or truncated list would silently put every ETF back on the board."""
    from engines import free_sources as fs
    class R:
        text = "Nasdaq Traded|Symbol|Security Name|ETF\nY|SPY|SPDR|Y\nY|AAPL|Apple|N\n"
    monkeypatch.setattr(fs, "_get", lambda *a, **k: R())
    fs.load_etf_symbols.cache_clear()
    with pytest.raises(RuntimeError, match="implausible"):
        fs.load_etf_symbols()
    class Bad:
        text = "Symbol|Name\nSPY|x\n"
    monkeypatch.setattr(fs, "_get", lambda *a, **k: Bad())
    fs.load_etf_symbols.cache_clear()
    with pytest.raises(RuntimeError, match="header"):
        fs.load_etf_symbols()
    fs.load_etf_symbols.cache_clear()


def test_recession_series_is_the_whole_window_on_shared_dates():
    """It was the last 160 points while the chart claimed three years."""
    import pandas as pd
    import run_all
    idx = pd.date_range("2023-09-12", periods=800, freq="D")
    hy = pd.Series(range(800), index=idx, dtype=float)
    ccc = hy.copy(); ccc.iloc[5] = float("nan")
    out = run_all._aligned_series({"hy": hy, "ccc": ccc, "bb": hy})
    assert len(out["t"]) == 799 and out["t"][0] == "2023-09-12"
    assert run_all._last_obs(ccc)["date"] == idx[-1].strftime("%Y-%m-%d")


def test_custom_tabs_render_from_their_files_not_constants():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "index.html").read_text()
    assert "const RENDERER_UNWIRED={};" in html
    for fn in ("function renderCycleLive(e,d)", "function renderRecessionLive(e,d)"):
        assert fn in html
    live = html.split("LIVE RENDERERS")[1].split("TICKER DETAIL PAGE")[0]
    # the live renderers must not reach for the embedded sample constants
    import re
    for const in (r"\bREC\.[a-z]", r"\bBTC\.[a-z]", r"\bBTCD\.[a-z]", r"\bOAS\.[a-z]",
                  r"\be\.regime\b", r"\be\.clock\b", r"\be\.alerts\b"):
        assert not re.search(const, live), f"live renderer reads sample constant {const}"


def test_darkpool_rows_satisfy_the_dashboard_contract():
    """21 real rows crashed the tab on `r.name` — every earlier board was empty."""
    import re
    from pathlib import Path
    from engines import dashboard_adapter as da
    html = (Path(__file__).resolve().parents[1] / "index.html").read_text()
    block = html.split("darkpool:{", 1)[1].split("rows:", 1)[0]
    keys = re.findall(r'\{k:"(\w+)"', block)
    comps = re.findall(r'comps:\[([^\]]*)\]', block)[0].count('"') // 2
    assert keys, "no darkpool columns found"
    r = {"symbol": "BF/B", "score": 71, "state": "Neutral", "dpi_5d": 52.5, "dpi_z": 0.78,
         "oe_share": 54.9, "rvol": 1.15, "compression": 87, "ret_20d": 0.8, "dollar_adv": 4.1e8,
         "components": {k: 60 for k in da.DARKPOOL_ORDER}}
    row = da.darkpool_row(r, {"BFB": "Brown-Forman"})
    missing = [k for k in keys if k not in row]
    assert not missing, f"darkpool columns not produced: {missing}"
    assert isinstance(row["name"], str) and row["name"] == "Brown-Forman"
    assert len(row["comp"]) == comps and row["note"] and len(row["facts"]) == 4
    # and it survives unknowns without inventing numbers
    bare = da.darkpool_row({"symbol": "X", "score": 62, "components": {}}, None)
    assert bare["name"] == "" and bare["dpi"] is None and bare["si"] is None


# ------------------------------------------------------- REIT vs real estate

def test_only_6798_is_a_reit_not_the_whole_6500_block(monkeypatch):
    """
    6500-6599 is real-estate SERVICES — CBRE, JLL, Zillow. Labelling them
    `reit` handed them the trust allowances (payout 85/FCF 90/debt 6.0x) and a
    6.5% cost of capital, making them easier to pass than an industrial.
    """
    from engines import free_sources as fs
    cases = {6798: "reit", 6500: "general", 6531: "general", 6552: "general",
             6599: "general", 6022: "financial", 4911: "utility", 7372: "general"}
    for sic, want in cases.items():
        monkeypatch.setattr(fs, "company_sic", lambda t, _s=sic: _s)
        assert fs.company_sector("X") == want, f"SIC {sic} -> {want}"


def test_real_estate_services_face_the_general_dividend_caps():
    """The allowance must not be reachable by a services business."""
    base = dict(symbol="CBRE", name="CBRE Group", increase_streak_years=10,
                years_since_cut=99, dps_cagr_5y=8.0, dps_cagr_3y=8.0,
                revenue_cagr_5y=6.0, market_cap=3e10, dollar_adv=2e8,
                eps_payout=80.0, fcf_payout=80.0, net_debt_ebitda=4.5,
                interest_coverage=9.0)
    assert sc.dividend_gates(sc.Fundamentals(sector="general", **base)), \
        "80% payout and 4.5x leverage must fail the general caps"
    assert not sc.dividend_gates(sc.Fundamentals(sector="reit", **base)), \
        "the trust allowance is what this name used to pass on"


def test_fcf_yield_is_voided_when_capex_was_never_tagged():
    """
    FCF is `ocf - capex.fillna(0)`, so an untagged capex turns the YIELD into an
    OCF yield. fcf_margin and fcf_payout were voided; fcf_yield was not, and
    149 of 1,449 records carried one — Alexandria 17.4%, PBR 1.6e12%.
    """
    from engines import fundamentals_builder as fb
    f = sc.Fundamentals(symbol="ARE", name="Alexandria")
    f.fcf_unavailable, f.capex_voided_fcf = True, True
    f.fcf_yield, f.fcf_margin, f.fcf_payout = 17.39, 40.0, 55.0
    voided = fb.void_derived_fields(f)
    assert "fcf_yield" in voided and f.fcf_yield is None


def test_a_voided_fcf_yield_drops_out_rather_than_scoring_neutral():
    """Substituting a neutral 50 made missing data outrank a real low yield."""
    base = dict(symbol="X", name="X", ev_ebit_percentile_10y=10.0)
    real_low = sc.score_dividend(sc.Fundamentals(fcf_yield=1.0, **base))["components"]["valuation"]
    voided = sc.score_dividend(sc.Fundamentals(fcf_yield=None, **base))["components"]["valuation"]
    only_pct = sc._clamp(sc._scale(100 - 10.0, 20, 90))
    assert voided == round(only_pct), "the voided term must leave the mean, not sit at 50"
    assert voided > real_low, "a real 1% yield legitimately scores below the percentile alone"
    q = sc.score_quality_value(sc.Fundamentals(fcf_yield=None, **base))["components"]["fcf_yield"]
    assert q == 0 or q is not None


# ---------------------------------------- flow facts: instants and quarters

def _dur(start, end, val, form="10-K", filed=None):
    return {"start": start, "end": end, "val": val, "fy": int(end[:4]),
            "fp": "FY", "form": form, "filed": filed or f"{int(end[:4])+1}-02-15"}


def _inst(end, val, form="10-K", filed=None):
    return {"end": end, "val": val, "fy": int(end[:4]), "fp": "FY",
            "form": form, "filed": filed or f"{int(end[:4])+1}-02-15"}


def _dps(items):
    return {"facts": {"us-gaap": {
        "CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": items}}}}}


def test_a_flow_tagged_as_an_instant_is_not_an_annual_figure():
    """Boston Properties: 56 quarterly declarations, each a bare instant."""
    from engines import free_sources as fs
    years = range(2020, 2024)
    items = [_inst(f"{y}-{m}-28", 0.98) for y in years for m in ("01", "04", "07", "10")]
    df = fs.extract_series(_dps(items), "dividends_per_share")
    # four instants in a year ARE a complete quarterly cadence: sum them
    assert set(df["val"]) == {0.98 * 4}
    assert df.attrs["years_from_instants"] == len(list(years))
    # three is ambiguous — a year ends honestly rather than being guessed
    partial = [_inst(f"2025-{m}-28", 0.98) for m in ("01", "04", "07")]
    df2 = fs.extract_series(_dps(items + partial), "dividends_per_share")
    import pandas as _pd
    assert 2025 not in set(_pd.to_datetime(df2["end"]).dt.year)


def test_a_stock_concept_still_accepts_instants():
    """The rule must not reach balance-sheet concepts, which ARE instants."""
    from engines import free_sources as fs
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        _inst("2024-12-31", 5e9), _inst("2025-12-31", 6e9)]}}}}}
    assert len(fs.extract_series(facts, "assets")) == 2


def test_an_annual_fact_smaller_than_its_own_quarters_is_rejected():
    """
    Extra Space Storage 2020: two 365-day facts, 0.90 and 3.60. The
    first-reported dedup kept 0.90 while three tagged quarters sum to 2.70 —
    a median cannot see this, arithmetic can.
    """
    from engines import free_sources as fs
    items = [_dur("2020-01-01", "2020-12-31", 0.90, filed="2021-02-01"),
             _dur("2020-01-01", "2020-12-31", 3.60, filed="2021-03-01"),
             _dur("2020-01-01", "2020-03-31", 0.90, form="10-Q"),
             _dur("2020-04-01", "2020-06-30", 0.90, form="10-Q"),
             _dur("2020-07-01", "2020-09-30", 0.90, form="10-Q")]
    df = fs.extract_series(_dps(items), "dividends_per_share")
    assert list(df["val"]) == [3.60], "the year must not read as one quarter"


def test_quarters_that_do_not_cover_the_year_may_not_replace_it():
    """Apple 2020: three quarters sum to 0.60 against a correct annual 0.795."""
    from engines import free_sources as fs
    items = [_dur("2020-01-01", "2020-12-31", 0.795),
             _dur("2020-01-01", "2020-03-31", 0.20, form="10-Q"),
             _dur("2020-04-01", "2020-06-30", 0.20, form="10-Q"),
             _dur("2020-07-01", "2020-09-30", 0.20, form="10-Q")]
    df = fs.extract_series(_dps(items), "dividends_per_share")
    assert list(df["val"]) == [0.795]


def test_year_to_date_durations_are_never_summed_as_quarters():
    """Filers tag 180- and 272-day cumulatives beside the discrete quarters."""
    from engines import free_sources as fs
    items = [_dur("2021-01-01", "2021-12-31", 4.00),
             _dur("2021-01-01", "2021-03-31", 1.00, form="10-Q"),
             _dur("2021-01-01", "2021-06-30", 2.00, form="10-Q"),   # cumulative
             _dur("2021-04-01", "2021-06-30", 1.00, form="10-Q"),
             _dur("2021-01-01", "2021-09-30", 3.00, form="10-Q"),   # cumulative
             _dur("2021-07-01", "2021-09-30", 1.00, form="10-Q"),
             _dur("2021-10-01", "2021-12-31", 1.00, form="10-Q")]
    df = fs.extract_series(_dps(items), "dividends_per_share")
    assert list(df["val"]) == [4.00]


# ================================================== REIT screen: contract

def _reit_record(**kw):
    f = sc.Fundamentals(symbol="VICI", name="VICI Properties", sector="reit")
    f.reit_in_scope = True
    f.ffo_ps, f.ffo_payout, f.p_ffo, f.p_ffo_percentile_10y = 2.40, 74.0, 13.5, 28.0
    f.affo_yield, f.ffo_cagr_5y, f.net_debt_ebitda = 6.1, 7.4, 5.2
    f.dividend_yield, f.ffo_positive, f.deep_cut_3y = 5.6, True, False
    f.ffo, f.ffo_unavailable, f.affo_unavailable = 2.4e9, False, False
    f.interest_coverage, f.share_count_cagr_5y, f.revenue_cagr_5y = 4.0, 2.0, 6.0
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def test_reit_dashboard_contract_is_satisfied():
    """
    Written BEFORE the row builder, like the financials screen. The engine
    emits {symbol, components, gates_failed}; the tab asks for ticker, pffo,
    ffopay, affoy. Wiring them straight through renders correctly-ranked blanks.
    """
    html = (Path(__file__).resolve().parents[1] / "index.html").read_text()
    block = html.split("reit:{", 1)[1].split("rows:", 1)[0]
    keys = re.findall(r'\{k:"(\w+)"', block)
    comps = re.findall(r'comps:\[([^\]]*)\]', block)[0].count('"') // 2
    f = _reit_record()
    res = sc.score_reit(f)
    row = da.to_rows("reit", [(f, res)])[0]
    missing = [k for k in keys if k not in row]
    assert not missing, f"reit columns not produced: {missing}"
    assert len(row["comp"]) == comps
    assert row["note"] and len(row["facts"]) == 4
    assert isinstance(row["ticker"], str) and row["score"] == res["score"]


def test_reit_gates_enforce_coverage_not_a_streak():
    """
    The statute forces distributions to track taxable income, so a smoothing
    streak is the wrong test — 19 of 62 trusts show a zero-year streak. The
    replacement is coverage plus a deep-cut test.
    """
    f = _reit_record()
    assert not sc.reit_gates(f), sc.reit_gates(f)
    assert not any("streak" in g for g in sc.reit_gates(_reit_record(increase_streak_years=0)))
    assert any("90%" in g for g in sc.reit_gates(_reit_record(ffo_payout=104.0)))
    assert any("third" in g for g in sc.reit_gates(_reit_record(deep_cut_3y=True)))
    assert any("distribut" in g for g in sc.reit_gates(_reit_record(dividend_yield=0.0)))
    # an industrial's 3.5x leverage cap describes nothing here
    assert not sc.reit_gates(_reit_record(net_debt_ebitda=5.5))
    assert any("8.0x" in g for g in sc.reit_gates(_reit_record(net_debt_ebitda=9.0)))


def test_reit_screen_does_not_inherit_fcf_or_ebit_gates():
    """16 of 62 failed on "capex not tagged" for a screen with no FCF in it."""
    f = _reit_record(fcf_unavailable=True, capex_voided_fcf=True, ebit_unavailable=True)
    assert not sc.reit_gates(f), sc.reit_gates(f)
    # the general screens still enforce both
    assert any("capex" in g for g in sc.dividend_gates(f))


def test_ffo_overstated_without_a_gains_tag_is_gated():
    """Nothing subtracted means the payout reads safer than it is."""
    assert any("gains-on-sale" in g for g in sc.reit_gates(_reit_record(ffo_degraded=True)))


# --------------------------------------------- one concept, one scale

def test_a_scale_switch_inside_one_tag_is_snapped():
    """
    ConocoPhillips tags 1,245,440 (thousands) then 1,253,446,000 (shares) —
    the same 1.25bn — and share_count_cagr_5y read +310%/yr. 67 of 1,449 names
    carried a CAGR above 50%/yr, which is a scale, not a company.
    """
    from engines import free_sources as fs
    items = [_inst(f"{y}-12-31", v) for y, v in
             [(2019, 1_123_536), (2020, 1_078_030), (2021, 1_328_151),
              (2022, 1_278_163_000), (2023, 1_205_675_000), (2024, 1_253_446_000)]]
    facts = {"facts": {"us-gaap": {"CommonStockSharesOutstanding":
                                   {"units": {"shares": items}}}}}
    df = fs.extract_series(facts, "shares")
    vals = sorted(df["val"])
    assert all(9e8 < v < 2e9 for v in vals), vals
    assert df.attrs["facts_rescaled"] == 3


def test_a_stock_split_is_not_a_scale_switch():
    """Apple's 4:1 and NVIDIA's 10:1 must pass through to split_adjust."""
    from engines import free_sources as fs
    items = [_inst(f"{y}-12-31", v) for y, v in
             [(2021, 628_000_000), (2022, 2_535_000_000),      # 4:1
              (2023, 2_507_000_000), (2024, 24_804_000_000)]]  # 10:1
    facts = {"facts": {"us-gaap": {"CommonStockSharesOutstanding":
                                   {"units": {"shares": items}}}}}
    df = fs.extract_series(facts, "shares")
    assert df.attrs["facts_rescaled"] == 0
    assert max(df["val"]) == 24_804_000_000


def test_a_clean_series_is_left_alone():
    from engines import free_sources as fs
    items = [_inst(f"{y}-12-31", v) for y, v in
             [(2022, 4_350_000_000), (2023, 4_339_000_000), (2024, 4_320_000_000)]]
    facts = {"facts": {"us-gaap": {"CommonStockSharesOutstanding":
                                   {"units": {"shares": items}}}}}
    df = fs.extract_series(facts, "shares")
    assert df.attrs["facts_rescaled"] == 0 and len(df) == 3


def test_the_floor_rule_never_touches_a_flow_that_can_go_negative():
    """
    Three positive quarters and a fourth-quarter loss make an annual net income
    legitimately smaller than their sum. Applying the floor there deleted 42
    net-income periods for Elastic and moved its ROIC from -37.7% to +1.7%.
    """
    from engines import free_sources as fs
    items = [_dur("2024-01-01", "2024-12-31", 5e6),
             _dur("2024-01-01", "2024-03-31", 10e6, form="10-Q"),
             _dur("2024-04-01", "2024-06-30", 10e6, form="10-Q"),
             _dur("2024-07-01", "2024-09-30", 10e6, form="10-Q"),
             _dur("2024-10-01", "2024-12-31", -25e6, form="10-Q")]
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": items}}}}}
    df = fs.extract_series(facts, "net_income")
    assert list(df["val"]) == [5e6]
    assert df.attrs["periods_reconciled"] == 0 and df.attrs["periods_inconsistent"] == 0


def test_a_disagreeing_year_is_kept_not_deleted():
    """
    A 4:1 split moves every per-share value by exactly the factor a quarterly
    straggler does, so a year that merely disagrees must not be deleted —
    Tractor Supply lost three years to an earlier version of this.
    """
    from engines import free_sources as fs
    items = [_dur("2023-01-01", "2023-12-31", 0.82),
             _dur("2023-01-01", "2023-03-31", 1.02, form="10-Q"),
             _dur("2023-04-01", "2023-06-30", 1.02, form="10-Q"),
             _dur("2023-07-01", "2023-09-30", 1.02, form="10-Q")]
    df = fs.extract_series(_dps(items), "dividends_per_share")
    assert len(df) == 1, "the year must survive"
    assert df.attrs["periods_inconsistent"] == 1, "and the disagreement recorded"


def test_the_quarter_floor_is_restricted_to_dividends():
    """
    A 52/53-week filer can end FIVE quarters inside one calendar year.
    Extending the floor to revenue made Tractor Supply's 2020 read $18.4bn
    against a real $10.6bn and flipped its 5y CAGR from +7.9% to -3.3%.
    """
    from engines import free_sources as fs
    assert fs.RECONCILABLE_FLOWS == {"dividends_per_share", "dividends_paid"}
    items = [_dur("2020-01-01", "2020-12-31", 10.6e9)] + [
        _dur(f"2020-{a}", f"2020-{b}", 4.5e9, form="10-Q")
        for a, b in (("01-01", "03-31"), ("04-01", "06-30"),
                     ("07-01", "09-30"), ("10-01", "12-31"))]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": items}}}}}
    df = fs.extract_series(facts, "revenue")
    assert list(df["val"]) == [10.6e9], "revenue must not be rewritten by its quarters"


# ------------------------------------------- a CAGR measures what it claims

def test_a_cagr_declines_rather_than_relabelling_a_shorter_span():
    """
    `span = min(years, len(s) - 1)` relabelled whatever it had: 21 names in a
    1,449-name universe carried a ONE-YEAR change presented as a five-year
    share-count CAGR, which is how a newly listed company reads as a serial
    diluter.
    """
    import pandas as pd
    from engines import fundamentals_builder as fb
    idx = pd.date_range("2020-12-31", periods=6, freq="YE")
    full = pd.Series([100, 110, 121, 133, 146, 161], index=idx, dtype=float)
    assert fb._cagr(full, 5) == pytest.approx(10.0, abs=0.1)
    # two or three points cannot support a five-year rate
    assert fb._cagr(full.iloc[-2:], 5) is None
    assert fb._cagr(full.iloc[-3:], 5) is None
    # three years of span is the floor, and it measures what it reports
    assert fb._cagr(full.iloc[-4:], 5) == pytest.approx(10.0, abs=0.1)
    # a one-year measure still works off two points
    assert fb._cagr(full.iloc[-2:], 1) == pytest.approx(10.3, abs=0.2)


def test_a_newly_listed_name_does_not_read_as_a_serial_diluter():
    import pandas as pd
    from engines import fundamentals_builder as fb
    idx = pd.date_range("2024-12-31", periods=2, freq="YE")
    shares = pd.Series([1e6, 120e6], index=idx, dtype=float)   # IPO
    assert fb._cagr(shares, 5) is None, "a 1-year listing jump is not a 5y CAGR"
    f = sc.Fundamentals(symbol="IPO", name="Newly listed", share_count_cagr_5y=None)
    assert not any("share count growing" in g for g in sc.quality_gates(f))


def test_build_survives_a_share_count_that_cannot_support_a_five_year_rate():
    """
    Unary minus on a None CAGR crashed build() for 147 of 1,449 names, and
    every one was counted as "never became a record" rather than as a crash.
    """
    from engines import fundamentals_builder as fb
    facts = _gaap(_ann("Revenues", {y: 2.0e10 for y in YEARS}),
                  _ann("NetIncomeLoss", {y: 1.0e9 for y in YEARS}))
    facts["facts"]["us-gaap"]["CommonStockSharesOutstanding"] = {"units": {"shares": [
        _inst("2024-12-31", 1.0e9), _inst("2025-12-31", 1.02e9)]}}      # 2 points
    f = fb.build("THIN", facts)
    assert f.share_count_cagr_5y is None and f.buyback_yield == 0.0


def test_an_unmeasurable_dilution_rate_does_not_pass_as_safe():
    f = sc.Fundamentals(symbol="X", name="X", share_count_cagr_5y=None)
    assert any("not measurable" in g for g in sc.quality_gates(f))
    assert any("not measurable" in g for g in sc.recovery_gates(f))


# ------------------------------------------------- the shares chain

def _dei(items):
    return {"facts": {"dei": {"EntityCommonStockSharesOutstanding":
                              {"units": {"shares": items}}},
                      "us-gaap": _ann("Revenues", {y: 1e10 for y in YEARS})}}


def test_cover_page_shares_are_used_when_the_statement_tags_are_absent():
    """Baker Hughes tags 100 shares on a 10-Q; its real 992m is in `dei`."""
    from engines import free_sources as fs
    items = [_inst(f"{y}-02-10", 990_000_000 + y) for y in YEARS]
    df = fs.extract_series(_dei(items), "shares")
    assert len(df) == len(list(YEARS))
    assert df.iloc[-1]["tag"].startswith("dei:")


def test_two_share_classes_are_summed_not_picked():
    """Picking one class understates the count on every multi-class filer."""
    from engines import free_sources as fs
    items = []
    for y in YEARS:
        items += [_inst(f"{y}-02-10", 600_000_000),      # class A
                  _inst(f"{y}-02-10", 300_000_000)]      # class B
    df = fs.extract_series(_dei(items), "shares")
    assert set(df["val"]) == {900_000_000}


def test_an_amended_filing_does_not_double_count_a_class():
    """Shell files a 20-F and a 20-F/A carrying the SAME value on one date."""
    from engines import free_sources as fs
    items = []
    for y in YEARS:
        items += [_inst(f"{y}-02-10", 6_486_295_984, form="20-F"),
                  _inst(f"{y}-02-10", 6_486_295_984, form="20-F/A")]
    df = fs.extract_series(_dei(items), "shares")
    assert set(df["val"]) == {6_486_295_984}


def test_a_cover_page_that_stopped_is_refused_not_carried_forward():
    """
    Visa's only un-dimensioned dei facts are from 2009-2010 — its classes are
    DIMENSIONED and companyfacts omits those. A 2010 count feeding today's
    market cap is the stale-series failure, so it is refused outright.
    """
    from engines import free_sources as fs
    items = [_inst("2009-11-13", 470_210_301), _inst("2010-01-27", 469_280_842)]
    assert fs.extract_series(_dei(items), "shares").empty


def test_ifrs_filers_get_their_weighted_average_share_count():
    """WeightedAverageShares covers 64 of the 69 IFRS filers that had none."""
    from engines import free_sources as fs
    facts = {"facts": {"ifrs-full": {
        "WeightedAverageShares": {"units": {"shares": [
            _dur(f"{y}-01-01", f"{y}-12-31", 25_929_000_000) for y in YEARS]}},
        **_ann("Revenue", {y: 1e10 for y in YEARS})}}}
    df = fs.extract_series(facts, "shares")
    assert len(df) == len(list(YEARS)) and df.iloc[-1]["val"] == 25_929_000_000


def test_a_builder_crash_is_not_reported_as_missing_data(monkeypatch):
    """
    147 of 1,449 names were lost to a TypeError in build() and counted beside
    "no annual revenue", as though the filers were at fault. A crash is a bug
    to fix; missing data is a fact about the filer.
    """
    from engines import fundamentals_builder as fb
    from engines import free_sources as fs
    facts = _gaap(_ann("Revenues", {y: 1e10 for y in YEARS}),
                  _ann("NetIncomeLoss", {y: 1e9 for y in YEARS}))
    monkeypatch.setattr(fs, "company_facts", lambda t: facts)
    monkeypatch.setattr(fs, "company_sector", lambda t: "general")
    monkeypatch.setattr(fs, "company_sic", lambda t: 3571)
    monkeypatch.setattr(fb, "build", lambda *a, **k: (_ for _ in ()).throw(TypeError("bad operand")))
    rep = fb.load_fundamentals_report(["BOOM"], with_prices=False)
    assert list(rep["by_reason"]) == ["BUILDER CRASH (a bug, not missing data)"]
    monkeypatch.setattr(fb, "build", lambda *a, **k: (_ for _ in ()).throw(ValueError("X: no annual revenue facts")))
    rep2 = fb.load_fundamentals_report(["NOREV"], with_prices=False)
    assert list(rep2["by_reason"]) == ["no annual revenue"]


def test_a_single_year_step_voids_the_rate_rather_than_shortening_it():
    """
    Measuring from after the step would redefine the label — a "5-year CAGR"
    over two years is the span bug through a different door. And the steps are
    not one shape: Grab and Bitdeer are SPAC listings, Nu is a predecessor
    basis (306, 405, 334, 184, 4,858 — two series concatenated, with no
    meaningful "after"), AngloGold a real secondary.
    """
    import pandas as pd
    from engines import fundamentals_builder as fb
    idx = pd.date_range("2020-12-31", periods=6, freq="YE")
    nu = pd.Series([306e6, 405e6, 334e6, 184e6, 4858e6, 4889e6], index=idx)
    steps = {}
    assert fb._cagr(nu, 5, sink=steps, name="shares") is None
    assert steps["shares"]["ratio"] == 26.4 and steps["shares"]["year"] == 2024


def test_the_inverse_step_is_the_same_discontinuity():
    """A reverse split or a spin-off runs the break the other way."""
    import pandas as pd
    from engines import fundamentals_builder as fb
    idx = pd.date_range("2020-12-31", periods=6, freq="YE")
    rev = pd.Series([500e6, 505e6, 510e6, 50e6, 51e6, 52e6], index=idx)  # 1:10
    steps = {}
    assert fb._cagr(rev, 5, sink=steps, name="shares") is None
    assert steps["shares"]["ratio"] <= 0.2


def test_an_old_step_outside_the_window_does_not_void_the_rate():
    """A break ten years ago says nothing about a five-year rate."""
    import pandas as pd
    from engines import fundamentals_builder as fb
    idx = pd.date_range("2014-12-31", periods=12, freq="YE")
    v = pd.Series([10e6, 100e6] + [100e6 * 1.05 ** i for i in range(10)], index=idx)
    assert fb._cagr(v, 5) == pytest.approx(5.0, abs=0.3)


def test_a_one_year_growth_rate_is_not_voided_by_its_own_step():
    """
    revenue_growth_ttm measures a single transition, so a 5x year IS the
    measurement. Voiding it took AST SpaceMobile, CRISPR and QXO out of the
    universe entirely.
    """
    import pandas as pd
    from engines import fundamentals_builder as fb
    idx = pd.date_range("2021-12-31", periods=5, freq="YE")
    rev = pd.Series([1e6, 2e6, 3e6, 4e6, 40e6], index=idx)      # 10x final year
    assert fb._cagr(rev, 1) == pytest.approx(900.0, abs=1.0)
    steps = {}
    assert fb._cagr(rev, 5, sink=steps, name="revenue") is None   # 5y still void
    assert steps["revenue"]["ratio"] == 10.0
