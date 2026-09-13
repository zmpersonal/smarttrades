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
    src = inspect.getsource(fb.void_derived_fields)
    names = {f.name for f in dataclasses.fields(sc.Fundamentals)}
    voidable = set(re.findall(r'"([a-z_0-9]+)"', src)) & names
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
