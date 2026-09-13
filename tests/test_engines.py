"""
Tests for the bugs that actually happened, plus the invariants that must hold.

Not exhaustive coverage — targeted at the three classes of failure this
codebase has already produced:

  1. numpy scalars leaking into payloads and breaking JSON serialisation
  2. alert rules re-firing on unchanged input
  3. indicators giving different answers on daily vs weekly input

Run:  python -m pytest tests -q
"""

import io
import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engines import btc_cycle as bc
from engines import indicators as ind
from engines import recession as rec
from engines import screeners as sc


# --------------------------------------------------- JSON serialisability

def _assert_json(obj, label):
    """np.bool_ and np.float64 are not JSON-serialisable. This bit twice."""
    try:
        json.dumps(obj)
    except TypeError as e:
        pytest.fail(f"{label} is not JSON-serialisable: {e}")


def test_cycle_position_json():
    out = bc.cycle_position(date(2026, 9, 6), 126296, date(2025, 10, 6), 77400)
    _assert_json(out, "cycle_position")
    assert out["peak_day_post_halving"] == 534
    assert -0.40 < out["drawdown"] < -0.38


def test_divergence_json(weekly):
    pushes = bc.find_pushes(weekly)
    _assert_json(bc.detect_divergence(pushes), "detect_divergence")


def test_recession_outputs_json(oas):
    _assert_json(rec.oas_state(oas["hy"]), "oas_state")
    _assert_json(rec.quality_dispersion(oas["ccc"], oas["bb"]), "quality_dispersion")
    _assert_json(rec.curve_probit(-0.08), "curve_probit")


# ------------------------------------------------------------- fixtures

@pytest.fixture
def weekly():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-07", periods=140, freq="W-SUN")
    px = 60000 * np.exp(np.cumsum(rng.normal(0.002, 0.05, len(idx))))
    df = pd.DataFrame({"close": px}, index=idx)
    df["rsi"] = bc.rsi(df["close"])
    return df


@pytest.fixture
def oas():
    idx = pd.date_range("2023-09-11", periods=780, freq="B")
    rng = np.random.default_rng(3)
    hy = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.02, len(idx))).clip(-1.0, 2.0), index=idx)
    return {"hy": hy, "bb": hy * 0.65 + 0.2, "ccc": 1.4 + hy * 2.1}


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2025-01-01", periods=400, freq="B")
    px = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, len(idx))))
    return pd.DataFrame({"close": px, "high": px * 1.012, "low": px * 0.988,
                         "volume": rng.lognormal(15, 0.4, len(idx))}, index=idx)


# ------------------------------------------------- timeframe independence

def test_oas_lookbacks_are_date_based(oas):
    """
    Positional lookbacks made iloc[-22] mean '22 weeks' on weekly input.
    The 1-month change must be broadly the same on daily and weekly data.
    """
    daily = oas["hy"]
    weekly = daily.resample("W-FRI").last().dropna()
    d = rec.oas_state(daily)["change_1m"]
    w = rec.oas_state(weekly)["change_1m"]
    assert abs(d - w) < 0.35, f"daily {d} vs weekly {w} — lookback is positional again"


# ---------------------------------------------------------- gate ordering

def test_gate_failure_is_disqualifying():
    """A name that fails a gate must never rank, however good its score."""
    f = sc.Fundamentals(
        symbol="TRAP", name="Value Trap Inc", roic_5y=2.0, fcf_yield=11.0,
        ev_ebit=8.0, ev_ebit_percentile_10y=3.0, gross_margin=20.0,
        fcf_margin=2.0, revenue_cagr_5y=-2.0, market_cap=5e9, dollar_adv=1e7,
    )
    res = sc.score_quality_value(f)
    assert res["gates_failed"], "deep-value trap should fail quality gates"
    assert not sc.run_screen([f], "quality", strict=True, min_score=0)


def test_recovery_rejects_excessive_drawdown():
    f = sc.Fundamentals(symbol="X", name="X", drawdown_from_ath=-94.0,
                        gross_margin=45.0, altman_z=3.0, cash_runway_quarters=99)
    assert any("drawdown" in g for g in sc.recovery_gates(f))


def test_multiple_only_thesis_is_penalised():
    """A 3x carried by re-rating alone is fragile and must score lower."""
    balanced = sc.decompose_upside(60.0, 1.2, 1.6)
    multiple = sc.decompose_upside(5.0, 1.0, 2.85)
    assert multiple["leg_share"]["multiple"] > 70
    assert balanced["leg_share"]["revenue"] > multiple["leg_share"]["revenue"]


# ------------------------------------------------------- regime gating

def test_dip_buy_gated_in_drawdown(weekly):
    """The dip-buy rule must never fire outside a confirmed bull regime."""
    close = weekly["close"]
    deep_ath = float(close.max()) * 3        # force a large drawdown
    out = bc.dip_buy_signal(close, deep_ath, 0.8, 45.0)
    assert out["status"] == "GATED"
    assert "drawdown_shallow" in out["reason"]


def test_adaptive_bands_differ_by_regime():
    assert bc.RSI_BANDS["bull"]["oversold"] > bc.RSI_BANDS["bear"]["oversold"]


# --------------------------------------------------------- entry ladder

def test_tier3_requires_confirmation():
    broken = ind.entry_ladder(100, 140, solvency_ok=False,
                              insider_buying=False, estimate_revision_3m=-30)
    assert broken["tiers"][2]["confirmation"]["met"] is False

    healthy = ind.entry_ladder(100, 140, solvency_ok=True,
                               insider_buying=True, estimate_revision_3m=1.0)
    assert healthy["tiers"][2]["confirmation"]["met"] is True


def test_ladder_marks_tiers_already_reached():
    """Price below a tier is a different situation from waiting for it."""
    out = ind.entry_ladder(60, 140, support_levels=[118, 104, 86, 72])
    assert out["tiers"][0]["status"] == "reached"
    assert out["tiers"][2]["status"] == "reached"


def test_ladder_tiers_descend():
    out = ind.entry_ladder(100, 140)
    prices = [t["price"] for t in out["tiers"]]
    assert prices == sorted(prices, reverse=True)


# ------------------------------------------------------ indicator sanity

def test_rsi_bounds(ohlcv):
    r = ind.rsi(ohlcv["close"]).dropna()
    assert r.between(0, 100).all()


def test_independence_flags_redundancy(ohlcv):
    """
    The whole point of the panel grouping: momentum indicators are not
    independent. If this ever reports 5 of 5, the report is broken.
    """
    p = ind.stock_panel(ohlcv)
    assert p["independence"]["effective_signals"] < 5


def test_mvrv_and_oscillator_agree_by_construction():
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    rng = np.random.default_rng(2)
    price = pd.Series(60000 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))), index=idx)
    realized = price.rolling(60, min_periods=1).mean()
    osc = ind.realized_price_oscillator(price, realized)
    mvrv = price / realized - 1
    assert abs(float(np.corrcoef(osc.dropna(), (mvrv * 100).dropna())[0, 1]) - 1) < 1e-9


def test_atr_is_positive(ohlcv):
    a = ind.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
    assert (a > 0).all()


# -------------------------------------------------------------- alerts

def test_alerts_do_not_refire_on_unchanged_state(tmp_path, monkeypatch):
    """The single most important alerting invariant."""
    import alerts

    monkeypatch.setattr(alerts, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(alerts, "WEBHOOK", "")

    prev = {"entry": {"status": "GATED"}, "divergence": {}, "cycle": {}}
    cur = {"entry": {"status": "TRIGGERED", "regime_label": "bull",
                     "weekly_rsi": 54.0,
                     "daily": {"rsi": 38.0, "bands": {"oversold": 40},
                               "days_oversold": 2, "distance_to_oversold": -2.0,
                               "bullish_divergence": None}},
           "divergence": {}, "cycle": {"alerts": []}}

    assert alerts.send_btc(cur, prev, dry_run=True) >= 1
    assert alerts.send_btc(cur, cur, dry_run=True) == 0, "re-fired on unchanged state"
    assert alerts.send_btc(cur, cur, dry_run=True) == 0


def test_digest_is_biweekly_and_sunday_only():
    import alerts

    sundays = [date(2026, 9, 6) + timedelta(weeks=i) for i in range(10)]
    flags = [alerts.digest_due(d) for d in sundays]
    assert sum(flags) == 5, "exactly every other Sunday over a 10-week span"
    for a, b in zip(flags, flags[1:]):
        assert a != b, "digest weeks must alternate"
    assert not alerts.digest_due(date(2026, 9, 9)), "must not fire midweek"

    # The reason for the fixed-epoch approach: isocalendar().week % 2 skips a
    # fortnight across a 53-week ISO year (2026 is one), because weeks 53 and 1
    # are both odd. Walk every Sunday to 2032 and assert a clean 2-week cadence.
    d, last, gaps = date(2026, 1, 4), None, []
    while d < date(2032, 1, 4):
        if alerts.digest_due(d):
            if last:
                gaps.append((d - last).days // 7)
            last = d
        d += timedelta(weeks=1)
    assert set(gaps) == {2}, f"cadence drifted: gaps seen {sorted(set(gaps))}"


def test_digest_stays_inside_slack_block_limit():
    import alerts

    engines = {k: {"rows": [{"symbol": f"T{i}", "score": 90 - i} for i in range(10)]}
               for k in ("darkpool", "dividend", "recovery", "value", "politicians")}
    payload = alerts.digest_blocks(engines, alerts.compute_deltas(engines, {}))
    blocks = payload["attachments"][0]["blocks"]
    assert len(blocks) <= 50, "Slack caps a message at 50 blocks"
    for b in blocks:
        if b["type"] == "section" and "text" in b:
            assert len(b["text"]["text"]) <= 3000, "Slack caps a text object at 3000 chars"


# ---------------------------------------------------------- probit sanity

def test_probit_monotonic_and_bounded():
    probs = [rec.curve_probit(s)["probability"] for s in (-1.0, -0.5, 0.0, 0.5, 1.5)]
    assert probs == sorted(probs, reverse=True), "inversion must raise probability"
    assert all(0 <= p <= 100 for p in probs)


def test_composite_returns_two_scores_not_one():
    out = rec.composite(
        {"curve_probit": 60, "uninversion_clock": 70, "quality_dispersion": 10,
         "claims_trend": 40, "lei_trend": 50},
        {"oas_level": 0, "oas_momentum": 5, "sahm": 20, "financial_conditions": 15})
    assert "lead_score" in out and "stress_score" in out and "divergence" in out
    assert not math.isclose(out["lead_score"], out["stress_score"])


# ------------------------------------------------- free sources (offline)
# Network is unavailable in CI for these hosts, so these test the parsing and
# point-in-time logic against fixtures. The HTTP paths need a live smoke test.

from engines import free_sources as fs


def _facts_fixture():
    """Two filings for FY2023: an original and a later restatement."""
    return {"entityName": "Test Co", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2022-12-31", "filed": "2023-02-15", "fy": 2022, "fp": "FY",
             "form": "10-K", "val": 900},
            {"end": "2023-12-31", "filed": "2024-02-20", "fy": 2023, "fp": "FY",
             "form": "10-K", "val": 1000},
            {"end": "2023-12-31", "filed": "2025-02-18", "fy": 2023, "fp": "FY",
             "form": "10-K", "val": 1050},          # restated a year later
            {"end": "2023-03-31", "filed": "2023-05-01", "fy": 2023, "fp": "Q1",
             "form": "10-Q", "val": 240},
        ]}}}}}


def test_point_in_time_excludes_unfiled_facts():
    """The whole reason to use EDGAR over a cheap API."""
    f = _facts_fixture()
    # On 1 Jan 2024 the FY2023 10-K had not been filed yet.
    early = fs.extract_series(f, "revenue", as_of=date(2024, 1, 1))
    assert 2023 not in set(early["fy"]), "leaked a fact filed after as_of"
    assert 2022 in set(early["fy"])

    later = fs.extract_series(f, "revenue", as_of=date(2024, 6, 1))
    assert 2023 in set(later["fy"])


def test_first_reported_value_wins_over_restatement():
    """Backtests must see what the market saw, not the rewritten number."""
    df = fs.extract_series(_facts_fixture(), "revenue")
    row = df[df["fy"] == 2023].iloc[0]
    assert row["val"] == 1000, "used the restated value instead of first-reported"


def test_annual_only_filters_quarters():
    df = fs.extract_series(_facts_fixture(), "revenue", annual_only=True)
    assert len(df) == 2 and set(df["fy"]) == {2022, 2023}


def test_tag_fallback_chain():
    """Companies switch XBRL tags; the chain must find whichever is present."""
    alt = {"facts": {"us-gaap": {"SalesRevenueNet": {"units": {"USD": [
        {"end": "2020-12-31", "filed": "2021-02-10", "fy": 2020, "fp": "FY",
         "form": "10-K", "val": 500}]}}}}}
    df = fs.extract_series(alt, "revenue")
    assert len(df) == 1 and df.iloc[0]["tag"] == "SalesRevenueNet"


def test_missing_field_returns_empty_frame_not_exception():
    df = fs.extract_series({"facts": {"us-gaap": {}}}, "revenue")
    assert df.empty and list(df.columns)


def test_q4_derivation():
    """Q4 is never filed — it is FY minus the first three quarters."""
    annual = pd.Series({pd.Timestamp("2023-12-31"): 1000})
    quarterly = pd.Series({pd.Timestamp("2023-03-31"): 240,
                           pd.Timestamp("2023-06-30"): 250,
                           pd.Timestamp("2023-09-30"): 260})
    assert fs.derive_q4(annual, quarterly).iloc[0] == 250


def test_lei_substitute_is_neutral_not_invented():
    assert fs.load_lei_score() == 50.0


def test_scores_stay_bounded(monkeypatch):
    claims = pd.Series(range(300_000, 300_000 + 40 * 1000, 1000),
                       index=pd.date_range("2026-01-01", periods=40, freq="W"))
    assert 0 <= fs.claims_trend_score(claims) <= 100
    for v in (-2.0, 0.0, 3.0):
        assert 0 <= fs.nfci_score(pd.Series([v])) <= 100


# ------------------------------ regressions from the first live smoke test

def test_uninversion_bool_dtype_not_object():
    """
    Live bug: pos.shift(1).fillna(False) upcasts bool -> object, after which
    ~True is Python's integer bitwise-not (-2, truthy) and almost every day
    reads as a curve crossing. It flagged 10,487 of 12,566 observations.
    """
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    vals = [1.0] * 100 + [-1.0] * 200 + [1.0] * 100      # exactly ONE uninversion
    s = pd.Series(vals, index=idx)
    pos = s > 0

    good = pos & ~pos.shift(1, fill_value=False)
    assert good.sum() == 2, "one crossing at the start, one true uninversion"

    with pytest.warns(DeprecationWarning):
        # Python itself flags this: "Bitwise inversion '~' on bool is
        # deprecated ... usually not what you expect from negating a bool."
        bad = pos & ~pos.shift(1).fillna(False)
    assert bad.sum() > 100, "the buggy form should over-trigger — guarding the fix"
    assert good.sum() != bad.sum()


def test_finra_trailer_row_is_dropped():
    """The daily file ends with a bare record-count line."""
    raw = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
           "20260910|AAPL|1000.0|0.0|2500.0|Q\n"
           "20260910|MSFT|900.0|0.0|1800.0|Q\n"
           "Records: 12326\n")
    df = pd.read_csv(io.StringIO(raw), sep="|")
    df = df[df["Symbol"].notna() & (df["Symbol"] != "")]
    df = df[df["Date"].astype(str).str.fullmatch(r"\d{8}")]
    assert len(df) == 2, "trailer line leaked into the frame"
    assert pd.to_numeric(df["TotalVolume"]).sum() == 4300


def test_load_ptrs_raises_rather_than_returning_empty():
    """Stock Watcher is dead; silence would read as 'no congressional trades'."""
    with pytest.raises(NotImplementedError):
        fs.load_ptrs()


def test_load_ohlcv_raises_when_every_symbol_fails(monkeypatch):
    monkeypatch.setattr(fs, "equity_ohlcv",
                        lambda s: (_ for _ in ()).throw(RuntimeError("blocked")))
    with pytest.raises(RuntimeError, match="source is down"):
        fs.load_ohlcv(["AAPL", "MSFT"])


def test_load_ohlcv_tolerates_partial_failure(monkeypatch):
    def one_bad(sym):
        if sym == "BAD":
            raise ValueError("no data")
        return pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-09-10"]))
    monkeypatch.setattr(fs, "equity_ohlcv", one_bad)
    out = fs.load_ohlcv(["AAPL", "BAD"])
    assert set(out) == {"AAPL"}


def test_predecessor_cik_is_consulted_for_empty_registrant():
    assert fs.PREDECESSOR_CIK["XOM"] == 34088
    empty = {"facts": {"us-gaap": {}}}
    assert not fs._has_annual_facts(empty)
    good = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"fp": "FY", "end": "2024-12-31", "filed": "2025-02-01", "val": 1}]}}}}}
    assert fs._has_annual_facts(good)


# ---------------------- regressions from the second live smoke test

def test_finra_keeps_ticker_NA():
    """
    'NA' is a real NYSE ticker. pandas' default NA list coerces it to NaN and
    the notna() filter then drops it — silently, every day.
    """
    raw = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
           "20260910|NA|500.0|0.0|1200.0|Q\n"
           "20260910|AAPL|1000.0|0.0|2500.0|Q\n"
           "Records: 2\n")

    naive = pd.read_csv(io.StringIO(raw), sep="|")
    naive = naive[naive["Symbol"].notna() & (naive["Symbol"] != "")]
    assert "NA" not in set(naive["Symbol"]), "guard: the old form ate it"

    fixed = pd.read_csv(io.StringIO(raw), sep="|", keep_default_na=False)
    fixed = fixed[fixed["Symbol"].notna() & (fixed["Symbol"] != "")]
    fixed = fixed[fixed["Date"].astype(str).str.fullmatch(r"\d{8}")]
    assert set(fixed["Symbol"]) == {"NA", "AAPL"}
    assert len(fixed) == 2, "trailer must still be dropped"


def test_coinmetrics_truncation_is_caught(monkeypatch):
    """
    A 7-day response to a 2013 request is truncation, not an answer. HTTP 200
    is not evidence of completeness, and a short series propagates as NaN
    through every rolling window before surfacing as an unrelated error.
    """
    seven = [{"time": f"2026-09-{d:02d}T00:00:00Z", "PriceUSD": "77000"}
             for d in range(1, 8)]

    class R:
        @staticmethod
        def json():
            return {"data": seven}

    monkeypatch.setattr(fs, "_get", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="community-tier window cap"):
        fs.coinmetrics(["PriceUSD"])


def test_reference_rate_is_swapped_for_price_usd(monkeypatch):
    captured = {}

    class R:
        @staticmethod
        def json():
            return {"data": [{"time": f"2020-01-{(i % 28) + 1:02d}T00:00:00Z",
                              "PriceUSD": "100"} for i in range(400)]}

    def fake(url, headers=None, **kw):
        captured["metrics"] = kw.get("params", {}).get("metrics")
        return R()

    monkeypatch.setattr(fs, "_get", fake)
    fs.coinmetrics(["ReferenceRateUSD"])
    assert captured["metrics"] == "PriceUSD"


def test_rate_limit_backoff_matches_the_window_per_host():
    """
    3s against a 60s limiter fails every backfill loop. But ONE global 62s
    backoff is the opposite error: SEC allows ~10 requests per second and
    throttles briefly, so 62s there is pure dead time. Applied globally across
    4,500 SEC requests it turned a 45-minute run into seven hours.
    """
    assert fs._RATE_LIMIT_WAIT["api.polygon.io"] >= 60, "Polygon needs minutes"
    assert fs._RATE_LIMIT_WAIT["data.sec.gov"] <= 5, "SEC needs seconds"
    assert 0 < fs._DEFAULT_RATE_WAIT < 60


def test_sector_and_sic_share_one_request():
    """They fetched the same endpoint separately: 3,000 requests for 1,500."""
    assert "company_submissions" in fs.company_sic.__code__.co_names
    assert "company_sic" in fs.company_sector.__code__.co_names


def test_dispersion_is_weighted_into_lead_not_stress():
    """
    Dispersion leads headline HY OAS, so averaging it with coincident stress
    measures buries it by construction — whenever it is extreme, the others
    MUST still be calm.
    """
    cfg = rec.Config()
    assert "quality_dispersion" in cfg.lead_weights
    assert "quality_dispersion" not in cfg.stress_weights
    assert abs(sum(cfg.lead_weights.values()) - 1.0) < 1e-9
    assert abs(sum(cfg.stress_weights.values()) - 1.0) < 1e-9


def test_extreme_dispersion_is_not_diluted():
    """The live case: 99.9th percentile must not read as an unremarkable score."""
    lead = {"curve_probit": 31.6, "uninversion_clock": 41.7,
            "quality_dispersion": 99.9, "claims_trend": 42.0, "lei_trend": 50.0}
    stress = {"oas_level": 0.0, "oas_momentum": 2.9, "sahm": 20.0,
              "financial_conditions": 15.0}
    disp = {"dispersion": 9.06, "percentile_in_window": 99.9,
            "change_3m": 1.8, "widening": True}
    out = rec.composite(lead, stress, disp)
    assert out["credit_early_warning"] is not None
    assert out["lead_score"] > 45, "extreme dispersion must move the lead score"
    assert out["divergence"] > 30


def test_early_warning_needs_widening_not_just_level():
    """A high but compressing dispersion is a late signal, not an early one."""
    lead = {"curve_probit": 30, "uninversion_clock": 40,
            "quality_dispersion": 95, "claims_trend": 40, "lei_trend": 50}
    stress = {"oas_level": 0, "oas_momentum": 0, "sahm": 10,
              "financial_conditions": 10}
    compressing = {"dispersion": 9.0, "percentile_in_window": 95,
                   "change_3m": -1.2, "widening": False}
    assert rec.composite(lead, stress, compressing)["credit_early_warning"] is None


# -------------------- fundamentals builder (third live smoke test)

from engines import fundamentals_builder as fb


def _fact(tag, rows, instant=False, unit="USD"):
    items = []
    for end, val in rows:
        it = {"end": end, "filed": f"{int(end[:4]) + 1}-02-15",
              "fy": int(end[:4]), "fp": "FY", "form": "10-K", "val": val}
        if not instant:
            it["start"] = f"{int(end[:4])}-01-01"
        items.append(it)
    return {tag: {"units": {unit: items}}}


def test_fp_FY_alone_does_not_mean_annual():
    """
    The 5x error. A 10-K tags its quarterly comparatives fp="FY", form="10-K",
    so filtering on fp alone mixes 90-day and 365-day periods in one series and
    a CAGR compares a quarter against a year.
    """
    items = [
        {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-15",
         "fy": 2024, "fp": "FY", "form": "10-K", "val": 416_000},
        {"start": "2024-10-01", "end": "2024-12-31", "filed": "2025-02-15",
         "fy": 2024, "fp": "FY", "form": "10-K", "val": 53_000},   # the trap
    ]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": items}}}}}

    df = fs.extract_series(facts, "revenue", annual_only=True)
    assert len(df) == 1, "quarterly comparative leaked into the annual series"
    assert df.iloc[0]["val"] == 416_000

    assert fs._is_annual(items[0]) is True
    assert fs._is_annual(items[1]) is False


def test_instants_survive_the_duration_filter():
    """Balance-sheet facts have no `start` and no duration to test."""
    inst = {"end": "2024-12-31", "filed": "2025-02-15", "fy": 2024,
            "fp": "FY", "form": "10-K", "val": 100}
    assert fs._is_annual(inst) is True


def test_altman_uses_market_equity_when_available():
    """
    Original 1968 coefficients need MARKET equity in X4. Pairing them with book
    value understates Z badly — enough to read a healthy company as distressed.
    """
    p = {"ta": 320e9, "tl": 200e9, "wc": 10e9, "re": 40e9,
         "ebit": 60e9, "sales": 200e9, "book_equity": 120e9}
    with_mkt = fb._altman(p, market_equity=3889e9)
    private = fb._altman(p, market_equity=None)
    assert with_mkt > 2.99, "healthy firm must clear the original safe threshold"
    assert with_mkt > private, "market equity should exceed book for this firm"

    wrong = (1.2 * p["wc"] / p["ta"] + 1.4 * p["re"] / p["ta"]
             + 3.3 * p["ebit"] / p["ta"] + 0.6 * p["book_equity"] / p["tl"]
             + 1.0 * p["sales"] / p["ta"])
    assert wrong < 2.0, "guard: the mixed formula reads as distress"


def test_dividend_record_detects_a_cut():
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2016, 2026)])
    rising = pd.Series([1.0 + 0.1 * i for i in range(10)], index=idx)
    assert fb.dividend_record(rising)["streak"] == 9
    assert fb.dividend_record(rising)["years_since_cut"] == 99

    cut = rising.copy()
    cut.iloc[7] = cut.iloc[6] * 0.5          # 3M in 2024
    rec = fb.dividend_record(cut)
    assert rec["years_since_cut"] <= 3, "a cut inside 10y must be visible"
    assert rec["streak"] < 9


def test_builder_fills_the_quality_gate_fields():
    """
    The regression that matters: load_fundamentals used to fill 1 of 44 fields,
    so every screen returned zero rows for reasons unrelated to the companies.
    """
    yrs = [f"{2016 + i}-12-31" for i in range(10)]
    rev = [(y, 200e9 * 1.087 ** i) for i, y in enumerate(yrs)]
    us = {}
    for tag, mult in [("GrossProfit", .44), ("OperatingIncomeLoss", .30),
                      ("NetIncomeLoss", .24),
                      ("NetCashProvidedByUsedInOperatingActivities", .28),
                      ("PaymentsToAcquirePropertyPlantAndEquipment", .04),
                      ("DepreciationDepletionAndAmortization", .03),
                      ("InterestExpense", .01)]:
        us.update(_fact(tag, [(y, v * mult) for y, v in rev]))
    us.update(_fact("Revenues", rev))
    for tag, mult in [("Assets", 1.6), ("StockholdersEquity", .35),
                      ("CashAndCashEquivalentsAtCarryingValue", .15),
                      ("LongTermDebt", .30), ("AssetsCurrent", .5),
                      ("LiabilitiesCurrent", .45),
                      ("RetainedEarningsAccumulatedDeficit", .2)]:
        us.update(_fact(tag, [(y, v * mult) for y, v in rev], instant=True))
    us.update(_fact("CommonStockSharesOutstanding",
                    [(y, 16e9 * 0.97 ** i) for i, (y, _) in enumerate(rev)], instant=True))
    # A dividend payer, so the dividend fields are exercised too. Without
    # these the builder legitimately leaves them at default and coverage
    # lands near 58%.
    dps = [(y, 0.60 * 1.09 ** i) for i, (y, _) in enumerate(rev)]
    us.update(_fact("CommonStockDividendsPerShareDeclared", dps, unit="USD/shares"))
    us.update(_fact("PaymentsOfDividendsCommonStock", [(y, d * 16e9) for y, d in dps]))
    us.update(_fact("EarningsPerShareDiluted",
                    [(y, v * 0.24 / 16e9) for y, v in rev], unit="USD/shares"))
    facts = {"entityName": "Test Co", "facts": {"us-gaap": us}}

    # EDGAR alone: enough for every quality gate, not for history-relative
    # scoring. Measured, not assumed — ~32% of fields.
    f = fb.build("TEST", facts)
    cov = fb.coverage_report(f)
    assert cov["pct"] > 28, f"only {cov['pct']}% populated from EDGAR alone"

    # Adding a price series roughly doubles it: every "vs its own 5-10y
    # history" measure needs the denominator.
    rng = np.random.default_rng(4)
    idx = pd.date_range("2016-01-01", periods=2600, freq="B")
    px = pd.DataFrame(
        {"close": 120 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(idx)))),
         "volume": rng.lognormal(16, .3, len(idx))}, index=idx)
    cov_px = fb.coverage_report(fb.build("TEST", facts, px))
    assert cov_px["pct"] > 70, f"with prices only {cov_px['pct']}%"
    assert cov_px["filled"] > cov["filled"] + 8

    assert 8.0 < f.revenue_cagr_5y < 9.5, f"CAGR {f.revenue_cagr_5y} — duration bug?"
    for name in ("gross_margin", "fcf_margin", "roic_5y", "net_debt_ebitda",
                 "interest_coverage", "altman_z"):
        assert getattr(f, name) != 0.0, f"{name} left at default"

    from engines import screeners as scr
    gates = scr.quality_gates(f)
    assert not any("ROIC" in g and "0.0" in g for g in gates), \
        "gate failing on an unpopulated field rather than on the company"


def test_coverage_report_counts_defaults():
    blank = sc.Fundamentals(symbol="X", name="X")
    assert fb.coverage_report(blank)["filled"] <= 2


# ------------------- fourth live smoke test: five real tickers

def test_ev_ebit_is_not_just_the_price_percentile():
    """
    The bug: ev_ebit was (close * shares_today) / ebit_today — a constant times
    price — so its percentile equalled the raw close percentile to machine
    precision across all four tickers checked. It carried ~30% of the quality
    score while measuring "near its high".
    """
    rng = np.random.default_rng(9)
    idx = pd.date_range("2016-01-01", periods=2600, freq="B")
    close = 120 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, len(idx))))
    px = pd.DataFrame({"close": close, "volume": rng.lognormal(16, .3, len(idx))},
                      index=idx)
    yrs = pd.to_datetime([f"{2016 + i}-12-31" for i in range(10)])
    # EBIT accelerates in the back half, so the multiple must COMPRESS even as
    # price rises. A price-tracking measure cannot show that.
    ebit = pd.Series([20e9 * (1.05 ** i if i < 5 else 1.05 ** 5 * 1.28 ** (i - 5))
                      for i in range(10)], index=yrs)
    out = fb.price_context(
        px, 5e9, pd.Series(dtype=float), 20e9, 24e9, 8e9, 60e9,
        ebit_hist=ebit,
        shares_hist=pd.Series([5e9 * 0.98 ** i for i in range(10)], index=yrs),
        net_debt_hist=pd.Series([10e9] * 10, index=yrs),
        revenue_hist=pd.Series([60e9 * 1.08 ** i for i in range(10)], index=yrs))

    s = pd.Series(close, index=idx)
    price_pct = float((s < s.iloc[-1]).mean() * 100)
    assert abs(out["ev_ebit_percentile_10y"] - price_pct) > 5, \
        "EV/EBIT percentile still tracks price — the scalar bug is back"
    assert out["_ev_hist_points"] > 1000


def test_fundamentals_are_lagged_to_filing_date():
    """Using a FY figure before it was filed is lookahead."""
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    annual = pd.Series([100.0], index=pd.to_datetime(["2023-12-31"]))
    aligned = fb._align(annual, idx, lag_days=75)
    assert pd.isna(aligned.loc[pd.Timestamp("2024-01-15")]), \
        "FY2023 was not public in January 2024"
    assert aligned.loc[pd.Timestamp("2024-06-03")] == 100.0


def test_splits_must_be_authoritative_never_inferred():
    """
    A 2:1 split and a 50% dividend cut are identical in a DPS series. The ratio
    heuristic that was tried explained away 3M's real 2024 halving.
    """
    mmm = pd.Series([4.10, 4.70, 5.44, 5.76, 5.88, 5.92, 5.96, 6.00, 2.86, 2.90],
                    index=pd.to_datetime([f"{y}-12-31" for y in range(2016, 2026)]))
    rec = fb.dividend_record(fb.split_adjust(mmm, pd.Series(dtype=float)))
    assert rec["years_since_cut"] <= 2, "a genuine cut was adjusted away"
    assert rec["split_ambiguous"] is True, "unexplained drop must be flagged"


def test_authoritative_splits_clear_the_dividend_record():
    idx = pd.to_datetime([f"{y}-09-30" for y in range(2018, 2026)])
    dps = pd.Series([2.40, 2.72, 0.795, 0.85, 0.90, 0.94, 0.98, 1.02], index=idx)
    splits = pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))
    rec = fb.dividend_record(fb.split_adjust(dps, splits))
    assert rec["split_ambiguous"] is False
    assert rec["years_since_cut"] == 99, "split wrongly read as a cut"


def test_ambiguous_dividend_record_fails_the_gate():
    """Cannot-verify must exclude, not silently pass."""
    f = sc.Fundamentals(symbol="X", name="X", increase_streak_years=20,
                        years_since_cut=99, dps_cagr_5y=8.0, eps_payout=40,
                        fcf_payout=50, net_debt_ebitda=1.5, interest_coverage=12,
                        revenue_cagr_5y=5, market_cap=5e10, dollar_adv=1e8)
    assert not sc.dividend_gates(f)
    f.dividend_record_ambiguous = True
    assert any("unverifiable" in g for g in sc.dividend_gates(f))


def test_tag_chain_picks_the_richest_not_the_first():
    """J&J: 1 fact under Declared, 51 under CashPaid. Breaking early took the 1."""
    one = [{"end": "2021-12-31", "start": "2021-01-01", "filed": "2022-02-15",
            "fy": 2021, "fp": "FY", "form": "10-K", "val": 4.19}]
    many = [{"end": f"{y}-12-31", "start": f"{y}-01-01", "filed": f"{y + 1}-02-15",
             "fy": y, "fp": "FY", "form": "10-K", "val": 1.0 + 0.1 * (y - 2015)}
            for y in range(2015, 2026)]
    facts = {"facts": {"us-gaap": {
        "CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": one}},
        "CommonStockDividendsPerShareCashPaid": {"units": {"USD/shares": many}}}}}
    df = fs.extract_series(facts, "dividends_per_share")
    assert len(df) == len(many), f"took the sparse tag: {len(df)} rows"
    assert df.iloc[0]["tag"] == "CommonStockDividendsPerShareCashPaid"


def test_missing_ebit_does_not_read_as_maximum_safety():
    """`or 99.0` turned a zero EBIT into interest coverage of 99."""
    yrs = [f"{2016 + i}-12-31" for i in range(6)]
    rev = [(y, 300e9) for y in yrs]
    us = {}
    us.update(_fact("Revenues", rev))
    us.update(_fact("InterestExpense", [(y, 1e9) for y, _ in rev]))
    us.update(_fact("Assets", [(y, 400e9) for y, _ in rev], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 200e9) for y, _ in rev], instant=True))
    f = fb.build("OILCO", {"entityName": "Oil Co", "facts": {"us-gaap": us}})
    assert f.ebit_unavailable is True
    assert f.interest_coverage == 0.0, "unknown coverage must not read as safe"
    # Excluded via the honest data-quality reason. The coverage gate itself is
    # now skipped, because reporting "coverage 0.0x under 4x" beside "no
    # derivable EBIT" puts a business label next to the real cause.
    gates = sc.dividend_gates(f)
    assert any("no derivable EBIT" in g for g in gates)
    assert not any("interest coverage" in g for g in gates)


def test_ebit_derived_from_pretax_when_operating_income_absent():
    yrs = [f"{2016 + i}-12-31" for i in range(6)]
    rev = [(y, 300e9) for y in yrs]
    us = {}
    us.update(_fact("Revenues", rev))
    us.update(_fact("InterestExpense", [(y, 2e9) for y, _ in rev]))
    us.update(_fact(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
        "NoncontrollingInterest", [(y, 30e9) for y, _ in rev]))
    us.update(_fact("Assets", [(y, 400e9) for y, _ in rev], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 200e9) for y, _ in rev], instant=True))
    f = fb.build("OILCO", {"entityName": "Oil Co", "facts": {"us-gaap": us}})
    assert f.ebit_unavailable is False
    assert f.interest_coverage > 10


def test_stale_data_is_gated():
    """XOM's predecessor CIK stopped filing in 2021; 55% 'growth' was 2020 vs 2021."""
    f = sc.Fundamentals(symbol="XOM", name="X", increase_streak_years=20,
                        years_since_cut=99, dps_cagr_5y=8.0, eps_payout=40,
                        fcf_payout=50, net_debt_ebitda=1.5, interest_coverage=12,
                        revenue_cagr_5y=5, market_cap=5e11, dollar_adv=1e9)
    assert not sc.dividend_gates(f)
    f.data_stale_days = 1700
    assert any("days old" in g for g in sc.dividend_gates(f))


def test_roic_gate_does_not_disqualify_excellent_mean_reversion():
    """MSFT went 33.0 -> 27.1 with EBIT +17%. That is not erosion."""
    good = sc.Fundamentals(
        symbol="MSFT", name="M", roic_5y=30.2, roic_ttm=27.1,
        roic_declining_years=4, wacc=9.0, gross_margin=68, fcf_margin=30,
        net_debt_ebitda=0.3, fcf_positive_years_of_10=10,
        share_count_cagr_5y=-0.5, revenue_cagr_5y=14)
    assert not any("eroding" in g for g in sc.quality_gates(good))

    eroding = sc.Fundamentals(
        symbol="BAD", name="B", roic_5y=22.0, roic_ttm=11.0,
        roic_declining_years=4, wacc=9.0, gross_margin=40, fcf_margin=9,
        net_debt_ebitda=2.0, fcf_positive_years_of_10=9,
        share_count_cagr_5y=0.0, revenue_cagr_5y=5)
    assert any("eroding" in g for g in sc.quality_gates(eroding))


# ------------------- fifth live smoke test: the fix that broke three tickers

def test_tag_chain_stitches_across_asc606_migration():
    """
    'Richest tag by count' was worse than the bug it fixed. Most large filers
    migrated revenue tags at ASC 606 in 2018, so the deprecated tag carries a
    LONGER history that stops dead at the migration — 11 periods ending 2017
    beat 9 ending 2025, and Apple's revenue series silently ended eight years
    ago. Tags in a chain are the same concept, so the answer is both, stitched.
    """
    def rows(yrs, base):
        return [{"end": f"{y}-09-30", "start": f"{y - 1}-10-01",
                 "filed": f"{y}-11-01", "fy": y, "fp": "FY", "form": "10-K",
                 "val": base * (1.08 ** (y - 2012))} for y in yrs]

    facts = {"facts": {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": rows(range(2007, 2018), 100e9)}},
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            {"units": {"USD": rows(range(2017, 2026), 100e9)}}}}}

    df = fs.extract_series(facts, "revenue")
    ends = pd.to_datetime(df["end"])
    assert ends.max().year == 2025, "series froze at the tag migration"
    assert ends.min().year == 2007, "legacy history was discarded"
    assert len(df) == 19, "overlap year should dedupe"
    assert len(df.attrs.get("tags_used", [])) == 2


def test_ratios_use_a_common_period():
    """
    Apple's gross margin read 85.15% — FY2025 gross profit over FY2017 revenue.
    Taking _last() of each series independently is the whole bug.
    """
    rev = pd.Series([100.0, 110.0],
                    index=pd.to_datetime(["2016-12-31", "2017-12-31"]))
    gp = pd.Series([44.0, 48.0, 95.0],
                   index=pd.to_datetime(["2016-12-31", "2017-12-31", "2025-12-31"]))
    val, stale = fb._ratio(gp, rev, as_of=date(2018, 6, 1))
    assert abs(val - 43.6) < 0.5, "ratio crossed periods"
    assert stale is False

    naive = float(gp.iloc[-1]) / float(rev.iloc[-1]) * 100
    assert naive > 80, "guard: the old form gives a nonsense margin"


def test_period_mismatch_is_measured_and_gated():
    a = pd.Series([1.0], index=pd.to_datetime(["2025-12-31"]))
    b = pd.Series([1.0], index=pd.to_datetime(["2014-12-31"]))
    assert fb._period_mismatch_days(a, b) > 400

    f = sc.Fundamentals(symbol="X", name="X", roic_5y=20, roic_ttm=20,
                        gross_margin=50, fcf_margin=15, net_debt_ebitda=1.0,
                        fcf_positive_years_of_10=10, share_count_cagr_5y=-1,
                        revenue_cagr_5y=8, wacc=9)
    assert not sc.quality_gates(f)
    f.period_mismatch_days = 4000
    assert any("cross-period" in g for g in sc.quality_gates(f))


def test_stale_flags_gate_every_screen_not_just_dividends():
    """
    data_stale_days and ebit_unavailable were set by the builder and read only
    by dividend_gates, so quality and recovery ranked names whose revenue
    series stopped in 2017 — Coca-Cola failed on 'revenue declining' purely as
    an artifact.
    """
    f = sc.Fundamentals(symbol="X", name="X", data_stale_days=3268)
    for gates in (sc.quality_gates, sc.recovery_gates, sc.dividend_gates):
        assert any("days old" in g for g in gates(f)), gates.__name__

    g = sc.Fundamentals(symbol="Y", name="Y", ebit_unavailable=True)
    for gates in (sc.quality_gates, sc.recovery_gates, sc.dividend_gates):
        assert any("EBIT" in x for x in gates(g)), gates.__name__


def test_stale_ebit_does_not_silently_restore_the_price_percentile():
    """
    The most dangerous regression in this project: an unlimited forward-fill
    carried J&J's single 2014 EBIT across the whole window, making the
    denominator constant, so EV/EBIT became a constant times price and its
    percentile matched the close percentile to two decimal places. The degraded
    output was indistinguishable from the fixed one.
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2016-01-01", periods=2600, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.014, len(idx))))
    px = pd.DataFrame({"close": close, "volume": rng.lognormal(16, .3, len(idx))},
                      index=idx)
    s = pd.Series(close, index=idx)
    price_pct = float((s < s.iloc[-1]).mean() * 100)

    stale = pd.Series([20e9] * 6,
                      index=pd.to_datetime([f"{y}-12-31" for y in range(2009, 2015)]))
    out = fb.price_context(px, 5e9, pd.Series(dtype=float), 20e9, 24e9, 8e9, 60e9,
                           ebit_hist=stale)
    assert out["ev_history_degraded"] is True
    assert out["ev_ebit_percentile_10y"] == 50.0, "must not rank on a constant"
    assert abs(out["ev_ebit_percentile_10y"] - price_pct) > 5

    current = pd.Series([20e9 * 1.09 ** i for i in range(11)],
                        index=pd.to_datetime([f"{2015 + i}-12-31" for i in range(11)]))
    ok = fb.price_context(px, 5e9, pd.Series(dtype=float), 20e9, 24e9, 8e9, 60e9,
                          ebit_hist=current)
    assert ok["ev_history_degraded"] is False
    assert ok["_ev_hist_points"] > 2000


def test_align_drops_values_past_the_staleness_limit():
    idx = pd.date_range("2016-01-01", periods=2600, freq="B")
    old = pd.Series([100.0], index=pd.to_datetime(["2014-12-31"]))
    aligned = fb._align(old, idx, max_stale_days=450)
    assert aligned.notna().sum() < 400, "unlimited ffill is back"


def test_degraded_ev_history_fails_the_value_screen():
    f = sc.Fundamentals(symbol="X", name="X", roic_5y=20, roic_ttm=20,
                        gross_margin=50, fcf_margin=15, net_debt_ebitda=1.0,
                        fcf_positive_years_of_10=10, share_count_cagr_5y=-1,
                        revenue_cagr_5y=8, wacc=9)
    assert not sc.quality_gates(f)
    f.ev_history_degraded = True
    assert any("too short or stale" in g for g in sc.quality_gates(f))


# --------------- sixth live smoke test: stitching and staleness

def test_dedup_tolerates_52_53_week_fiscal_drift():
    """
    J&J files on a 52/53-week calendar, so two tags report the same fiscal year
    one day apart. Exact-date dedup kept both, and the second was a QUARTERLY
    declared rate sitting beside the annual figure — read mid-series as a 3.94x
    drop and recovery, giving a 5-year streak and a 38% DPS CAGR.
    """
    cash = [{"end": f"{y}-01-03", "start": f"{y - 1}-01-04",
             "filed": f"{y}-02-20", "fy": y - 1, "fp": "FY", "form": "10-K",
             "val": 3.5 + 0.12 * (y - 2016)} for y in range(2016, 2026)]
    decl = [{"end": "2021-01-04", "start": "2020-01-06", "filed": "2021-02-20",
             "fy": 2020, "fp": "FY", "form": "10-K", "val": 1.01}]
    facts = {"facts": {"us-gaap": {
        "CommonStockDividendsPerShareCashPaid": {"units": {"USD/shares": cash}},
        "CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": decl}}}}}

    df = fs.extract_series(facts, "dividends_per_share")
    assert len(df) == 10, "the one-day-apart duplicate survived"
    assert df["val"].min() > 2.0, "quarterly rate leaked into the annual series"

    v = pd.Series(df["val"].values, index=pd.to_datetime(df["end"])).sort_index()
    rec = fb.dividend_record(v)
    assert rec["years_since_cut"] == 99, "phantom cut from the duplicate"
    assert rec["split_ambiguous"] is False


def test_ev_history_guard_checks_recency_not_just_count():
    """
    A series can be long AND entirely historical. J&J's aligned EBIT had 1,564
    valid points, every one before June 2016, so the count guard passed and
    pct_rank ranked against a series that had already ended — reporting a
    mid-2016 multiple as current.
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2016-01-01", periods=2600, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.014, len(idx))))
    px = pd.DataFrame({"close": close, "volume": rng.lognormal(16, .3, len(idx))},
                      index=idx)
    s = pd.Series(close, index=idx)
    price_pct = float((s < s.iloc[-1]).mean() * 100)

    historical = pd.Series([20e9 * 1.03 ** i for i in range(8)],
                           index=pd.to_datetime([f"{2009 + i}-12-31" for i in range(8)]))
    out = fb.price_context(px, 5e9, pd.Series(dtype=float), 20e9, 24e9, 8e9, 60e9,
                           ebit_hist=historical)
    assert out["_ev_hist_points"] > 500, "fixture must clear the COUNT guard"
    assert out["ev_history_degraded"] is True, "recency guard missed it"
    assert out["ev_ebit_percentile_10y"] == 50.0
    # None, not 0.0 — a zero renders as the cheapest possible stock.
    assert out["ev_ebit"] is None, "stale multiple reported as current"
    assert abs(out["ev_ebit_percentile_10y"] - price_pct) > 5


def test_ebit_falls_back_when_stale_not_only_when_empty():
    """
    J&J has six OperatingIncomeLoss facts ending 2014 while pretax runs to
    2025. Testing emptiness alone meant ROIC and EV/EBIT used eleven-year-old
    earnings, making the name look ~25% cheaper than it was.
    """
    yrs_old = [f"{2009 + i}-12-31" for i in range(6)]
    yrs_new = [f"{2016 + i}-12-31" for i in range(10)]
    us = {}
    us.update(_fact("Revenues", [(y, 80e9) for y in yrs_new]))
    us.update(_fact("OperatingIncomeLoss", [(y, 15e9) for y in yrs_old]))
    us.update(_fact(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
        "NoncontrollingInterest", [(y, 22e9) for y in yrs_new]))
    us.update(_fact("InterestExpense", [(y, 1e9) for y in yrs_new]))
    us.update(_fact("Assets", [(y, 180e9) for y in yrs_new], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 70e9) for y in yrs_new], instant=True))

    f = fb.build("JNJ", {"entityName": "J", "facts": {"us-gaap": us}})
    assert f.ebit_unavailable is False
    # 22 + 1 = 23bn from pretax, not the stale 15bn operating income.
    assert f.interest_coverage > 20, "used the stale OperatingIncomeLoss series"


def test_missing_gross_profit_reads_as_unknown_not_zero():
    """
    Exxon reports no GrossProfit tag — normal for oil majors. Absent data was
    surfacing as 'gross margin 0% under 35%', a business label on a data
    problem, which would reject the whole sector.
    """
    yrs = [f"{2016 + i}-12-31" for i in range(10)]
    us = {}
    us.update(_fact("Revenues", [(y, 300e9) for y in yrs]))
    us.update(_fact("OperatingIncomeLoss", [(y, 30e9) for y in yrs]))
    us.update(_fact("Assets", [(y, 400e9) for y in yrs], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 200e9) for y in yrs], instant=True))

    f = fb.build("XOM", {"entityName": "X", "facts": {"us-gaap": us}})
    assert f.gross_profit_unavailable is True

    # No longer a hard failure — EDGAR covers gross profit badly enough that
    # gating on it rejects real companies for a filing-presentation choice.
    # It must still never appear as a business failure, and it must still be
    # COUNTED.
    gates = sc.quality_gates(f)
    assert not any("gross margin 0" in g for g in gates), \
        "absent data still wearing a business label"
    rep = sc.data_quality_report(f)
    assert rep["flags"]["gross_profit_unavailable"] is True
    assert rep["any_flag"] is True


def test_gross_profit_derived_from_cost_of_revenue():
    yrs = [f"{2016 + i}-12-31" for i in range(10)]
    us = {}
    us.update(_fact("Revenues", [(y, 300e9) for y in yrs]))
    us.update(_fact("CostOfRevenue", [(y, 180e9) for y in yrs]))
    us.update(_fact("OperatingIncomeLoss", [(y, 30e9) for y in yrs]))
    us.update(_fact("Assets", [(y, 400e9) for y in yrs], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 200e9) for y in yrs], instant=True))

    f = fb.build("XOM", {"entityName": "X", "facts": {"us-gaap": us}})
    assert f.gross_profit_unavailable is False
    assert abs(f.gross_margin - 40.0) < 0.5


# --------------- seventh live smoke test: universe run

def test_period_mismatch_ignores_abandoned_series():
    """
    It measured raw `oi` even after the builder switched to the pretax
    fallback. J&J's used series all end 2025-12-28 — a true mismatch of 0 —
    but it read 4,018 and failed the gate, inflating the very data-quality
    rate the universe run is meant to measure.
    """
    new = [f"{2016 + i}-12-31" for i in range(10)]
    old = [f"{2009 + i}-12-31" for i in range(6)]
    us = {}
    us.update(_fact("Revenues", [(y, 88e9) for y in new]))
    us.update(_fact("GrossProfit", [(y, 60e9) for y in new]))
    us.update(_fact("NetCashProvidedByUsedInOperatingActivities",
                    [(y, 24e9) for y in new]))
    us.update(_fact("OperatingIncomeLoss", [(y, 15e9) for y in old]))  # abandoned
    us.update(_fact("IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                    "ExtraordinaryItemsNoncontrollingInterest",
                    [(y, 22e9) for y in new]))
    us.update(_fact("InterestExpense", [(y, 1e9) for y in new]))
    us.update(_fact("Assets", [(y, 180e9) for y in new], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 70e9) for y in new], instant=True))

    f = fb.build("JNJ", {"entityName": "J", "facts": {"us-gaap": us}})
    assert f.period_mismatch_days == 0, f"read {f.period_mismatch_days}"
    assert not any("cross-period" in g for g in sc.data_quality_gates(f))


def test_ifrs_filers_produce_a_record():
    """
    20-F filers report under ifrs-full with zero us-gaap concepts. They
    returned no revenue and dropped out before becoming a record, so they
    never appeared in any coverage or failure statistic — an invisible
    exclusion, which is worse than a counted one.
    """
    def ifrs(tag, rows, instant=False):
        return {tag: {"units": {"USD": [
            {"end": e, "filed": f"{int(e[:4]) + 1}-03-01", "fy": int(e[:4]),
             "fp": "FY", "form": "20-F", "val": v}
            | ({} if instant else {"start": f"{int(e[:4])}-01-01"})
            for e, v in rows]}}}

    yrs = [f"{2016 + i}-12-31" for i in range(10)]
    tags = {}
    tags.update(ifrs("Revenue", [(y, 10e9 * 1.06 ** i) for i, y in enumerate(yrs)]))
    tags.update(ifrs("GrossProfit", [(y, 3.1e9 * 1.06 ** i) for i, y in enumerate(yrs)]))
    tags.update(ifrs("ProfitLossFromOperatingActivities", [(y, 1.6e9) for y in yrs]))
    tags.update(ifrs("Assets", [(y, 14e9) for y in yrs], instant=True))
    facts = {"entityName": "Wipro", "facts": {"ifrs-full": tags}}

    assert fs.taxonomy_of(facts) == "ifrs-full"
    assert not fs.extract_series(facts, "revenue").empty
    f = fb.build("WIT", facts)
    assert f.revenue_cagr_5y > 0
    assert abs(f.gross_margin - 31.0) < 1.0


def test_taxonomy_detection():
    assert fs.taxonomy_of({"facts": {"us-gaap": {"x": 1}}}) == "us-gaap"
    assert fs.taxonomy_of({"facts": {"ifrs-full": {"x": 1}}}) == "ifrs-full"
    assert fs.taxonomy_of({"facts": {}}) == "unknown"


# ------------- eighth live smoke test: 207-name universe run

def test_ratio_refuses_an_ancient_common_period():
    """
    Amazon's gross profit series ends 2009, so the index intersection is 2009
    and the margin came back 22.57% — internally consistent, sixteen years old,
    and it would have rendered. Third instance of the same class, after
    _align's unlimited ffill and the count-only EV guard.
    """
    rev = pd.Series([100.0 + 10 * i for i in range(20)],
                    index=pd.to_datetime([f"{2006 + i}-12-31" for i in range(20)]))
    gp = pd.Series([22.6, 24.0, 25.4, 26.8],
                   index=pd.to_datetime([f"{2006 + i}-12-31" for i in range(4)]))

    val, stale = fb._ratio(gp, rev, as_of=date(2026, 9, 12))
    assert stale is True
    assert val == 0.0, "published a 2009 margin as current"

    fresh_gp = pd.Series([40.0, 44.0],
                         index=pd.to_datetime(["2024-12-31", "2025-12-31"]))
    val2, stale2 = fb._ratio(fresh_gp, rev, as_of=date(2026, 9, 12))
    assert stale2 is False and val2 > 0


def test_stale_gross_profit_marks_unavailable():
    yrs = [f"{2016 + i}-12-31" for i in range(10)]
    us = {}
    us.update(_fact("Revenues", [(y, 300e9) for y in yrs]))
    us.update(_fact("GrossProfit", [(f"{2008 + i}-12-31", 60e9) for i in range(3)]))
    us.update(_fact("OperatingIncomeLoss", [(y, 30e9) for y in yrs]))
    us.update(_fact("Assets", [(y, 400e9) for y in yrs], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 200e9) for y in yrs], instant=True))
    f = fb.build("AMZN", {"entityName": "A", "facts": {"us-gaap": us}})
    assert f.gross_profit_unavailable is True, "stale is not more usable than missing"
    assert f.gross_margin == 0.0


def test_implausible_derived_margin_is_rejected():
    """Verizon's revenue-minus-cost derivation gave 82.4% — a partial cost line."""
    yrs = [f"{2016 + i}-12-31" for i in range(10)]
    us = {}
    us.update(_fact("Revenues", [(y, 130e9) for y in yrs]))
    us.update(_fact("CostOfRevenue", [(y, 22e9) for y in yrs]))   # partial
    us.update(_fact("OperatingIncomeLoss", [(y, 30e9) for y in yrs]))
    us.update(_fact("Assets", [(y, 380e9) for y in yrs], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 90e9) for y in yrs], instant=True))
    f = fb.build("VZ", {"entityName": "V", "facts": {"us-gaap": us}})
    assert f.gross_profit_unavailable is True, "published an 83% telco margin"


def test_period_mismatch_never_returns_a_sentinel():
    """MPT rendered 99999 — a sentinel in a field the UI prints as days."""
    a = pd.Series([1.0], index=pd.to_datetime(["2025-12-31"]))
    b = pd.Series([1.0], index=pd.to_datetime(["2014-12-31"]))
    d = fb._period_mismatch_days(a, b)
    assert d != 99999 and 0 < d < 20000


def test_data_quality_report_counts_screen_specific_flags():
    """
    A universe run counted 35.3% from shared gates while 46.5% carried a
    degraded EV history and 23.5% an unverifiable dividend record — invisible
    in the headline because those gate only one screen each.
    """
    f = sc.Fundamentals(symbol="X", name="X")
    f.ev_history_degraded = True
    rep = sc.data_quality_report(f)
    assert rep["flags"]["ev_history_degraded"] is True
    assert rep["any_flag"] is True
    assert rep["hard_failed"] is False, "must not become a shared hard failure"
    # It IS a hard exclusion for the value screen, so it is not "degraded
    # only" — that reading undercounted a data reason by 44% of the universe.
    assert rep["degraded_only"] is False
    assert rep["hard_by_screen"]["quality"]


def test_unavailable_gross_margin_scores_neutral_not_bad():
    base = dict(roic_5y=20.0, roic_ttm=20.0, wacc=9.0, fcf_margin=15.0,
                share_count_cagr_5y=-1.0, revenue_cagr_5y=8.0,
                ev_ebit=12.0, ev_ebit_median_10y=15.0)
    have = sc.score_quality_value(sc.Fundamentals(symbol="A", name="A",
                                                  gross_margin=55.0, **base))
    missing = sc.score_quality_value(sc.Fundamentals(
        symbol="B", name="B", gross_profit_unavailable=True, **base))
    poor = sc.score_quality_value(sc.Fundamentals(symbol="C", name="C",
                                                  gross_margin=0.0, **base))
    # Absent must not be treated as zero...
    assert missing["components"]["quality"] > poor["components"]["quality"], \
        "absent gross margin scored as if it were 0%"
    # ...nor as a bonus. Substituting a neutral 50 made missing data score
    # BETTER than a real 55% margin, because _scale(55, 35, 85) is only 40.
    strong = sc.score_quality_value(sc.Fundamentals(symbol="D", name="D",
                                                    gross_margin=80.0, **base))
    assert missing["components"]["quality"] < strong["components"]["quality"], \
        "absent gross margin outscored an excellent one"


# ------------- ninth live smoke test: split-adjusted share counts

def test_share_counts_are_split_adjusted():
    """
    split_adjust was applied to dividends but never to share counts, though
    splits were already being fetched. NVIDIA steps 628m -> 2.535bn across the
    2021 4:1 and 2.494bn -> 24.804bn across the 2024 10:1, so
    share_count_cagr_5y read +108% a year. The gate on it was the single
    largest business exclusion — 126 of 190 on quality — so the most aggressive
    buyback names in the market were rejected as serial diluters.
    """
    idx = pd.to_datetime([f"{y}-01-31" for y in range(2020, 2026)])
    raw = pd.Series([616e6, 628e6, 2.50e9, 2.47e9, 2.494e9, 24.4e9], index=idx)
    splits = pd.Series([4.0, 10.0],
                       index=pd.to_datetime(["2021-07-20", "2024-06-10"]))

    assert fb._cagr(raw, 5) > 100, "guard: unadjusted must look absurd"
    adj = fb.split_adjust(raw, splits, kind="count")
    assert abs(fb._cagr(adj, 5)) < 3, f"adjusted CAGR {fb._cagr(adj, 5)}"

    f = sc.Fundamentals(symbol="NVDA", name="N",
                        share_count_cagr_5y=fb._cagr(adj, 5))
    assert not any("share count growing" in g for g in sc.quality_gates(f))


def test_split_direction_differs_for_counts_and_per_share():
    """Counts go UP at a split; per-share figures go down. Backwards is silent."""
    v = pd.Series([4.0, 1.0], index=pd.to_datetime(["2021-01-31", "2022-01-31"]))
    sp = pd.Series([4.0], index=pd.to_datetime(["2021-07-20"]))

    per_share = fb.split_adjust(v, sp, kind="per_share")
    assert per_share.iloc[0] == 1.0, "prior per-share value should be divided"

    counts = fb.split_adjust(v, sp, kind="count")
    assert counts.iloc[0] == 16.0, "prior count should be multiplied"

    with pytest.raises(ValueError):
        fb.split_adjust(v, sp, kind="nonsense")


def test_buyback_yield_recovers_once_counts_are_adjusted():
    idx = pd.to_datetime([f"{y}-01-31" for y in range(2020, 2026)])
    raw = pd.Series([1.0e9, 0.98e9, 0.95e9, 2.76e9, 2.70e9, 2.64e9], index=idx)
    splits = pd.Series([3.0], index=pd.to_datetime(["2023-06-01"]))
    adj = fb.split_adjust(raw, splits, kind="count")
    cagr = fb._cagr(adj, 5)
    assert cagr < 0, "a genuine buyback should show a shrinking count"
    assert max(-cagr, 0.0) > 0, "buyback_yield read 0.0 for every split name"


def test_report_counts_screen_specific_hard_exclusions():
    """
    ev_history_degraded hard-excluded 84 of 190 from the value screen while
    being reported as 'degraded only'. A data reason causing a hard exclusion
    must be counted as one, for the screen it affects.
    """
    f = sc.Fundamentals(symbol="X", name="X")
    f.ev_history_degraded = True
    rep = sc.data_quality_report(f)

    assert rep["hard_failed"] is False, "must not become a SHARED hard failure"
    assert rep["hard_by_screen"]["quality"], "value screen exclusion uncounted"
    assert not rep["hard_by_screen"]["dividend"], "must not leak to other screens"
    assert rep["hard_failed_any_screen"] is True
    assert rep["degraded_only"] is False, "it is not merely degraded"

    g = sc.Fundamentals(symbol="Y", name="Y")
    g.dividend_record_ambiguous = True
    rep2 = sc.data_quality_report(g)
    assert rep2["hard_by_screen"]["dividend"]
    assert not rep2["hard_by_screen"]["quality"]


# ------------- tenth live smoke test: cash tag, debt sum, ROIC base

def test_cash_chain_covers_the_asu_2016_18_migration():
    """
    The cash chain held ONE tag, so there was nothing to stitch. Filers moved
    off CashAndCashEquivalentsAtCarryingValue around fiscal 2019, and net_debt
    inherits cash's index — which killed the EV history for 84 of 190 names.
    Repairing shares_hist moved that number not at all; cash was the cause.
    """
    chain = fs.TAG_CHAINS["cash"]
    assert len(chain) > 1, "single-tag chain cannot stitch a migration"
    assert any("RestrictedCash" in t for t in chain)

    def rows(tag, yrs, val):
        return {tag: {"units": {"USD": [
            {"end": f"{y}-06-30", "filed": f"{y}-08-01", "fy": y, "fp": "FY",
             "form": "10-K", "val": val} for y in yrs]}}}

    facts = {"facts": {"us-gaap": {
        **rows("CashAndCashEquivalentsAtCarryingValue", range(2012, 2020), 8e9),
        **rows("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
               range(2019, 2027), 9e9)}}}
    df = fs.extract_series(facts, "cash")
    assert pd.to_datetime(df["end"]).max().year == 2026, "cash still dies at 2019"
    assert len(df) == 15


def test_total_debt_sums_components():
    """
    Oracle and Comcast read debt of exactly ZERO against roughly $90bn each,
    because the chain held long-term only. That understates invested capital
    and INFLATES ROIC — Oracle came out at 138.3%. False passes are worse than
    false failures, because gates run before scores.
    """
    # Oracle's FY ends 31 May, so periods run June-to-May. _fact assumes a
    # calendar year, which would make these 150-day periods the duration filter
    # correctly rejects.
    yrs = [f"{2018 + i}-05-31" for i in range(8)]
    def _fy(tag, rows, instant=False):
        return {tag: {"units": {"USD": [
            {"end": e, "start": f"{int(e[:4]) - 1}-06-01",
             "filed": f"{int(e[:4])}-07-15", "fy": int(e[:4]), "fp": "FY",
             "form": "10-K", "val": v}
            | ({} if not instant else {}) for e, v in rows]}}}
    us = {}
    us.update(_fy("Revenues", [(y, 50e9) for y in yrs]))
    us.update(_fy("OperatingIncomeLoss", [(y, 15e9) for y in yrs]))
    for tag, val in [("Assets", 140e9), ("StockholdersEquity", 12e9),
                     ("CashAndCashEquivalentsAtCarryingValue", 10e9),
                     ("LongTermDebtNoncurrent", 75e9), ("LongTermDebtCurrent", 12e9),
                     ("FinanceLeaseLiabilityNoncurrent", 3e9)]:
        us.update(_fy(tag, [(y, val) for y in yrs], instant=True))

    f = fb.build("ORCL", {"entityName": "O", "facts": {"us-gaap": us}})
    assert f.debt_unavailable is False
    # invested capital = 12 + 90 - 10 = 92bn, NOPAT = 15 * 0.79 = 11.85bn
    assert 10 < f.roic_ttm < 16, f"ROIC {f.roic_ttm:.1f}% — debt components dropped"


def test_nonpositive_invested_capital_refuses_to_publish_roic():
    """SLS read +2,422%, ASAN +341%, PTON -313%. The positive ones passed the gate."""
    yrs = [f"{2018 + i}-12-31" for i in range(8)]
    us = {}
    us.update(_fact("Revenues", [(y, 1e9) for y in yrs]))
    us.update(_fact("OperatingIncomeLoss", [(y, 0.2e9) for y in yrs]))
    us.update(_fact("Assets", [(y, 5e9) for y in yrs], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 2e9) for y in yrs], instant=True))
    us.update(_fact("CashAndCashEquivalentsAtCarryingValue",
                    [(y, 3e9) for y in yrs], instant=True))          # cash > equity
    us.update(_fact("LongTermDebtNoncurrent", [(y, 0.1e9) for y in yrs], instant=True))

    f = fb.build("SLS", {"entityName": "S", "facts": {"us-gaap": us}})
    assert f.roic_unavailable is True
    assert f.roic_ttm == 0.0, "published a ROIC on a negative capital base"
    assert any("ROIC unavailable" in g for g in sc.quality_gates(f))


def test_roic_5y_flags_when_it_is_really_ttm():
    """
    equity.get(i) is an exact-date lookup and fiscal year-ends drift, so the
    history list emptied and roic_5y silently became roic_ttm for 75 of 190 —
    a one-year figure labelled as a five-year mean.
    """
    yrs_ebit = [f"{2019 + i}-12-31" for i in range(6)]
    yrs_bs = [f"{2019 + i}-12-28" for i in range(6)]      # 3-day drift
    us = {}
    us.update(_fact("Revenues", [(y, 30e9) for y in yrs_ebit]))
    us.update(_fact("OperatingIncomeLoss", [(y, 6e9) for y in yrs_ebit]))
    us.update(_fact("Assets", [(y, 60e9) for y in yrs_bs], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 20e9) for y in yrs_bs], instant=True))
    us.update(_fact("CashAndCashEquivalentsAtCarryingValue",
                    [(y, 4e9) for y in yrs_bs], instant=True))
    us.update(_fact("LongTermDebtNoncurrent", [(y, 10e9) for y in yrs_bs], instant=True))

    f = fb.build("CMCSA", {"entityName": "C", "facts": {"us-gaap": us}})
    assert f.roic_unavailable is False
    assert f.roic_5y_is_ttm_fallback is False, \
        "nearest-period lookup failed on a 3-day fiscal drift"
    assert f.roic_5y > 0


def test_fcf_gate_tests_a_rate_not_a_count():
    """
    `fcf_positive_years_of_10 < 8` failed any company listed under eight years
    regardless of profitability — 56 of 102 failures.
    """
    base = dict(roic_5y=25, roic_ttm=25, wacc=9, gross_margin=60, fcf_margin=20,
                net_debt_ebitda=0.5, share_count_cagr_5y=-2, revenue_cagr_5y=15)

    young = sc.Fundamentals(symbol="A", name="A", fcf_positive_years_of_10=6,
                            fcf_history_years=6, **base)
    assert not any("FCF positive" in g for g in sc.quality_gates(young)), \
        "six clean years still failing"

    patchy = sc.Fundamentals(symbol="B", name="B", fcf_positive_years_of_10=5,
                             fcf_history_years=10, **base)
    assert any("FCF positive" in g for g in sc.quality_gates(patchy))

    tiny = sc.Fundamentals(symbol="C", name="C", fcf_positive_years_of_10=3,
                           fcf_history_years=3, **base)
    assert any("too short to judge" in g for g in sc.quality_gates(tiny))


def test_missing_capex_years_are_counted():
    """`ocf - capex.fillna(0)` treats an absent capex year as zero capex."""
    yrs = [f"{2018 + i}-12-31" for i in range(8)]
    us = {}
    us.update(_fact("Revenues", [(y, 40e9) for y in yrs]))
    us.update(_fact("NetCashProvidedByUsedInOperatingActivities",
                    [(y, 12e9) for y in yrs]))
    us.update(_fact("PaymentsToAcquirePropertyPlantAndEquipment",
                    [(y, 3e9) for y in yrs[:3]]))          # only 3 of 8
    us.update(_fact("OperatingIncomeLoss", [(y, 9e9) for y in yrs]))
    us.update(_fact("Assets", [(y, 80e9) for y in yrs], instant=True))
    us.update(_fact("StockholdersEquity", [(y, 30e9) for y in yrs], instant=True))

    f = fb.build("NVDA", {"entityName": "N", "facts": {"us-gaap": us}})
    assert f.capex_years_missing == 5, f"counted {f.capex_years_missing}"


def test_no_high_risk_chain_is_single_tag():
    """
    `cash` died at ASU 2016-18 because its chain held one tag and there was
    nothing to stitch. That is a CLASS, not an instance. Cash-flow and
    income-statement concepts all have variants filers switch between —
    notably "...ContinuingOperations" whenever a disposal is reported.

    Only stable balance-sheet concepts may stay single-tag.
    """
    allowed_single = {
        # Stable balance-sheet concepts.
        "assets", "current_assets", "current_liabilities", "retained_earnings",
        # Has its own derivation from cost_of_revenue plus an unavailable flag.
        "gross_profit",
        # Optional preference only: when absent, total debt is summed from its
        # components, so a migration away from this tag costs nothing.
        "debt_total",
        # Stable balance-sheet concept, measured rather than assumed: across 45
        # sampled financials, `Goodwill` appears on 82% and ZERO names carry a
        # variant (GoodwillGross, IntangibleAssetsNetIncludingGoodwill) without
        # also carrying the plain tag. There is no migration to stitch across.
        "goodwill",
    }
    single = {k for k, v in fs.TAG_CHAINS.items() if len(v) == 1}
    assert single <= allowed_single, f"new single-tag chain: {single - allowed_single}"


def test_ocf_continuing_operations_variant_resolves():
    """
    A filer reporting only the ContinuingOperations variant returned an EMPTY
    ocf series, setting fcf_unavailable and hard-failing data quality for a
    company that reports the figure perfectly well.
    """
    rows = [{"end": f"{y}-12-31", "start": f"{y}-01-01", "filed": f"{y + 1}-02-15",
             "fy": y, "fp": "FY", "form": "10-K", "val": 12e9}
            for y in range(2016, 2026)]
    facts = {"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations":
            {"units": {"USD": rows}}}}}
    assert len(fs.extract_series(facts, "ocf")) == 10


def test_capex_alternate_tags_resolve():
    for tag in ("PaymentsToAcquireProductiveAssets",
                "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"):
        rows = [{"end": f"{y}-12-31", "start": f"{y}-01-01",
                 "filed": f"{y + 1}-02-15", "fy": y, "fp": "FY",
                 "form": "10-K", "val": 3e9} for y in range(2018, 2026)]
        facts = {"facts": {"us-gaap": {tag: {"units": {"USD": rows}}}}}
        assert len(fs.extract_series(facts, "capex")) == 8, tag


# ---------- eleventh live smoke test: the systemic check

def test_concept_freshness_detects_a_tag_migration():
    """
    The SAME bug has now been found four times in four universe runs: revenue
    (ASC 606, 2018), cash (ASU 2016-18, 2019), gross profit, and debt. Each
    time a concept stopped being tagged and the truncated series silently
    substituted for the full quantity.

    Comparing every concept against revenue turns an open-ended bug class into
    a measured one. A filer still reporting revenue through 2025 whose debt
    stops in 2013 is describing a migration, not a repaid balance sheet.
    """
    def fy(tag, yrs, val):
        return {tag: {"units": {"USD": [
            {"end": f"{y}-12-31", "start": f"{y}-01-01",
             "filed": f"{y + 1}-02-15", "fy": y, "fp": "FY", "form": "10-K",
             "val": val} for y in yrs]}}}

    facts = {"facts": {"us-gaap": {
        **fy("Revenues", range(2013, 2026), 130e9),
        **fy("LongTermDebtNoncurrent", range(2013, 2014), 100e9),   # stops 2013
        **fy("StockholdersEquity", range(2013, 2026), 90e9),
    }}}

    rep = fs.concept_freshness(facts, stale_days=450)
    assert rep["available"] is True
    assert "debt_noncurrent" in rep["stale"], rep["stale"]
    assert "equity" not in rep["stale"]
    assert rep["lags"]["debt_noncurrent"] > 4000


def test_stale_critical_concept_fails_the_gate():
    f = sc.Fundamentals(symbol="VZ", name="V")
    f.stale_concepts = ["debt_noncurrent", "cash"]
    f.concept_lags = {"debt_noncurrent": 4380, "cash": 2500}
    assert any("tag migration" in g for g in sc.data_quality_gates(f))

    # A non-critical concept lagging must not fail the name.
    g_ = sc.Fundamentals(symbol="X", name="X")
    g_.stale_concepts = ["tax_expense"]
    g_.concept_lags = {"tax_expense": 900}
    assert not any("tag migration" in x for x in sc.data_quality_gates(g_))


def test_debt_components_intersect_rather_than_zero_fill():
    """
    The union form reindexed components onto a combined index and filled gaps
    with 0, so a component whose series ENDED became "zero debt" instead of
    "unknown debt". Coca-Cola's long-term debt stops in 2023 while current runs
    to 2025, so invested capital used $0.1bn and dropped $35.5bn — ROIC 49.5%
    against a reality near 15-18%.
    """
    def fy(tag, yrs, val):
        return {tag: {"units": {"USD": [
            {"end": f"{y}-12-31", "start": f"{y}-01-01",
             "filed": f"{y + 1}-02-15", "fy": y, "fp": "FY", "form": "10-K",
             "val": val} for y in yrs]}}}

    us = {}
    us.update(fy("Revenues", range(2016, 2026), 47e9))
    us.update(fy("OperatingIncomeLoss", range(2016, 2026), 13.76e9))
    us.update(fy("Assets", range(2016, 2026), 100e9))
    us.update(fy("StockholdersEquity", range(2016, 2026), 32.2e9))
    us.update(fy("CashAndCashEquivalentsAtCarryingValue", range(2016, 2026), 10.3e9))
    us.update(fy("LongTermDebtNoncurrent", range(2016, 2024), 35.5e9))   # ends 2023
    us.update(fy("LongTermDebtCurrent", range(2016, 2026), 0.1e9))

    f = fb.build("KO", {"entityName": "K", "facts": {"us-gaap": us}})
    assert 12 < f.roic_ttm < 25, f"ROIC {f.roic_ttm:.1f}% — long-term debt dropped"


def test_unprofitable_is_not_filed_as_a_data_problem():
    """14 of 75 flagged names were loss-makers, not short or stale histories."""
    f = sc.Fundamentals(symbol="RIOT", name="R")
    f.ev_history_degraded = True
    f.ev_short_because_unprofitable = True
    gates = sc.quality_gates(f)
    assert any("loss-making years" in g for g in gates)
    assert not any("too short or stale" in g for g in gates)


# ---------- twelfth live smoke test: composition, not freshness

def _mk(tag, yrs, val, end="12-31"):
    return {tag: {"units": {"USD": [
        {"end": f"{y}-{end}", "start": f"{y - 1}-{end}",
         "filed": f"{y + 1}-02-15", "fy": y, "fp": "FY", "form": "10-K",
         "val": val} for y in yrs]}}}


def _debtco(name, nc_yrs, cur_yrs, fl_yrs, nc=35.5e9, cur=0.1e9, fl=0.2e9):
    us = {}
    us.update(_mk("Revenues", range(2013, 2026), 47e9))
    us.update(_mk("OperatingIncomeLoss", range(2013, 2026), 13.76e9))
    us.update(_mk("Assets", range(2013, 2026), 100e9))
    us.update(_mk("StockholdersEquity", range(2013, 2026), 32.2e9))
    us.update(_mk("CashAndCashEquivalentsAtCarryingValue", range(2013, 2026), 10.3e9))
    if nc_yrs:
        us.update(_mk("LongTermDebtNoncurrent", nc_yrs, nc))
    if cur_yrs:
        us.update(_mk("LongTermDebtCurrent", cur_yrs, cur))
    if fl_yrs:
        us.update(_mk("FinanceLeaseLiabilityNoncurrent", fl_yrs, fl))
    return fb.build(name, {"entityName": name, "facts": {"us-gaap": us}})


def test_stale_optional_component_does_not_veto_the_total():
    """
    NVIDIA stopped tagging finance leases in 2017, so a strict intersection
    across all three components gave ZERO dates and dropped a company whose two
    core components are perfectly current. Every freshness number read 0 while
    the derived quantity was empty.
    """
    f = _debtco("NVDA", range(2013, 2026), range(2020, 2026), range(2009, 2018),
                nc=8.5e9, cur=1.0e9, fl=0.2e9)
    assert f.debt_unavailable is False, "an immaterial dead component vetoed debt"
    assert f.roic_unavailable is False
    assert f.roic_ttm > 0


def test_stale_CORE_component_truncates_rather_than_being_dropped():
    """
    The mirror error. Dropping a stale component is right for finance leases
    and catastrophic for long-term debt — Coca-Cola's ends 2023 and is $35.5bn,
    so dropping it reproduces the zero-fill bug exactly (ROIC 49.5%).
    """
    f = _debtco("KO", range(2013, 2024), range(2013, 2026), None)
    assert 12 < f.roic_ttm < 25, f"ROIC {f.roic_ttm:.1f}% — core component dropped"
    assert any("truncating the join" in w for w in f.derivation_warnings)


def test_derivation_integrity_catches_an_empty_result():
    """
    The invariant freshness cannot express: a derived quantity must be no more
    absent, and no more stale, than its freshest input.
    """
    inputs = {"a": pd.Series([1.0], index=pd.to_datetime(["2026-01-25"])),
              "b": pd.Series([1.0], index=pd.to_datetime(["2026-01-25"]))}
    warn = fb._derivation_integrity({"debt": (pd.Series(dtype=float), inputs)})
    assert warn and "composition failure" in warn[0]

    stale = pd.Series([1.0], index=pd.to_datetime(["2019-01-25"]))
    warn2 = fb._derivation_integrity({"debt": (stale, inputs)})
    assert warn2 and "truncating the join" in warn2[0]

    ok = pd.Series([1.0], index=pd.to_datetime(["2026-01-25"]))
    assert fb._derivation_integrity({"debt": (ok, inputs)}) == []


def test_debt_free_company_is_not_treated_as_missing_data():
    """
    The blind spot: freshness sees a concept that STOPPED, never one that never
    started. Interest expense is the corroborating witness — a company paying
    none plausibly has no debt and deserves a real ROIC.
    """
    us = {}
    us.update(_mk("Revenues", range(2016, 2026), 10e9))
    us.update(_mk("OperatingIncomeLoss", range(2016, 2026), 3e9))
    us.update(_mk("Assets", range(2016, 2026), 14e9))
    us.update(_mk("StockholdersEquity", range(2016, 2026), 11e9))
    us.update(_mk("CashAndCashEquivalentsAtCarryingValue", range(2016, 2026), 4e9))
    f = fb.build("DEBTFREE", {"entityName": "D", "facts": {"us-gaap": us}})
    assert f.debt_assumed_zero is True
    assert f.debt_unavailable is False
    assert f.roic_unavailable is False, "a debt-free company was excluded"

    # Paying material interest with no balance tagged is the opposite case.
    us["InterestExpense"] = _mk("InterestExpense", range(2016, 2026), 0.9e9)["InterestExpense"]
    g_ = fb.build("MISSING", {"entityName": "M", "facts": {"us-gaap": us}})
    assert g_.debt_unavailable is True
    assert g_.debt_assumed_zero is False


def test_interest_expense_nonoperating_resolves():
    """The fifth migration, named by concept_freshness before a universe run."""
    facts = {"facts": {"us-gaap": _mk("InterestExpenseNonoperating",
                                      range(2016, 2026), 0.5e9)}}
    assert len(fs.extract_series(facts, "interest_expense")) == 10


# ---------- thirteenth: the hand-check finding (NextEra)

def test_absent_capex_voids_fcf_rather_than_fabricating_it():
    """
    Found by hand-checking five RANDOM non-megacaps. NextEra tags no capex
    concept at all, so `ocf - capex.fillna(0)` set free cash flow equal to
    operating cash flow: +48.4% FCF margin and 10 of 10 positive years, for a
    utility spending ~$25bn a year on capital and persistently FCF-negative.
    The quality screen scored the fabrication as a strength.

    Neither existing check could see it. Freshness looks for a concept that
    STOPPED; derivation integrity looks for a join that COLLAPSED. This is a
    concept that never started, whose absence is filled with a value that
    flatters the result — and the fill is silent because the arithmetic works.
    """
    yrs = range(2016, 2026)
    us = {}
    for tag, val in [("Revenues", 26e9), ("OperatingIncomeLoss", 8e9),
                     ("NetCashProvidedByUsedInOperatingActivities", 12.5e9),
                     ("Assets", 190e9), ("StockholdersEquity", 48e9),
                     ("CashAndCashEquivalentsAtCarryingValue", 2e9),
                     ("LongTermDebtNoncurrent", 75e9)]:
        us.update(_mk(tag, yrs, val))

    f = fb.build("NEE", {"entityName": "NextEra", "facts": {"us-gaap": us}})
    assert f.fcf_unavailable is True, "published a fabricated FCF margin"
    assert f.capex_voided_fcf is True
    assert f.fcf_margin == 0.0
    assert f.fcf_positive_years_of_10 == 0, "10/10 positive years on no capex data"
    assert any("fabricated" in w for w in f.derivation_warnings)
    assert any("capex not tagged" in g for g in sc.quality_gates(f))


def test_partial_capex_warns_but_does_not_void():
    yrs = range(2016, 2026)
    us = {}
    for tag, val in [("Revenues", 26e9), ("OperatingIncomeLoss", 8e9),
                     ("NetCashProvidedByUsedInOperatingActivities", 12.5e9),
                     ("Assets", 190e9), ("StockholdersEquity", 48e9),
                     ("CashAndCashEquivalentsAtCarryingValue", 2e9),
                     ("LongTermDebtNoncurrent", 75e9)]:
        us.update(_mk(tag, yrs, val))
    us.update(_mk("PaymentsToAcquirePropertyPlantAndEquipment", range(2016, 2024), 25e9))

    f = fb.build("PART", {"entityName": "P", "facts": {"us-gaap": us}})
    assert f.fcf_unavailable is False, "two missing years should not void it"
    assert f.derivation_warnings, "but it must still be flagged"


def test_fill_integrity_distinguishes_fabricated_from_incomplete():
    idx = pd.to_datetime([f"{2016 + i}-12-31" for i in range(10)])
    result = pd.Series([1.0] * 10, index=idx)

    none_present = pd.Series([np.nan] * 10, index=idx)
    w = fb._fill_integrity("free cash flow", result, none_present, "capex")
    assert w and "fabricated" in w[0]

    mostly = pd.Series([1.0] * 8 + [np.nan] * 2, index=idx)
    w2 = fb._fill_integrity("free cash flow", result, mostly, "capex")
    assert w2 and "overstate" in w2[0] and "fabricated" not in w2[0]

    assert fb._fill_integrity("x", result, pd.Series([1.0] * 10, index=idx), "c") == []


def test_altman_not_applied_outside_manufacturing():
    """
    Altman fitted Z on manufacturers. NextEra scores 0.54 — that describes a
    regulated utility's capital structure, not distress.
    """
    base = dict(drawdown_from_ath=-55, gross_margin=40, cash_runway_quarters=99,
                net_debt_ebitda=3.5, share_count_cagr_5y=1.0, revenue_cagr_5y=9,
                altman_z=0.54)
    util = sc.Fundamentals(symbol="NEE", name="N", sector="utility", **base)
    util.altman_not_applicable = True
    assert not any("Altman" in g for g in sc.recovery_gates(util))

    mfg = sc.Fundamentals(symbol="XYZ", name="X", sector="general", **base)
    assert any("Altman" in g for g in sc.recovery_gates(mfg))


def test_long_term_debt_total_tag_does_not_double_count():
    """
    `LongTermDebt` is the TOTAL carrying amount in most filers' usage. Leaving
    it in debt_noncurrent and then adding debt_current double-counted: Oracle's
    invested capital read $141.3bn against roughly $100bn actual.
    """
    assert "LongTermDebt" in fs.TAG_CHAINS["debt_total"]
    assert "LongTermDebt" not in fs.TAG_CHAINS["debt_noncurrent"]


# ---------- fourteenth: fresh-five hand-check findings

def test_roic_history_skips_years_rather_than_zero_filling_debt():
    """
    The SAME zero-fill removed from inv_cap, surviving in roic_hist. Coca-Cola's
    debt series ends 2023, so 2024 and 2025 took capital bases of $14.0bn and
    $21.9bn instead of ~$52bn and read 56.3% and 49.6%. The five-year mean went
    from 16.1% to 30.9%.
    """
    us = {}
    us.update(_mk("Revenues", range(2016, 2026), 47e9))
    us.update(_mk("OperatingIncomeLoss", range(2016, 2026), 11e9))
    us.update(_mk("Assets", range(2016, 2026), 100e9))
    us.update(_mk("StockholdersEquity", range(2016, 2026), 25e9))
    us.update(_mk("CashAndCashEquivalentsAtCarryingValue", range(2016, 2026), 10e9))
    us.update(_mk("LongTermDebtNoncurrent", range(2016, 2024), 37e9))   # ends 2023

    f = fb.build("KO", {"entityName": "K", "facts": {"us-gaap": us}})
    assert 12 < f.roic_5y < 22, f"roic_5y {f.roic_5y:.1f}% — debt zero-filled again"
    assert abs(f.roic_5y - f.roic_ttm) < 10


def test_multi_unit_concepts_are_pinned_not_blended():
    """
    Seventh bug class, invisible to all three integrity checks by construction:
    both units are current, nothing absent, nothing zero-filled. HDFC Bank tags
    dividends in INR/shares AND USD/shares; blended, the rupee figure was
    divided by a dollar ADR price and the yield published as 47.45% against a
    real 1.11%.
    """
    inr = [{"end": f"{y}-03-31", "start": f"{y - 1}-04-01", "filed": f"{y}-06-30",
            "fy": y, "fp": "FY", "form": "20-F", "val": 19.0 + 0.3 * (y - 2015)}
           for y in range(2015, 2026)]
    usd = [{"end": f"{y}-03-31", "start": f"{y - 1}-04-01", "filed": f"{y}-06-30",
            "fy": y, "fp": "FY", "form": "20-F", "val": 0.24 + 0.002 * (y - 2019)}
           for y in range(2019, 2026)]
    facts = {"facts": {"us-gaap": {"CommonStockDividendsPerShareDeclared":
                                   {"units": {"INR/shares": inr, "USD/shares": usd}}}}}

    df = fs.extract_series(facts, "dividends_per_share")
    assert df.attrs["multi_unit"] is True, "mixed filer not flagged"
    assert df.attrs["unit_used"] == "USD/shares"
    assert len(set(df["unit"])) == 1, "series still blends currencies"
    assert df["val"].max() < 1.0, "rupee values leaked into a dollar series"


def test_single_unit_concept_is_unflagged():
    facts = {"facts": {"us-gaap": _mk("Revenues", range(2018, 2026), 5e9)}}
    df = fs.extract_series(facts, "revenue")
    assert df.attrs["multi_unit"] is False
    assert df.attrs["unit_used"] == "USD"


def test_near_zero_denominator_voids_the_ratio():
    """Archer published a 33.33% gross margin on ~zero revenue, unflagged."""
    idx = pd.to_datetime([f"{2020 + i}-12-31" for i in range(5)])
    rev = pd.Series([2e5] * 5, index=idx)          # $200k
    gp = pd.Series([6.7e4] * 5, index=idx)
    val, stale = fb._ratio(gp, rev, as_of=date(2025, 6, 1), min_den=1e7)
    assert stale is True and val == 0.0

    real = pd.Series([4e9] * 5, index=idx)
    val2, stale2 = fb._ratio(pd.Series([1.4e9] * 5, index=idx), real,
                             as_of=date(2025, 6, 1), min_den=1e7)
    assert stale2 is False and 30 < val2 < 40


def test_pre_revenue_company_is_gated_and_altman_skipped():
    us = {}
    us.update(_mk("Revenues", range(2020, 2026), 2e5))
    us.update(_mk("OperatingIncomeLoss", range(2020, 2026), -4.3e8))
    us.update(_mk("Assets", range(2020, 2026), 9e8))
    us.update(_mk("StockholdersEquity", range(2020, 2026), 7e8))
    us.update(_mk("CashAndCashEquivalentsAtCarryingValue", range(2020, 2026), 5e8))
    us.update(_mk("LongTermDebtNoncurrent", range(2020, 2026), 1e7))

    f = fb.build("ACHR", {"entityName": "A", "facts": {"us-gaap": us}})
    assert f.pre_revenue is True
    assert f.gross_margin == 0.0, "published a margin on ~zero revenue"
    assert f.altman_not_applicable is True, "Z dominated by the equity term"
    assert any("not meaningful" in g for g in sc.data_quality_gates(f))


def test_sector_branches_can_actually_fire():
    """
    `sector` defaulted to "general" and nothing ever classified it, so the
    REIT/utility payout, FCF and debt caps in dividend_gates — and the Altman
    skip — had never applied to any name in any run.
    """
    assert hasattr(fs, "company_sector")
    assert fs._SIC_SECTORS, "no SIC mapping"

    reit = sc.Fundamentals(symbol="R", name="R", sector="reit",
                           increase_streak_years=10, years_since_cut=99,
                           dps_cagr_5y=6.0, eps_payout=80.0, fcf_payout=85.0,
                           net_debt_ebitda=5.0, interest_coverage=5.0,
                           revenue_cagr_5y=4.0, market_cap=8e9, dollar_adv=2e7,
                           fcf_history_years=10, fcf_positive_years_of_10=10)
    assert not sc.dividend_gates(reit), "REIT allowances still not applying"

    general = sc.Fundamentals(**{**reit.__dict__, "sector": "general"})
    assert sc.dividend_gates(general), "general caps should reject these payouts"


# ---------- the dead-code class, guarded rather than patched

def test_every_quality_field_is_both_written_and_read():
    """
    Two instances of one class, found one session apart:

      sector      READ by dividend_gates and the Altman skip, never SET —
                  three payout allowances had never fired for any name ever.
      multi_unit  SET on df.attrs by extract_series, never READ — the
                  currency-mixing flag existed and gated nothing.

    A field written but never read is a check that does not run. A field read
    but never written is a branch that cannot be reached. Both look implemented.
    This asserts every data-quality field on Fundamentals is assigned in the
    builder AND consumed by the screeners.
    """
    import re
    from pathlib import Path
    from dataclasses import fields as dc_fields

    root = Path(sc.__file__).parent
    builder_src = (root / "fundamentals_builder.py").read_text()
    screener_src = (root / "screeners.py").read_text()

    # Fields the builder derives and the gates consult, as opposed to raw
    # statement values that only feed scoring arithmetic.
    quality_fields = [
        fld.name for fld in dc_fields(sc.Fundamentals)
        if fld.name.endswith(("_unavailable", "_ambiguous", "_degraded",
                              "_applicable", "_fallback", "_days", "_concepts",
                              "_warnings", "_zero", "_fcf"))
        or fld.name in ("pre_revenue", "mixed_unit_concepts", "sector")
    ]
    assert len(quality_fields) > 10, "field discovery is broken, not the code"

    unwritten, unread = [], []
    for name in quality_fields:
        written = (re.search(rf"\bf\.{name}\s*=", builder_src)
                   or re.search(rf"\b{name}\s*=", builder_src))
        read = re.search(rf"\bf\.{name}\b", screener_src)
        if not written:
            unwritten.append(name)
        if not read:
            unread.append(name)

    assert not unwritten, f"read but never set — unreachable branch: {unwritten}"
    assert not unread, f"set but never read — a check that never runs: {unread}"


def test_mixed_unit_concepts_reaches_a_gate():
    f = sc.Fundamentals(symbol="NBIS", name="N")
    f.mixed_unit_concepts = ["revenue", "net_income", "assets", "equity"]
    assert any("multiple currencies" in g for g in sc.data_quality_gates(f))
    assert sc.data_quality_report(f)["flags"]["mixed_units"] is True

    # One mixed concept on a CRITICAL field is still gated — separately.
    #
    # The >= 3 threshold sat above the case that motivated the whole check:
    # HDFC Bank has exactly one mixed concept, dividends_per_share, and that
    # single concept produced the 47.45% yield. Field written, read reachable,
    # threshold above the motivating case — a check that effectively never ran.
    g_ = sc.Fundamentals(symbol="HDB", name="H")
    g_.mixed_unit_concepts = ["dividends_per_share"]
    gates = sc.data_quality_gates(g_)
    assert any("a unit was chosen" in x for x in gates)
    assert not any("not internally comparable" in x for x in gates)

    # A peripheral concept alone is a quirk and is not gated.
    p_ = sc.Fundamentals(symbol="X", name="X")
    p_.mixed_unit_concepts = ["tax_expense"]
    assert not sc.data_quality_gates(p_)


def test_builder_records_mixed_units_from_extraction():
    """df.attrs is dropped when a DataFrame becomes a Series — that is how the
    flag came to be set and never read."""
    inr = [{"end": f"{y}-03-31", "start": f"{y - 1}-04-01", "filed": f"{y}-06-30",
            "fy": y, "fp": "FY", "form": "20-F", "val": 19.0} for y in range(2016, 2026)]
    usd = [{"end": f"{y}-03-31", "start": f"{y - 1}-04-01", "filed": f"{y}-06-30",
            "fy": y, "fp": "FY", "form": "20-F", "val": 0.25} for y in range(2019, 2026)]
    us = {}
    us.update(_mk("Revenues", range(2016, 2026), 8e9))
    us.update(_mk("OperatingIncomeLoss", range(2016, 2026), 2e9))
    us.update(_mk("Assets", range(2016, 2026), 40e9))
    us.update(_mk("StockholdersEquity", range(2016, 2026), 12e9))
    us["CommonStockDividendsPerShareDeclared"] = {
        "units": {"INR/shares": inr, "USD/shares": usd}}

    f = fb.build("HDB", {"entityName": "H", "facts": {"us-gaap": us}})
    assert "dividends_per_share" in f.mixed_unit_concepts


def test_pinning_records_the_coverage_it_costs():
    """
    USD wins unconditionally because every ratio divides by a USD price. But
    HDFC Bank has 35 INR facts and 11 USD, so that choice truncates its
    dividend history from 35 years to 11 — silently capping the increase
    streak and every "own history" percentile. The choice is right; making it
    invisible is not.
    """
    inr = [{"end": f"{y}-03-31", "start": f"{y - 1}-04-01", "filed": f"{y}-06-30",
            "fy": y, "fp": "FY", "form": "20-F", "val": 19.0}
           for y in range(1991, 2026)]
    usd = [{"end": f"{y}-03-31", "start": f"{y - 1}-04-01", "filed": f"{y}-06-30",
            "fy": y, "fp": "FY", "form": "20-F", "val": 0.25}
           for y in range(2015, 2026)]
    facts = {"facts": {"us-gaap": {"CommonStockDividendsPerShareDeclared":
                                   {"units": {"INR/shares": inr, "USD/shares": usd}}}}}

    df = fs.extract_series(facts, "dividends_per_share")
    cost = df.attrs["unit_coverage_cost"]
    assert cost is not None, "truncation from 35 facts to 11 went unrecorded"
    assert cost["chosen"] == "USD/shares" and cost["chosen_facts"] == 11
    assert cost["richest_facts"] == 35


def test_no_coverage_cost_when_usd_is_also_the_richest():
    facts = {"facts": {"us-gaap": _mk("Revenues", range(2016, 2026), 5e9)}}
    assert fs.extract_series(facts, "revenue").attrs["unit_coverage_cost"] is None


# ---------- fifteenth: containment is not correctness

def test_fields_derived_from_unavailable_inputs_hold_no_value():
    """
    The gates exclude these names, so nothing downstream of a gate is wrong.
    But PayPal STORED an EV/EBIT of 7.25 and an Altman Z of 1.95 computed on an
    empty debt series, and anything reading a Fundamentals outside the gate
    path — a dashboard detail page, an export — renders them as fact.

    The meta-test cannot see this: both fields are written, both are read, and
    the read is reachable. Containment by gate is not correctness by
    construction.
    """
    f = sc.Fundamentals(symbol="PYPL", name="P", ev_ebit=7.25, altman_z=1.95,
                        net_debt_ebitda=0.4, roic_ttm=18.0, roic_5y=17.0,
                        ev_ebit_percentile_10y=42.0)
    f.debt_unavailable = True
    f.roic_unavailable = True

    voided = fb.void_derived_fields(f)
    for name in ("ev_ebit", "altman_z", "net_debt_ebitda", "roic_ttm", "roic_5y"):
        # None, not 0.0. Setting zero made the FIELD say unavailable while the
        # VALUE said zero — 1,284 fields across the universe.
        assert getattr(f, name) is None, f"{name} voided to a value, not to None"
    assert set(voided) >= {"ev_ebit", "altman_z", "net_debt_ebitda"}

    clean = sc.Fundamentals(symbol="OK", name="O", ev_ebit=12.0, altman_z=4.0)
    assert fb.void_derived_fields(clean) == [], "voided a healthy record"


def test_freshness_reference_is_checked_against_the_calendar():
    """
    Revenue is the reference, so its own lag is ZERO BY CONSTRUCTION. A filer
    whose revenue is 500 days old reads lag 0 on every concept and sails
    through — nothing measures the yardstick.
    """
    facts = {"facts": {"us-gaap": _mk("Revenues", range(2016, 2024), 5e9)}}
    rep = fs.concept_freshness(facts, as_of=date(2026, 9, 12), stale_days=450)
    assert rep["lags"]["revenue"] == 0, "self-lag is zero by construction"
    assert rep["reference_stale"] is True, "the yardstick went unchecked"
    assert rep["reference_lag_days"] > 900

    current = {"facts": {"us-gaap": _mk("Revenues", range(2016, 2027), 5e9)}}
    rep2 = fs.concept_freshness(current, as_of=date(2026, 9, 12), stale_days=450)
    assert rep2["reference_stale"] is False


def test_altman_exemption_is_narrower_than_the_sector_label():
    """
    SIC files crypto miners and SPAC-descended issuers under 6199 "finance
    services". Eleven of twenty-four `financial` names were exactly that, and
    all eleven lost the Altman gate — including Core Scientific at -$0.96bn
    equity, which is precisely the case Altman exists for.
    """
    assert fs.altman_exempt(6022) is True, "national bank"
    assert fs.altman_exempt(6798) is True, "REIT"
    assert fs.altman_exempt(4911) is True, "electric utility"
    assert fs.altman_exempt(6199) is False, "finance services — crypto miners"
    assert fs.altman_exempt(6770) is False, "blank check / SPAC"
    assert fs.altman_exempt(7372) is False, "software"


def test_foreign_per_share_yield_is_voided_not_published():
    """
    An ADR trades at a MULTIPLE of the underlying share and XBRL never says
    what it is. HDFC Bank came out at 0.56% against a real 1.11% — out by
    roughly its ADR ratio. Pinning fixed the currency and could not fix this,
    because the information is not in the filing.
    """
    us = {}
    us.update(_mk("Revenues", range(2016, 2026), 8e9))
    us.update(_mk("OperatingIncomeLoss", range(2016, 2026), 2e9))
    us.update(_mk("Assets", range(2016, 2026), 40e9))
    us.update(_mk("StockholdersEquity", range(2016, 2026), 12e9))
    us["CommonStockDividendsPerShareDeclared"] = {"units": {"USD/shares": [
        {"end": f"{y}-03-31", "start": f"{y - 1}-04-01", "filed": f"{y}-06-30",
         "fy": y, "fp": "FY", "form": "20-F", "val": 0.25}
        for y in range(2016, 2026)]}}

    f = fb.build("HDB", {"entityName": "H", "facts": {"us-gaap": us}})
    assert f.foreign_private_issuer is True
    assert f.adr_ratio_unknown is True
    assert f.dividend_yield == 0.0, "published a yield wrong by the ADR ratio"
    assert any("ADR price" in g for g in sc.dividend_gates(f))


def test_domestic_filer_keeps_its_yield():
    us = {}
    us.update(_mk("Revenues", range(2016, 2026), 8e9))
    us.update(_mk("OperatingIncomeLoss", range(2016, 2026), 2e9))
    us.update(_mk("Assets", range(2016, 2026), 40e9))
    us.update(_mk("StockholdersEquity", range(2016, 2026), 12e9))
    us.update(_mk("CommonStockDividendsPerShareDeclared", range(2016, 2026), 1.2))
    f = fb.build("KO", {"entityName": "K", "facts": {"us-gaap": us}})
    assert f.foreign_private_issuer is False
    assert f.adr_ratio_unknown is False


# ---------- sixteenth: threshold audit

def test_reference_and_calendar_lag_use_different_thresholds():
    """
    One threshold was serving two different measurements, which left a dead
    band exactly one reporting cycle wide. Reference-relative: two concepts
    from the SAME filing share a period end, so a 365-day lag means the concept
    was NOT TAGGED in the latest filing. At 450 that passed unflagged.
    Calendar-relative: a healthy annual fact is legitimately ~440 days old.
    """
    facts = {"facts": {"us-gaap": {
        **_mk("Revenues", range(2016, 2026), 5e9),
        **_mk("LongTermDebtNoncurrent", range(2016, 2025), 3e9)}}}   # one cycle behind

    rep = fs.concept_freshness(facts, as_of=date(2026, 6, 1))
    assert rep["lags"]["debt_noncurrent"] == 365
    assert "debt_noncurrent" in rep["stale"], "the dead band is back"

    old = fs.concept_freshness(facts, as_of=date(2026, 6, 1), stale_days=450)
    assert "debt_noncurrent" not in old["stale"], "guard: 450 let it through"

    assert fs.REFERENCE_LAG_DAYS < 365 < fs.CALENDAR_LAG_DAYS


def test_fiscal_drift_does_not_trip_the_tighter_threshold():
    """52/53-week filers shift a few days; that must stay clean at 200."""
    drift = {"facts": {"us-gaap": {
        **_mk("Revenues", range(2016, 2026), 5e9),
        "StockholdersEquity": {"units": {"USD": [
            {"end": f"{y}-12-28", "start": f"{y}-01-01", "filed": f"{y + 1}-02-15",
             "fy": y, "fp": "FY", "form": "10-K", "val": 2e9}
            for y in range(2016, 2026)]}}}}}
    rep = fs.concept_freshness(drift, as_of=date(2026, 6, 1))
    assert "equity" not in rep["stale"]
    assert rep["lags"]["equity"] < 10


def test_revenue_floor_scales_to_the_business():
    """
    An ABSOLUTE $10m floor caught Archer at $200k but left a band unguarded: a
    $1.4bn-asset biotech with $12m of revenue cleared it and published a 33%
    gross margin that is still noise.
    """
    def case(rev, assets, sector="general"):
        yrs = range(2018, 2026)
        us = {}
        us.update(_mk("Revenues", yrs, rev))
        us.update(_mk("GrossProfit", yrs, rev * 0.33))
        us.update(_mk("OperatingIncomeLoss", yrs, rev * 0.1))
        us.update(_mk("Assets", yrs, assets))
        us.update(_mk("StockholdersEquity", yrs, assets * 0.6))
        return fb.build("X", {"entityName": "X", "facts": {"us-gaap": us}},
                        sector=sector)

    assert case(2e5, 9e8).pre_revenue is True, "Archer"
    assert case(1.2e7, 1.4e9).pre_revenue is True, "clears $10m, still noise"
    assert case(5e8, 2e9).pre_revenue is False, "a real smallcap"
    assert case(2.6e10, 1.9e11).pre_revenue is False, "a utility"
    # Revenue is structurally a low share of assets for a bank; not pre-commercial.
    assert case(9e10, 3.2e12, "financial").pre_revenue is False


def test_ev_count_guard_never_caught_its_own_case():
    """
    Documenting a threshold that is live but did NOT catch what motivated it.
    J&J had 1,564 aligned EBIT points, every one pre-2016 — the count guard
    passed it and the RECENCY test caught it. The count guard earns its place
    on a different case (MRVL at 358 usable points), not on J&J.
    """
    assert 1564 >= 500, "J&J cleared the count guard"
    assert 358 < 500, "MRVL is what the count guard is actually for"


# ---------- the contract between the screeners and the dashboard

def test_dashboard_contract_is_satisfied():
    """
    The two halves have never met. `run_screen` emits
    {symbol, score, components, gates_failed}; index.html's column definitions
    ask for {ticker, roic, fcfy, ev, evp, impl, act, gap, up} plus comp, note
    and facts. Wiring the pipeline straight through would have produced a table
    of correctly-ranked BLANKS, and the obvious diagnosis would have been "the
    screener returned nothing".

    This asserts every column key declared in the HTML is actually produced.
    """
    import re
    from pathlib import Path
    from engines import dashboard_adapter as da

    html = (Path(sc.__file__).parent.parent / "index.html").read_text()

    f = sc.Fundamentals(
        symbol="MSFT", name="Microsoft", roic_5y=28.0, roic_ttm=27.0, wacc=9.0,
        gross_margin=68.0, fcf_margin=30.0, net_debt_ebitda=0.3,
        fcf_positive_years_of_10=10, fcf_history_years=10,
        share_count_cagr_5y=-1.0, revenue_cagr_5y=14.0, ev_ebit=22.0,
        ev_ebit_median_10y=26.0, ev_ebit_percentile_10y=30.0, fcf_yield=3.4,
        reverse_dcf_implied_growth=5.0, dividend_yield=0.8,
        yield_median_5y=0.9, dps_cagr_5y=10.0, increase_streak_years=16,
        eps_payout=25.0, fcf_payout=30.0, interest_coverage=40.0,
        years_since_cut=99, drawdown_from_ath=-12.0, cash_runway_quarters=99,
        ev_sales_percentile_5y=40.0, altman_z=8.0)

    for engine, scorer in (("value", sc.score_quality_value),
                           ("dividend", sc.score_dividend),
                           ("recovery", lambda x: sc.score_recovery(x, 40.0, 1.2, 1.6))):
        res = scorer(f)
        row = da.to_rows(engine, [(f, res)])[0]

        block = html[html.index(f"{engine}:{{"):]
        block = block[:block.index("comps:[")]
        declared = re.findall(r'\{k:"(\w+)"', block)
        assert declared, f"no columns parsed for {engine}"

        missing = [k for k in declared if k not in row]
        assert not missing, f"{engine} would render blanks for {missing}"

        for key in ("ticker", "name", "score", "comp", "note", "facts"):
            assert key in row, f"{engine} missing {key}"
        assert len(row["comp"]) == 5, f"{engine} component count mismatch"
        assert row["note"].startswith("<p>")


def test_adapter_surfaces_data_caveats_where_they_are_read():
    """A name that ranks despite a degraded input must say so on the page."""
    from engines import dashboard_adapter as da

    f = sc.Fundamentals(symbol="X", name="X", roic_5y=20.0, wacc=9.0,
                        revenue_cagr_5y=8.0, reverse_dcf_implied_growth=3.0)
    f.gross_profit_unavailable = True
    f.voided_fields = ["altman_z", "net_debt_ebitda"]

    row = da.to_rows("value", [(f, sc.score_quality_value(f))])[0]
    assert "Data caveats" in row["note"]
    assert "gross profit unavailable" in row["note"]
    assert "Voided as unavailable" in row["note"]


# ---------- seventeenth: the 1,500 run and the live dashboard

def test_stale_debt_total_does_not_override_current_components():
    """
    EnerSys tags debt_total with 4 points ending 2014 ($0.288bn) while
    debt_noncurrent runs to 2026 ($1.080bn). Taking the combined tag
    unconditionally produced net debt of -$0.151bn and a leverage ratio of
    -0.30x — reading NET CASH for a company carrying +$0.64bn of net debt.
    net_debt_ebitda gates all three screens, so a wrong SIGN lets a levered
    company clear a leverage gate.
    """
    def mk(tag, yrs, val):
        return {tag: {"units": {"USD": [
            {"end": f"{y}-03-31", "start": f"{y - 1}-03-31",
             "filed": f"{y}-06-01", "fy": y, "fp": "FY", "form": "10-K",
             "val": val} for y in yrs]}}}

    us = {}
    us.update(mk("Revenues", range(2010, 2027), 3.6e9))
    us.update(mk("OperatingIncomeLoss", range(2010, 2027), 0.42e9))
    us.update(mk("DepreciationDepletionAndAmortization", range(2010, 2027), 0.086e9))
    us.update(mk("Assets", range(2010, 2027), 3.9e9))
    us.update(mk("StockholdersEquity", range(2010, 2027), 1.6e9))
    us.update(mk("CashAndCashEquivalentsAtCarryingValue", range(2010, 2027), 0.439e9))
    us.update(mk("LongTermDebtNoncurrent", range(2010, 2027), 1.080e9))
    us.update(mk("DebtLongtermAndShorttermCombinedAmount", range(2011, 2015), 0.288e9))

    f = fb.build("ENS", {"entityName": "EnerSys", "facts": {"us-gaap": us}})
    assert f.net_debt_ebitda > 0, "wrong-signed leverage: reads net cash"
    assert 0.8 < f.net_debt_ebitda < 1.8


def test_near_zero_invested_capital_voids_roic():
    """
    Deckers: equity $2.50bn, no debt, cash $1.91bn leaves $0.59bn of invested
    capital against $1.26bn of EBIT — it published 281.1% against a real 35-40%
    and ranked FIRST on the value screen. The guard caught <= 0 but not
    near-zero.
    """
    def mk(tag, yrs, val):
        return {tag: {"units": {"USD": [
            {"end": f"{y}-03-31", "start": f"{y - 1}-03-31",
             "filed": f"{y}-06-01", "fy": y, "fp": "FY", "form": "10-K",
             "val": val} for y in yrs]}}}

    us = {}
    us.update(mk("Revenues", range(2016, 2027), 4.9e9))
    us.update(mk("OperatingIncomeLoss", range(2016, 2027), 1.26e9))
    us.update(mk("Assets", range(2016, 2027), 4.5e9))
    us.update(mk("StockholdersEquity", range(2016, 2027), 2.50e9))
    us.update(mk("CashAndCashEquivalentsAtCarryingValue", range(2016, 2027), 1.91e9))
    us.update(mk("LongTermDebtNoncurrent", range(2016, 2027), 0.0))

    f = fb.build("DECK", {"entityName": "Deckers", "facts": {"us-gaap": us}})
    assert f.roic_unavailable is True, "published 281% on a thin denominator"
    assert f.roic_ttm == 0.0
    assert any("ROIC unavailable" in g for g in sc.quality_gates(f))


def test_truncating_join_gates_like_composition_failure():
    """
    Two variants of one warning, and only one gated. 'composition failure'
    means EMPTY; 'truncating the join' means STALE. EnerSys shipped a
    wrong-signed ratio through the second with data_quality_gates empty.
    """
    f = sc.Fundamentals(symbol="ENS", name="E")
    f.derivation_warnings = ["debt ends 2014-03-31 but its inputs run to "
                             "2026-03-31 (4383d) — a component is truncating the join"]
    assert any("truncating the join" in g for g in sc.data_quality_gates(f))


def test_reverse_dcf_is_computed_not_a_placeholder():
    """
    It read 0.0 for every one of 37 rows, and it is a SCORED component — so the
    expectations gap was a constant and the gap/upside columns were both
    derived from it. It needs nothing the builder does not already have.
    """
    assert fb.reverse_dcf_growth(30e9, 2.0e9, 9.0) < 5, "15x FCF implies low growth"
    assert fb.reverse_dcf_growth(120e9, 2.0e9, 9.0) > 15, "60x FCF implies high"
    assert fb.reverse_dcf_growth(30e9, -1e9, 9.0) is None, "negative FCF"
    assert fb.reverse_dcf_growth(0, 2e9, 9.0) is None

    a = fb.reverse_dcf_growth(30e9, 2.0e9, 9.0)
    b = fb.reverse_dcf_growth(60e9, 2.0e9, 9.0)
    assert b > a, "a higher price must imply higher embedded growth"


def test_unavailable_reverse_dcf_scores_neutral_and_renders_blank():
    from engines import dashboard_adapter as da

    f = sc.Fundamentals(symbol="X", name="X", roic_5y=20.0, wacc=9.0,
                        revenue_cagr_5y=16.5, fcf_margin=12.0,
                        ev_ebit=15.0, ev_ebit_median_10y=18.0)
    f.reverse_dcf_unavailable = True
    res = sc.score_quality_value(f)
    assert res["expectations_gap"] == 0.0, "gap computed against a placeholder"
    assert res["components"]["reverse_dcf_gap"] == 50.0

    row = da.to_rows("value", [(f, res)])[0]
    assert row["impl"] is None and row["gap"] is None, "rendered a placeholder"


def test_min_score_is_set_per_screen():
    """
    One number for three screens, set when the only 'data' was invented sample
    rows. Recovery's highest achievable score across 1,500 names is 68, so a 60
    cut sits at 88% of the observed ceiling.
    """
    import run_all
    assert run_all.MIN_SCORE["recovery"] < run_all.MIN_SCORE["value"]
    assert run_all.MIN_SCORE["recovery"] <= 55


# ---------- eighteenth: the 1,500 run, WACC and the denominator class

def test_wacc_varies_by_sector():
    """
    A flat 9% is not a minor input to a reverse DCF — it is the thing being
    inverted. A 2-point error moves implied growth ~4.5 points, the same size
    as the signal, and it has a DIRECTION: it overcharges low-beta utilities so
    they screen expensive, and undercharges high-beta software so it screens
    cheap. Backwards for a value screen.
    """
    assert fb.SECTOR_WACC["utility"] < fb.SECTOR_WACC["general"]
    assert fb.SECTOR_WACC["reit"] < fb.SECTOR_WACC["general"]
    assert all(w >= fb.MIN_WACC for w in fb.SECTOR_WACC.values()), \
        "a WACC at or under terminal growth makes the DCF unsolvable"

    flat = fb.reverse_dcf_growth(50e9, 2e9, 9.0)
    util = fb.reverse_dcf_growth(50e9, 2e9, fb.SECTOR_WACC["utility"])
    assert flat - util > 5, "the tilt this is meant to remove"


def test_negative_book_equity_voids_roic():
    """
    Otis carries NEGATIVE equity of -$5.35bn from its spinoff; netting it
    against $7.12bn of debt and $1.10bn of cash leaves $0.66bn of invested
    capital and a published ROIC of 204.2%. Deckers' was $0.59bn — the ratio
    guard sat between them. Buyback-heavy and spun-off companies break the
    equity+debt-cash definition structurally, however clean the inputs.
    """
    def mk(tag, yrs, val):
        return {tag: {"units": {"USD": [
            {"end": f"{y}-12-31", "start": f"{y}-01-01", "filed": f"{y + 1}-02-15",
             "fy": y, "fp": "FY", "form": "10-K", "val": val} for y in yrs]}}}

    us = {}
    us.update(mk("Revenues", range(2020, 2027), 14.2e9))
    us.update(mk("OperatingIncomeLoss", range(2020, 2027), 2.3e9))
    us.update(mk("Assets", range(2020, 2027), 10.4e9))
    us.update(mk("StockholdersEquity", range(2020, 2027), -5.35e9))
    us.update(mk("CashAndCashEquivalentsAtCarryingValue", range(2020, 2027), 1.10e9))
    us.update(mk("LongTermDebtNoncurrent", range(2020, 2027), 7.12e9))

    f = fb.build("OTIS", {"entityName": "Otis", "facts": {"us-gaap": us}})
    assert f.roic_unavailable is True, "published 204% on negative equity"
    assert f.roic_ttm == 0.0


def test_implausible_roic_is_voided():
    """Any ROIC above 100% on a business of scale is a denominator artifact."""
    def mk(tag, yrs, val):
        return {tag: {"units": {"USD": [
            {"end": f"{y}-12-31", "start": f"{y}-01-01", "filed": f"{y + 1}-02-15",
             "fy": y, "fp": "FY", "form": "10-K", "val": val} for y in yrs]}}}
    us = {}
    us.update(mk("Revenues", range(2020, 2027), 5e9))
    us.update(mk("OperatingIncomeLoss", range(2020, 2027), 1.5e9))
    us.update(mk("Assets", range(2020, 2027), 4e9))
    us.update(mk("StockholdersEquity", range(2020, 2027), 1.9e9))
    us.update(mk("CashAndCashEquivalentsAtCarryingValue", range(2020, 2027), 1.0e9))
    us.update(mk("LongTermDebtNoncurrent", range(2020, 2027), 0.05e9))
    f = fb.build("X", {"entityName": "X", "facts": {"us-gaap": us}})
    assert f.roic_unavailable is True


def test_offcycle_periods_do_not_fake_a_dividend_cut():
    """
    Eaton reads "cut dividend 9y ago" and has never cut. Two stray points —
    quarterly amounts carrying an ANNUAL-length period context, dated two
    MONTHS off the December year end — defeat the +/-7 day dedup tolerance.
    """
    idx = pd.to_datetime(["2014-02-26", "2014-12-31", "2015-12-31", "2016-12-31",
                          "2017-02-22", "2017-12-31", "2018-12-31", "2019-12-31",
                          "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31",
                          "2024-12-31", "2025-12-31"])
    vals = [0.49, 2.28, 2.28, 2.28, 0.60, 2.40, 2.64, 2.84, 2.92, 3.04, 3.24,
            3.44, 3.76, 4.16]
    s = pd.Series(vals, index=idx)

    assert fb.dividend_record(s)["years_since_cut"] < 20, "guard: raw shows a cut"
    cleaned = fb.drop_offcycle_periods(s)
    assert len(cleaned) == 12
    assert fb.dividend_record(cleaned)["years_since_cut"] == 99, "phantom cut"


def test_offcycle_drop_keeps_a_consistent_offbeat_year_end():
    """A filer whose year ends in June must not have its whole series dropped."""
    idx = pd.to_datetime([f"{y}-06-30" for y in range(2016, 2026)])
    s = pd.Series([1.0 + 0.1 * i for i in range(10)], index=idx)
    assert len(fb.drop_offcycle_periods(s)) == 10


def test_broker_dealers_are_altman_exempt_like_banks():
    """Raymond James read Z 0.77 and was not exempt while every bank was."""
    assert fs.altman_exempt(6211) is True, "broker-dealer"
    assert fs.altman_exempt(6022) is True, "depository"
    assert fs.altman_exempt(6199) is False, "crypto miner — must stay gated"


def test_splits_come_from_the_history_call():
    """A second yfinance round-trip per ticker was 19.8% of a 41.8-minute run."""
    import inspect
    src = inspect.getsource(fs.yfinance_ohlcv)
    assert "actions=True" in src
    assert "_SPLIT_CACHE" in src
    assert "_SPLIT_CACHE" in inspect.getsource(fs.equity_splits)


# ---------- nineteenth: the denominator framework and the inverted trap

def test_ev_ebit_bound_voids_rather_than_zeroes():
    """
    The inverted trap, introduced while fixing the last one. Bounding by
    ZEROING turned CoStar's 2,998x artifact into 0.0x — the cheapest possible
    stock. 286 names carried an unvoided 0.0, 33 a negative, and three were
    published in the recovery table.
    """
    f = sc.Fundamentals(symbol="CSGP", name="CoStar", roic_5y=14.0, wacc=9.0,
                        revenue_cagr_5y=12.0)
    f.ev_ebit = None
    f.ev_ebit_implausible = True
    assert any("denominator not a valuation" in g for g in sc.quality_gates(f))

    from engines import dashboard_adapter as da
    row = da.to_rows("value", [(f, sc.score_quality_value(f))])[0]
    assert row["ev"] is None, "rendered as the cheapest possible stock"


def test_quarterly_stragglers_caught_by_magnitude_not_date():
    """
    Third instance of one failure on a third date offset: J&J one day off,
    Eaton two months off, Cognex in early October against a December year end.
    The calendar filter cannot catch the third. The invariant that holds across
    all three is MAGNITUDE — a value roughly a quarter of its neighbours is a
    quarterly figure whatever date it carries.
    """
    idx = pd.to_datetime(["2020-12-31", "2021-10-01", "2021-12-31", "2022-10-02",
                          "2022-12-31", "2023-10-01", "2023-12-31", "2024-09-29",
                          "2024-12-31", "2025-12-31"])
    s = pd.Series([0.245, 0.060, 0.255, 0.065, 0.265, 0.070, 0.275, 0.075,
                   0.285, 0.295], index=idx)

    assert fb.dividend_record(s)["years_since_cut"] < 10, "guard: raw shows a cut"
    cleaned = fb.drop_dividend_outliers(s)
    assert len(cleaned) == 6, f"caught {10 - len(cleaned)} of 4 — iterate"
    assert fb.dividend_record(cleaned)["years_since_cut"] == 99


def test_magnitude_filter_spares_a_genuine_halving():
    """A real cut lands near 0.5 of its neighbours; a quarterly figure near 0.25."""
    mmm = pd.Series([4.70, 5.44, 5.76, 5.88, 5.92, 5.96, 6.00, 2.86, 2.90, 2.94],
                    index=pd.to_datetime([f"{y}-12-31" for y in range(2016, 2026)]))
    assert len(fb.drop_dividend_outliers(mmm)) == 10, "adjusted a real cut away"
    assert fb.dividend_record(fb.drop_dividend_outliers(mmm))["years_since_cut"] <= 3


def test_cancellation_check_is_necessary_but_not_sufficient():
    """
    The hypothesis was that one rule unified all four bounds. It does not, and
    Trex is the proof: c = 0.93, comparable to Microsoft, and it still published
    a 116.9% ROIC. Cancellation explains the negative-equity cases and says
    nothing about the rest.
    """
    otis = fb.denominator_reliability(-5.35e9 + 7.12e9 - 1.10e9,
                                      terms={"equity": -5.35e9, "debt": 7.12e9,
                                             "cash": 1.10e9})
    assert otis["reliable"] is False and otis["cancellation_ratio"] < 0.25

    msft = fb.denominator_reliability(2.9e11 + 6.0e10 - 7.5e10,
                                      terms={"equity": 2.9e11, "debt": 6.0e10,
                                             "cash": 7.5e10})
    assert msft["reliable"] is True

    trex = fb.denominator_reliability(1.2e9 + 0.1e9 - 0.18e9,
                                      terms={"equity": 1.2e9, "debt": 0.1e9,
                                             "cash": 0.18e9})
    assert trex["reliable"] is True, "the counter-example that killed one rule"


def test_materiality_check_separates_cleanly():
    """
    EBIT is a residual of revenue, so EV/EBIT explodes as margin approaches
    zero. This IS a clean separation, and it is cause-level where a 150x cap is
    symptom-level: CoStar's 2,998x is not an expensive stock, it is a 2.2%
    margin.
    """
    for rev, ebit, ok in [(2.7e9, 0.059e9, False), (0.45e9, 0.010e9, False),
                          (65e9, 9.6e9, True), (0.78e9, 0.315e9, True)]:
        r = fb.denominator_reliability(ebit, residual_of=(rev, "revenue"))
        assert r["reliable"] is ok, f"margin {ebit / rev:.1%}"


def test_near_zero_margin_gates_with_the_cause():
    f = sc.Fundamentals(symbol="CSGP", name="C", roic_5y=14.0, wacc=9.0)
    f.ebit_margin_check = fb.denominator_reliability(0.059e9,
                                                     residual_of=(2.7e9, "revenue"))
    gates = sc.quality_gates(f)
    assert any("materiality" in g and "revenue" in g for g in gates), \
        "gated on the symptom rather than the cause"


# ---------- twentieth: the adapter undoing the builder's void

def test_pct_preserves_none_rather_than_zeroing_it():
    """
    The builder enforced the void and the adapter undid it one layer out:
    `round(float(v or 0), nd)` turned every voided field into 0.0, which
    renders as the cheapest possible value on a multiple. Only `ev` was guarded
    explicitly, so nothing published hit it — but any future voided field
    flowing through here would have.

    Invisible to the written-and-read meta-test: both a write and a read exist.
    The failure is in the VALUE, not the wiring.
    """
    from engines.dashboard_adapter import _pct
    assert _pct(None) is None, "a voided field rendered as 0.0"
    assert _pct(0) == 0.0, "a real zero must survive"
    assert _pct(12.345) == 12.3
    assert _pct("nonsense") is None


def test_no_adapter_column_turns_a_void_into_zero():
    """Every numeric column must carry None through, not just the guarded ones."""
    from engines import dashboard_adapter as da

    f = sc.Fundamentals(symbol="X", name="X", roic_5y=20.0, wacc=9.0,
                        revenue_cagr_5y=8.0)
    f.ev_ebit = None
    f.fcf_yield = None
    row = da.to_rows("value", [(f, sc.score_quality_value(f))])[0]
    assert row["ev"] is None
    assert row["fcfy"] is None, "an unguarded voided column rendered as 0.0"


def test_bundled_specials_are_not_read_as_a_cut():
    """
    The third category. Cognex shows $2.2250 in 2020 against $0.2450 in 2021,
    a ratio of 37 — a regular dividend bundled with a special, which is neither
    a cut nor a split. Magnitude-below alone cannot separate it from a cut, so
    the filter is symmetric: a value far ABOVE the running median is a bundled
    special and the following year is not a reduction.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2017, 2026)])
    cgnx = pd.Series([0.205, 0.215, 0.225, 2.2250, 0.2450, 0.255, 0.265,
                      0.275, 0.285], index=idx)

    assert fb.dividend_record(cgnx)["years_since_cut"] < 10, "guard: raw shows a cut"
    cleaned = fb.drop_dividend_outliers(cgnx)
    assert len(cleaned) == 8, "the bundled special was not removed"
    assert fb.dividend_record(cleaned)["years_since_cut"] == 99


def test_symmetric_filter_still_spares_a_real_cut_and_a_real_raise():
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2017, 2026)])
    mmm = pd.Series([4.70, 5.44, 5.76, 5.88, 5.92, 5.96, 6.00, 2.86, 2.90],
                    index=idx)
    assert fb.dividend_record(fb.drop_dividend_outliers(mmm))["years_since_cut"] <= 2

    # A large but legitimate raise — under the 2.5x bar — must survive.
    riser = pd.Series([1.0, 1.2, 1.5, 1.9, 2.4, 3.0, 3.8, 4.7, 5.9], index=idx)
    assert len(fb.drop_dividend_outliers(riser)) == 9


# ---------- twenty-first: the void, one layer below where it was fixed

def test_void_sets_none_in_the_builder_not_zero():
    """
    `_pct` was fixed in the adapter and `void_derived_fields` still wrote 0.0 —
    so the FIELD said unavailable while the VALUE said zero, across 1,284
    fields. Trade Desk listed ev_ebit, altman_z and net_debt_ebitda as voided
    with all three holding 0.0. The invariant held in the adapter and not in
    the builder, one layer below where it was fixed.
    """
    f = sc.Fundamentals(symbol="TTD", name="Trade Desk", ev_ebit=7.25,
                        altman_z=1.95, net_debt_ebitda=0.4, roic_ttm=18.0,
                        roic_5y=17.0)
    f.debt_unavailable = True
    f.roic_unavailable = True
    fb.void_derived_fields(f)

    for name in ("ev_ebit", "altman_z", "net_debt_ebitda", "roic_ttm", "roic_5y"):
        assert getattr(f, name) is None, f"{name} voided to a value"


def test_business_gates_do_not_compare_against_unknown():
    """
    Voided fields hold None, so every comparison must say what None means
    rather than crash or coerce. An unknown cannot FAIL a business gate —
    data_quality_gates already excluded the name for the reason it is unknown.
    """
    f = sc.Fundamentals(symbol="X", name="X")
    f.net_debt_ebitda = None
    f.altman_z = None
    f.interest_coverage = None
    f.fcf_payout = None

    for gates in (sc.quality_gates, sc.recovery_gates, sc.dividend_gates):
        out = gates(f)                          # must not raise
        assert not any("None" in g for g in out), \
            f"{gates.__name__} rendered None into a gate message"

    assert sc._below(None, 5) is False
    assert sc._above(None, 5) is False
    assert sc._below(3, 5) is True


def test_mean_available_omits_rather_than_substitutes():
    """
    `_scale(None) -> 50.0` is a neutral SUBSTITUTION, which is what made
    missing data outscore a real 55% gross margin. This is the intended
    mechanism wherever an input is legitimately absent; `_scale`'s handling
    stays only as a last-resort net.
    """
    assert sc._mean_available([90, None]) == 90.0, "dragged toward a substitute"
    assert sc._mean_available([80, None, 20]) == 50.0
    assert sc._mean_available([None, None]) == 50.0, "nothing present -> neutral"


def test_specials_are_counted_so_the_question_is_answerable():
    """
    "How many were bundled specials rather than quarterly stragglers" could not
    be answered, because there was no field. The same written-and-read lesson,
    applied to a number rather than a flag.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2017, 2026)])
    with_special = pd.Series([0.205, 0.215, 0.225, 2.2250, 0.2450, 0.255,
                              0.265, 0.275, 0.285], index=idx)
    cleaned = fb.drop_dividend_outliers(with_special)
    assert cleaned.attrs["specials_dropped"] == 1
    assert cleaned.attrs["outliers_dropped"] == 1

    quarterly = pd.Series([0.245, 0.060, 0.255, 0.065, 0.265, 0.070, 0.275,
                           0.075, 0.285], index=idx)
    c2 = fb.drop_dividend_outliers(quarterly)
    assert c2.attrs["specials_dropped"] == 0
    assert c2.attrs["outliers_dropped"] > 0


def test_scope_paragraph_covers_every_excluded_bloc():
    """
    It claimed utilities, REITs AND banks were excluded while citing evidence
    for only "131 utilities and REITs" — the 176 financials were not in the
    count, and they are excluded for a DIFFERENT reason.
    """
    from pathlib import Path
    html = (Path(sc.__file__).parent.parent / "index.html").read_text()
    block = html[html.index("value:{"):]
    scope = block[block.index("scope:"):block.index("method:")]
    assert "131" in scope and "176" in scope, "a bloc is claimed without evidence"
    assert "tangible book" in scope and "AFFO" in scope, "no anchor named"


def test_funnel_is_on_the_page_not_only_in_the_json():
    """27 rows means nothing without 1,500 considered and 1,405 built."""
    from pathlib import Path
    html = (Path(sc.__file__).parent.parent / "index.html").read_text()
    assert "e.funnel" in html
    for k in ("universe", "built", "data_gated", "business_gated", "passed"):
        assert f"funnel.{k}" in html, f"{k} not rendered"
