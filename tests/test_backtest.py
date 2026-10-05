"""
Tests for the backtest harness.

Hermetic: every test here runs on synthetic frames. The one assertion that
genuinely needs the network — that a past `as_of` produces a historical market
cap for a real ticker — lives in test_live_smoke.py.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engines import backtest as bt


def _frame(vals, end="2023-06-01"):
    idx = pd.bdate_range(end=end, periods=len(vals))
    return pd.DataFrame({"close": vals, "volume": [1e6] * len(vals)}, index=idx)


def _lin(a, b, n):
    return list(np.linspace(a, b, n))


# ------------------------------------------------- the ticker-reuse guard

def test_a_reused_ticker_does_not_cover_the_as_of_date():
    """
    The dangerous survivorship case. yfinance returns NOTHING for most
    delisted tickers, which is a clean failure. BBBY returns 55 rows starting
    2026-07-17 and SBNY 535 from 2024-08-15, because both tickers were REUSED
    after the original company died — so joining 2021 fundamentals to them
    produces a plausible number about two different companies with no error
    raised. Presence is not the test; coverage is.
    """
    as_of = date(2021, 6, 15)
    reuse = _frame([2.0] * 55, end="2026-09-30")
    assert not bt.price_covers(reuse, as_of)

    # Dies in 2019, ticker reissued in 2026: covers neither side of 2021.
    dead = _frame(_lin(50, 60, 300), end="2019-06-01")
    assert not bt.price_covers(pd.concat([dead, reuse]), as_of)

    live = _frame(_lin(50, 90, 2200), end="2026-10-01")   # starts 2018
    assert bt.price_covers(live, as_of)


def test_coverage_is_a_reuse_guard_not_a_history_requirement():
    """
    How much history a MEASURE needs is decided where that measure is built —
    `ev_history_degraded`, `panel_sessions_required`. Duplicating it here
    would exclude a recent listing for the wrong reason, and report it as a
    survivorship exit.
    """
    as_of = date(2021, 6, 15)
    recent = _frame(_lin(20, 25, 80), end="2021-06-14")
    assert bt.price_covers(recent, as_of), "one quarter of tape is enough"
    tiny = _frame(_lin(20, 25, 20), end="2021-06-14")
    assert not bt.price_covers(tiny, as_of)


# --------------------------------------------------------- forward returns

def test_a_truncated_forward_window_is_absent_not_flat():
    """
    A window that cannot run its full length is not a short return, it is no
    return. Treating it as one would read a delisting as a flat quarter —
    which is the survivorship bias arriving through the back door, dressed as
    data.
    """
    px = _frame(_lin(100, 110, 60), end="2021-03-01")
    assert bt.forward_return(px, date(2021, 1, 4), bt.HORIZONS["12m"]) is None
    long = _frame(_lin(100, 120, 700), end="2023-06-01")
    r = bt.forward_return(long, date(2021, 6, 15), bt.HORIZONS["3m"])
    assert r is not None and r > 0


def test_return_is_total_not_price():
    """
    A dividend screen measured on price return is measured on everything
    except the component it selects for.
    """
    px = _frame([100.0] * 700, end="2023-06-01")
    as_of = date(2021, 6, 15)
    flat = bt.forward_return(px, as_of, bt.HORIZONS["3m"])
    divs = pd.Series([2.0], index=[pd.Timestamp("2021-07-01")])
    withdiv = bt.forward_return(px, as_of, bt.HORIZONS["3m"], dividends=divs)
    assert abs(flat) < 1e-9
    assert withdiv == pytest.approx(0.02, abs=1e-9)


# ------------------------------------------------------- the exit classifier

@pytest.mark.parametrize("label,vals,want", [
    ("jump to deal price then flat",
     _lin(70, 95, 510) + [118.0] * 120, bt.EXIT_ACQUIRED),
    ("collapse to near zero",
     _lin(60, 40, 430) + _lin(40, 1.2, 200), bt.EXIT_DISTRESS),
    ("steady fade, genuinely ambiguous",
     _lin(100, 65, 630), bt.EXIT_UNKNOWN),
])
def test_exits_split_three_ways_by_where_the_tape_ends(label, vals, want):
    """
    The two exits point in OPPOSITE directions, so the survivorship bias
    cannot be signed until they are told apart: an acquisition pays a premium
    so missing it biases the measured return DOWN, a bankruptcy biases it UP.

    Measured against the symbol's own trailing-year HIGH, not the price a
    fixed quarter earlier — that first version mis-sorted two of these three,
    because a collapse spread over eighteen months has a mild final quarter
    and a steady fade ends close to its own falling year high.
    """
    assert bt.classify_exit(_frame(vals), date(2023, 6, 1)) == want


def test_an_unclassifiable_exit_is_unknown_not_forced():
    assert bt.classify_exit(None, date(2023, 6, 1)) == bt.EXIT_UNKNOWN
    assert bt.classify_exit(_frame([1.0] * 10), None) == bt.EXIT_UNKNOWN
    assert bt.classify_exit(_frame([1.0] * 10), date(2023, 6, 1)) == bt.EXIT_UNKNOWN


# ----------------------------------------------------- the honest sample size

def _paired(horizon, wins, n=23, start=date(2020, 3, 31)):
    p = bt.Paired("value", horizon, "score")
    for i in range(n):
        p.dates.append(start + timedelta(days=91 * i))
        p.screen_mean.append(0.02 if i < wins else -0.01)
        p.null_mean.append(0.0)
        p.n_screen.append(20)
        p.n_null.append(55)
    return p


def test_effective_n_counts_non_overlapping_windows_not_dates():
    """
    Quarterly dates with a 12m horizon overlap 75%, so 23 dates carry about
    five independent observations. Reporting 23 there would imply four and a
    half times the precision the data has.
    """
    assert _paired("3m", 17).effective_n >= 20
    assert _paired("12m", 17).effective_n <= 6
    assert _paired("1m", 17).effective_n == 23


def test_a_thin_horizon_refuses_to_claim_a_result():
    """A null result must be a finding; a five-observation result is neither."""
    v = _paired("12m", 23).verdict()
    assert "not enough to support a claim" in v
    assert "p=" not in v, "a p-value on five observations invites belief"


def test_the_sign_test_needs_seventeen_of_twenty_three():
    assert _paired("3m", 16).sign_test_p() > 0.05
    assert _paired("3m", 17).sign_test_p() < 0.05
    assert "consistent sign" in _paired("3m", 17).verdict()
    assert "no consistent sign" in _paired("3m", 13).verdict()


def test_the_headline_is_not_a_footnote():
    """
    "This can detect a consistent sign, not estimate a magnitude" is the
    result's own error bar, so it is the first thing the report carries.
    """
    assert "consistent sign" in bt.HEADLINE
    assert "not estimate a magnitude" in bt.HEADLINE


# ------------------------------------------------------- the empty-board trap

class _Rec:
    """Minimal stand-in for a scored Fundamentals."""
    def __init__(self, sym):
        self.symbol = sym


def test_preflight_passes_when_every_screen_has_a_board(monkeypatch):
    monkeypatch.setattr(bt, "arms", lambda recs, screen: {
        "published": ["A", "B", "C", "D"], "near_miss": ["E"],
        "gate_clean": ["A", "B", "C", "D", "E"], "universe": ["A", "B", "C", "D", "E", "F"]})
    found = bt.preflight(date(2020, 3, 31), [_Rec("A")])
    # `found` now also carries the diagnostic keys the raise decision reads.
    assert set(bt.SCREENS) <= set(found)
    assert found["value"]["published"] == 4
    assert found["_empty"] == []


# -------------------------------------------------------------- the date grid

def test_the_window_stops_twelve_months_short_of_today():
    """A 12m forward return needs twelve months of tape after the date."""
    d = bt.as_of_dates(today=date(2026, 10, 4))
    assert d[0] >= bt.WINDOW_START
    assert d[-1] <= date(2025, 10, 4)
    assert 20 <= len(d) <= 24, f"{len(d)} quarterly dates"


def test_not_cached_is_not_a_survivorship_fact(monkeypatch):
    """
    The cache holds today's universe, so reading it alone would count every
    2021 symbol we simply never fetched as delisted — turning a cache
    boundary into a finding. An uncached symbol is resolved live, and one
    that cannot be resolved either way is `unresolved`, not an exit.
    """
    from engines import backtest_cache as bc

    def no_cache(sym):
        raise FileNotFoundError(sym)

    monkeypatch.setattr(bc, "prices", no_cache)
    r = bt.survivorship_report(["A", "B", "C"], {}, date(2021, 6, 15),
                               tail=date(2026, 10, 1), sample=None, live=False)
    assert r["unresolved"] == 3
    assert r["unreachable"] == 0, "an uncached name is not a delisting"
    assert sum(r["exits"].values()) == 0


def test_the_sample_size_is_reported_not_implied(monkeypatch):
    from engines import backtest_cache as bc
    monkeypatch.setattr(bc, "prices",
                        lambda s: _frame(_lin(50, 90, 2200), end="2026-10-01"))
    r = bt.survivorship_report([f"S{i}" for i in range(900)], {},
                               date(2021, 6, 15), tail=date(2026, 10, 1),
                               sample=100, live=False)
    assert r["as_of_universe"] == 900 and r["sampled"] == 100
    assert r["reachable"] == 100


def test_build_at_makes_no_live_request(monkeypatch):
    """
    `build_at` documented itself as cache-only and was not: sector and SIC came
    from the SEC submissions endpoint, one request per symbol, lru_cached only
    WITHIN a process. A 23-date sweep across four parallel processes made
    ~1,446 live requests per process on its first date and stalled on the rate
    limiter before a single date finished. A stated guarantee that is not true
    is worse than none, because the sweep was designed around it.
    """
    from engines import free_sources as free

    def boom(*a, **k):
        raise AssertionError("build_at made a live request")

    for name in ("company_sector", "company_sic", "company_submissions",
                 "company_facts", "equity_ohlcv", "equity_splits"):
        monkeypatch.setattr(free, name, boom)
    # No cached symbols: must return empty, not reach for the network.
    recs, skipped = bt.build_at(date(2021, 6, 15), ["NOPE1", "NOPE2"],
                                verbose=False)
    assert recs == []
    assert skipped["no_cache"] == 2


def test_a_symbol_without_cached_meta_is_skipped_not_defaulted(monkeypatch):
    """
    Falling back to sector "general" for an uncached symbol would silently
    change which gates apply — the REIT and utility payout allowances and the
    Altman skip all read sector — so the name is skipped and counted instead.
    """
    from engines import backtest_cache as bc
    monkeypatch.setattr(bc, "has", lambda s: True)
    monkeypatch.setattr(bc, "prices",
                        lambda s: _frame(_lin(50, 90, 2200), end="2026-10-01"))
    monkeypatch.setattr(bc, "load_meta", lambda: {})
    recs, skipped = bt.build_at(date(2021, 6, 15), ["AAA"], verbose=False)
    assert recs == [] and skipped["no_meta"] == 1


def test_a_thin_horizon_withholds_the_number_not_just_the_claim():
    """
    The verdict said "five observations, not enough to support a claim" and
    the table printed p=0.001 beside it, which lets a reader take the number
    and discard the sentence. A refusal in one layer undone by the next — the
    same shape as a void erased by its own caller.
    """
    thin = _paired("12m", 17)
    s = thin.summary()
    assert thin.too_thin and s["sign_test_p"] is None
    assert s["median_excess_withheld"] is True
    rep = {"headline": bt.HEADLINE, "window": {"from": "a", "to": "b",
           "as_of_dates": 19}, "results": [s]}
    out = bt.format_report(rep)
    assert "0.001" not in out and "p=" not in out.split("verdict")[1]

    fat = _paired("3m", 17).summary()
    assert fat["sign_test_p"] is not None


def test_losing_to_the_null_is_not_described_as_beating_it():
    """
    "beat the null on 4 of 19 dates (below)" read as a win while the
    parenthesis said the opposite — on the one line where the direction IS
    the result.
    """
    lost = _paired("3m", 4).verdict()          # 4 wins of 23 dates
    assert "WRONG WAY" in lost and "lost to the null on 19 of 23" in lost
    assert "beat the null" not in lost
    assert "beat the null on 17 of 23" in _paired("3m", 17).verdict()


class _R:
    def __init__(self, sym, ev=10.0, deg=False):
        self.symbol, self.ev_ebit, self.ev_history_degraded = sym, ev, deg


def test_an_empty_board_raises_only_when_the_INPUTS_collapsed(monkeypatch):
    """
    The first version raised on ANY empty board and guessed the cause from
    universe size. That discarded all four 2021 dates and told them the as-of
    path had failed. It had not: ev_ebit was present for 69% of records,
    identically to 2020 and 2022, while discount_to_own_history averaged 11
    against 50 a year later. Mid-2021 was the most expensive market in the
    sample and a value screen SHOULD publish nothing at a top.
    """
    monkeypatch.setattr(bt, "arms", lambda recs, screen: {
        "published": [], "near_miss": ["A"] * 45,
        "gate_clean": ["A"] * 45, "universe": ["A"] * 1228})

    # Inputs sound (69%, as measured in 2021): a real zero, recorded.
    sound = [_R(f"S{i}", ev=10.0 if i < 690 else None) for i in range(1000)]
    found = bt.preflight(date(2021, 6, 30), sound)
    assert found["_input_health"]["sound"] is True
    assert found["_empty"], "the empty board still has to be recorded"

    # Inputs collapsed (the original bug's signature was ZERO): a failure.
    broken = [_R(f"S{i}", ev=None) for i in range(1000)]
    with pytest.raises(bt.EmptyBoardError, match="ranking inputs have collapsed"):
        bt.preflight(date(2021, 6, 30), broken)


def test_a_full_board_never_raises_however_thin_the_inputs(monkeypatch):
    monkeypatch.setattr(bt, "arms", lambda recs, screen: {
        "published": ["A"] * 20, "near_miss": ["B"] * 30,
        "gate_clean": ["A"] * 50, "universe": ["A"] * 1200})
    bt.preflight(date(2021, 6, 30), [_R("X", ev=None) for _ in range(100)])


def test_input_health_witnesses_the_original_failure():
    """ev_ebit is the witness because its absence WAS the original bug."""
    assert bt.input_health([_R("A"), _R("B")])["ev_ebit_rate"] == 1.0
    assert bt.input_health([_R("A", ev=None)])["sound"] is False
    assert bt.input_health([])["records"] == 0
