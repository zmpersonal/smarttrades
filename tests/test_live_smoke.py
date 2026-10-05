"""
Live network smoke tests.

The rest of the suite runs against fixtures, deliberately: parsing and
point-in-time logic should be testable without a network. But the HTTP paths
themselves have their own failure modes, and every one this project has hit so
far was invisible offline — a silently truncated Coin Metrics window, a ticker
eaten by pandas' NA coercion, a trailer row, a geo-block.

These are opt-in so CI and the normal `pytest tests -q` run stay hermetic and
fast. Enable with:

    SMOKE_LIVE=1 python -m pytest tests/test_live_smoke.py -q -s
"""

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from engines import finra_darkpool as fd

pytestmark = pytest.mark.skipif(
    os.environ.get("SMOKE_LIVE") != "1",
    reason="live network test; set SMOKE_LIVE=1 to run",
)

VOL_COLS = ("oe_short_vol", "oe_short_exempt", "oe_total_vol")


@pytest.fixture(scope="module")
def recent_day():
    """
    Most recent trading day FINRA has published. Walks back rather than
    assuming: the file lands ~6pm ET, so 'today' is empty all morning, and
    holidays 404.
    """
    import requests

    s = requests.Session()
    for i in range(10):
        d = date.today() - timedelta(days=i)
        df = fd.fetch_finra_day(d, s)
        if not df.empty:
            return d, df, s
    pytest.fail("no FINRA file in the last 10 calendar days")


def test_returns_thousands_of_symbols(recent_day):
    d, df, _ = recent_day
    assert len(df) > 5000, f"{d}: only {len(df)} symbols, expected >5000"
    assert df["symbol"].is_unique, "duplicate symbols in a single day's file"


def test_trailer_row_is_dropped(recent_day):
    """
    The file ends with a bare record count. It must not survive as a row, and
    the loader must not drop a real row along with it.
    """
    d, df, s = recent_day
    raw = s.get(fd.FINRA_DAILY.format(d=d.strftime("%Y%m%d")), timeout=30).text
    lines = raw.strip().split("\n")

    assert lines[-1].strip().isdigit(), (
        f"expected a bare record-count trailer, got {lines[-1]!r}")

    # 1 header + N data rows + 1 trailer
    expected = len(lines) - 2
    assert len(df) == expected, (
        f"{d}: loader returned {len(df)} rows, file has {expected} data rows")

    assert not df["symbol"].isna().any()
    assert (df["symbol"] != "").all()
    # The trailer's own value must not have leaked in as a Date.
    assert df["Date"].nunique() == 1 and df["Date"].iloc[0] == pd.Timestamp(d)


def test_na_ticker_survives_pandas_na_coercion(recent_day):
    """
    'NA' is a real NYSE ticker. Without keep_default_na=False, read_csv turns
    it into NaN and the notna() filter drops it from every day's file.
    """
    _, df, _ = recent_day
    assert "NA" in set(df["symbol"]), "ticker NA was eaten by NA coercion"


def test_volumes_parse_as_numbers(recent_day):
    d, df, _ = recent_day
    for c in VOL_COLS:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} is {df[c].dtype}"
        assert df[c].notna().all(), f"{c} has NaNs"
        assert (df[c] >= 0).all(), f"{c} has negative share counts"

    assert (df["oe_total_vol"] > 0).any()
    # Short volume is a subset of total. Allow a hair of float slop.
    over = df[df["oe_short_vol"] > df["oe_total_vol"] * 1.0001]
    assert over.empty, f"short > total for {len(over)} symbols, e.g.\n{over.head()}"

    # Off-exchange short share sits near half; market makers facilitating
    # buyers print as short. A median far from that means a parsing shift.
    ratio = (df["oe_short_vol"] / df["oe_total_vol"]).median()
    assert 0.30 < ratio < 0.70, f"median short/total = {ratio:.3f}, suspect parse"
    print(f"\n  {d}: {len(df)} symbols, "
          f"total off-exchange {df['oe_total_vol'].sum():,.0f} sh, "
          f"median short/total {ratio:.3f}")


# --------------------------------------------------------------- Polygon
# Separate gate: this one needs a credential as well as a network. Get a free
# key at polygon.io (no card), then:
#
#     POLYGON_API_KEY=... SMOKE_LIVE=1 python -m pytest tests/test_live_smoke.py -q -s

needs_polygon = pytest.mark.skipif(
    not os.environ.get("POLYGON_API_KEY"),
    reason="POLYGON_API_KEY unset",
)


@pytest.fixture(scope="module")
def polygon_day():
    """Most recent date polygon has a grouped bar for. Walks back over weekends."""
    from engines import free_sources as fs

    for i in range(1, 8):
        d = date.today() - timedelta(days=i)
        df = fs.polygon_grouped_daily(d)
        if not df.empty:
            return d, df
    pytest.fail("no polygon grouped bars in the last 7 days")


@needs_polygon
def test_polygon_returns_thousands_of_tickers(polygon_day):
    d, df = polygon_day
    assert len(df) > 3000, f"{d}: only {len(df)} tickers"
    assert df["symbol"].is_unique


@needs_polygon
def test_polygon_has_usable_volume(polygon_day):
    d, df = polygon_day
    assert pd.api.types.is_numeric_dtype(df["volume"])
    assert (df["volume"] >= 0).all()
    assert (df["volume"] > 0).sum() > len(df) * 0.8, "most tickers should have traded"

    for c in ("open", "high", "low", "close"):
        assert pd.api.types.is_numeric_dtype(df[c])
    ok = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    assert len(ok) == len(df), "high/low inverted or non-positive close"
    assert df["Date"].nunique() == 1 and df["Date"].iloc[0] == pd.Timestamp(d)
    print(f"\n  polygon {d}: {len(df)} tickers, "
          f"total volume {df['volume'].sum():,.0f} sh")


def test_a_past_as_of_uses_a_past_price():
    """
    `as_of` filtered the EDGAR facts and never reached the price frame, so a
    historical build produced a HYBRID that existed at no point in time.
    AAPL at 2020-06-30 published a market cap of $6.21tn — today's price times
    a 2020 share count, and LARGER than its actual ~$5.0tn today. It also
    voided ev_ebit outright, because the recency guard correctly compared an
    EBIT series ending 2020 against a price index running to 2026, so the
    value screen published NOTHING at any historical date.

    AAPL's real market cap on 2020-06-30 was ~$1.58tn.
    """
    from engines import free_sources as fs, fundamentals_builder as fb

    facts = fs.company_facts("AAPL")
    px, sp = fs.equity_ohlcv("AAPL"), fs.equity_splits("AAPL")
    kw = dict(sector=fs.company_sector("AAPL"), sic=fs.company_sic("AAPL"))

    hist = fb.build("AAPL", facts, px, as_of=date(2020, 6, 30), splits=sp, **kw)
    now = fb.build("AAPL", facts, px, splits=sp, **kw)

    assert 1.2e12 < hist.market_cap < 2.1e12, (
        f"2020 market cap came out {hist.market_cap/1e12:.2f}tn, real ~1.58tn")
    assert hist.market_cap < now.market_cap, "a 2020 build cannot exceed today"
    assert hist.ev_ebit is not None, "ev_ebit voided at a past as_of"
    assert not hist.ev_history_degraded


def test_splits_are_not_truncated_at_the_as_of_date():
    """
    The counter-intuitive half. yfinance retroactively split-adjusts prices
    for EVERY split, including ones after as_of, so the share count must be
    adjusted on the same set. Truncating splits — the obvious "fix" — put
    AAPL's mid-2020 market cap at $0.42tn against $1.70tn, because the 4:1 of
    August 2020 was removed from the counts and left in the prices.
    """
    from engines import free_sources as fs, fundamentals_builder as fb

    facts = fs.company_facts("AAPL")
    px, sp = fs.equity_ohlcv("AAPL"), fs.equity_splits("AAPL")
    kw = dict(sector=fs.company_sector("AAPL"), sic=fs.company_sic("AAPL"))
    as_of = date(2020, 6, 30)

    full = fb.build("AAPL", facts, px, as_of=as_of, splits=sp, **kw)
    cut = fb.build("AAPL", facts, px, as_of=as_of,
                   splits=sp.loc[:str(as_of)], **kw)
    assert cut.market_cap < full.market_cap / 3, (
        "truncating splits should break by roughly the split factor — if this "
        "no longer holds, check whether the price source still back-adjusts")
    assert 1.2e12 < full.market_cap < 2.1e12


def test_the_reuse_guard_holds_against_the_real_bbby_and_sbny():
    """
    The guard asserted against the actual tickers, not a synthetic frame.

    BBBY was delisted in 2023 and SBNY failed in March 2023, and both tickers
    were REUSED — yfinance returns a live series for each, for a different
    company. A backtest is exactly where that survives: joining 2021
    fundamentals to 2026 prices yields a plausible number with no error
    raised anywhere.

    If this test starts failing because the series now covers 2021, the
    provider has changed what it returns and the guard needs re-deriving —
    which is the point of asserting on the live case rather than trusting it.
    """
    from engines import free_sources as fs
    from engines import backtest as bt

    as_of = date(2021, 6, 15)
    checked = 0
    for t in ("BBBY", "SBNY"):
        try:
            px = fs.equity_ohlcv(t)
        except Exception:
            continue                      # cleanly gone is the safe outcome
        checked += 1
        assert not bt.price_covers(px, as_of), (
            f"{t} passed the reuse guard: its series starts "
            f"{px.index.min().date()} and would be joined to 2021 fundamentals")
        assert bt.forward_return(px, as_of, bt.HORIZONS["12m"]) is None, (
            f"{t} produced a 12m return from a reused ticker")
    assert checked, "neither reused ticker resolved; the case is untested"


def test_a_cleanly_delisted_ticker_fails_rather_than_returning_something():
    """The safe half: 8 of 10 delisted tickers return nothing at all."""
    from engines import free_sources as fs

    gone = ["ATVI", "TWTR", "SIVB", "FRC", "CERN", "XLNX", "VMW", "ZNGA"]
    served = []
    for t in gone:
        try:
            fs.equity_ohlcv(t)
            served.append(t)
        except Exception:
            pass
    assert len(served) <= 2, (
        f"{served} now return prices; each needs the reuse guard checked, "
        f"because a delisted ticker that serves data is either a shell or a "
        f"reassignment")
