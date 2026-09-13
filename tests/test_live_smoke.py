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
