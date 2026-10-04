"""
On-disk cache for the backtest's inputs.

The backtest rebuilds the same universe at ~23 as-of dates. The INPUTS are
identical for every one of them — `companyfacts` carries its own `filed` dates
so one fetch serves every as-of, and a price frame is full history that each
build slices. Re-fetching per date would be 23 network runs of ~35 minutes;
fetched once, the rebuilds are pure compute.

This caches raw provider responses, not derived values. Nothing here decides
anything: `extract_series(as_of=...)` and `build(as_of=...)` do the
point-in-time work on whatever they are handed, so a cached fetch and a live
one produce the same record.

Two deliberate properties:

Stored as pickle, not parquet: the project is pandas/numpy/requests only and
pyarrow is not a dependency. A local cache needs lossless dtypes and speed,
not a portable archive format.

  * A corrupt or unreadable cache entry RAISES rather than returning empty. A
    truncated price frame is indistinguishable from a short listing, and a
    short listing is exactly how the percentile measures silently change
    meaning.
  * The fetch date is recorded per symbol. A cache built in October and reused
    in March would quietly shift every forward-return window, and the report
    has to be able to say when its inputs were pulled.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import free_sources as free

CACHE = Path("data/backtest_cache")
FACTS = CACHE / "facts"
PRICES = CACHE / "prices"
MANIFEST = CACHE / "manifest.json"


def _ensure_dirs() -> None:
    for d in (CACHE, FACTS, PRICES):
        d.mkdir(parents=True, exist_ok=True)


def _safe(symbol: str) -> str:
    """A filename for a ticker. BRK/B and YCY/U are real symbols."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol.upper())


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            pass
    return {"fetched_on": {}, "failed": {}, "built_at": None}


def _save_manifest(m: dict) -> None:
    _ensure_dirs()
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True))


# ------------------------------------------------------------------- write

def warm(symbols: list[str], *, progress_every: int = 100,
         skip_existing: bool = True) -> dict:
    """
    Fetch facts and prices for `symbols` once and store them.

    Per-symbol failure is recorded and does not stop the sweep — a delisted or
    unmapped ticker is a FACT about the universe and the backtest needs the
    count, not an exception. Total failure of a provider still raises, via
    free_sources.
    """
    _ensure_dirs()
    m = load_manifest()
    fetched, failed = m["fetched_on"], m["failed"]
    today = date.today().isoformat()

    for i, sym in enumerate(symbols, 1):
        if progress_every and i % progress_every == 0:
            print(f"  cache: {i}/{len(symbols)} "
                  f"({len(fetched)} cached, {len(failed)} unavailable)")
        if skip_existing and (sym in fetched or sym in failed):
            continue
        why = []
        fp = FACTS / f"{_safe(sym)}.json"
        try:
            fp.write_text(json.dumps(free.company_facts(sym)))
        except Exception as e:                       # noqa: BLE001
            why.append(f"facts: {str(e)[:90]}")
        pp = PRICES / f"{_safe(sym)}.pkl"
        try:
            px = free.equity_ohlcv(sym)
            sp = free.equity_splits(sym)
            px.to_pickle(pp)
            if sp is not None and len(sp):
                sp.rename("split").to_frame().to_pickle(
                    PRICES / f"{_safe(sym)}.splits.pkl")
        except Exception as e:                       # noqa: BLE001
            why.append(f"prices: {str(e)[:90]}")

        if why:
            failed[sym] = "; ".join(why)
            fetched.pop(sym, None)
        else:
            fetched[sym] = today
            # A symbol that succeeds must LEAVE the failed list. `failed` is
            # the unreachable-price count the survivorship bias is reported
            # from, so a stale entry from an earlier sweep would inflate the
            # bias with names that are actually present — and a symbol in both
            # lists has no defined meaning at all.
            failed.pop(sym, None)
        if i % 250 == 0:
            _save_manifest(m)

    m["built_at"] = today
    _save_manifest(m)
    print(f"  cache: {len(fetched)} symbols cached, {len(failed)} unavailable")
    return m


# -------------------------------------------------------------------- read

def has(symbol: str) -> bool:
    return (FACTS / f"{_safe(symbol)}.json").exists() and \
           (PRICES / f"{_safe(symbol)}.pkl").exists()


def facts(symbol: str) -> dict:
    fp = FACTS / f"{_safe(symbol)}.json"
    if not fp.exists():
        raise FileNotFoundError(f"{symbol}: no cached facts")
    try:
        return json.loads(fp.read_text())
    except json.JSONDecodeError as e:
        # Never fall back to a live fetch here. A half-written cache entry
        # would otherwise mix one symbol's fresh fetch into a run whose other
        # inputs are weeks old, and nothing downstream could see it.
        raise RuntimeError(f"{symbol}: cached facts are corrupt ({e})") from e


def prices(symbol: str) -> pd.DataFrame:
    pp = PRICES / f"{_safe(symbol)}.pkl"
    if not pp.exists():
        raise FileNotFoundError(f"{symbol}: no cached prices")
    df = pd.read_pickle(pp)
    if df.empty or "close" not in df:
        raise RuntimeError(f"{symbol}: cached prices are unusable")
    return df


def splits(symbol: str) -> pd.Series | None:
    sp = PRICES / f"{_safe(symbol)}.splits.pkl"
    if not sp.exists():
        return None
    df = pd.read_pickle(sp)
    return df["split"] if "split" in df and len(df) else None
