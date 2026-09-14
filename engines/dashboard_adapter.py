"""
Screener output -> the row shape index.html actually renders.

These two halves have never met. `run_screen` emits
`{symbol, score, components, gates_failed, ...}`; the dashboard's column
definitions ask for `{ticker, name, roic, fcfy, ev, evp, impl, act, gap, up}`
plus `comp`, `note` and `facts`. Wiring the pipeline straight through would
have produced a table of correctly-ranked blanks, and the obvious diagnosis
would have been "the screener returned nothing".

`test_dashboard_contract_is_satisfied` asserts every column key declared in
index.html is produced here, so the contract breaks at test time rather than
on screen.
"""

from __future__ import annotations

from engines.screeners import Fundamentals, data_quality_report


def _pct(v, nd=1):
    """
    None stays None. `round(float(v or 0), nd)` silently turned every voided
    field into 0.0 — the builder enforced the void and the adapter undid it,
    one layer out, and 0.0 renders as the cheapest possible value on a
    multiple. Only `ev` was guarded explicitly, so nothing published hit it,
    but any future voided field flowing through here would have.

    Invisible to the written-and-read meta-test, because both a write and a
    read exist. The failure is in the VALUE, not the wiring.
    """
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _int(v):
    """`int(round(v))` that leaves a voided field voided instead of raising."""
    v = _pct(v, 0)
    return None if v is None else int(v)


def _txt(v, unit="", nd=1):
    """A number inside prose or a fact cell. A voided field reads as a dash,
    never as "None%" and never as a zero."""
    v = _pct(v, nd)
    return "\u2014" if v is None else f"{v}{unit}"


def _diff(a, b):
    """a - b, unavailable if either side is."""
    return None if a is None or b is None else a - b


def _facts(pairs):
    """Four label/value pairs for the expandable detail row."""
    return [[k, v] for k, v in pairs][:4]


def _note(f: Fundamentals, res: dict, thesis: str) -> str:
    """
    The engine's own reasoning, rendered as the detail-panel prose.

    Data-quality caveats are appended rather than hidden: a name that ranks
    despite a degraded input should say so where it is read, not only in a log.
    """
    parts = [f"<p>{thesis}</p>"]

    flagged = [k for k, v in data_quality_report(f)["flags"].items() if v]
    if flagged:
        parts.append("<p><b>Data caveats:</b> "
                     + ", ".join(k.replace("_", " ") for k in flagged) + ".</p>")
    if f.voided_fields:
        parts.append(f"<p><b>Voided as unavailable:</b> "
                     f"{', '.join(f.voided_fields[:6])}.</p>")
    if res.get("gates_failed"):
        parts.append("<p><b>Gate failures:</b> "
                     + "; ".join(res["gates_failed"][:3]) + ".</p>")
    return "".join(parts)


def _comp(res: dict, order: list[str]) -> list[int]:
    """Component scores in the order the engine's `comps` list declares."""
    c = res.get("components", {})
    return [int(round(c.get(k, 0))) for k in order]


# Component keys in the order each engine's `comps` array expects.
COMPONENT_ORDER = {
    "value": ["quality", "discount_to_own_history", "reverse_dcf_gap",
              "fcf_yield", "fundamental_momentum"],
    "dividend": ["yield_vs_own_history", "dividend_safety", "growth_durability",
                 "valuation", "balance_sheet"],
    "recovery": ["valuation_gap", "durability", "reacceleration",
                 "insider_and_buyback", "technical_base"],
    "financial": ["returns", "capital_strength", "valuation_vs_own_history",
                  "growth_durability", "shareholder_return"],
}


def value_row(f: Fundamentals, res: dict) -> dict:
    return {
        "ticker": f.symbol, "name": f.name, "score": int(res["score"]),
        "roic": _pct(f.roic_5y), "fcfy": _pct(f.fcf_yield, 2),
        "ev": None if f.ev_ebit in (None, 0) else _pct(f.ev_ebit), "evp": _int(f.ev_ebit_percentile_10y),
        # Show nothing rather than a number derived from a placeholder.
        "impl": None if f.reverse_dcf_unavailable else _pct(f.reverse_dcf_implied_growth),
        "act": _pct(f.revenue_cagr_5y),
        "gap": None if f.reverse_dcf_unavailable else _pct(res.get("expectations_gap", 0)),
        "up": (None if (f.ev_history_degraded
                        or res.get("upside_to_own_median") is None)
               else int(round(res["upside_to_own_median"]))),
        "comp": _comp(res, COMPONENT_ORDER["value"]),
        "note": _note(f, res,
                      f"Reverse-DCF implies {_pct(f.reverse_dcf_implied_growth)}% "
                      f"growth against {_txt(f.revenue_cagr_5y, '%')} delivered over "
                      f"five years. ROIC of {_txt(f.roic_5y, '%')} against an assumed "
                      f"{_txt(f.wacc, '%')} cost of capital."),
        "facts": _facts([
            ("ROIC vs WACC", _txt(_diff(f.roic_5y, f.wacc), "pts")),
            ("Share count 5y", f"{_pct(f.share_count_cagr_5y)}%"),
            ("EV/EBIT pctile", _txt(_int(f.ev_ebit_percentile_10y), "th", 0)),
            ("Sector", f.sector),
        ]),
    }


def dividend_row(f: Fundamentals, res: dict) -> dict:
    return {
        "ticker": f.symbol, "name": f.name, "score": int(res["score"]),
        "yld": _pct(f.dividend_yield, 2), "yz": _pct(res.get("yield_z", 0), 2),
        "med": _pct(f.yield_median_5y, 2), "cagr": _pct(f.dps_cagr_5y),
        "chow": _pct(res.get("chowder", 0)), "pay": _int(f.fcf_payout),
        "streak": int(f.increase_streak_years), "nd": _pct(f.net_debt_ebitda, 2),
        "comp": _comp(res, COMPONENT_ORDER["dividend"]),
        "note": _note(f, res,
                      f"Yield of {_pct(f.dividend_yield, 2)}% against a five-year "
                      f"median of {_pct(f.yield_median_5y, 2)}%, with "
                      f"{f.increase_streak_years} consecutive years of increases. "
                      "Note the XBRL horizon caps streaks near 18 years."),
        "facts": _facts([
            ("EPS payout", _txt(_int(f.eps_payout), "%", 0)),
            ("Interest coverage", _txt(f.interest_coverage, "x")),
            ("Years since cut", str(f.years_since_cut)),
            ("Sector", f.sector),
        ]),
    }


def recovery_row(f: Fundamentals, res: dict) -> dict:
    path = res.get("path", {})
    return {
        "ticker": f.symbol, "name": f.name, "score": int(res["score"]),
        "dd": int(round(f.drawdown_from_ath)),
        "upside": round(float(path.get("total_multiple", 0)), 1),
        "rev": _pct(f.revenue_cagr_5y), "gm": _int(f.gross_margin),
        "fcf": _pct(f.fcf_margin), "runway": int(round(f.cash_runway_quarters)),
        "evs": _int(f.ev_sales_percentile_5y), "z": _pct(f.altman_z),
        "comp": _comp(res, COMPONENT_ORDER["recovery"]),
        "legs": [[k.title(), f"{v}% of the move"]
                 for k, v in path.get("leg_share", {}).items()],
        "note": _note(f, res,
                      f"Down {int(round(f.drawdown_from_ath))}% from the high. "
                      f"Base case {round(float(path.get('total_multiple', 0)), 1)}x, "
                      f"of which {path.get('leg_share', {}).get('multiple', 0)}% "
                      "is multiple re-rating."),
        "facts": _facts([
            ("Altman Z", "n/a" if f.altman_not_applicable else _txt(f.altman_z)),
            ("FCF runway", _txt(_int(f.cash_runway_quarters), "q", 0)),
            ("Net debt/EBITDA", _txt(f.net_debt_ebitda, "x", 2)),
            ("Sector", f.sector),
        ]),
    }


def financial_row(f: Fundamentals, res: dict) -> dict:
    """
    Financials rank on ROE against cost of equity and price to tangible book
    against their own history. EV/EBIT and ROIC have no meaning where debt is
    raw material rather than financing.
    """
    return {
        "ticker": f.symbol, "name": f.name, "score": int(res["score"]),
        "roe": _pct(f.roe_5y), "rotce": _pct(f.rotce),
        "ea": _pct(f.equity_to_assets),
        "ptbv": _pct(f.price_to_tangible_book, 2),
        "ptbvp": _int(f.ptbv_percentile_10y),
        "spread": _pct(res.get("roe_spread")),
        "tbv": _pct(f.tbvps_cagr_5y),
        "sub": f.financial_subtype or "—",
        "comp": _comp(res, COMPONENT_ORDER["financial"]),
        "note": _note(f, res,
                      f"ROE of {_txt(f.roe_5y, '%')} against an assumed "
                      f"{_txt(f.cost_of_equity, '%')} cost of equity, on "
                      f"{_txt(f.equity_to_assets, '%')} equity-to-assets. Priced at "
                      f"{_txt(f.price_to_tangible_book, 'x', 2)} tangible book."),
        "facts": _facts([
            ("ROE less cost of equity", _txt(res.get("roe_spread"), "pts")),
            ("ROTCE", "n/a" if f.rotce is None else _txt(f.rotce, "%")),
            ("TBVPS 5y CAGR", _txt(f.tbvps_cagr_5y, "%")),
            ("Sub-bucket", f.financial_subtype or "unresolved"),
        ]),
    }


BUILDERS = {"value": value_row, "dividend": dividend_row,
            "recovery": recovery_row, "financial": financial_row}


def to_rows(engine: str, scored: list[tuple[Fundamentals, dict]]) -> list[dict]:
    """`scored` is (Fundamentals, score_result) pairs, already gate-filtered."""
    build = BUILDERS[engine]
    return [build(f, res) for f, res in scored]
