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

from engines import screeners
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


# --------------------------------------------------------------- why panel
#
# "Why it's on the list", generated. Three rules, and they are the point:
#
# 1. It states the COUNTER-case, not just the case. A generator that only
#    argues for a name is marketing.
# 2. Every numeral traces to a field the engine computed. Each entry carries
#    `vals` naming the fields it used, and a test asserts that every number in
#    the prose appears there — which careful writing cannot satisfy, only
#    actually sourcing each figure can.
# 3. A counter-case from a VOIDED INPUT and one from a WEAK COMPONENT are
#    different sentences and stay in different kinds. "Its weakest component is
#    growth durability at 70" is a judgement about the company; "gross profit
#    is unavailable, so the margin check could not run" is a judgement about
#    the data. A reader acts differently on each, so they never share a kind.
#
# No derived figures in prose. The mockup writes "about 40% above its own
# median", which is 0.21/0.15 rounded for rhythm — exactly the interpolation
# rule 2 forbids. The z-score says the same thing, is already computed, and
# traces to one field.
_KIND = {"case": "case", "counter": "counter", "data": "data"}
# The one threshold quoted in prose. Named rather than inlined so the test can
# accept it as a source alongside the record's own fields.
DIVIDEND_STREAK_MIN = 7
# Components are scored 0-100. Named because it appears in prose, and the test
# accepts a numeral only if the generator declares where it came from — which
# is the whole point: an undeclared constant and an invented one look the same.
COMPONENT_SCALE = 100


def _nums(text):
    """Numerals inside a threshold string like "under 65%"."""
    import re
    return re.findall(r"\d+(?:\.\d+)?", str(text or ""))


def _shown(v):
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


def _cap(t):
    t = str(t or "")
    return t[0].upper() + t[1:] if t else t


def _w(kind, title, detail, vals=None):
    return {"k": kind, "t": title, "d": detail, "vals": vals or []}


def _v(field, shown):
    """
    A number used in prose, with the field it came from. `shown` is the
    DISPLAYED form — _pct returns a float, and comparing a float against the
    text actually rendered would let a formatting difference through.
    """
    return {"field": field, "shown": str(shown)}


# Where each half stops being a claim worth making.
_VAL_HIGH, _VAL_LOW = 70, 40
# The percentile half reads a different field on each screen, and the detail
# page cites the field it actually used.
def _ord(n) -> str:
    """1st, 2nd, 3rd, 11th. `f"{pct}th"` published "3th" and "2th"."""
    i = _int(n)
    if i is None:
        return "n/a"
    suf = "th" if 10 <= i % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


_PCT_FIELD = {"value": "ev_ebit_percentile_10y",
              "recovery": "ev_sales_percentile_5y"}


def _valuation_why(f: Fundamentals, screen: str) -> list[dict]:
    """
    Explain a BLENDED component by naming the half that drives it.

    Four readings, and they are genuinely different stories rather than
    gradations of one. The relative half is `_scale` of the percentile; the
    absolute half is `_scale(30 - ev_ebit, ...)`. Either may be absent, and an
    absent half is stated rather than silently halving the sentence.
    """
    pct = (f.ev_ebit_percentile_10y if screen == "value"
           else f.ev_sales_percentile_5y)
    lo, hi = (20, 95) if screen == "value" else (40, 95)
    rel = None if pct is None else screeners._scale(100 - pct, lo, hi)
    ab = None if f.ev_ebit is None else screeners._scale(30 - f.ev_ebit, 5, 22)
    own = "its own ten-year range" if screen == "value" else "its own five years"

    if rel is None and ab is None:
        return [_w("data", "Valuation could not be measured.",
                   "Neither the multiple nor its history is derivable, so this "
                   "component scored on nothing and the name ranks on the "
                   "others.", [])]
    if ab is None:
        # WHICH half is missing is not enough — WHY it is missing is a
        # different fact about the business each time, and it must be read off
        # the flag that actually voided the field rather than guessed. An
        # earlier draft of this branch told all four published names their
        # "price and earnings history do not overlap"; measured, CSGP's EBIT
        # is NEGATIVE (-$72m, 2.2% of revenue) and AKAM, PINS and ALNY are
        # `debt_unavailable`, so no enterprise value can be formed at all.
        # Three different facts, one of which is the recovery thesis itself.
        # They score identically — the half drops out of the mean — so the
        # prose is the only place the difference can survive.
        chk = f.ebit_margin_check or {}
        ebit = chk.get("value")
        if f.ev_ebit_implausible and ebit is not None and ebit <= 0:
            why = ("EBIT is negative, so there is no multiple to be cheap or "
                   "expensive on — the absolute half does not apply to a "
                   "company not yet earning.")
        elif f.ev_ebit_implausible:
            why = ("EV/EBIT comes out above 150x, which describes a near-zero "
                   "denominator rather than a valuation, so it is voided "
                   "rather than scored.")
        elif f.debt_unavailable:
            why = ("Total debt is not derivable, so no enterprise value can be "
                   "formed and EV/EBIT is voided — the percentile here is on "
                   "EV/sales, which needs no debt.")
        elif f.ev_history_degraded:
            why = ("Its multiple history is too short or too stale to stand "
                   "up, so EV/EBIT is voided rather than ranked.")
        else:
            why = ("EV/EBIT is not derivable, so whether that is cheap in "
                   "absolute terms is unknown.")
        return [_w("data", "Only half the valuation measure is available.",
                   f"It sits in the {_ord(pct)} percentile of {own}. {why}",
                   [_v(_PCT_FIELD[screen], str(_int(pct)))])]
    if rel is None:
        return [_w("case", "Cheap on the multiple.",
                   f"EV/EBIT of {_pct(f.ev_ebit)}x, though its own history is "
                   f"too short or too stale to say whether that is unusual "
                   f"for it.", [_v("ev_ebit", str(_pct(f.ev_ebit)))])]

    vals = [_v("ev_ebit", str(_pct(f.ev_ebit))),
            _v(_PCT_FIELD[screen], str(_int(pct)))]
    both = f"EV/EBIT of {_pct(f.ev_ebit)}x, in the {_ord(pct)} percentile of {own}"
    if rel >= _VAL_HIGH and ab >= _VAL_HIGH:
        return [_w("case", "Cheap on both measures.",
                   f"{both} — inexpensive outright and unusually so for itself.",
                   vals)]
    if rel >= _VAL_HIGH and ab <= _VAL_LOW:
        # The case this whole blend exists to catch.
        return [_w("counter", "The least expensive it has been, not cheap.",
                   f"{both}. The percentile is doing the work here: it is near "
                   f"the bottom of its own range and still a high multiple, so "
                   f"read this as a de-rating rather than a discount.", vals)]
    if ab >= _VAL_HIGH and rel <= _VAL_LOW:
        return [_w("counter", "Cheap outright, expensive against itself.",
                   f"{both}. A low multiple, but the name has usually traded "
                   f"lower still, so its own history argues the other way.",
                   vals)]
    return [_w("counter", "Middling on valuation.",
               f"{both}. Neither half of the measure is making a strong "
               f"claim — this is not where the score comes from.", vals)]


def _why(f: Fundamentals, res: dict, screen: str) -> list[dict]:
    out, comp = [], res.get("components", {}) or {}

    # ---- the case: the components that actually carried the score ----------
    ranked = sorted(comp.items(), key=lambda kv: -kv[1])
    if ranked:
        top, tv = ranked[0]
        out.append(_w("case", f"Strongest on {top.replace('_', ' ')}.",
                      f"It scores {tv} out of {COMPONENT_SCALE} there, its "
                      f"highest component.",
                      [_v(f"components.{top}", str(tv)),
                       _v("component scale", str(COMPONENT_SCALE))]))

    if screen == "dividend":
        if f.yield_std_5y and res.get("yield_z") is not None:
            out.append(_w("case", "Cheaper than usual.",
                          f"The yield is {_pct(f.dividend_yield, 2)}% against its own "
                          f"five-year median of {_pct(f.yield_median_5y, 2)}% — "
                          f"{_pct(res['yield_z'], 1)} standard deviations above it.",
                          [_v("dividend_yield", _pct(f.dividend_yield, 2)),
                           _v("yield_median_5y", _pct(f.yield_median_5y, 2)),
                           _v("yield_z", _pct(res["yield_z"], 1))]))
        if f.dps_cagr_5y is not None and f.dps_cagr_5y >= 5:
            out.append(_w("case", "The dividend is growing.",
                          f"Raised {_pct(f.dps_cagr_5y)}% a year over five years, "
                          f"with {f.increase_streak_years} consecutive years of increases.",
                          [_v("dps_cagr_5y", _pct(f.dps_cagr_5y)),
                           _v("increase_streak_years", str(f.increase_streak_years))]))
        if not f.fcf_unavailable and f.fcf_payout is not None and f.fcf_payout < 50:
            out.append(_w("case", "Room to keep raising.",
                          f"The dividend takes {_int(f.fcf_payout)}% of free cash flow.",
                          [_v("fcf_payout", str(_int(f.fcf_payout)))]))

    # ---- the blended valuation component ----------------------------------
    #
    # `discount_to_own_history` (value) and `valuation_gap` (recovery) average
    # a RELATIVE half — where the multiple sits in the name's own history —
    # with an ABSOLUTE one, because a purely relative measure cannot tell
    # "cheap" from "least expensive it has ever been".
    #
    # So the prose has to say WHICH half is talking. Measured on the current
    # value board, 13 of 22 names have halves more than 20 points apart and 8
    # more than 40: MELI scores 50 from 100 and 0, QCOM scores 50 from 55 and
    # 45. Writing "cheaper than usual" over both would be true of one half of
    # MELI and badly misleading about the other.
    if screen in ("value", "recovery"):
        out += _valuation_why(f, screen)

    # ---- the counter-case: about the COMPANY ------------------------------
    if len(ranked) > 1:
        low, lv = ranked[-1]
        out.append(_w("counter", f"Weakest on {low.replace('_', ' ')}.",
                      f"It scores {lv} out of {COMPONENT_SCALE} there, its "
                      f"lowest component.",
                      [_v(f"components.{low}", str(lv)),
                       _v("component scale", str(COMPONENT_SCALE))]))

    if screen == "dividend":
        if f.dividend_yield is not None and f.dividend_yield < 1.5:
            out.append(_w("counter", "Low income today.",
                          f"At {_pct(f.dividend_yield, 2)}%, this is a dividend-growth "
                          f"idea rather than an income stock.",
                          [_v("dividend_yield", _pct(f.dividend_yield, 2))]))
        if f.increase_streak_years < 10:
            at_min = f.increase_streak_years == DIVIDEND_STREAK_MIN
            out.append(_w("counter", "Short track record.",
                          f"{f.increase_streak_years} years of raises"
                          + (f", exactly the minimum this screen allows."
                             if at_min else
                             f", against a minimum of {DIVIDEND_STREAK_MIN}."),
                          [_v("increase_streak_years", str(f.increase_streak_years)),
                           _v("threshold", str(DIVIDEND_STREAK_MIN))]))

    # Any rule sitting close to its threshold is a real counter-point.
    for r in (screeners.rule_table(f, screen)["rules"] if screen in screeners.RULES else []):
        if r["state"] == "fail":
            shown = _shown(r["value"])
            out.append(_w("counter", f"Fails {r['label'].lower()}.",
                          f"It needs {r['need']}" +
                          (f", and reads {shown}." if shown else "."),
                          ([_v(f"rule.{r['label']}", shown)] if shown else [])
                          + [_v("threshold", n) for n in _nums(r["need"])]))

    # ---- judgements about the DATA, kept separate -------------------------
    # voided_fields has its own sentence below; listing it here as a flag name
    # would say the same thing twice in different words.
    flagged = [k for k, v in data_quality_report(f)["flags"].items()
               if v and k != "voided_fields"]
    for k in flagged[:4]:
        out.append(_w("data", f"{k.replace('_', ' ').capitalize()}.",
                      "This name ranks despite that input being degraded.", []))
    if f.voided_fields:
        out.append(_w("data", "Some figures are voided as unavailable.",
                      ", ".join(f.voided_fields[:6])
                      + " could not be derived, so nothing is shown for them "
                        "rather than a zero.", []))
    if screen in screeners.RULES:
        for r in screeners.rule_table(f, screen)["rules"]:
            if r["state"] == "unchecked":
                # "No source" and "not derivable for this company" are
                # different facts and a reader acts differently on each: one
                # is never coming, the other may be there next quarter.
                why = r["note"] or (
                    screeners.UNCHECKABLE.get(r["field"])
                    if r["field"] in screeners.UNCHECKABLE
                    else "the input is unavailable for this company")
                out.append(_w("data", f"The {r['check']} check could not run.",
                              _cap(why) + ".", []))
    return out


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
    "reit": ["valuation_vs_own_history", "coverage", "growth_durability",
             "balance_sheet", "shareholder_return"],
    "financial": ["returns", "capital_strength", "valuation_vs_own_history",
                  "growth_durability", "shareholder_return"],
}


def value_row(f: Fundamentals, res: dict) -> dict:
    return {
        "ticker": f.symbol, "name": f.name, "score": int(res["score"]),
        "why": _why(f, res, "value"),
        "rules": screeners.rule_table(f, "value"),
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
        # Published, not back-derived. sigma = (yld - med) / yz is wrong by a
        # little everywhere because yz is rounded to 2dp, and undefined at
        # yz == 0 — the class of silent error this project exists to remove.
        "ysd": _pct(f.yield_std_5y, 4),
        "chow": _pct(res.get("chowder", 0)), "pay": _int(f.fcf_payout),
        "streak": int(f.increase_streak_years), "nd": _pct(f.net_debt_ebitda, 2),
        "comp": _comp(res, COMPONENT_ORDER["dividend"]),
        "note": _note(f, res,
                      f"Yield of {_pct(f.dividend_yield, 2)}% against a five-year "
                      f"median of {_pct(f.yield_median_5y, 2)}%, with "
                      f"{f.increase_streak_years} consecutive years of increases. "
                      "Note the XBRL horizon caps streaks near 18 years."),
        "why": _why(f, res, "dividend"),
        "rules": screeners.rule_table(f, "dividend"),
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
        "why": _why(f, res, "recovery"),
        "rules": screeners.rule_table(f, "recovery"),
        "dd": int(round(f.drawdown_from_ath)),
        "upside": round(float(path.get("total_multiple", 0)), 1),
        "rev": _pct(f.revenue_cagr_5y), "gm": _int(f.gross_margin),
        "fcf": _pct(f.fcf_margin), "runway": int(round(f.cash_runway_quarters)),
        "evs": _int(f.ev_sales_percentile_5y), "z": _pct(f.altman_z),
        # The ABSOLUTE half of valuation_gap. Without it the detail page can
        # show the blended score and cannot say which half produced it — and
        # a 50 from two halves disagreeing is a different story from a 50
        # where both are middling.
        "ev": None if f.ev_ebit in (None, 0) else _pct(f.ev_ebit),
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
        "why": _why(f, res, "financial"),
        "rules": screeners.rule_table(f, "financial"),
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


# The dark pool board row. Its engine output never met the table: rows came
# back as {symbol, dpi_5d, oe_share, ...} while the tab reads {ticker, name,
# dpi, dpiz, oe, rvol, coil, drift, si}. Every earlier board was EMPTY, so the
# mismatch never rendered — until a 62 cut produced 21 rows and `r.name`
# crashed the tab. Short interest has no free source and stays unavailable.
DARKPOOL_ORDER = ["dpi_persistence", "off_exch_share", "block_trend",
                  "rel_volume", "compression", "price_stealth"]


def darkpool_row(r: dict, titles: dict | None = None) -> dict:
    sym = str(r["symbol"])
    key = "".join(ch for ch in sym.upper() if ch.isalnum())
    comps = r.get("components", {})
    unwired = [k.replace("_", " ") for k in ("block_trend", "rel_volume")
               if comps.get(k) == 50]
    adv = r.get("dollar_adv")
    return {
        "ticker": sym, "name": (titles or {}).get(key) or "",
        "score": int(r["score"]),
        "dpi": _pct(r.get("dpi_5d")), "dpiz": _pct(r.get("dpi_z"), 2),
        "oe": _pct(r.get("oe_share")), "rvol": _pct(r.get("rvol"), 2),
        "coil": _int(r.get("compression")), "drift": _pct(r.get("ret_20d")),
        "si": None,
        "state": r.get("state") or "Neutral",
        "comp": [int(round(comps.get(k, 0))) for k in DARKPOOL_ORDER],
        "note": (f"<p>DPI {_txt(r.get('dpi_5d'), '%')} over five sessions "
                 f"(z {_txt(r.get('dpi_z'), '', 2)}), off-exchange share "
                 f"{_txt(r.get('oe_share'), '%')}, relative volume "
                 f"{_txt(r.get('rvol'), 'x', 2)}, range coil {_txt(r.get('compression'), '', 0)}, "
                 f"20-day move {_txt(r.get('ret_20d'), '%')}. Read: {r.get('state')}.</p>"
                 + (f"<p><b>Scored at a neutral 50, not measured:</b> {', '.join(unwired)}.</p>"
                    if unwired else "")),
        "facts": _facts([
            ("Dollar ADV (20d)", "\u2014" if adv is None else f"${adv/1e6:,.0f}M"),
            ("DPI z", _txt(r.get("dpi_z"), "", 2)),
            ("Off-exchange share", _txt(r.get("oe_share"), "%")),
            ("Short interest", "not sourced"),
        ]),
    }


def reit_row(f: Fundamentals, res: dict) -> dict:
    """
    Trusts rank on FFO, so the row carries FFO payout and price to FFO rather
    than EPS payout and EV/EBIT, which describe the tax code and an irrelevant
    denominator respectively.
    """
    return {
        "ticker": f.symbol, "name": f.name, "score": int(res["score"]),
        "why": _why(f, res, "reit"),
        "rules": screeners.rule_table(f, "reit"),
        "yld": _pct(f.dividend_yield, 2),
        "pffo": _pct(f.p_ffo), "pffop": _int(f.p_ffo_percentile_10y),
        "ffopay": _int(f.ffo_payout), "affoy": _pct(f.affo_yield),
        "ffog": _pct(f.ffo_cagr_5y), "nd": _pct(f.net_debt_ebitda),
        "comp": _comp(res, COMPONENT_ORDER["reit"]),
        "note": _note(f, res,
                      f"FFO payout {_txt(f.ffo_payout, '%', 0)} of funds from "
                      f"operations, priced at {_txt(f.p_ffo, 'x')} FFO against "
                      f"{_txt(f.p_ffo_percentile_10y, 'th', 0)} percentile of its own "
                      f"ten-year history. FFO is derived — no trust files the tag."
                      + (" AFFO is unavailable: this filer tags no recurring "
                         "capital improvements, only growth spend."
                         if f.affo_unavailable else "")
                      + (" FFO is overstated: no gains-on-sale tag to subtract."
                         if f.ffo_degraded else "")),
        "facts": _facts([
            ("FFO payout", _txt(f.ffo_payout, "%", 0)),
            ("AFFO yield", "n/a" if f.affo_unavailable else _txt(f.affo_yield, "%")),
            ("FFO 5y CAGR", _txt(f.ffo_cagr_5y, "%")),
            ("Net debt/EBITDA", _txt(f.net_debt_ebitda, "x")),
        ]),
    }


BUILDERS = {"value": value_row, "dividend": dividend_row,
            "recovery": recovery_row, "financial": financial_row,
            "reit": reit_row}


def to_rows(engine: str, scored: list[tuple[Fundamentals, dict]]) -> list[dict]:
    """`scored` is (Fundamentals, score_result) pairs, already gate-filtered."""
    build = BUILDERS[engine]
    return [build(f, res) for f, res in scored]
