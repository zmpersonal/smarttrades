"""
SmartTrades.AI — Slack alerting
===============================

Two message types, deliberately different in character:

    BTC ALERTS    event-driven. Fire only when something changes state and
                  only when there is a decision to make. Silence is the
                  default and is meaningful.

    DIGEST        every two weeks. Top 10 per screener, with what entered
                  and left the list since last time.

The hard part is not sending messages, it is not sending them. A rule that
alerts whenever `rsi < 30` will message you every day for a week during one
oversold stretch, and you will mute the channel by day three. Everything here
fires on the *transition* into a state, never on the state persisting.

State lives in data/alert_state.json, committed by the Action, so the runner
knows what it already told you.

Setup:
    Slack → Apps → Incoming Webhooks → Add to a channel → copy the URL
    GitHub → Settings → Secrets → Actions → SLACK_WEBHOOK_URL
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import requests

STATE = Path(__file__).parent / "data" / "alert_state.json"
WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", "")

# Biweekly parity from a fixed epoch. Do NOT use isocalendar().week % 2: in a
# 53-week ISO year (2026 is one) weeks 53 and 1 are both odd, so the cadence
# skips a fortnight and you get a 3-week gap. Counting weeks from a fixed date
# is immune. Verified in tests/test_engines.py over 2026-2032.
EPOCH = date(2026, 1, 4)

COLORS = {"act": "#4FC08D", "warn": "#E8A13A", "risk": "#E4626F", "info": "#35C9D4"}


@dataclass
class Alert:
    key: str                 # stable id — used for dedup
    severity: str            # act | warn | risk | info
    title: str
    body: str
    fields: list[tuple[str, str]] = field(default_factory=list)
    cooldown_days: int = 3   # minimum gap before this key may fire again


# ------------------------------------------------------------------ state

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"fired": {}, "last_digest": None, "last_top": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str))


def _due(st: dict, a: Alert, today: date) -> bool:
    prev = st["fired"].get(a.key)
    if not prev:
        return True
    gap = (today - date.fromisoformat(prev["on"])).days
    # Same key, same state, inside the cooldown → stay quiet.
    return gap >= a.cooldown_days or prev.get("body") != a.body


# ------------------------------------------------------------- BTC rules

def btc_alerts(cur: dict, prev: dict | None) -> list[Alert]:
    """
    Compare this run's engine output to the previous run and emit only
    genuine transitions.

    `cur` and `prev` are the payloads written by run_all.run_bitcoin().
    """
    out: list[Alert] = []
    prev = prev or {}

    entry, p_entry = cur.get("entry", {}), prev.get("entry", {})
    dip, p_dip = cur.get("dip_buy", {}), prev.get("dip_buy", {})
    cyc = cur.get("cycle", {})
    daily = entry.get("daily", {})

    # --- the one you actually want to be woken for -----------------------
    if entry.get("status") == "TRIGGERED" and p_entry.get("status") != "TRIGGERED":
        out.append(Alert(
            "btc_entry", "act",
            "BTC entry triggered",
            "Weekly regime is confirmed and the daily has pulled the trigger. "
            "Both timeframes agree for the first time this cycle.",
            [("Daily RSI", str(daily.get("rsi"))),
             ("Oversold band", str(daily.get("bands", {}).get("oversold"))),
             ("Days oversold", str(daily.get("days_oversold"))),
             ("Weekly RSI", str(entry.get("weekly_rsi")))],
            cooldown_days=7))

    # --- regime flips: the whole rule set turns on or off -----------------
    # If entry already fired this run, "now armed" is redundant noise —
    # the stronger message already says the regime confirmed.
    already_triggered = any(x.key == "btc_entry" for x in out)
    if (entry.get("status") != "GATED" and p_entry.get("status") == "GATED"
            and not already_triggered):
        out.append(Alert(
            "btc_armed", "warn",
            "BTC dip-buy is now armed",
            "The weekly regime just confirmed. The dip-buy rule is live for the "
            "first time since it was gated — daily oversold prints now count.",
            [("Regime", str(entry.get("regime_label"))),
             ("Weekly RSI", str(entry.get("weekly_rsi")))],
            cooldown_days=14))

    if entry.get("status") == "GATED" and p_entry.get("status") not in (None, "GATED"):
        out.append(Alert(
            "btc_disarmed", "risk",
            "BTC regime lost — dip-buy gated",
            "Price has lost the weekly support band or the drawdown reopened. "
            "Daily oversold prints are no longer actionable on their own.",
            [("Regime", str(entry.get("regime_label")))],
            cooldown_days=14))

    # --- be at the desk ---------------------------------------------------
    dist = daily.get("distance_to_oversold")
    if dist is not None and 0 < dist <= 5 and entry.get("status") == "ARMED":
        out.append(Alert(
            "btc_approaching", "warn",
            "BTC daily RSI approaching oversold",
            f"Daily RSI is {dist} points from the oversold line with the weekly "
            "regime already granting permission. Worth watching intraday.",
            [("Daily RSI", str(daily.get("rsi"))),
             ("Trigger at", str(daily.get("bands", {}).get("oversold")))],
            cooldown_days=5))

    if daily.get("bullish_divergence") and not (
            p_entry.get("daily") or {}).get("bullish_divergence"):
        d = daily["bullish_divergence"]
        out.append(Alert(
            "btc_bull_div", "warn",
            "BTC daily bullish divergence",
            "Price made a lower low while RSI made a higher low — selling "
            "pressure decaying into new lows.",
            [("Window", f"{d['from']} → {d['to']}"),
             ("Price", f"{d['price_delta_pct']}%"),
             ("RSI", f"+{d['rsi_delta']}")],
            cooldown_days=10))

    # --- risk-off ---------------------------------------------------------
    div = cur.get("divergence", {})
    if div.get("signal") and div["signal"] != (prev.get("divergence") or {}).get("signal"):
        s = div["signal"]
        out.append(Alert(
            "btc_bear_div", "risk",
            f"BTC weekly bearish divergence — push {s['push']}",
            "Price made a higher high, RSI did not. This is the pattern that "
            "marked the October 2025 top. Tighten risk rather than act.",
            [("Strength", s["strength"]),
             ("Price", f"{s['price_delta_pct']}%"),
             ("RSI", str(s["rsi_delta"]))],
            cooldown_days=14))

    # --- cycle-level, rare by construction --------------------------------
    for note in cyc.get("alerts", []):
        out.append(Alert(
            "btc_cycle_" + note.split()[0].lower().strip("."), "info",
            "BTC cycle milestone", note,
            [("Drawdown", f"{cyc.get('drawdown', 0) * 100:.1f}%"),
             ("Days since peak", str(cyc.get("days_since_peak"))),
             ("Cycle progress", f"{cyc.get('cycle_pct')}%")],
            cooldown_days=30))

    return out


def recession_alerts(cur: dict, prev: dict | None) -> list[Alert]:
    """
    Same transition-only discipline as BTC. Credit spreads sit still for months
    and then move fast, so a level-based rule would say nothing for a year and
    then repeat itself daily for three weeks. These fire on crossings.
    """
    out: list[Alert] = []
    prev = prev or {}
    oas, p_oas = cur.get("oas", {}), prev.get("oas", {})
    comp, p_comp = cur.get("composite", {}), prev.get("composite", {})
    disp, p_disp = cur.get("dispersion", {}), prev.get("dispersion", {})
    sahm, p_sahm = cur.get("sahm", {}), prev.get("sahm", {})

    # Momentum from a tight base beats any level. This is the one that matters.
    if oas.get("change_1m", 0) >= 1.00 and p_oas.get("change_1m", 0) < 1.00:
        out.append(Alert(
            "rec_oas_momentum", "risk",
            "HY OAS widening fast",
            f"Spreads have widened {oas['change_1m']:.2f}pp in a month to "
            f"{oas['level']:.2f}%. Rate of change from a tight base is the "
            "informative signal, not the level.",
            [("HY OAS", f"{oas['level']:.2f}%"), ("1m change", f"+{oas['change_1m']:.2f}pp"),
             ("Band", oas.get("band", "")), ("vs 200d", f"{oas.get('vs_200d', 0):+.2f}pp")],
            cooldown_days=5))

    # Band escalation, both directions.
    order = [b[2] for b in __import__("engines.recession", fromlist=["x"]).OAS_BANDS]
    if oas.get("band") and p_oas.get("band") and oas["band"] != p_oas["band"]:
        worse = order.index(oas["band"]) > order.index(p_oas["band"])
        out.append(Alert(
            "rec_oas_band", "risk" if worse else "info",
            f"HY OAS moved to the {oas['band']} band",
            f"From {p_oas['band']} to {oas['band']} at {oas['level']:.2f}%.",
            [("Level", f"{oas['level']:.2f}%"),
             ("Percentile", f"{oas.get('percentile_in_window', 0):.0f}th of 3y")],
            cooldown_days=10))

    # Dispersion turns before headline HY, so this should fire earlier.
    if disp.get("percentile_in_window", 0) >= 75 and p_disp.get("percentile_in_window", 0) < 75:
        out.append(Alert(
            "rec_dispersion", "warn",
            "Credit quality dispersion widening",
            f"CCC-BB at {disp['dispersion']:.2f}pp, "
            f"{disp['percentile_in_window']:.0f}th percentile. The low-quality tail "
            "reprices before the index does, so this usually leads headline HY OAS.",
            [("CCC-BB", f"{disp['dispersion']:.2f}pp"), ("3m change", f"{disp.get('change_3m', 0):+.2f}pp")],
            cooldown_days=10))

    if sahm.get("triggered") and not p_sahm.get("triggered"):
        out.append(Alert(
            "rec_sahm", "risk", "Sahm rule triggered",
            f"Real-time Sahm at {sahm['value']:.2f}, through the 0.50 threshold. "
            "Coincident to slightly lagging — it confirms rather than predicts, and "
            "post-pandemic labour supply swings can lift it without a demand collapse.",
            [("Sahm", f"{sahm['value']:.2f}")], cooldown_days=30))

    # The configuration change worth knowing about: lead and stress converging.
    if (comp.get("stress_score", 0) >= 45 and p_comp.get("stress_score", 0) < 45
            and comp.get("lead_score", 0) >= 45):
        out.append(Alert(
            "rec_converged", "risk",
            "Recession scores converged",
            "Lead and stress are both elevated. The gap that has been open all "
            "year has closed, meaning the market is now repricing what the curve "
            "flagged months ago.",
            [("Lead", str(comp["lead_score"])), ("Stress", str(comp["stress_score"])),
             ("Gap", str(comp.get("divergence")))],
            cooldown_days=21))

    return out


# ---------------------------------------------------------- message build

def btc_blocks(a: Alert) -> dict:
    icon = {"act": ":rotating_light:", "warn": ":eyes:",
            "risk": ":warning:", "info": ":hourglass:"}[a.severity]
    blocks = [
        {"type": "header",
         "text": {"type": "plain_text", "text": f"{icon} {a.title}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": a.body}},
    ]
    if a.fields:
        blocks.append({"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in a.fields[:10]]})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
        "text": f"SmartTrades · {datetime.now(timezone.utc):%d %b %H:%M UTC} · not advice"}]})
    return {"attachments": [{"color": COLORS[a.severity], "blocks": blocks}]}


def digest_blocks(engines: dict, deltas: dict) -> dict:
    """
    Biweekly top-10 digest.

    Slack caps a message at 50 blocks and 3,000 characters per text object,
    so each engine gets one compact section rather than a block per stock.
    Five engines at two blocks each stays well inside the limit.
    """
    names = {"darkpool": "Dark Pool Radar", "dividend": "Dividend Growers",
             "recovery": "Recovery Compounders", "value": "Undervalued Quality",
             "politicians": "Capitol Flow"}

    blocks = [
        {"type": "header", "text": {"type": "plain_text",
         "text": f":bar_chart: Screener digest — {date.today():%d %b %Y}"}},
        {"type": "context", "elements": [{"type": "mrkdwn",
         "text": "Top 10 per engine. *New* entered since the last digest, "
                 "*dropped* fell out of the top 10."}]},
    ]

    for key, label in names.items():
        rows = (engines.get(key) or {}).get("rows", [])[:10]
        if not rows:
            continue
        lines = []
        for i, r in enumerate(rows, 1):
            tag = " :new:" if r.get("symbol", r.get("ticker")) in deltas.get(key, {}).get("new", []) else ""
            sym = r.get("symbol") or r.get("ticker", "?")
            lines.append(f"`{i:>2}` *{sym}*  {r.get('score', '')}{tag}")
        text = "\n".join(lines)[:2900]

        gone = deltas.get(key, {}).get("dropped", [])
        if gone:
            text += f"\n_dropped: {', '.join(gone[:8])}_"

        blocks.append({"type": "divider"})
        blocks.append({"type": "section",
                       "text": {"type": "mrkdwn", "text": f"*{label}*\n{text}"}})

    blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
        "text": "Screens surface candidates for research, not decisions. "
                "Sample data until the loaders are wired."}]})
    return {"attachments": [{"color": COLORS["info"], "blocks": blocks}]}


def compute_deltas(engines: dict, last_top: dict) -> dict:
    """What entered and left each top 10 since the previous digest."""
    out = {}
    for key, payload in engines.items():
        cur = [r.get("symbol") or r.get("ticker")
               for r in (payload or {}).get("rows", [])[:10]]
        prev = last_top.get(key, [])
        out[key] = {"new": [s for s in cur if s not in prev],
                    "dropped": [s for s in prev if s not in cur],
                    "current": cur}
    return out


# ------------------------------------------------------------------- send

def post(payload: dict, dry_run: bool = False) -> bool:
    if dry_run or not WEBHOOK:
        print(json.dumps(payload, indent=1)[:1600])
        if not WEBHOOK:
            print("  (SLACK_WEBHOOK_URL unset — printed instead of sent)")
        return True
    r = requests.post(WEBHOOK, json=payload, timeout=15)
    if r.status_code != 200:
        print(f"  slack error {r.status_code}: {r.text[:200]}")
        return False
    return True


def send_btc(cur: dict, prev: dict | None, dry_run: bool = False) -> int:
    st, today = load_state(), date.today()
    sent = 0
    for a in btc_alerts(cur, prev):
        if not _due(st, a, today):
            print(f"  [quiet] {a.key} — inside cooldown, unchanged")
            continue
        if post(btc_blocks(a), dry_run):
            st["fired"][a.key] = {"on": today.isoformat(), "body": a.body}
            sent += 1
            print(f"  [sent ] {a.key} ({a.severity})")
    save_state(st)
    if sent == 0:
        print("  nothing to say — no state changes")
    return sent


def send_generic(alert_list: list[Alert], dry_run: bool = False) -> int:
    """Dispatch any alert list through the same dedup and cooldown path."""
    st, today, sent = load_state(), date.today(), 0
    for a in alert_list:
        if not _due(st, a, today):
            print(f"  [quiet] {a.key} — inside cooldown, unchanged")
            continue
        if post(btc_blocks(a), dry_run):
            st["fired"][a.key] = {"on": today.isoformat(), "body": a.body}
            sent += 1
            print(f"  [sent ] {a.key} ({a.severity})")
    save_state(st)
    if sent == 0:
        print("  nothing to say — no state changes")
    return sent


def digest_due(today: date | None = None) -> bool:
    today = today or date.today()
    return today.weekday() == 6 and ((today - EPOCH).days // 7) % 2 == 0


def send_digest(engines: dict, dry_run: bool = False) -> bool:
    st = load_state()
    deltas = compute_deltas(engines, st.get("last_top", {}))
    ok = post(digest_blocks(engines, deltas), dry_run)
    if ok:
        st["last_digest"] = date.today().isoformat()
        st["last_top"] = {k: v["current"] for k, v in deltas.items()}
        save_state(st)
    return ok


if __name__ == "__main__":
    # Dry run against a synthetic transition: gated -> triggered.
    prev = {"entry": {"status": "GATED"}, "divergence": {}, "cycle": {}}
    cur = {"entry": {"status": "TRIGGERED", "regime_label": "bull", "weekly_rsi": 54.2,
                     "daily": {"rsi": 38.4, "bands": {"oversold": 40},
                               "days_oversold": 2, "distance_to_oversold": -1.6,
                               "bullish_divergence": None}},
           "divergence": {}, "cycle": {"alerts": [], "drawdown": -0.11,
                                       "days_since_peak": 40, "cycle_pct": 72.0}}
    print("=== BTC alert dry run ===")
    send_btc(cur, prev, dry_run=True)
    print(f"\n=== digest due today? {digest_due()} ===")
