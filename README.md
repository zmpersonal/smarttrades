# SmartTrades.AI

Six screening engines behind one dashboard, refreshed nightly by GitHub Actions
and published to GitHub Pages.

```
index.html                     dashboard — opens standalone, no build step
run_all.py                     daily runner, writes data/*.json
engines/
  finra_darkpool.py            off-exchange accumulation
  screeners.py                 dividends, recovery, quality-value
  capitol_flow.py              congressional disclosures
  btc_cycle.py                 halving cycle, RSI divergence, dip buy
data/                          engine output + status.json (committed by CI)
.github/workflows/daily.yml    the automation
```

The dashboard fetches `data/*.json` on load and falls back to embedded sample
data when those files are absent, so the page renders before anything is wired.

## Getting it running

1. Push to GitHub. Settings → Pages → source **GitHub Actions**.
2. Add provider keys under Settings → Secrets → Actions:
   `FMP_API_KEY`, `POLYGON_API_KEY`, `GLASSNODE_API_KEY`, `QUIVER_API_KEY`.
3. Fill in the six loader stubs at the bottom of `run_all.py`. They are
   deliberately `NotImplementedError` rather than half-working defaults —
   guessing a provider would produce code that looks finished and silently
   returns wrong numbers.
4. Actions → Daily screens → **Run workflow** to test before trusting the cron.

Until step 3 is done the runner reports `not_wired` per engine and exits 0.
That is the intended buildout state.

## The automation

Two crons. Weekday **23:15 UTC** (6:15pm ET) for the daily engines, just after
FINRA posts its off-exchange file at 6:00pm ET. Sunday **06:00 UTC** for the
fundamental screens, which barely move day to day.

Things that will bite you:

- **GitHub's scheduler is best-effort.** Runs can lag 10-30 minutes under load.
  The 15-minute cushion after the FINRA deadline exists for that reason. If you
  need reliable timing, use an external trigger hitting `workflow_dispatch`.
- **Scheduled workflows auto-disable after 60 days without repo activity.**
  The nightly data commit counts as activity, so this is self-sustaining once
  running — but it will not restart itself if you pause it.
- **Cron is UTC and ignores daylight saving.** The 23:15 slot is 6:15pm ET in
  winter and 7:15pm in summer. Both land after the FINRA post, so it holds
  either way, but check this if you tighten the window.
- One engine failing must never take down the others. Each runs in its own
  try/except and writes its own file; `data/status.json` records per-engine
  state and the dashboard shows each tab's age.

## Working on this in Claude Code

`CLAUDE.md` is read at the start of every session and carries the decisions
that are expensive to rediscover: why high DPI reads bullish, why politicians
rank on replicable alpha, why the BTC dip-buy rule is gated, why alerts fire on
transitions only. Read it before changing engine logic.

Four project skills in `.claude/skills/`:

| Command | Use |
|---|---|
| `/wire-loader [name]` | Connect a stubbed data loader to its real source |
| `/backtest [engine]` | Walk-forward backtest without lookahead or survivorship bias |
| `/check-alerts` | Preview Slack alerts safely — dry run only, never sends |
| `/add-engine [name]` | Add a screener following gate-then-score structure |

`.claude/settings.json` pre-approves the project's own commands and the data
domains, asks before `git push` and workflow edits, and blocks reads of `.env`.
Accept the workspace trust prompt on first run or the allow rules stay inactive.

## Slack alerts

Set `SLACK_WEBHOOK_URL` in repo secrets (Slack → Apps → Incoming Webhooks).
Two message types, deliberately different:

**BTC — event-driven.** Fires only when something changes state, never when a
condition merely persists. If daily RSI sits oversold for a week that is one
message, not seven. Most days send nothing.

| Alert | Severity | Cooldown |
|---|---|---|
| Entry triggered — weekly permits, daily triggers | act | 7d |
| Dip-buy armed — weekly regime just confirmed | warn | 14d |
| Regime lost — dip-buy gated again | risk | 14d |
| Daily RSI within 5 points of oversold, regime armed | warn | 5d |
| Daily bullish divergence | warn | 10d |
| Weekly bearish divergence, push 2 or 3 | risk | 14d |
| Cycle milestone — trough window, drawdown thresholds | info | 30d |

**Digest — every second Sunday, 14:00 UTC.** Top 10 per screener with what
entered and what dropped out since last time. At a two-week cadence the
changes matter more than the list.

Two things that will silently break this if you miss them:

- **`data/alert_state.json` must be committed.** It is what remembers which
  alerts you have already seen. If it is gitignored, every run looks like a
  first run and re-sends everything.
- **Cron cannot express "every two weeks."** The Sunday job runs weekly and
  `alerts.digest_due()` decides whether this is a digest week, counting weeks
  from a fixed epoch. Do not use `isocalendar().week % 2`: in a 53-week ISO year
  such as 2026, weeks 53 and 1 are both odd, so the cadence skips a fortnight.

Test without spamming yourself:

```bash
python run_all.py --notify --dry-run          # prints payloads
python run_all.py --digest --dry-run          # force a digest preview
```

## Data sources

**Everything runs on free data.** `run_all.py` defaults to `engines/free_sources.py`:
FINRA, FRED, SEC EDGAR, Stooq, Binance, Coin Metrics Community, Stock Watcher
and Yahoo RSS — none of which need a paid key, and most of which need no key at
all. See [COSTS.md](COSTS.md) for what you give up (short answer: STH-MVRV, the
Conference Board LEI, and about two weekends of wiring).

**Free** — FINRA daily off-exchange short volume (same day, 6pm ET), FINRA ATS
transparency (weekly, 2-4 week lag), House and Senate PTR portals, SEC EDGAR.

**Paid, and you need at least one** — fundamentals with 5-10y history are the
hard requirement, since the screens rank against each company's *own* multiple
and yield history. FMP ~$50-100/mo, EODHD ~$80/mo, Sharadar ~$150/mo for clean
point-in-time data, Polygon ~$30-200/mo for tape volume. Glassnode ~$40/mo for
STH-MVRV, which has no reliable free source.

Realistic total: **$100-250/month** for 2,000-3,000 US names.

## What each engine does

**Dark Pool Radar.** FINRA's actual dark pool data is too slow to trade — ATS
reports run 2-4 weeks late by design. The daily off-exchange short volume file
is the fast feed. DPI (off-exchange short volume ÷ off-exchange total) z-scored
against each symbol's own history, plus off-exchange share, block-size trend,
relative volume, range compression and a price-stealth term. High DPI reads
*bullish* — those prints are mostly market makers facilitating buyers. The main
false positive is heavily shorted names, so the signal is damped above 15%
short interest.

**Dividend Growers.** Yield ranked against each company's own 5-year median,
not an absolute threshold. Hard gates on streak, payout, coverage and leverage
first. A yield-trap filter removes anything elevated because forward estimates
collapsed rather than because price fell.

**Recovery Compounders.** 200% is 3.0x. Every candidate carries an explicit
decomposition — revenue growth × margin change × multiple re-rating — and cases
carried mostly by multiple expansion get penalised. Survival gates come first.

**Undervalued Quality.** Quality is pass/fail, not a tiebreaker. Survivors rank
on a reverse DCF: solve for the growth the price implies, compare to delivered.

**Capitol Flow.** Members rank on *replicable* alpha — the return from entering
on the filing date, not the trade date. The STOCK Act's 45-day window means
those differ enormously and only the first is available to you. Member names in
the sample are fictional; attaching invented performance figures to real
officials would be defamatory. Live pulls real names from the filings.

**Bitcoin Cycle.** Halving clock, RSI three-push divergence detector, and a
regime-gated dip-buy rule. Run over the last cycle, the divergence detector
flags the October 2025 top: RSI 89.9 → 77.7 → 78.0 while price rose 21% across
the three pushes. The dip-buy rule combines the 20W/21W support band with
STH-MVRV ≤ 1.0 and is currently **gated**, because price sits 38.7% below the
high — it buys dips within an uptrend, and firing it in a markdown phase is
exactly how it fails.

**Recession Radar.** Two scores rather than one: LEAD (curve probit,
un-inversion clock, LEI, claims) fires 6-18 months out and is noisy; STRESS
(HY OAS, CCC-BB dispersion, financial conditions, Sahm) fires at the event and
is rarely early. The gap between them is the read. Currently lead 48.5 against
stress 18.4 — the curve flags risk while credit prices almost none, which is
late-cycle-unpriced rather than all-clear.

Two constraints worth knowing before wiring it. FRED serves only a rolling
three-year window for ICE BofA series as of April 2026, so percentile ranks are
against three years rather than a credit cycle; cache the daily prints forward
in `data/oas_history.csv`. And Conference Board LEI is licensed, so it is not
on FRED at all.

## Ticker detail pages

Every symbol in every table is clickable, and BTC has its own. Each page has the
bull and bear case with checkable triggers, a three-tier entry ladder, the daily
indicator panel, and a news slot.

Two design points that are load-bearing rather than cosmetic:

**Indicators are grouped by independence, not listed flat.** MACD, RSI and
StochRSI all derive from close and correlate 0.6-0.9 in practice, so the panel
presents them as one signal with three readings and reports the effective signal
count. OBV and Z-Score are the two that can genuinely disagree with price.

**The ladder's third tier carries a confirmation gate.** A price that good
usually means the thesis broke. Tier 3 checks solvency, insider buying and
estimate trajectory, and renders a falling-knife warning when those fail.

News is category placeholders, deliberately — fabricating headlines about real
companies is misinformation. Wire Finnhub, Marketaux or Alpha Vantage (all have
usable free tiers) and tag each item against the triggers above.

## Before you trust any of it

**Backtest first.** The gates are defensible from first principles. The weights
are reasoned priors, not fitted parameters, and they need testing against
forward returns. This is the step people skip.

**The Bitcoin tab rests on three completed cycles.** That is an anecdote count,
not a sample, and the current cycle is already breaking the pattern on drawdown
depth while matching it on timing.

**Sample data throughout** until the loaders are wired. Nothing here is
investment advice; a screen surfaces candidates for research rather than
replacing it.
