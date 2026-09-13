# Running this for $0

Every engine runs on free data. This documents how, and what you give up.

## The free stack

| Need | Source | Key? | Notes |
|---|---|---|---|
| Off-exchange volume | FINRA `cdn.finra.org` | No | Plain files, same day by 6pm ET |
| HY OAS, curve, Sahm, NFCI, claims | FRED CSV | No | Key only raises rate limits |
| Fundamentals | SEC EDGAR `companyfacts` | No | **User-Agent with contact email required** |
| EOD OHLCV | Stooq CSV | No | Full history, `.us` suffix for US tickers |
| BTC candles | Binance klines | No | Better than CoinGecko free |
| BTC on-chain | Coin Metrics Community | No | Market cap + MVRV; realized cap derived |
| Congressional PTRs | House / Senate Stock Watcher | No | Pre-parsed JSON |
| News | Yahoo Finance RSS | No | Headlines + links |
| Insider trades | SEC EDGAR Form 4 | No | Same User-Agent rule |

`python run_all.py` uses all of these by default. No accounts needed beyond a
Slack webhook.

## The genuinely good surprise

<cite index="50-1">SEC's XBRL endpoints are completely free, require no API key, and return JSON — the only requirement is a proper User-Agent header identifying your application and a contact email.</cite>
Every fact carries the date it was **filed**, so you can reconstruct what was
knowable on any past date.

That is the point-in-time property I earlier said costs ~$150/month, and on one
axis the free version is **better**. Budget paid APIs store one mutable value
per period: <cite index="44-1">restatements silently overwrite history, so the "point-in-time" join can still leak, and you can't reconstruct what the market originally saw.</cite>
EDGAR keeps every filed version, so `extract_series()` takes the
first-reported value rather than the restatement.

The cost is engineering, not money. <cite index="44-1">Companies switch XBRL tags over time (Apple's revenue has lived under three different tags since 2014), stub periods and quarters can leak into annual figures, and reconstructing first-reported values through restatements is genuinely fiddly.</cite>
The tag fallback chains, first-reported logic and Q4 derivation are already in
`free_sources.py` and under test. <cite index="44-1">Budget a few weekends, not a few hours.</cite>

## What you actually give up

**1. Short-term holder MVRV — the one real loss.**
The 155-day cohort cost basis is Glassnode proprietary with no free equivalent.
The free path substitutes **aggregate MVRV** from Coin Metrics: all holders
rather than recent ones. It is excellent at cycle extremes — below 1.0 has
marked generational lows — and much duller mid-cycle, which is exactly where
the dip-buy rule wants to fire. That rule's second leg is degraded; lean harder
on the support band. Cost to fix: Glassnode from about $39/month.

**2. Conference Board LEI is licensed.**
Not on FRED. `load_lei_score()` returns a neutral 50 rather than inventing a
reading. It is 15% of the lead score, so the recession engine loses a little
resolution and nothing else.

**3. Stooq has no SLA.**
Free service, no guarantees, and delisted tickers can vanish — which matters
for survivorship if you backtest off it. Fine for live screening. For a serious
backtest, cache daily into your own store and build history forward.

**4. Universe construction.**
EDGAR has no screener, so you must supply a ticker list in
`data/universe.json`. The S&P 1500 constituents or a Nasdaq symbol dump both
work.

**5. Time.**
The free path costs roughly two weekends of wiring versus roughly two hours for
a paid key. That is the real trade.

## Non-API costs

- **GitHub Actions** — free on public repos. Private gets 2,000 min/month free;
  this uses ~200–300.
- **GitHub Pages** — free.
- **Claude Code** — included with Claude Pro or Max.

## When to start paying

Only two things are worth money later:

| Spend | Buys |
|---|---|
| ~$39/mo Glassnode | STH-MVRV back, restoring the dip-buy rule's second leg |
| ~$150/mo Sharadar | Saves the EDGAR normalisation work, not better data |

Neither is worth paying before the pipeline runs nightly and the Slack cadence
proves it fits how you work.

## Status of the free path

Written and unit-tested offline — point-in-time filtering, first-reported
selection, tag fallback and Q4 derivation all have tests.

**The HTTP calls have not been smoke-tested against live endpoints**, because
the environment this was built in blocks those hosts (`x-deny-reason:
host_not_allowed`). They work from a GitHub runner or your machine. First run,
expect to fix small things: a Stooq ticker suffix, a Coin Metrics page token,
an EDGAR 403 from a missing User-Agent. Run `python run_all.py --only recession
--force` first — FRED is the simplest of them and will tell you fastest whether
the plumbing is sound.
