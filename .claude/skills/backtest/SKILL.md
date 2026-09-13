---
description: Build or run a walk-forward backtest for an engine, avoiding lookahead and survivorship bias. Use when the user asks to backtest, validate, test the weights, check hit rates, or asks whether an engine actually works.
argument-hint: [engine-name]
---

Backtest the `$ARGUMENTS` engine. If none was named, ask which one.

## Before writing any code, confirm the data is point-in-time

This is the step that decides whether the result means anything, and it is the
one that gets skipped.

- **Restatements.** Screening today's fundamentals against 2023 prices uses
  numbers nobody had in 2023. Sharadar SF1 has genuine point-in-time data;
  most cheap providers do not.
- **Survivorship.** If the universe comes from today's listed names, every
  company that went to zero is missing. That alone can turn a losing strategy
  into a winner on paper.
- **Signal timing.** FINRA daily files post 6pm ET, so a signal on day T is
  tradeable at day T+1 open at the earliest. Congressional PTRs are tradeable
  from the *filing* date, never the transaction date.

If the data is not point-in-time, say so plainly and either stop or label the
result as indicative only. Do not quietly proceed.

## Method

1. **Walk forward, never in-sample.** Fit or tune on a window, evaluate on the
   next unseen window, roll. A single in-sample pass over the full history
   tells you nothing.
2. **Report per-signal, not portfolio-level only.** Hit rate, mean and median
   forward return at 1w/1m/3m, expectancy, worst decile. A 60% hit rate with a
   fat left tail is a losing system.
3. **Benchmark honestly.** Versus SPY for equity engines, versus buy-and-hold
   BTC for the bitcoin engine. An engine that underperforms holding is a
   finding worth reporting, not a bug to tune away.
4. **Count the sample.** State n for every claim. The BTC cycle engine has
   three completed cycles; no amount of processing makes that significant.
5. **Test the gates separately from the weights.** Gates are structural and
   should hold up. Weights are untested priors. Knowing which half is doing the
   work matters more than the headline number.

## Guard against fooling yourself

- Do not tune weights until the walk-forward harness exists, or you are fitting
  noise and calling it validation.
- If a result looks excellent, look for the bug before celebrating. Suspiciously
  good results usually mean lookahead leaked in.
- Report negative results with the same prominence as positive ones. An engine
  that does not work is the single most valuable thing a backtest can tell us.

Write results to `backtests/<engine>-<date>.md` with the method, the sample
size, and the caveats stated inline rather than in a footnote.
