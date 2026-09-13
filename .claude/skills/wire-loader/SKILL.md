---
description: Wire one of the NotImplementedError data loaders in run_all.py against its real source. Use when the user says wire the loader, connect FINRA, hook up fundamentals, get real data flowing, or names a provider like FMP, Polygon, Glassnode, or CoinGecko.
argument-hint: [loader-name]
allowed-tools: Bash(python -c *) Bash(python run_all.py *)
---

Wire the `$ARGUMENTS` loader in `run_all.py`. If no loader was named, list the
six stubs with what each needs and ask which one.

## The rule that matters

Never write a loader from the API docs alone. Fetch real data first, print the
actual shape, then write the parser against what came back. Every stub is a
`NotImplementedError` precisely because a plausible-looking loader that returns
subtly wrong numbers is worse than one that raises.

## Steps

1. **Pull one real sample before writing anything.** For FINRA, fetch a single
   recent trading day and print the raw first three lines plus the parsed
   dtypes. For a paid provider, request one symbol. Confirm column names,
   date format, and whether the volume column means what the code assumes.

2. **Check the assumption the engine depends on.** Each loader feeds a specific
   calculation and the wrong column silently breaks it:

   | Loader | What must be true |
   |---|---|
   | `load_tape` | `volume` is CONSOLIDATED tape volume, not off-exchange. `off_exch_pct` is meaningless otherwise. |
   | `load_fundamentals` | 5–10y of history per field, point-in-time if it will feed a backtest. Snapshots are not enough — the screens rank against each company's own history. |
   | `load_ptrs` | Both `transaction_date` and `filing_date` present and distinct. The whole engine is built on the gap between them. |
   | `load_btc_weekly` | Weekly closes, not resampled daily. |
   | `load_btc_daily` | Daily closes, UTC candle close. |
   | `load_sth_mvrv` | Short-term holder cost basis, 155-day cohort. Not aggregate MVRV. |

3. **Handle the failure modes the source actually has.** FINRA 404s on
   holidays (already handled), occasionally reposts corrected files, and the
   House portal serves scanned PDFs needing OCR for many older PTRs. Do not
   assume a clean fetch.

4. **Write the loader**, cast numpy scalars with `float()`/`bool()` so the
   payload stays JSON-serialisable, and keep secrets in `os.environ`.

5. **Verify end to end** with `python run_all.py --only <engine> --force` and
   read the written `data/<engine>.json`. Sanity-check two or three values by
   hand against the source rather than trusting that it ran without error.

6. **Report what you could not verify.** If a field was inferred rather than
   confirmed against real output, say so.

Do not touch scoring logic while wiring a loader. If the real data reveals a
scoring bug, report it and ask before changing it.
