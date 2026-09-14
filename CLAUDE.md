# SmartTrades.AI

Six screening engines behind one dashboard. Nightly GitHub Action runs the
engines, writes `data/*.json`, posts Slack alerts, and publishes to Pages.

```
index.html                  dashboard (fetches data/*.json, falls back to embedded sample)
run_all.py                  orchestrator — cadence, per-engine isolation, Slack dispatch
alerts.py                   Slack: transition-only BTC alerts + biweekly digest
engines/
  finra_darkpool.py         off-exchange accumulation
  screeners.py              dividends, recovery, quality-value
  capitol_flow.py           congressional disclosures
  btc_cycle.py              halving cycle, RSI, regime-gated dip buy
  recession.py              HY OAS, curve probit, lead-vs-stress split
  indicators.py             MACD/RSI/StochRSI/OBV/Z, MFI, ATR, MVRV, entry ladders
  free_sources.py           the $0 data layer — every loader, no paid keys
data/                       engine output, status.json, alert_state.json (all committed)
```

## Non-obvious decisions — do not silently reverse these

These were reasoned through and cost real work to get right. If a change
requires breaking one, say so explicitly and explain why rather than quietly
changing it.

**Gates run before scores, everywhere.** A gate failure disqualifies a name no
matter how well it scores. Never "soften" a gate to let an attractive name
through — that inverts the design. Gate failures are surfaced with reasons, not
hidden.

**High DPI reads bullish.** Off-exchange prints marked short are mostly market
makers facilitating a *buyer*, so high DPI means absorbed demand, not bearish
positioning. The main false positive is heavily shorted names where real short
initiation inflates it; the signal is damped above 15% short interest. Anyone
"fixing" this to treat high short volume as bearish has broken the engine.

**FINRA has two feeds with very different latency.** The ATS transparency
report (true per-dark-pool volume) is weekly and 2–4 weeks late by design — it
can never be a trigger, only a slow confirming overlay. The daily off-exchange
short volume file posts by 6pm ET same day and is the actual signal.

**Politicians rank on replicable alpha, not raw alpha.** The STOCK Act's 45-day
window means what the member earned (entry at trade date) and what a follower
earns (entry at filing date) differ enormously. Only the second is available to
you. Ranking on raw alpha flatters members who file on day 44.

**BTC: weekly grants permission, daily pulls the trigger.** Weekly alone fires
2–3 times a cycle, too rarely to time anything. Daily alone fires all the way
down a bear market. The characteristic daily failure is an oversold print early
in a downtrend — the demo series' one loser fired three weeks after the cycle
top. The weekly gate exists to remove exactly that case. Never let the dip-buy
rule fire outside a confirmed regime.

**RSI bands adapt to regime, not fixed 30/70.** RSI lives ~40–80 in an uptrend
and ~20–60 in a downtrend (Cardwell). Fixed 30 under-fires in bull markets and
over-fires in bear markets. See `RSI_BANDS` in `btc_cycle.py`.

**RSI divergence: the oversold print is the anchor, not the signal.** It opens
the sequence. Divergence appears later at the highs of pushes 2 and 3, where
price makes a higher high and RSI does not. Push 1 must exceed RSI 65 or the
sequence lacks the momentum for its decay to mean anything.

**Alerts fire on transitions only.** A rule that messages whenever a condition
is *true* will send seven messages during one oversold week and the channel
gets muted. Everything in `alerts.py` fires on entering a state, with a
per-key cooldown, against `data/alert_state.json`.

**`data/alert_state.json` must stay committed.** It is what remembers what was
already sent. Gitignore it and every run looks like a first run and re-sends
everything.

**Cron cannot express "every two weeks."** The Sunday job runs weekly and
`alerts.digest_due()` decides, counting weeks from a fixed epoch. Do NOT switch
this to `isocalendar().week % 2`: in a 53-week ISO year (2026 is one) weeks 53
and 1 are both odd, so the cadence skips a fortnight — a 3-week gap and one
missed digest. Counting from a fixed date is immune. A regression test walks
every Sunday from 2026 to 2032 and asserts the gap is always exactly 2.

**HY OAS is a confirmer, not a predictor.** Spreads stay tight until trouble is
visible, then widen violently. In June 2007 HY OAS sat near 250bp and passed
900bp within eighteen months. It therefore carries most of the STRESS score and
almost none of the LEAD score. Anyone "improving" the recession engine by
weighting OAS into the lead score has reintroduced the exact error the split
exists to prevent.

**The recession engine returns two scores, never one.** Lead and stress do not
share a time horizon, and averaging them destroys the only information that
matters. The gap between them is the output. Do not collapse it into a single
recession probability.

**CCC-BB dispersion turns before headline HY OAS.** An index dominated by
well-bid BB paper looks calm while the low-quality tail reprices. Dispersion is
the early line inside credit; momentum beats level for both.

**FRED serves only a rolling three-year window for ICE BofA series** as of April
2026. Percentile ranks against three years are near-meaningless for a
cycle-scale question. Either license history from ICE Data Indices or append
every daily pull to `data/oas_history.csv` and build it forward. Do not present
a 3-year percentile as though it were a cycle percentile.

**State the base rate.** The US is in recession ~15% of months, so a model that
always says "no" is right 85% of the time. Every lead-time claim rests on eight
recessions since 1976. Prefer a late high-confidence signal to an early noisy
one — calling ten of the last three recessions means sitting in cash through the
expansion that pays for everything.

**Correlated indicators are one signal, not several.** MACD, RSI and StochRSI
are all momentum derivatives of close; on real data they correlate 0.6-0.9. The
detail page groups them as one vote with three readings, with OBV (flow) and
Z-Score (mean reversion) as the two genuinely independent inputs. Same for BTC:
MVRV Z-Score and the realized price oscillator share the numerator
(price - realized price) and agree by construction. `independence_report()`
measures this rather than asserting it — run it before adding any indicator to
a panel, and drop anything correlating above 0.8 with something already there.

**Entry ladder tier 3 has a confirmation gate, not just a lower price.** A price
that good usually means the thesis broke, and price alone cannot separate
"cheap" from "impaired". Tier 3 requires solvency intact plus either insider
buying or estimates not collapsing. Without it the tier renders as a falling
knife warning. Do not remove the gate to make the ladder look cleaner.

**Never fabricate news headlines for real tickers.** The news panel ships
category placeholders labelled PLACEHOLDER for exactly this reason. When wired,
tag each item by whether it confirms or contradicts a stated trigger — an
untagged feed is noise.

**Bull and bear cases must have checkable triggers.** The bear case is the bull
case failing at its own trigger points, not a separate list of generic risks. A
risk you cannot check is decoration; an inverted trigger is a monitor.

**Engines return a `components` dict.** The UI shows why each name ranks, not
just that it ranks. Any new scoring function keeps this contract.

**The free path is the default and should stay that way.** `run_all.py` binds
every loader to `free_sources.py`. Do not swap in a paid provider unless asked.

**SEC EDGAR gives point-in-time for free, and better than cheap paid APIs.**
Every XBRL fact carries its `filed` date, so `extract_series(as_of=...)` returns
what was knowable then. It also keeps first-reported values rather than
restatements, which budget providers overwrite. Never "simplify" that to take
the latest value — it reintroduces lookahead silently.

**Free STH-MVRV does not exist.** `load_sth_mvrv()` returns AGGREGATE MVRV from
Coin Metrics, which is a different and duller measure. This is a labelled
substitution, not an equivalence — do not present it as STH-MVRV in the UI.

## Scope, decided not defaulted

The quality screen is a screen for **asset-light compounders**. Utilities,
REITs and banks are excluded by construction: zero of 131 utilities and REITs
were gate-clean across 1,500 names, because ROIC pinned near an allowed return,
leverage that would disqualify an industrial, and capex exceeding operating
cash flow are what those businesses ARE. The page says so.

Admitting them needs separate screens, not relaxed gates — the SCORING
components are wrong too, not just the gates. `discount_to_own_history` is
built on EV/EBIT, meaningless for a bank where the anchor is price to tangible
book; `fcf_yield` is meaningless for a REIT where it is AFFO yield. Patching
only the gates lets names through to a score assembled from inapplicable
components.

| Sector | Verdict |
|---|---|
| Financials, 176 names | **Build.** Equity, net income, net interest income and noninterest expense are all present; ROE, ROTCE and the efficiency ratio are computable today. ~1 session. |
| REITs | **Buildable.** FFO is not a GAAP tag — derive as net income + depreciation - gains on sale, with two competing gains tags, so it inherits the same stitching and unavailability discipline. ~1 session plus the derivation. |
| Utilities | **Do not build on the free path.** `PublicUtilitiesAllowedRateOfReturnOnEquity` and rate base are absent from EDGAR entirely; they live in FERC Form 1 and state rate-case filings. A screen built from what EDGAR has would rank utilities on metrics that do not describe the business. |

## The recurring bug class — read this first

The same bug has been found in four separate universe runs, on four concepts:

| Concept | Cause | Damage |
|---|---|---|
| revenue | ASC 606 tag migration, 2018 | series froze at 2017 for 3 of 5 megacaps |
| cash | ASU 2016-18 migration, 2019 | killed the EV history for 44% of names |
| gross profit | ASC 606 presentation | Amazon's margin read its 2009 figure |
| debt | truncated for 43.7% of names | inflated Coca-Cola's ROIC 2.4x, to 49.5% |
| interest expense | stale for 46.4% — `InterestExpenseNonoperating` | caught by the check BEFORE a universe run |
| capex | never tagged at all by some filers | FCF fabricated as +48.4% for NextEra; found by random hand-check |
| dividends per share | tagged in TWO currencies at once | HDFC Bank yield 47.45% vs a real 1.11%; found by random hand-check |
| bank revenue | ASC 606 excludes interest income, so `RevenueFromContract...` is only the FEE SLICE | Regions' revenue 1/62 of its total, M&T's 5y CAGR -22.6% against a real +10.2%; Goldman and 13 other banks never built at all |
| taxonomy | ONE stray us-gaap concept made an IFRS filer read as American | BAT, Santander, TotalEnergies and 25 more "had no revenue" |

The shape is always identical: a concept stops being tagged, the truncated
series silently substitutes for the full quantity, and the output is wrong in a
way that looks plausible enough to render.

**`free_sources.concept_freshness()` is the systemic answer.** It compares every
concept's newest annual period against revenue's in one pass. A filer still
reporting revenue through 2025 whose debt stops in 2013 is describing a tag
migration, not a repaid balance sheet. `build()` records `stale_concepts` and
`concept_lags`; critical ones (debt, cash, equity, ocf) hard-fail the shared
gate. Reach for this before adding another tag by hand.

**Freshness is not enough — check COMPOSITION too.** `concept_freshness`
measures each concept against revenue. It cannot see a join that fails while
every input is current: NVIDIA's debt concepts all read lag 0 while total debt
came out EMPTY, because an optional component that died in 2017 vetoed the
intersection. `_derivation_integrity()` asserts the complementary invariant —
a derived quantity must be no more absent, and no more stale, than its freshest
input.

**CORE versus OPTIONAL, not stale versus fresh.** Dropping a stale component is
right for finance leases (NVIDIA stopped tagging them; immaterial) and
catastrophic for long-term debt (Coca-Cola's ends 2023 and is $35.5bn —
dropping it reproduces the zero-fill bug exactly, ROIC 49.5%). A stale CORE
component means the TOTAL is stale: intersect, let the series end honestly, and
let `data_stale_days` report it. A stale OPTIONAL component is left out.

**Freshness has a blind spot: it sees a concept that STOPPED, never one that
never started.** For an absent debt tag, "no debt" and "migrated tag" are
indistinguishable from the series. Interest expense is the corroborating
witness — material interest with no balance means the balance is missing; no
interest means plausibly debt-free, and that company deserves a real ROIC
rather than exclusion. See `debt_assumed_zero`.

**Watch for tags that BUNDLE other components.**
`LongTermDebtAndCapitalLeaseObligations` already includes leases; adding
`finance_leases` on top double-counts them.

**THIRD invariant: no derived quantity may be computed from a zero-filled
input without declaring itself unavailable.** Found by hand-checking five
RANDOM non-megacaps, which is why random sampling is now part of the process.
NextEra tags no capex concept at all, so `ocf - capex.fillna(0)` set free cash
flow equal to operating cash flow — a +48.4% FCF margin and 10 of 10 positive
years, for a utility spending ~$25bn a year on capital and persistently
FCF-negative. The quality screen scored the fabrication as a strength. Five more
names did the same, including one at 100.3%, which would mean free cash flow
equals revenue.

Neither earlier check could see it, and the reason generalises:

| Check | Sees |
|---|---|
| `concept_freshness` | a concept that STOPPED |
| `_derivation_integrity` | a join that COLLAPSED |
| `_fill_integrity` | a concept that NEVER STARTED, filled with a flattering value |

The fill is silent precisely because the arithmetic succeeds. Half the periods
missing voids the series; fewer warns.

**Altman Z does not apply outside manufacturing.** It was fitted on
manufacturers, and regulated utilities, banks and REITs run leverage that reads
as distress without being distressed — NextEra scores 0.54. Skipped via
`altman_not_applicable`.

**`LongTermDebt` is the TOTAL, not the noncurrent portion**, in most filers'
usage. Leaving it in `debt_noncurrent` and then adding `debt_current`
double-counted — Oracle's invested capital read $141.3bn against ~$100bn.

**FOURTH class, and it is invisible to all three checks: MIXED UNITS.** HDFC
Bank tags dividends per share in `INR/shares` (35 facts, ending Rs22.00) AND
`USD/shares` (11 facts, ending $0.26). `extract_series` appended both, the
rupee figure got divided by a dollar ADR price, and the yield published as
47.45% against a real 1.11%. Nebius blends RUB and USD across revenue, net
income, assets, equity, cash, every debt component and operating cash flow.

Freshness sees both units as current. Derivation integrity sees nothing absent.
Fill integrity sees nothing zero-filled. The numbers are simply not the same
kind of thing. `extract_series` now PINS one unit — preferring USD, then
coverage — and records `multi_unit`, `unit_used` and `units_available`. Never
merge units. Expect this to matter far more on a wider list, since the FINRA
ranking pulls in ADRs heavily.

**Record what a fix COSTS, not just that it worked.** Pinning to USD is right
— every ratio divides by a USD price, and mixing bases across concepts is worse
than a short series. But HDFC Bank has 35 INR facts and 11 USD, so pinning
truncates its dividend history from 35 years to 11, capping the increase streak
and every own-history percentile. `unit_coverage_cost` records it.

**Audit every threshold against the case that motivated it.** Four results,
and only one was cleanly live:

| Threshold | Verdict |
|---|---|
| mixed units `>= 3` | sat ABOVE its case — HDB had exactly one. Split in two. |
| freshness `450d` | dead band one reporting cycle wide. Now 200 reference-relative, 550 calendar. |
| `min_den` $10m | live, but left $10m-$200m unguarded. Now scales to 2% of assets. |
| FCF rate `0.8` | live; its case (6 of 6) scores 1.0 |
| ev_hist `500` | live, but NEVER caught its own case — J&J's 1,564 points passed; recency caught it |

**One threshold was serving two different measurements.** Reference-relative
lag (concepts against revenue) should be near zero for a healthy filer, since
two concepts from the same filing share a period end — a 365-day lag means the
concept was not tagged in the latest filing. Calendar-relative lag
(`data_stale_days`, `_align`, `_ratio`) legitimately runs to ~440, being 365 of
cycle plus ~75 of filing lag. Conflating them at 450 let exactly one missed
reporting cycle through unflagged.

**A flat WACC is not an assumption, it is the thing the reverse DCF inverts.**
A 2-point error moves implied growth ~4.5 points — the same size as the signal,
since delivered growth mostly sits in the mid single to low double digits. And
it has a DIRECTION: a flat 9% overcharges a low-beta utility (true ~5.5%) so
the model reads 8.6% embedded growth where the price embeds about -1%, the gap
collapses and it screens EXPENSIVE; the same 9% undercharges high-beta software
so it screens CHEAP. Backwards for a value screen, and it tilts toward the
names whose valuations depend most on the discount rate. `SECTOR_WACC` buckets
off the SIC sector already computed, floored at 4% so `r <= terminal_growth`
never fires.

**Negative book equity breaks equity+debt-cash structurally.** Otis carries
-$5.35bn of equity from its spinoff; netting that against $7.12bn of debt and
$1.10bn of cash leaves $0.66bn of invested capital and a published ROIC of
204.2%. Deckers' was $0.59bn — the ratio guard sat BETWEEN them. Three separate
conditions now void ROIC: non-positive capital, negative equity, and any
computed ROIC over 100%, which on a business of scale is always a denominator
artifact.

**The same near-zero-denominator shape recurs per metric and must be bounded
per metric.** CoStar published an EV/EBIT of 2,998x, 50 names exceeded 200x and
the maximum was 26,625x. Above ~150x the multiple describes the denominator,
not the valuation.

**Off-cycle period ends fake dividend cuts.** Eaton reads "cut 9y ago" and has
never cut: two stray points are quarterly amounts carrying an ANNUAL-length
period context, dated two MONTHS off the December year end, so `_is_annual`
admits them and the +/-7 day dedup cannot merge them. `drop_offcycle_periods`
takes the modal month-day of the filer's own series and drops the outliers.

**Flag denominators at CONSTRUCTION, not by bounding the ratio afterwards.**
Four bounds were added one at a time, each after a different metric blew up:
ROIC 100%, EV/EBIT 150x, revenue 2% of assets, `min_den` $10m. Testing whether
one rule unified them showed it does NOT — two do, distinguishable by how the
denominator is built. `denominator_reliability()` implements both:

- **Cancellation** — a signed sum small relative to its terms. Invested
  capital is equity + debt - cash; Otis nets -$5.35bn of equity against
  $7.12bn of debt for $0.66bn, c = 0.09. **Necessary but not sufficient**, and
  that was the surprise: Trex scores 0.93, comparable to Microsoft, and still
  published a 116.9% ROIC.
- **Materiality** — a residual approaching zero. EBIT is revenue minus costs,
  so EV/EBIT explodes as margin approaches zero. CoStar at 2.2% published
  2,998x; Accenture at 14.8% publishes 10.3x. This separation IS clean.

Materiality is **cause-level** where a 150x cap is symptom-level: CoStar's
2,998x is not an expensive stock, it is a 2.2% margin, and an output bound
cannot tell a real 150x from an artifact one while a margin floor can. A new
ratio inherits its check from how its denominator is built, rather than needing
someone to notice it blew up first.

**A bound must VOID, never substitute — including with zero.** Bounding EV/EBIT
at 150x by ZEROING turned CoStar's artifact into 0.0x, which renders as the
cheapest possible stock. 286 names carried an unvoided 0.0 and 33 a negative,
three of them published. This is "unknown must never read as safe" inverted,
and it was introduced while fixing the bound above it.

**Quarterly stragglers are a MAGNITUDE problem, not a calendar one.** Three
instances on three different date offsets: J&J one day off, Eaton two months
off, Cognex in early October against a December year end — which no calendar
filter catches. A DPS value roughly a quarter of its neighbours is a quarterly
figure whatever date it carries. Compare against a running median so a genuine
halving (3M, near 0.5) is not mistaken for a quarterly figure (near 0.25), and
ITERATE: one pass caught two of Cognex's four, because a centred median
includes the neighbouring stragglers.

**Sector-conditional WACC is not the binding constraint for utilities and
REITs.** It corrects a real tilt — a flat 9% understates low-beta implied
growth by ~8 points — but zero of 131 utilities and REITs were gate-clean, so
the discount rate is never reached. PG&E has 3.3% ROIC and 11.9x leverage;
Welltower 0.5% ROIC and 10.2% annual share issuance. Those are not assumption
artifacts, they are what those businesses structurally look like. **The quality
screen is a screen for asset-light compounders and excludes them by design.**
Admitting them needs sector-conditional GATES, which would change what the
screen means — a decision, not a fix.

**A void enforced in one layer can be undone in the next.** The builder voided
`ev_ebit` to None and the adapter's `_pct` was `round(float(v or 0), nd)` —
turning every voided field back into 0.0, which renders as the cheapest
possible value on a multiple. Only `ev` was guarded explicitly, so nothing
published hit it. Invisible to the written-and-read meta-test, because both a
write and a read exist: **the failure is in the VALUE, not the wiring.** None
must survive the builder, the scorer, the adapter and the renderer. `_scale`
treats None as NEUTRAL rather than bottom-of-range, so an unavailable field
never ranks a name down for missing data.

**None is a CONTRACT, tested exhaustively, not a site-by-site fix.** The same
crash shipped twice, one layer apart: `100 - f.fcf_payout` in the dividend
scorer, then `int(round(f.fcf_payout))` in the adapter building that scorer's
rows. The first fix was verified through `run_screen` and never through
`run_screener`, which is where the adapter runs — a verification path that
skips a layer passes while production fails. A field may hold None only if it
is annotated `| None` on `Fundamentals` (`screeners.voidable_fields()`);
`build()` raises if any other field comes out None; a meta-test fails if
anything `void_derived_fields` or the builder can null is unannotated; and
every scorer, gate, report and row builder runs against ALL voidable fields
None and EACH one alone, across every sector and sub-bucket. Annotating a field
is what enrols it. Proven by reverting each fix and watching the test fail.

**A carried result must say it was carried.** Merging `status.json` kept the
weekly screeners' last outcome through weekday runs — correct — but a carried
Sunday traceback with no timestamp read as a Monday crash, against code that
had been fixed and had not yet run. Carried entries keep their original time
and are marked `carried`.

**Bundled specials are a third dividend category.** Cognex shows $2.2250 in
2020 against $0.2450 in 2021, a ratio of 37 — a regular dividend bundled with
a special, which is neither a cut nor a split, and which magnitude-below alone
cannot separate from a cut. The outlier filter is symmetric: far BELOW the
running median is a quarterly figure, far ABOVE is a bundled special, and a
genuine cut sits near 0.5 between them.

**Guard the denominator from BOTH sides — near-zero, not just non-positive.**
Deckers has equity $2.50bn, no debt and $1.91bn of cash, leaving $0.59bn of
invested capital against $1.26bn of EBIT. It published a ROIC of 281.1% against
a real 35-40% and ranked FIRST on the value screen. The guard caught `<= 0` and
not "smaller than half of EBIT". ADP showed 105.0% the same way.

**A stale derived value feeding a gate is no better than a missing one.**
`derivation_integrity` emits two variants and only one gated: "composition
failure" (EMPTY) did, "truncating the join" (STALE) did not. EnerSys shipped a
net_debt_ebitda of **-0.30x, wrong-signed**, reading net cash for a company
carrying +$0.64bn of net debt — and `net_debt_ebitda` gates all three screens,
so a wrong sign lets a levered company clear a leverage gate.

**Prefer a combined tag only if it is FRESHER than its components.** EnerSys
tags `debt_total` with 4 points ending 2014 while `debt_noncurrent` runs to
2026. Taking the combined tag unconditionally was the mechanism above.

**Never let a placeholder feed a scored component.**
`reverse_dcf_implied_growth` read 0.0 for all 37 value rows, so the
expectations gap was a CONSTANT and the gap and upside columns were both
derived from it. It is now computed — `reverse_dcf_growth()` bisects for the
growth the price embeds from EV, FCF and WACC, all of which the builder already
had. Where it cannot be solved (negative FCF), the component scores neutral and
the column renders blank rather than zero.

**`lru_cache` does nothing on a single sweep of unique tickers.**
`company_sector` is cached and still cost 363 seconds across 1,500 names,
because the cache never hits when every key is distinct. It helps repeated runs
within a process, not this workload. The real wins are removing round-trips:
splits now arrive with the price history (`actions=True`) instead of a second
call, which was 19.8% of a 41.8-minute run.

**Rate-limit backoff must be PER HOST.** A single 62s wait tuned for Polygon's
5-per-minute limiter was applied to SEC, which allows ~10 per second — and
`company_sector` and `company_sic` each fetched the same endpoint separately,
3,000 requests where 1,500 would do. Together they turned a 45-minute run into
a seven-hour one.

**`min_score` must come from each screen's OWN distribution.** One number for
three screens, chosen when the only "data" was invented sample rows. Value
reaches 85 and dividend 91, so 60 sits mid-distribution. Recovery's highest
achievable score across 1,500 names is 68 — a 60 cut there is at 88% of the
observed ceiling. Re-audit whenever component weights change.

**Guard the DENOMINATOR.** Archer Aviation published a 33.33% gross margin on
essentially zero revenue and an FCF margin of -170,566%, with nothing flagged.
`_ratio` takes `min_den`; margins use a $10m revenue floor and `pre_revenue`
gates the name.

**Check that every flag is BOTH written and read.** This class produced six
instances across two sessions, in both directions:

| Field | Failure |
|---|---|
| `sector` | read by three dividend allowances and the Altman skip, never set — those branches had never fired for any name, ever |
| `multi_unit` | set on `df.attrs`, never read — the currency-mixing flag gated nothing |
| `ev_history_degraded` | reached the dataclass only via a `setattr` loop, invisible to any write check |
| `derivation_warnings` | produced by `_derivation_integrity` and only ever printed — the check ran and nothing consumed it |
| `debt_assumed_zero`, `roic_5y_is_ttm_fallback` | set, never surfaced |

A field written but never read is a check that does not run. A field read but
never written is a branch that cannot be reached. **Both look implemented.**

**A third variant the meta-test cannot catch: the read is reachable but the
THRESHOLD sits above the motivating case.** The mixed-unit gate fired at
`>= 3` concepts. HDFC Bank has exactly ONE — `dividends_per_share` in INR and
USD — and that single concept produced the 47.45% yield the check was built
for. Written, read, reachable, and it would never have fired on the case that
justified it. The gate now asks two questions: any mixed unit on a
ratio-critical concept says "a unit was chosen, verify it"; three or more says
"this statement is not internally comparable". A regex proves a field is used;
only a test against the motivating case proves the check works.
`test_every_quality_field_is_both_written_and_read` asserts the invariant by
scanning both modules; it found four of the six on its first run. Avoid
`setattr` loops for anything a gate consults — assign explicitly.

**Containment by a gate is not correctness by construction.** The sharpest
residual the meta-test cannot reach: PayPal STORED an EV/EBIT of 7.25 and an
Altman Z of 1.95 computed on an empty debt series. Both fields written, both
read, read reachable — harmless only because a gate excluded the name first.
Anything reading a `Fundamentals` outside the gate path, such as the ticker
detail page, renders them as fact. `void_derived_fields()` enforces the real
invariant: a field derived from an unavailable input must hold no value.

**The freshness reference cannot check itself.** Revenue is the yardstick, so
its own lag is zero by construction — a filer whose revenue is 500 days old
read lag 0 on every concept and passed. `concept_freshness` now also reports
`reference_lag_days` against the calendar.

**The Altman exemption must be NARROWER than the sector label.** SIC files
crypto miners and SPAC-descended issuers under 6199 "finance services", and
eleven of twenty-four `financial` names were exactly that — all eleven lost the
Altman gate, including Core Scientific at -$0.96bn equity, which is precisely
the case Altman exists for. Use `altman_exempt(sic)`: depository institutions,
insurers, REITs, utilities. Not the coarse label.

**An ADR trades at a MULTIPLE of the underlying share and XBRL never says
what.** HDFC Bank's yield came out at 0.56% against a real 1.11% — out by
roughly its ADR ratio. Unit pinning fixed the currency and could not fix this,
because the information is not in the filing. Foreign per-share yields are
voided, not published: a number wrong by an unknown integer factor is worse
than no number.

**A default that nothing ever sets is dead code.** `sector` defaulted to
"general" and nothing classified it, so the REIT/utility payout, FCF and debt
caps in `dividend_gates` and the Altman skip had NEVER fired for any name in
any run. `free_sources.company_sector()` now reads the SEC's own SIC code.
Check that a branch can actually be reached before trusting that it works.

**Two rules that fall out of it:**

- **No high-risk chain may be single-tag.** Only stable balance-sheet concepts
  qualify, plus ones with their own fallback. A test enforces the allowlist.
- **Sum components on an INTERSECTION, never a zero-filled union.** The union
  form made a component whose series ENDED read as "zero debt" rather than
  "unknown debt" — which is how $35.5bn of Coca-Cola's long-term debt
  disappeared from invested capital.

**A partial concept can hide under a total's name.** Every earlier instance
was a series that STOPPED. Bank revenue never stopped — it was current,
complete, and a fifth of the business, because ASC 606 scopes out financial
instruments and the chain's first tag is the ASC 606 one. None of the three
checks can see it: fresh, composed, not filled. What caught it was comparing
the chain against an independent construction of the same total — net interest
income plus noninterest income matches `Revenues` to the dollar at JPM, BAC, C,
COF and PNC. `_bank_format_revenue` replaces the chain only when it falls
materially SHORT of that total, or is stale beside it, since a slice can only
be smaller: StoneX's gross revenue far above its net figure is left alone, as
is T. Rowe Price, whose bank-format tags end in 2015. Zero non-financials
changed basis across the 1,500.

**Presence is the wrong test, a third time.** After deposits (Franklin's zero
tag) and SIC codes, `taxonomy_of` chose us-gaap on ANY us-gaap key. It now
takes the taxonomy with the newest annual fact — not the larger concept count,
which Itau's 339 stale us-gaap concepts against 336 current IFRS ones defeats.
The same shape is still live in the financial sub-bucket witness, which reads
`interest_income` tag presence at any date: Franklin Resources is a "broker" on
a tag that ended 2014. Not yet fixed — reading stale as absent would also move
Ally, a real bank, to fee_based, because Ally's current interest income is
tagged only by component.

**Fixing one exclusion exposes the next hole.** The taxonomy fix made 28
foreign filers buildable, and those with no USD facts arrive with statements
in CAD, GBP or SEK divided by a USD price. Telus scored valuation_gap 100 and
would have published. Nothing gated it because nothing could reach it before.
`statement_currency` now gates all four screens and voids price ratios.

## Normalise within a peer group ONLY when the difference says nothing about quality

Two structural differences look alike and must be treated oppositely.

**Leverage is structural.** Banks run 7-11% equity-to-assets, insurers 15-30%.
Both are normal for their model, so a shared 6-to-16 scale read a bank's
defining leverage as weakness and capped it near 45 however good it was — the
same failure ROIC imposes on utilities, one level down. Scored as a percentile
WITHIN the sub-bucket, the gap between banks and insurers fell from 42 points
to 23. Use a percentile rather than per-bucket constants: it self-corrects if
the population shifts and cannot silently acquire the same bias again.

**Return spread is economic.** Depositories average a 3.4pt spread over cost of
equity, asset managers 18.8pts. That is a real difference in value creation,
not a difference in what normal looks like. Normalising it within bucket would
say "this is a good bank" while concealing "banks create less surplus than
capital-light financials", so it stays on an absolute scale.

**The test:** if the two groups' normal ranges differ but neither is better,
normalise within group. If one range genuinely is better, leave it absolute.

**Gate thresholds follow the same rule as scoring scales.** If a scale had to
be per-bucket, the floor does too — `FINANCIAL_FLOORS` sets ROE, equity-to-
assets and tangible book per sub-bucket. 6% equity-to-assets is roughly a
10-12% Tier 1 ratio for a bank and alarming for an underwriter.

## Relative cheapness needs an absolute anchor

`discount_to_own_history` was purely `ev_ebit_percentile_10y`, so a name that
had been expensive for a decade and was now merely very expensive scored 100.
MercadoLibre at 31.8x, Autodesk at 29.1x and Intuit at 19.1x all did. A purely
relative measure cannot tell "cheap" from "less expensive than it has ever
been". Both the value screen's `discount_to_own_history` and the recovery
screen's `valuation_gap` now average the percentile with `_scale(30 - ev_ebit,
5, 22)` via `_mean_available`, so an absent multiple drops out rather than
substituting. MELI 98.7 -> 49.3, ADSK 100 -> 50.0, INTU 100 -> 67.4, a genuine
9x name 87.1. The dividend screen already blends its percentile with FCF yield
and needs no change.

## Sub-buckets are assigned by what the filer REPORTS, not by its SIC code

Three rounds of SIC-led assignment produced three rounds of the same error:
6200 "security and commodity brokers" holds the EXCHANGES, 6411 "insurance
agents and brokers" holds commission brokers who never underwrite, 6211 holds
BlackRock and SEI who are asset managers. Each time the witness was right and
the map was wrong. SIC now decides SCOPE only — it is the only thing that
excludes 6199 "Finance services", which holds MicroStrategy, IREN, Coinbase and
Circle alongside genuine lenders.

**Test materiality, not presence.** Franklin Resources carries the deposits tag
with a value of zero and a presence test read it as a bank. Ameriprise holds
$34bn of deposits and $1.4bn of premiums against $191bn of assets and $19bn of
revenue — it HAS a bank and an insurer without BEING either. Real depositories
run deposits above half of assets (JPMorgan 57.8%, Schwab 52.1%) and real
underwriters run premiums at most of revenue (Travelers 89.9%), so a 20% floor
separates them with room to spare.

## Traps found live — do not reintroduce

**`ev_ebit` must use a historical EBIT series, never a scalar.** It was
`(close * shares_today) / ebit_today` — a constant times price — so its
percentile equalled the raw close percentile to machine precision on all four
tickers tested. It carried ~30% of the quality score while measuring "near its
high" under the label "expensive versus its own history". Historical EBIT,
share count and net debt are now stepped onto the price index via `_align`,
each lagged 75 days to its filing date. Anything that reduces one of those to a
scalar reintroduces the bug.

**Never infer splits from a dividend series.** A 2:1 split and a 50% cut are
numerically identical in DPS. A ratio heuristic was tried and explained away
3M's real 2024 halving as a split — turning a disqualifying cut into an
unbroken record. Splits come from yfinance (`equity_splits`) or not at all;
unexplained drops set `dividend_record_ambiguous`, which fails the gate.
Excluding on "cannot verify" is the safe error.

**The tag chain STITCHES across the chain — not first, not richest.** Two
failed approaches, in order. Taking the first tag with any data returned J&J's
single stale 2021 dividend fact instead of the 51 under `...CashPaid`. Taking
the richest by period count was worse: most large filers migrated revenue tags
at ASC 606 in 2018, so the deprecated tag carries a LONGER history ending
exactly at the migration. `SalesRevenueNet` gave Apple 11 periods to 2017-09-30
and beat the current tag's 9 to 2025-09-27 — Apple's revenue silently ended
eight years ago, on three of five test tickers. Tags within a chain are the
same concept, so the answer is both: prefer the tag with the most recent
coverage, backfill gaps from older ones, dedupe on period end.

**Ratios must use a COMMON period.** `_last()` of two series independently gave
Apple's gross margin as FY2025 gross profit over FY2017 revenue: 85.15%. Use
`_ratio()`, which intersects the indices. `_period_mismatch_days` measures the
spread and gates above 400 days.

**`_align` must cap the forward-fill, and this is the subtle one.** Without
`max_stale_days`, J&J's single 2014 EBIT was carried across the whole ten-year
window, making the denominator constant — which turns EV/EBIT back into a
constant times price, so its percentile matched the raw close percentile to two
decimal places. The degraded output was indistinguishable from the fixed one.
Past 450 days the value goes NaN; under 500 surviving points the percentile
returns a neutral 50 and sets `ev_history_degraded`, which fails the value
screen.

**Dedup periods with tolerance, not on the exact end date.** 52/53-week filers
drift: J&J reports the same fiscal year under two tags one day apart
(2021-01-03 and 2021-01-04). Exact-key dedup kept both, and the second was a
QUARTERLY declared rate ($1.01) beside the annual figure ($3.98) — mid-series
that reads as a 3.94x drop and recovery, giving a 5-year streak and a 38% DPS
CAGR. `_PERIOD_TOL_DAYS = 7`.

**Fall back when a series is EMPTY *or* STALE.** J&J has six
`OperatingIncomeLoss` facts ending 2014 while pretax runs to 2025. Testing
emptiness alone meant ROIC and EV/EBIT were computed on eleven-year-old
earnings and reported as current, making the name look ~25% cheaper than it
was, with no flag raised.

**Staleness guards must test RECENCY, not count.** A series can be long and
entirely historical. J&J's aligned EBIT had 1,564 valid points, every one
before June 2016, so a `len(...) < 500` guard passed and `pct_rank`'s `.tail()`
ranked against a series that had already ended. Check the age of the newest
valid point.

**Unknown must read as unknown in BOTH directions, for every input.** The EBIT
trap has two siblings. Absent `GrossProfit` surfaced as "gross margin 0% under
35%" — a business label on a missing tag, which would reject oil, financials
and REITs wholesale. Absent operating cash flow produced two more. There are
now `gross_profit_unavailable` and `fcf_unavailable` beside `ebit_unavailable`,
gross profit is derived from revenue minus cost of revenue where possible, and
**business gates skip any check whose input is flagged** — otherwise an honest
data reason appears next to a misleading business one.

**Measure period mismatch across the series ACTUALLY USED.** It included raw
`oi` even after the builder abandoned it for the pretax fallback, so any filer
that stopped tagging `OperatingIncomeLoss` got a false data-quality failure —
J&J read 4,018 days when its used series were all within 0. This inflates the
data-quality rate, which is the number the universe run depends on.

**Foreign private issuers file under `ifrs-full`, not `us-gaap`.** Reading only
us-gaap made every 20-F filer (WIT, GGB) return no revenue and drop out before
becoming a record, so they never appeared in any coverage or failure statistic.
`taxonomy_of()` detects it and `IFRS_CHAINS` maps the concepts. An invisible
exclusion is worse than a counted one — use `load_fundamentals_report()`, which
returns records plus a categorised account of everything that never became one.

**Ranking a universe by FINRA SHARE volume biases toward low-priced names.**
The 300-name sample came back with a median close of $21.90. FINRA publishes
shares, not dollars, so dollar-volume ranking needs a price join first — or one
Polygon grouped-daily call, which returns close and volume for every US ticker
at once. This is the strongest argument for the free Polygon key.

**A correct ratio on an ancient period is not correct.** Third instance of the
same class, after `_align`'s unlimited ffill and the count-only EV guard.
Intersecting indices fixed the cross-period bug but introduced a subtler one:
Amazon's gross profit series ends 2009, so the intersection IS 2009 and the
margin returned 22.57% — internally consistent, sixteen years old, and it would
have rendered. Netflix 2011, Oracle 2018. `_ratio()` now returns
`(value, stale)` and refuses anything past 550 days.

**Gross margin is a SCORING input, not a hard gate.** A 207-name universe run
found gross profit missing or stale for a large fraction of US large caps —
ASC 606 presentation does not require the tag. Gating on it rejected real
companies for a filing-presentation choice, which measures the filing rather
than the business. ROIC, FCF margin and leverage stay hard gates and are well
covered.

**Omit an unavailable component, never substitute for it.** Substituting a
neutral 50 for gross margin made MISSING data outscore a real 55% margin,
because `_scale(55, 35, 85)` is only 40. Average over the components that
exist.

**Reject implausible derived values.** Revenue minus cost of revenue gave
Verizon an 82.4% gross margin — the filer tags a partial cost line. A number
that looks authoritative and is wrong is worse than an absent one.

**Never leak a sentinel into a rendered field.** `_period_mismatch_days`
returned 99999 when no common period existed, and MPT printed it.

**The `cash` chain must cover the ASU 2016-18 migration.** Filers moved off
`CashAndCashEquivalentsAtCarryingValue` around fiscal 2019 to the
restricted-cash bundle. The chain held ONE tag, so there was nothing to stitch
and cash died in 2019 for most large filers. `net_debt` inherits cash's index,
which killed the EV history for 84 of 190 names — repairing `shares_hist`
moved that number not at all. The successor bundles restricted cash, so net
debt shifts slightly in definition: a labelled approximation.

**Total debt is a SUM of components, not a choice between alternatives.** The
chain held long-term only, so Oracle and Comcast read debt of exactly zero
against roughly $90bn each. That understates invested capital and INFLATES
ROIC — Oracle came out at 138.3%, correctly 12.9%. This is the dangerous
direction: gates run before scores, so a false pass is worse than a false fail.

**Refuse to publish ROIC on a non-positive capital base.** SLS read +2,422%,
ASAN +341%, PTON -313%. The positive ones passed the gate. `roic_unavailable`
now fires when invested capital <= 0, debt is absent, or EBIT is unavailable.

**Balance-sheet lookups need nearest-period matching.** `equity.get(i)` is an
exact-date lookup and fiscal year-ends drift by days across series, so the
history list emptied and `roic_5y` silently became `roic_ttm` for 75 of 190 —
a one-year figure labelled as a five-year mean. `roic_5y_is_ttm_fallback`
flags it when it still happens.

**Gate FCF on a RATE, not an absolute count.** `fcf_positive_years_of_10 < 8`
failed any company listed under eight years regardless of profitability: 56 of
102 failures. Now requires 4+ years of history and 80% positive.

**Absent capex reads as zero capex.** `ocf - capex.fillna(0)` overstates free
cash flow wherever capex is missing — entirely absent for QCOM and Verizon,
three points for NVIDIA. `capex_years_missing` counts it.

**Split-adjust SHARE COUNTS, not just dividends — and mind the direction.**
Counts go UP at a split, per-share figures go down, so `split_adjust` takes
`kind="count"` or `kind="per_share"` and getting it backwards is silent.
Adjusting shares was simply missing while splits were already being fetched:
NVIDIA read +108%/yr share growth, and `share_count_cagr_5y > 0.5` became the
single largest business gate at 126 of 190, rejecting the most aggressive
buyback names in the market as serial diluters. It also poisoned `shares_hist`,
so implied market cap read $0.01T in 2021 against ~$5.3T today and EV/EBIT was
incomparable across any split date.

**Measurement and gating are different jobs.** `ev_history_degraded` is read
only by the value screen and `dividend_record_ambiguous` only by the dividend
screen — correctly. But that made the true loss rate unmeasurable: a run
counted 35.3% from shared gates while 46.5% carried a degraded EV history and
23.5% an unverifiable dividend record. Use `data_quality_report()` for the
honest denominator — and note it reports `hard_by_screen`, because
`ev_history_degraded` hard-excludes 44% of names from the value screen while
being merely a flag elsewhere. A data reason causing a hard exclusion must be
counted as one for the screen it affects.

**Data-quality flags must be read by EVERY screen.** `data_stale_days` and
`ebit_unavailable` were set by the builder and gated only in `dividend_gates`,
so quality and recovery went on ranking names with frozen series — Coca-Cola
failed on "revenue declining -5.9%" purely as an artifact. They now live in
`data_quality_gates`, shared by all three.

**The XBRL dividend horizon caps streaks near 17-18 years.** Coca-Cola's real
record is ~63 years; EDGAR only holds ~18. All five test tickers sit at their
data horizon. The 7-year gate is satisfiable, but a true aristocrat cannot be
distinguished from a merely good grower. Show as ">=18y", never as precise.

**Unknown must never read as safe.** `interest_coverage` used `or 99.0`, so a
company with no derivable EBIT scored maximum coverage from missing data. Oil
majors and many financials report neither `GrossProfit` nor
`OperatingIncomeLoss` — EBIT now falls back to pretax + interest, and when even
that fails, `ebit_unavailable` is set and coverage is 0.

**The predecessor-CIK path can return a series that stops years ago.** XOM's
predecessor stopped filing after the restructuring, so its newest annual period
was 2021 and the 54.95% "growth" was a 2020 COVID trough against 2021,
presented as current. `data_stale_days` is measured and gated above 550 days.

**`min_score` silently dropped names that passed every gate.** Microsoft
cleared every dividend gate and scored 51 against a default of 60; the screen
reported zero rows with no explanation. `run_screen` now logs near-misses.

**The ROIC-decline gate needs a floor.** Declining alone disqualified Microsoft
at 27% ROIC — mean reversion from exceptional to excellent, not erosion. Now
requires decline AND (ROIC below max(15%, 1.5x WACC) OR a >30% relative fall).

**`fp == "FY"` does NOT mean annual.** A 10-K tags its own quarterly
comparatives `fp="FY"`, `form="10-K"`. Filtering on `fp` alone mixed 90-day and
365-day periods in one series, so `revenue_cagr_5y` compared a quarter against
a year and read 45.1% for Apple instead of 8.7%. Period LENGTH via the `start`
field is the only reliable discriminator — `free_sources._is_annual` accepts
330-400 days and passes instants (no `start`) through unchanged. Any new
extraction path must go through it.

**Altman Z needs MARKET equity in X4** when using the original 1968
coefficients. Pairing them with book equity reads a healthy company as
distressed — the same firm scored 13.12 correctly and 1.82 with the mix, which
would trip the recovery engine's `altman_z < 1.8` gate. Without a price series,
use Altman's Z' private-firm variant, which re-fits every coefficient. See
`fundamentals_builder._altman`.

**Field coverage is measured, not assumed.** EDGAR alone fills ~32% of
`Fundamentals`; joining a price series takes it to ~78%. All 11 quality gates
are satisfiable from EDGAR. Prices are what enable every "vs its own 5-10y
history" measure, which is what all three screens actually rank on. Run
`fundamentals_builder.coverage_report()` before trusting a screen — a record
that is mostly defaults fails gates for reasons unrelated to the company.

**`eps_revision_3m/6m` have no free source** and default to 0. That is safe
rather than convenient: the dividend yield-trap gate only fires below -20%, so
zero passes without admitting a trap. It does mean the trap filter is inactive
on the free tier, and the UI should say so.

**`ReferenceRateUSD` is 7-day capped on the Coin Metrics community tier.** It
returns HTTP 200, seven rows, no error field and no page token — nothing says
it was truncated. Use **`PriceUSD`**, same series, full history to 2013.
`coinmetrics()` now swaps it automatically and raises if any series comes back
implausibly short. HTTP 200 is not evidence of a complete response.

**`NA` is a real NYSE ticker** and pandas coerces it to NaN under the default
NA list, after which a `notna()` filter drops it silently, every day.
`keep_default_na=False` in `fetch_finra_day` is load-bearing.

**Rate-limit backoff must match the limiter's window.** `_get` slept 1s then 2s
and gave up three seconds into Polygon's 60-second window. It now waits 62s on
a 429 and honours `Retry-After`. Fine for the nightly single call either way;
essential for any backfill loop.

**Dispersion belongs in LEAD, not STRESS.** It was in stress until a live run
showed 99.9th-percentile dispersion producing a stress score of 24.2. That is
structural, not bad luck: if dispersion leads headline HY OAS, then whenever it
is extreme the coincident measures beside it MUST be calm, so an average buries
it every time. Same error the lead/stress split exists to prevent, one level
down. It is now weighted into lead AND surfaced as a standalone
`credit_early_warning` flag requiring high percentile plus active widening.

## Live source status — verified 10 Sep 2026

| Source | State |
|---|---|
| FRED | works. HY OAS 2.71, CCC-BB dispersion 9.06 |
| SEC EDGAR | works. Point-in-time verified live |
| Coin Metrics | works. MVRV 1.441 |
| FINRA | endpoint works, 12,326 symbols same-day |
| Binance | **HTTP 451 from US IPs, including GitHub runners.** Replaced by Coinbase + Coin Metrics |
| Stooq | **behind a JS proof-of-work challenge.** Replaced by Polygon grouped-daily / yfinance |
| Stock Watcher | **dead — 403 + NXDOMAIN.** No free replacement; politicians engine is blocked |

Do not bypass Stooq's bot challenge. It is an explicit anti-bot control.

**Polygon grouped-daily is the right shape** for the tape denominator: one call
returns OHLCV for every US ticker on a date. Free tier is 5 calls/min, no card.
One call per trading day, not one per symbol.

**`load_ptrs` raises rather than returning `[]`.** Stock Watcher's death would
otherwise read as "no congressional trades" instead of "no data source". Any
loader that can fail totally must raise; per-item failure may warn and continue.

**XBRL ticker maps point at the current registrant**, which after a
restructuring can be a shell with no history — XOM maps to CIK 2115436 with
zero annual facts while the history sits under predecessor 34088. See
`PREDECESSOR_CIK`. Exclusions are logged loudly; a silently dropped mega-cap
changes every screen.

## Honesty constraints

**Sample data must never render silently — a footer label is not a label.**
This replaces "the dashboard says so in the footer", which failed on the live
site. On 13 Sep 2026 every tab rendered the invented rows hardcoded in
index.html (OGN 8.94%, LYB 6.84%) with nothing on screen to say so, for three
independent reasons, any one of which was enough:

| Cause | Mechanism |
|---|---|
| `bitcoin.json` held 14 bare `NaN` tokens | the browser's JSON parse threw inside a shared `Promise.all`, `boot()` fell into its catch, and ALL tabs kept the sample |
| `--only recession` rewrote `status.json` with one engine | every other tab lost its status and was skipped as "not ok" |
| weekday runs recorded `skipped` for weekly screeners | Sunday's `ok` was overwritten six days in seven |

Rules now: every tab carries a state — live, stale, failed, missing or
sample — read by the banner, nav, tape, table and footer alike. A failed or
missing run renders an explicit empty state with the reason and timestamp,
never older rows and never sample rows. Sample appears only when
`status.json` is unreachable (a cold-start demo) or a tab's renderer is not
yet wired to its file, and then under a sticky banner with every row and card
stamped. Every tab shows its file's age; stale is judged against the engine's
cadence. `run_all.dump_json` writes strict JSON (`allow_nan=False`), and
`status.json` is MERGED across runs, never replaced.

The Bitcoin and Recession tabs are still sample by construction: their charts
draw from constants embedded in index.html and do not read `bitcoin.json` or
`recession.json`. The ticker detail page generates its price, ladder and
indicators from the ticker string. All three are bannered until wired.

**Politician names in sample data are fictional, deliberately.** Attaching
invented performance figures to real named officials is defamatory. Live data
pulls real names from public filings; sample data never does.

**Weights are reasoned priors, not fitted parameters.** Gates are defensible
from first principles; weights are not yet tested. Do not describe any engine
as backtested until it has been.

**The BTC four-year cycle rests on three completed cycles.** That is an
anecdote count, not a sample. The current cycle matched on timing (peak day 535
post-halving vs 526 and 548) and broke on depth (-39% vs -78/-84/-86%). Report
both; do not resolve it.

**Point-in-time data before any backtest.** Screening today's fundamentals
against historical prices is lookahead bias — financials get restated and
delisted companies vanish, so you test survivors only. Backtests built the
wrong way look spectacular and are fiction.

## Conventions

- Python 3.12, pandas/numpy/requests only. No framework.
- Cast numpy scalars with `float()`/`bool()` before JSON — `np.bool_` is not
  serialisable and this has bitten twice.
- Data loaders in `run_all.py` are intentionally `NotImplementedError` stubs.
  Do not replace one with a plausible-looking default; an unwired loader that
  raises is better than one that silently returns wrong numbers.
- `index.html` is a single file with no build step. Keep it that way unless
  asked. No `localStorage`.
- Engine accent colours are fixed: darkpool violet `#8B7BF0`, dividend jade
  `#4FC08D`, recovery amber `#E8A13A`, value azure `#4A9EE8`, politicians pink
  `#E06C9F`, bitcoin teal `#35C9D4`.

## Interpretation choices worth knowing

- "Stock RSI" is implemented as **Stochastic RSI**, since it was requested
  alongside plain RSI. One-line swap if that reading is wrong.
- "Z-Score" on the detail page is a **price z-score** vs a rolling mean, not the
  Altman Z-Score — Altman already serves as a solvency gate in `screeners.py`.
- BTC indicator sets are **regime-conditional**: bear runs MFI, MVRV Z, realized
  price oscillator and RSI across monthly/weekly/daily; bull collapses to RSI
  and ATR on the daily. ATR is directionless — it is for trailing stops and
  blowoff detection, never direction.

## Commands

```bash
python run_all.py                          # all engines, respects cadence
python run_all.py --only bitcoin --force   # one engine, ignore cadence
python run_all.py --notify --dry-run       # print Slack payloads, send nothing
python run_all.py --digest --dry-run       # preview the biweekly digest
python alerts.py                           # alert dry run against a fake transition
```

Always use `--dry-run` when testing anything that touches Slack.

## Current state

Engines and scoring are written and tested. The six data loaders are stubs, so
the dashboard runs on embedded sample data. Highest-value next steps, in order:

1. Wire the loaders (start with FINRA daily — free, no key, verifies fastest).
2. Point-in-time fundamentals, then a walk-forward backtest.
3. A signal ledger: log every fired signal with forward returns at 1w/1m/3m.
   After six months that gives real hit rates for these exact rules.
