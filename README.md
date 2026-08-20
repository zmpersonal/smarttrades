# SmartTrades.ai — Free SEC Beta

This version deliberately removes the paid-market-data dependency from the public beta.

## What V0.2 does

- covers a focused list of ~50 large U.S. operating companies
- downloads standardized company facts from SEC EDGAR
- calculates Quality, Growth, Financial Strength and Smart Scores
- publishes five free Top 10 rankings:
  - Quality Growth
  - Dividend Growth
  - Compounders
  - Cash Machines
  - Low-Debt Growth
- provides a free public screener and stock snapshots
- refreshes every weekday and deploys GitHub Pages in the same workflow

## What it intentionally does NOT do yet

- live stock prices
- P/E, P/FCF or other current valuation multiples
- price momentum
- analyst estimates
- paid accounts / paywall in the visible UI

Those require a suitable licensed market-data source. They should be added after the SEC-only engine is stable.

## API keys

**None are required for V0.2.**

The updater uses SEC EDGAR Company Facts. SEC's APIs do not require authentication or API keys.

The old `FMP_API_KEY` GitHub secret can remain in the repository, but this version does not read it.

## Launch / update

1. Replace the existing repository files with this build.
2. Keep GitHub Pages source set to **GitHub Actions**.
3. Open **Actions**.
4. Run **Refresh SmartTrades fundamentals**.
5. Confirm the workflow reaches:
   - `Refresh SEC fundamentals and rankings`
   - `Commit refreshed public rankings`
   - `Deploy Pages`
6. Open `/data/public.json` on the live site and confirm:
   - `is_demo` is `false`
   - `coverage_count` is at least 25
   - rankings contain real ticker rows

## SEC automated-access behavior

The script identifies itself with a User-Agent and throttles requests to roughly 5.5 requests/second. If you prefer, change `SEC_USER_AGENT` in `.github/workflows/update-data.yml` to a monitored email address on the SmartTrades domain.

## Cloudflare

Cloudflare is **not required for this data release**. The old Worker/D1 files remain in `worker/` for later use when adding email capture, accounts, watchlists and subscriptions.

Recommended sequence:

1. Make the SEC rankings deploy successfully.
2. Inspect the real ranking output for sanity.
3. Then configure Cloudflare D1 + Worker for lead capture/accounts.
4. Add licensed valuation data only after that.
