# SmartTrades.ai

GitHub Pages front end + scheduled quantitative ranking engine + optional Cloudflare Worker/D1 membership backend.

## Product included

- Smart Score (quality, growth, valuation, financial strength, momentum)
- 8 automated ranking engines
- Dividend Growth Leaders
- Quality Growth at a Reasonable Price
- GARP
- Quality on Sale
- Compounders
- Cash Machines
- Fallen Angels
- Low-Debt Growth
- Public limited screener
- Pricing / conversion flow
- Newsletter lead capture
- Accounts + watchlists in Cloudflare D1
- Stripe subscription checkout wiring
- Full paid rankings stored behind the Worker rather than shipped in the public GitHub Pages artifact
- Daily weekday refresh workflow with direct GitHub Pages deployment

## Important launch state

The repo intentionally contains **no fabricated live rankings**. `data/public.json` starts empty with `is_demo: true`. The banner disappears after the first successful licensed-data refresh.

## 1. GitHub Pages

Create a repo, upload everything including `.github`, then:

1. Settings → Pages → Source: **GitHub Actions**
2. Set custom domain `smarttrades.ai`
3. Add the required market-data secret described below.
4. Run **Refresh SmartTrades data** manually once.

DNS for GitHub Pages root domain:

- `185.199.108.153`
- `185.199.109.153`
- `185.199.110.153`
- `185.199.111.153`

`www` CNAME → your GitHub Pages hostname.

## 2. Market data

V0.1 implements the **Financial Modeling Prep adapter**. Set GitHub secret:

`FMP_API_KEY`

**Do not treat a personal/developer FMP account as permission to publicly display or resell its data.** SmartTrades is a commercial multi-user product, so obtain the applicable display/redistribution rights before launching live rankings.

The updater calls standardized profile, quote, financial statement, ratio, metric and dividend endpoints, computes SmartTrades' own derived scores, and publishes only the top public results. Full rankings are written temporarily to `.private/`, synced to the Worker, then deleted before Pages deployment.

Universe: `config/universe.csv` (roughly 180 liquid U.S. companies initially). Expand after provider limits/licensing are settled.

## 3. Cloudflare Worker / D1

The backend is optional for the public preview but required for accounts, watchlists, leads and paid rankings.

From `worker/`:

```bash
npx wrangler d1 create smarttrades
```

Copy the database ID into a new `worker/wrangler.toml` based on `wrangler.toml.example`.

Apply schema:

```bash
npx wrangler d1 execute smarttrades --file=schema.sql --remote
```

Set secrets:

```bash
npx wrangler secret put JWT_SECRET
npx wrangler secret put SYNC_TOKEN
npx wrangler secret put STRIPE_SECRET_KEY
npx wrangler secret put STRIPE_WEBHOOK_SECRET
npx wrangler secret put STRIPE_PRICE_PRO
npx wrangler secret put STRIPE_PRICE_PREMIUM
```

Use a long random value for `JWT_SECRET` and `SYNC_TOKEN`.

Deploy:

```bash
npx wrangler deploy
```

Then edit `/assets/config.js`:

```js
window.SMARTTRADES_CONFIG = {
  apiBase: "https://api.smarttrades.ai",
  paymentsEnabled: false,
  demoMode: false
};
```

Add GitHub secrets:

- `SMARTTRADES_API_BASE` — Worker URL
- `SMARTTRADES_SYNC_TOKEN` — same value as Worker `SYNC_TOKEN`

## 4. Stripe

Create two recurring monthly prices:

- SmartTrades Pro — $29/month
- SmartTrades Premium — $79/month

Set their Price IDs as Worker secrets `STRIPE_PRICE_PRO` / `STRIPE_PRICE_PREMIUM`.

Create a Stripe webhook pointing to:

`https://YOUR-WORKER/api/stripe-webhook`

Subscribe at minimum to `checkout.session.completed`.

After testing, set `PAYMENTS_ENABLED = "true"` in the Worker config and `paymentsEnabled: true` in `/assets/config.js`.

**Before accepting money, harden the webhook/subscription-state handling and finalize Terms, Privacy, refund/cancellation language, and financial-data licensing.** V0.1 checkout proves the flow; production billing should also process renewals, cancellations and failed payments.

## 5. Ranking methodology

General Smart Score:

- Quality 30%
- Growth 20%
- Valuation 20%
- Financial strength 20%
- Momentum 10%

The eight specialized ranking formulas are published on their pages. Most factors are percentile-ranked within the coverage universe, which reduces unit-scale problems. V0.1 does not claim sector-neutral purity; sector-relative normalization is the next major research enhancement.

## 6. Security / monetization architecture

Do **not** put the full paid dataset into a public JSON file on GitHub Pages. The updater:

1. fetches licensed raw data,
2. calculates rankings,
3. writes a small public preview to `data/public.json`,
4. writes full data to `.private/`,
5. syncs `.private/` to Cloudflare D1,
6. deletes `.private/`,
7. deploys only the public site.

This makes the paid dataset meaningfully gateable.

## 7. Next production improvements

- Sector-relative scoring
- 5Y historical valuation percentiles
- analyst estimate revisions (if licensed)
- ranking-entry / exit history
- email alerts through Resend/Postmark
- password reset + email verification
- full Stripe lifecycle webhook handling
- custom screens for Premium
- backtest engine with point-in-time constituent controls
- SEC CompanyFacts adapter for selected raw fundamentals

## Disclaimer

SmartTrades is an informational research product, not individualized investment advice. Rankings are derived from financial inputs and can be wrong, incomplete, delayed or become stale. High scores do not guarantee returns.

## FMP entitlement test (run this before a full refresh)

Use **Actions → Test FMP access → Run workflow**. This makes only two requests: an AAPL quote and an AAPL annual income statement. If either returns HTTP 402, the current FMP subscription does not grant the endpoint/data entitlement needed by the v0.2 FMP adapter. If it returns HTTP 429, wait for the FMP quota reset before testing again.

Do not repeatedly run the full updater to diagnose FMP access. The production adapter requests multiple datasets per security and can consume a low-tier daily quota quickly.
