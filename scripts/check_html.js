#!/usr/bin/env node
// Parses the <script> block in index.html and renders every engine's view.
// index.html is edited by string replacement, and a missing comma between row
// objects has silently broken the page twice. This catches it in ~200ms.
const fs = require("fs");
const path = process.argv[2] || "index.html";
if (!fs.existsSync(path)) process.exit(0);

const html = fs.readFileSync(path, "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error("check_html: no <script> block found"); process.exit(1); }

const stub = () => ({
  innerHTML: "", firstElementChild: {}, style: { setProperty() {} },
  appendChild() {}, remove() {}, focus() {}, setSelectionRange() {},
  dataset: {}, addEventListener() {}, onclick: null, oninput: null,
  // The views wire their own controls by querying inside the element they
  // just filled, rather than from document — so the stub needs this too.
  querySelectorAll: () => [],
  classList: { add() {}, remove() {}, toggle() {} },
});
const doc = {
  getElementById: stub, querySelectorAll: () => [], createElement: stub,
  body: { style: {}, appendChild() {} }, addEventListener() {},
  documentElement: { style: { setProperty() {} } },
};

// The watchlist and compare views read localStorage. Node has none, and the
// page is written to degrade to memory when a browser refuses it, so this
// stub exercises the working path rather than the fallback.
const store = {};
global.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: (k) => { delete store[k]; },
};

try {
  const api = new Function("document", "window", "fetch",
    "return (function(){" + m[1] +
    "; return {ENGINES,ORDER,VIEW_ORDER,VIEWS,BEST_DIR,tkDetail,btcDetail," +
    "renderWatchlist,renderCompare,renderOverview,SCREEN_KEYS,SIGNAL_KEYS," +
    "wlToggle,cmpToggle,cmpGet};})()"
  )(doc, { scrollTo() {} }, () => Promise.reject("no net"));

  // Fields renderTab interpolates. A missing one printed the literal string
  // "undefined" into a styled box on the Property Trusts and Financial
  // Returns tabs, for as long as those tabs have existed: both were authored
  // with `how:` while the renderer reads `method`, and nothing read `how`.
  // Rendering the page cannot catch that — the template produces valid HTML
  // containing the word "undefined" — so it is asserted here.
  const REQUIRED = ["name", "title", "thesis", "method"];

  let rows = 0, tabs = 0;
  for (const k of api.ORDER) {
    const e = api.ENGINES[k];
    if (!e) throw new Error(`ORDER references missing engine: ${k}`);
    for (const f of REQUIRED) {
      if (e[f] === undefined || e[f] === null || e[f] === "")
        throw new Error(`${k} is missing ${f} — renderTab would print "undefined"`);
    }
    if (e.custom) continue;
    tabs++;
    // A tab may legitimately ship no SAMPLE rows: live mode discards the
    // embedded rows anyway, and an empty demo table is this project's stated
    // preference over invented figures attached to real tickers. What must
    // not happen is a malformed row, which is what this check exists for.
    for (const r of e.rows || []) {
      if (!r.ticker) throw new Error(`${k} row missing ticker`);
      api.tkDetail(r.ticker, r.name, k);
      rows++;
    }
  }
  api.btcDetail();

  // The views are not engines and so are not in ORDER, which means nothing
  // above renders them. They are the newest code on the page and the most
  // likely to carry a typo, so they get exercised here: once empty, once with
  // a slate drawn from a real board. A Best mark on the wrong end of a row is
  // worse than no mark, so the direction table is checked for sanity too —
  // every entry must be exactly +1 or -1, never 0 or a truthy accident.
  for (const [k, d] of Object.entries(api.BEST_DIR)) {
    if (d !== 1 && d !== -1)
      throw new Error(`BEST_DIR.${k} is ${d} — must be +1 or -1`);
  }
  api.renderOverview();           // every engine on sample rows
  api.renderWatchlist();          // empty state
  api.renderCompare();            // empty state
  const seed = api.ORDER.map((k) => api.ENGINES[k])
    .filter((e) => !e.custom && e.rows && e.rows.length >= 2)[0];
  if (!seed) throw new Error("no engine with sample rows to seed the views");
  const two = seed.rows.slice(0, 3).map((r) => r.ticker);
  for (const t of two) { api.wlToggle(t); api.cmpToggle(t); }
  if (api.cmpGet().length < 2)
    throw new Error("compare slate did not accept two tickers");
  api.renderWatchlist();          // populated
  api.renderCompare();            // populated, with Best marks
  api.renderOverview();           // again, now with a watchlist behind it
  // Compare caps the slate; a fourth must be refused rather than silently
  // dropping one of the three.
  if (api.cmpToggle("ZZZZ") !== false)
    throw new Error("compare accepted a 4th ticker past CMP_MAX");
  // Every engine must sit in exactly one nav group. A key in neither is
  // invisible on the page; a key in both renders twice.
  const grouped = [...api.SCREEN_KEYS, ...api.SIGNAL_KEYS];
  for (const k of api.ORDER) {
    const n = grouped.filter((g) => g === k).length;
    if (n !== 1) throw new Error(`${k} appears in ${n} nav groups, must be 1`);
  }
  for (const k of grouped) {
    if (!api.ORDER.includes(k)) throw new Error(`nav group lists unknown engine ${k}`);
  }
  const views = api.VIEW_ORDER.length;
  console.log(`check_html OK — ${api.ORDER.length} engines, ${tabs} table tabs, `
    + `${views} views, ${rows} detail pages render`);
} catch (err) {
  console.error("check_html FAILED:", err.message);
  process.exit(1);
}
