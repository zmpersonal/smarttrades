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
});
const doc = {
  getElementById: stub, querySelectorAll: () => [], createElement: stub,
  body: { style: {}, appendChild() {} }, addEventListener() {},
  documentElement: { style: { setProperty() {} } },
};

try {
  const api = new Function("document", "window", "fetch",
    "return (function(){" + m[1] + "; return {ENGINES,ORDER,tkDetail,btcDetail};})()"
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
  console.log(`check_html OK — ${api.ORDER.length} engines, ${tabs} table tabs, ${rows} detail pages render`);
} catch (err) {
  console.error("check_html FAILED:", err.message);
  process.exit(1);
}
