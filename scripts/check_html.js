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

  let rows = 0;
  for (const k of api.ORDER) {
    const e = api.ENGINES[k];
    if (!e) throw new Error(`ORDER references missing engine: ${k}`);
    if (e.custom) continue;
    if (!e.rows || !e.rows.length) throw new Error(`${k} has no rows`);
    for (const r of e.rows) {
      if (!r.ticker) throw new Error(`${k} row missing ticker`);
      api.tkDetail(r.ticker, r.name, k);
      rows++;
    }
  }
  api.btcDetail();
  console.log(`check_html OK — ${api.ORDER.length} engines, ${rows} detail pages render`);
} catch (err) {
  console.error("check_html FAILED:", err.message);
  process.exit(1);
}
