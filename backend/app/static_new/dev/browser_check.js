#!/usr/bin/env node
/* node backend/app/static_new/dev/browser_check.js [base] [outDir] [realPageId]
   Drives headless Chromium (DevTools protocol, debugging port 8014) against a running app — default
   http://127.0.0.1:8013/demo/ — and checks the v3 table UI end to end: the dev fixture on the review page (tables,
   editable cells, kind badge + override, split export + menu, per-table confidence, text / overlay panes), the
   process / settings / help / documents pages, a stubbed v3 backend (structure engine selector, LLM re-run) and,
   with a real page_id, that a page without doc_kind / tables renders exactly as the pre-v3 docview did.
   Screenshots go to outDir. Exit code 1 when a check fails. Needs `chromium` on PATH and node >= 22. */
"use strict";
const {spawn} = require("child_process"), fs = require("fs"), path = require("path"), os = require("os");
const BASE = process.argv[2] || "http://127.0.0.1:8013/demo/";
const OUT = process.argv[3] || path.join(os.tmpdir(), "khocr_shots");
const REAL = process.argv[4] || "";
const PORT = 8014, FID = "0de00000000000000000000000007ab1";
fs.mkdirSync(OUT, {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));
let fails = 0, n = 0;
const check = (name, ok, extra) => { n++; if (ok) console.log("ok   " + name); else { fails++; console.log("FAIL " + name + (extra == null ? "" : "  -> " + JSON.stringify(extra).slice(0, 400))); } };
const DOC = `JSON.parse(localStorage.getItem('khocr.docs.v1')).find(x => x.page_id === '${FID}')`;

async function main() {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "khocr_profile_"));
  const chrome = spawn("chromium", ["--headless", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, "--window-size=1440,1100", "--hide-scrollbars",
    "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--font-render-hinting=none", "about:blank"], {stdio: ["ignore", "ignore", "pipe"]});
  let err = ""; chrome.stderr.on("data", d => { err += d; });
  try {
    let targets = null;
    for (let i = 0; i < 60 && !targets; i++) { await sleep(250); try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); } catch {} }
    if (!targets) throw new Error("chromium did not start: " + err.slice(-600));
    const target = targets.find(t => t.type === "page");
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = e => rej(new Error("ws: " + (e.message || e))); });
    let mid = 0; const pending = new Map(); const consoleErrors = [];
    ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(m.error.message)) : p.res(m.result); }
      else if (m.method === "Runtime.exceptionThrown") consoleErrors.push(m.params.exceptionDetails.exception && m.params.exceptionDetails.exception.description || m.params.exceptionDetails.text);
      else if (m.method === "Log.entryAdded" && m.params.entry.level === "error") consoleErrors.push(m.params.entry.text + " " + (m.params.entry.url || "")); };
    const send = (method, params = {}) => new Promise((res, rej) => { const id = ++mid; pending.set(id, {res, rej}); ws.send(JSON.stringify({id, method, params})); });
    const evaluate = async expression => { const r = await send("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
      if (r.exceptionDetails) throw new Error("page JS: " + (r.exceptionDetails.exception && r.exceptionDetails.exception.description || r.exceptionDetails.text)); return r.result.value; };
    const waitFor = async (expression, ms = 15000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { try { if (await evaluate(expression)) return true; } catch {} await sleep(150); } return false; };
    const shot = async (name, full) => { let clip = null; if (full) { const m = await send("Page.getLayoutMetrics"); clip = {x: 0, y: 0, width: 1440, height: Math.min(5000, Math.ceil(m.cssContentSize.height)), scale: 1}; }
      const r = await send("Page.captureScreenshot", {format: "png", captureBeyondViewport: !!full, ...(clip ? {clip} : {})}); const p = path.join(OUT, name); fs.writeFileSync(p, Buffer.from(r.data, "base64")); console.log("     screenshot " + p); };
    const nav = async url => { await send("Page.navigate", {url}); await sleep(300); await waitFor("document.readyState === 'complete'"); };
    const go = async hash => { await evaluate(`location.hash === '${hash}' ? route() : (location.hash = '${hash}')`); await sleep(250); };  // same hash: no hashchange, so route()
    const text = sel => evaluate(`(document.querySelector('${sel}') || {}).textContent || ''`);
    const count = sel => evaluate(`document.querySelectorAll('${sel}').length`);
    await send("Page.enable"); await send("Runtime.enable"); await send("Log.enable");

    // ---- 1. the dev fixture on the review page
    await nav(BASE + "?dev=table_page");
    check("dev hook opens the fixture on the review page", await waitFor("location.hash.startsWith('#/review/') && !!document.querySelector('.dv-table')"), await evaluate("location.hash"));
    await sleep(1500);  // fonts + fit
    await shot("01_review_fixture.png", true);
    const nCells = await evaluate(`${DOC}.result.document.elements.filter(e => e.type === 'table').reduce((a, e) => a + e.cells.length, 0)`);
    check("two tables rendered", await count(".dv-tablewrap") === 2);
    check("every cell is contenteditable", await count('.dv-table [contenteditable="true"]') === nCells && nCells === 36, [await count('.dv-table [contenteditable="true"]'), nCells]);
    check("merged cells (3 rowspan, 2 colspan) and 9 header cells", await count(".dv-table [rowspan]") === 3 && await count(".dv-table [colspan]") === 2 && await count(".dv-table th.dv-th") === 9);
    check("one low cell highlighted, colgroup widths present", await count(".dv-table .dv-lowcell") === 1 && await count(".dv-table colgroup col") === 8);
    check("table geometry: first table ~1040 px wide at scale", await evaluate("(() => { const w = document.querySelector('.dv-tablewrap[data-block=\"b3\"]'); const s = parseFloat(w.style.width); return s > 300 && Math.abs(s / parseFloat(document.querySelector('.dv-page').style.width) - 1040 / 1240) < 0.01; })()"));
    check("kind badge: Mixed · small model", /Mixed/.test(await text("#r_kind")) && /small model/.test(await text("#r_kind")), await text("#r_kind"));
    check("split export button defaults to Word", /Export Word/.test(await text("#r_exportdef")), await text("#r_exportdef"));
    check("About: Kind and per-table confidence rows", /Mixed · small model · 81%/.test(await text("#view")) && /T1 6×5/.test(await text(".kv .tconf")) && /91%/.test(await text(".kv .tconf")) && /78%/.test(await text(".kv .tconf")), await text(".kv .tconf"));
    check("Re-run with AI model is disabled without a reachable LLM", await evaluate("document.querySelector('#r_llm').disabled === true"));
    check("table tooltip carries the confidence", await evaluate("document.querySelector('.dv-tablewrap[data-block=\"b3\"] table').title") === "Table 1 · 91 %");

    // ---- 2. kind override
    await evaluate("document.querySelector('#r_kind').click()"); await sleep(200);
    check("kind menu: hint + 3 kinds", await count(".menu button[data-k]") === 3 && /Detected: Mixed · small model · 81% · 2 tables/.test(await text(".menu .hint")), await text(".menu .hint"));
    await shot("02_kind_menu.png");
    await evaluate("document.querySelector('.menu button[data-k=\"sheet\"]').click()"); await sleep(400);
    check("override -> Spreadsheet · you, stored with source=user", /Spreadsheet/.test(await text("#r_kind")) && /you/.test(await text("#r_kind")) && await evaluate(`${DOC}.result.doc_kind.kind === 'sheet' && ${DOC}.result.doc_kind.source === 'user' && ${DOC}.result.doc_kind.tables === 2`));
    check("default export follows the kind: Excel", /Export Excel/.test(await text("#r_exportdef")), await text("#r_exportdef"));
    await shot("03_review_sheet_kind.png");

    // ---- 3. export menu
    await evaluate("document.querySelector('#r_export').click()"); await sleep(200);
    const items = await evaluate("[...document.querySelectorAll('.menu button')].map(b => (b.dataset.a || '') + ':' + b.textContent.trim() + (b.disabled ? ':disabled' : ''))");
    check("export menu: Word, Excel (default), CSV, text, JSON, image", items.some(x => x.startsWith("docx:Word")) && items.some(x => /^xlsx:Excel.*default$/.test(x)) && items.some(x => x.startsWith("csv:CSV") && !/disabled/.test(x)) && items.some(x => x.startsWith("txt:")) && items.some(x => x.startsWith("json:")) && items.some(x => x.startsWith("png:")), items);
    await shot("04_export_menu.png");
    await evaluate("document.querySelector('.menu button[data-a=\"docx\"]').click()");
    check("Word export against this backend works (toast)", await waitFor("/downloaded \\.docx|export failed/.test(document.querySelector('#toast').textContent)", 20000) && /downloaded \.docx/.test(await text("#toast")), await text("#toast"));
    await evaluate("document.querySelector('#r_export').click()"); await sleep(150); await evaluate("document.querySelector('.menu button[data-a=\"xlsx\"]').click()");
    const xl = await waitFor("/downloaded \\.xlsx|export failed/.test(document.querySelector('#toast').textContent)", 20000);
    console.log("     Excel export toast: " + await text("#toast") + (xl ? "" : " (no toast)"));
    check("Excel export reaches POST /export/xlsx (ok or a clean failure toast on a pre-v3 backend)", xl);
    await evaluate("document.querySelector('#r_export').click()"); await sleep(150); await evaluate("document.querySelector('.menu button[data-a=\"csv\"]').click()");
    const cs = await waitFor("/downloaded \\.csv|export failed/.test(document.querySelector('#toast').textContent)", 20000);
    console.log("     CSV export toast: " + await text("#toast") + (cs ? "" : " (no toast)"));
    check("CSV export reaches POST /export/csv", cs);
    check("plain text export has tab-separated table rows", await evaluate(`pageText(${DOC}.result).includes('សរុប\\t\\t\\t២៥៤\\t១៨០') && pageText(${DOC}.result).includes('Item\\tQty\\tPrice')`));

    // ---- 4. editing a cell writes back to the element, the block cell and its line
    const j = await evaluate(`${DOC}.result.document.elements.find(e => e.type === 'table').cells.findIndex(c => c.r === 2 && c.c === 1)`);
    await evaluate(`(() => { const td = document.querySelector('.dv-tablewrap[data-block="b3"] [data-cell="${j}"]'); td.focus(); td.innerText = 'ម៉ម ចាន់ថា'; td.dispatchEvent(new Event('blur')); return true; })()`);
    await sleep(200);
    const wb = await evaluate(`(() => { const d = ${DOC}; const el = d.result.document.elements.find(e => e.type === 'table'); const b = d.result.blocks.find(x => x.id === 'b3'); const bc = b.table.cells.find(c => c.r === 2 && c.c === 1);
      return {el: el.cells[${j}].text, run: el.cells[${j}].runs[0].text, bc: bc.text, line: b.lines[bc.lines[0]].text, conf: b.lines[bc.lines[0]].confidence, saved: document.querySelector('#r_saved').textContent, tsv: b.text.split('\\n')[2]}; })()`);
    check("cell edit -> document cell + runs", wb.el === "ម៉ម ចាន់ថា" && wb.run === "ម៉ម ចាន់ថា", wb);
    check("cell edit -> block cell + line (confidence 0.99) + block TSV", wb.bc === "ម៉ម ចាន់ថា" && wb.line === "ម៉ម ចាន់ថា" && wb.conf >= 0.99 && wb.tsv.split("\t")[1] === "ម៉ម ចាន់ថា", wb);
    check("saved indicator", /Saved just now/.test(wb.saved));
    const je = await evaluate(`${DOC}.result.document.elements.find(e => e.type === 'table').cells.findIndex(c => c.r === 3 && c.c === 4)`);
    await evaluate(`(() => { const td = document.querySelector('.dv-tablewrap[data-block="b3"] [data-cell="${je}"]'); td.innerText = '80'; td.dispatchEvent(new Event('blur')); return true; })()`); await sleep(200);
    check("empty cell filled -> stored", await evaluate(`${DOC}.result.blocks.find(x => x.id === 'b3').table.cells.find(c => c.r === 3 && c.c === 4).text === '80'`));
    check("edited low-confidence line (review list) -> cell follows; clicking the card focuses the cell", await evaluate(`(() => { const card = document.querySelector('.rev'); if (!card) return 'no card'; card.click(); const a = document.activeElement; return !!(a && a.dataset && a.dataset.cell !== undefined && a.closest('.dv-tablewrap')); })()`));
    await evaluate("(() => { const c = document.querySelector('.rev'); c.querySelector('input').value = '៩៧'; c.querySelector('[data-a=\"accept\"]').click(); return true; })()"); await sleep(300);
    check("Accept on a table line updates the cell text and clears low", await evaluate(`(() => { const d = ${DOC}; const el = d.result.document.elements.find(e => e.type === 'table'); const c = el.cells.find(x => x.r === 4 && x.c === 3); return c.text === '៩៧' && c.low === false && d.result.blocks.find(x => x.id === 'b3').table.cells.find(x => x.r === 4 && x.c === 3).text === '៩៧'; })()`));
    check("no low cell left after the accept", await count(".dv-table .dv-lowcell") === 0);

    // ---- 5. text pane (tables as small grids) and overlay pane (unchanged per-line editing)
    await evaluate("document.querySelector('#r_mode button[data-m=\"text\"]').click()"); await sleep(300);
    check("text pane shows the two tables as grids with header cells", await count(".textout table.tsv") === 2 && await count(".textout table.tsv th") === 13, [await count(".textout table.tsv"), await count(".textout table.tsv th")]);
    await shot("05_review_text.png", true);
    await evaluate("document.querySelector('#r_mode button[data-m=\"side\"]').click()"); await sleep(300);
    await evaluate("document.querySelector('#r_pane button[data-p=\"overlay\"]').click()"); await sleep(600);
    check("overlay pane: one editable line per recognised line (40)", await count(".docpage .ln") === 40, await count(".docpage .ln"));
    await shot("06_review_overlay.png");
    await evaluate("document.querySelector('#r_pane button[data-p=\"document\"]').click()"); await sleep(300);

    // ---- 6. process page with the fixture result (timing rows)
    await evaluate(`(() => { const d = ${DOC}; JOB = {id: 'dev', name: 'table_page.png', files: [{name: 'table_page.png', url: d.image_url, size: 0, status: 'done', page_id: d.page_id, error: null}], total: 1, done: 1, status: 'done', started: Date.now() - 1500, results: [d.result], current: 0}; location.hash = '#/process'; return true; })()`);
    await sleep(500);
    const steps = await evaluate("[...document.querySelectorAll('.pipe .step')].map(s => s.textContent.replace(/\\s+/g, ' ').trim())");
    check("process page: Table structure row with count and time", steps.some(s => /Table structure.*2 tables.*49 ms/.test(s)), steps);
    check("process page: Build the document names the kind", steps.some(s => /Build the document.*Spreadsheet \(you\)/.test(s)), steps);
    await shot("07_process.png");
    await evaluate(`(() => { const d = ${DOC}; d.result.timings.llm_ms = 2310.4; JOB.results = [d.result]; renderProcess(); return true; })()`); await sleep(200);
    const steps2 = await evaluate("[...document.querySelectorAll('.pipe .step')].map(s => s.textContent.replace(/\\s+/g, ' ').trim())");
    check("process page: Structure engine row when llm_ms is present", steps2.some(s => /Structure engine.*AI model \(LLM\).*2\.3 s/.test(s)), steps2);
    await shot("07b_process_llm.png");

    // ---- 7. settings (this backend) + help + documents list
    await go("#/settings"); await waitFor("!!document.querySelector('#s_out')");
    check("settings: default export offers Excel", await count("#s_out option[value=\"xlsx\"]") === 1);
    const engineText = await evaluate("[...document.querySelectorAll('.card')].map(c => c.textContent).find(t => /Structure engine/.test(t)) || ''");
    check("settings: structure engine card present", /Structure engine/.test(engineText));
    console.log("     engine card says: " + engineText.replace(/\s+/g, " ").slice(0, 160));
    await shot("08_settings.png", true);
    await go("#/help"); await sleep(200);
    check("help: Spreadsheets and tables section", /Spreadsheets and tables/.test(await text(".help")) && /\/export\/xlsx/.test(await text(".help")));
    await shot("09_help.png", true);
    await go("#/documents"); await sleep(300);
    check("documents list shows the kind", /Spreadsheet/.test(await text("#doclist")), await text("#doclist"));
    await evaluate("document.querySelector('#doclist .more').click()"); await sleep(200);
    const items2 = await evaluate("[...document.querySelectorAll('.menu button')].map(b => (b.dataset.a || '') + ':' + b.textContent.trim() + (b.disabled ? ':disabled' : ''))");
    check("documents menu: Word, Excel (default for a sheet page), CSV, text", items2.some(x => x.startsWith("docx:")) && items2.some(x => /^xlsx:.*default$/.test(x)) && items2.some(x => x.startsWith("csv:") && !/disabled/.test(x)) && items2.some(x => x.startsWith("txt:")), items2);
    await shot("10_documents_menu.png");

    // ---- 8. a v3 backend, stubbed at the fetch level (the app's own api.* functions and request bodies run for real):
    //         engine selector + LLM status + re-run with the AI model
    await evaluate(`(() => { window.__calls = []; const real = window.fetch.bind(window); const J = o => new Response(JSON.stringify(o), {headers: {'Content-Type': 'application/json'}});
      window.fetch = async (url, opts) => {
        if (url === '/pipeline/info') { const base = await (await real(url, opts)).json(); return J({...base, structure_engine: window.__engine || 'model', llm: {configured: true, provider: 'openai', model: 'qwen3-14b', reachable: !!window.__reach}}); }
        if (url === '/pipeline/settings') { const b = JSON.parse(opts.body); window.__calls.push('engine:' + b.structure_engine + ':' + opts.method); window.__engine = b.structure_engine; return J({}); }
        if (url === '/llm/structure') { const b = JSON.parse(opts.body); window.__calls.push('llm:' + (b.blocks ? b.blocks.length : 0) + ':' + (b.page && b.page.blocks ? 'page' : 'nopage') + ':' + opts.method);
          const r = JSON.parse(JSON.stringify(b)); delete r.page; r.doc_kind = {kind: 'document', confidence: 0.95, source: 'llm', tables: 2}; r.timings.llm_ms = 1800; return J(r); }
        return real(url, opts); };
      return true; })()`);
    await go("#/settings"); await waitFor("!!document.querySelector('#s_engine')");
    check("settings (v3 backend): engine selector with 3 options, Small model active", await count("#s_engine button") === 3 && /Small model/.test(await text("#s_engine button.on")));
    check("settings (v3 backend): LLM status not reachable + model name", /not reachable/.test(engineTextNow = await evaluate("[...document.querySelectorAll('.card')].map(c => c.textContent).find(t => /Structure engine/.test(t))")) && /qwen3-14b/.test(engineTextNow) && /via openai/.test(engineTextNow), engineTextNow);
    await shot("11_settings_v3.png", true);
    await evaluate("document.querySelector('#s_engine button[data-e=\"llm\"]').click()"); await waitFor("window.__calls.includes('engine:llm:POST') && document.querySelector('#s_engine button.on').dataset.e === 'llm'");
    check("switching the engine POSTs /pipeline/settings {structure_engine} and re-renders", await evaluate("JSON.stringify(window.__calls) === '[\"engine:llm:POST\"]' && document.querySelector('#s_engine button.on').dataset.e === 'llm'"), await evaluate("window.__calls"));
    await evaluate("window.__reach = true; INFO = null; true"); await go("#/settings"); await waitFor("/reachable/.test(document.body.textContent) && !/not reachable/.test(document.body.textContent)");
    check("settings: LLM reachable badge", await evaluate("[...document.querySelectorAll('.badge.ok')].some(b => /reachable/.test(b.textContent))"));
    await shot("11b_settings_v3_reachable.png", true);
    await go("#/review/" + FID); await waitFor("!!document.querySelector('#r_llm')");
    check("review: Re-run with AI model enabled when the LLM is reachable", await evaluate("document.querySelector('#r_llm').disabled === false"));
    await evaluate("document.querySelector('#r_llm').click()"); await waitFor("/AI model/.test(document.querySelector('#r_kind').textContent)", 5000);
    check("re-run replaces the stored result (kind from the LLM, source AI model)", /Document/.test(await text("#r_kind")) && /AI model/.test(await text("#r_kind")) && await evaluate(`${DOC}.result.doc_kind.source === 'llm' && ${DOC}.result.timings.llm_ms === 1800`), await text("#r_kind"));
    check("re-run POSTs /llm/structure with the page result (blocks at top level and under page)", await evaluate("window.__calls.some(c => c === 'llm:6:page:POST')"), await evaluate("window.__calls"));
    await shot("12_review_after_llm.png");

    // ---- 9. backwards compatibility on a real page (no doc_kind, no tables)
    if (REAL) {
      await nav(BASE + "?page_id=" + REAL);
      check("real page loads through ?page_id (reprocess on the backend)", await waitFor(`location.hash === '#/review/${REAL}' && !!document.querySelector('.dv-page')`, 120000), await evaluate("location.hash"));
      await sleep(1200);
      check("real page: no kind badge, no table rows, Word default", await count("#r_kind") === 0 && await count(".dv-tablewrap") === 0 && !/Tables/.test(await evaluate("[...document.querySelectorAll('.kv span')].map(s => s.textContent).join('|')")) && /Export Word/.test(await text("#r_exportdef")));
      check("real page: paragraphs editable as before", await count(".dv-page [contenteditable=\"true\"]") > 0);
      await shot("13_review_real_page.png", true);
      const cmp = await evaluate(`(async () => { const d = JSON.parse(localStorage.getItem('khocr.docs.v1')).find(x => x.page_id === '${REAL}'); const page = d.result, opts = {scale: 0.6, editable: true};
        const a = DocView.render(page, opts).outerHTML;
        await new Promise((res, rej) => { const s = document.createElement('script'); s.src = 'docview.js.pre_v3tables?x=' + Date.now(); s.onload = res; s.onerror = () => rej(new Error('old docview not served')); document.head.appendChild(s); });
        const b = DocView.render(page, opts).outerHTML; let i = 0; while (i < a.length && a[i] === b[i]) i++;
        return {same: a === b, la: a.length, lb: b.length, at: i, ctx: a.slice(Math.max(0, i - 60), i + 60)}; })()`);
      check("real page: new docview HTML == pre-v3 docview HTML (byte for byte)", cmp.same, cmp);
    } else console.log("     (no real page id given: backwards-compatibility render comparison skipped)");

    const errs = consoleErrors.filter(e => !/favicon|404|Failed to load resource/.test(e));
    check("no uncaught page errors", errs.length === 0, errs);
    ws.close();
  } finally { chrome.kill("SIGTERM"); await sleep(300); try { chrome.kill("SIGKILL"); } catch {} try { fs.rmSync(profile, {recursive: true, force: true}); } catch {} }
  console.log(`\n${n - fails}/${n} browser checks passed · screenshots in ${OUT}`);
  process.exit(fails ? 1 : 0);
}
let engineTextNow = "";
main().catch(e => { console.error("browser check crashed:", e); process.exit(2); });
