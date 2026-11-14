/* ThaOCR — tables, document kind and export helpers (v3). Pure functions over the pipeline JSON described in
   docs/TABLES_CONTRACT.md §2 (doc_kind, blocks[].table, document[] "table" elements): no DOM, so app.js uses them in
   the browser (window.Tables) and dev/test_tables.js runs them under node (module.exports). */
(function (root) {
  "use strict";
  const KIND = {document: "Document", sheet: "Spreadsheet", mixed: "Mixed"};           // doc_kind.kind -> label
  const SOURCE = {rules: "rules", model: "small model", llm: "AI model", user: "you"};  // doc_kind.source -> label
  const ENGINE = {rules: "Rules", model: "Small model (phone)", llm: "AI model (LLM)"}; // structure_engine -> label
  const EXPORTS = [["docx", "Word (.docx)"], ["xlsx", "Excel (.xlsx)"], ["txt", "Plain text"], ["json", "JSON"]];
  const EXPORT_LABEL = {docx: "Word", xlsx: "Excel", txt: "text", json: "JSON", csv: "CSV"};

  /** The block a document element refers to: `block` is the block id ("b3") in the current backend, an index in the
      contract example; both are accepted. */
  function findBlock(page, ref) {
    const bs = (page && page.blocks) || [];
    if (ref == null) return null;
    return bs.find(b => b.id === ref) || (Number.isInteger(ref) ? bs[ref] : null) || bs.find(b => String(b.id) === String(ref)) || null;
  }
  /** document[] elements of type "table" in reading order (also inside row / footer_row columns). */
  function tableElements(page) {
    const out = [];
    const walk = els => { for (const e of els || []) { if (e.type === "table") out.push(e); else if (e.columns) for (const c of e.columns) walk(c.elements); } };
    walk(page && page.document && page.document.elements);
    return out;
  }
  /** [rows, cols] of a table (element: numbers; block table: band lists), grown to cover every cell span. */
  function dims(t) {
    let R = Array.isArray(t.rows) ? t.rows.length : (+t.rows || 0), C = Array.isArray(t.cols) ? t.cols.length : (+t.cols || 0);
    for (const c of t.cells || []) { R = Math.max(R, (c.r | 0) + Math.max(1, c.rs | 0)); C = Math.max(C, (c.c | 0) + Math.max(1, c.cs | 0)); }
    return [R, C];
  }
  /** R×C grid of cell texts: the text sits at the top-left position of a merged cell, spanned positions are "". */
  function tableGrid(t) {
    const [R, C] = dims(t), g = Array.from({length: R}, () => Array(C).fill(""));
    for (const c of t.cells || []) if (c.r >= 0 && c.r < R && c.c >= 0 && c.c < C) g[c.r][c.c] = String(c.text == null ? "" : c.text).replace(/[\t\r\n]+/g, " ").trim();
    return g;
  }
  /** Tab-separated rows (plain-text export / copy). */
  function tableTsv(t) { return tableGrid(t).map(r => r.join("\t")).join("\n"); }
  /** Index (into tableElements) of the biggest table = most grid positions; -1 without tables. */
  function biggestTable(page) {
    let best = -1, size = 0;
    tableElements(page).forEach((e, i) => { const [R, C] = dims(e); if (R * C > size) { size = R * C; best = i; } });
    return best;
  }
  /** Per-table summary for the review page: one entry per table block (falls back to the elements). */
  function tableStats(page) {
    const els = tableElements(page), out = [];
    els.forEach((e, i) => {
      const b = findBlock(page, e.block), bt = b && b.table, [R, C] = dims(bt || e);
      const conf = bt && bt.confidence != null ? bt.confidence : (e.confidence != null ? e.confidence : null);
      out.push({n: i + 1, rows: R, cols: C, confidence: conf, source: (bt && bt.source) || e.source || null, borders: (bt && bt.borders) || e.borders || null,
        low: (e.cells || []).filter(c => c.low).length, block: b ? b.id : e.block});
    });
    return out;
  }
  const kindLabel = k => KIND[k] || (k ? String(k) : "Document");
  const sourceLabel = s => SOURCE[s] || (s ? String(s) : "");
  const engineLabel = s => ENGINE[s] || (s ? String(s) : "");
  /** The export format a one-click export uses: Excel for a spreadsheet page, else the user's default. */
  function defaultExport(settings, page) {
    const k = page && page.doc_kind && page.doc_kind.kind;
    if (k === "sheet") return "xlsx";
    const o = settings && settings.output;
    return EXPORTS.some(([v]) => v === o) ? o : "docx";
  }
  /** The user overrides the kind: kept on the stored result with source "user" (the exports and the badge follow). */
  function setKind(page, kind) {
    if (!KIND[kind]) return page.doc_kind || null;
    page.doc_kind = Object.assign({}, page.doc_kind || {}, {kind, source: "user"});
    if (page.doc_kind.confidence == null) page.doc_kind.confidence = 1;
    return page.doc_kind;
  }
  /** A cell edited in the document view: writes the text to document[].cells[j] (text + runs, no longer low), to the
      matching blocks[].table cell (same r/c, else same index) and to that cell's recognised lines (first line gets
      the text, the rest are cleared, like a paragraph edit). Returns {cell, block, blockCell, lines} or null. */
  function applyCellEdit(page, el, j, text) {
    const cell = el && el.cells && el.cells[j];
    if (!cell) return null;
    text = String(text == null ? "" : text);
    cell.text = text; cell.runs = text ? [{text, bold: !!cell.bold}] : []; cell.low = false;
    const b = findBlock(page, el.block), lines = [];
    let bc = null;
    if (b && b.table && Array.isArray(b.table.cells)) {
      bc = b.table.cells.find(c => c.r === cell.r && c.c === cell.c) || b.table.cells[j] || null;
      if (bc) {
        bc.text = text; bc.confidence = Math.max(+bc.confidence || 0, 0.99);
        (bc.lines || []).forEach((li, i) => { const l = b.lines && b.lines[li]; if (!l) return; l.text = i ? "" : text; l.spans = []; l.confidence = Math.max(+l.confidence || 0, 0.99); lines.push(l); });
      }
      b.text = tableTsv(b.table);
    }
    return {cell, block: b, blockCell: bc, lines};
  }
  const KHMER = /[ក-៿]/, LOW = 0.8;
  /** Lines of one cell joined: a Khmer line break carries no space, a Latin one a space (contract §2.2). */
  function joinLines(texts) { let out = ""; for (const t of texts || []) { if (!t) continue; if (out && !(KHMER.test(out[out.length - 1]) && KHMER.test(t[0]))) out += " "; out += t; } return out; }
  /** A recognised line inside a table block was edited or accepted (review list): the block's cell and the document
      element's cell follow (text, runs, low, confidence = the lowest of the cell's lines). */
  function syncCell(page, b, l) {
    if (!b || !b.table || !Array.isArray(b.table.cells)) return null;
    const li = (b.lines || []).indexOf(l); if (li < 0) return null;
    const bc = b.table.cells.find(c => (c.lines || []).includes(li)); if (!bc) return null;
    const ls = bc.lines.map(i => b.lines[i] || {}), text = joinLines(ls.map(x => x.text));
    bc.text = text; bc.confidence = Math.min(1, ...ls.map(x => +x.confidence || 0));
    const el = tableElements(page).find(e => findBlock(page, e.block) === b), cell = el && (el.cells || []).find(c => c.r === bc.r && c.c === bc.c);
    if (cell) { cell.text = text; cell.runs = text ? [{text, bold: !!cell.bold}] : []; cell.low = ls.some(x => (+x.confidence || 0) < LOW); }
    b.text = tableTsv(b.table);
    return {blockCell: bc, cell: cell || null, text};
  }
  /** Timing rows of the process page for the v3 stages (null when the result has none). */
  function tableTiming(t) {
    if (!t || (t.table_ms == null && t.tables == null)) return null;
    const n = t.tables == null ? null : +t.tables;
    return {ms: t.table_ms == null ? null : +t.table_ms, text: n == null ? "" : `${n} table${n === 1 ? "" : "s"}`};
  }

  const Tables = {KIND, SOURCE, ENGINE, EXPORTS, EXPORT_LABEL, findBlock, tableElements, dims, tableGrid, tableTsv, biggestTable, tableStats,
                  kindLabel, sourceLabel, engineLabel, defaultExport, setKind, applyCellEdit, joinLines, syncCell, tableTiming};
  if (typeof module !== "undefined" && module.exports) module.exports = Tables;
  root.Tables = Tables;
})(typeof window !== "undefined" ? window : globalThis);
