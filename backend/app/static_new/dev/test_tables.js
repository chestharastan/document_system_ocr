#!/usr/bin/env node
/* node backend/app/static_new/dev/test_tables.js
   Exercises the pure table functions of the web app without a browser or a backend: DocView.tableHtml (docview.js)
   and Tables.* (tables.js: cell write-back, line -> cell sync, default export, kind override, TSV, timings) on the
   dev fixture table_page.json plus a few hand-made cases. Exit code 1 when a check fails. */
"use strict";
const path = require("path"), fs = require("fs");
const DocView = require(path.join(__dirname, "..", "docview.js"));
const Tables = require(path.join(__dirname, "..", "tables.js"));
const FIXTURE = JSON.parse(fs.readFileSync(path.join(__dirname, "table_page.json"), "utf8"));
const clone = o => JSON.parse(JSON.stringify(o));
const count = (s, re) => (s.match(new RegExp(re, "g")) || []).length;
let fails = 0, n = 0;
const check = (name, ok, extra) => { n++; if (ok) console.log("ok   " + name); else { fails++; console.log("FAIL " + name + (extra == null ? "" : "  -> " + JSON.stringify(extra).slice(0, 300))); } };

// ---- fixture shape (contract §2.1 - §2.3)
const page = clone(FIXTURE);
const els = Tables.tableElements(page);
check("fixture has doc_kind", page.doc_kind && page.doc_kind.kind === "mixed" && page.doc_kind.source === "model");
check("fixture has two table elements", els.length === 2, els.length);
check("table elements point at table blocks with blocks[].table", els.every(e => { const b = Tables.findBlock(page, e.block); return b && b.cls === "table" && b.table && Array.isArray(b.table.cells); }));
check("biggest table is the first (6x5)", Tables.biggestTable(page) === 0);
check("findBlock by id and by index", Tables.findBlock(page, "b3") === page.blocks[2] && Tables.findBlock(page, 2) === page.blocks[2] && Tables.findBlock(page, "nope") === null);
check("dims of element and block table agree", JSON.stringify(Tables.dims(els[0])) === "[6,5]" && JSON.stringify(Tables.dims(Tables.findBlock(page, els[0].block).table)) === "[6,5]");

// ---- tableHtml: grid table with merged cells, header rows, empty / low / bold cells, widths, row heights
const t1 = els[0], b3 = Tables.findBlock(page, t1.block);
const html1 = DocView.tableHtml(t1, {blockTable: b3.table, scale: 1, title: "Table 1 · 91 %"});
check("grid class + title", /^<table class="dv-table dv-grid" title="Table 1 · 91 %">/.test(html1), html1.slice(0, 80));
check("6 rows", count(html1, "<tr") === 6, count(html1, "<tr"));
check("one cell node per element cell", count(html1, 'data-cell="') === t1.cells.length, [count(html1, 'data-cell="'), t1.cells.length]);
check("rowspan=2 x3, colspan=2 x1, colspan=3 x1", count(html1, 'rowspan="2"') === 3 && count(html1, 'colspan="2"') === 1 && count(html1, 'colspan="3"') === 1);
check("6 header cells are <th> (2 header rows)", count(html1, "<th ") === 6 && count(html1, "</th>") === 6, count(html1, "<th "));
check("col widths from col_widths", html1.includes('<colgroup><col style="width:8.00%"><col style="width:34.00%"><col style="width:14.00%"><col style="width:22.00%"><col style="width:22.00%"></colgroup>'));
check("row heights from the block's row bands (scale 1)", count(html1, 'style="height:44.00px"') === 2 && count(html1, 'style="height:48.00px"') === 4);
check("row heights follow the scale", count(DocView.tableHtml(t1, {blockTable: b3.table, scale: 0.5}), 'style="height:22.00px"') === 2);
const emptyIdx = t1.cells.findIndex(c => c.r === 3 && c.c === 4);
check("empty cell is present and empty", emptyIdx >= 0 && html1.includes(`<td class="dv-cell dv-ar" data-cell="${emptyIdx}"></td>`), emptyIdx);
const lowIdx = t1.cells.findIndex(c => c.low);
check("low cell carries dv-lowcell", lowIdx >= 0 && new RegExp(`<td class="dv-cell dv-lowcell[^"]*" data-cell="${lowIdx}"`).test(html1));
check("bold total row: dv-b + colspan=3 + right aligned", /<td class="dv-cell dv-ar dv-b" data-cell="\d+" colspan="3"><b>សរុប<\/b><\/td>/.test(html1));
check("bold runs render as <b>", count(html1, "<b>") >= 3);
check("header cell text present", html1.includes("ឈ្មោះសិស្ស"));

// ---- borderless table
const t2 = els[1], b5 = Tables.findBlock(page, t2.block);
const html2 = DocView.tableHtml(t2, {blockTable: b5.table, scale: 1});
check("borderless class", html2.startsWith('<table class="dv-table dv-noborder">'));
check("4 rows x 3 cols, 3 <th>, no spans", count(html2, "<tr") === 4 && count(html2, "<th ") === 3 && !/rowspan|colspan/.test(html2));
check("centre / right alignment classes (header cells are bold in the fixture)", html2.includes('class="dv-cell dv-th dv-ac dv-b"') && html2.includes('class="dv-cell dv-ar dv-b"') && html2.includes('class="dv-cell dv-ar" data-cell'));

// ---- tableHtml robustness
check("rows/cols grown from the cells when missing", count(DocView.tableHtml({cells: [{r: 2, c: 1, text: "x"}]}), "<tr") === 3);
check("header_rows alone makes <th>", count(DocView.tableHtml({rows: 2, cols: 1, header_rows: 1, cells: [{r: 0, c: 0, text: "h"}, {r: 1, c: 0, text: "d"}]}), "<th ") === 1);
check("cell text is escaped", DocView.tableHtml({rows: 1, cols: 1, cells: [{r: 0, c: 0, text: "<b>&"}]}).includes("&lt;b&gt;&amp;"));
check("no cells -> empty table, no crash", DocView.tableHtml({rows: 0, cols: 0, cells: []}) === '<table class="dv-table dv-grid"><tbody></tbody></table>');
check("col_widths ignored when the count does not match", !DocView.tableHtml({rows: 1, cols: 2, col_widths: [1], cells: []}).includes("<colgroup>"));
check("hlines class", DocView.tableHtml({rows: 1, cols: 1, borders: "hlines", cells: []}).includes("dv-hlines"));

// ---- cell edit write-back (document[].cells -> blocks[].table.cells -> lines)
{
  const pg = clone(FIXTURE), el = Tables.tableElements(pg)[0], j = el.cells.findIndex(c => c.r === 2 && c.c === 1);
  const r = Tables.applyCellEdit(pg, el, j, "ម៉ម ចាន់ថា");
  const b = Tables.findBlock(pg, el.block), bc = b.table.cells.find(c => c.r === 2 && c.c === 1);
  check("element cell text + runs + low", el.cells[j].text === "ម៉ម ចាន់ថា" && el.cells[j].runs.length === 1 && el.cells[j].runs[0].text === "ម៉ម ចាន់ថា" && el.cells[j].low === false);
  check("block cell text follows", bc.text === "ម៉ម ចាន់ថា" && bc.confidence >= 0.99);
  check("the cell's recognised line follows", r.lines.length === 1 && b.lines[bc.lines[0]].text === "ម៉ម ចាន់ថា" && b.lines[bc.lines[0]].confidence >= 0.99);
  check("block text (TSV) follows", b.text.split("\n")[2].split("\t")[1] === "ម៉ម ចាន់ថា");
  check("other cells untouched", el.cells.filter(c => c !== el.cells[j]).every((c, i) => c.text === FIXTURE.document.elements.find(e => e.type === "table").cells.filter((_, k) => k !== j)[i].text));
  const je = el.cells.findIndex(c => c.r === 3 && c.c === 4), r2 = Tables.applyCellEdit(pg, el, je, "80");
  check("empty cell gets text (no line to update)", el.cells[je].text === "80" && r2.blockCell.text === "80" && r2.lines.length === 0 && b.text.split("\n")[3].split("\t")[4] === "80");
  const r3 = Tables.applyCellEdit(pg, el, je, "");
  check("cleared cell -> no runs", el.cells[je].text === "" && el.cells[je].runs.length === 0 && r3.blockCell.text === "");
  check("bad cell index -> null", Tables.applyCellEdit(pg, el, 999, "x") === null && Tables.applyCellEdit(pg, null, 0, "x") === null);
  const lowCell = el.cells.find(c => c.low), jl = el.cells.indexOf(lowCell);
  Tables.applyCellEdit(pg, el, jl, "៩១");
  check("editing the low cell clears low", lowCell.low === false);
}
// ---- an edited / accepted line inside a table (review list) syncs the cell
{
  const pg = clone(FIXTURE), b = Tables.findBlock(pg, "b3"), bc = b.table.cells.find(c => c.r === 4 && c.c === 3), l = b.lines[bc.lines[0]];
  l.text = "៩៧"; l.confidence = 0.99;
  const r = Tables.syncCell(pg, b, l), cell = Tables.tableElements(pg)[0].cells.find(c => c.r === 4 && c.c === 3);
  check("syncCell: block cell + element cell + low cleared", r && bc.text === "៩៧" && cell.text === "៩៧" && cell.low === false && cell.runs[0].text === "៩៧" && bc.confidence >= 0.99);
  check("syncCell: TSV updated", b.text.split("\n")[4].split("\t")[3] === "៩៧");
  check("syncCell ignores lines outside a table", Tables.syncCell(pg, Tables.findBlock(pg, "b2"), Tables.findBlock(pg, "b2").lines[0]) === null);
  check("joinLines: Khmer without space, Latin with", Tables.joinLines(["កខ", "គឃ"]) === "កខគឃ" && Tables.joinLines(["ab", "cd"]) === "ab cd" && Tables.joinLines(["", "x", ""]) === "x");
}
// ---- TSV / grid
{
  const g = Tables.tableGrid(b3.table), tsv = Tables.tableTsv(b3.table);
  check("grid 6x5 with the merged text at the anchor only", g.length === 6 && g.every(r => r.length === 5) && g[0][3] === "ពិន្ទុ" && g[0][4] === "" && g[5][0] === "សរុប" && g[5][1] === "" && g[5][2] === "");
  check("tsv rows", tsv.split("\n").length === 6 && tsv.split("\n")[5] === "សរុប\t\t\t២៥៤\t១៨០");
  check("tsv from the element (rows/cols as numbers)", Tables.tableTsv(t1) === tsv);
  check("tabs / newlines inside a cell become spaces", Tables.tableTsv({rows: 1, cols: 1, cells: [{r: 0, c: 0, text: "a\tb\nc"}]}) === "a b c");
}
// ---- default export / kind override
{
  const s = {output: "docx"};
  check("default = settings.output for a mixed page", Tables.defaultExport(s, page) === "docx");
  check("sheet page -> xlsx whatever the setting", Tables.defaultExport({output: "txt"}, {doc_kind: {kind: "sheet"}}) === "xlsx");
  check("no doc_kind -> settings.output", Tables.defaultExport({output: "json"}, {blocks: []}) === "json" && Tables.defaultExport({output: "json"}, null) === "json");
  check("unknown setting -> docx", Tables.defaultExport({output: "bogus"}, page) === "docx" && Tables.defaultExport(null, null) === "docx");
  const pg = clone(FIXTURE), dk = Tables.setKind(pg, "sheet");
  check("setKind stores kind + source=user, keeps the rest", dk.kind === "sheet" && dk.source === "user" && dk.tables === 2 && pg.doc_kind === dk);
  check("setKind changes the default export", Tables.defaultExport(s, pg) === "xlsx");
  check("setKind ignores an unknown kind", Tables.setKind(pg, "weird") === dk && dk.kind === "sheet");
  const old = {blocks: []}; Tables.setKind(old, "document");
  check("setKind on a page without doc_kind creates it", old.doc_kind.kind === "document" && old.doc_kind.source === "user" && old.doc_kind.confidence === 1);
  check("labels", Tables.kindLabel("sheet") === "Spreadsheet" && Tables.sourceLabel("model") === "small model" && Tables.sourceLabel("user") === "you" && Tables.engineLabel("llm") === "AI model (LLM)" && Tables.kindLabel(undefined) === "Document");
  check("EXPORTS has Excel", Tables.EXPORTS.some(([k]) => k === "xlsx") && Tables.EXPORT_LABEL.xlsx === "Excel");
}
// ---- stats / timings
{
  const st = Tables.tableStats(page);
  check("tableStats: 2 tables with block confidence", st.length === 2 && st[0].rows === 6 && st[0].cols === 5 && st[0].confidence === 0.91 && st[1].confidence === 0.78 && st[0].low === 1 && st[0].source === "tsr_yolo11n_v1");
  check("tableTiming", JSON.stringify(Tables.tableTiming({table_ms: 48.7, tables: 2})) === '{"ms":48.7,"text":"2 tables"}' && Tables.tableTiming({}) === null && Tables.tableTiming(null) === null && Tables.tableTiming({tables: 1}).text === "1 table");
}
// ---- backwards compatibility: a page without doc_kind / tables
{
  const old = {width: 1000, height: 1400, blocks: [{id: "b1", cls: "text", lines: [{text: "hello", confidence: 0.9, reading_order: 1, box: {x: 0, y: 0, width: 10, height: 10}}]}],
               document: {margins: {left: 0, right: 0, top: 0, bottom: 0}, elements: [{type: "paragraph", block: "b1", lines: [1], text: "hello", runs: [{text: "hello"}]}]}};
  check("old page: no tables, no stats, default from settings", Tables.tableElements(old).length === 0 && Tables.tableStats(old).length === 0 && Tables.biggestTable(old) === -1 && Tables.defaultExport({output: "docx"}, old) === "docx" && Tables.tableTiming({total_ms: 5}) === null);
}

console.log(`\n${n - fails}/${n} checks passed`);
process.exit(fails ? 1 : 0);
