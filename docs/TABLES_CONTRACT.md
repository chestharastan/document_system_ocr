# Tables, document kind and the two-tier structure engine — shared contract (v3, 2026-11-12)

Every agent on the v3 work reads this first. It fixes the JSON that the pipeline, the exports, the web app, the
LLM engine and the training labels exchange, so the parts can be built in parallel. Change it only through the lead.

## 1. Goals
1. A user photographs / uploads a page. The system decides whether it is a **document** (Word-like), a **sheet**
   (one big table: spreadsheet printout, price list, attendance list, timetable, ledger) or **mixed** (document with
   tables inside) and exports **.docx** (with real Word tables) or **.xlsx** (one worksheet per table / page) — the
   user can override the kind and the export format.
2. Two tiers of "structure understanding":
   - **phone tier** — small ONNX models: layout YOLO11n -> text-line DB -> CRNN -> font CNN ->
     **table-structure YOLO11n** (rows / columns / header / spanning cells) -> **structure transformer v3**
     (roles, paragraphs, reading order, doc kind). Every model < 10 MB, CPU-only inference.
   - **laptop / server tier** — one **LLM structure engine** (`engine = "llm"`): the OCR lines (text + geometry +
     font hints) go to an LLM (OpenAI-compatible endpoint such as the owner's local llama-server, or Claude) which
     returns the full document structure (kind, elements, tables with cells, order). It replaces rules + structure
     model + table-structure model + kind rules when it is available, and it is also the **teacher** that labels real
     pages for training the phone models.
3. Engine selection: `structure_engine` setting = `rules` | `model` (phone) | `llm`; `llm` falls back to `model`
   when the endpoint is unreachable, `model` falls back to `rules`. The response always says which engine produced
   the structure (`source` fields below).

## 2. Pipeline response additions (`POST /process`, `/reprocess`, `/process_lines`)
Existing keys stay unchanged: `width, height, blocks[], layout, document[], timings, models, device`.

### 2.1 `doc_kind` (page level, new)
```json
"doc_kind": {"kind": "document" | "sheet" | "mixed", "confidence": 0.0-1.0,
             "source": "rules" | "model" | "llm" | "user",
             "tables": 2, "table_area": 0.63, "text_area": 0.12}
```
`table_area` / `text_area` = share of the page covered by table / non-table text blocks (0-1).
Rules (phone tier, `backend/app/services/tables.py::decide_kind`): `sheet` when table_area >= 0.45 and the page's
non-table text is only header/footer/title/short lines (text_area < 0.12 or < 4 paragraphs); `mixed` when there is at
least one table with >= 2 rows and >= 2 columns and the page is not `sheet`; else `document`.

### 2.2 Block-level table structure (`blocks[i].table`, new, only when `cls == "table"`)
Coordinates are **page pixels** like every other box in the response (`[x0, y0, x1, y1]`).
```json
"table": {
  "rows": [[y0, y1], ...],            // row bands, top to bottom (len R)
  "cols": [[x0, x1], ...],            // column bands, left to right (len C)
  "header_rows": 1,                   // number of leading header rows (0 when unknown / none)
  "borders": "grid" | "hlines" | "none",
  "cells": [                          // one entry per *logical* cell, row-major; spanned positions are not repeated
    {"r": 0, "c": 0, "rs": 1, "cs": 2, "box": [x0, y0, x1, y1], "header": true,
     "text": "...", "lines": [12, 13], "align": "l" | "c" | "r", "bold": false, "confidence": 0.93}
  ],
  "source": "tsr_yolo11n_v1" | "rules" | "llm",
  "confidence": 0.9
}
```
`lines` = indices into the block's `lines[]` list (the recognised text lines that fell into the cell, reading order).
`text` = cell text already joined (Khmer line breaks join without a space, Latin with a space). Empty cells are present
with `text: ""` and `lines: []`. `rs`/`cs` are span counts (>= 1).

### 2.3 Document elements (`document[]`, new element type)
`build_document` emits one element per table instead of `table_row` items:
```json
{"type": "table", "rows": R, "cols": C, "header_rows": 1, "borders": "grid",
 "cells": [{"r":0,"c":0,"rs":1,"cs":1,"header":true,"align":"l","bold":true,"text":"...",
            "runs":[{"text":"...","bold":true}], "low": false}],
 "col_widths": [0.12, 0.48, 0.2, 0.2],   // fractions of the table width, sum 1
 "block": 3, "low": false, ...style keys (font_px, family, weight, color) like other elements}
```
`table_row` is **removed** (replaced by `table`). Elements keep reading order as today.

### 2.4 Settings / engine (`GET /pipeline/info`, `POST /pipeline/models`, `POST /pipeline/settings`)
`/pipeline/info` adds `"structure_engine": "rules"|"model"|"llm"`, `"llm": {"configured": bool, "provider": "openai"|"anthropic"|null,
"model": "...", "reachable": bool}`. `POST /pipeline/settings {"structure_engine": "llm"}` switches the engine for the
process. LLM configuration is server-side only (environment / config): `LLM_PROVIDER=openai|anthropic`,
`LLM_BASE_URL` (OpenAI-compatible, default `http://127.0.0.1:8082/v1`), `LLM_MODEL` (default `qwen3-14b`),
`LLM_API_KEY`, `LLM_TIMEOUT_S` (default 120). The UI never holds API keys.

**LLM jobs (added 2026-11-13).** The LLM never blocks a request. With `structure_engine=llm`, `POST /process` returns
the phone-tier result at once plus `"llm_job": {"id", "status": "queued"|"running"|"done"|"error", "poll": "/llm/jobs/<id>"}`;
one background worker refines pages in order. `GET /llm/jobs/<id>` returns the status and, when done, the refined page
result (same schema, `structure_engine: "llm"`); on failure the phone-tier result stays and `error` says why.
`POST /llm/structure {..., "async": true}` returns 202 + a job; without `async` it is synchronous and capped by
`LLM_SYNC_TIMEOUT_S` (default 60). `LLM_SYNC=1` restores the synchronous pipeline hook with that cap. The engine fails
fast when the server is unreachable, busy, or lists the model as unloaded (it never makes the server swap models).
Rule added by integration_qa: a page holding one table over 20-45 % of its area plus only a title is `sheet`, not `mixed`.

## 3. Exports
- `POST /export/docx` (exists) — `table` elements become real Word tables (merged cells, header row bold, column
  widths from `col_widths`, cell alignment, Khmer font).
- `POST /export/xlsx` (new) — body `{"pages": [<process result>...], "title": "...", "mode": "auto"|"tables"|"sheet"}`.
  `auto`: `sheet` pages -> one worksheet per page holding its biggest table (plus the other tables below, separated by
  a blank row); other pages -> one worksheet per table named `P{page}T{n}`. Merged cells via `rs/cs`, header rows bold
  with a light fill, numbers that parse (`1,234.50`, `-12`, `12%`, Khmer digits converted) stored as numbers with a
  number format, everything else as text; column widths from text length; font "Khmer OS" 11 for Khmer cells.
  Response: the .xlsx bytes (`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`).
- `POST /export/csv` (new) — same body, `table_index` optional; UTF-8 with BOM, one table (the biggest when not given).

## 4. Web app (backend/app/static_new)
- Review page: a **kind badge** (Document / Spreadsheet / Mixed, with the source) and a switch to override it
  (`doc_kind.source = "user"` on the stored result; it changes the default export).
- Tables render as editable grids inside the reflowed document (docview.js), cells `contenteditable`, edits write back
  to `document[].cells[].text` and the block's cell text; merged cells use `rowspan` / `colspan`.
- Export menu: Word (.docx), Excel (.xlsx), CSV (current / biggest table), plain text, JSON; the default follows
  the kind (`sheet` -> Excel). Settings: default export, structure engine selector (Rules / Small model (phone) /
  AI model (LLM)) showing whether the LLM is reachable. Process page: timing rows for "table structure" and the engine.

## 5. Training labels (docgen v5, `training/docgen`)
`manifest.jsonl` records gain a `tables` list (page pixel coordinates, already warped like `objs`):
```json
"tables": [{"box": [x0,y0,x1,y1], "rows": [[y0,y1],...], "cols": [[x0,x1],...], "header_rows": 1,
            "borders": "grid"|"hlines"|"none"|"zebra", "sheet": false,
            "cells": [{"r":0,"c":0,"rs":1,"cs":1,"box":[x0,y0,x1,y1],"header":true,"text":"...","align":"l",
                       "bold":true,"line_ids":[3,4]}]}]
```
`line_ids` index the record's `lines` / `line_texts` (printed lines inside the cell). Row / column bands are the
**ruling-line positions** for bordered tables and the mid-gaps between text for borderless ones. Multi-row cells
(spans) are labelled on the top-left position only. Every page also carries `doc_kind` (`document|sheet|mixed`).

YOLO table-structure (TSR) dataset (`training/table/make_tsr_dataset.py`): one image per table crop (crop = table
box padded by 2 % of its size, long side resized to 800), labels in YOLO format with classes
`0 row, 1 column, 2 column_header, 3 spanning_cell, 4 projected_row_header` (PubTables-1M formulation, so licensed
real sets convert 1:1). `row` boxes span the full table width, `column` boxes the full height, `column_header` is one
box over all header rows, `spanning_cell` one box per cell with rs > 1 or cs > 1.

## 6. Evaluation (`training/table/eval_tsr.py`)
Cell-level: adjacency-relation F1 (ICDAR 2013 style) and **TEDS-struct** (tree edit distance on the HTML structure,
without text) on held-out docgen v5 test tables and on every licensed real set; grid-level row/column mAP50 from
ultralytics. Report per set, phone-tier vs LLM-tier.

## 7. Ownership (do not edit files you do not own; ask the lead)
| area | owner |
|---|---|
| `training/docgen/*` tables + sheet doc types, `training/table/{make_tsr_dataset,eval_tsr}.py` | synthetic-table agent |
| `training/data/ext_tables_*.py`, `work/data/ext/tables_*`, `training/DATA_SOURCES.md` (append) | real-table-data agent |
| `backend/app/services/tables.py`, `stages/table_yolo_onnx.py`, `routers/export.py`, `services/document.py` (table part), `pipeline_service.py` (table + kind insertion, marked `# v3 tables`), `backend/tests/test_tables.py`, `test_export_xlsx.py` | backend-tables agent |
| `backend/app/services/llm_structure.py`, `llm_providers.py`, `routers/llm.py`, `training/structure/llm_label.py`, `backend/tests/test_llm_structure.py` | llm-engine agent |
| `backend/app/static_new/*` | frontend agent |
| `training/table/train_tsr.py`, `training/structure/*` (v3 heads), `.model/v2/models.json`, this file | lead |
