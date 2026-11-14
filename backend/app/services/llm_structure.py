"""LLM structure engine (contract §1 goal 2, `structure_engine = "llm"`): ONE call per page (or per band of a long
page) understands the whole page.

In:  the OCR lines of a process result (text + geometry + font hints), serialised compactly (``serialize_lines``):
     one line per text line ``L<id> b<block> [x0 y0 x1 y1] s<px> [B] [m:<marker>] | text`` with coordinates
     quantised to 1/1000 of the page and the text cut to a per-page character budget (Khmer costs ~1 token per
     character, so long pages get shorter texts). Pages over LLM_MAX_LINES lines (default 150) are cut into vertical
     bands at the biggest vertical gaps, one call per band; a table cut by a band edge is joined again.
Out: a JSON answer constrained by SCHEMA (strict objects, an anyOf of two element shapes; llama.cpp turns it into a
     grammar, Claude's structured output accepts it): document kind + elements in reading order. Text elements are
     {"t", "lv", "b", "l"}; tables are written compactly as rows of cells of line ids ({"t": "table", "hr": header
     rows, "rows": [[[ids], [ids], ...], ...], "sp": [[r, c, rs, cs], ...]}) because generation is the slow part on a
     laptop (~2-5 tokens/s for a 14B model partly on the CPU): ~3-4 tokens per cell instead of ~25.
     ``normalize_answer`` turns that into the internal cell form ({"t": "table", "c": [{r, c, rs, cs, h, l}]}, also
     accepted from the model), ``validate_and_repair`` makes it safe (ids exist, every line used once, unused lines
     appended as paragraphs, cells inside a consistent grid, spans >= 1, kind consistent with the tables).
``apply_llm_structure(res)`` rewrites a process result in place: text blocks are rebuilt from the elements (one block
per element, consecutive list items share one block), tables get ``block["table"]`` (contract §2.2, page pixels),
``doc_kind`` (source "llm"), reading order, ``layout`` and ``document`` (``build_document``) like the phone tier.
On any LLM failure (unreachable, timeout, page budget LLM_PAGE_BUDGET_S exceeded, malformed twice, page too long) the
result is returned unchanged with ``timings["llm_error"]`` set: the caller keeps the phone-tier result."""

from __future__ import annotations

import copy
import logging
import os
import re
import statistics
import time

from app.services.doc_structure import block_text, paragraph_text
from app.services.document import build_document, paragraph_runs
from app.services.llm_providers import LLMError, LLMProvider, LLMTruncated, get_provider, llm_config, parse_partial_json
from app.services.page_structure import page_layout
from app.taxonomy import CLS, COLORS, LABELS, TEXT_CLASSES, legacy_type

log = logging.getLogger("app.llm")


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


MAX_LINES_PER_CALL = int(_env_num("LLM_MAX_LINES", 150))      # longer pages go in vertical bands (one call each)
MAX_PAGE_LINES = int(_env_num("LLM_MAX_PAGE_LINES", 900))     # longer pages are not sent at all (phone tier stands)
PROMPT_CHAR_BUDGET = int(_env_num("LLM_PROMPT_CHARS", 7000))  # line texts of one call share this many characters
TEXT_MAX = 120             # characters of a line shown to the model at most
TEXT_MIN = 24              # ... and at least (cells / short lines are rarely cut)
BAND_OVERLAP = 0           # bands are cut at vertical gaps, no line is sent twice (kept for compatibility)
MAX_ROWS, MAX_COLS = 400, 60
KINDS = ["document", "sheet", "mixed"]
TYPES = ["heading", "para", "list", "header", "footer", "pagenum", "caption", "footnote", "field", "table"]
TEXT_TYPES = [t for t in TYPES if t != "table"]
TYPE_TO_CLS = {"para": "text", "list": "list", "header": "header", "footer": "footer", "pagenum": "page_number",
               "caption": "caption", "footnote": "footnote", "field": "form_field", "table": "table"}

_IDS = {"type": "array", "items": {"type": "integer"}}
TEXT_ELEMENT_SCHEMA = {"type": "object",
                       "properties": {"t": {"type": "string", "enum": TEXT_TYPES}, "lv": {"type": "integer"},
                                      "b": {"type": "string"}, "l": _IDS},
                       "required": ["t", "lv", "b", "l"], "additionalProperties": False}
TABLE_ELEMENT_SCHEMA = {"type": "object",
                        "properties": {"t": {"type": "string", "enum": ["table"]}, "hr": {"type": "integer"},
                                       "rows": {"type": "array", "items": {"type": "array", "items": {"anyOf": [{"type": "integer"}, _IDS]}}},
                                       "sp": {"type": "array", "items": _IDS}},
                        "required": ["t", "hr", "rows", "sp"], "additionalProperties": False}
SCHEMA = {"type": "object",
          "properties": {"kind": {"type": "string", "enum": KINDS},
                         "els": {"type": "array", "items": {"anyOf": [TEXT_ELEMENT_SCHEMA, TABLE_ELEMENT_SCHEMA]}}},
          "required": ["kind", "els"], "additionalProperties": False}

SYSTEM_PROMPT = """You are a document-structure engine for scanned Cambodian (Khmer) and English pages. You get the OCR output of ONE page (or one part of a long page): its text lines with geometry. Answer with ONE JSON object matching the schema, MINIFIED on one line (no line breaks, no indentation, no spaces between tokens), nothing else.

INPUT
First line: page size and line count. Then "blocks:" = the layout detector's boxes (b<n> class [x0 y0 x1 y1]; a hint: often wrong for headings; a "table" block is usually right, its lines then form ONE table element even when its first column is numbered; the detector misses some tables, so lines outside table blocks can still form one). Then one line per OCR text line:
  L<id> b<block> r<row> [x0 y0 x1 y1] s<size> [B] [m:<marker>] | <text>
id = line id; r = visual row: lines with the same r sit side by side on one horizontal band (in a table: one table row, unless a cell wraps onto several lines); x0 y0 x1 y1 = box in 1/1000 of the page width/height, origin top-left; s = font size in px; B = bold; m:<marker> = a list number or bullet that the OCR split off this line; text may be cut with "…".

OUTPUT
{"kind":"document"|"sheet"|"mixed","els":[element,...]} with the elements in READING ORDER.
Text element {"t":type,"lv":level,"b":marker,"l":[line ids]}, types:
 "heading" (lv 1 = document title, 2 = section heading, 3 = sub-heading), "para" (one paragraph = ONE element with all its wrapped lines), "list" (ONE list item per element; lv = depth 1, 2, ...; b = its marker "•" "-" "1." "១." "ក." "a)"), "header" (letterhead / running head: kingdom motto, ministry or company name, reference number, date at the top), "footer" (address, phone, website at the bottom), "pagenum", "caption", "footnote", "field" (form field "label: ____").
 lv = 0 and b = "" for every type other than list / heading (headings: b = "").
Table element {"t":"table","hr":header rows,"rows":[row,...],"sp":[[r,c,rs,cs],...]}:
 rows = the table rows top to bottom. A row lists its cells left to right, ONE entry per column, so every row has the same number of entries: a cell is its line id (12), or a list of ids when one cell holds several wrapped lines ([12,13]), or 0 when the cell is empty or covered by a span. Example row of 4 columns: [17,18,0,[19,20]].
 hr = number of column-header rows at the top (0 if none). sp = ONLY merged cells, i.e. one cell that stretches over several columns or rows (a group heading over two columns, a total label across the row); the positions it covers are 0 in rows. Each entry is [row, column, rows spanned, columns spanned] (0-based, of the merged cell's top-left position). An ordinary row is NOT a span: most tables have "sp":[].
Every line id of the input appears exactly once in the answer. Never invent ids.

RULES
- Khmer documents: the title is usually centred, larger or bold; "ព្រះរាជាណាចក្រកម្ពុជា", "ជាតិ សាសនា ព្រះមហាក្សត្រ", ministry / organisation names and a reference number or date line at the top are header; "ប្រការ ១" / "មាត្រា ២" clauses, "១." "២." "ក." "ខ." "1." "a)" numbered or bulleted lines are list items (b = the number) unless they clearly are section headings.
- Tables: lines that line up in rows (same y band) and columns (same x bands) form a table: price lists, attendance lists, timetables, ledgers, schedules, invoice item grids, form grids. Use the x positions to place every cell in its column (keep empty cells as []), the y positions for rows; several wrapped lines inside one cell go in the same cell. A title row running across the table is one cell with a span in sp. Numbered rows ("១" "២" "1" "2" in the first column) next to other aligned columns (several lines with the same r) are table rows, NOT list items. Text above / below the table (title, notes, totals written as sentences, signatures) are text elements, not rows.
- kind: "sheet" when the page is essentially one big table (a spreadsheet printout, list or form grid; only a title / header / footer around it); "mixed" when a document also contains a table; "document" when there is no table.
- Reading order: header first, then top to bottom; in a multi-column layout finish the left column before the right one; footer and page number last. Do not merge different paragraphs: a date / place line, a signature line, a name under a signature are separate elements.

EXAMPLE
{"kind":"mixed","els":[{"t":"header","lv":0,"b":"","l":[1,2]},{"t":"heading","lv":1,"b":"","l":[3]},{"t":"para","lv":0,"b":"","l":[4,5,6]},{"t":"list","lv":1,"b":"១.","l":[7,8]},{"t":"table","hr":1,"rows":[[9,10,11],[12,[13,14],0],[15,0,0]],"sp":[[2,0,1,3]]},{"t":"footer","lv":0,"b":"","l":[16]}]}"""

_KHMER_COMBINING = set(range(0x17B4, 0x17D4)) | {0x17DD}
_COENG = "្"


# ------------------------------------------------------------------------------------------------ serialisation
def truncate_text(text: str, n: int = TEXT_MAX) -> str:
    """Cut long texts but never inside a Khmer cluster (a consonant and the vowels / signs / coeng stack on it)."""
    if len(text) <= n:
        return text
    i = n
    while i < len(text) and (ord(text[i]) in _KHMER_COMBINING or text[i - 1] == _COENG):
        i += 1
    return text[:i] + "…"


def text_budget(n_lines: int) -> int:
    """Characters shown per line: the page's lines share PROMPT_CHAR_BUDGET (short pages keep TEXT_MAX)."""
    return int(max(TEXT_MIN, min(TEXT_MAX, PROMPT_CHAR_BUDGET // max(1, n_lines))))


def _line_box(ln: dict) -> list[float]:
    box = ln.get("box")
    if isinstance(box, dict):
        x0, y0 = float(box.get("x", 0)), float(box.get("y", 0))
        return [x0, y0, x0 + float(box.get("width", 0)), y0 + float(box.get("height", 0))]
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(v) for v in box]
    b = ln.get("bbox")
    return [float(v) for v in b] if b else [0.0, 0.0, 0.0, 0.0]


def _block_box(blk: dict) -> list[float]:
    return _line_box(blk)


def page_lines(res: dict) -> list[dict]:
    """Recognised text lines of a process result in block reading order -> neutral line dicts
    {id, bi, li, block, cls, box [x0 y0 x1 y1], font_px, bold, bullet, text, confidence}. Ids are 1-based."""
    blocks = res.get("blocks") or []
    order = sorted(range(len(blocks)), key=lambda i: (blocks[i].get("reading_order") or i + 1, i))
    out: list[dict] = []
    for bi in order:
        blk = blocks[bi]
        cls = blk.get("cls") or "text"
        if cls not in TEXT_CLASSES:
            continue
        for li, ln in enumerate(blk.get("lines") or []):
            text = (ln.get("text") or "").strip()
            if not text:
                continue
            x0, y0, x1, y1 = _line_box(ln)
            size = ln.get("font_px") or blk.get("font_px") or (y1 - y0)
            out.append({"id": len(out) + 1, "bi": bi, "li": li, "block": blk.get("reading_order") or bi + 1, "cls": cls,
                        "box": [x0, y0, x1, y1], "font_px": float(size or 0), "bold": bool(ln.get("bold")),
                        "bullet": ln.get("bullet") or "", "text": text, "confidence": float(ln.get("confidence", 1.0) or 0.0)})
    return out


def row_bands(lines: list[dict]) -> dict:
    """Line id -> visual row number (1-based, top to bottom). Each line is chained to its nearest right-hand
    neighbour that overlaps it vertically by >= 40 % of the smaller height (and best among the close ones); the chains
    are the rows. Chaining neighbours instead of cutting y bands keeps slightly rotated / curved rows together."""
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    boxes = [l["box"] for l in lines]
    for i, a in enumerate(boxes):
        ha = max(1.0, a[3] - a[1])
        best, best_key = None, None
        for j, b in enumerate(boxes):
            if j == i or b[0] < a[2] - 0.3 * ha:   # must start right of a (small overlap allowed)
                continue
            hb = max(1.0, b[3] - b[1])
            ov = min(a[3], b[3]) - max(a[1], b[1])
            if ov < 0.4 * min(ha, hb):
                continue
            key = (b[0] - a[2], -ov)
            if best_key is None or key < best_key:
                best, best_key = j, key
        if best is not None:
            parent[find(i)] = find(best)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups.values(), key=lambda g: (sum((boxes[i][1] + boxes[i][3]) / 2 for i in g) / len(g), min(boxes[i][0] for i in g)))
    out = {}
    for k, g in enumerate(order, start=1):
        for i in g:
            out[lines[i]["id"]] = k
    return out


def serialize_lines(lines: list[dict], W: int, H: int, blocks: list[dict] | None = None, part: tuple[int, int] | None = None) -> str:
    """The compact page text the model reads (see the module doc). ``blocks`` (process-result blocks) are listed as
    hints: class and box, text and non-text alike (only those that overlap the lines' vertical range).
    ``part`` = (k, n) when the page is sent in bands."""
    sx, sy = 1000.0 / max(1, W), 1000.0 / max(1, H)
    head = f"page {W}x{H}px, {len(lines)} lines"
    if part and part[1] > 1:
        head += f" (part {part[0]} of {part[1]} of a long page, top to bottom: answer for these lines only)"
    out = [head]
    if blocks:
        ylo = min((l["box"][1] for l in lines), default=0.0)
        yhi = max((l["box"][3] for l in lines), default=float(H))
        parts = []
        for k, blk in sorted(enumerate(blocks), key=lambda t: (t[1].get("reading_order") or t[0] + 1, t[0])):
            x0, y0, x1, y1 = _block_box(blk)
            if y1 < ylo or y0 > yhi:
                continue
            parts.append(f"b{blk.get('reading_order') or k + 1} {blk.get('cls') or 'text'} [{round(x0 * sx)} {round(y0 * sy)} {round(x1 * sx)} {round(y1 * sy)}]")
        if parts:
            out.append("blocks: " + "; ".join(parts))
    n = text_budget(len(lines))
    rb = row_bands(lines)
    out.append("lines:")
    for ln in lines:
        x0, y0, x1, y1 = ln["box"]
        s = f"L{ln['id']} b{ln['block']} r{rb[ln['id']]} [{round(x0 * sx)} {round(y0 * sy)} {round(x1 * sx)} {round(y1 * sy)}] s{round(ln['font_px'] or 0)}"
        if ln.get("bold"):
            s += " B"
        if ln.get("bullet"):
            s += f" m:{ln['bullet']}"
        out.append(s + " | " + truncate_text(ln["text"].replace("\n", " "), n))
    return "\n".join(out)


def serialize_page(res: dict) -> str:
    return serialize_lines(page_lines(res), int(res.get("width") or 0), int(res.get("height") or 0), res.get("blocks"))


def max_tokens_for(n_lines: int) -> int:
    """Answer budget: minified JSON needs ~10 tokens per text line (ids + element overhead) and ~4 per table cell."""
    return int(min(12000, 400 + 20 * n_lines))


# ------------------------------------------------------------------------------------------------ validation
class _Taker:
    """Hands out line ids once; counts bad / duplicate ids."""

    def __init__(self, valid: set, stats: dict):
        self.valid, self.stats, self.used = valid, stats, set()

    def take(self, seq) -> list[int]:
        out: list[int] = []
        for v in seq if isinstance(seq, (list, tuple)) else []:
            try:
                i = int(v)
            except (TypeError, ValueError):
                self.stats["bad_ids"] += 1
                continue
            if i not in self.valid:
                self.stats["bad_ids"] += 1
            elif i in self.used:
                self.stats["dup_ids"] += 1
            else:
                self.used.add(i)
                out.append(i)
        return out

    def release(self, ids) -> None:
        self.used.difference_update(ids)


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _ids(v) -> list:
    """A cell / element id field -> list of ids: [ids] stays, a bare id becomes [id], 0 / null (empty cell) -> []."""
    if isinstance(v, list):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [int(v)] if int(v) > 0 else []
    return []


def table_from_rows(el: dict) -> dict:
    """Compact table element ({"hr", "rows", "sp"}; a row = cells left to right, a cell = id | [ids] | 0) -> internal cell form {"t": "table", "lv": 0, "b": "", "l": [],
    "c": [{r, c, rs, cs, h, l}]}. Spans come from ``sp``; ids the model put in positions covered by a span join the
    span's cell; short rows are padded (missing positions are empty cells)."""
    rows = el.get("rows") if isinstance(el.get("rows"), list) else []
    hr = max(0, _int(el.get("hr"), 0))
    grid: dict = {}
    for r, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for c, cell in enumerate(row):
            grid[(r, c)] = _ids(cell)
    # the same ids written into neighbouring positions (a common way to show a merged cell) = a span
    rep_spans: dict = {}
    for (r, c), ids in sorted(grid.items()):
        if not ids or any((r, c) in v for v in rep_spans.values()):
            continue
        cs = 1
        while grid.get((r, c + cs)) == ids:
            cs += 1
        rs = 1
        while all(grid.get((r + rs, cc)) == ids for cc in range(c, c + cs)):
            rs += 1
        if rs > 1 or cs > 1:
            cov = {(rr, cc) for rr in range(r, r + rs) for cc in range(c, c + cs)} - {(r, c)}
            rep_spans[(r, c)] = cov
            for pos in cov:
                grid[pos] = []
            el = {**el, "sp": list(el.get("sp") or []) + [[r, c, rs, cs]]}
    spans = {}
    for s in el.get("sp") if isinstance(el.get("sp"), list) else []:
        if isinstance(s, list) and len(s) == 4:
            r, c, rs, cs = (_int(v, -1) for v in s)
            if r >= 0 and c >= 0 and (rs > 1 or cs > 1):
                rs, cs = max(1, rs), max(1, cs)
                # a real span covers positions the model left empty (0); a "span" over cells that hold their own
                # lines is a misunderstanding (seen: [r, 0, 1, C] for every row) and is ignored
                if any(grid.get((rr, cc)) for rr in range(r, r + rs) for cc in range(c, c + cs) if (rr, cc) != (r, c)):
                    continue
                spans[(r, c)] = (rs, cs)
    covered: dict = {}
    for (r, c), (rs, cs) in spans.items():
        for rr in range(r, r + rs):
            for cc in range(c, c + cs):
                if (rr, cc) != (r, c):
                    covered.setdefault((rr, cc), (r, c))
    cells: dict = {}
    for (r, c) in sorted(set(grid) | set(spans)):
        if (r, c) in covered and (r, c) not in spans:
            continue
        rs, cs = spans.get((r, c), (1, 1))
        cells[(r, c)] = {"r": r, "c": c, "rs": rs, "cs": cs, "h": r < hr, "l": list(grid.get((r, c), []))}
    for pos, origin in covered.items():
        ids = grid.get(pos)
        if ids and origin in cells:
            cells[origin]["l"].extend(ids)
    return {"t": "table", "lv": 0, "b": "", "l": _ids(el.get("l")), "c": [cells[k] for k in sorted(cells)]}


def normalize_answer(answer) -> dict:
    """Model answer -> internal form: every element has t / lv / b / l / c; compact tables become cell lists."""
    if not isinstance(answer, dict):
        return {"kind": None, "els": []}
    els = []
    raw = answer.get("els")
    for el in raw if isinstance(raw, list) else []:
        if not isinstance(el, dict):
            els.append(el)  # counted as dropped by the validation
            continue
        if el.get("t") == "table" and "rows" in el and not el.get("c"):
            els.append(table_from_rows(el))
        else:
            els.append({"t": el.get("t"), "lv": el.get("lv", 0), "b": el.get("b", ""), "l": _ids(el.get("l")),
                        "c": el.get("c") if isinstance(el.get("c"), list) else []})
    return {"kind": answer.get("kind"), "els": els}


def _clean_cells(raw, taker: _Taker, stats: dict) -> list[dict]:
    cells = []
    for cell in raw if isinstance(raw, list) else []:
        if not isinstance(cell, dict):
            stats["bad_cells"] += 1
            continue
        r, c = _int(cell.get("r"), -1), _int(cell.get("c"), -1)
        rs, cs = max(1, _int(cell.get("rs"), 1)), max(1, _int(cell.get("cs"), 1))
        if r < 0 or c < 0 or r >= MAX_ROWS or c >= MAX_COLS:
            stats["bad_cells"] += 1
            continue  # its lines stay unused and come back as paragraphs
        cells.append({"r": r, "c": c, "rs": min(rs, MAX_ROWS - r), "cs": min(cs, MAX_COLS - c), "h": bool(cell.get("h")),
                      "l": taker.take(cell.get("l"))})
    occupied: set = set()
    clean = []
    for cell in sorted(cells, key=lambda k: (k["r"], k["c"])):
        pos = {(rr, cc) for rr in range(cell["r"], cell["r"] + cell["rs"]) for cc in range(cell["c"], cell["c"] + cell["cs"])}
        if pos & occupied:
            if (cell["r"], cell["c"]) in occupied:  # the origin itself is taken: drop the cell, free its lines
                stats["bad_cells"] += 1
                taker.release(cell["l"])
                continue
            cell["rs"], cell["cs"] = 1, 1  # the span ran into another cell: shrink it
            stats["bad_cells"] += 1
            pos = {(cell["r"], cell["c"])}
        occupied |= pos
        clean.append(cell)
    return _trim_grid(clean)


def _trim_grid(cells: list[dict]) -> list[dict]:
    """Drop empty cells, then rows / columns that hold no text at all (the model padded a row or a column): the grid
    shrinks, spans over removed rows / columns shrink with it."""
    if not cells:
        return cells
    R, C = grid_size(cells)
    used_r = {rr for c in cells if c["l"] for rr in range(c["r"], c["r"] + c["rs"])}
    used_c = {cc for c in cells if c["l"] for cc in range(c["c"], c["c"] + c["cs"])}
    keep_r = [r for r in range(R) if r in used_r]
    keep_c = [c for c in range(C) if c in used_c]
    if len(keep_r) == R and len(keep_c) == C:
        return [c for c in cells if c["l"] or c["rs"] > 1 or c["cs"] > 1]
    rmap = {r: k for k, r in enumerate(keep_r)}
    cmap = {c: k for k, c in enumerate(keep_c)}
    out = []
    for cell in cells:
        if not cell["l"]:
            continue
        rows = [rmap[rr] for rr in range(cell["r"], cell["r"] + cell["rs"]) if rr in rmap]
        cols = [cmap[cc] for cc in range(cell["c"], cell["c"] + cell["cs"]) if cc in cmap]
        out.append({**cell, "r": rows[0], "c": cols[0], "rs": len(rows), "cs": len(cols)})
    return out


def grid_size(cells: list[dict]) -> tuple[int, int]:
    if not cells:
        return 0, 0
    return max(c["r"] + c["rs"] for c in cells), max(c["c"] + c["cs"] for c in cells)


def _group_unused(ids: list[int], boxes: dict | None) -> list[list[int]]:
    """Unused lines -> paragraphs: geometric order; vertically adjacent, horizontally overlapping lines share one."""
    if not ids:
        return []
    if not boxes:
        return [[i] for i in ids]
    ids = sorted(ids, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    groups: list[list[int]] = []
    for i in ids:
        b = boxes[i]
        if groups:
            a = boxes[groups[-1][-1]]
            h = max(1.0, min(a[3] - a[1], b[3] - b[1]))
            if b[1] - a[3] < 0.8 * h and min(a[2], b[2]) - max(a[0], b[0]) > 0 and b[1] >= a[1] - 0.5 * h:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def _insert_unused(els: list[dict], group: list[int], boxes: dict | None) -> None:
    """Put a paragraph of forgotten lines after the last element that starts above it (geometric reading position)."""
    para = {"t": "para", "lv": 0, "b": "", "l": group, "c": []}
    if not boxes:
        els.append(para)
        return
    y = boxes[group[0]][1]

    def top(e):
        ids = e["l"] or [i for c in e["c"] for i in c["l"]]
        return min((boxes[i][1] for i in ids if i in boxes), default=0.0)
    pos = len(els)
    for k in range(len(els) - 1, -1, -1):
        if els[k]["t"] in ("footer", "pagenum"):
            pos = k
            continue
        if top(els[k]) <= y:
            pos = k + 1
            break
        pos = k
    els.insert(pos, para)


def _bands_1d(intervals: list[tuple[float, float]], n_groups: int, min_gap: float) -> list[tuple[float, float]]:
    """Intervals (one per text line) -> separated bands: positions covered by at most 10 % of the groups (rows for
    columns) are gaps, gaps narrower than ``min_gap`` do not separate. A line spanning several bands (a merged header)
    therefore does not merge them."""
    if not intervals:
        return []
    lo, hi = min(a for a, _ in intervals), max(b for _, b in intervals)
    n = max(1, int(hi - lo) + 1)
    cov = [0] * (n + 1)
    for a, b in intervals:
        cov[int(a - lo)] += 1
        cov[int(b - lo)] -= 1
    run, prof = 0, []
    for k in range(n):
        run += cov[k]
        prof.append(run)
    limit = max(0, int(0.1 * n_groups))
    bands, start, gap_start = [], None, None
    for k, v in enumerate(prof + [0]):
        if v > limit:
            if start is None:
                start = k
            elif gap_start is not None and k - gap_start < min_gap:
                pass  # a narrow gap inside a band
            elif gap_start is not None:
                bands.append((lo + start, lo + gap_start))
                start = k
            gap_start = None
        elif start is not None and gap_start is None:
            gap_start = k
    if start is not None:
        bands.append((lo + start, lo + (gap_start if gap_start is not None else n)))
    return bands


def regrid_cells(cells: list[dict], boxes: dict) -> list[dict]:
    """Model cells -> the same rows with columns re-assigned from the geometry of the table's lines. The model is
    good at deciding which lines form the table and its rows, weak at column indices (rows listed without the empty
    cells, ragged rows); the x positions are reliable. Each model cell (its lines) is split per column band it covers
    when its lines fall into different bands, and spans the bands one wide line covers. Returns the model cells when
    the geometry finds fewer than 2 columns."""
    lines = [i for c in cells for i in c["l"] if i in boxes]
    if len(lines) < 4:
        return cells
    hs = sorted(boxes[i][3] - boxes[i][1] for i in lines)
    mh = max(1.0, hs[len(hs) // 2])
    n_rows = max(1, len({c["r"] for c in cells if c["l"]}))
    cols = _bands_1d([(boxes[i][0], boxes[i][2]) for i in lines], n_rows, max(2.0, 0.1 * mh))
    if len(cols) < 2:
        return cells
    # geometry that finds fewer columns than the model's typical row has merged columns (skewed photo, cells that
    # touch): keep the model's columns then
    lens = sorted(len({c["c"] for c in cells if c["r"] == r and c["l"]}) for r in {c["r"] for c in cells if c["l"]})
    if lens and len(cols) < lens[len(lens) // 2]:
        return cells

    def col_range(b):
        hit = [k for k, (a, z) in enumerate(cols) if min(b[2], z) - max(b[0], a) > 0.3 * min(b[2] - b[0], z - a)]
        if not hit:
            cx = (b[0] + b[2]) / 2
            hit = [min(range(len(cols)), key=lambda k: abs((cols[k][0] + cols[k][1]) / 2 - cx))]
        return hit[0], hit[-1]
    rows: dict = {}
    for cell in cells:
        if not cell["l"]:
            continue
        rows.setdefault(cell["r"], []).append(cell)
    out, r_new = [], 0
    for r in sorted(rows):
        groups: dict = {}
        for cell in rows[r]:
            for i in cell["l"]:
                if i not in boxes:
                    continue
                c0, c1 = col_range(boxes[i])
                g = groups.setdefault(c0, {"c1": c1, "l": [], "h": cell["h"], "rs": cell["rs"]})
                b = boxes[i]
                for j in g["l"]:  # two lines side by side in one "column": the bands merged real columns
                    o = boxes[j]
                    if min(b[3], o[3]) - max(b[1], o[1]) > 0.5 * min(b[3] - b[1], o[3] - o[1]):
                        return cells
                g["c1"] = max(g["c1"], c1)
                g["l"].append(i)
        if not groups:
            continue
        taken: set = set()
        for c0 in sorted(groups):
            g = groups[c0]
            if c0 in taken:  # covered by a wide line of an earlier group: join it
                owner = max(k for k in groups if k < c0 and groups[k]["c1"] >= c0)
                groups[owner]["l"] += g["l"]
                g["l"] = []
                continue
            taken.update(range(c0, g["c1"] + 1))
        for c0 in sorted(groups):
            g = groups[c0]
            if g["l"]:
                out.append({"r": r_new, "c": c0, "rs": 1, "cs": g["c1"] - c0 + 1, "h": g["h"],
                            "l": sorted(g["l"], key=lambda i: (boxes[i][1], boxes[i][0]))})
        r_new += 1
    # model row spans: a cell the model marked rs > 1 keeps it when nothing else sits below it in its columns
    return out


def _absorb_row_lines(els: list[dict], boxes: dict, stats: dict) -> list[dict]:
    """Short text elements (<= 2 lines) that sit inside a table row (vertical centre within the row's band, no
    overlap with that row's cells, within the table's vertical extent) are cells the model left out (seen: the row
    label column of a table written as paragraphs): they join the table, which is re-gridded."""
    tables = [e for e in els if e["t"] == "table"]
    if not tables:
        return els
    absorbed = set()
    for tab in tables:
        rows: dict = {}
        for c in tab["c"]:
            for i in c["l"]:
                b = boxes[i]
                lo, hi = rows.get(c["r"], (b[1], b[3]))
                rows[c["r"]] = (min(lo, b[1]), max(hi, b[3]))
        if len(rows) < 2:
            continue
        added = []
        for k, e in enumerate(els):
            if e["t"] in ("table", "footer", "pagenum", "header") or k in absorbed or not e["l"] or len(e["l"]) > 2:
                continue
            hit = None
            for i in e["l"]:
                b = boxes[i]
                cy = (b[1] + b[3]) / 2
                r = next((r for r, (lo, hi) in rows.items() if lo <= cy <= hi), None)
                if r is None or any(min(b[2], boxes[j][2]) - max(b[0], boxes[j][0]) > 0 for c in tab["c"] if c["r"] == r for j in c["l"]):
                    hit = None
                    break
                hit = r if hit in (None, r) else -1
            if hit is not None and hit >= 0:
                absorbed.add(k)
                added.append({"r": hit, "c": 999, "rs": 1, "cs": 1, "h": False, "l": list(e["l"])})
        if added:
            stats["absorbed"] = stats.get("absorbed", 0) + sum(len(a["l"]) for a in added)
            cells = regrid_cells(tab["c"] + added, boxes)
            tab["c"] = _trim_grid(cells) if cells is not tab["c"] + added else tab["c"] + added
    return [e for k, e in enumerate(els) if k not in absorbed]


def validate_and_repair(answer: dict, line_ids: list[int], boxes: dict | None = None) -> tuple[dict, dict]:
    """-> (clean answer in the internal form, repair counters). Every id exists and is used once; tables keep a
    consistent grid (>= 2 x 2, spans >= 1, no overlapping cells, no empty rows / columns) or become paragraphs; unused
    lines are inserted as paragraphs at their geometric place; the kind agrees with the tables that survived."""
    stats = {"bad_ids": 0, "dup_ids": 0, "unused": 0, "bad_cells": 0, "tables_dropped": 0, "elements_dropped": 0}
    answer = normalize_answer(answer)
    valid = {int(i) for i in line_ids}
    taker = _Taker(valid, stats)
    kind = answer.get("kind")
    if kind not in KINDS:
        kind = "document"
    els_out: list[dict] = []
    # tables claim their lines first: a model that lists the same lines as paragraphs AND as table cells (seen on
    # table-only crops) keeps the table
    pre_cells: dict = {}
    for k, el in enumerate(answer["els"]):
        if isinstance(el, dict) and el.get("t") == "table":
            cells = _clean_cells(el.get("c"), taker, stats)
            if boxes:
                cells = _trim_grid(_clean_cells(regrid_cells(cells, boxes), _Taker(valid, dict(stats)), dict(stats)))
            pre_cells[k] = cells
    for k, el in enumerate(answer["els"]):
        if not isinstance(el, dict):
            stats["elements_dropped"] += 1
            continue
        t = el.get("t") if el.get("t") in TYPES else "para"
        lv = _int(el.get("lv"), 0)
        marker = str(el.get("b") or "").strip()[:8]
        if t == "table":
            cells = pre_cells.get(k, [])
            stray = taker.take(el.get("l"))  # a table's own l should be empty: lines put there become paragraphs
            R, C = grid_size(cells)
            if R >= 2 and C >= 2 and sum(1 for c in cells if c["l"]) >= 2:
                els_out.append({"t": "table", "lv": 0, "b": "", "l": [], "c": cells})
                if stray:
                    taker.release(stray)  # come back as paragraphs
            else:
                stats["tables_dropped"] += 1
                for cell in sorted(cells, key=lambda k: (k["r"], k["c"])):
                    if cell["l"]:
                        els_out.append({"t": "para", "lv": 0, "b": "", "l": cell["l"], "c": []})
                if stray:
                    els_out.append({"t": "para", "lv": 0, "b": "", "l": stray, "c": []})
            continue
        ids = taker.take(el.get("l"))
        if not ids:
            stats["elements_dropped"] += 1
            continue
        if t == "heading":
            lv = min(3, lv) if lv >= 1 else 2
        elif t == "list":
            lv = min(4, lv) if lv >= 1 else 1
        else:
            lv, marker = 0, ""
        els_out.append({"t": t, "lv": lv, "b": marker if t == "list" else "", "l": ids, "c": []})
    if boxes:
        els_out = _absorb_row_lines(els_out, boxes, stats)
    unused = [i for i in sorted(valid) if i not in taker.used]
    stats["unused"] = len(unused)
    for group in _group_unused(unused, boxes):
        _insert_unused(els_out, group, boxes)
    has_table = any(e["t"] == "table" for e in els_out)
    if not has_table:
        kind = "document"
    else:
        # the kind must agree with how much of the page the tables hold (the model often says "mixed" for a sheet
        # with a title and a footer): >= 85 % of the lines in tables -> sheet, < 35 % -> mixed (docgen v5: sheets 0.89-1.0,
        # invoices / statements 0.7-0.8)
        in_tab = sum(len(c["l"]) for e in els_out if e["t"] == "table" for c in e["c"])
        share = in_tab / max(1, len(valid))
        if share >= 0.85:
            kind = "sheet"
        elif kind == "sheet" and share < 0.35 or kind == "document":
            kind = "mixed"
    return {"kind": kind, "els": els_out}, stats


# ------------------------------------------------------------------------------------------------ the call
def _bands(lines: list[dict], max_lines: int | None = None) -> list[list[dict]]:
    """Long pages -> vertical bands of at most ``max_lines`` lines, each cut at the biggest vertical gap in the last
    third of its range (so a band edge does not fall inside a paragraph or a table row when it can be avoided)."""
    m = max_lines or MAX_LINES_PER_CALL
    if len(lines) <= m:
        return [lines]
    by_y = sorted(lines, key=lambda l: ((l["box"][1] + l["box"][3]) / 2, l["box"][0]))
    out, start = [], 0
    while len(by_y) - start > m:
        lo, hi = start + max(1, (2 * m) // 3), start + m
        best, best_gap = hi, float("-inf")
        for k in range(lo, hi + 1):
            above = max(l["box"][3] for l in by_y[max(start, k - 8):k])
            gap = by_y[k]["box"][1] - above
            if gap > best_gap:
                best, best_gap = k, gap
        out.append(by_y[start:best])
        start = best
    out.append(by_y[start:])
    return out


def _merge_bands(answers: list[dict], bands: list[list[dict]]) -> dict:
    """Band answers (internal form) -> one answer. Ids outside a band are ignored (each band answers for its lines);
    a table that ends one band and a table that starts the next one with the same number of columns are one table
    (rows appended); the kind is "sheet" when the biggest band says so, else mixed / document from the tables."""
    assigned: set = set()
    els: list[dict] = []
    kinds = []
    for ans, band in zip(answers, bands):
        ans = normalize_answer(ans)
        if ans.get("kind") in KINDS:
            kinds.append((len(band), ans["kind"]))
        allowed = {l["id"] for l in band}
        first = True
        for el in ans["els"]:
            if not isinstance(el, dict):
                continue
            el = copy.deepcopy(el)

            def keep(seq):
                k = [i for i in _ids(seq) if isinstance(i, int) and i in allowed and i not in assigned]
                assigned.update(k)
                return k
            el["l"] = keep(el.get("l"))
            cells = [c for c in (el.get("c") or []) if isinstance(c, dict)]
            for cell in cells:
                cell["l"] = keep(cell.get("l"))
            el["c"] = cells
            if not el["l"] and not any(c["l"] for c in cells):
                continue
            if first and el.get("t") == "table" and els and els[-1].get("t") == "table":
                prev = els[-1]
                if grid_size(prev["c"])[1] == grid_size(cells)[1]:
                    r0 = min(c["r"] for c in cells if c["l"])
                    off = grid_size(prev["c"])[0] - r0
                    prev["c"] += [{**c, "r": c["r"] + off, "h": False} for c in cells if c["r"] >= r0]
                    first = False
                    continue
            first = False
            els.append(el)
    kind = max(kinds)[1] if kinds else "document"
    if kind != "sheet" and any(k == "sheet" for _, k in kinds):
        kind = "mixed"
    return {"kind": kind, "els": els}


_NUM_RE = re.compile(r"\d+")


def _loop_check(n_lines: int):
    """A streamed answer mentions each line id about once (plus a few numbers: levels, header rows, spans); far more
    numbers than that means the model is looping (seen: 1400+ tokens of repeated elements for a 57-line page)."""
    limit = 2 * n_lines + 60
    char_limit = 40 * n_lines + 400   # minified answers need ~10-20 characters per line

    def check(text: str) -> bool:
        return len(text) > char_limit or len(_NUM_RE.findall(text)) > limit
    return check


def _answer_ids(ans) -> set:
    """Line ids an (unvalidated) answer mentions, in elements and table cells of both forms."""
    out: set = set()
    for el in normalize_answer(ans)["els"]:
        if isinstance(el, dict):
            out.update(i for i in el.get("l") or [] if isinstance(i, int))
            for c in el.get("c") or []:
                if isinstance(c, dict):
                    out.update(i for i in _ids(c.get("l")) if isinstance(i, int))
    return out


def _with_progress(check, band_ids: set, done_before: int, total: int, progress):
    """The loop check, also reporting progress: the band's line ids the streamed answer already names (each line
    appears once in the answer, so this is the share of the page written so far)."""
    if progress is None:
        return check
    best = [0]

    def wrapped(text: str) -> bool:
        if text:
            seen = {int(n) for n in _NUM_RE.findall(text) if len(n) < 6} & band_ids
            best[0] = max(best[0], len(seen))
            progress("writing", done_before + best[0], total)
        return check(text)
    return wrapped


def structure_lines(lines: list[dict], W: int, H: int, provider: LLMProvider | None = None,
                    blocks: list[dict] | None = None, budget_s: float | None = None,
                    progress=None) -> tuple[dict, dict]:
    """Neutral lines -> (validated answer, meta {ms, tokens, model, provider, calls, bands, prompt_chars, repairs}).
    Raises LLMError when the provider cannot answer or the page budget (``budget_s``, default LLM_PAGE_BUDGET_S or
    2.5 x LLM_TIMEOUT_S) is used up before the last band. ``progress(stage, lines_done, lines_total)``: optional
    callback, stage "reading" (the model reads a band's prompt) / "writing" (it streams the answer)."""
    prov = provider or get_provider()
    if prov is None:
        raise LLMError("LLM engine not configured (LLM_PROVIDER)")
    if len(lines) > MAX_PAGE_LINES:
        raise LLMError(f"page has {len(lines)} lines (> LLM_MAX_PAGE_LINES={MAX_PAGE_LINES})")
    if budget_s is None:
        budget_s = _env_num("LLM_PAGE_BUDGET_S", 2.5 * float(llm_config()["timeout_s"]))
    t0 = time.perf_counter()
    bands = _bands(lines)
    answers, raws, calls, prompt_chars, salvaged = [], [], 0, 0, 0
    tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    server: dict = {}
    done_lines = 0
    for k, band in enumerate(bands, start=1):
        if progress is not None:
            progress("reading", done_lines, len(lines))
        if k > 1 and time.perf_counter() - t0 > budget_s:
            raise LLMError(f"page budget of {budget_s:.0f}s used up after {k - 1} of {len(bands)} bands")
        user = serialize_lines(band, W, H, blocks, part=(k, len(bands)))
        prompt_chars += len(SYSTEM_PROMPT) + len(user)
        budget = max_tokens_for(len(band))
        for attempt in range(2):
            try:
                calls += 1
                with prov.stop_when(_with_progress(_loop_check(len(band)), {l["id"] for l in band}, done_lines,
                                                   len(lines), progress)):
                    ans = prov.complete_json(SYSTEM_PROMPT, user, SCHEMA, budget)
                break
            except LLMTruncated as exc:
                for key in tokens:
                    tokens[key] += int(prov.last_usage.get(key, 0))
                # keep what was complete when it already places most lines (the validation adds the rest as
                # paragraphs); a second, twice as long call costs minutes on a laptop model
                part = parse_partial_json(getattr(exc, "partial", ""))
                placed = len(_answer_ids(part)) if part else 0
                if part and (placed >= 0.6 * len(band) or attempt or time.perf_counter() - t0 > budget_s):
                    salvaged += 1
                    ans = part
                    break
                if attempt or time.perf_counter() - t0 > budget_s:
                    raise
                budget *= 2
        for key in tokens:
            tokens[key] += int(prov.last_usage.get(key, 0))
        for key, v in (prov.last_usage.get("server_timings") or {}).items():
            if key.endswith("_ms"):
                server[key] = server.get(key, 0.0) + v
            else:
                server[key] = v  # the last call's rate
        answers.append(ans)
        raws.append(copy.deepcopy(ans))
        done_lines += len(band)
        if progress is not None:
            progress("writing", done_lines, len(lines))
    merged = _merge_bands(answers, bands) if len(bands) > 1 else answers[0]
    clean, stats = validate_and_repair(merged, [l["id"] for l in lines], {l["id"]: l["box"] for l in lines})
    meta = {"ms": round((time.perf_counter() - t0) * 1000, 1), "tokens": tokens, "model": prov.model, "provider": prov.name,
            "calls": calls, "bands": len(bands), "prompt_chars": prompt_chars, "repairs": stats, "lines": len(lines),
            "server_timings": server, "raw": raws, "salvaged": salvaged}
    return clean, meta


# ------------------------------------------------------------------------------------------------ tables
def _fill_extents(ext: list, lo: float, hi: float) -> list:
    """Unknown row / column extents are interpolated between the known ones (or spread evenly)."""
    known = [k for k, e in enumerate(ext) if e is not None]
    n = len(ext)
    if not known:
        step = (hi - lo) / max(1, n)
        return [[lo + k * step, lo + (k + 1) * step] for k in range(n)]
    out = list(ext)
    for k in range(n):
        if out[k] is not None:
            continue
        prev = max((j for j in known if j < k), default=None)
        nxt = min((j for j in known if j > k), default=None)
        if prev is not None and nxt is not None:
            a, b = ext[prev][1], ext[nxt][0]
            span = (b - a) / (nxt - prev)
            out[k] = [a + span * (k - prev - 1) + 0.1 * span, a + span * (k - prev) - 0.1 * span]
        elif prev is not None:
            size = ext[prev][1] - ext[prev][0]
            d = k - prev
            out[k] = [ext[prev][1] + size * 0.3 * d + size * (d - 1), ext[prev][1] + size * 0.3 * d + size * d]
        else:
            size = ext[nxt][1] - ext[nxt][0]
            d = nxt - k
            out[k] = [ext[nxt][0] - size * 0.3 * d - size * d, ext[nxt][0] - size * 0.3 * d - size * (d - 1)]
    return out


def _bands_from_extents(ext: list, pad: float) -> list[list[float]]:
    """Text extents per row / column -> contiguous bands cut at the mid-gaps (the table edge gets ``pad``)."""
    n = len(ext)
    edges = [ext[0][0] - pad]
    for k in range(1, n):
        a, b = ext[k - 1], ext[k]
        edges.append((a[1] + b[0]) / 2 if b[0] >= a[1] else ((a[0] + a[1]) / 2 + (b[0] + b[1]) / 2) / 2)
    edges.append(ext[-1][1] + pad)
    for k in range(1, len(edges)):  # monotonic
        edges[k] = max(edges[k], edges[k - 1] + 1.0)
    return [[round(edges[k], 1), round(edges[k + 1], 1)] for k in range(n)]


def _cell_align(box: list, x0: float, x1: float) -> str:
    w = max(1.0, x1 - x0)
    cx = (box[0] + box[2]) / 2
    if abs(cx - (x0 + x1) / 2) < 0.08 * w and box[0] - x0 > 0.05 * w:
        return "c"
    if x1 - box[2] < 0.1 * w and box[0] - x0 > 0.1 * w:
        return "r"
    return "l"


def table_structure(cells: list[dict], boxes: list, texts: list[str], bold: list | None = None, conf: list | None = None,
                    source: str = "llm") -> dict:
    """Validated cells ({r, c, rs, cs, h, idx: [indices into boxes / texts]}) -> contract §2.2 table dict in the
    coordinate space of ``boxes`` (page pixels). Row / column bands come from the text extents of the single-span
    cells (mid-gaps); empty grid positions become empty cells; spanned positions are not repeated."""
    R, C = grid_size(cells)
    ext: dict = {}
    for cell in cells:
        bx = [boxes[i] for i in cell["idx"]]
        if bx:
            ext[(cell["r"], cell["c"])] = [min(b[0] for b in bx), min(b[1] for b in bx), max(b[2] for b in bx), max(b[3] for b in bx)]
    hs = [boxes[i][3] - boxes[i][1] for cell in cells for i in cell["idx"]]
    mh = statistics.median(hs) if hs else 10.0
    allb = list(ext.values())
    if not allb:
        raise ValueError("table_structure needs at least one cell with lines")
    tx0, ty0 = min(b[0] for b in allb), min(b[1] for b in allb)
    tx1, ty1 = max(b[2] for b in allb), max(b[3] for b in allb)
    row_ext: list = [None] * R
    col_ext: list = [None] * C
    for cell in cells:
        e = ext.get((cell["r"], cell["c"]))
        if e is None:
            continue
        if cell["rs"] == 1:
            cur = row_ext[cell["r"]]
            row_ext[cell["r"]] = [min(e[1], cur[0]), max(e[3], cur[1])] if cur else [e[1], e[3]]
        if cell["cs"] == 1:
            cur = col_ext[cell["c"]]
            col_ext[cell["c"]] = [min(e[0], cur[0]), max(e[2], cur[1])] if cur else [e[0], e[2]]
    rows = _bands_from_extents(_fill_extents(row_ext, ty0, ty1), 0.3 * mh)
    cols = _bands_from_extents(_fill_extents(col_ext, tx0, tx1), 0.5 * mh)
    hr = 0
    while hr < R and any(c["h"] and c["r"] <= hr < c["r"] + c["rs"] for c in cells):
        hr += 1
    origin = {(c["r"], c["c"]): c for c in cells}
    covered = {(rr, cc) for c in cells for rr in range(c["r"], c["r"] + c["rs"]) for cc in range(c["c"], c["c"] + c["cs"])
               if (rr, cc) != (c["r"], c["c"])}
    out = []
    for r in range(R):
        for c in range(C):
            if (r, c) in covered:
                continue
            cell = origin.get((r, c))
            band = [cols[c][0], rows[r][0], cols[min(C - 1, c + (cell["cs"] if cell else 1) - 1)][1], rows[min(R - 1, r + (cell["rs"] if cell else 1) - 1)][1]]
            if cell is None or not cell["idx"]:
                out.append({"r": r, "c": c, "rs": cell["rs"] if cell else 1, "cs": cell["cs"] if cell else 1,
                            "box": [round(v, 1) for v in band], "header": r < hr, "text": "", "lines": [], "align": "l",
                            "bold": False, "confidence": 1.0})
                continue
            idx = list(cell["idx"])
            e = ext[(r, c)]
            out.append({"r": r, "c": c, "rs": cell["rs"], "cs": cell["cs"], "box": [round(v, 1) for v in e], "header": r < hr,
                        "text": paragraph_text([{"text": texts[i]} for i in idx]), "lines": idx,
                        "align": _cell_align(e, band[0], band[2]),
                        "bold": bool(bold) and all(bool(bold[i]) for i in idx),
                        "confidence": round(min(float(conf[i]) for i in idx), 4) if conf else 1.0})
    return {"rows": rows, "cols": cols, "header_rows": hr, "borders": "none", "cells": out, "source": source,
            "confidence": 0.8, "box": [round(tx0 - 0.3 * mh, 1), round(ty0 - 0.3 * mh, 1), round(tx1 + 0.3 * mh, 1), round(ty1 + 0.3 * mh, 1)]}


# ------------------------------------------------------------------------------------------------ apply
def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def _cls_of(el: dict, src_classes: set) -> str:
    if el["t"] == "heading":
        return "title" if el["lv"] <= 1 else "section_header"
    if el["t"] == "para" and src_classes and src_classes <= {"handwriting"}:
        return "handwriting"
    return TYPE_TO_CLS.get(el["t"], "text")


def _put_marker_back(ln: dict) -> None:
    """A marker the rules split off ("2." of a clause heading) belongs to the text when the element is no list item."""
    m = ln.get("bullet")
    if not m:
        return
    ln["text"] = f"{m} {ln['text']}"
    ln["spans"] = ([{"text": m, "bold": bool(ln.get("bold"))}] + list(ln["spans"])) if ln.get("spans") else []
    ln["bullet"], ln["bullet_box"] = None, None
    ln.pop("char_pos", None)


def _copy_line(src_blocks: list, ln_ref: dict) -> dict:
    ln = copy.deepcopy(src_blocks[ln_ref["bi"]]["lines"][ln_ref["li"]])
    ln["spans"] = list(ln.get("spans") or [])
    ln.setdefault("indent_level", 0)
    return ln


def _add_element_lines(blk: dict, el: dict, by_id: dict, src_blocks: list, owner: dict) -> list[dict]:
    pl = []
    lv = int(el.get("lv") or 0)
    for k, lid in enumerate(el["l"]):
        ref = by_id[lid]
        ln = _copy_line(src_blocks, ref)
        if el["t"] == "list":
            if k == 0:
                marker = el.get("b") or ln.get("bullet") or "•"
                if el.get("b") and ln["text"].startswith(el["b"]) and len(ln["text"]) > len(el["b"]) + 1:
                    ln["text"], ln["spans"] = ln["text"][len(el["b"]):].lstrip(), []
                    ln.pop("char_pos", None)
                ln["bullet"] = marker
            else:
                _put_marker_back(ln)
            ln["indent_level"], ln["tab"] = lv, lv > 0
        else:
            _put_marker_back(ln)
            if el["t"] in ("heading", "header", "footer", "pagenum"):
                ln["indent_level"], ln["tab"] = 0, False
        ln["reading_order"] = len(blk["lines"]) + 1
        owner[id(ln)] = ref["bi"]
        blk["lines"].append(ln)
        pl.append(ln)
    return pl


def _paragraph(pl: list[dict], is_list: bool, lv: int) -> dict:
    levels = [int(l.get("indent_level", 0)) for l in pl]
    first_indent = len(pl) > 1 and levels[0] > min(levels[1:])
    return {"indent_level": lv if is_list else (min(levels) if first_indent else levels[0]),
            "bullet": pl[0].get("bullet") if is_list else None, "text": paragraph_text(pl),
            "lines": [l["reading_order"] for l in pl], "runs": paragraph_runs(pl), "first_line_indent": bool(first_indent)}


def _new_block(cls: str) -> dict:
    return {"cls": cls, "lines": [], "paragraphs": [], "crop": None, "source": "llm", "recognized": True}


def _finish_block(blk: dict, src_blocks: list, owner: dict) -> dict:
    """Box, typeface, colour and the display fields from the source block that gave most of the lines."""
    lines = blk["lines"]
    votes: dict[int, int] = {}
    for ln in lines:
        votes[owner[id(ln)]] = votes.get(owner[id(ln)], 0) + 1
    src_i = max(votes.items(), key=lambda kv: kv[1])[0]
    src = src_blocks[src_i]
    boxes = [_line_box(ln) for ln in lines]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cls = blk["cls"]
    src_cls = {src_blocks[i].get("cls") for i in votes}
    blk.update({"type": legacy_type(cls), "label": cls, "cls_id": CLS.get(cls, -1), "display": LABELS.get(cls, cls),
                "color": _hex(COLORS.get(cls, (120, 120, 120))), "model_label": src.get("model_label") or src.get("cls"),
                "score": max(float(src_blocks[i].get("score") or 0.0) for i in votes), "source": "llm",
                "box": {"x": round(x0, 1), "y": round(y0, 1), "width": round(x1 - x0, 1), "height": round(y1 - y0, 1)},
                "text": block_text(lines), "bold": bool(lines) and all(l.get("bold") for l in lines if l.get("text")),
                "font_px": src.get("font_px"), "font_src": src.get("font_src"), "font": src.get("font"),
                "text_color": src.get("text_color"), "align": src.get("align", "left"),
                "relabel": (src.get("cls") if src_cls != {cls} else src.get("relabel"))})
    return blk


def _table_block(el: dict, by_id: dict, src_blocks: list, owner: dict) -> dict:
    blk = _new_block("table")
    cells = []
    for cell in sorted(el["c"], key=lambda k: (k["r"], k["c"])):
        idx = []
        for lid in cell["l"]:
            ref = by_id[lid]
            ln = _copy_line(src_blocks, ref)
            _put_marker_back(ln)
            ln["indent_level"], ln["tab"] = 0, False
            ln["reading_order"] = len(blk["lines"]) + 1
            owner[id(ln)] = ref["bi"]
            blk["lines"].append(ln)
            idx.append(len(blk["lines"]) - 1)
        cells.append({**cell, "idx": idx})
    lines = blk["lines"]
    blk["paragraphs"] = [_paragraph([ln], False, 0) for ln in lines]
    table = table_structure(cells, [_line_box(l) for l in lines], [l["text"] for l in lines],
                            [l.get("bold") for l in lines], [l.get("confidence", 1.0) for l in lines])
    tbox = table.pop("box")
    blk["table"] = table
    _finish_block(blk, src_blocks, owner)
    blk["box"] = {"x": round(tbox[0], 1), "y": round(tbox[1], 1), "width": round(tbox[2] - tbox[0], 1), "height": round(tbox[3] - tbox[1], 1)}
    return blk


def _area(blk: dict) -> float:
    b = blk.get("box") or {}
    return max(0.0, float(b.get("width", 0))) * max(0.0, float(b.get("height", 0)))


def rebuild_result(res: dict, lines: list[dict], answer: dict) -> dict:
    """Process result + validated answer -> blocks rebuilt from the elements, doc_kind, layout, document (in place)."""
    W, H = int(res["width"]), int(res["height"])
    src_blocks = res.get("blocks") or []
    by_id = {l["id"]: l for l in lines}
    consumed = {l["bi"] for l in lines}
    owner: dict[int, int] = {}
    new_blocks: list[dict] = []
    pending_list: dict | None = None
    for el in answer["els"]:
        if el["t"] == "table":
            new_blocks.append(_table_block(el, by_id, src_blocks, owner))
            pending_list = None
            continue
        if el["t"] == "list":
            if pending_list is None:
                pending_list = _new_block("list")
                new_blocks.append(pending_list)
            pl = _add_element_lines(pending_list, el, by_id, src_blocks, owner)
            pending_list["paragraphs"].append(_paragraph(pl, True, el["lv"]))
            continue
        pending_list = None
        blk = _new_block(_cls_of(el, {by_id[i]["cls"] for i in el["l"]}))
        pl = _add_element_lines(blk, el, by_id, src_blocks, owner)
        if el["t"] in ("header", "footer", "pagenum"):
            # letterhead / footer lines are printed lines, not wrapped text: one paragraph each keeps the line breaks
            # (build_document then stacks them in the letterhead row / footer columns like the phone tier)
            blk["paragraphs"] = [_paragraph([ln], False, 0) for ln in pl]
        else:
            blk["paragraphs"].append(_paragraph(pl, False, 0))
        new_blocks.append(blk)
    for blk in new_blocks:
        if "table" not in blk:
            _finish_block(blk, src_blocks, owner)
    # non-text blocks (figures, stamps, logos ...) and untouched blocks keep their place by geometry (order_blocks)
    kept = sorted((b for i, b in enumerate(src_blocks) if i not in consumed), key=lambda b: (b.get("box") or {}).get("y", 0))
    ordered: list[dict] = []
    placed: set = set()
    for blk in new_blocks:
        bottom = blk["box"]["y"] + blk["box"]["height"]
        for o in kept:
            ob = o.get("box") or {}
            if id(o) not in placed and ob.get("y", 0) + ob.get("height", 0) <= bottom:
                ordered.append(o)
                placed.add(id(o))
        ordered.append(blk)
    ordered += [o for o in kept if id(o) not in placed]
    for k, blk in enumerate(ordered, start=1):
        blk["id"], blk["reading_order"] = f"b{k}", k
    # layout (margins / zones / alignment) from the rebuilt blocks, like page_structure.page_layout in the pipeline
    internal = [{"cls": b.get("cls") or "text", "box": _block_box(b), "lines": [{"bbox": _line_box(l)} for l in (b.get("lines") or []) if l.get("text")]}
                for b in ordered]
    try:
        layout = page_layout(internal, W, H, TEXT_CLASSES)
        for b, ib in zip(ordered, internal):
            if b.get("source") == "llm":
                b["align"] = ib.get("align", "left")
    except Exception as exc:  # noqa: BLE001 - never lose the page over a layout statistic
        log.warning("page_layout failed on the LLM blocks: %r", exc)
        layout = res.get("layout")
    tables = [b for b in ordered if b.get("cls") == "table" and b.get("table")]
    page_area = max(1.0, float(W * H))
    table_area = sum(_area(b) for b in tables) / page_area
    text_area = sum(_area(b) for b in ordered if b.get("cls") in TEXT_CLASSES and b.get("cls") != "table" and b.get("lines")) / page_area
    rule_kind = "sheet" if table_area >= 0.45 and text_area < 0.12 else ("mixed" if tables else "document")
    res["blocks"] = ordered
    res["layout"] = layout
    res["doc_kind"] = {"kind": answer["kind"], "confidence": 0.95 if rule_kind == answer["kind"] else 0.7, "source": "llm",
                       "tables": len(tables), "table_area": round(table_area, 3), "text_area": round(text_area, 3)}
    res["document"] = build_document(ordered, layout, W, H)
    res["llm_structure"] = answer  # the validated answer (line ids = page_lines order of the INPUT result), for debugging
    return res


def sync_timeout_s() -> float:
    """Time a synchronous web request may spend on the LLM (LLM_SYNC_TIMEOUT_S, default 60 s)."""
    return _env_num("LLM_SYNC_TIMEOUT_S", 60.0)


def apply_llm_structure(res: dict, timeout_s: float | None = None, wait_if_busy: bool | None = None,
                        progress=None) -> dict:
    """Contract: rewrite ``res`` (a full process result) with the LLM's structure; on failure return it unchanged with
    ``timings["llm_error"]`` set so the caller can fall back to the phone tier.
    ``timeout_s``: total time for the page (all bands, queueing on the server included); None = LLM_PAGE_BUDGET_S /
    LLM_TIMEOUT_S (background jobs, the teacher). ``wait_if_busy``: queue behind a request already running on a
    single-slot server (default LLM_WAIT_IF_BUSY, off: fail at once so an interactive caller keeps the phone tier)."""
    timings = res.setdefault("timings", {})
    try:
        W, H = int(res.get("width") or 0), int(res.get("height") or 0)
    except (TypeError, ValueError):
        W, H = 0, 0
    if W <= 0 or H <= 0:
        timings["llm_error"] = "result has no page size"
        return res
    lines = page_lines(res)
    if len(lines) < 2:
        timings["llm_error"] = "fewer than 2 recognised lines"
        return res
    prov = get_provider()
    if prov is None:
        timings["llm_error"] = "LLM engine not configured (LLM_PROVIDER)"
        return res
    if not prov.reachable():  # cached probe: fail fast instead of waiting for a connect / read timeout
        timings["llm_error"], timings["llm_model"] = f"LLM endpoint not reachable ({prov.name} / {prov.model})", prov.model
        return res
    if wait_if_busy is None:
        wait_if_busy = os.environ.get("LLM_WAIT_IF_BUSY", "0") in ("1", "true", "yes")
    if not wait_if_busy and prov.busy():
        # a single-slot server busy with another request would queue this page for minutes: phone tier stands
        timings["llm_error"], timings["llm_model"] = f"LLM busy (another request is running on {prov.model})", prov.model
        return res
    try:
        with prov.deadline(timeout_s):
            answer, meta = structure_lines(lines, W, H, prov, res.get("blocks"), budget_s=timeout_s, progress=progress)
    except LLMError as exc:
        log.warning("llm structure failed: %s", exc)
        timings["llm_error"], timings["llm_model"] = str(exc), prov.model
        return res
    if progress is not None:
        progress("building", len(lines), len(lines))
    rebuild_result(res, lines, answer)
    timings.pop("llm_error", None)
    timings.update({"llm_ms": meta["ms"], "llm_tokens": meta["tokens"]["prompt_tokens"] + meta["tokens"]["completion_tokens"],
                    "llm_prompt_tokens": meta["tokens"]["prompt_tokens"], "llm_completion_tokens": meta["tokens"]["completion_tokens"],
                    "llm_model": meta["model"], "llm_provider": meta["provider"], "llm_calls": meta["calls"],
                    "llm_prompt_chars": meta["prompt_chars"], "llm_lines": meta["lines"], "llm_repairs": meta["repairs"]})
    if meta.get("server_timings"):
        timings["llm_server"] = meta["server_timings"]
    return res


# ------------------------------------------------------------------------------------------------ background jobs
class LLMJobs:
    """The LLM refinement as a background job, so no web request waits minutes for a slow laptop model: one worker
    thread (the local server has one slot anyway) takes jobs in order; finished jobs keep their result until
    ``keep`` newer jobs exist. Job = {"id", "status": queued | running | done | failed, "page_id", "created",
    "started", "finished", "elapsed_s", "position", "error", "result" (the rewritten page result, status done),
    "stage": queued | starting | reading | writing | building | done, "progress" 0..1 = "lines_done" / "lines_total"
    (the page's lines the streamed answer already places)}."""

    def __init__(self, keep: int = 50, runner=None):
        import queue
        import threading
        self.keep = keep
        self.runner = runner or (lambda res: apply_llm_structure(res, timeout_s=None, wait_if_busy=True,
                                                                 progress=self._progress))
        self._current = None
        self._jobs: dict = {}
        self._order: list = []
        self._q: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._thread = None
        self._threading = threading

    def submit(self, res: dict) -> dict:
        import uuid
        job = {"id": uuid.uuid4().hex[:12], "status": "queued", "page_id": res.get("page_id"), "created": time.time(),
               "started": None, "finished": None, "error": None, "result": None, "stage": "queued", "progress": 0.0,
               "lines_done": 0, "lines_total": len(page_lines(res))}
        with self._lock:
            self._jobs[job["id"]] = job
            self._order.append(job["id"])
            while len(self._order) > self.keep:
                old = self._order[0]
                if self._jobs[old]["status"] in ("queued", "running"):
                    break
                self._order.pop(0)
                self._jobs.pop(old, None)
            if self._thread is None or not self._thread.is_alive():
                self._thread = self._threading.Thread(target=self._work, name="llm-jobs", daemon=True)
                self._thread.start()
        self._q.put((job["id"], res))
        return self.public(job["id"])

    def _progress(self, stage: str, done: int, total: int) -> None:
        job = self._current
        if job is not None:
            job["stage"], job["lines_done"], job["lines_total"] = stage, int(done), int(total)
            job["progress"] = round(min(1.0, done / total), 3) if total else 0.0

    def _work(self) -> None:
        while True:
            jid, res = self._q.get()
            job = self._jobs.get(jid)
            if job is None:
                continue
            job["status"], job["started"], job["stage"] = "running", time.time(), "starting"
            self._current = job
            try:
                out = self.runner(res)
                err = (out.get("timings") or {}).get("llm_error") if isinstance(out, dict) else "no result"
                if err:
                    job["status"], job["error"] = "failed", err
                else:
                    out["structure_engine"] = "llm"
                    out.pop("llm_job", None)
                    job["status"], job["result"] = "done", out
            except Exception as exc:  # noqa: BLE001 - a job never kills the worker
                log.warning("llm job %s failed: %r", jid, exc)
                job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"[:300]
            self._current = None
            job["finished"] = time.time()
            if job["status"] == "done":
                job["progress"], job["stage"] = 1.0, "done"

    def public(self, jid: str, with_result: bool = True) -> dict | None:
        job = self._jobs.get(jid)
        if job is None:
            return None
        out = {k: v for k, v in job.items() if k != "result"}
        now = time.time()
        out["elapsed_s"] = round((job["finished"] or now) - (job["started"] or now), 1) if job["started"] else 0.0
        if job["status"] == "queued":
            with self._lock:
                ahead = [j for j in self._order if j != jid and self._jobs.get(j, {}).get("status") in ("queued", "running")
                         and self._jobs[j]["created"] <= job["created"]]
            out["position"] = len(ahead)
        if with_result and job["status"] == "done":
            out["result"] = job["result"]
        return out

    def list(self) -> list[dict]:
        return [self.public(j, with_result=False) for j in list(self._order)]


JOBS = LLMJobs()


def submit_llm_job(res: dict) -> dict:
    """Queue the LLM refinement of a (copied) process result -> the public job dict (contract §2.4, v3 llm jobs)."""
    return JOBS.submit(copy.deepcopy(res))
