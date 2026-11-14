"""Export a processed page (the pipeline's JSON, possibly edited in the UI) as an editable Word document, an Excel
workbook or a CSV file (docs/TABLES_CONTRACT.md §3).

POST /export/docx  {"pages": [<page result>, ...], "title": "..."}  -> application/vnd...document
POST /export/xlsx  {"pages": [...], "title": "...", "mode": "auto" | "tables" | "sheet"}  -> .xlsx
POST /export/csv   {"pages": [...], "title": "...", "table_index": n?}  -> text/csv (UTF-8 with BOM, one table)

Word: each text block becomes paragraphs (the pipeline's ``paragraphs``: wrapped lines joined, Khmer without a space),
bold words come from the lines' ``spans``, typeface / size / colour / alignment / indent from the block; logo, stamp,
figure ... crops are inserted as pictures; page margins from ``layout.margins``; a ``table`` block with a cell grid
(v3 ``block.table``) becomes a real Word table (merged cells, header row bold, column widths, cell alignment, Khmer
font). Pixel sizes are converted assuming the page is A4 wide (points = px * 72 / (page_width / 8.27 in)).
Excel: ``auto`` puts a ``sheet`` page's tables on one worksheet (biggest first, a blank row between), every other
table on a worksheet of its own ``P{page}T{n}``; merged cells follow rs / cs, header rows are bold on a light fill,
numbers that parse (1,234.50, -12, 12%, $1,200, Khmer digits) are stored as numbers with a number format."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Inches, Pt, RGBColor
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pydantic import BaseModel, Field

from app.config import settings
from app.services.document import table_element

router = APIRouter(tags=["export"])
_HEX = re.compile(r"^#([0-9a-fA-F]{6})$")
_ID = re.compile(r"^[0-9a-f]{32}$")

# classifier family -> Word font name (the Khmer OS fonts are installed under exactly these names)
FONT_NAMES = {"Latin serif": "Times New Roman", "Latin sans": "Arial"}
CROP_CLASSES = {"figure", "logo", "stamp", "photo", "qr_code", "barcode", "signature", "decoration", "checkbox", "formula"}
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XLSX_MEDIA = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX_MEDIA = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
HEADER_FILL = "DDEBF7"
KHMER_FONT = "Khmer OS"


class ExportRequest(BaseModel):
    pages: list[dict] = Field(min_length=1, max_length=200)
    title: str = ""


class XlsxRequest(ExportRequest):
    mode: str = "auto"  # auto | tables | sheet


class CsvRequest(ExportRequest):
    table_index: int | None = None


def _px_to_pt(px: float, page_w: float) -> float:
    dpi = max(50.0, page_w / 8.27)
    return px * 72.0 / dpi


def _px_to_in(px: float, page_w: float) -> float:
    return px / max(50.0, page_w / 8.27)


def _rgb(hexcolor):
    m = _HEX.match(hexcolor or "")
    if not m:
        return None
    v = int(m.group(1), 16)
    return RGBColor((v >> 16) & 255, (v >> 8) & 255, v & 255)


def _khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿" or "᧠" <= ch <= "᧿"


def _has_khmer(text: str) -> bool:
    return any(_khmer(ch) for ch in text or "")


def _runs_for_lines(lines: list[dict]) -> list[tuple[str, bool]]:
    """Wrapped lines -> [(text, bold)] runs; a Khmer line break carries no space, a Latin one does."""
    runs: list[tuple[str, bool]] = []
    for ln in lines:
        spans = ln.get("spans") or []
        pieces = [(s["text"], bool(s.get("bold"))) for s in spans if s.get("text")] if spans else [(ln.get("text", ""), bool(ln.get("bold")))]
        if not pieces:
            continue
        if runs:
            prev, nxt = runs[-1][0], pieces[0][0]
            if prev and nxt and not (_khmer(prev[-1]) and _khmer(nxt[0])):
                runs.append((" ", False))
        for i, (t, b) in enumerate(pieces):
            if i and spans:
                runs.append((" ", False))
            runs.append((t, b))
    return runs


def _crop_path(page: dict, crop: dict) -> Path | None:
    page_id = page.get("page_id") or ""
    name = (crop or {}).get("file") or ""
    if not _ID.match(page_id) or not re.match(r"^[A-Za-z0-9_.-]+\.png$", name):
        return None
    p = Path(settings.crop_dir) / page_id / name
    return p if p.exists() else None


def _style_run(r, family: str, size_pt: float, bold: bool, italic: bool, colour) -> None:
    r.font.name = family
    r.font.size = Pt(size_pt)
    r.bold = bool(bold)
    r.italic = bool(italic)
    if colour is not None:
        r.font.color.rgb = colour
    rpr = r._element.get_or_add_rPr()
    rfonts = rpr.find(W_NS + "rFonts")
    if rfonts is not None:  # complex-script font too, so Khmer is not rendered in a fallback face
        rfonts.set(W_NS + "cs", family)


# ---------------------------------------------------------------------------------------------- tables (v3)

def _page_tables(page: dict) -> list[dict]:
    """The page's tables as ``table`` elements (document.table_element over the blocks, reading order), each with
    ``area`` (page px^2) for "the biggest table"."""
    out = []
    for b in sorted(page.get("blocks") or [], key=lambda b: b.get("reading_order", 0)):
        if (b.get("cls") or b.get("label")) != "table":
            continue
        el = table_element(b)
        if el is None or el["rows"] < 1 or el["cols"] < 1:
            continue
        box = b.get("box") or {}
        el["area"] = float(box.get("width") or 0) * float(box.get("height") or 0)
        out.append(el)
    return out


def _add_docx_table(doc, el: dict, usable_in: float, block_w_in: float, family: str, size_pt: float, colour, italic: bool) -> None:
    R, C = int(el["rows"]), int(el["cols"])
    if R < 1 or C < 1:
        return
    table = doc.add_table(rows=R, cols=C)
    if el.get("borders") != "none":
        table.style = "Table Grid"
    table.autofit = False
    width_in = max(1.0, min(usable_in, block_w_in or usable_in))
    cw = [max(0.3, f * width_in) for f in (el.get("col_widths") or [1.0 / C] * C)]
    for c in el["cells"]:
        r0, c0, rs, cs = c["r"], c["c"], c["rs"], c["cs"]
        if r0 >= R or c0 >= C:
            continue
        if rs > 1 or cs > 1:
            table.cell(r0, c0).merge(table.cell(min(R - 1, r0 + rs - 1), min(C - 1, c0 + cs - 1)))
    align_map = {"c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
    for c in el["cells"]:
        r0, c0 = c["r"], c["c"]
        if r0 >= R or c0 >= C:
            continue
        cell = table.cell(r0, c0)
        p = cell.paragraphs[0]
        p.alignment = align_map.get(c.get("align"), WD_ALIGN_PARAGRAPH.LEFT)
        runs = c.get("runs") or ([{"text": c.get("text", ""), "bold": c.get("bold")}] if c.get("text") else [])
        for rn in runs:
            if rn.get("text"):
                _style_run(p.add_run(rn["text"]), family, size_pt, bool(rn.get("bold")) or bool(c.get("header")), italic, colour)
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Inches(cw[min(i, C - 1)])


def build_docx(pages: list[dict], title: str = "") -> bytes:
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    if title:
        doc.core_properties.title = title
    for pi, page in enumerate(pages):
        W = float(page.get("width") or 1000)
        margins = ((page.get("layout") or {}).get("margins") or {})
        if pi == 0:
            sec.left_margin = Inches(max(0.4, min(1.5, _px_to_in(margins.get("left", 0.1 * W), W))))
            sec.right_margin = Inches(max(0.4, min(1.5, _px_to_in(margins.get("right", 0.1 * W), W))))
            sec.top_margin = Inches(max(0.4, min(1.5, _px_to_in(margins.get("top", 0.08 * W), W))))
            sec.bottom_margin = Inches(max(0.4, min(1.5, _px_to_in(margins.get("bottom", 0.08 * W), W))))
        else:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        usable_in = (sec.page_width - sec.left_margin - sec.right_margin) / 914400.0
        blocks = sorted(page.get("blocks") or [], key=lambda b: b.get("reading_order", 0))
        for b in blocks:
            cls = b.get("cls") or b.get("label") or "text"
            align = {"center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}.get(b.get("align"), WD_ALIGN_PARAGRAPH.LEFT)
            font = b.get("font") or {}
            family = FONT_NAMES.get(font.get("family"), font.get("family")) or "Khmer OS Battambang"
            size_pt = max(7.0, min(40.0, _px_to_pt(float(b.get("font_px") or 0.014 * W), W)))
            colour = _rgb(b.get("text_color"))
            if cls == "table":  # v3 tables: a real Word table from the cell grid
                el = table_element(b)
                if el is not None:
                    bw = float((b.get("box") or {}).get("width") or 0)
                    _add_docx_table(doc, el, usable_in, _px_to_in(bw, W) if bw else 0.0, family, size_pt, colour, font.get("slant") == "italic")
                    doc.add_paragraph().paragraph_format.space_after = Pt(2)
                    continue
            if cls in CROP_CLASSES or not b.get("lines"):
                if cls == "decoration":
                    continue
                path = _crop_path(page, b.get("crop"))
                if path is not None:
                    p = doc.add_paragraph()
                    p.alignment = align
                    bw = float((b.get("box") or {}).get("width") or 0.2 * W)
                    p.add_run().add_picture(str(path), width=Inches(max(0.3, min(7.0, _px_to_in(bw, W)))))
                continue
            lines = b.get("lines") or []
            paras = b.get("paragraphs") or [{"lines": [l.get("reading_order", i + 1) for i, l in enumerate(lines)], "indent_level": lines[0].get("indent_level", 0) if lines else 0, "bullet": lines[0].get("bullet") if lines else None}]
            by_order = {l.get("reading_order", i + 1): l for i, l in enumerate(lines)}
            for para in paras:
                plines = [by_order[i] for i in para.get("lines", []) if i in by_order]
                if not plines:
                    continue
                lf = plines[0].get("font") or {}
                if lf.get("family") and float(lf.get("confidence") or 0) >= 0.5:  # the line's own typeface when sure
                    family = FONT_NAMES.get(lf["family"], lf["family"])
                else:
                    family = FONT_NAMES.get(font.get("family"), font.get("family")) or "Khmer OS Battambang"
                p = doc.add_paragraph()
                p.alignment = align
                pf = p.paragraph_format
                pf.space_after = Pt(4)
                levels = [int(l.get("indent_level") or 0) for l in plines]
                first_indent = bool(para.get("first_line_indent")) or (len(plines) > 1 and levels[0] > min(levels[1:]))
                level = min(levels) if first_indent else int(para.get("indent_level") or 0)
                if level:
                    pf.left_indent = Inches(0.3 * level)
                if first_indent:
                    pf.first_line_indent = Inches(0.4)
                bullet = para.get("bullet")
                if bullet:
                    r = p.add_run(f"{bullet} ")
                    r.font.name, r.font.size = family, Pt(size_pt)
                runs = [(r["text"], bool(r.get("bold"))) for r in para.get("runs") or []] or _runs_for_lines(plines)
                for text, bold in runs:
                    _style_run(p.add_run(text), family, size_pt, bool(bold) or (font.get("weight") == "bold"), font.get("slant") == "italic", colour)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------- Excel / CSV (v3)

_KH_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
_NUM_RE = re.compile(r"^(?P<sign>[-+−])?\s*(?P<cur>[$€£¥₹៛])?\s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?P<cur2>[$€£¥₹៛])?\s*(?P<pct>%)?$")


def parse_number(text: str):
    """A cell text that is a number -> (value, number_format) else None. Thousands separators, a sign, a currency
    symbol, a percent sign and Khmer digits are understood; dates, phone numbers and mixed text stay text."""
    t = (text or "").strip().translate(_KH_DIGITS)
    neg = False
    if len(t) >= 3 and t[0] == "(" and t[-1] == ")":
        t, neg = t[1:-1].strip(), True
    m = _NUM_RE.match(t)
    if not m or (m.group("cur") and m.group("cur2")):
        return None
    num = m.group("num")
    value = float(num.replace(",", ""))
    if m.group("sign") in ("-", "−") or neg:
        value = -value
    decimals = len(num.split(".")[1]) if "." in num else 0
    thousands = "," in num
    cur = m.group("cur") or m.group("cur2")
    if m.group("pct"):
        return value / 100.0, "0%" if decimals == 0 else "0." + "0" * decimals + "%"
    fmt = ("#,##0" if thousands else "0") + ("." + "0" * decimals if decimals else "")
    if cur:
        fmt = f'"{cur}"{fmt}' if cur != "៛" else f'{fmt}"៛"'
    if decimals == 0 and abs(value) < 1e15:
        value = int(value)
    return value, fmt


def _sheet_name(base: str, used: set) -> str:
    name = re.sub(r"[\[\]:*?/\\]", "_", base or "Sheet").strip() or "Sheet"
    name = name[:31]
    cand, k = name, 2
    while cand.lower() in used:
        suffix = f"~{k}"
        cand = name[:31 - len(suffix)] + suffix
        k += 1
    used.add(cand.lower())
    return cand


def _set_text(cell, text: str) -> None:
    """Recognised text is data: a cell starting with "=" stays a string, never an Excel formula."""
    cell.value = text or None
    if text and text.startswith("="):
        cell.data_type = "s"


def _write_table(ws, el: dict, r0: int, widths: dict) -> int:
    """Write one table element at row ``r0`` (1-based) -> the last row written. ``widths`` collects the longest
    text per column letter for the column widths."""
    R, C = int(el["rows"]), int(el["cols"])
    header_fill = PatternFill("solid", fgColor=HEADER_FILL)
    h_align = {"l": "left", "c": "center", "r": "right"}
    for c in el["cells"]:
        r, k = r0 + int(c["r"]), 1 + int(c["c"])
        rs, cs = int(c.get("rs", 1)), int(c.get("cs", 1))
        text = c.get("text") or ""
        cell = ws.cell(row=r, column=k)
        num = parse_number(text) if text else None
        if num is not None:
            cell.value, cell.number_format = num
        else:
            _set_text(cell, text)
        bold = bool(c.get("bold")) or bool(c.get("header"))
        cell.font = Font(name=KHMER_FONT if _has_khmer(text) else None, size=11, bold=bold)
        horizontal = h_align.get(c.get("align")) if (num is None or c.get("align") == "c") else None
        cell.alignment = Alignment(horizontal=horizontal, vertical="top", wrap_text=num is None and len(text) > 40)
        if c.get("header"):
            cell.fill = header_fill
        if rs > 1 or cs > 1:
            ws.merge_cells(start_row=r, start_column=k, end_row=r + rs - 1, end_column=k + cs - 1)
        if cs == 1:
            col = get_column_letter(k)
            widths[col] = max(widths.get(col, 0), max((len(s) for s in text.splitlines()), default=0))
    return r0 + R - 1


def _page_paragraphs(page: dict) -> list[str]:
    """The page's text in reading order, one entry per paragraph (the blocks' ``paragraphs``; a block without them
    gives its lines joined, else its text). Picture blocks (no text) are skipped."""
    out = []
    for b in sorted(page.get("blocks") or [], key=lambda b: b.get("reading_order", 0)):
        paras = [p.get("text") or "" for p in (b.get("paragraphs") or [])]
        if not any(t.strip() for t in paras):
            lines = [l.get("text") or "" for l in (b.get("lines") or [])]
            paras = [paragraph_text_join(lines)] if any(t.strip() for t in lines) else [b.get("text") or ""]
        out += [t.strip() for t in paras if t and t.strip()]
    return out


def paragraph_text_join(texts: list[str]) -> str:
    """Lines of one block -> one string (Khmer lines join without a space, Latin with one)."""
    out = ""
    for t in (t.strip() for t in texts):
        if not t:
            continue
        out = t if not out else out + ("" if _has_khmer(out[-1:]) and _has_khmer(t[:1]) else " ") + t
    return out


def _write_text(ws, paragraphs: list[str]) -> None:
    for i, t in enumerate(paragraphs, start=1):
        cell = ws.cell(row=i, column=1)
        _set_text(cell, t)
        cell.font = Font(name=KHMER_FONT if _has_khmer(t) else None, size=11)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 100


def build_xlsx(pages: list[dict], title: str = "", mode: str = "auto") -> tuple[bytes, int]:
    """-> (xlsx bytes, number of tables written)."""
    mode = (mode or "auto").lower()
    if mode not in ("auto", "tables", "sheet"):
        raise ValueError("mode must be auto, tables or sheet")
    wb = Workbook()
    wb.remove(wb.active)
    used: set = set()
    n_tables = 0
    for pi, page in enumerate(pages, start=1):
        tabs = _page_tables(page)
        if not tabs:  # a page without tables: its text, one paragraph per row in column A
            ws = wb.create_sheet(_sheet_name(f"P{pi}" if len(pages) > 1 or not title else title, used))
            _write_text(ws, _page_paragraphs(page))
            continue
        kind = ((page.get("doc_kind") or {}).get("kind") or "").lower()
        as_sheet = mode == "sheet" or (mode == "auto" and kind == "sheet")
        if as_sheet:
            ws = wb.create_sheet(_sheet_name(f"P{pi}" if len(pages) > 1 or not title else title, used))
            widths: dict = {}
            row = 1
            biggest = max(tabs, key=lambda t: t["area"])
            for t in [biggest] + [t for t in tabs if t is not biggest]:
                row = _write_table(ws, t, row, widths) + 2  # one blank row between tables
                n_tables += 1
            for col, n in widths.items():
                ws.column_dimensions[col].width = max(8.0, min(60.0, n * 1.3 + 2))
        else:
            for n, t in enumerate(tabs, start=1):
                ws = wb.create_sheet(_sheet_name(f"P{pi}T{n}", used))
                widths = {}
                _write_table(ws, t, 1, widths)
                for col, m in widths.items():
                    ws.column_dimensions[col].width = max(8.0, min(60.0, m * 1.3 + 2))
                n_tables += 1
    if not wb.sheetnames:  # no pages at all (the request model requires one): keep the workbook valid
        wb.create_sheet("Sheet1")
    if title:
        wb.properties.title = title
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue(), n_tables


def build_csv(pages: list[dict], table_index: int | None = None) -> str:
    """One table (``table_index`` over all tables of all pages in order, else the biggest) as CSV text; spanned
    positions are empty cells. ValueError when there is no table / the index is out of range."""
    tabs = [t for page in pages for t in _page_tables(page)]
    if not tabs:
        raise ValueError("no table on the exported pages")
    if table_index is None:
        el = max(tabs, key=lambda t: t["area"])
    elif 0 <= table_index < len(tabs):
        el = tabs[table_index]
    else:
        raise ValueError(f"table_index out of range (the pages hold {len(tabs)} tables)")
    R, C = int(el["rows"]), int(el["cols"])
    grid = [["" for _ in range(C)] for _ in range(R)]
    for c in el["cells"]:
        if 0 <= c["r"] < R and 0 <= c["c"] < C:
            grid[c["r"]][c["c"]] = c.get("text") or ""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    for row in grid:
        w.writerow(row)
    return buf.getvalue()


def _filename(title: str, default: str, ext: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", title or default)[:60] or default
    return f"{name}.{ext}"


@router.post("/export/docx")
async def export_docx(req: ExportRequest):
    try:
        data = build_docx(req.pages, req.title)
    except Exception as exc:  # malformed client JSON
        raise HTTPException(status_code=422, detail=f"Cannot build the document: {type(exc).__name__}: {exc}") from exc
    return StreamingResponse(io.BytesIO(data), media_type=DOCX_MEDIA,
                             headers={"Content-Disposition": f'attachment; filename="{_filename(req.title, "document", "docx")}"'})


@router.post("/export/xlsx")
async def export_xlsx(req: XlsxRequest):
    try:
        data, n = build_xlsx(req.pages, req.title, req.mode)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot build the workbook: {type(exc).__name__}: {exc}") from exc
    return StreamingResponse(io.BytesIO(data), media_type=XLSX_MEDIA,
                             headers={"Content-Disposition": f'attachment; filename="{_filename(req.title, "tables", "xlsx")}"',
                                      "X-Tables": str(n)})


@router.post("/export/csv")
async def export_csv(req: CsvRequest):
    try:
        text = build_csv(req.pages, req.table_index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot build the CSV: {type(exc).__name__}: {exc}") from exc
    return Response(content=("﻿" + text).encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{_filename(req.title, "table", "csv")}"'})
