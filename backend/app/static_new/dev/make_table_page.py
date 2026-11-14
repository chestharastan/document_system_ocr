"""Dev fixture for the KhmerOCR web app: a mock pipeline result with doc_kind, two "table" document elements (one grid
table with merged cells, two header rows and an empty cell; one borderless table) and the matching blocks[].table,
following docs/TABLES_CONTRACT.md §2. Writes table_page.json and table_page.png (the drawn page, so the review page's
Original pane and the overlay have an image). Open it with /demo/?dev=table_page (no backend result needed).

    /home/hello/ocr/khm_repo_clone/det_venv/bin/python backend/app/static_new/dev/make_table_page.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = Path("/home/hello/ocr/work/fonts")
W, H = 1240, 1754
MARGINS = {"left": 100.0, "right": 100.0, "top": 90.0, "bottom": 90.0}
PAGE_ID = "0de00000000000000000000000007ab1"
INK = "#1a1a1a"


def font(name: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), px)


def khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿"


def join_lines(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if not t:
            continue
        if out and not (khmer(out[-1]) and khmer(t[0])):
            out += " "
        out += t
    return out


class Page:
    def __init__(self) -> None:
        self.img = Image.new("RGB", (W, H), "white")
        self.draw = ImageDraw.Draw(self.img)
        self.blocks: list[dict] = []
        self.elements: list[dict] = []

    # ---- text --------------------------------------------------------------------------------------------------
    def text_line(self, x: float, y: float, text: str, f: ImageFont.FreeTypeFont, bold=False, color=INK) -> dict:
        """Draw one line with its top-left at (x, y); return the line record with the measured box."""
        self.draw.text((x, y), text, font=f, fill=color)
        x0, y0, x1, y1 = self.draw.textbbox((x, y), text, font=f)
        ascent, _ = f.getmetrics()
        return {"box": {"x": round(x0 - 2, 1), "y": round(y0 - 2, 1), "width": round(x1 - x0 + 4, 1), "height": round(y1 - y0 + 4, 1)},
                "text": text, "confidence": 0.97, "score": 0.9, "reading_order": 0, "indent_level": 0, "indent": 0, "tab": False,
                "bullet": None, "bullet_box": None, "baseline": round(y + ascent, 1), "core_h": round(f.size * 0.55, 1),
                "font": {"family": "Khmer OS", "confidence": 0.9, "weight": "bold" if bold else "regular", "slant": "upright"},
                "bold": bold, "spans": [], "text_color": color}

    def text_block(self, bid: str, cls: str, lines: list[dict], px: int, align="left", el_type="paragraph", heading=None) -> None:
        for i, ln in enumerate(lines):
            ln["reading_order"] = i + 1
        xs0 = min(l["box"]["x"] for l in lines); ys0 = min(l["box"]["y"] for l in lines)
        xs1 = max(l["box"]["x"] + l["box"]["width"] for l in lines); ys1 = max(l["box"]["y"] + l["box"]["height"] for l in lines)
        box = {"x": round(xs0, 1), "y": round(ys0, 1), "width": round(xs1 - xs0, 1), "height": round(ys1 - ys0, 1)}
        style = {"font_px": px, "family": "Khmer OS", "weight": "bold" if lines[0]["bold"] else "regular", "slant": "upright", "heavy": False, "color": INK, "size_model": True}
        block = {"id": bid, "type": "paragraph", "label": cls, "cls": cls, "cls_id": 0, "display": cls.title(), "color": "#3c78d8", "model_label": cls,
                 "score": 0.95, "source": "layout", "reading_order": len(self.blocks) + 1, "box": box, "lines": lines, "crop": None,
                 "text": "\n".join(l["text"] for l in lines), "recognized": True, "font_px": px, "font_src": "model", "align": align, "relabel": None,
                 "font": {"family": "Khmer OS", "confidence": 0.9, "weight": style["weight"], "slant": "upright"}, "text_color": INK, "bold": lines[0]["bold"],
                 "paragraphs": [{"indent_level": 0, "bullet": None, "text": l["text"], "lines": [l["reading_order"]], "runs": [{"text": l["text"], "bold": l["bold"]}], "first_line_indent": False} for l in lines]}
        self.blocks.append(block)
        base = {"block": bid, "role": cls, "align": align, "style": style, "box": [box["x"], box["y"], box["width"], box["height"]]}
        for l in lines:
            el = {**base, "runs": [{"text": l["text"], "bold": l["bold"]}], "text": l["text"], "level": 0, "lines": [l["reading_order"]], "low": l["confidence"] < 0.8, "type": el_type}
            if el_type == "heading":
                el["heading"] = heading
            elif el_type == "paragraph":
                el["first_line_indent"] = False
            self.elements.append(el)

    # ---- tables ------------------------------------------------------------------------------------------------
    def table_block(self, bid: str, x0: float, y0: float, col_widths: list[float], width: float, row_h: list[float], header_rows: int,
                    borders: str, spec: list[dict], px: int, confidence: float) -> None:
        """spec: [{r, c, rs, cs, text, align, bold, low}] (one per logical cell; spanned positions absent)."""
        xs = [x0]
        for f in col_widths:
            xs.append(xs[-1] + f * width)
        ys = [y0]
        for h in row_h:
            ys.append(ys[-1] + h)
        R, C = len(row_h), len(col_widths)
        rows = [[round(ys[i], 1), round(ys[i + 1], 1)] for i in range(R)]
        cols = [[round(xs[i], 1), round(xs[i + 1], 1)] for i in range(C)]
        # header fill, then rulings
        if borders != "none" and header_rows:
            self.draw.rectangle([xs[0], ys[0], xs[-1], ys[header_rows]], fill="#f0ede6")
        if borders == "grid":
            for x in xs:
                self.draw.line([(x, ys[0]), (x, ys[-1])], fill="#333333", width=2)
            for y in ys:
                self.draw.line([(xs[0], y), (xs[-1], y)], fill="#333333", width=2)
        elif borders == "hlines":
            for y in ys:
                self.draw.line([(xs[0], y), (xs[-1], y)], fill="#333333", width=2)
        # merged positions get no ruling: paint them over
        reg, bold = font("KhmerOS.ttf", px), font("Battambang-Bold.ttf", px)
        lines: list[dict] = []
        bcells: list[dict] = []
        ecells: list[dict] = []
        for cell in spec:
            r, c, rs, cs = cell["r"], cell["c"], cell.get("rs", 1), cell.get("cs", 1)
            cx0, cx1, cy0, cy1 = xs[c], xs[c + cs], ys[r], ys[r + rs]
            if borders == "grid" and (rs > 1 or cs > 1):  # erase inner rulings of a merged cell
                self.draw.rectangle([cx0 + 2, cy0 + 2, cx1 - 2, cy1 - 2], fill="#f0ede6" if r < header_rows else "white")
            head = r < header_rows
            is_bold = bool(cell.get("bold")) or head
            f = bold if is_bold else reg
            text = cell.get("text", "")
            align = cell.get("align", "l")
            line_ids: list[int] = []
            if text:
                tx0, ty0, tx1, ty1 = self.draw.textbbox((0, 0), text, font=f)
                tw, th = tx1 - tx0, ty1 - ty0
                pad = 10
                x = cx0 + pad if align == "l" else (cx1 - pad - tw if align == "r" else (cx0 + cx1 - tw) / 2)
                y = (cy0 + cy1) / 2 - th / 2 - ty0
                ln = self.text_line(x, y, text, f, bold=is_bold)
                ln["confidence"] = cell.get("conf", 0.6 if cell.get("low") else 0.97)
                ln["reading_order"] = len(lines) + 1
                line_ids = [len(lines)]
                lines.append(ln)
            low = bool(cell.get("low")) or (bool(text) and lines[-1]["confidence"] < 0.8)
            cconf = cell.get("conf", 0.6 if low else 0.95)
            bcells.append({"r": r, "c": c, "rs": rs, "cs": cs, "box": [round(cx0, 1), round(cy0, 1), round(cx1, 1), round(cy1, 1)], "header": head,
                           "text": text, "lines": line_ids, "align": align, "bold": is_bold, "confidence": cconf})
            ecells.append({"r": r, "c": c, "rs": rs, "cs": cs, "header": head, "align": align, "bold": is_bold, "text": text,
                           "runs": [{"text": text, "bold": is_bold}] if text else [], "low": low})
        box = {"x": round(xs[0], 1), "y": round(ys[0], 1), "width": round(xs[-1] - xs[0], 1), "height": round(ys[-1] - ys[0], 1)}
        table = {"rows": rows, "cols": cols, "header_rows": header_rows, "borders": borders, "cells": bcells, "source": "tsr_yolo11n_v1", "confidence": confidence}
        grid = [[""] * C for _ in range(R)]
        for cell in bcells:
            grid[cell["r"]][cell["c"]] = cell["text"]
        tsv = "\n".join("\t".join(row) for row in grid)
        style = {"font_px": px, "family": "Khmer OS", "weight": "regular", "slant": "upright", "heavy": False, "color": INK, "size_model": True}
        self.blocks.append({"id": bid, "type": "table", "label": "table", "cls": "table", "cls_id": 4, "display": "Table", "color": "#8e44ad", "model_label": "table",
                            "score": 0.93, "source": "layout", "reading_order": len(self.blocks) + 1, "box": box, "lines": lines, "crop": None, "text": tsv,
                            "recognized": True, "font_px": px, "font_src": "model", "align": "left", "relabel": None,
                            "font": {"family": "Khmer OS", "confidence": 0.88, "weight": "regular", "slant": "upright"}, "text_color": INK, "bold": False,
                            "paragraphs": [], "table": table})
        self.elements.append({"block": bid, "role": "table", "align": "left", "style": style, "box": [box["x"], box["y"], box["width"], box["height"]],
                              "type": "table", "rows": R, "cols": C, "header_rows": header_rows, "borders": borders, "cells": ecells,
                              "col_widths": [round(f, 4) for f in col_widths], "low": any(c["low"] for c in ecells)})


def main() -> None:
    p = Page()
    title_f, body_f, small_f = font("KhmerOSmuollight.ttf", 40), font("KhmerOS.ttf", 24), font("KhmerOS.ttf", 20)
    # title (centred)
    t = "បញ្ជីវត្តមាន និងពិន្ទុសិស្ស"
    tw = p.draw.textbbox((0, 0), t, font=title_f)[2]
    p.text_block("b1", "title", [p.text_line((W - tw) / 2, 110, t, title_f)], 40, align="center", el_type="heading", heading=1)
    # two body lines
    p.text_block("b2", "text", [p.text_line(100, 200, "ថ្នាក់ទី៧ក ឆ្នាំសិក្សា ២០២៥-២០២៦", body_f),
                                p.text_line(100, 238, "គ្រូបន្ទុកថ្នាក់៖ លោកគ្រូ សុខ វណ្ណា", body_f)], 24)
    # table 1: grid, 2 header rows, merged header cells, an empty cell, a low cell, a bold total row with colspan 3
    p.table_block("b3", 100, 300, [0.08, 0.34, 0.14, 0.22, 0.22], 1040, [44, 44, 48, 48, 48, 48], 2, "grid", [
        {"r": 0, "c": 0, "rs": 2, "text": "ល.រ", "align": "c"}, {"r": 0, "c": 1, "rs": 2, "text": "ឈ្មោះសិស្ស", "align": "l"},
        {"r": 0, "c": 2, "rs": 2, "text": "ភេទ", "align": "c"}, {"r": 0, "c": 3, "cs": 2, "text": "ពិន្ទុ", "align": "c"},
        {"r": 1, "c": 3, "text": "ភាសាខ្មែរ", "align": "c"}, {"r": 1, "c": 4, "text": "គណិតវិទ្យា", "align": "c"},
        {"r": 2, "c": 0, "text": "១", "align": "c"}, {"r": 2, "c": 1, "text": "សុខ សុភា"}, {"r": 2, "c": 2, "text": "ស្រី", "align": "c"}, {"r": 2, "c": 3, "text": "៨៥", "align": "r"}, {"r": 2, "c": 4, "text": "៩២", "align": "r"},
        {"r": 3, "c": 0, "text": "២", "align": "c"}, {"r": 3, "c": 1, "text": "ចាន់ ដារ៉ា"}, {"r": 3, "c": 2, "text": "ប្រុស", "align": "c"}, {"r": 3, "c": 3, "text": "៧៨", "align": "r"}, {"r": 3, "c": 4, "text": "", "align": "r"},
        {"r": 4, "c": 0, "text": "៣", "align": "c"}, {"r": 4, "c": 1, "text": "លី មេង"}, {"r": 4, "c": 2, "text": "ប្រុស", "align": "c"}, {"r": 4, "c": 3, "text": "៩១", "align": "r", "low": True, "conf": 0.62}, {"r": 4, "c": 4, "text": "៨៨", "align": "r"},
        {"r": 5, "c": 0, "cs": 3, "text": "សរុប", "align": "r", "bold": True}, {"r": 5, "c": 3, "text": "២៥៤", "align": "r", "bold": True}, {"r": 5, "c": 4, "text": "១៨០", "align": "r", "bold": True},
    ], 24, 0.91)
    # a note paragraph
    p.text_block("b4", "text", [p.text_line(100, 620, "កំណត់សម្គាល់៖ ពិន្ទុខាងលើគឺជាពិន្ទុប្រចាំឆមាសទីមួយ។", body_f),
                                p.text_line(100, 658, "សិស្សដែលអវត្តមានត្រូវប្រឡងសងនៅសប្តាហ៍ក្រោយ។", body_f)], 24)
    # table 2: borderless, Latin, 1 header row, numbers right-aligned, bold total, one empty cell
    p.table_block("b5", 100, 730, [0.5, 0.2, 0.3], 600, [42, 42, 42, 42], 1, "none", [
        {"r": 0, "c": 0, "text": "Item"}, {"r": 0, "c": 1, "text": "Qty", "align": "c"}, {"r": 0, "c": 2, "text": "Price", "align": "r"},
        {"r": 1, "c": 0, "text": "Rice 25 kg"}, {"r": 1, "c": 1, "text": "2", "align": "c"}, {"r": 1, "c": 2, "text": "$ 45.00", "align": "r"},
        {"r": 2, "c": 0, "text": "Fish sauce"}, {"r": 2, "c": 1, "text": "6", "align": "c"}, {"r": 2, "c": 2, "text": "$ 12.50", "align": "r"},
        {"r": 3, "c": 0, "text": "Total", "bold": True}, {"r": 3, "c": 1, "text": "", "align": "c"}, {"r": 3, "c": 2, "text": "$ 57.50", "align": "r", "bold": True},
    ], 22, 0.78)
    # page number
    pn = "ទំព័រ ១"
    pw = p.draw.textbbox((0, 0), pn, font=small_f)[2]
    p.text_block("b6", "page_number", [p.text_line((W - pw) / 2, 1660, pn, small_f)], 20, align="center", el_type="page_number")

    n_lines = sum(len(b["lines"]) for b in p.blocks)
    table_area = sum(b["box"]["width"] * b["box"]["height"] for b in p.blocks if b["cls"] == "table") / (W * H)
    text_area = sum(b["box"]["width"] * b["box"]["height"] for b in p.blocks if b["cls"] != "table") / (W * H)
    result = {
        "page_id": PAGE_ID, "index": 0, "filename": "table_page.png", "image_url": "dev/table_page.png", "width": W, "height": H,
        "blocks": p.blocks,
        "layout": {"width": W, "height": H, "margins": MARGINS, "zones": []},
        "document": {"width": W, "height": H, "margins": MARGINS, "elements": p.elements},
        "doc_kind": {"kind": "mixed", "confidence": 0.81, "source": "model", "tables": 2, "table_area": round(table_area, 3), "text_area": round(text_area, 3)},
        "timings": {"layout_ms": 71.2, "stamps_found": 0, "stamps_ms": 3.1, "textdet_ms": 540.3, "lines_detected": n_lines, "bullets_found": 0, "bullets_ms": 4.1,
                    "rec_ms": 610.5, "lines_recognized": n_lines, "bold_words": 8, "bold_ms": 3.2, "font_ms": 120.4, "structure_ms": 2.1, "order_ms": 12.0,
                    "table_ms": 48.7, "tables": 2, "total_ms": 1420.5},
        "models": [{"stage": "layout", "id": "layout_yolo11n_v3_22cls"}, {"stage": "textdet", "id": "db_ppocrv5_khmer_v2"}, {"stage": "rec", "id": "crnn_m_v2_stamp2"},
                   {"stage": "font", "id": "font_cnn_v2"}, {"stage": "structure", "id": "struct_v1"}, {"stage": "table", "id": "tsr_yolo11n_v1"}],
        "device": "cpu", "ignored_regions": [],
    }
    (HERE / "table_page.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    p.img.save(HERE / "table_page.png", optimize=True)
    print(f"wrote {HERE / 'table_page.json'} ({len(p.blocks)} blocks, {n_lines} lines, {len(p.elements)} elements) and table_page.png")


if __name__ == "__main__":
    main()
