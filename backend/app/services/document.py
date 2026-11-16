"""The page as a structured document (what a word processor would hold), built from the rendered blocks.

Elements in reading order, each with a role, alignment, typeface, size, colour and text runs (bold / regular):
    heading (level 1 title, 2 section header, 3 short bold line), paragraph (first_line_indent, level),
    list (items with one bullet style per level), header, footer, page_number, figure (crop), decoration,
    table (v3: the cell grid of the block's ``table`` structure - rows, cols, header_rows, borders, cells with
    runs / spans / alignment, col_widths as fractions of the table width; one element per table, no table_row)
Consecutive bulleted paragraphs form one list; footer blocks on the same row become one footer with columns.
Nothing here keeps photo coordinates: a renderer reflows it with the page margins, like Word."""

from __future__ import annotations

HEAVY = {"Khmer OS Muol", "Khmer OS Muol Light", "Khmer OS Muol Pali", "Khmer OS Bokor", "Khmer OS Metal Chrieng",
         "Koulen", "Dangrek", "Bayon", "Angkor", "Taprom"}
CROP = {"figure", "logo", "stamp", "photo", "qr_code", "barcode", "signature", "decoration", "checkbox", "formula"}


def _khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿" or "᧠" <= ch <= "᧿"


def paragraph_runs(lines: list[dict]) -> list[dict]:
    """Wrapped lines -> [{text, bold}] runs; a Khmer line break carries no space, a Latin one a space."""
    runs: list[dict] = []
    for ln in lines:
        spans = ln.get("spans") or []
        pieces = [(s["text"], bool(s.get("bold"))) for s in spans if s.get("text")] if spans else [(ln.get("text", ""), bool(ln.get("bold")))]
        pieces = [(t, b) for t, b in pieces if t]
        if not pieces:
            continue
        if runs:
            prev, nxt = runs[-1]["text"], pieces[0][0]
            if prev and not (_khmer(prev[-1]) and _khmer(nxt[0])):
                runs.append({"text": " ", "bold": False})
        for i, (t, b) in enumerate(pieces):
            if i and spans:
                runs.append({"text": " ", "bold": False})
            if runs and runs[-1]["bold"] == b:
                runs[-1]["text"] += t
            else:
                runs.append({"text": t, "bold": b})
    return runs


def _style(b: dict) -> dict:
    f = b.get("font") or {}
    return {"font_px": b.get("font_px"), "family": f.get("family"), "weight": f.get("weight", "regular"), "slant": f.get("slant", "upright"),
            "heavy": f.get("family") in HEAVY, "color": b.get("text_color") or "#1a1a1a",
            "size_model": b.get("font_src") == "model"}  # font_px is the learned em size: render it as is


BODY = {"title", "text", "list", "section_header", "table", "caption", "footnote", "form_field", "handwriting"}
LOW_CONF = 0.8


def table_element(b: dict, base: dict | None = None) -> dict | None:
    """v3 tables: one ``table`` element (contract §2.3) from a table block's ``table`` structure. Cell text comes from
    the block's lines (indices in ``cells[].lines``) joined Khmer-aware; a cell whose ``text`` differs from its lines
    (edited in the UI, or re-read without lines) keeps that text as one run. A table block without a structure (an
    older result) becomes a one-column table with one row per line. None when there is nothing to show."""
    t = b.get("table") or {}
    all_lines = b.get("lines") or []
    cells_in, rows, cols = t.get("cells") or [], t.get("rows") or [], t.get("cols") or []
    if not (cells_in and rows and cols):
        texted = [l for l in all_lines if (l.get("text") or "").strip()]
        if not texted:
            return None
        cells_in = [{"r": i, "c": 0, "rs": 1, "cs": 1, "lines": [all_lines.index(l)], "header": False, "align": "l", "bold": bool(l.get("bold"))}
                    for i, l in enumerate(texted)]
        rows, cols, t = [[0, 0]] * len(texted), [[0, 1]], {"header_rows": 0, "borders": "none", "source": "lines"}
    R, C = len(rows), len(cols)
    widths = [max(0.0, float(c[1]) - float(c[0])) for c in cols]
    tot = sum(widths)
    col_widths = [round(w / tot, 4) for w in widths] if tot > 0 else [round(1.0 / C, 4)] * C
    col_widths[-1] = round(max(0.0, 1.0 - sum(col_widths[:-1])), 4)  # fractions of the table width, sum 1
    cells, low_any = [], False
    for c in cells_in:
        idx = [i for i in (c.get("lines") or []) if isinstance(i, int) and 0 <= i < len(all_lines)]
        cl = [all_lines[i] for i in idx if (all_lines[i].get("text") or "").strip()]
        runs = paragraph_runs(cl) if cl else []
        joined = "".join(r["text"] for r in runs).strip()
        text = (c.get("text") if c.get("text") is not None else joined) or ""
        if text != joined:
            runs = [{"text": text, "bold": bool(c.get("bold"))}] if text else []
        low = any(float(l.get("confidence", 1.0)) < LOW_CONF for l in cl) or (not cl and float(c.get("confidence", 1.0)) < LOW_CONF and bool(text))
        low_any = low_any or low
        cells.append({"r": int(c.get("r", 0)), "c": int(c.get("c", 0)), "rs": max(1, int(c.get("rs", 1))), "cs": max(1, int(c.get("cs", 1))),
                      "header": bool(c.get("header")), "align": c.get("align") or "l", "bold": bool(c.get("bold")),
                      "text": text, "runs": runs, "low": low})
    el = dict(base or {"block": b.get("id"), "role": "table", "align": b.get("align", "left"), "style": _style(b)})
    el.update(type="table", rows=R, cols=C, header_rows=int(t.get("header_rows") or 0), borders=t.get("borders") or "none",
              cells=cells, col_widths=col_widths, low=low_any, source=t.get("source"))
    return el


def _x_clusters(blocks: list[dict]) -> list[list[dict]]:
    """Blocks whose horizontal extents overlap (>= 30 % of the narrower) share a column; columns left to right."""
    cols: list[dict] = []
    for b in blocks:
        bx = b.get("box") or {}
        x0, x1 = bx.get("x", 0), bx.get("x", 0) + bx.get("width", 0)
        for c in cols:
            ov = min(c["x1"], x1) - max(c["x0"], x0)
            if ov >= 0.3 * max(1.0, min(x1 - x0, c["x1"] - c["x0"])):
                c["blocks"].append(b); c["x0"], c["x1"] = min(c["x0"], x0), max(c["x1"], x1)
                break
        else:
            cols.append({"x0": x0, "x1": x1, "blocks": [b]})
    # merge columns that came to overlap after growing
    merged = True
    while merged and len(cols) > 1:
        merged = False
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                ov = min(cols[i]["x1"], cols[j]["x1"]) - max(cols[i]["x0"], cols[j]["x0"])
                if ov >= 0.2 * max(1.0, min(cols[i]["x1"] - cols[i]["x0"], cols[j]["x1"] - cols[j]["x0"])):  # a few px of touch is not an overlap
                    cols[i]["blocks"] += cols[j]["blocks"]; cols[i]["x0"] = min(cols[i]["x0"], cols[j]["x0"]); cols[i]["x1"] = max(cols[i]["x1"], cols[j]["x1"])
                    del cols[j]; merged = True; break
            if merged:
                break
    return [sorted(c["blocks"], key=lambda b: (b.get("box") or {}).get("y", 0)) for c in sorted(cols, key=lambda c: c["x0"])]


def build_document(blocks: list[dict], layout: dict | None, W: int, H: int) -> dict:
    margins = (layout or {}).get("margins") or {"left": 0.08 * W, "right": 0.08 * W, "top": 0.06 * H, "bottom": 0.06 * H}
    content_w = max(1.0, W - margins["left"] - margins["right"])
    ordered = sorted(blocks, key=lambda x: x.get("reading_order", 0))
    # the letterhead zone (everything before the first body block) is often two columns: emblem + ministry name on
    # the left, motto + ornament on the right -> one row with columns, each column stacked top to bottom
    first_body = next((i for i, b in enumerate(ordered) if (b.get("cls") or "text") in BODY), len(ordered))
    head, rest = ordered[:first_body], ordered[first_body:]
    cols = _x_clusters(head) if len(head) >= 2 else []
    if len(cols) >= 2:
        row = {"type": "row", "block": head[0].get("id"), "box": [round(min((b["box"]["x"] for b in head), default=0), 1), round(min((b["box"]["y"] for b in head), default=0), 1), 0, 0],
               "columns": [{"x0": round(min(b["box"]["x"] for b in c), 1), "x1": round(max(b["box"]["x"] + b["box"]["width"] for b in c), 1),
                            "elements": _elements(c, content_w, W)} for c in cols]}
        return _finish([row] + _elements(rest, content_w, W), margins, W, H)
    return _finish(_elements(ordered, content_w, W), margins, W, H)


def _finish(els: list[dict], margins: dict, W: int, H: int) -> dict:
    # footer blocks -> one footer row with columns (address left, phone right); lines of one column stack even when
    # the layout split them into several blocks
    out: list[dict] = []
    i = 0
    while i < len(els):
        if els[i]["type"] != "footer":
            out.append(els[i]); i += 1
            continue
        j = i
        while j < len(els) and els[j]["type"] == "footer":
            j += 1
        group = els[i:j]
        fake_blocks = [{"id": k, "box": {"x": e["box"][0], "y": e["box"][1], "width": e["box"][2], "height": e["box"][3]}} for k, e in enumerate(group)]
        cols = _x_clusters(fake_blocks)
        if len(cols) >= 2:
            columns = [{"block": group[c[0]["id"]]["block"], "align": group[c[0]["id"]]["align"],
                        "elements": [group[fb["id"]] for fb in c]} for c in cols]
            out.append({"type": "footer_row", "block": group[0]["block"], "box": group[0]["box"], "columns": columns})
        else:
            out.extend(group)
        i = j
    return {"width": W, "height": H, "margins": {k: round(float(v), 1) for k, v in margins.items()}, "elements": out}


def _elements(blocks: list[dict], content_w: float, W: int) -> list[dict]:
    els: list[dict] = []
    for b in blocks:
        cls = b.get("cls") or "text"
        box = b.get("box") or {}
        base = {"block": b.get("id"), "role": cls, "align": b.get("align", "left"), "style": _style(b),
                "box": [round(box.get("x", 0), 1), round(box.get("y", 0), 1), round(box.get("width", 0), 1), round(box.get("height", 0), 1)]}
        if cls == "table":  # v3 tables: one element per table (contract §2.3)
            el = table_element(b, base)
            if el is not None:
                els.append(el)
            continue
        if cls in CROP or not (b.get("lines") or []):
            if cls == "decoration":
                els.append({**base, "type": "decoration", "crop": (b.get("crop") or {}).get("url")})
            elif b.get("crop"):
                els.append({**base, "type": "figure", "crop": b["crop"].get("url"), "width_frac": round(min(1.0, box.get("width", 0) / content_w), 3),
                            "aspect": round(box.get("height", 1) / max(1.0, box.get("width", 1)), 4)})
            continue
        lines = [l for l in b.get("lines") or [] if l.get("text")]
        if not lines:  # every line of the block read as empty: nothing to show
            continue
        by_order = {l.get("reading_order"): l for l in lines}
        paras = b.get("paragraphs") or [{"lines": [l.get("reading_order") for l in lines], "indent_level": lines[0].get("indent_level", 0), "bullet": lines[0].get("bullet")}]
        short_bold = len(paras) == 1 and len(lines) == 1 and (b.get("bold") or base["style"]["heavy"]) and box.get("width", 0) < 0.6 * content_w and len(lines[0]["text"]) <= 60
        for p in paras:
            pl = [by_order[i] for i in p.get("lines", []) if i in by_order]
            if not pl:
                continue
            runs = p.get("runs") or paragraph_runs(pl)
            levels = [int(l.get("indent_level", 0)) for l in pl]
            first_indent = len(pl) > 1 and levels[0] > min(levels[1:])
            level = min(levels) if first_indent else int(p.get("indent_level", levels[0]))
            item = {**base, "runs": runs, "text": "".join(r["text"] for r in runs), "level": level, "lines": [l.get("reading_order") for l in pl],
                    "low": any(l.get("confidence", 1) < 0.8 for l in pl)}
            if p.get("bullet"):
                item.update(type="list_item", bullet=p["bullet"])
                if els and els[-1]["type"] == "list":
                    els[-1]["items"].append(item)
                else:
                    els.append({**base, "type": "list", "items": [item]})
                continue
            if cls == "title":
                item.update(type="heading", heading=1)
            elif cls == "section_header" or (cls in ("text", "list") and short_bold):
                item.update(type="heading", heading=2 if cls == "section_header" else 3)
            elif cls in ("header", "footer", "page_number", "caption", "footnote"):
                item.update(type=cls)
            else:
                item.update(type="paragraph", first_line_indent=bool(first_indent))
            els.append(item)
    return els
