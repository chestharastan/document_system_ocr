"""Layout blocks. Each block draws onto a Page and registers its labels (v2 taxonomy).

Labelling rules (consistent with DocLayNet, see README in this package):
  text = paragraph block, title = document title, section_header = heading, list = whole list block,
  table = whole grid, caption = caption block, header/footer = running header/footer block, page_number = the number
  alone, form_field = label + input region (+ value), checkbox = box glyph (+ tick) only, handwriting = handwritten
  value ink, signature / stamp / logo / photo / qr_code / barcode / figure / formula = their ink / image box.
Blocks return the y coordinate below what they drew (or the input y if nothing fitted).
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from . import assets2 as X
from . import tables as T
from .canvas import Page, bbox, rect_quad, to_pts, wrap
from .ctx import Ctx
from .res import R, font, font_ok

from det import assets as A  # noqa: E402  (old repo, via res sys.path)


def text_block(pg: Page, ctx: Ctx, x, y, w, text, role="body", size=None, fill=None, align="l", justify=False, lh=None,
               max_lines=99, indent=0, ymax=None, kind="print", squeeze=1.0, path=None, srole=None):
    """Wrapped paragraph. Returns (y_end, [quads]). ``srole`` = structure role recorded for its lines (when the caller
    has not opened a paragraph with pg.begin())."""
    own = pg.cur is None
    if own:
        pg.begin(srole or "text", 0, None, indent=indent)
    try:
        return _text_block(pg, ctx, x, y, w, text, role, size, fill, align, justify, lh, max_lines, indent, ymax, kind, squeeze, path)
    finally:
        if own:
            pg.end()


def _text_block(pg, ctx, x, y, w, text, role, size, fill, align, justify, lh, max_lines, indent, ymax, kind, squeeze, path):
    size = int(size or ctx.size)
    path = path or ctx.font(role, text)
    if not path:
        return y, []
    f = font(path, size)
    lines = wrap(text, f, (w - indent) / squeeze)[:max_lines]
    fill = fill or ctx.fg
    ymax = ymax or pg.H - 8
    step = size * (lh or ctx.lh)
    quads, yy = [], y
    for i, ln in enumerate(lines):
        if yy + size * 1.45 > ymax:
            break
        xx = x + (indent if i == 0 else 0)
        toks = ln.split(" ")
        q = None
        if justify and i < len(lines) - 1 and len(toks) > 3 and align == "l":
            tw = sum(f.getlength(t) for t in toks) * squeeze
            gap = (w - (xx - x) - tw) / (len(toks) - 1)
            if 0 < gap < size * 1.0:
                qs, cx = [], xx
                for t in toks:
                    qs.append(pg.text(cx, yy, t, path, size, fill, kind=None, squeeze=squeeze))
                    cx += f.getlength(t) * squeeze + gap
                if all(qs):
                    q = list(to_pts(qs).min(0)) + list(to_pts(qs).max(0))
                    q = [round(float(v), 1) for v in rect_quad(*q)]
                    if kind == "print":
                        pg._rec(q, " ".join(toks), path)
                quads.append(q)
                yy += step
                continue
        ax = xx if align == "l" else (x + w / 2 if align == "c" else x + w)
        q = pg.text(ax, yy, ln, path, size, fill, align=align, kind=kind, squeeze=squeeze)
        if q:
            quads.append(q)
        yy += step
    if not quads:
        return y, quads
    bottom = max(max(q[1::2]) for q in quads)
    return max(yy, bottom + size * 0.25), quads


def fit_text(pg, x, y, text, path, size, max_w, fill, align="l", min_scale=0.6, **kw):
    """Draw a single line, shrinking the font (down to min_scale) so it fits into max_w."""
    if not path:
        return None
    s = size
    while s >= size * min_scale:
        if font(path, int(s)).getlength(text) <= max_w:
            return pg.text(x, y, text, path, int(s), fill, align=align, **kw)
        s *= 0.9
    return None


# --------------------------------------------------------------------------- text blocks
def b_para(pg, ctx, x, y, w, ymax=None, n=None, size=None, cls="text", align="l", text=None, role="body", fill=None, indent=None, lh=None):
    rng = ctx.rng
    n = n or rng.randint(2, 7)
    sz = int(size or ctx.size)
    t = text or ctx.para(int(w / sz * (1.5 if ctx.lang == "km" else 1.9) * n))
    ind = indent if indent is not None else (int(sz * 1.6) if rng.random() < 0.35 else 0)
    ye, qs = text_block(pg, ctx, x, y, w, t, role, sz, fill, align=align, justify=ctx.justify and align == "l", max_lines=n,
                        indent=ind, ymax=ymax, lh=lh, srole=cls)
    if qs:
        pg.add(cls, qs)
        return ye + sz * rng.uniform(0.4, 0.9)
    return y


def b_heading(pg, ctx, x, y, w, ymax=None, cls="section_header", text=None, size=None, align="l", role="bold", fill=None,
              number=None, underline=False, max_lines=2):
    rng = ctx.rng
    sz = int(size or ctx.size * (rng.uniform(1.05, 1.4) if cls == "section_header" else rng.uniform(1.5, 2.2)))
    t = text or ctx.heading()
    if number:
        t = f"{number} {t}"
    ye, qs = text_block(pg, ctx, x, y, w, t, role, sz, fill or (ctx.accent if rng.random() < 0.35 else ctx.fg), align=align,
                        max_lines=max_lines, ymax=ymax, lh=1.35 if ctx.lang == "km" else 1.15, srole=cls)
    if not qs:
        return y
    b = pg.add(cls, qs)
    if underline and b:
        pg.d.line([b[0], b[3] + 5, b[2] if rng.random() < 0.5 else x + w, b[3] + 5], fill=ctx.fg, width=2)
        ye += 6
    return ye + sz * rng.uniform(0.25, 0.6)


def list_marker(ctx, i, style):
    kd = ctx.d
    return {"bullet": "•", "dash": "-", "num": f"{kd(i + 1)}.", "paren": f"{kd(i + 1)})", "alpha": ("abcdefgh"[i % 8] + ")") if ctx.lang == "en" else ("កខគឃងចឆជ"[i % 9] + "."),
            "square": "▪", "arrow": "➢", "roman": ["i.", "ii.", "iii.", "iv.", "v.", "vi.", "vii.", "viii."][i % 8],
            "ring": "o", "circle": "◦", "endash": "–"}[style]


def b_list(pg, ctx, x, y, w, ymax=None, n=None, items=None, size=None, style=None):
    rng = ctx.rng
    sz = int(size or ctx.size)
    style = style or rng.choice(["bullet", "bullet", "dash", "num", "paren", "alpha", "square", "arrow", "roman"])
    n = len(items) if items else (n or rng.randint(3, 6))
    ind = int(sz * rng.uniform(1.4, 2.2))
    allq, yy = [], y
    sub_style = rng.choice(["ring", "circle", "dash", "square", "endash"]) if style not in ("ring", "circle") else "dash"
    sub_p = 0.3 if not items else 0.0  # Khmer official lists: "-" items with "o" sub-items
    ind2 = ind + int(sz * rng.uniform(1.4, 2.0))
    for i in range(n):
        mark = list_marker(ctx, i, style)
        mp = ctx.font("body", mark) or ctx.font("latin", mark)
        if not mp:
            style = "dash"
            mark = "-"
            mp = ctx.font("body", mark)
        t = items[i] if items else (ctx.phrase(20, 90) if rng.random() < 0.6 else ctx.para(rng.randint(60, 180)))
        y0 = yy
        pg.begin("list_item", 1, style, indent=0)
        yy2, qs = text_block(pg, ctx, x + ind, yy, w - ind, t, "body", sz, max_lines=3, ymax=ymax)
        if not qs:
            pg.end()
            break
        mq = pg.text(x + sz * 0.3, y0, mark, mp, sz, ctx.fg, marker=True)
        pg.end()
        allq += qs + ([mq] if mq else [])
        yy = yy2 + sz * rng.uniform(0.1, 0.4)
        if rng.random() < sub_p:
            for j in range(rng.randint(1, 3)):
                smark = list_marker(ctx, j, sub_style)
                smp = ctx.font("body", smark) or ctx.font("latin", smark)
                if not smp:
                    smark, smp = "-", ctx.font("body", "-")
                st = ctx.phrase(15, 70)
                sy0 = yy
                pg.begin("list_item", 2, sub_style, indent=0)
                yy3, sqs = text_block(pg, ctx, x + ind2, yy, w - ind2, st, "body", sz, max_lines=2, ymax=ymax)
                if not sqs:
                    pg.end()
                    break
                smq = pg.text(x + ind + sz * 0.3, sy0, smark, smp, int(sz * 0.9), ctx.fg, marker=True)
                pg.end()
                allq += sqs + ([smq] if smq else [])
                yy = yy3 + sz * rng.uniform(0.05, 0.3)
    if allq:
        pg.add("list", allq)
        return yy + sz * 0.5
    return y


def b_caption(pg, ctx, x, y, w, ymax=None, prefix=None, size=None, align=None):
    rng = ctx.rng
    sz = int(size or ctx.size * rng.uniform(0.8, 0.92))
    t = (prefix + " " if prefix else "") + ctx.phrase(15, 80)
    ye, qs = text_block(pg, ctx, x, y, w, t, "italic" if (ctx.lang == "en" and rng.random() < 0.5) else "body", sz,
                        align=align or rng.choice(["l", "c"]), max_lines=2, ymax=ymax, srole="caption")
    if qs:
        pg.add("caption", qs)
        return ye + sz * 0.3
    return y


def fig_prefix(ctx, i):
    return f"រូបភាពទី{ctx.d(i)}៖" if ctx.lang == "km" else ctx.rng.choice([f"Figure {i}.", f"Fig. {i}:", f"Figure {i}:"])


def tab_prefix(ctx, i):
    return f"តារាងទី{ctx.d(i)}៖" if ctx.lang == "km" else ctx.rng.choice([f"Table {i}.", f"Table {i}:", f"TABLE {i}"])


def b_figure(pg, ctx, x, y, w, ymax=None, kind=None, caption=True, fw=None, fh=None, num=None):
    rng = ctx.rng
    ymax = ymax or pg.H - 40
    fw = int(fw or w * rng.uniform(0.45, 1.0))
    fh = int(fh or fw * rng.uniform(0.45, 0.8))
    if y + fh + (ctx.size * 3 if caption else 0) > ymax:
        fh = int(ymax - y - (ctx.size * 3 if caption else 0))
        if fh < 120:
            return y
    kind = kind or rng.choice(["chart", "chart", "scene", "scene", "diagram", "map"])
    im = {"chart": X.make_chart, "scene": X.make_scene, "diagram": X.make_diagram, "map": X.make_map}[kind](rng, fw, fh)
    im = im.resize((fw, fh)) if im.size != (fw, fh) else im
    xx = x + (w - fw) * rng.choice([0, 0.5, 0.5, 1])
    top_cap = caption and rng.random() < 0.15
    if top_cap:
        y = b_caption(pg, ctx, xx, y, fw, ymax, fig_prefix(ctx, num or rng.randint(1, 12)))
    if not pg.paste(im, (xx, y), "figure"):
        return y
    ye = y + fh + 8
    if caption and not top_cap:
        ye = b_caption(pg, ctx, xx, ye, fw, ymax, fig_prefix(ctx, num or rng.randint(1, 12)))
    return ye + ctx.size * 0.6


# --------------------------------------------------------------------------- tables
def table_data(ctx, ncol, nrow, kind="generic"):
    rng, fk, L = ctx.rng, ctx.fake, ctx.lang
    if kind == "generic":
        hdr = [ctx.phrase(3, 14) for _ in range(ncol)]
        types = [rng.choice(["txt", "num", "num", "money", "date"]) for _ in range(ncol)]
        types[0] = rng.choice(["idx", "txt"])
        rows = []
        for r in range(nrow):
            row = []
            for t in types:
                row.append({"idx": ctx.d(r + 1), "txt": ctx.phrase(3, 22), "num": ctx.d(rng.choice([rng.randint(0, 999), rng.randint(1000, 99999)])),
                            "money": ctx.money(1, 20000), "date": ctx.date() if L == "en" else ctx.d(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2018, 2026)}")}[t])
            rows.append(row)
        al = ["c" if t == "idx" else ("r" if t in ("num", "money") else "l") for t in types]
        return hdr, rows, al
    if kind == "items":
        if L == "km":
            hdr = rng.choice([["ល.រ", "បរិយាយ", "ចំនួន", "តម្លៃឯកតា", "សរុប"], ["ល.រ", "ឈ្មោះទំនិញ", "បរិមាណ", "តម្លៃ", "ទឹកប្រាក់"],
                              ["No.", "Description / បរិយាយ", "Qty", "Unit Price", "Amount"]])
        else:
            hdr = rng.choice([["#", "Description", "Qty", "Unit Price", "Amount"], ["Item", "Description", "Quantity", "Rate", "Total"],
                              ["SKU", "Product", "Qty", "Price", "Line Total"]])
        rows = []
        for r in range(nrow):
            q = rng.randint(1, 50)
            p = rng.uniform(0.5, 400)
            if L == "km":
                desc = rng.choice(X_PRODUCTS_KM()) if rng.random() < 0.75 else fk.word().title() + " " + rng.choice(["Pro", "Plus", "A4", "500ml", "1kg"])
            else:
                desc = rng.choice([fk.catch_phrase(), f"{fk.word().title()} {rng.choice(['service', 'license', 'kit', 'unit', 'hours', 'support', 'subscription'])}",
                                   f"{fk.color_name()} {fk.word()} ({rng.choice(['S', 'M', 'L', 'XL'])})"])[:34]
            first = ctx.d(r + 1) if hdr[0] in ("ល.រ", "No.", "#") else (f"{rng.randint(1000, 99999)}" if hdr[0] == "SKU" else ctx.d(r + 1))
            rows.append([first, desc, ctx.d(q), f"{p:,.2f}", f"{p * q:,.2f}"])
        return hdr, rows, ["c", "l", "r", "r", "r"]
    if kind == "statement":
        if L == "km":
            hdr = rng.choice([["កាលបរិច្ឆេទ", "ប្រតិបត្តិការ", "ដក", "ដាក់", "សមតុល្យ"], ["Date", "Description / បរិយាយ", "Debit", "Credit", "Balance"]])
        else:
            hdr = rng.choice([["Date", "Description", "Withdrawals", "Deposits", "Balance"], ["Posted", "Transaction", "Debit", "Credit", "Balance"]])
        bal = rng.uniform(100, 20000)
        rows = []
        for r in range(nrow):
            amt = rng.uniform(1, 1500)
            deb = rng.random() < 0.65
            bal = bal - amt if deb else bal + amt
            if L == "km":
                desc = rng.choice(["ផ្ទេរប្រាក់ចេញ", "ដកប្រាក់ ATM", "ទទួលប្រាក់ខែ", "បង់ថ្លៃអគ្គិសនី", "ទូទាត់ KHQR", "ការប្រាក់", "ផ្ទេរប្រាក់ចូល",
                                   f"POS {fk.company()[:14]}", "Transfer to " + ctx.kh_name(True)[0]])
                dt = ctx.d(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2023, 2026)}")
            else:
                desc = rng.choice([f"POS PURCHASE {fk.company()[:18].upper()}", "ATM WITHDRAWAL", f"ACH DEPOSIT {fk.company()[:14].upper()} PAYROLL",
                                   f"ONLINE TRANSFER TO {fk.last_name().upper()}", "MONTHLY SERVICE FEE", f"CHECK #{rng.randint(1000, 9999)}",
                                   f"DEBIT CARD {fk.city().upper()[:12]}", "INTEREST PAYMENT"])
                dt = fk.date(pattern="%m/%d")
            rows.append([dt, desc, f"{amt:,.2f}" if deb else "", "" if deb else f"{amt:,.2f}", f"{bal:,.2f}"])
        return hdr, rows, ["l", "l", "r", "r", "r"]
    raise ValueError(kind)


def X_PRODUCTS_KM():
    from .ctx import KH_PRODUCTS
    return KH_PRODUCTS


def b_table(pg, ctx, x, y, w, ymax=None, hdr=None, rows=None, al=None, kind="generic", ncol=None, nrow=None, style=None,
            size=None, caption=None, cap_num=None, col_w=None, hdr_fill=None, spec=None, sheet=False, variety=None, tstyle=None,
            min_body=1):
    """Table block (v5 engine, training/docgen/tables.py). Keeps the old call signature: ``hdr``/``rows``/``al`` =
    explicit data (rendered as given, no structural variety unless ``variety=True``), ``kind``/``ncol``/``nrow`` =
    generated data (with spans / group rows / totals / empty + wrapped cells), ``style`` = legacy or v5 style name.
    ``spec`` = a prepared tables.py spec (sheet doc types), ``tstyle`` = a prepared style dict, ``sheet`` = the
    contract's sheet flag. Labels: the ``table`` layout box + a ``pg.meta["tables"]`` entry (cells, rows, cols)."""
    pg.begin("table")
    try:
        return _b_table_v5(pg, ctx, x, y, w, ymax=ymax, hdr=hdr, rows=rows, al=al, kind=kind, ncol=ncol, nrow=nrow, style=style, size=size,
                           caption=caption, cap_num=cap_num, col_w=col_w, hdr_fill=hdr_fill, spec=spec, sheet=sheet, variety=variety, tstyle=tstyle,
                           min_body=min_body)
    finally:
        pg.end()


def _b_table_v5(pg, ctx, x, y, w, ymax=None, hdr=None, rows=None, al=None, kind="generic", ncol=None, nrow=None, style=None,
                size=None, caption=None, cap_num=None, col_w=None, hdr_fill=None, spec=None, sheet=False, variety=None, tstyle=None,
                min_body=1):
    rng = ctx.rng
    ymax = ymax or pg.H - 60
    sz = int(size or ctx.size * rng.uniform(0.72, 0.92))
    if spec is None:
        if hdr is None:
            ncol = ncol or (5 if kind in ("items", "statement") else rng.randint(2, 6))
            nrow = nrow or rng.randint(3, 10)
            hdr, rows, al = table_data(ctx, ncol, nrow, kind)
            variety = True if variety is None else variety
        else:
            variety = False if variety is None else variety
        spec = T.spec_from_data(hdr, rows, al)
        if variety:
            T.add_variety(ctx, spec, kind=kind)
    if col_w is not None:
        spec["col_frac"] = list(col_w)
    st = tstyle or T.choose_style(rng, style, hdr_fill=hdr_fill, sheet=sheet, ncol=spec["ncol"])
    cap_h = sz * 3 if caption else 0
    yc = y
    if caption and rng.random() < 0.7:  # table captions usually above
        pg.end()  # the caption is its own paragraph, not part of the table
        yc = b_caption(pg, ctx, x, y, w, ymax, tab_prefix(ctx, cap_num or rng.randint(1, 9)))
        pg.begin("table")
        caption = False
        cap_h = 0
    y_end, lab = T.render_table(pg, ctx, x, int(yc), w, ymax - cap_h, spec, st, sz, sheet=sheet, min_body=min_body)
    if lab is None:
        return y
    ye = y_end + sz * 0.6
    if caption:
        pg.end()
        ye = b_caption(pg, ctx, x, ye, w, ymax, tab_prefix(ctx, cap_num or rng.randint(1, 9)))
        pg.begin("table")
    return ye + ctx.size * 0.5


def _b_table_v4(pg, ctx, x, y, w, ymax=None, hdr=None, rows=None, al=None, kind="generic", ncol=None, nrow=None, style=None,
                size=None, caption=None, cap_num=None, col_w=None, hdr_fill=None):
    """docgen v2-v4 table renderer (kept for reference; b_table now uses _b_table_v5)."""
    rng = ctx.rng
    ymax = ymax or pg.H - 60
    sz = int(size or ctx.size * rng.uniform(0.72, 0.92))
    if hdr is None:
        ncol = ncol or (5 if kind in ("items", "statement") else rng.randint(2, 6))
        nrow = nrow or rng.randint(3, 10)
        hdr, rows, al = table_data(ctx, ncol, nrow, kind)
    ncol = len(hdr)
    rh = int(sz * (rng.uniform(1.9, 2.4) if ctx.lang == "km" else rng.uniform(1.6, 2.1)))
    cap_h = sz * 3 if caption else 0
    max_rows = int((ymax - y - cap_h) / rh) - 1
    if max_rows < 2:
        return y
    rows = rows[:max_rows]
    if col_w is None:
        lens = np.array([max([len(hdr[c])] + [len(r[c]) for r in rows]) for c in range(ncol)], np.float32)
        ws = np.clip(lens, 3, 40) ** 0.9
    else:
        ws = np.array(col_w, np.float32)
    ws = (ws / ws.sum() * w).astype(int)
    xs = np.concatenate([[0], np.cumsum(ws)]) + int(x)
    style = style or rng.choice(["grid", "grid", "hlines", "none", "zebra"])
    lw = rng.choice([1, 1, 2])
    col = (rng.randint(20, 110),) * 3
    yc = y
    if caption and rng.random() < 0.7:  # table captions usually above
        yc = b_caption(pg, ctx, x, y, w, ymax, tab_prefix(ctx, cap_num or rng.randint(1, 9)))
        caption = False
    ty = int(yc)
    nr = len(rows) + 1
    d = pg.d
    hf = hdr_fill if hdr_fill is not None else (rng.choice([(225, 230, 240), (235, 235, 235), (210, 225, 215), (40, 60, 110), (240, 225, 200)]) if rng.random() < 0.6 else None)
    if hf:
        d.rectangle([xs[0], ty, xs[-1], ty + rh], fill=hf)
    if style == "zebra":
        for r in range(1, nr):
            if r % 2 == 0:
                d.rectangle([xs[0], ty + r * rh, xs[-1], ty + (r + 1) * rh], fill=(242, 244, 248))
    ink = []
    for r in range(nr + 1):
        yy = ty + r * rh
        if style in ("grid", "hlines") or (style in ("none", "zebra") and r in (0, 1, nr)):
            d.line([xs[0], yy, xs[-1], yy], fill=col, width=lw)
            ink.append((xs[0], yy, xs[-1], yy + lw))
    if style == "grid":
        for c in range(ncol + 1):
            d.line([xs[c], ty, xs[c], ty + nr * rh], fill=col, width=lw)
    dark_hdr = hf is not None and sum(hf) < 300
    for r in range(nr):
        cells = hdr if r == 0 else rows[r - 1]
        for c in range(ncol):
            t = cells[c]
            if not t:
                continue
            role = "bold" if r == 0 else "body"
            a = "c" if r == 0 else (al[c] if al else "l")
            path = ctx.font(role, t)
            if not path:
                continue
            f = font(path, sz)
            pad = 6
            lines = wrap(t, f, ws[c] - 2 * pad)[:2]
            lh = sz * 1.25
            for k, s in enumerate(lines):
                cx = xs[c] + (ws[c] / 2 if a == "c" else (ws[c] - pad if a == "r" else pad))
                cy = ty + r * rh + (rh - lh * len(lines)) / 2 + k * lh - sz * 0.05
                q = pg.text(cx, cy, s, path, sz, (250, 250, 250) if (r == 0 and dark_hdr) else ctx.fg, align=a,
                            bound=(xs[c] + 1, ty + r * rh - sz * 0.3, xs[c + 1] - 1, ty + (r + 1) * rh + sz * 0.3))
                if q:
                    ink.append(q)
    if style in ("grid", "zebra") or hf:
        pg.add("table", (xs[0], ty, xs[-1] + 1, ty + nr * rh + 1))
    else:
        pg.add("table", ink)
    ye = ty + nr * rh + sz * 0.6
    if caption:
        ye = b_caption(pg, ctx, x, ye, w, ymax, tab_prefix(ctx, cap_num or rng.randint(1, 9)))
    return ye + ctx.size * 0.5


# --------------------------------------------------------------------------- running header / footer / page number / footnote
def b_header(pg, ctx, x, y, w, text=None, rule=None):
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.7, 0.85))
    t = text or (ctx.phrase(10, 50) if rng.random() < 0.7 else ctx.company())
    p = ctx.font("italic" if rng.random() < 0.3 else "body", t)
    al = rng.choice(["l", "r", "c"])
    pg.begin("header")
    q = pg.text({"l": x, "r": x + w, "c": x + w / 2}[al], y, t, p, sz, (90, 90, 90))
    pg.end()
    qs = [q]
    if rng.random() < 0.35 and q:  # second item at the other side (e.g. date / chapter)
        t2 = ctx.date() if rng.random() < 0.5 else ctx.phrase(5, 25)
        p2 = ctx.font("body", t2)
        pg.begin("header")
        q2 = pg.text(x + w if al != "r" else x, y, t2, p2, sz, (90, 90, 90), align="r" if al != "r" else "l")
        pg.end()
        if q2 and q and not overlaps(q, q2):
            pg.add("header", [q2])
    b = pg.add("header", qs)
    if b and (rule if rule is not None else rng.random() < 0.6):
        pg.d.line([x, b[3] + 8, x + w, b[3] + 8], fill=(140, 140, 140), width=1)
    return b


def overlaps(a, b):
    A_ = bbox([a])
    B_ = bbox([b])
    return not (A_[2] < B_[0] or B_[2] < A_[0] or A_[3] < B_[1] or B_[3] < A_[1])


def b_footer(pg, ctx, x, y, w, page_no=True, text=True):
    """Footer text block and/or page number (separate boxes). y = top of the footer line."""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.68, 0.82))
    pn_side = rng.choice(["c", "r", "l"])
    if text:
        t = rng.choice([ctx.company(), ctx.phrase(10, 60), " | ".join([ctx.phone(), ctx.fake.free_email() if ctx.lang == "en" else ctx.fake.company_email()])])
        if rng.random() < 0.5:
            pg.d.line([x, y - 10, x + w, y - 10], fill=(150, 150, 150), width=1)
        al = "l" if pn_side != "l" else "r"
        p = ctx.font("body", t)
        f = font(p, sz) if p else None
        if f and f.getlength(t) < w * 0.7:
            pg.begin("footer")
            q = pg.text(x if al == "l" else x + w, y, t, p, sz, (90, 90, 90), align=al)
            pg.end()
            pg.add("footer", [q])
    if page_no:
        b_page_number(pg, ctx, {"c": x + w / 2, "r": x + w, "l": x}[pn_side], y, pn_side, sz)


def b_page_number(pg, ctx, x, y, align="c", size=None):
    rng = ctx.rng
    n = rng.randint(1, 240)
    t = rng.choice([ctx.d(n), f"- {ctx.d(n)} -", f"{ctx.d(n)}", (f"ទំព័រ {ctx.d(n)}" if ctx.lang == "km" else f"Page {n}"),
                    (f"ទំព័រ {ctx.d(n)}/{ctx.d(n + rng.randint(0, 9))}" if ctx.lang == "km" else f"Page {n} of {n + rng.randint(0, 20)}")])
    p = ctx.font("body", t)
    pg.begin("page_number")
    q = pg.text(x, y, t, p, int(size or ctx.size * 0.8), (70, 70, 70), align=align)
    pg.end()
    return pg.add("page_number", [q])


def b_footnotes(pg, ctx, x, y_bottom, w, n=None):
    """Footnotes stacked upward from y_bottom; returns the top y used (so the body can stop above it)."""
    rng = ctx.rng
    n = n or rng.randint(1, 3)
    sz = int(ctx.size * rng.uniform(0.66, 0.78))
    texts = [ctx.para(rng.randint(50, 200)) for _ in range(n)]
    p = ctx.font("body", "".join(texts)) or ctx.font("body", texts[0])
    if not p:
        return y_bottom
    f = font(p, sz)
    lh = sz * (1.45 if ctx.lang == "km" else 1.25)
    heights = [len(wrap(t, f, w - sz * 1.2)[:3]) * lh + sz * 0.45 for t in texts]
    top = y_bottom - sum(heights)
    pg.d.line([x, top - sz * 0.6, x + w * rng.uniform(0.25, 0.4), top - sz * 0.6], fill=(80, 80, 80), width=1)
    yy = top
    for i, t in enumerate(texts):
        mark = ctx.d(i + 1) if rng.random() < 0.7 else "*" * (i + 1)
        pg.begin("footnote", 0, "note")
        mq = pg.text(x, yy - sz * 0.15, mark, ctx.font("body", mark) or p, int(sz * 0.75), ctx.fg, marker=True)
        ye, qs = text_block(pg, ctx, x + sz * 1.2, yy, w - sz * 1.2, t, "body", sz, max_lines=3, lh=lh / sz, path=p if font_ok(p, t) else None)
        pg.end()
        if qs:
            pg.add("footnote", qs + ([mq] if mq else []))
        yy = max(yy + heights[i], ye + sz * 0.2)
    return top - sz * 1.2


def b_formula(pg, ctx, x, y, w, ymax=None, number=None):
    rng = ctx.rng
    hpx = int(ctx.size * rng.uniform(1.6, 2.4))
    im = X.make_formula(rng, hpx, ctx.fg)
    if im is None or im.width > w * 0.85:
        if im is not None:
            s = w * 0.8 / im.width
            im = im.resize((max(4, int(im.width * s)), max(4, int(im.height * s))), Image.LANCZOS)
        else:
            return y
    ymax = ymax or pg.H - 40
    if y + im.height + 10 > ymax:
        return y
    fx = x + (w - im.width) / 2 if rng.random() < 0.75 else x + ctx.size * 2
    yy = y + ctx.size * 0.3
    b = pg.ink(im, (fx, yy), None)
    if not b:
        return y
    from .canvas import hull_pts
    pts = hull_pts(np.asarray(im)[..., 3] > 40, int(fx), int(yy))
    if pts is None:
        return y
    pg.formula_lines.append([round(float(v), 1) for v in rect_quad(*pts.min(0), *pts.max(0))])
    allp = [pts]
    if number or rng.random() < 0.5:
        t = f"({number or rng.randint(1, 30)})"
        p = ctx.font("body", t)
        q = pg.text(x + w, yy + im.height / 2 - ctx.size * 0.6, t, p, ctx.size, ctx.fg, align="r")
        if q:
            allp.append(to_pts([q]))
    pg.add("formula", np.concatenate(allp))
    return yy + im.height + ctx.size * 0.8


# --------------------------------------------------------------------------- forms
VALUE_RULES = [  # (keywords in label, kind)
    (("email", "អ៊ីមែល"), "email"), (("phone", "ទូរស័ព្ទ"), "phone"), (("birth", "កំណើត", "date", "កាលបរិច្ឆេទ"), "date"),
    (("zip",), "zip"), (("city",), "city"), (("state",), "state"), (("country",), "country"), (("address", "អាសយដ្ឋាន", "apt"), "address"),
    (("amount", "$", "ទឹកប្រាក់"), "money"), (("account", "number", "no.", "id", "លេខ", "social"), "digits"), (("occupation", "employer", "មុខរបរ"), "job"),
    (("name", "ឈ្មោះ", "នាម", "contact"), "name"), (("sex", "ភេទ"), "sex"), (("nationality", "សញ្ជាតិ"), "nat"), (("reason", "មូលហេតុ"), "reason"),
]


def value_for(ctx: Ctx, label: str) -> str:
    rng, fk = ctx.rng, ctx.fake
    lab = label.lower()
    kind = next((k for keys, k in VALUE_RULES if any(kw in lab for kw in keys)), None)
    km = ctx.lang == "km"
    if kind is None:
        return hw_value(ctx)
    if kind == "email":
        return fk.free_email()
    if kind == "phone":
        return ctx.phone()
    if kind == "date":
        return ctx.d(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1950, 2026)}") if km else fk.date(pattern=rng.choice(["%m/%d/%Y", "%m-%d-%y"]))
    if kind == "zip":
        return fk.zipcode()
    if kind == "city":
        return fk.city()
    if kind == "state":
        return rng.choice(["CA", "TX", "NY", "Ohio", "Florida", "WA"])
    if kind == "country":
        return rng.choice(["USA", "United States", "Cambodia", "Canada"])
    if kind == "address":
        return (rng.choice(X_KH_PROV()) if rng.random() < 0.5 else ctx.address()[0]) if km else fk.street_address()
    if kind == "money":
        return f"${rng.uniform(10, 9000):,.2f}"
    if kind == "digits":
        return ctx.d(str(rng.randint(10 ** 5, 10 ** 10))) if rng.random() < 0.7 else f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
    if kind == "job":
        return rng.choice(["គ្រូបង្រៀន", "កសិករ", "និស្សិត", "មន្ត្រីរាជការ", "អាជីវករ", "វិស្វករ"]) if km else fk.job()[:28]
    if kind == "name":
        return (ctx.kh_name()[0] if rng.random() < 0.7 else ctx.kh_name(True)[0].title()) if km else fk.name()
    if kind == "sex":
        return rng.choice(["ប្រុស", "ស្រី"]) if km else rng.choice(["M", "F", "Male", "Female"])
    if kind == "nat":
        return rng.choice(["ខ្មែរ", "Khmer", "Cambodian"]) if km else rng.choice(["American", "US"])
    return ctx.phrase(8, 30)


def hw_value(ctx: Ctx, label_kind="generic") -> str:
    rng, fk = ctx.rng, ctx.fake
    if ctx.lang == "km":
        return rng.choice([ctx.kh_name()[0], ctx.kh_name(True)[0].title(), ctx.d(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1960, 2010)}"),
                           rng.choice(X_KH_PROV()), ctx.phone(), rng.choice(["ប្រុស", "ស្រី", "ខ្មែរ", "គ្រូបង្រៀន", "កសិករ", "និស្សិត", "មន្ត្រីរាជការ", "អាជីវករ"]),
                           ctx.d(str(rng.randint(10000000, 99999999))), "ភូមិ" + rng.choice(["ថ្មី", "ស្វាយ", "ពោធិ៍", "ក្រាំង"])])
    return rng.choice([fk.name(), fk.first_name(), fk.date(pattern="%m/%d/%Y"), fk.city(), fk.street_address(), fk.phone_number(),
                       fk.job()[:28], fk.free_email(), str(rng.randint(100, 99999)), fk.last_name(), f"${rng.uniform(10, 9000):,.2f}"])


def X_KH_PROV():
    from .ctx import KH_PROV
    return KH_PROV


def draw_hw(pg, ctx, x, y, text, size, bound=None):
    """Handwritten value: hand font, ink colour, slight rotation -> hw_lines + handwriting box."""
    rng = ctx.rng
    from .canvas import KHMER
    if KHMER.search(text):
        p = ctx.font("hand", text, fallback=False)
    else:  # Latin value: real Latin handwriting fonts (the Khmer hand fonts' Latin glyphs look printed)
        cand = [f for f in ctx.F["en_hand"] if font_ok(f, text)]
        p = rng.choice(cand) if cand else None
    if not p:
        return None
    q = pg.text(x, y, text, p, int(size), ctx.hw_ink, angle=rng.uniform(-2.5, 2.5), kind="hw", bound=bound)
    if q:
        pg.add("handwriting", [q])
    return q


def b_field(pg, ctx, x, y, w, label=None, value=None, vkind=None, style=None, size=None, ymax=None):
    pg.begin("form_field")
    try:
        return _b_field(pg, ctx, x, y, w, label=label, value=value, vkind=vkind, style=style, size=size, ymax=ymax)
    finally:
        pg.end()


def _b_field(pg, ctx, x, y, w, label=None, value=None, vkind=None, style=None, size=None, ymax=None):
    """Label + input region (+ value). Returns y_end."""
    rng = ctx.rng
    sz = int(size or ctx.size)
    label = label or rng.choice(FIELD_LABELS[ctx.lang])
    label = label + ctx.colon() if rng.random() < 0.85 else label
    ymax = ymax or pg.H - 20
    if y + sz * 2.2 > ymax:
        return y
    lp = ctx.font("bold" if rng.random() < 0.3 else "body", label)
    lq = pg.text(x, y, label, lp, sz, ctx.fg)
    if not lq:
        return y
    lb = bbox([lq])
    style = style or rng.choice(["line", "line", "dots", "box", "comb", "none_line"])
    x0 = lb[2] + sz * 0.4
    x1 = x + w
    if x1 - x0 < sz * 3:
        pg.add("text", [lq])  # no room for an input region: the label is plain text
        return y + sz * 1.8
    base = y + sz * 1.25
    parts = [lq]
    d = pg.d
    lc = rng.choice([(90, 90, 90), (40, 40, 40), (120, 120, 140)])
    if style in ("line", "none_line"):
        d.line([x0, base, x1, base], fill=lc, width=1)
        parts.append((x0, base - 1, x1, base + 1))
    elif style == "dots":
        for dx in range(int(x0), int(x1), 7):
            d.ellipse([dx, base - 1, dx + 2, base + 1], fill=lc)
        parts.append((x0, base - 1, x1, base + 1))
    elif style == "box":
        d.rectangle([x0, y - sz * 0.35, x1, base + sz * 0.25], outline=lc, width=1)
        parts.append((x0, y - sz * 0.35, x1 + 1, base + sz * 0.25 + 1))
    else:  # comb boxes (one char per box)
        cw = sz * 1.1
        n = int(min(rng.randint(6, 20), (x1 - x0) // cw))
        x1 = x0 + n * cw
        d.rectangle([x0, y - sz * 0.3, x1, base + sz * 0.2], outline=lc, width=1)
        for i in range(1, n):
            d.line([x0 + i * cw, base - sz * 0.35, x0 + i * cw, base + sz * 0.2], fill=lc, width=1)
        parts.append((x0, y - sz * 0.3, x1 + 1, base + sz * 0.2 + 1))
    vkind = vkind or rng.choices(["hw", "print", "empty"], [0.55, 0.2, 0.25])[0]
    if vkind != "empty":
        v = value or value_for(ctx, label)
        bound = (x0, y - sz * 1.0, x1, base + sz * 0.6)
        if vkind == "hw":
            q = draw_hw(pg, ctx, x0 + rng.uniform(4, 30), y - sz * rng.uniform(0.3, 0.6), v, sz * rng.uniform(1.0, 1.35), bound)
        else:
            vp = ctx.font("mono" if rng.random() < 0.3 else "body", v)
            q = pg.text(x0 + rng.uniform(4, 20), y - sz * 0.05, v, vp, sz, ctx.fg, bound=bound)
        if q:
            parts.append(q)
    pg.add("form_field", parts)
    return base + sz * rng.uniform(0.9, 1.4)


FIELD_LABELS = {
    "km": ["ឈ្មោះ", "នាមត្រកូល និងនាមខ្លួន", "ភេទ", "ថ្ងៃខែឆ្នាំកំណើត", "ទីកន្លែងកំណើត", "អាសយដ្ឋាន", "លេខទូរស័ព្ទ", "មុខរបរ", "សញ្ជាតិ",
           "លេខអត្តសញ្ញាណប័ណ្ណ", "កាលបរិច្ឆេទ", "អ៊ីមែល", "ឈ្មោះឪពុក", "ឈ្មោះម្តាយ", "កម្រិតវប្បធម៌", "ស្ថានភាពគ្រួសារ", "លេខគណនី", "មូលហេតុ",
           "ឈ្មោះ / Name", "ថ្ងៃខែឆ្នាំកំណើត / Date of birth", "អាសយដ្ឋាន / Address", "លេខទូរស័ព្ទ / Phone", "អ៊ីមែល / Email"],
    "en": ["Name", "Full name", "First name", "Last name", "Date of birth", "Address", "City", "State", "ZIP code", "Phone", "Email",
           "Occupation", "Employer", "Social Security No.", "Signature date", "Account number", "Amount", "Spouse's name", "Date",
           "Apt./Suite", "Country", "Reason for application", "Emergency contact", "Policy number", "Member ID"],
}
CHECK_OPTS = {
    "km": [["ប្រុស", "ស្រី"], ["បាទ/ចាស", "ទេ"], ["នៅលីវ", "រៀបការ", "លែងលះ"], ["សាធារណៈ", "ឯកជន"], ["ថ្មី", "បន្ត", "ផ្លាស់ប្តូរ"],
           ["សាច់ប្រាក់", "ផ្ទេរតាមធនាគារ", "KHQR"], ["Male / ប្រុស", "Female / ស្រី"]],
    "en": [["Male", "Female", "Other"], ["Yes", "No"], ["Single", "Married filing jointly", "Married filing separately", "Head of household"],
           ["Cash", "Check", "Credit card"], ["New", "Renewal", "Change of address"], ["Full-time", "Part-time", "Contract"], ["Checking", "Savings"]],
}
CHECK_Q = {"km": ["ភេទ", "ស្ថានភាព", "ប្រភេទ", "វិធីទូទាត់", "ជម្រើស"], "en": ["Sex", "Filing status", "Type", "Payment method", "Status", "Account type"]}


def checkbox(pg, ctx, x, y, s, checked=None, round_=None):
    """Box glyph + optional tick -> checkbox label (box incl. tick ink). Returns bbox."""
    rng = ctx.rng
    round_ = rng.random() < 0.2 if round_ is None else round_
    lc = rng.choice([(30, 30, 30), (70, 70, 70), (20, 40, 100)])
    wd = rng.choice([1, 2, 2])
    d = pg.d
    pts = [(x, y, x + s + 1, y + s + 1)]
    if round_:
        d.ellipse([x, y, x + s, y + s], outline=lc, width=wd)
    else:
        d.rectangle([x, y, x + s, y + s], outline=lc, width=wd)
    checked = rng.random() < 0.5 if checked is None else checked
    if checked:
        kind = rng.choice(["tick", "x", "fill", "printed"])
        ink = ctx.hw_ink if kind in ("tick", "x") else lc
        if kind == "tick":
            p = [(x + s * 0.15, y + s * 0.5), (x + s * 0.42, y + s * 0.85), (x + s * rng.uniform(0.95, 1.3), y - s * rng.uniform(0.0, 0.35))]
            d.line(p, fill=ink, width=max(2, s // 7), joint="curve")
        elif kind == "x":
            e = s * rng.uniform(0.0, 0.2)
            p = [(x - e, y - e), (x + s + e, y + s + e)]
            d.line(p, fill=ink, width=max(2, s // 8))
            p2 = [(x + s + e, y - e), (x - e, y + s + e)]
            d.line(p2, fill=ink, width=max(2, s // 8))
            p = p + p2
        elif kind == "fill":
            if round_:
                d.ellipse([x + s * 0.25, y + s * 0.25, x + s * 0.75, y + s * 0.75], fill=lc)
            else:
                d.rectangle([x + s * 0.25, y + s * 0.25, x + s * 0.75, y + s * 0.75], fill=lc)
            p = []
        else:
            d.line([(x + s * 0.2, y + s * 0.55), (x + s * 0.42, y + s * 0.78), (x + s * 0.82, y + s * 0.22)], fill=lc, width=max(2, s // 6))
            p = []
        hw = max(2, s // 7) / 2 + 1
        for (px, py) in p:
            pts.append((px - hw, py - hw, px + hw, py + hw))
    return pg.add("checkbox", pts)


def b_check_group(pg, ctx, x, y, w, ymax=None, q=None, opts=None, size=None, vertical=None):
    """'Question: [ ] A  [x] B' -> form_field (whole group) + checkbox per box. Returns y_end."""
    rng = ctx.rng
    sz = int(size or ctx.size)
    ymax = ymax or pg.H - 20
    q = q or rng.choice(CHECK_Q[ctx.lang])
    opts = opts or rng.choice(CHECK_OPTS[ctx.lang])
    vertical = rng.random() < 0.25 if vertical is None else vertical
    if y + sz * (2 + (len(opts) * 1.6 if vertical else 0)) > ymax:
        return y
    parts = []
    t = q + ctx.colon()
    qq = pg.text(x, y, t, ctx.font("body", t), sz, ctx.fg)
    if not qq:
        return y
    parts.append(qq)
    cx = bbox([qq])[2] + sz * 0.8
    cy = y
    s = int(sz * rng.uniform(0.75, 0.95))
    one = rng.random() < 0.8
    pick = rng.randrange(len(opts))
    for i, o in enumerate(opts):
        p = ctx.font("body", o)
        if not p:
            continue
        ow = font(p, sz).getlength(o) + s + sz * 1.6
        if vertical:
            cx, cy = x + sz * 1.5, cy + sz * 1.6
        elif cx + ow > x + w:
            cx, cy = x + sz * 1.5, cy + sz * 1.6
        if cy + sz * 1.6 > ymax:
            break
        top = cy + (sz * 1.2 - s) / 2
        cb = checkbox(pg, ctx, cx, top, s, (i == pick) if one else None)
        oq = pg.text(cx + s + sz * 0.4, cy, o, p, sz, ctx.fg)
        parts += [cb, oq]
        cx += ow
    pg.add("form_field", parts)
    return cy + sz * 2.0


# --------------------------------------------------------------------------- signatures / stamps
def make_signature(ctx: Ctx):
    rng = ctx.rng
    r = rng.random()
    if r < 0.35 and R["sig_real"]:
        s = X.real_ink(R["sig_real"], rng, X.INKS)
        if s is not None:
            return s
    if r < 0.6:
        names = (ctx.kh_name(True)[0].title(), ctx.kh_name()[0]) if ctx.lang == "km" else (ctx.fake.name(), ctx.fake.last_name())
        nm = rng.choice(names)
        fonts = [p for p in ctx.pools["sig"] if font_ok(p, nm)]
        if fonts:
            A_names = A.NAMES_LATIN
            A.NAMES_LATIN = [nm]
            A_kh = A.NAMES_KHMER
            A.NAMES_KHMER = [nm]
            try:
                s = A.make_font_signature(rng, fonts, int(rng.uniform(60, 110)))
            finally:
                A.NAMES_LATIN, A.NAMES_KHMER = A_names, A_kh
            if s is not None:
                return s
    return A.make_signature(rng, int(rng.uniform(60, 120)), ink=ctx.hw_ink)


def make_stamp(ctx: Ctx, d=None, org=None):
    rng = ctx.rng
    if R["stamp_real"] and rng.random() < 0.3:
        s = X.real_ink(R["stamp_real"], rng, X.STAMP_INKS)
        if s is not None:
            return s
    fonts = ctx.pools["bold"] + ctx.pools["body"]
    if ctx.lang == "km":
        from .ctx import MINISTRIES
        top = org or rng.choice([m[0] for m in MINISTRIES] + [ctx.company()])
        bot = rng.choice(["KINGDOM OF CAMBODIA", "PHNOM PENH", None, rng.choice(MINISTRIES)[1].upper()[:30], "រាជធានីភ្នំពេញ"])
        centre = rng.sample(["ទទួលបាន", "បានទូទាត់", "អនុម័ត", "ច្បាប់ដើម", "បានពិនិត្យ", "ហត្ថលេខា"], 1) + [ctx.d(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2020, 2027)}")]
        rect_words = rng.sample(["ទទួលបាន", "ច្បាប់ចម្លងត្រឹមត្រូវ", "បានទូទាត់", "អនុម័ត", "ច្បាប់ដើម", "PAID", "RECEIVED", "បានពិនិត្យ"], rng.choice([1, 2]))
    else:
        top = org or ctx.fake.company().upper()[:34]
        bot = rng.choice([ctx.fake.city().upper(), "NOTARY PUBLIC", "CERTIFIED", None, f"EST. {rng.randint(1950, 2020)}"])
        centre = [rng.choice(["SEAL", "OFFICIAL", "APPROVED", "★"]), ctx.fake.date(pattern="%m/%d/%Y")]
        rect_words = rng.sample(["PAID", "RECEIVED", "APPROVED", "COPY", "CONFIDENTIAL", "VOID", "POSTED", "VERIFIED", "ENTERED"], 1)
        if rng.random() < 0.6:
            rect_words.append(ctx.fake.date(pattern="%b %d %Y").upper())
    if rng.random() < 0.6:
        return X.make_round_stamp(rng, fonts, int(d or rng.uniform(150, 250)), top, bot, centre)
    s = X.make_rect_stamp(rng, fonts, int((d or 220) * rng.uniform(0.9, 1.3)), rect_words)
    return s or X.make_round_stamp(rng, fonts, int(d or 200), top, bot, centre)


def b_sign(pg, ctx, x, y, w, ymax=None, role=None, name=None, date=False, stamp_p=0.5, align="c"):
    """Signature block in a column of width w: [date] role / signature (+stamp) / name."""
    rng = ctx.rng
    sz = ctx.size
    ymax = ymax or pg.H - 20
    cx = {"c": x + w / 2, "l": x, "r": x + w}[align]
    yy = y
    tq = []
    if date:
        t = (rng.choice(["រាជធានីភ្នំពេញ ", "ខេត្តសៀមរាប ", "ធ្វើនៅ "]) + ctx.date()) if ctx.lang == "km" else ctx.date()
        q = pg.text(cx, yy, t, ctx.font("body", t), int(sz * 0.9), ctx.fg, align=align)
        if q:
            pg.add("text", [q])
            yy = bbox([q])[3] + sz * 0.5
    role = role or (rng.choice(ROLES_KM) if ctx.lang == "km" else rng.choice(["Sincerely,", "Authorized Signature", "Approved by", "Best regards,", ctx.fake.job()[:30]]))
    p = ctx.font("moul" if ctx.lang == "km" and rng.random() < 0.5 else "bold", role)
    q = pg.text(cx, yy, role, p, int(sz * 0.95), ctx.fg, align=align)
    if q:
        pg.add("text", [q])
        yy = bbox([q])[3] + sz * 0.3
    gap = int(rng.uniform(2.4, 4.0) * sz)
    if yy + gap + sz * 2 > ymax:
        return yy
    sig = make_signature(ctx)
    s = min(1.0, (w * 0.85) / sig.width, gap * 1.1 / sig.height)
    if s < 1:
        sig = sig.resize((max(8, int(sig.width * s)), max(8, int(sig.height * s))), Image.LANCZOS)
    sx = (cx - sig.width / 2 if align == "c" else (x if align == "l" else x + w - sig.width)) + rng.uniform(-0.1, 0.1) * w
    if rng.random() < stamp_p:
        stp = make_stamp(ctx, d=min(gap * rng.uniform(1.3, 2.0), w * 0.7))
        if stp.width > w * 1.5 or stp.height > gap * 2.2:
            k = min(w * 1.5 / stp.width, gap * 2.2 / stp.height)
            stp = stp.resize((max(8, int(stp.width * k)), max(8, int(stp.height * k))), Image.LANCZOS)
        px = sx + rng.uniform(-0.6, 0.2) * stp.width
        pg.ink(stp, (px, yy + gap / 2 - stp.height / 2 + rng.uniform(-0.15, 0.15) * gap), "stamp")
    pg.ink(sig, (sx, yy + (gap - sig.height) / 2 + rng.uniform(-0.15, 0.1) * gap), "signature")
    yy += gap
    name = name or ctx.name()
    p = ctx.font("bold", name)
    q = pg.text(cx, yy, name, p, int(sz * 0.95), ctx.fg, align=align)
    if q:
        pg.add("text", [q])
        yy = bbox([q])[3]
    return yy + sz


ROLES_KM = ["នាយក", "ប្រធាន", "រដ្ឋមន្ត្រី", "អគ្គនាយក", "ប្រធានមន្ទីរ", "អ្នកស្នើសុំ", "អនុប្រធាន", "អ្នកទទួល", "ប្រធាននាយកដ្ឋាន", "ជ.រដ្ឋមន្ត្រី"]


# --------------------------------------------------------------------------- codes / logo / photo
def b_qr(pg, ctx, x, y, size, data=None, cls="qr_code", color=(0, 0, 0)):
    im = X.make_qr(ctx.rng, int(size), data, color)
    return pg.paste(im, (x, y), cls)


def b_barcode(pg, ctx, x, y, w, h, digits=True, code=None):
    rng = ctx.rng
    code = code or "".join(rng.choice("0123456789") for _ in range(rng.choice([10, 12, 13])))
    im = X.make_barcode(rng, int(w), int(h), code)
    b = pg.paste(im, (x, y), "barcode")
    if b and digits:
        t = code if rng.random() < 0.6 else " ".join([code[:1], code[1:7], code[7:]])
        p = ctx.font("mono", t) or ctx.font("latin", t)
        q = pg.text(x + w / 2, y + h + 4, t, p, int(max(12, h * 0.22)), (0, 0, 0), align="c")
        if q:
            pg.add("text", [q])
            return bbox([q])[3]
    return (b[3] if b else y)


def b_logo(pg, ctx, x, y, size, kind=None, wordmark=None):
    """Emblem / company logo, optionally with a wordmark beside it (wordmark = printed text line, inside the logo box)."""
    rng = ctx.rng
    kind = kind or rng.choice(["emblem", "corp"])
    im = X.make_emblem(rng, int(size)) if kind == "emblem" else X.make_corp_logo(rng, ctx.pools["latin_bold"], int(size))
    b = pg.paste(im, (x, y), None)
    if not b:
        return None
    from .canvas import hull_pts
    pts = [hull_pts(np.asarray(im)[..., 3] > 40, int(x), int(y))] if im.mode == "RGBA" else [to_pts((x, y, x + im.width, y + im.height))]
    if wordmark:
        p = ctx.font("bold", wordmark) or ctx.font("latin_bold", wordmark)
        q = pg.text(x + im.width + size * 0.15, y + im.height / 2 - size * 0.25, wordmark, p, int(size * 0.36), rng.choice(X.LOGO_COLS))
        if q:
            pts.append(to_pts([q]))
    pts = [p for p in pts if p is not None]
    return pg.add("logo", np.concatenate(pts)) if pts else None


def b_photo(pg, ctx, x, y, w, h, frame=True):
    im = X.make_portrait(ctx.rng, int(w), int(h))
    b = pg.paste(im, (x, y), "photo")
    if b and frame and ctx.rng.random() < 0.4:
        pg.d.rectangle([x - 2, y - 2, x + w + 1, y + h + 1], outline=(90, 90, 90), width=1)
    return b


def watermark(pg, ctx, text=None):
    """Very light diagonal text behind the content - intentionally NOT labelled (like human annotators)."""
    rng = ctx.rng
    text = text or (rng.choice(["ច្បាប់ចម្លង", "សម្ងាត់", "គំរូ", "COPY", "SAMPLE"]) if ctx.lang == "km" else rng.choice(["COPY", "DRAFT", "SAMPLE", "CONFIDENTIAL", "VOID", "ORIGINAL"]))
    p = ctx.font("bold", text)
    if not p:
        return
    sz = int(pg.W * rng.uniform(0.09, 0.15))
    f = font(p, sz)
    wl = f.getlength(text)
    m = Image.new("L", (int(wl) + sz, sz * 3), 0)
    ImageDraw.Draw(m).text((sz // 2, sz // 2), text, font=f, fill=int(255 * rng.uniform(0.06, 0.14)))
    m = m.rotate(rng.uniform(20, 45), expand=True, resample=Image.BICUBIC)
    col = rng.choice([(200, 30, 30), (120, 120, 120), (30, 60, 160)])
    pg.img.paste(col, (int(pg.W / 2 - m.width / 2), int(pg.H / 2 - m.height / 2)), m)
    pg.meta["watermark"] = text


def fill_flow(pg, ctx, x, y, w, ymax, kinds, max_blocks=30):
    """Fill a column with random blocks. kinds: [(name, weight)]."""
    rng = ctx.rng
    names, wts = zip(*kinds)
    sec = [0]
    fig_n = [rng.randint(1, 5)]
    tab_n = [rng.randint(1, 5)]
    for _ in range(max_blocks):
        if y > ymax - ctx.size * 2.5:
            break
        k = rng.choices(names, wts)[0]
        before = y
        if k == "para":
            y = b_para(pg, ctx, x, y, w, ymax)
        elif k == "heading":
            if y > ymax - ctx.size * 7:  # no orphan headings at the column bottom
                continue
            sec[0] += 1
            num = (f"{ctx.d(sec[0])}." if rng.random() < 0.5 else None)
            y = b_heading(pg, ctx, x, y, w, ymax, number=num)
            if y > before:
                y = b_para(pg, ctx, x, y, w, ymax)
        elif k == "list":
            y = b_list(pg, ctx, x, y, w, ymax)
        elif k == "table":
            tab_n[0] += 1
            y = b_table(pg, ctx, x, y, w, ymax, caption=rng.random() < 0.7, cap_num=tab_n[0])
        elif k == "figure":
            fig_n[0] += 1
            y = b_figure(pg, ctx, x, y, w, ymax, num=fig_n[0])
        elif k == "formula":
            y = b_formula(pg, ctx, x, y, w, ymax)
        elif k == "field":
            y = b_field(pg, ctx, x, y, w, ymax=ymax)
        elif k == "check":
            y = b_check_group(pg, ctx, x, y, w, ymax)
        if y - before < 3:
            if k in ("para", "heading"):
                break
    return y


# --------------------------------------------------------------------------- ornaments
def b_ornament(pg, ctx, cx, y, w, col=None, style=None):
    """Symmetric divider under a title / motto: thin rule with mirrored small shapes. Recorded as class "decoration"
    (not a text line). Returns its box (x0, y0, x1, y1) or None."""
    rng = ctx.rng
    d = pg.d
    col = col or (ctx.fg if rng.random() < 0.7 else ctx.accent)
    style = style or rng.choice(["diamond", "rings", "dots", "arrows", "wave", "bars"])
    w = max(40, int(w))
    t = rng.choice([1, 1, 2])
    h = max(6, int(w * rng.uniform(0.04, 0.09)))
    cx, y = int(cx), int(y)
    if style == "diamond":
        d.line([cx - w / 2, y, cx - h * 1.6, y], fill=col, width=t)
        d.line([cx + h * 1.6, y, cx + w / 2, y], fill=col, width=t)
        d.polygon([(cx, y - h), (cx + h, y), (cx, y + h), (cx - h, y)], fill=col)
        for sgn in (-1, 1):
            d.ellipse([cx + sgn * w / 2 - h * 0.5, y - h * 0.5, cx + sgn * w / 2 + h * 0.5, y + h * 0.5], outline=col, width=t)
    elif style == "rings":
        d.line([cx - w / 2, y, cx + w / 2, y], fill=col, width=t)
        for k in (-2, -1, 0, 1, 2):
            r = h * (0.9 if k == 0 else 0.55)
            d.ellipse([cx + k * h * 2.2 - r, y - r, cx + k * h * 2.2 + r, y + r], fill=pg.meta.get("paper", (255, 255, 255)), outline=col, width=t)
    elif style == "dots":
        for k in range(-5, 6):
            r = h * (0.7 if k == 0 else 0.35)
            d.ellipse([cx + k * w / 11 - r, y - r, cx + k * w / 11 + r, y + r], fill=col)
    elif style == "arrows":
        d.line([cx - w / 2, y, cx + w / 2, y], fill=col, width=t)
        for sgn in (-1, 1):
            tip = cx + sgn * w / 2
            d.polygon([(tip, y), (tip - sgn * h * 1.6, y - h * 0.8), (tip - sgn * h * 1.6, y + h * 0.8)], fill=col)
        d.polygon([(cx, y - h), (cx + h * 0.8, y), (cx, y + h), (cx - h * 0.8, y)], fill=col)
    elif style == "wave":
        pts = [(cx - w / 2 + i * w / 60, y + h * 0.6 * math.sin(i / 60 * 6 * math.pi)) for i in range(61)]
        d.line(pts, fill=col, width=t)
        d.ellipse([cx - h * 0.6, y - h * 0.6, cx + h * 0.6, y + h * 0.6], fill=col)
    else:  # bars
        d.line([cx - w / 2, y - h * 0.5, cx + w / 2, y - h * 0.5], fill=col, width=t)
        d.line([cx - w / 2 * 0.7, y + h * 0.5, cx + w / 2 * 0.7, y + h * 0.5], fill=col, width=t)
        d.polygon([(cx, y - h * 1.3), (cx + h * 0.7, y), (cx, y + h * 1.3), (cx - h * 0.7, y)], fill=col)
    box = (cx - w / 2 - h, y - h * 1.4, cx + w / 2 + h, y + h * 1.4)
    pg.add("decoration", box)
    return box


def b_stamp_at(pg, ctx, cx, cy, d=None, org=None):
    """A stamp centred near (cx, cy) - over the text that is already there (dates, signatures)."""
    rng = ctx.rng
    stp = make_stamp(ctx, d=d or rng.uniform(170, 260), org=org)
    pg.ink(stp, (cx - stp.width / 2 + rng.uniform(-0.15, 0.15) * stp.width, cy - stp.height / 2 + rng.uniform(-0.15, 0.15) * stp.height), "stamp")
    return stp
