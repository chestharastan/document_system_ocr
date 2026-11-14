"""Table engine v5: logical table spec -> rendered table + cell-level labels (docs/TABLES_CONTRACT.md §5).

A table is a *spec*: ``{"ncol": C, "header_rows": H, "rows": [{"kind": "hdr"|"body"|"group"|"total", "cells": [cell|None]*C}],
"align": [..C..], "col_frac": [..C..]|None, "row_header": bool}``. A cell is ``{"text", "rs", "cs", "align", "bold", "fill",
"max_lines", "header", "color", "role"}``; grid positions covered by a span hold ``None``. ``spec_from_data`` builds a
plain spec from header + rows, ``add_variety`` adds merged header groups (2-row headers), row-group header rows that span
the width, body cells spanning 2-3 rows, a totals row, empty cells, wrapped cells and a shaded row-header column.

``render_table`` lays the spec out (uneven column widths from the text, row heights from the wrapped lines), draws it in
one of the styles from ``choose_style`` (grid, hlines, outer border, none, zebra, thick header rule, dark header, dotted
leaders, Excel-like thin grey grid) and registers the labels: the ``table`` layout box (as before) plus an entry in
``pg.meta["tables"]`` with ``box, rows, cols, header_rows, borders, sheet, cells[{r,c,rs,cs,box,header,text,align,bold,
line_ids}]`` in page pixels. ``rows``/``cols`` are the ruling positions when that axis is ruled (vertical rules ->
cols, horizontal rules or zebra fills -> rows) and the mid-gaps between the printed text otherwise; a cell ``box`` is the
intersection of its row and column bands. generate.py warps all of it with the page homography. Every cell is its own
structure paragraph (``pg.begin("table")`` per cell) so ``struct`` groups wrapped lines per cell.
"""

from __future__ import annotations

import random
import re

import numpy as np

from .canvas import KHMER, bbox, to_pts, wrap
from .res import font

NUM_RE = re.compile(r"^[\s$€£៛]*[-+]?[\d,]*\.?\d+\s*%?\s*(៛|KHR|USD|\$)?$")


def cell(text="", rs=1, cs=1, align="l", bold=False, fill=None, max_lines=2, header=False, color=None, role=None):
    return {"text": text or "", "rs": int(rs), "cs": int(cs), "align": align, "bold": bool(bold), "fill": fill,
            "max_lines": int(max_lines), "header": bool(header), "color": color, "role": role}


def spec_from_data(hdr, rows, al=None, hdr_align=None, max_lines=2):
    """Plain spec: one header row + body rows (lists of strings)."""
    C = len(hdr)
    al = list(al) if al else ["l"] * C
    al = (al + ["l"] * C)[:C]
    out = [{"kind": "hdr", "cells": [cell(h, align=hdr_align or "c", bold=True, header=True, max_lines=max_lines) for h in hdr]}]
    for r in rows:
        out.append({"kind": "body", "cells": [cell(r[c] if c < len(r) else "", align=al[c], max_lines=max_lines) for c in range(C)]})
    return {"ncol": C, "header_rows": 1, "rows": out, "align": al, "col_frac": None, "row_header": False}


# --------------------------------------------------------------------------- structure variety
def body_rows(spec):
    return [i for i, r in enumerate(spec["rows"]) if r["kind"] == "body"]


def two_row_header(ctx, spec, labels=None, groups=None):
    """Merged header groups over 2-4 consecutive columns (row 0) with the old header texts as sub-headers (row 1);
    columns outside a group span both header rows."""
    rng = ctx.rng
    C = spec["ncol"]
    if C < 4 or spec["header_rows"] != 1:
        return False
    hdr = spec["rows"][0]["cells"]
    if groups is None:
        groups = []
        c = 1 + rng.randint(0, 1)
        while c < C - 1 and len(groups) < 2:
            span = rng.randint(2, min(4, C - c))
            if span >= 2 and rng.random() < 0.8:
                groups.append((c, span))
                c += span + rng.randint(0, 1)
            else:
                c += 1
    if not groups:
        return False
    labels = list(labels or [])
    top, sub = [None] * C, [None] * C
    grouped = set()
    for (c0, span) in groups:
        lab = labels.pop(0) if labels else ctx.phrase(4, 16)
        top[c0] = cell(lab, cs=span, align="c", bold=True, header=True)
        for k in range(c0, c0 + span):
            grouped.add(k)
            sub[k] = dict(hdr[k])
    for k in range(C):
        if k not in grouped:
            top[k] = dict(hdr[k], rs=2)
    spec["rows"][0] = {"kind": "hdr", "cells": top}
    spec["rows"].insert(1, {"kind": "hdr", "cells": sub})
    spec["header_rows"] = 2
    return True


def add_group_rows(ctx, spec, n=None, texts=None, fill=None, align="l"):
    """Row-group header rows: one bold cell spanning the full width, inserted before body rows (never last)."""
    rng = ctx.rng
    C = spec["ncol"]
    bi = body_rows(spec)
    if len(bi) < 3:
        return 0
    n = n or rng.randint(1, 3)
    cand = bi[:-1]
    pos = sorted(rng.sample(cand, min(n, len(cand))))
    texts = list(texts or [])
    k = 0
    for p in reversed(pos):
        if p == bi[0] and rng.random() < 0.5 and len(pos) > 1:
            continue  # a group row right under the header is fine but keep some tables without one
        t = texts.pop() if texts else ctx.phrase(6, 26)
        spec["rows"].insert(p, {"kind": "group", "cells": [cell(t, cs=C, align=align, bold=True, fill=fill, max_lines=1)] + [None] * (C - 1)})
        k += 1
    return k


def add_row_spans(ctx, spec, col=None, n=None, max_rs=3, same_text=True):
    """Body cells spanning 2-3 consecutive body rows in one column (category / name repeated -> merged)."""
    rng = ctx.rng
    C = spec["ncol"]
    rows = spec["rows"]
    if col is None:
        cands = [c for c in range(C) if spec["align"][c] == "l"] or [c for c in range(1, C)] or [0]
        col = rng.choice(cands)
    done = 0
    for _ in range(n or rng.randint(1, 3)):
        i = rng.randrange(len(rows))
        if rows[i]["kind"] != "body" or rows[i]["cells"][col] is None or rows[i]["cells"][col]["rs"] != 1 or rows[i]["cells"][col]["cs"] != 1:
            continue
        rs = rng.randint(2, max_rs)
        if not all(i + k < len(rows) and rows[i + k]["kind"] == "body" and rows[i + k]["cells"][col] is not None
                   and rows[i + k]["cells"][col]["rs"] == 1 and rows[i + k]["cells"][col]["cs"] == 1 for k in range(rs)):
            continue
        rows[i]["cells"][col]["rs"] = rs
        for k in range(1, rs):
            rows[i + k]["cells"][col] = None
        done += 1
    return done


def _num(t: str) -> float | None:
    t = t.strip()
    if not t or not NUM_RE.match(t):
        return None
    s = "".join(str("០១២៣៤៥៦៧៨៩".index(ch)) if ch in "០១២៣៤៥៦៧៨៩" else ch for ch in t)
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def add_totals(ctx, spec, label=None, span_label=None, numeric_cols=None, bold=True):
    """Totals row: label in the first text column (sometimes spanning the leading columns), sums in numeric columns."""
    rng = ctx.rng
    C = spec["ncol"]
    al = spec["align"]
    numeric = numeric_cols if numeric_cols is not None else [c for c in range(C) if al[c] == "r"]
    if not numeric:
        numeric = [c for c in range(1, C) if all((r["cells"][c] is None or _num(r["cells"][c]["text"]) is not None or not r["cells"][c]["text"])
                                                  for r in spec["rows"] if r["kind"] == "body")]
    if not numeric:
        return False
    label = label or (rng.choice(["សរុប", "សរុបរួម", "សរុបទាំងអស់"]) if ctx.lang == "km" else rng.choice(["Total", "TOTAL", "Grand Total", "Sum"]))
    first_num = min(numeric)
    lead = list(range(0, first_num)) if first_num > 0 else [0]
    cells: list = [None] * C
    span_label = (rng.random() < 0.4 and len(lead) > 1) if span_label is None else span_label
    if span_label:
        cells[lead[0]] = cell(label, cs=len(lead), align=rng.choice(["r", "c", "l"]), bold=bold)
    else:
        lc = lead[-1] if len(lead) > 1 else lead[0]
        for c in lead:
            cells[c] = cell("", align=al[c])
        cells[lc] = cell(label, align=al[lc] if al[lc] != "c" or lc == 0 else "l", bold=bold)
    for c in range(first_num, C):
        if c in numeric:
            vals = [_num(r["cells"][c]["text"]) for r in spec["rows"] if r["kind"] == "body" and r["cells"][c] is not None]
            vals = [v for v in vals if v is not None]
            ex = next((r["cells"][c]["text"] for r in spec["rows"] if r["kind"] == "body" and r["cells"][c] and r["cells"][c]["text"]), "")
            if vals and rng.random() < 0.85:
                s = sum(vals)
                txt = f"{s:,.2f}" if "." in ex else f"{int(round(s)):,}"
                if "$" in ex:
                    txt = "$" + txt
                if "៛" in ex:
                    txt += " ៛"
                if "%" in ex:
                    txt = f"{s / max(1, len(vals)):.1f}%"
                txt = ctx.d(txt)
            else:
                txt = ctx.money(1000, 90000) if rng.random() < 0.5 else ctx.d(str(rng.randint(100, 99999)))
            cells[c] = cell(txt, align="r", bold=bold)
        else:
            cells[c] = cell("", align=al[c])
    spec["rows"].append({"kind": "total", "cells": cells})
    return True


def add_empties(ctx, spec, frac, skip_cols=(0,)):
    rng = ctx.rng
    C = spec["ncol"]
    n = 0
    for r in spec["rows"]:
        if r["kind"] != "body":
            continue
        for c in range(C):
            if c in skip_cols:
                continue
            cl = r["cells"][c]
            if cl and cl["rs"] == 1 and cl["cs"] == 1 and cl["text"] and rng.random() < frac:
                cl["text"] = ""
                n += 1
    return n


def add_wraps(ctx, spec, col=None, frac=0.3, lo=28, hi=70, max_lines=3):
    """Longer phrases in one text column so some cells wrap to 2-3 lines."""
    rng = ctx.rng
    C = spec["ncol"]
    cands = [c for c in range(C) if spec["align"][c] == "l"]
    if not cands:
        return 0
    col = cands[0] if col is None else col
    n = 0
    for r in spec["rows"]:
        if r["kind"] != "body":
            continue
        cl = r["cells"][col]
        if cl and cl["text"] and rng.random() < frac:
            cl["text"] = ctx.phrase(lo, hi)
            cl["max_lines"] = max_lines
            n += 1
    return n


def add_variety(ctx, spec, kind="generic"):
    """Random structural variety for generated tables (the probabilities depend on the data kind)."""
    rng = ctx.rng
    C = spec["ncol"]
    P = {"generic": dict(h2=0.25, grp=0.3, span=0.25, tot=0.3, rowhdr=0.35, empty=0.5, wrap=0.3),
         "items": dict(h2=0.1, grp=0.15, span=0.1, tot=0.2, rowhdr=0.1, empty=0.3, wrap=0.3),
         "statement": dict(h2=0.15, grp=0.15, span=0.05, tot=0.3, rowhdr=0.0, empty=0.0, wrap=0.15)}.get(kind, {})
    if not P:
        return spec
    if rng.random() < P["h2"]:
        two_row_header(ctx, spec)
    if rng.random() < P["grp"]:
        add_group_rows(ctx, spec)
    if rng.random() < P["span"]:
        add_row_spans(ctx, spec)
    if rng.random() < P["tot"]:
        add_totals(ctx, spec)
    if rng.random() < P["rowhdr"]:
        spec["row_header"] = True
    if rng.random() < P["empty"]:
        add_empties(ctx, spec, rng.uniform(0.1, 0.25))
    if rng.random() < P["wrap"]:
        add_wraps(ctx, spec)
    for r in spec["rows"]:  # Khmer digits in some numeric cells of Khmer tables are already handled by ctx.d; nothing here
        pass
    return spec


# --------------------------------------------------------------------------- styles
LIGHT_FILLS = [(225, 230, 240), (235, 235, 235), (210, 225, 215), (240, 225, 200), (217, 225, 242), (226, 239, 218), (252, 228, 214), (230, 230, 230)]
DARK_FILLS = [(40, 60, 110), (30, 30, 30), (90, 20, 20), (20, 80, 60), (60, 60, 60), (31, 78, 121)]
ZEBRA_FILLS = [(242, 244, 248), (245, 245, 245), (236, 242, 236), (248, 242, 236), (240, 240, 240)]
STYLE_NAMES = ["grid", "grid", "grid", "hlines", "hlines", "outer", "none", "zebra", "thick_hdr", "dark_hdr", "dotted"]


def choose_style(rng: random.Random, name=None, hdr_fill=None, sheet=False, excel=False, ncol=99):
    """Style dict for render_table. ``name`` may be a legacy name (grid | hlines | none | zebra) or a v5 name."""
    st = {"name": name or rng.choice(STYLE_NAMES), "pad_x": rng.uniform(0.3, 0.5), "pad_y": rng.uniform(0.28, 0.55),
          "hdr_scale": rng.uniform(1.0, 1.15), "lw": rng.choice([1, 1, 2]), "col": (rng.randint(20, 110),) * 3,
          "zebra": False, "zebra_fill": rng.choice(ZEBRA_FILLS), "hdr_fill": None, "thick_hdr": False, "dotted": False,
          "double_total": False, "group_fill": rng.choice([None, (246, 246, 246), (235, 238, 242)]), "rowhdr_fill": rng.choice([None, (245, 245, 245), (238, 240, 246)]),
          "total_fill": rng.choice([None, None, (240, 240, 240)]), "excel": False, "fill_width": True, "hdr_rule": rng.random() < 0.7,
          "outer_rule": rng.random() < 0.5, "hdr_align": None, "font_role": None, "lh": None, "letters_fill": None}
    if st["name"] == "dotted" and ncol > 3:
        st["name"] = "none"
    n = st["name"]
    st["rules"] = {"grid": "grid", "hlines": "hlines", "outer": "outer", "none": "none", "zebra": rng.choice(["none", "none", "hlines"]),
                   "thick_hdr": "hlines", "dark_hdr": rng.choice(["grid", "hlines"]), "dotted": "none"}[n]
    if n == "zebra":
        st["zebra"] = True
    if n == "thick_hdr":
        st["thick_hdr"] = True
        st["lw"] = 1
    if n == "dotted":
        st["dotted"] = True
    if hdr_fill is not None:
        st["hdr_fill"] = hdr_fill
    elif n == "dark_hdr":
        st["hdr_fill"] = rng.choice(DARK_FILLS)
    elif rng.random() < 0.6:
        st["hdr_fill"] = rng.choice(LIGHT_FILLS if rng.random() < 0.85 else DARK_FILLS)
    if st["rules"] != "grid" and rng.random() < 0.5:
        st["double_total"] = True
    if excel:
        st.update(name="excel", rules="grid", excel=True, lw=1, col=(rng.randint(175, 215),) * 3, zebra=rng.random() < 0.15, thick_hdr=False, dotted=False,
                  hdr_fill=rng.choice([None, (217, 225, 242), (226, 239, 218), (242, 242, 242), (221, 235, 247)]) if rng.random() < 0.7 else None,
                  letters_fill=rng.choice([(230, 230, 230), (238, 238, 238), (225, 225, 225)]), pad_x=rng.uniform(0.25, 0.4), pad_y=rng.uniform(0.22, 0.4),
                  hdr_scale=1.0, double_total=False, group_fill=rng.choice([None, (242, 242, 242)]))
    if sheet:
        st["fill_width"] = True
    return st


def borders_label(st) -> str:
    if st["rules"] == "grid":
        return "grid"
    if st["rules"] == "hlines":
        return "hlines"
    if st["zebra"]:
        return "zebra"
    return "none"


# --------------------------------------------------------------------------- layout + render
def _join(lines: list[str]) -> str:
    out = ""
    for s in lines:
        if not out:
            out = s
        elif KHMER.search(out[-1:]) or KHMER.search(s[:1]):
            out += s
        else:
            out += " " + s
    return out


def render_table(pg, ctx, x, y, w, ymax, spec, st, sz, sheet=False, min_body=1, x_offset=None):
    """Lay out, draw and label one table. Returns (y_below, label dict) or (y, None) if nothing fitted."""
    rng = ctx.rng
    d = pg.d
    C = spec["ncol"]
    H = spec["header_rows"]
    rows = [dict(r, cells=[dict(c) if c else None for c in r["cells"]]) for r in spec["rows"]]
    km = ctx.lang == "km" or any(KHMER.search(c["text"]) for r in rows for c in r["cells"] if c and c["text"])
    lh = sz * (st.get("lh") or (1.5 if km else 1.3))
    pad_x, pad_y = sz * st["pad_x"], sz * st["pad_y"]
    fg = ctx.fg

    # ---- fonts
    for r in rows:
        for c in r["cells"]:
            if c and c["text"]:
                role = c["role"] or st.get("font_role") or ("bold" if (c["bold"] or c["header"]) else "body")
                c["font"] = ctx.font(role, c["text"]) or ctx.font("body", c["text"])
                if not c["font"]:
                    c["text"] = ""
    # ---- column widths
    if spec.get("col_frac"):
        fr = np.array(spec["col_frac"], np.float64)
        ws = fr / fr.sum() * w
        xt = x
    else:
        nat = np.full(C, sz * 2.0)
        for r in rows:
            for ci, c in enumerate(r["cells"]):
                if c and c["text"] and c["cs"] == 1:
                    f = font(c["font"], sz)
                    cap = sz * (9 if c["header"] else (14 if c["max_lines"] > 1 else 30))
                    nat[ci] = max(nat[ci], min(f.getlength(c["text"]), cap))
        nat = nat + 2 * pad_x
        nat = nat * np.array([rng.uniform(0.92, 1.12) for _ in range(C)])
        total = nat.sum()
        if total <= w:
            if st["fill_width"] or sheet or rng.random() < 0.8:
                wt = nat * np.array([2.0 if spec["align"][c] == "l" else 1.0 for c in range(C)])
                ws = nat + (w - total) * wt / wt.sum()
                xt = x
            else:
                ws = nat
                xt = x + (w - total) * rng.choice([0, 0, 0.5])
        else:
            ws = nat * (w / total)
            mn = sz * 2.2
            for _ in range(3):
                low = ws < mn
                if not low.any() or low.all():
                    break
                deficit = (mn - ws[low]).sum()
                ws[low] = mn
                ws[~low] -= deficit * ws[~low] / ws[~low].sum()
            xt = x
    xs = np.concatenate([[0], np.cumsum(ws)]) + float(xt)
    xs = np.round(xs).astype(int)
    xs[-1] = min(xs[-1], int(x + w) + 1)

    # ---- wrap cell text, row heights
    for r in rows:
        for ci, c in enumerate(r["cells"]):
            if c and c["text"]:
                avail = xs[ci + c["cs"]] - xs[ci] - 2 * pad_x
                c["lines"] = wrap(c["text"], font(c["font"], sz), max(sz, avail))[:max(1, c["max_lines"])]
            elif c:
                c["lines"] = []
    R = len(rows)
    hs = np.zeros(R)
    for ri, r in enumerate(rows):
        n = max([len(c["lines"]) for c in r["cells"] if c and c["rs"] == 1] + [1])
        hs[ri] = n * lh + 2 * pad_y
        if r["kind"] == "hdr":
            hs[ri] *= st["hdr_scale"]
    for ri, r in enumerate(rows):  # spans need enough height
        for c in r["cells"]:
            if c and c["rs"] > 1:
                need = max(1, len(c["lines"])) * lh + 2 * pad_y
                have = hs[ri:ri + c["rs"]].sum()
                if have < need:
                    hs[min(R - 1, ri + c["rs"] - 1)] += need - have
    ys = np.concatenate([[0], np.cumsum(hs)]) + float(y)
    # ---- cut to ymax
    keep = int(np.searchsorted(ys, ymax + 0.5, side="right") - 1)  # number of rows that fit
    keep = min(keep, R)
    while keep > 0 and rows[keep - 1]["kind"] in ("group", "hdr") and keep > H:
        keep -= 1
    if keep < H + min_body or keep < 2:
        return y, None
    if keep < R:
        rows = rows[:keep]
        R = keep
        ys = ys[:R + 1]
        for ri, r in enumerate(rows):
            for c in r["cells"]:
                if c and ri + c["rs"] > R:
                    c["rs"] = R - ri
    ys = np.round(ys).astype(int)

    # ---- logical cells + cover map
    cover = [[None] * C for _ in range(R)]
    cells = []
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r["cells"]):
            if c is None:
                continue
            rs, cs = min(c["rs"], R - ri), min(c["cs"], C - ci)
            c["rs"], c["cs"] = rs, cs
            c["r"], c["c"] = ri, ci
            c["kind"] = r["kind"]
            for a in range(ri, ri + rs):
                for b in range(ci, ci + cs):
                    cover[a][b] = (ri, ci)
            cells.append(c)
    for ri in range(R):  # positions nobody owns (data rows shorter than C): empty cells
        for ci in range(C):
            if cover[ri][ci] is None:
                c = cell("", align=spec["align"][ci])
                c.update(r=ri, c=ci, kind=rows[ri]["kind"], lines=[], font=None)
                cover[ri][ci] = (ri, ci)
                cells.append(c)
                rows[ri]["cells"][ci] = c
    cells.sort(key=lambda c: (c["r"], c["c"]))

    # ---- fills
    ink = []
    body_i = 0
    row_fill = {}
    for ri, r in enumerate(rows):
        k = r["kind"]
        if k == "hdr":
            row_fill[ri] = st["hdr_fill"]
        elif k == "group":
            row_fill[ri] = st["group_fill"]
        elif k == "total":
            row_fill[ri] = st["total_fill"]
        else:
            row_fill[ri] = st["zebra_fill"] if (st["zebra"] and body_i % 2 == 1) else None
            body_i += 1
    for c in cells:
        box = (xs[c["c"]], ys[c["r"]], xs[c["c"] + c["cs"]], ys[c["r"] + c["rs"]])
        fl = c["fill"] or row_fill.get(c["r"])
        if fl is None and spec.get("row_header") and c["c"] == 0 and c["kind"] == "body":
            fl = st["rowhdr_fill"]
        if fl is None and st["excel"] and spec.get("excel_letters") and (c["r"] == 0 or c["c"] == 0):
            fl = st["letters_fill"]
        if fl is not None:
            d.rectangle([box[0], box[1], box[2], box[3]], fill=fl)
            c["_fill"] = fl
            ink.append(box)
    # ---- rules
    lw, col = st["lw"], st["col"]
    rules = st["rules"]

    def hseg(k, c0, c1, width=None):
        yy = ys[k]
        d.line([xs[c0], yy, xs[c1], yy], fill=col, width=width or lw)
        ink.append((xs[c0], yy, xs[c1], yy + (width or lw)))

    def vseg(k, r0, r1):
        xx = xs[k]
        d.line([xx, ys[r0], xx, ys[r1]], fill=col, width=lw)
        ink.append((xx, ys[r0], xx + lw, ys[r1]))

    total_rows = [ri for ri, r in enumerate(rows) if r["kind"] == "total"]
    for k in range(R + 1):
        want = rules in ("grid", "hlines") or (rules == "outer" and k in (0, R)) or (k in (0, R) and st["outer_rule"] and rules == "none") \
            or (k == H and st["hdr_rule"] and rules in ("none", "outer")) or (st["zebra"] and k in (0, R) and st["outer_rule"])
        thick = st["thick_hdr"] and k == H
        dbl = st["double_total"] and k in total_rows and rules != "grid"
        if not (want or thick or dbl):
            continue
        c0 = 0
        while c0 < C:
            if 0 < k < R and cover[k - 1][c0] == cover[k][c0]:
                c0 += 1
                continue
            c1 = c0 + 1
            while c1 < C and not (0 < k < R and cover[k - 1][c1] == cover[k][c1]):
                c1 += 1
            if dbl:
                d.line([xs[c0], ys[k] - 3, xs[c1], ys[k] - 3], fill=col, width=1)
                ink.append((xs[c0], ys[k] - 3, xs[c1], ys[k] - 2))
            hseg(k, c0, c1, width=(3 if thick else None))
            c0 = c1
    for k in range(C + 1):
        want = rules == "grid" or (rules == "outer" and k in (0, C))
        if not want:
            continue
        r0 = 0
        while r0 < R:
            if 0 < k < C and cover[r0][k - 1] == cover[r0][k]:
                r0 += 1
                continue
            r1 = r0 + 1
            while r1 < R and not (0 < k < C and cover[r1][k - 1] == cover[r1][k]):
                r1 += 1
            vseg(k, r0, r1)
            r0 = r1
    # ---- text
    for c in cells:
        c["line_ids"] = []
        c["quads"] = []
        if not c["lines"]:
            continue
        x0, x1 = xs[c["c"]], xs[c["c"] + c["cs"]]
        y0, y1 = ys[c["r"]], ys[c["r"] + c["rs"]]
        a = c["align"]
        if c["kind"] == "hdr" and st.get("hdr_align"):
            a = st["hdr_align"]
        n = len(c["lines"])
        fl = c.get("_fill")
        dark = fl is not None and sum(fl) < 300
        color = (250, 250, 250) if dark else (c["color"] or fg)
        top = y0 + ((y1 - y0) - lh * n) / 2 - sz * 0.05
        pg.begin("table")
        for k, s in enumerate(c["lines"]):
            cx = x0 + ((x1 - x0) / 2 if a == "c" else ((x1 - x0) - pad_x if a == "r" else pad_x))
            q = pg.text(cx, top + k * lh, s, c["font"], sz, color, align=a, bound=(x0 + 1, y0 - sz * 0.3, x1 - 1, y1 + sz * 0.3))
            if q:
                c["line_ids"].append(len(pg.lines) - 1)
                c["quads"].append(q)
                ink.append(q)
        pg.end()
        c["text"] = _join([pg.line_texts[i] for i in c["line_ids"]])
    # ---- dotted leaders (menu / price-list look): between the first text column and the last column
    if st["dotted"]:
        tc = next((ci for ci in range(C) if spec["align"][ci] == "l"), 0)
        for ri, r in enumerate(rows):
            if r["kind"] != "body":
                continue
            a, b = r["cells"][tc], r["cells"][C - 1]
            if not (a and b and a["quads"] and b["quads"]) or tc == C - 1:
                continue
            xa = bbox(a["quads"])[2] + sz * 0.5
            xb = bbox(b["quads"])[0] - sz * 0.5
            yy = (ys[ri] + ys[ri + 1]) / 2 + sz * 0.3
            for dx in np.arange(xa, xb, sz * 0.35):
                d.ellipse([dx, yy - 1, dx + 2, yy + 1], fill=(90, 90, 90))
    # ---- layout label
    x_ruled = rules == "grid"
    y_ruled = rules in ("grid", "hlines") or st["zebra"]
    rect_like = rules in ("grid", "hlines", "outer") or st["zebra"] or any(c.get("_fill") for c in cells) or st["thick_hdr"]
    if rect_like:
        box = (float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1))
        pg.add("table", box)
    else:
        box = bbox(ink) if ink else None
        if box is None:
            return y, None
        pg.add("table", ink)
    # ---- bands
    row_lo = [None] * R
    row_hi = [None] * R
    col_lo = [None] * C
    col_hi = [None] * C
    for c in cells:
        if not c["quads"]:
            continue
        _ink_extents(c, bbox(c["quads"]), row_lo, row_hi, col_lo, col_hi)
    rb = _bands(R, ys, row_lo, row_hi, box[1], box[3], y_ruled)
    cb = _bands(C, xs, col_lo, col_hi, box[0], box[2], x_ruled)
    out_cells = []
    for c in cells:
        cb0, cb1 = cb[c["c"]], cb[c["c"] + c["cs"] - 1]
        rb0, rb1 = rb[c["r"]], rb[c["r"] + c["rs"] - 1]
        out_cells.append({"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": [cb0[0], rb0[0], cb1[1], rb1[1]],
                          "header": c["kind"] == "hdr", "text": c["text"], "align": c["align"], "bold": bool(c["bold"] or c["kind"] in ("hdr", "total")),
                          "line_ids": c["line_ids"], "kind": c["kind"]})
    lab = {"box": [round(v, 1) for v in box], "rows": rb, "cols": cb, "header_rows": H, "borders": borders_label(st), "sheet": bool(sheet),
           "cells": out_cells, "style": st["name"], "excel": bool(spec.get("excel_letters"))}
    pg.meta.setdefault("tables", []).append(lab)
    return float(ys[-1]) + 1, lab


def _ink_extents(c, b, row_lo, row_hi, col_lo, col_hi):
    """Feed one cell's ink bbox into the per-row / per-column ink extents. A spanning cell's ink counts for the first
    row / column it covers (top / left edge) and the last one (bottom / right edge), so a merged label that is right-
    aligned in its merged area still lies inside the mid-gap bands."""
    r0, r1 = c["r"], c["r"] + c["rs"] - 1
    c0, c1 = c["c"], c["c"] + c["cs"] - 1
    row_lo[r0] = b[1] if row_lo[r0] is None else min(row_lo[r0], b[1])
    row_hi[r1] = b[3] if row_hi[r1] is None else max(row_hi[r1], b[3])
    col_lo[c0] = b[0] if col_lo[c0] is None else min(col_lo[c0], b[0])
    col_hi[c1] = b[2] if col_hi[c1] is None else max(col_hi[c1], b[2])


def _bands(n, pos, lo_ink, hi_ink, lo, hi, ruled):
    """n bands between lo and hi: ruling positions when ruled, otherwise mid-gaps between the ink of neighbours."""
    b = [float(lo)]
    for i in range(1, n):
        if ruled or hi_ink[i - 1] is None or lo_ink[i] is None:
            b.append(float(pos[i]))
        else:
            b.append((hi_ink[i - 1] + lo_ink[i]) / 2.0)
    b.append(float(hi))
    for i in range(1, len(b)):
        b[i] = max(b[i], b[i - 1])
    b = [min(max(v, lo), hi) for v in b]
    return [[round(b[i], 1), round(b[i + 1], 1)] for i in range(n)]


def register_table(pg, xs, ys, cells, header_rows, borders="grid", sheet=False, box=None, x_ruled=True, y_ruled=True, style="legacy"):
    """Label a table drawn by other code (e.g. templates_km_v4.group_table) without changing how it is drawn.
    cells: [{r, c, rs, cs, text, header, align, bold, line_ids}] (line_ids into pg.lines); xs / ys: grid positions."""
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    R, C = len(ys) - 1, len(xs) - 1
    box = box or (xs[0], ys[0], xs[-1] + 1, ys[-1] + 1)
    row_lo, row_hi, col_lo, col_hi = [None] * R, [None] * R, [None] * C, [None] * C
    for c in cells:
        qs = [pg.lines[i] for i in c.get("line_ids", []) if i < len(pg.lines)]
        if not qs:
            continue
        _ink_extents(c, bbox(qs), row_lo, row_hi, col_lo, col_hi)
    rb = _bands(R, ys, row_lo, row_hi, box[1], box[3], y_ruled)
    cb = _bands(C, xs, col_lo, col_hi, box[0], box[2], x_ruled)
    out = []
    for c in sorted(cells, key=lambda c: (c["r"], c["c"])):
        out.append({"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": [cb[c["c"]][0], rb[c["r"]][0], cb[c["c"] + c["cs"] - 1][1], rb[c["r"] + c["rs"] - 1][1]],
                    "header": bool(c.get("header")), "text": c.get("text", ""), "align": c.get("align", "l"), "bold": bool(c.get("bold")),
                    "line_ids": list(c.get("line_ids", [])), "kind": c.get("kind", "hdr" if c.get("header") else "body")})
    lab = {"box": [round(float(v), 1) for v in box], "rows": rb, "cols": cb, "header_rows": int(header_rows), "borders": borders, "sheet": bool(sheet),
           "cells": out, "style": style, "excel": False}
    pg.meta.setdefault("tables", []).append(lab)
    return lab


def check_label(lab, n_lines=None):
    """Structural sanity checks for one table label; returns a list of problems (empty = ok)."""
    bad = []
    R, C = len(lab["rows"]), len(lab["cols"])
    seen = {}
    for c in lab["cells"]:
        if c["rs"] < 1 or c["cs"] < 1 or c["r"] < 0 or c["c"] < 0 or c["r"] + c["rs"] > R or c["c"] + c["cs"] > C:
            bad.append(f"cell out of grid {c['r']},{c['c']} rs={c['rs']} cs={c['cs']} (R={R},C={C})")
            continue
        for a in range(c["r"], c["r"] + c["rs"]):
            for b in range(c["c"], c["c"] + c["cs"]):
                if (a, b) in seen:
                    bad.append(f"overlap at {a},{b}")
                seen[(a, b)] = 1
        if n_lines is not None and any(i >= n_lines or i < 0 for i in c["line_ids"]):
            bad.append(f"line id out of range in cell {c['r']},{c['c']}")
        if (c["text"] == "") != (len(c["line_ids"]) == 0):
            bad.append(f"text/line_ids mismatch in cell {c['r']},{c['c']}")
    if len(seen) != R * C:
        bad.append(f"grid not covered: {len(seen)} of {R * C}")
    # band order: compare centres (after a rotation / keystone warp the axis-aligned bounding bands overlap)
    for i in range(1, R):
        if sum(lab["rows"][i]) < sum(lab["rows"][i - 1]) - 1.0:
            bad.append("rows not sorted")
            break
    for i in range(1, C):
        if sum(lab["cols"][i]) < sum(lab["cols"][i - 1]) - 1.0:
            bad.append("cols not sorted")
            break
    if not (0 <= lab["header_rows"] <= R):
        bad.append("header_rows out of range")
    return bad
