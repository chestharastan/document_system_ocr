"""Page-level structure the layout model does not give us (yet):

* ``find_decorations``   an ornament / divider the text-line detector returned as a "text" line: mirror-symmetric ink,
                         read with low confidence or much smaller than the page's text -> class "decoration" (crop).
* ``snap_crop_boxes``    figure / logo / stamp ... boxes grown to the connected ink they cut through (a tight box on a
                         round emblem clips its ring) plus a small margin.
* ``relabel_by_position``  layout "text" blocks above the body are the letterhead (header), a centred short block
                         just above the body is the title, blocks under the body are footer / page number; a small
                         square figure at the top of the page is a logo.
* ``measure_fonts``      per line: the dense ink core (consonant body) height and the baseline; per block: one font
                         size for all its lines - the font model's em size per line (font_cnn v2, ``font.em_px``)
                         when available, else median core x K - snapped to the page's few size levels so the body
                         text of every paragraph renders at the same size.
* ``page_layout``        margins (left / right / top / bottom), header / body / footer zones and each block's
                         horizontal alignment, for rebuilding the page as a document.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

K_KHMER = 1.4   # CSS font-size / consonant-body height, Khmer OS fonts (Noto Sans Khmer: 1.7)
K_LATIN = 1.9   # CSS font-size / x-height..cap-height core, Latin fonts
CROP_SNAP = ("figure", "logo", "photo", "qr_code", "barcode", "stamp")
_DIGITS = re.compile(r"^[\d០-៩\s./\-–—()|]+$")


def _bbox_int(b, W, H):
    x0, y0, x1, y1 = b
    return max(0, int(x0)), max(0, int(y0)), min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))


def _is_khmer_text(t: str) -> bool:
    k = sum(1 for c in t if "ក" <= c <= "៿")
    return k >= max(1, 0.4 * sum(1 for c in t if not c.isspace()))


# ---------------------------------------------------------------------------------------------- decorations

def symmetry(ink: np.ndarray, box, W, H) -> float:
    """IoU of the (slightly dilated) ink around ``box`` with its mirror image about the centre of the ink extent (best
    of small shifts). Text lines: 0.4-0.6, an ornament / rule: > 0.7. The box is widened by two line heights so the
    thin side rules of a divider the detector cut off are included."""
    x0, y0, x1, y1 = box
    h = max(1.0, y1 - y0)
    x0, y0, x1, y1 = _bbox_int((x0 - h, y0 - 2, x1 + h, y1 + 2), W, H)
    m = ink[y0:y1, x0:x1] > 0
    if m.size == 0 or not m.any():
        return 0.0
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cols = np.where(m.any(0))[0]
    cx = (cols[0] + cols[-1]) / 2
    xs = np.arange(m.shape[1])
    best = 0.0
    for s in range(-6, 7):
        xm = np.round(2 * (cx + s) - xs).astype(int)
        ok = (xm >= 0) & (xm < m.shape[1])
        a, b = m[:, xs[ok]], m[:, xm[ok]]
        inter, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
        if union:
            best = max(best, inter / union)
    return float(best)


def is_decoration(ink: np.ndarray, ln: dict, page_line_h: float, W: int, H: int) -> bool:
    """A line is a decoration when it reads badly / is far smaller than the page's text AND its ink is mirror-symmetric
    (ornament) or a single rule; a tiny garbage line sitting on the page's centre line is one too."""
    x0, y0, x1, y1 = ln["bbox"]
    h, w = y1 - y0, x1 - x0
    if w < 2 * h:
        return False
    conf = float(ln.get("confidence", 1.0))
    text = (ln.get("text") or "").strip()
    small = page_line_h > 0 and h < 0.55 * page_line_h
    garbage = bool(text) and not _is_khmer_text(text) and not re.search(r"[A-Za-z]{3,}", text) and not _DIGITS.match(text)
    if not (conf < 0.75 or small or garbage):
        return False
    if symmetry(ink, ln["bbox"], W, H) >= 0.7:
        return True
    centred = abs((x0 + x1) / 2 - W / 2) < 0.05 * W
    if small and centred and (garbage or conf < 0.6):
        return True
    # a plain horizontal rule: a few inked rows spanning the box
    xi0, yi0, xi1, yi1 = _bbox_int(ln["bbox"], W, H)
    m = ink[yi0:yi1, xi0:xi1] > 0
    if m.size and small:
        rows = m.sum(1)
        return bool((rows >= 0.85 * m.shape[1]).sum() >= 1 and (rows > 0.1 * m.shape[1]).sum() <= max(3, 0.25 * m.shape[0]))
    return False


def find_decorations(ink: np.ndarray, blocks: list, text_classes: set, block_factory) -> list:
    """Move decoration lines out of text blocks into "decoration" blocks (crop class). Returns the new block list."""
    H, W = ink.shape[:2]
    hs = [l["bbox"][3] - l["bbox"][1] for b in blocks if b["cls"] in text_classes for l in b["lines"]]
    page_h = float(np.median(hs)) if hs else 0.0
    out, extra = [], []
    for b in blocks:
        if b["cls"] not in text_classes or b["cls"] in ("table", "form_field", "handwriting") or not b["lines"]:
            out.append(b)
            continue
        deco = [l for l in b["lines"] if is_decoration(ink, l, page_h, W, H)]
        if not deco:
            out.append(b)
            continue
        deco_ids = {id(l) for l in deco}  # identity: line dicts hold numpy quads, == on them is ambiguous
        keep = [l for l in b["lines"] if id(l) not in deco_ids]
        for l in deco:
            d = block_factory("decoration", l["bbox"], b["score"], "decoration", source="heuristic")
            extra.append(d)
        if keep:
            b["lines"] = keep
            b["box"] = [min(l["bbox"][0] for l in keep), min(l["bbox"][1] for l in keep),
                        max(l["bbox"][2] for l in keep), max(l["bbox"][3] for l in keep)]
            out.append(b)
    return out + extra


# ---------------------------------------------------------------------------------------------- crop boxes

def snap_crop_box(ink: np.ndarray, box, W: int, H: int, grow: float = 0.15, margin: float = 0.03):
    """Grow ``box`` to cover the connected ink components that lie mostly (>= 50 % of their area) inside it, then add
    a margin. Growth is capped at ``grow`` x the box size per side so a neighbouring paragraph cannot pull it."""
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    gx, gy = grow * bw + 6, grow * bh + 6
    wx0, wy0, wx1, wy1 = _bbox_int((x0 - gx, y0 - gy, x1 + gx, y1 + gy), W, H)
    if wx1 - wx0 < 4 or wy1 - wy0 < 4:
        return list(box)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink[wy0:wy1, wx0:wx1], 8)
    nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    for i in range(1, n):
        cx, cy, cw, ch, area = (int(v) for v in st[i])
        cx, cy = cx + wx0, cy + wy0
        if area < 8:
            continue
        inter_w = min(cx + cw, x1) - max(cx, x0)
        inter_h = min(cy + ch, y1) - max(cy, y0)
        if inter_w <= 0 or inter_h <= 0 or inter_w * inter_h < 0.5 * cw * ch:
            continue  # mostly outside: another object
        nx0, ny0, nx1, ny1 = min(nx0, cx), min(ny0, cy), max(nx1, cx + cw), max(ny1, cy + ch)
    mx, my = margin * (nx1 - nx0) + 3, margin * (ny1 - ny0) + 3
    return [max(0.0, nx0 - mx), max(0.0, ny0 - my), min(float(W), nx1 + mx), min(float(H), ny1 + my)]


def snap_crop_boxes(ink: np.ndarray, blocks: list, text_classes: set = frozenset()) -> int:
    """Snap every figure / logo / stamp box; the grown box is cut back where it would run into a text line of
    another block (the letterhead under an emblem)."""
    H, W = ink.shape[:2]
    lines = [ln["bbox"] for o in blocks if o["cls"] in text_classes for ln in o["lines"]]
    n = 0
    for b in blocks:
        if b["cls"] not in CROP_SNAP:
            continue
        ox0, oy0, ox1, oy1 = b["box"]
        nb = snap_crop_box(ink, b["box"], W, H)
        for lx0, ly0, lx1, ly1 in lines:
            if lx0 >= nb[2] or lx1 <= nb[0] or ly0 >= nb[3] or ly1 <= nb[1]:
                continue  # no overlap
            if ly0 >= oy1 - 1:        # text below the original box: stop above it
                nb[3] = min(nb[3], max(oy1, ly0 - 1))
            elif ly1 <= oy0 + 1:      # text above
                nb[1] = max(nb[1], min(oy0, ly1 + 1))
            elif lx0 >= ox1 - 1:      # text right
                nb[2] = min(nb[2], max(ox1, lx0 - 1))
            elif lx1 <= ox0 + 1:      # text left
                nb[0] = max(nb[0], min(ox0, lx1 + 1))
        if nb != b["box"]:
            b["box"], n = nb, n + 1
    return n


# ---------------------------------------------------------------------------------------------- relabelling

def relabel_logos(blocks: list, W: int, H: int) -> None:
    """A small, roughly square figure in the top part of the page is an emblem / logo; a page-wide strip thinner than
    a text line is a rule (decoration)."""
    for b in blocks:
        if b["cls"] != "figure":
            continue
        x0, y0, x1, y1 = b["box"]
        w, h = x1 - x0, y1 - y0
        if h <= 0.015 * H and w >= 0.5 * W:
            b["cls"], b["relabel"] = "decoration", "figure"
            continue
        if y1 <= 0.3 * H and w <= 0.3 * W and h <= 0.25 * H and 0.6 <= w / max(1.0, h) <= 1.6:
            b["cls"], b["relabel"] = "logo", "figure"


def relabel_by_position(blocks: list, W: int, H: int) -> None:
    """Letterhead / title / footer / page number from the position relative to the body text."""
    def lh(b):
        hs = [l["bbox"][3] - l["bbox"][1] for l in b["lines"]]
        return float(np.median(hs)) if hs else 0.0
    wide = [b for b in blocks if b["cls"] in ("text", "list") and len(b["lines"]) >= 2 and (b["box"][2] - b["box"][0]) >= 0.45 * W]
    body = [b for b in wide if len(b["lines"]) >= 3] or wide  # a two-line centred title is not the body
    # a two-line address block at the very bottom (or a letterhead at the very top) is not body text
    core = [b for b in body if 0.12 * H < (b["box"][1] + b["box"][3]) / 2 < 0.88 * H] or body
    if not core:
        return
    body_top = min(b["box"][1] for b in core)
    body_bottom = max(b["box"][3] for b in core)
    # title: centred short blocks right above the body (a reference number beside them is skipped; stacked centred
    # lines are all title)
    cands = [b for b in blocks if b["cls"] == "text" and b["lines"] and b["box"][3] <= body_top + 2 and len(b["lines"]) <= 2]
    limit = body_top
    for b in sorted(cands, key=lambda b: -b["box"][3]):
        x0, y0, x1, y1 = b["box"]
        h = max(lh(b), 1.0)
        if limit - y1 > 3.5 * h:
            break  # too far above the body (or the title found so far)
        if abs((x0 + x1) / 2 - W / 2) < 0.04 * W and (x1 - x0) < 0.6 * W:
            b["cls"], b["relabel"] = "title", "text"
            limit = y0
    text_blocks = [b for b in blocks if b["cls"] in ("text", "list", "title", "section_header") and b["lines"]]
    for b in blocks:
        if b["cls"] != "text" or not b["lines"]:
            continue
        x0, y0, x1, y1 = b["box"]
        if y1 <= body_top + 2 and y1 <= 0.35 * H:
            # a letterhead is separated from what follows; a short first paragraph is not (a list continued
            # from the previous page): gap to the next text block below >= 1.5 line heights
            below = [o["box"][1] for o in text_blocks if o is not b and o["box"][1] >= y1 - 2]
            gap = (min(below) - y1) if below else float("inf")
            centred_top = abs((x0 + x1) / 2 - W / 2) < 0.04 * W and (x1 - x0) < 0.6 * W and y1 <= 0.2 * H and len(b["lines"]) <= 3
            short_top = (x1 - x0) < 0.4 * W and y1 <= 0.2 * H and len(b["lines"]) == 1  # a reference number / date line
            if gap >= 1.5 * max(lh(b), 1.0) or centred_top or short_top:
                b["cls"], b["relabel"] = "header", "text"
        elif y0 >= body_bottom - 2 and y0 >= 0.85 * H and len(b["lines"]) <= 3:
            text = " ".join((l.get("text") or "") for l in b["lines"]).strip()
            if text and len(text) <= 8 and _DIGITS.match(text) and (x1 - x0) < 0.15 * W:
                b["cls"], b["relabel"] = "page_number", "text"
            else:
                b["cls"], b["relabel"] = "footer", "text"


# ---------------------------------------------------------------------------------------------- font metrics

def line_metrics(ink: np.ndarray, ln: dict, W: int, H: int) -> None:
    """``core_h`` (rows with >= 35 % of the peak ink: the consonant bodies) and ``baseline`` (bottom of the core)."""
    x0, y0, x1, y1 = _bbox_int(ln["bbox"], W, H)
    m = ink[y0:y1, x0:x1] > 0
    if m.size == 0 or not m.any():
        ln["core_h"], ln["baseline"] = float(ln["bbox"][3] - ln["bbox"][1]) * 0.5, float(ln["bbox"][3]) - 0.25 * (ln["bbox"][3] - ln["bbox"][1])
        return
    rows = m.sum(1).astype(np.float32)
    rows = cv2.GaussianBlur(rows[:, None], (1, 3), 0.8)[:, 0] if len(rows) >= 3 else rows
    core = np.where(rows >= 0.35 * rows.max())[0]
    ln["core_h"] = float(core[-1] - core[0] + 1)
    ln["baseline"] = float(y0 + core[-1] + 1)


def _trimmed(vals: list[float]) -> float:
    """Median up to 4 sorted values, interquartile mean from 5: the odd line (a bullet row, a line with a huge
    subscript, a mis-sized crop) does not pull the block's size."""
    if len(vals) >= 5:
        return float(np.mean(vals[len(vals) // 4: len(vals) - len(vals) // 4]))
    return float(vals[len(vals) // 2])


def measure_fonts(ink: np.ndarray, blocks: list, text_classes: set) -> dict:
    """Per line core_h / baseline; per block ``font_px`` (one size for the whole block) on the page's size levels.

    With font-model sizes (``font.em_px``, ``font_src`` = "model") see app/services/font_sizes.py: reliability-weighted
    lines, perspective / scale-field normalisation, weighted block medians, log-space level clustering and the page
    body typeface. Blocks without model sizes: ink core height x a per-script constant (``font_src`` = "ink"),
    clustered into levels the same way. Returns the page-level size info (levels, body size, scale field)."""
    from app.services.font_sizes import levels, page_sizes
    H, W = ink.shape[:2]
    for b in blocks:
        if b["cls"] not in text_classes or not b["lines"]:
            continue
        b.pop("font_src", None)
        for ln in b["lines"]:
            line_metrics(ink, ln, W, H)
    info = page_sizes(blocks, text_classes, W, H)
    ink_blocks = [b for b in blocks if b["cls"] in text_classes and b["lines"] and b.get("font_src") != "model"]
    for b in ink_blocks:
        text = " ".join((l.get("text") or "") for l in b["lines"])
        k = K_KHMER if _is_khmer_text(text) or not text.strip() else K_LATIN
        b["font_px"], b["font_src"] = round(_trimmed(sorted(l["core_h"] for l in b["lines"])) * k, 1), "ink"
    lab = levels([b["font_px"] for b in ink_blocks], [max(1, len(b["lines"])) for b in ink_blocks])
    for k in set(lab):
        grp = [b for b, l in zip(ink_blocks, lab) if l == k]
        level = float(np.average([b["font_px"] for b in grp], weights=[max(1, len(b["lines"])) for b in grp]))
        for b in grp:
            b["font_px"] = round(level, 1)
    return info


# ---------------------------------------------------------------------------------------------- page layout

def page_layout(blocks: list, W: int, H: int, text_classes: set) -> dict:
    """Margins + zones + per-block alignment. Margins come from the body text lines (5th / 95th percentile edges)."""
    body = [b for b in blocks if b["cls"] in text_classes and b["cls"] not in ("header", "footer", "page_number") and b["lines"]]
    xs0 = [l["bbox"][0] for b in body for l in b["lines"]]
    xs1 = [l["bbox"][2] for b in body for l in b["lines"]]
    content = [b for b in blocks if b["lines"] or b["cls"] not in text_classes]
    left = float(np.percentile(xs0, 5)) if xs0 else 0.0
    right = float(W - np.percentile(xs1, 95)) if xs1 else 0.0
    top = min((b["box"][1] for b in content), default=0.0)
    bottom = H - max((b["box"][3] for b in content), default=float(H))
    header = [b for b in blocks if b["cls"] in ("header", "logo")]
    footer = [b for b in blocks if b["cls"] in ("footer", "page_number")]
    zones = {
        "header": [min(b["box"][1] for b in header), max(b["box"][3] for b in header)] if header else None,
        "footer": [min(b["box"][1] for b in footer), max(b["box"][3] for b in footer)] if footer else None,
    }
    body_blocks = [b for b in blocks if b["cls"] not in ("header", "logo", "footer", "page_number", "decoration")]
    zones["body"] = [min(b["box"][1] for b in body_blocks), max(b["box"][3] for b in body_blocks)] if body_blocks else None
    cx_page = W / 2
    for b in blocks:
        x0, _, x1, _ = b["box"]
        if b["lines"] and len(b["lines"]) >= 3 and (x1 - x0) >= 0.6 * (W - left - right):
            b["align"] = "left"  # a paragraph (a two-line centred title is judged by its centre below)
        elif abs((x0 + x1) / 2 - cx_page) < 0.03 * W and x0 - left > 0.05 * W:
            b["align"] = "center"
        elif (W - right) - x1 < 0.03 * W and x0 - left > 0.1 * W:
            b["align"] = "right"
        else:
            b["align"] = "left"
    return {"width": W, "height": H,
            "margins": {"left": round(left, 1), "right": round(right, 1), "top": round(top, 1), "bottom": round(bottom, 1)},
            "zones": {k: ([round(v[0], 1), round(v[1], 1)] if v else None) for k, v in zones.items()}}


# ---------------------------------------------------------------------------------------------- figures with text

def _picture_clusters(comps: list, gap: float, gap_x: float | None = None) -> list:
    """Group picture component boxes [x0,y0,x1,y1] that lie within ``gap`` (vertically) / ``gap_x`` (horizontally) of
    each other -> cluster boxes, largest first."""
    gx = gap if gap_x is None else gap_x
    boxes = [list(c) for c in comps]
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out: list = []
        for c in boxes:
            for o in out:
                if c[0] - gx < o[2] and c[2] + gx > o[0] and c[1] - gap < o[3] and c[3] + gap > o[1]:
                    o[0], o[1], o[2], o[3] = min(o[0], c[0]), min(o[1], c[1]), max(o[2], c[2]), max(o[3], c[3])
                    merged = True
                    break
            else:
                out.append(c)
        boxes = out
    return sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def _on_picture(l: dict, pic: list) -> bool:
    x0, y0, x1, y1 = l["bbox"]
    iw, ih = min(x1, pic[2]) - max(x0, pic[0]), min(y1, pic[3]) - max(y0, pic[1])
    if iw <= 0 or ih <= 0:
        return False
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    inside = pic[0] <= cx <= pic[2] and pic[1] <= cy <= pic[3]
    return inside or iw * ih >= 0.3 * max(1.0, (x1 - x0) * (y1 - y0))


SPLIT_CLASSES = ("figure", "logo")


def split_figures(ink: np.ndarray, blocks: list, nonpaper: np.ndarray | None = None) -> list:
    """A layout "figure" / "logo" that also holds detected text lines (an emblem with the document title under it, a
    footer the model took for a picture, a loose box around two logos and the letterhead between them): the pictures
    are the ink components that touch no text line (or reach far beyond the lines they touch, an emblem over a
    title), clustered by proximity. A line stays with the picture it lies on (centre inside or >= 30 % overlap);
    every other line leaves the block (returned, to be grouped into text blocks). The block shrinks to its largest
    picture; further picture clusters become blocks of their own (appended to ``blocks``); a picture that duplicates
    another figure / logo block is dropped. A figure with no picture ink at all becomes a text block. A box that is
    mostly non-paper (``nonpaper`` mask: a photo, a coloured seal) is a solid picture and is left alone.
    -> the released lines."""
    H, W = ink.shape[:2]
    released: list = []
    drop: list = []
    new_blocks: list = []
    all_lines = [tuple(l["bbox"]) for o in blocks for l in o["lines"]]
    for b in blocks:
        if b["cls"] not in SPLIT_CLASSES or not b["lines"]:
            continue
        x0, y0, x1, y1 = _bbox_int(b["box"], W, H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        if nonpaper is not None and float(nonpaper[y0:y1, x0:x1].mean()) >= 0.5:
            continue  # a photo / coloured seal: the text on it belongs to it
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1], 8)
        # text ink = components touching a line box; the box is widened by one line height sideways because the
        # detector often cuts the first / last glyph of a display-font title
        near = [(lx0 - (ly1 - ly0), ly0 - 2, lx1 + (ly1 - ly0), ly1 + 2) for lx0, ly0, lx1, ly1 in all_lines
                if lx0 < x1 + 2 and lx1 > x0 - 2 and ly0 < y1 + 2 and ly1 > y0 - 2]
        line_h = float(np.median([l["bbox"][3] - l["bbox"][1] for l in b["lines"]]))
        comps = []
        min_area = max(30.0, 0.0005 * (x1 - x0) * (y1 - y0))
        for i in range(1, n):
            cx, cy, cw, ch, area = (int(v) for v in st[i])
            if area < min_area:
                continue
            cx, cy = cx + x0, cy + y0
            touched = [(lx0, ly0, lx1, ly1) for lx0, ly0, lx1, ly1 in near if cx < lx1 and cx + cw > lx0 and cy < ly1 and cy + ch > ly0]
            if touched:
                top, bot = min(t[1] for t in touched), max(t[3] for t in touched)
                above, below = top - cy > line_h, (cy + ch) - bot > line_h
                if not above and not below:
                    continue  # ink of a text line (or a frame around the lines)
                if above and not below:   # an emblem whose ribbon touches the title under it: the picture ends at the title
                    comps.append([cx, cy, cx + cw, int(top)]); continue
                if below and not above:
                    comps.append([cx, int(bot), cx + cw, cy + ch]); continue
            comps.append([cx, cy, cx + cw, cy + ch])
        if not comps:  # nothing but text: it is a text block
            if b["cls"] == "figure":
                b["cls"], b["relabel"] = "text", "figure"
            continue
        gap = max(6.0, 0.04 * max(x1 - x0, y1 - y0), 0.5 * line_h)
        pics = _picture_clusters(comps, gap, max(gap, 1.2 * line_h))
        big = [p for p in pics if p[2] - p[0] >= line_h or p[3] - p[1] >= line_h]
        pics = big or pics[:1]
        owner = [next((k for k, pic in enumerate(pics) if _on_picture(l, pic)), None) for l in b["lines"]]
        out = [l for l, k in zip(b["lines"], owner) if k is None]
        if not out:
            continue  # the text lies on the picture (a chart): keep it a figure
        orig = list(b["box"])
        pieces = []
        for k, pic in enumerate(pics):
            px0, py0, px1, py1 = pic
            keep = [l for l, o in zip(b["lines"], owner) if o == k]
            for l in keep:  # text on the picture stays inside the box
                px0, py0, px1, py1 = min(px0, l["bbox"][0]), min(py0, l["bbox"][1]), max(px1, l["bbox"][2]), max(py1, l["bbox"][3])
            mx, my = 0.04 * (px1 - px0) + 3, 0.04 * (py1 - py0) + 3
            pieces.append(([max(orig[0], px0 - mx), max(orig[1], py0 - my), min(orig[2], px1 + mx), min(orig[3], py1 + my)], keep))
        others = [o for o in blocks if o is not b and o["cls"] in SPLIT_CLASSES]
        first = True
        for box, keep in pieces:
            if any(_box_iou(box, o["box"]) > 0.5 for o in others):
                continue  # this picture is already a block of its own
            if first:
                b["split_from"] = orig  # the page pass skipped this region as a picture: detect its text again
                b["box"], b["lines"], b["split"] = box, keep, len(out)
                first = False
            else:
                new_blocks.append({**{k: v for k, v in b.items() if k not in ("split_from", "split")}, "box": box, "lines": keep,
                                   "source": "split", "split": 0})
        if first:  # every picture duplicated another block: this loose box is redundant
            drop.append(b)
        released += out
    if drop:
        blocks[:] = [b for b in blocks if not any(b is d for d in drop)]
    blocks += new_blocks
    return released


def _box_iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / max(1.0, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def find_rules(ink: np.ndarray, blocks: list, text_classes: set, block_factory, max_rules: int = 3) -> list:
    """Horizontal rule lines drawn on the page outside every block (the line between letterhead and body, above the
    footer) -> "decoration" blocks. A page with more than ``max_rules`` such lines is ruled paper or a table: none."""
    H, W = ink.shape[:2]
    hs = [l["bbox"][3] - l["bbox"][1] for b in blocks if b["cls"] in text_classes for l in b["lines"]]
    page_h = float(np.median(hs)) if hs else 0.02 * H
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    found = []
    for i in range(1, n):
        cx, cy, cw, ch, area = (int(v) for v in st[i])
        if cw < 0.3 * W or ch > max(6.0, 0.8 * page_h) or area < 0.5 * cw * max(1, ch // 2):
            continue
        box = [cx, cy, cx + cw, cy + ch]
        if any(box[0] < o["box"][2] and box[2] > o["box"][0] and box[1] < o["box"][3] and box[3] > o["box"][1] for o in blocks):
            continue
        found.append(box)
    if not found or len(found) > max_rules:
        return []
    return [block_factory("decoration", [b[0], max(0, b[1] - 2), b[2], min(H, b[3] + 2)], 1.0, "decoration", source="heuristic") for b in found]


# ---------------------------------------------------------------------------------------------- line boxes

def _leans(ln: dict, min_deg: float = 1.5) -> bool:
    q = ln.get("quad")
    if q is None:
        return False
    q = np.asarray(q, np.float32).reshape(-1, 2)
    if q.shape[0] != 4:
        return False
    dx, dy = float(q[1, 0] - q[0, 0]), float(q[1, 1] - q[0, 1])
    return dx > 0 and abs(np.degrees(np.arctan2(dy, dx))) >= min_deg


def snap_line_boxes(ink: np.ndarray, blocks: list, text_classes: set) -> int:
    """Grow every text line box to the glyph ink it cuts through: a connected component that intersects exactly one
    line extends that line, by at most one line height sideways and 0.35 line heights up / down (the detector often
    cuts the first or last glyph of a display-font title, and a cut glyph is misread). Components touching two lines
    (a subscript reaching the next line), rule-like components (wider than 3 line heights and thin) and large ones (an
    emblem touching a title) are left alone. Returns the number of lines changed."""
    lines = [ln for b in blocks if b["cls"] in text_classes for ln in b["lines"]]
    if not lines:
        return 0
    L = np.array([ln["bbox"] for ln in lines], np.float32)                 # [M,4]
    hs = np.maximum(1.0, L[:, 3] - L[:, 1])
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return 0
    C = st[1:].astype(np.float32)                                          # x, y, w, h, area
    C = np.stack([C[:, 0], C[:, 1], C[:, 0] + C[:, 2], C[:, 1] + C[:, 3], C[:, 4]], 1)
    C = C[C[:, 4] >= 4]
    if not len(C):
        return 0
    ix = (C[:, None, 0] < L[None, :, 2] + 1) & (C[:, None, 2] > L[None, :, 0] - 1) & \
         (C[:, None, 1] < L[None, :, 3] + 1) & (C[:, None, 3] > L[None, :, 1] - 1)    # [N,M]
    hits = ix.sum(1)
    one = np.where(hits == 1)[0]
    grow = np.array(L)
    for i in one:
        j = int(np.argmax(ix[i]))
        h = hs[j]
        cx0, cy0, cx1, cy1, area = C[i]
        cw, ch = cx1 - cx0, cy1 - cy0
        if (cw > 3 * h and ch < 0.25 * h) or area > 40 * h * h or ch > 2.5 * h:
            continue
        grow[j, 0] = min(grow[j, 0], max(cx0, L[j, 0] - h))
        grow[j, 2] = max(grow[j, 2], min(cx1, L[j, 2] + h))
        grow[j, 1] = min(grow[j, 1], max(cy0, L[j, 1] - 0.35 * h))
        grow[j, 3] = max(grow[j, 3], min(cy1, L[j, 3] + 0.35 * h))
    changed = 0
    H, W = ink.shape[:2]
    for j, ln in enumerate(lines):
        if _leans(ln):  # integration_qa: a leaning line of a tilted photo keeps its rotated quad (an axis-aligned
            continue    # box would crop the neighbouring lines into the recognition and hide the page skew)
        nb = [float(max(0.0, grow[j, 0])), float(max(0.0, grow[j, 1])), float(min(W, grow[j, 2])), float(min(H, grow[j, 3]))]
        if any(abs(a - b) > 0.5 for a, b in zip(nb, ln["bbox"])):
            ln["bbox"] = nb
            ln["quad"] = np.array([[nb[0], nb[1]], [nb[2], nb[1]], [nb[2], nb[3]], [nb[0], nb[3]]], np.float32)
            changed += 1
    return changed


# ---------------------------------------------------------------------------------------------- text colour

def _ink_hex(bgr_region: np.ndarray, ink_region: np.ndarray) -> str | None:
    """Median colour of the darkest 40 % of the ink pixels (anti-aliased edges are lighter than the true ink)."""
    m = ink_region > 0
    if m.sum() < 6:
        return None
    px = bgr_region[m].astype(np.int32)
    lum = px @ np.array([114, 587, 299]) // 1000  # BGR -> luminance
    keep = px[lum <= np.percentile(lum, 40)]
    if len(keep) == 0:
        keep = px
    b, g, r = (int(v) for v in np.median(keep, axis=0))
    # small anti-aliased print measures grey even when printed black: a neutral dark colour is black ink
    if max(r, g, b) - min(r, g, b) < 28 and (299 * r + 587 * g + 114 * b) // 1000 < 170:
        return "#1a1a1a"
    return "#%02x%02x%02x" % (r, g, b)


def text_colors(bgr: np.ndarray, ink: np.ndarray, blocks: list, text_classes: set) -> None:
    """``text_color`` (hex) per line, per span (bold words) and per block (its most common line colour)."""
    H, W = ink.shape[:2]
    for b in blocks:
        if b["cls"] not in text_classes or not b["lines"]:
            continue
        colours = []
        for ln in b["lines"]:
            x0, y0, x1, y1 = _bbox_int(ln["bbox"], W, H)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            c = _ink_hex(bgr[y0:y1, x0:x1], ink[y0:y1, x0:x1])
            if c:
                ln["text_color"] = c
                colours.append(c)
            for sp in ln.get("spans") or []:
                sx0, sx1 = max(x0, int(sp["x0"])), min(x1, int(np.ceil(sp["x1"])))
                if sx1 - sx0 >= 2:
                    sc = _ink_hex(bgr[y0:y1, sx0:sx1], ink[y0:y1, sx0:sx1])
                    if sc:
                        sp["text_color"] = sc
        if colours:
            b["text_color"] = max(set(colours), key=colours.count)
