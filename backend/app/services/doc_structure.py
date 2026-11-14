"""Document structure that the detectors do not give us: list bullets, indentation (tab stops) and paragraphs.

* ``find_bullets``      the DB text-line detector is trained on lines and drops isolated marks of 2..8 px (the "-" and
                        "o" bullets of a list). For every text line we look at the ink strip just left of it: one small
                        connected component on the line's vertical centre, clear of every detected line, is its bullet.
* ``merge_bullet_lines``  a bullet the detector DID return comes out as its own tiny line read as "•" (pipeline
                        ``_bullet_or_text``); it is attached to the text line of the same row and removed.
* ``structure_blocks``  lines -> paragraphs (a bullet, a deeper start than the line above, a short previous line or a
                        blank line start a new one); the first-line left edges of all paragraphs of a column are
                        clustered into tab stops and every line gets its ``indent_level`` (0 = the column margin), so a
                        Khmer first-line indent, a list item and a sub-item get 1, 1 and 2 tabs whatever layout box they
                        ended up in. Khmer line breaks carry no space when wrapped lines are joined, Latin ones do.

All functions work on the pipeline's internal line dicts: {"bbox": [x0, y0, x1, y1], "text": str, ...}.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from app.services.reading_order import sort_lines

BULLET_READINGS = {"o", "O", "0", "០", "•", "-", "·", "*", "e", "c", "C", "D", "Q", "–", "—", "▪", "■", "○", "●", "◦", "➢", "►", "✓"}
DASH = "-"
DOT = "•"
MAX_LEVEL = 6


def _h(ln) -> float:
    return max(1.0, ln["bbox"][3] - ln["bbox"][1])


SYMBOL_READINGS = {"•", "-", "·", "–", "—", "▪", "■", "○", "●", "◦", "➢", "►", "✓", "❖", "◆", "◇", "♦", "➤", "*", "+"}


# a list number read on its own ("3.", "១.", "a)", "ក.", "(2)") - the detector often returns the number of a clause as a
# tiny box of its own, separated from the clause text by the hanging indent
NUM_MARK_RE = re.compile(r"^(?:\(?(?:\d{1,3}|[០-៩]{1,3})[.)]|[a-z][.)]|[A-Z]\)|[ក-អ][.)]|(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)[.)])$")
# the same marker at the start of a recognised line, followed by its text ("1. ភាគី...", "ក. ...", "a) ...")
LEAD_NUM_RE = re.compile(r"^\s*(\(?(?:\d{1,3}|[០-៩]{1,3})[.)]|[a-z][.)]|[A-Z]\)|[ក-អ][.)]|(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)[.)])\s+(?=\S)")
# a dash / glyph bullet glued to (or spaced from) the text: "-អតិថិជន...", "- item", "• item". A Khmer word never starts
# with a hyphen, so a leading "-" before a Khmer letter is a list dash even without a space.
LEAD_GLYPH_RE = re.compile(r"^\s*(?:([-–—•·▪■➢►✓◦○●❖◆◇♦➤➔→▸▶✔])(?:\s+(?=\S)|(?=[ក-៿]))|([*+])\s+(?=[ក-៿A-Za-z]))")
# (2026-11-13) "*" / "+" are how the recogniser reads a ❖ / ◆ bullet; they need a space and a word after them.


def is_number_marker(text: str, ln: dict) -> bool:
    """A short line whose whole reading is a list number ("3.", "១.", "a)")."""
    t = (text or "").strip()
    if not t or not NUM_MARK_RE.match(t):
        return False
    x0, y0, x1, y1 = ln["bbox"]
    return (x1 - x0) < 3.5 * max(1.0, (y1 - y0))


def split_text_marker(ln: dict) -> bool:
    """A recognised line that starts with a list marker gets it as ``bullet`` (and loses it from the text):
    "1. text" -> bullet "1.", "-text" / "- text" -> bullet "-". Lines that already have a bullet are left alone.
    -> True when a marker was split off."""
    if ln.get("bullet"):
        return False
    text = ln.get("text") or ""
    m = LEAD_NUM_RE.match(text)
    if m and len(text) - m.end() >= 2:
        ln["bullet"], ln["text"] = m.group(1), text[m.end():].lstrip()
        ln.pop("char_pos", None)
        _strip_spans(ln, m.end())
        _set_text_x0(ln, m.end(), len(text))
        return True
    m = LEAD_GLYPH_RE.match(text)
    if m and len(text) - m.end() >= 2:
        g = m.group(1) or m.group(2)
        ln["bullet"], ln["text"] = (DASH if g in "-–—" else DOT), text[m.end():].lstrip()
        ln.pop("char_pos", None)
        _strip_spans(ln, m.end())
        _set_text_x0(ln, m.end(), len(text))
        return True
    return False


def _set_text_x0(ln: dict, cut: int, n: int) -> None:
    """The line box still starts at the marker that was split off its text: remember where the text itself starts
    (``text_x0``) so the item's wrapped lines, which align with the text, stay in the item (split_paragraphs)."""
    x0, y0, x1, y1 = ln["bbox"]
    sp = ln.get("spans") or []
    if sp and sp[0].get("x0") is not None and x0 <= float(sp[0]["x0"]) < x1:
        ln["text_x0"] = float(sp[0]["x0"])
    else:  # a marker and its space are about one line height wide; the character share overshoots on short lines
        ln["text_x0"] = x0 + min((x1 - x0) * cut / max(1, n), 1.2 * max(1.0, y1 - y0))


def _strip_spans(ln: dict, n: int) -> None:
    """Drop the first ``n`` characters from the word spans (bold runs) of a line."""
    spans = ln.get("spans")
    if not spans:
        return
    out = []
    for sp in spans:
        t = sp.get("text", "")
        if n <= 0:
            out.append(sp); continue
        if len(t) <= n:
            n -= len(t) + 1  # spans are words joined by one space
            continue
        out.append({**sp, "text": t[n:].lstrip()}); n = 0
    ln["spans"] = [sp for sp in out if sp.get("text")]


def merge_row_overlaps(lines: list) -> list:
    """Two detected boxes of ONE printed line that overlap on the same row (the detector returned the line twice,
    split in overlapping pieces - dotted blanks do that) -> one box (union). Pieces with a gap stay apart (columns,
    fields). Run before recognition so the union is read once."""
    if len(lines) < 2:
        return lines
    lines = sorted(lines, key=lambda l: l["bbox"][0])
    out: list = []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        h = max(1.0, y1 - y0)
        for o in out:
            ox0, oy0, ox1, oy1 = o["bbox"]
            oh = max(1.0, oy1 - oy0)
            vov = min(y1, oy1) - max(y0, oy0)
            hov = min(x1, ox1) - max(x0, ox0)
            if vov >= 0.6 * min(h, oh) and hov >= 0.2 * min(h, oh) and 0.6 < h / oh < 1.6:  # (a piece inside the other too)
                nb = [min(x0, ox0), min(y0, oy0), max(x1, ox1), max(y1, oy1)]
                q = np.concatenate([np.asarray(o["quad"], np.float32).reshape(-1, 2), np.asarray(ln["quad"], np.float32).reshape(-1, 2)])
                r = cv2.minAreaRect(q)
                box = cv2.boxPoints(r)
                c = box.mean(0)
                box = box[np.argsort(np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0]))]
                o["quad"], o["bbox"], o["score"] = box.astype(np.float32), [float(v) for v in nb], max(o.get("score", 0), ln.get("score", 0))
                o["merged_dup"] = True
                break
        else:
            out.append(ln)
    return out


def is_bullet_glyph(text: str, ln: dict) -> bool:
    """A line whose whole reading is one bullet symbol is a list bullet; a near-square little box read as a single
    o/0/e/c (the recogniser has no class for the ring glyph) is one too - a wide box read "o" is a misread word."""
    if text in SYMBOL_READINGS:
        return True
    x0, y0, x1, y1 = ln["bbox"]
    return text in BULLET_READINGS and (x1 - x0) < 1.6 * max(1.0, (y1 - y0))


# ------------------------------------------------------------------------------------------------ bullets

def ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Dark-on-light ink as 0/255 (adaptive: phone photos have uneven lighting)."""
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)


def _overlaps_any(box, boxes) -> bool:
    x0, y0, x1, y1 = box
    return any(x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1] for b in boxes)


def _row_coverage(ink: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    """Fraction of the columns x0..x1 that carry ink somewhere in rows y0..y1 (1.0 = an unbroken horizontal line)."""
    if x1 <= x0 or y1 <= y0:
        return 0.0
    band = ink[y0:y1, x0:x1]
    return float((band.max(axis=0) > 0).mean()) if band.size else 0.0


def find_bullet(ink: np.ndarray, gray: np.ndarray, ln: dict, other_boxes: list, reach: float = 2.5):
    """-> (glyph, [x0, y0, x1, y1]) of the bullet left of line ``ln`` or None.

    Window: ``reach`` line heights left of the line, the line's own height. Accept one component (two for a broken
    ring) that
      * is small: <= 0.6 h tall, <= 1.2 h wide, >= 0.18 h in its larger dimension, and either a dash (flat: width
        >= 2 x height, height <= 0.3 h) or a compact mark (width/height within 0.5..2: ring, dot, square, arrow),
      * sits on the line's vertical centre (+-0.35 h), touches no edge of the window (ascenders of the lines above
        and below, a rule that continues left) and lies in no detected line (a word of the column to the left),
      * is not a fragment of a ruled line: its rows are not inked across the text to its right or the paper to
        its left (notebook paper, table rules),
      * is not much fainter than the text ink (bleed-through, shadows)."""
    x0, y0, x1, y1 = ln["bbox"]
    h = _h(ln)
    if x1 - x0 < 1.6 * h:  # not a text line (a bullet itself, a digit...)
        return None
    H, W = ink.shape[:2]
    sx0, sx1 = int(max(0, x0 - reach * h)), int(min(W, max(0, x0 - 0.15 * h)))
    sy0, sy1 = int(max(0, y0)), int(min(H, y1))
    if sx1 - sx0 < 3 or sy1 - sy0 < 3:
        return None
    win = ink[sy0:sy1, sx0:sx1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(win, 8)
    good = []
    for i in range(1, n):
        cx, cy, cw, ch, area = (int(v) for v in st[i])
        if cy <= 0 or cy + ch >= win.shape[0] or (cx <= 0 and sx0 > 0):  # touches the window edge
            continue
        cx, cy = cx + sx0, cy + sy0
        if ch > 0.6 * h or cw > 1.2 * h or max(cw, ch) < 0.18 * h or area < 3:
            continue
        if abs(cy + ch / 2 - (y0 + y1) / 2) > 0.35 * h:
            continue
        if _overlaps_any((cx, cy, cx + cw, cy + ch), other_boxes):
            continue
        good.append((cx, cy, cw, ch, area, i))
    if not good or len(good) > 2:
        return None
    cx, cy = min(g[0] for g in good), min(g[1] for g in good)
    cw, ch = max(g[0] + g[2] for g in good) - cx, max(g[1] + g[3] for g in good) - cy
    if len(good) == 2 and (cw > 0.9 * h or ch > 0.6 * h):  # two marks, not one broken ring
        return None
    dash = cw >= 2 * ch and ch <= 0.3 * h
    if not dash and not (0.5 <= cw / max(1, ch) <= 2.0):
        return None
    # a ruled line: its rows stay inked between the mark and the text, across the text, or over the paper to its left
    ry0, ry1 = max(0, cy - 1), min(H, cy + ch + 1)
    if _row_coverage(ink, ry0, ry1, cx + cw + 2, int(x0)) >= 0.5:
        return None
    if _row_coverage(ink, ry0, ry1, int(x0), int(min(W, x0 + 4 * h))) >= 0.9:
        return None
    if _row_coverage(ink, ry0, ry1, int(max(0, cx - 3 * h)), cx - 1) >= 0.5:
        return None
    # not much fainter than the text ink (bleed-through, shadows); thin printed marks are anti-aliased lighter
    comp = np.zeros(win.shape, bool)
    for g in good:
        comp |= lab == g[5]
    mark_dark = float((255 - gray[sy0:sy1, sx0:sx1].astype(np.float32))[comp].mean())
    tx0, tx1 = int(x0), int(min(W, x0 + 4 * h))
    tmask = ink[sy0:sy1, tx0:tx1] > 0
    if tmask.any():
        text_dark = float((255 - gray[sy0:sy1, tx0:tx1].astype(np.float32))[tmask].mean())
        if mark_dark < 0.4 * text_dark:
            return None
    return (DASH if dash else DOT), [float(cx), float(cy), float(cx + cw), float(cy + ch)]


def find_bullets(bgr: np.ndarray, blocks: list, skip_cls=("table", "form_field"), gray=None, ink=None) -> int:
    """Set ``ln["bullet"]`` / ``ln["bullet_box"]`` on lines of the text blocks (``blocks[i]["lines"]``). -> count.
    ``gray`` / ``ink`` (ink_mask) can be passed in when the caller already has them."""
    lines = [(b, ln) for b in blocks if b["cls"] not in skip_cls for ln in b["lines"]]
    if not lines:
        return 0
    if gray is None:
        gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if ink is None:
        ink = ink_mask(gray)
    all_boxes = [ln["bbox"] for b in blocks for ln in b["lines"]]
    found = 0
    for _, ln in lines:
        if ln.get("bullet"):
            continue
        hit = find_bullet(ink, gray, ln, [bx for bx in all_boxes if bx is not ln["bbox"]])
        if hit:
            ln["bullet"], ln["bullet_box"] = hit
            ln["bullet_src"] = "ink"
            found += 1
    return found


def reread_ink_markers(blocks: list, read) -> int:
    """A mark found in the ink left of a line (find_bullets) is classed dot / dash by its shape; a clause number "2."
    (digit + full stop, two small components) passes the same test and became a dot. Read every compact ink mark
    with the recogniser (``read(list of (x0, y0, x1, y1) boxes) -> list of texts``; the box spans the line's height)
    and keep the reading when it is a list number. -> count changed."""
    todo = []
    for b in blocks:
        for ln in b["lines"]:
            if ln.get("bullet_src") == "ink" and ln.get("bullet") == DOT and ln.get("bullet_box"):
                bx0, _, bx1, _ = ln["bullet_box"]
                _, y0, _, y1 = ln["bbox"]
                h = max(1.0, y1 - y0)
                todo.append((ln, (bx0 - 0.3 * h, y0, bx1 + 0.3 * h, y1)))
    if not todo:
        return 0
    n = 0
    for (ln, _), t in zip(todo, read([bx for _, bx in todo])):
        t = (t or "").strip().replace(" ", "")
        if t and NUM_MARK_RE.match(t) and not t[:-1] in ("o", "O", "0", "០", "e", "c"):
            ln["bullet"] = t
            n += 1
    return n


LEADING_BULLETS = ("-", "–", "—", "•", "·", "○", "●", "◦", "▪", "■")


def strip_leading_bullet(ln: dict) -> None:
    """A line with a found bullet whose reading also starts with the glyph (a long dash touching the text) drops it."""
    text = ln.get("text") or ""
    if ln.get("bullet") and text.startswith(LEADING_BULLETS):
        ln["text"] = text[1:].lstrip()


def refine_bullet_boxes(ink: np.ndarray, blocks: list) -> int:
    """Shrink every ``bullet_box`` to the ink inside it (a bullet the line detector returned comes with a loose
    box; the ones found from the ink are tight) so all bullets of a page measure alike. -> count changed."""
    H, W = ink.shape[:2]
    n = 0
    for b in blocks:
        for ln in b["lines"]:
            bb = ln.get("bullet_box")
            if not bb:
                continue
            x0, y0 = max(0, int(bb[0]) - 1), max(0, int(bb[1]) - 1)
            x1, y1 = min(W, int(np.ceil(bb[2])) + 1), min(H, int(np.ceil(bb[3])) + 1)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            ys, xs = np.where(ink[y0:y1, x0:x1] > 0)
            if len(xs) < 3:
                continue
            nb = [float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)]
            if nb != bb:
                ln["bullet_box"], n = nb, n + 1
    return n


def merge_bullet_lines(lines: list) -> list:
    """Lines of one block after recognition: a line whose text is a bullet glyph is attached to the text line of the
    same row immediately to its right (``bullet`` / ``bullet_box``) and dropped. Returns the remaining lines."""
    marks = [ln for ln in lines if is_bullet_glyph(ln.get("text", ""), ln) or is_number_marker(ln.get("text", ""), ln)]
    if not marks:
        return lines
    mark_ids = {id(m) for m in marks}
    keep = [ln for ln in lines if id(ln) not in mark_ids]
    for m in marks:
        mx0, my0, mx1, my1 = m["bbox"]
        mh = _h(m)
        best, dist = None, float("inf")
        for ln in keep:
            x0, y0, x1, y1 = ln["bbox"]
            if x0 < mx1 - 0.3 * mh:  # must start right of the mark
                continue
            ov = min(y1, my1) - max(y0, my0)
            if ov < 0.5 * min(mh, _h(ln)):
                continue
            d = x0 - mx1
            if d < dist and d <= (2.5 if not is_number_marker(m.get("text", ""), m) else 4.0) * _h(ln):
                best, dist = ln, d
        if best is None or best.get("bullet"):
            keep.append(m)  # nothing to attach to: keep it as a line of its own
            continue
        best["bullet"] = DASH if m["text"] in ("-", "–", "—") else (m["text"].strip() if is_number_marker(m.get("text", ""), m) else DOT)
        best["bullet_box"] = list(m["bbox"])
    return keep


# ------------------------------------------------------------------------------------------------ structure

def _cluster_gap(values: list[float], tol: float) -> list[tuple[float, int]]:
    """Sorted values -> [(centre, count)]; a new cluster starts where two neighbours are more than ``tol`` apart."""
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [(float(np.mean(c)), len(c)) for c in out]


def _column_groups(blocks: list) -> list[list[dict]]:
    """Blocks chained by horizontal overlap (>= half the narrower one) = one text column."""
    n = len(blocks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ax0, _, ax1, _ = blocks[i]["box"]
        for j in range(i + 1, n):
            bx0, _, bx1, _ = blocks[j]["box"]
            ov = min(ax1, bx1) - max(ax0, bx0)
            if ov >= 0.5 * max(1.0, min(ax1 - ax0, bx1 - bx0)):
                parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(blocks[i])
    return list(groups.values())


def split_paragraphs(lines: list, tol: float, col_right: float) -> list[dict]:
    """Lines of one block in reading order -> [{"lines": [...], "bullet": str|None, "anchor": x of the first line}].

    A line starts a new paragraph when it carries a bullet, starts further right than the line above it (a Khmer
    first-line indent, a sub-item), leaves the hanging text column of a bulleted item, follows a line that ended
    early (> 3 line heights before the column's right edge) or follows a blank line."""
    paras: list[dict] = []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        h = _h(ln)
        new = not paras or bool(ln.get("bullet"))
        if not new:
            p = paras[-1]
            px0, py0, px1, py1 = p["lines"][-1]["bbox"]
            px0 = p["lines"][-1].get("text_x0", px0)  # a bullet split off the text: the text start counts
            if x0 > px0 + tol:
                new = True
            elif p["bullet"] and abs(x0 - p["anchor"]) > tol:
                new = True
            elif px1 < col_right - 3.0 * h:
                new = True
            elif y0 - py1 > 0.8 * h:
                new = True
        if new:
            paras.append({"lines": [ln], "bullet": ln.get("bullet"), "anchor": float(ln.get("text_x0", x0))})
        else:
            paras[-1]["lines"].append(ln)
    return paras


def structure_blocks(blocks: list, text_classes: set,
                     skip_cls=("table", "form_field", "header", "footer", "page_number")) -> None:
    """Reading order, paragraphs and indentation for every text block.

    Lines are sorted into rows (``b["lines"]``), grouped into paragraphs (``b["paras"]``) and given
    ``indent_level`` / ``indent`` / ``tab``. Per text column (blocks chained by horizontal overlap) the left edges of
    the paragraph FIRST lines are clustered into tab stops (tolerance half a line height); the margin is the leftmost
    stop used by >= 2 paragraphs. A paragraph's level is the number of stops left of its first line; a line's own
    level is the same count for its own left edge (the continuation lines of a Khmer first-line-indented paragraph
    are back at the margin: level 0; the hanging lines of a list item keep the item's level)."""
    for b in blocks:
        b["lines"] = sort_lines(b["lines"])
        b["paras"] = []
    assign_levels(blocks, text_classes, skip_cls, keep_paras=False)


def assign_levels(blocks: list, text_classes: set, skip_cls=("table", "form_field", "header", "footer", "page_number"),
                  keep_paras: bool = False) -> None:
    """Tab stops per text column -> ``indent_level`` / ``indent`` / ``tab`` of every line and paragraph.
    keep_paras=False: paragraphs are (re)split from the geometry (split_paragraphs); True: ``b["paras"]`` is kept
    (built by the structure model) and only the levels are measured."""
    for b in blocks:
        for ln in b["lines"]:
            ln["indent_level"], ln["indent"], ln["tab"] = 0, 0, False
    cand = [b for b in blocks if b["cls"] in text_classes and b["cls"] not in skip_cls and b["lines"]]
    for group in _column_groups(cand):
        lines = [ln for b in group for ln in b["lines"]]
        hs = sorted(_h(ln) for ln in lines)
        tol = max(4.0, 0.5 * hs[len(hs) // 2])
        col_right = max(ln["bbox"][2] for ln in lines)
        paras = []
        for b in group:
            if not keep_paras or not b.get("paras"):
                b["paras"] = split_paragraphs(b["lines"], tol, col_right)
            paras += b["paras"]
        # the margin comes from ALL line left edges (the continuation lines of Khmer first-line-indented paragraphs
        # sit on it even when every paragraph start is indented); the other tab stops from the paragraph starts
        edges = _cluster_gap([ln["bbox"][0] for ln in lines], tol)
        multi = [c for c in edges if c[1] >= 2]
        margin = (multi[0] if multi else edges[0])[0]
        clusters = _cluster_gap([p["anchor"] for p in paras], tol)
        stops = [margin] + [c for c, k in clusters if k >= 2 and c > margin + tol]

        def level_of(x: float) -> int:
            lv = sum(1 for s in stops if s < x - tol)
            if lv == 0 and x - margin > tol:
                lv = 1  # alone beyond the margin: still indented
            return min(lv, MAX_LEVEL)

        for p in paras:
            p["indent_level"] = level_of(p["anchor"])
            for ln in p["lines"]:
                ln["indent_level"] = level_of(ln["bbox"][0])
                ln["indent"] = int(round(max(0.0, ln["bbox"][0] - margin)))
                ln["tab"] = ln["indent_level"] > 0
    for b in blocks:
        if not b.get("paras") and b["lines"]:  # tables, headers ...: one paragraph per row
            b["paras"] = [{"lines": [ln], "bullet": ln.get("bullet"), "anchor": ln["bbox"][0], "indent_level": 0}
                          for ln in b["lines"]]


# ------------------------------------------------------------------------------------------------ text

def _is_khmer(ch: str) -> bool:
    return "\u1780" <= ch <= "\u17ff" or "\u19e0" <= ch <= "\u19ff"


def join_wrapped(a: str, b: str) -> str:
    """Join two wrapped lines: Khmer breaks at a word boundary without a space, everything else with one."""
    if not a:
        return b
    if not b:
        return a
    return a + ("" if _is_khmer(a[-1]) and _is_khmer(b[0]) else " ") + b


def paragraph_text(lines: list) -> str:
    text = ""
    for ln in lines:
        text = join_wrapped(text, ln.get("text", ""))
    return text


def block_text(lines: list) -> str:
    """Line-per-line text with the layout: one tab per indent level, the bullet glyph before a list item."""
    rows = []
    for ln in lines:
        if not ln.get("text"):
            continue
        rows.append("\t" * int(ln.get("indent_level", 0)) + (f"{ln['bullet']} " if ln.get("bullet") else "") + ln["text"])
    return "\n".join(rows)
