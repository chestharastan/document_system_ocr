"""Table structure (v3): a ``table`` layout block -> row / column bands, logical cells with text, header rows and the
page's document kind (docs/TABLES_CONTRACT.md §2.1 - 2.2).

Three structure sources, merged in this order of trust:
  1. the table-structure model (stage ``table``, kind ``tsr_yolo_onnx``): row / column / column_header /
     spanning_cell / projected_row_header boxes on the table crop. Rows and columns are sorted, overlaps resolved
     (NMS by centre), the band edges snapped to ruling lines within 2 % of the table size, spanning cells override
     the grid; ``source`` names the model.
  2. the structure transformer v3 (``struct_tables``: cells as groups of text lines, source ``struct_v3``) for
     borderless tables when no TSR model is loaded and no ruling lines were found.
  3. rules (``source = "rules"``), always computed as the fallback and as the ruling-line prior for 1.:
     * ruling lines by morphology on the ink of the table crop (long horizontal / vertical openings, pieces chained
       along the line so a slightly skewed photo still yields one rule; rules spanning >= 50 % of the table kept);
     * rows between the horizontal rules, columns between the vertical rules; without rules, rows from the
       y-clustering of the text lines and columns from the gap profile of the word boxes (the per-character CTC
       positions the recogniser exposes, else the line boxes) across the whole table;
     * spanning cells: a text line crossing several column bands with no rule between them (bordered), or centred
       over the bands it covers (borderless) - deliberately conservative;
     * ``header_rows``: the first row when it is bold, shaded or separated by a heavier rule, else 0.

Cell text: the block's recognised lines go to the cell they overlap; a line that runs across several columns is split
at the column boundaries with its character positions (the pieces replace it in ``block["lines"]`` so the indices in
``cells[].lines`` stay valid), or re-recognised per cell crop when positions are missing (``read_boxes``). The lines
of one cell are joined Khmer-aware (document.paragraph_runs). Cells with no ink get ``text: ""``.

Coordinates: the block and its lines are page pixels; everything inside this module works in table-crop pixels and
converts back at the end. Output: ``block["table"]`` (contract §2.2) with ``cells[i]["_lines"]`` = the line dicts
(the pipeline turns them into indices when it renders the block).
"""

from __future__ import annotations

import numpy as np
import cv2

from app.services.bold import word_spans
from app.services.document import paragraph_runs
from app.services.reading_order import sort_lines
from app.taxonomy import TEXT_CLASSES

TSR_CLASSES = ["row", "column", "column_header", "spanning_cell", "projected_row_header"]
KINDS = ("document", "sheet", "mixed")
MIN_RULE_SPAN = 0.5      # a ruling line spans at least this fraction of the table
SNAP_FRAC = 0.02         # model band edges snap to a ruling line within this fraction of the table size
LOW_CONF = 0.8
SHEET_MIN_AREA = 0.2     # integration_qa: sheet also when the table covers >= 20 % and the other text < 5 % of the page
SHEET_MAX_TEXT = 0.05
BODY_TEXT = {"text", "list", "section_header", "caption", "footnote", "form_field", "handwriting"}
_RULE_TOL = 0.02         # a rule "at" a band boundary lies within this fraction of the table size


# ================================================================================================= document kind

def _box4(b: dict) -> list[float]:
    bx = b.get("box") or [0, 0, 0, 0]
    if isinstance(bx, dict):
        x, y = float(bx.get("x", 0)), float(bx.get("y", 0))
        return [x, y, x + float(bx.get("width", 0)), y + float(bx.get("height", 0))]
    return [float(v) for v in bx]


def _grid_size(t: dict | None) -> tuple[int, int]:
    if not t:
        return 0, 0
    r, c = t.get("rows"), t.get("cols")
    return (len(r) if isinstance(r, list) else int(r or 0)), (len(c) if isinstance(c, list) else int(c or 0))


def decide_kind(blocks: list, W: int, H: int, prior: dict | None = None) -> dict:
    """Page kind from the blocks (rendered or internal: ``box`` dict or list, ``table`` dicts on table blocks).
    sheet: tables cover >= 45 % of the page and the other text is only headers / titles / a few lines
    (text_area < 0.12 or < 4 paragraphs), or a real grid covers >= 20 % with almost no other text (< 5 %, < 1/4 of
    the table area); mixed: a table with >= 2 rows and >= 2 columns; else document.
    ``prior`` = the structure model's kind {"kind", "confidence"}: taken when its confidence >= 0.8 (source "model")."""
    page = max(1.0, float(W) * float(H))
    t_area = text_area = 0.0
    tables, grids, paras = [], [], 0
    for b in blocks:
        cls = b.get("cls") or b.get("label") or "text"
        x0, y0, x1, y1 = _box4(b)
        a = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if cls == "table":
            tables.append(b)
            t_area += a
            R, C = _grid_size(b.get("table"))
            if R >= 2 and C >= 2:
                grids.append(b)
        elif cls in TEXT_CLASSES:
            text_area += a
            if cls in BODY_TEXT:
                ps = b.get("paragraphs") or b.get("paras")
                paras += len(ps) if ps else (1 if b.get("lines") else 0)
    t_area, text_area = min(1.0, t_area / page), min(1.0, text_area / page)
    if tables and t_area >= 0.45 and (text_area < 0.12 or paras < 4):
        kind = "sheet"
        conf = 0.65 + min(0.2, t_area - 0.45) + (0.1 if text_area < 0.12 and paras < 4 else 0.0)
    elif grids and t_area >= SHEET_MIN_AREA and text_area < SHEET_MAX_TEXT and t_area >= 4 * text_area:
        # a smaller table (a landscape price list, a short timetable) that is all the page holds besides a title
        kind = "sheet"
        conf = 0.6 + min(0.25, t_area - SHEET_MIN_AREA)
    elif grids:
        kind = "mixed"
        conf = 0.6 + 0.35 * max(float((b.get("table") or {}).get("confidence") or 0.5) for b in grids)
    else:
        kind = "document"
        conf = 0.95 if not tables else 0.7
    source = "rules"
    if prior and prior.get("kind") in KINDS and float(prior.get("confidence") or 0.0) >= 0.8:
        kind, conf, source = prior["kind"], float(prior["confidence"]), "model"
    return {"kind": kind, "confidence": round(min(0.99, conf), 3), "source": source, "tables": len(tables),
            "table_area": round(t_area, 3), "text_area": round(text_area, 3)}


# ================================================================================================= ruling lines

class Rule:
    """One ruling line: ``pos`` across the line (y for a horizontal rule), extent ``lo..hi`` along it, thickness and
    the inked pieces along it (for "is there a rule between these two cells" tests)."""
    __slots__ = ("pos", "lo", "hi", "thick", "pieces")

    def __init__(self, pos: float, lo: float, hi: float, thick: float, pieces: list):
        self.pos, self.lo, self.hi, self.thick, self.pieces = float(pos), float(lo), float(hi), float(thick), pieces

    def coverage(self, a: float, b: float) -> float:
        if b - a <= 1e-6:
            return 0.0
        cov = sum(max(0.0, min(hi, b) - max(lo, a)) for lo, hi in self.pieces)
        return min(1.0, cov / (b - a))

    def __repr__(self) -> str:
        return f"Rule(pos={self.pos:.1f}, {self.lo:.0f}..{self.hi:.0f}, thick={self.thick:.1f})"


def _union(iv: list) -> list:
    out: list = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _chain_rules(opened: np.ndarray, horizontal: bool, along: int, across: int) -> list[Rule]:
    """Connected pieces of the opened ink -> rules. Pieces are chained along the line when their positions across
    agree within a tolerance (plus a 3 % slope allowance over the gap between them: a skewed photo)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(opened, 8)
    pieces = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i][:4])
        if horizontal:
            pieces.append((float(x), float(x + w), y + h / 2.0, max(1.0, h - 2.0)))
        else:
            pieces.append((float(y), float(y + h), x + w / 2.0, max(1.0, w - 2.0)))
    pieces.sort()
    tol = max(3.0, 0.006 * across)
    chains: list[dict] = []
    for lo, hi, pos, th in pieces:
        best, bd = None, None
        for ch in chains:
            gap = lo - ch["hi"]
            if gap > 0.15 * along:
                continue
            d = abs(pos - ch["pos_end"])
            if d <= tol + 0.03 * max(0.0, gap) and (bd is None or d < bd):
                best, bd = ch, d
        if best is None:
            chains.append({"pieces": [(lo, hi)], "pos": [pos], "th": [th], "lo": lo, "hi": hi, "pos_end": pos})
        else:
            best["pieces"].append((lo, hi)); best["pos"].append(pos); best["th"].append(th)
            best["lo"], best["hi"], best["pos_end"] = min(best["lo"], lo), max(best["hi"], hi), pos
    rules = [Rule(float(np.median(ch["pos"])), ch["lo"], ch["hi"], float(max(ch["th"])), _union(ch["pieces"]))
             for ch in chains if ch["hi"] - ch["lo"] >= MIN_RULE_SPAN * along]
    rules.sort(key=lambda r: r.pos)
    merged: list[Rule] = []  # double rules / a thick line the opening split -> one rule
    tol2 = max(4.0, 0.015 * across)
    for r in rules:
        if merged and r.pos - merged[-1].pos <= tol2:
            m = merged[-1]
            top = min(m.pos - m.thick / 2, r.pos - r.thick / 2)
            bot = max(m.pos + m.thick / 2, r.pos + r.thick / 2)
            merged[-1] = Rule((top + bot) / 2, min(m.lo, r.lo), max(m.hi, r.hi), bot - top, _union(m.pieces + r.pieces))
        else:
            merged.append(r)
    return merged


def find_ruling_lines(ink: np.ndarray) -> tuple[list[Rule], list[Rule]]:
    """Ink of the table crop (0/255 or bool) -> (horizontal rules, vertical rules) in crop pixels."""
    h, w = ink.shape[:2]
    if h < 4 or w < 4:
        return [], []
    bw = (ink > 0).astype(np.uint8)
    kh, kv = max(10, int(round(0.08 * w))), max(10, int(round(0.08 * h)))
    hor = cv2.morphologyEx(cv2.dilate(bw, np.ones((3, 1), np.uint8)), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    ver = cv2.morphologyEx(cv2.dilate(bw, np.ones((1, 3), np.uint8)), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)))
    return _chain_rules(hor, True, w, h), _chain_rules(ver, False, h, w)


def _rule_at(rules: list[Rule], pos: float, tol: float) -> Rule | None:
    best, bd = None, None
    for r in rules:
        d = abs(r.pos - pos)
        if d <= tol and (bd is None or d < bd):
            best, bd = r, d
    return best


# ================================================================================================= bands

def _clusters_1d(intervals: list, min_gap: float) -> list:
    """[(a, b, payload)] -> [[a, b, [payloads]]]: neighbours closer than ``min_gap`` join one cluster (sorted by a)."""
    out: list = []
    for a, b, p in sorted(intervals, key=lambda t: (t[0], t[1])):
        if out and a < out[-1][1] + min_gap:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2].append(p)
        else:
            out.append([a, b, [p]])
    return out


def _bounds_from_clusters(clusters: list, lo: float, hi: float) -> list[float]:
    return [lo] + [(clusters[i][1] + clusters[i + 1][0]) / 2 for i in range(len(clusters) - 1)] + [hi]


def _col_bounds_from_words(rows_words: list, lo: float, hi: float, min_gap: float, max_cross: int, strict_gap: float | None = None) -> list[float]:
    """rows_words: per text row the (x0, x1) word intervals. Boundaries at the centres of the gaps that (a) no row
    crosses, are >= ``strict_gap`` wide and are supported by >= 2 rows with words on both sides (real column gaps are
    often only half a line height in tight Khmer tables), or (b) are >= ``min_gap`` wide and crossed by at most
    ``max_cross`` rows. A gap touching an edge is a margin, not a separator."""
    L = int(np.ceil(hi - lo)) + 1
    if L < 4 or not rows_words:
        return [lo, hi]
    strict_gap = min_gap if strict_gap is None else strict_gap
    cover = np.zeros(L, np.int32)
    rows_iv = []
    for words in rows_words:
        row = np.zeros(L, bool)
        iv = []
        for a, b in words:
            i0, i1 = int(max(0, round(a - lo))), int(min(L, round(b - lo)))
            if i1 > i0:
                row[i0:i1] = True
                iv.append((i0, i1))
        cover += row
        rows_iv.append(iv)
    bounds = [lo]
    i = 0
    while i < L:
        if cover[i] > max_cross:
            i += 1
            continue
        j = i
        while j < L and cover[j] <= max_cross:
            j += 1
        if i > 0 and j < L:
            width = j - i
            if width >= min_gap:
                bounds.append(lo + (i + j) / 2.0)
            else:  # the widest strictly free sub-run inside this low-coverage run
                best, k = None, i
                while k < j:
                    if cover[k] == 0:
                        m = k
                        while m < j and cover[m] == 0:
                            m += 1
                        if best is None or m - k > best[1] - best[0]:
                            best = (k, m)
                        k = m
                    else:
                        k += 1
                if best is not None and best[1] - best[0] >= strict_gap:
                    a, b = best
                    support = sum(1 for iv in rows_iv if any(y <= a for _, y in iv) and any(x >= b for x, _ in iv))
                    if support >= 2:
                        bounds.append(lo + (a + b) / 2.0)
        i = j
    bounds.append(hi)
    return bounds


def _bands(bounds: list[float]) -> list[list[float]]:
    bounds = sorted(bounds)
    return [[a, b] for a, b in zip(bounds, bounds[1:]) if b - a > 1.0]


def _band_of(bands: list, v: float) -> int:
    """Index of the band holding ``v`` (nearest band when outside all of them)."""
    for i, (a, b) in enumerate(bands):
        if a <= v < b:
            return i
    return min(range(len(bands)), key=lambda i: min(abs(v - bands[i][0]), abs(v - bands[i][1])))


def _snap(bounds: list[float], rules: list[Rule], tol: float) -> list[float]:
    out = list(bounds)
    for i in range(1, len(out) - 1):
        r = _rule_at(rules, out[i], tol)
        if r is not None:
            out[i] = r.pos
    return sorted(set(round(v, 2) for v in out))


def _model_bands(boxes: list, axis: int, lo: float, hi: float, rules: list[Rule], size: float) -> list | None:
    """TSR row / column boxes (crop px) -> bands. Sorted by position, NMS by centre (a box whose centre lies inside
    a better-scored one is dropped), boundaries at the mid-gaps, snapped to ruling lines within SNAP_FRAC."""
    kept: list = []
    for b in sorted(boxes, key=lambda b: -b["score"]):
        a0, a1 = b["box"][axis], b["box"][axis + 2]
        if a1 - a0 < 2:
            continue
        c = (a0 + a1) / 2
        if any(k[0] <= c <= k[1] or a0 <= (k[0] + k[1]) / 2 <= a1 for k in kept):
            continue
        kept.append((a0, a1))
    if len(kept) < 2:
        return None
    kept.sort()
    bounds = [lo] + [(kept[i][1] + kept[i + 1][0]) / 2 for i in range(len(kept) - 1)] + [hi]
    bands = _drop_slivers(_bands(_snap(bounds, rules, SNAP_FRAC * size)))
    return bands if len(bands) >= 2 else None


SLIVER_FRAC = 0.12  # integration_qa: a model band thinner than this fraction of the median band is merged away


def _drop_slivers(bands: list) -> list:
    """Bands thinner than SLIVER_FRAC x the median width (a mid-gap snapped onto the frame rule, a stray narrow
    model box at the table edge) are merged into their thinner neighbour: no empty 3-px columns / rows."""
    if len(bands) < 3:
        return bands
    med = float(np.median([b - a for a, b in bands]))
    out = [list(b) for b in bands]
    i = 0
    while i < len(out) and len(out) >= 2:
        a, b = out[i]
        if b - a < SLIVER_FRAC * med:
            if i == 0:
                out[1][0] = a
            elif i == len(out) - 1:
                out[i - 1][1] = b
            elif out[i - 1][1] - out[i - 1][0] <= out[i + 1][1] - out[i + 1][0]:
                out[i - 1][1] = b
            else:
                out[i + 1][0] = a
            del out[i]
            continue
        i += 1
    return out


# ================================================================================================= words / lines

def _words_of(ln: dict, ox: float, pad_x: float) -> list:
    """[(text, x0, x1, bold)] in crop pixels: from the line's word spans (page px), else from the character
    positions (0..1 of the recognition crop, which is the line box padded by pad_x line heights each side)."""
    spans = ln.get("spans") or []
    if spans and all(s.get("x1") is not None for s in spans):
        return [(s["text"], float(s["x0"]) - ox, float(s["x1"]) - ox, bool(s.get("bold"))) for s in spans if s.get("text")]
    text, pos = ln.get("text") or "", ln.get("char_pos") or []
    if text.strip() and len(pos) == len(text):
        x0, y0, x1, y1 = ln["bbox"]
        h = max(1.0, y1 - y0)
        scale, off = (x1 - x0) + 2 * pad_x * h, x0 - pad_x * h - ox
        return [(w, off + a, off + b, bool(ln.get("bold"))) for w, a, b in word_spans(text, pos, int(round(scale)), int(round(h)))]
    return []


def _char_xs(ln: dict, ox: float, pad_x: float) -> list[float] | None:
    text, pos = ln.get("text") or "", ln.get("char_pos") or []
    if not text or len(pos) != len(text):
        return None
    x0, y0, x1, y1 = ln["bbox"]
    h = max(1.0, y1 - y0)
    scale, off = (x1 - x0) + 2 * pad_x * h, x0 - pad_x * h - ox
    return [off + p * scale for p in pos]


def _piece(ln: dict, text: str, x0: float, x1: float, spans: list, conf: float | None = None) -> dict:
    """A new line dict for part of ``ln`` (page pixels)."""
    y0, y1 = ln["bbox"][1], ln["bbox"][3]
    bold = bool(spans) and all(s["bold"] for s in spans) if spans else bool(ln.get("bold"))
    out = {"quad": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), "bbox": [float(x0), float(y0), float(x1), float(y1)],
           "score": float(ln.get("score", 0.0)), "text": text, "confidence": float(ln.get("confidence", 0.0) if conf is None else conf),
           "spans": spans, "bold": bold, "indent_level": 0, "indent": 0, "tab": False, "bullet": None, "split_from": True}
    for k in ("font", "text_color"):
        if ln.get(k) is not None:
            out[k] = ln[k]
    return out


# ================================================================================================= the structure

def _median_h(lines: list, default: float) -> float:
    hs = sorted(max(1.0, l["bbox"][3] - l["bbox"][1]) for l in lines)
    return hs[len(hs) // 2] if hs else default


def _ink_fraction(ink: np.ndarray, box: list, shrink: float) -> float:
    x0, y0, x1, y1 = box
    sx, sy = min(shrink, (x1 - x0) / 4), min(shrink, (y1 - y0) / 4)
    xi0, yi0, xi1, yi1 = int(round(x0 + sx)), int(round(y0 + sy)), int(round(x1 - sx)), int(round(y1 - sy))
    if xi1 - xi0 < 1 or yi1 - yi0 < 1:
        return 0.0
    m = ink[max(0, yi0):yi1, max(0, xi0):xi1]
    return float((m > 0).mean()) if m.size else 0.0


def _paper_level(gray: np.ndarray, ink: np.ndarray, box: list) -> float | None:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    g, m = gray[max(0, y0):y1, max(0, x0):x1], ink[max(0, y0):y1, max(0, x0):x1]
    if g.size < 16:
        return None
    px = g[m == 0]
    return float(np.median(px)) if px.size >= 16 else None


class _Table:
    """Working state of one table in crop pixels."""

    def __init__(self, block: dict, bgr: np.ndarray, ink: np.ndarray, pad_x: float):
        H, W = ink.shape[:2]
        bx0, by0, bx1, by1 = block["box"]
        pad = max(2.0, 0.015 * max(bx1 - bx0, by1 - by0))
        self.ox, self.oy = max(0, int(np.floor(bx0 - pad))), max(0, int(np.floor(by0 - pad)))
        self.ex, self.ey = min(W, int(np.ceil(bx1 + pad))), min(H, int(np.ceil(by1 + pad)))
        self.ink = ink[self.oy:self.ey, self.ox:self.ex]
        self.gray = (bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))[self.oy:self.ey, self.ox:self.ex]
        self.h, self.w = self.ink.shape[:2]
        self.pad_x = pad_x
        # the table extent inside the crop (the box; later the frame rules)
        self.lo_x, self.hi_x = max(0.0, bx0 - self.ox), min(float(self.w), bx1 - self.ox)
        self.lo_y, self.hi_y = max(0.0, by0 - self.oy), min(float(self.h), by1 - self.oy)
        self.lines = list(block["lines"])
        self.L = {id(l): self.to_crop(l["bbox"]) for l in self.lines}
        self.med_h = _median_h(self.lines, max(8.0, 0.03 * self.h))
        self.first_band_rows = 1  # text rows inside the first ruled band (a multi-row header block)

    def to_crop(self, box) -> list[float]:
        return [box[0] - self.ox, box[1] - self.oy, box[2] - self.ox, box[3] - self.oy]

    def to_page(self, box) -> list[float]:
        return [round(box[0] + self.ox, 1), round(box[1] + self.oy, 1), round(box[2] + self.ox, 1), round(box[3] + self.oy, 1)]

    def words(self, ln: dict) -> list:
        return _words_of(ln, self.ox, self.pad_x)


def _text_rows(T: _Table, lines: list, min_gap: float) -> list:
    """Lines -> row clusters [[y0, y1, [lines]]] by the gaps in the y projection of the (slightly shrunk) boxes."""
    iv = []
    for l in lines:
        y0, y1 = T.L[id(l)][1], T.L[id(l)][3]
        s = 0.1 * (y1 - y0)
        iv.append((y0 + s, y1 - s, l))
    return _clusters_1d(iv, min_gap)


def _col_of_words(T: _Table, cols: list, ln: dict) -> list[int]:
    """Column index of every word of the line (by the word centre); the line's own centre when it has no words."""
    ws = T.words(ln)
    if not ws:
        b = T.L[id(ln)]
        return [_band_of(cols, (b[0] + b[2]) / 2)]
    return [_band_of(cols, (a + b) / 2) for _, a, b, _ in ws]


def _merge_continuations(T: _Table, clusters: list, cols: list) -> list:
    """A row cluster whose lines all sit in ONE column, right under a line of the previous cluster in the same
    column that is left- (or right-) aligned with it and was wrapped (reaches the column's text edge), is the
    continuation of that cell (a date / description wrapped onto a second line), not a new row."""
    if len(clusters) < 2:
        return clusters
    # wrapped lines of one cell sit tighter than table rows; a one-column table (no other column to compare with)
    # only merges clearly tight, justified continuation lines
    gap_max = (0.7 if len(cols) >= 2 else 0.4) * T.med_h
    wrap_tol = (1.0 if len(cols) >= 2 else 2.0)
    col_right = {}
    for c in clusters:
        for l in c[2]:
            for k in set(_col_of_words(T, cols, l)):
                col_right[k] = max(col_right.get(k, -1e9), T.L[id(l)][2])
    out = [clusters[0]]
    for cl in clusters[1:]:
        prev = out[-1]
        ks = {k for l in cl[2] for k in _col_of_words(T, cols, l)}
        merged = False
        if len(ks) == 1:
            k = next(iter(ks))
            above = [l for l in prev[2] if k in set(_col_of_words(T, cols, l))]
            if above:
                a = max(above, key=lambda l: T.L[id(l)][3])
                ab, bb = T.L[id(a)], T.L[id(cl[2][0])]
                h = max(1.0, ab[3] - ab[1])
                aligned = abs(ab[0] - bb[0]) < 0.6 * h or abs(ab[2] - bb[2]) < 0.6 * h
                wrapped = ab[2] >= col_right.get(k, ab[2]) - wrap_tol * h
                if aligned and wrapped and bb[1] - ab[3] < gap_max:
                    merged = True
        if merged:
            prev[1] = max(prev[1], cl[1]); prev[2].extend(cl[2])
        else:
            out.append(cl)
    return out


def _row_bounds_rules(T: _Table, hl: list[Rule], cols: list, lines: list, min_gap: float) -> tuple[list[float], bool]:
    """Row boundaries from the horizontal rules; a band holding several text rows that look like full rows (>= 2
    clusters with text in >= 2 columns) is sub-divided by the text - a ruled header plus a long unruled body.
    -> (bounds, sub-divided?)"""
    inner = [r.pos for r in hl if T.lo_y + 2 < r.pos < T.hi_y - 2]
    bounds = sorted(set([T.lo_y] + inner + [T.hi_y]))
    # an empty sliver between the table edge and the first / last rule (the layout box reaching past the frame)
    def empty(a, b):
        return not any(a <= (T.L[id(l)][1] + T.L[id(l)][3]) / 2 < b for l in lines) and b - a < 1.5 * T.med_h
    if len(bounds) > 2 and empty(bounds[0], bounds[1]):
        bounds = bounds[1:]
    if len(bounds) > 2 and empty(bounds[-2], bounds[-1]):
        bounds = bounds[:-1]
    T.lo_y, T.hi_y = bounds[0], bounds[-1]
    sub = False
    first_band_rows = 1
    out = [bounds[0]]
    for bi, (a, b) in enumerate(zip(bounds, bounds[1:])):
        inside = [l for l in lines if a <= (T.L[id(l)][1] + T.L[id(l)][3]) / 2 < b]
        clusters = _text_rows(T, inside, min_gap) if len(inside) >= 2 else []
        if len(clusters) >= 2:
            clusters = _merge_continuations(T, clusters, cols)
            full = sum(1 for c in clusters if len({k for l in c[2] for k in _col_of_words(T, cols, l)}) >= 2)
            if len(clusters) >= 2 and (full >= 2 or len(cols) == 1):
                out += _bounds_from_clusters(clusters, a, b)[1:-1]
                sub = True
                if bi == 0:
                    first_band_rows = len(clusters)
        out.append(b)
    T.first_band_rows = first_band_rows
    return out, sub


def _build_rules_grid(T: _Table, hl: list[Rule], vl: list[Rule]) -> dict:
    """Rows / columns / borders from the rules and the text; -> {rows, cols, borders, hl, vl}."""
    tol_x, tol_y = max(4.0, _RULE_TOL * T.w), max(4.0, _RULE_TOL * T.h)
    # frame rules move the table extent onto the drawn border
    for r in vl:
        if abs(r.pos - T.lo_x) <= tol_x * 1.5:
            T.lo_x = r.pos
        if abs(r.pos - T.hi_x) <= tol_x * 1.5:
            T.hi_x = r.pos
    for r in hl:
        if abs(r.pos - T.lo_y) <= tol_y * 1.5:
            T.lo_y = r.pos
        if abs(r.pos - T.hi_y) <= tol_y * 1.5:
            T.hi_y = r.pos
    inner_v = [r for r in vl if T.lo_x + tol_x < r.pos < T.hi_x - tol_x]
    inner_h = [r for r in hl if T.lo_y + tol_y < r.pos < T.hi_y - tol_y]
    lines = [l for l in T.lines if l.get("text") or not l.get("split_from")]
    min_gap_y = max(2.0, 0.3 * T.med_h)
    # columns
    if inner_v:
        cols = _bands(sorted(set([T.lo_x] + [r.pos for r in inner_v] + [T.hi_x])))
    else:
        clusters = _text_rows(T, lines, min_gap_y)
        rows_words = []
        for c in clusters:
            ws = [(a, b) for l in c[2] for _, a, b, _ in T.words(l)] or [(T.L[id(l)][0], T.L[id(l)][2]) for l in c[2]]
            rows_words.append(ws)
        max_cross = int(np.floor(0.12 * len(rows_words))) if len(rows_words) >= 8 else 0
        cols = _bands(_col_bounds_from_words(rows_words, T.lo_x, T.hi_x, max(4.0, 0.6 * T.med_h), max_cross, max(4.0, 0.35 * T.med_h)))
    # rows
    if inner_h:
        bounds, _ = _row_bounds_rules(T, hl, cols, lines, min_gap_y)
        rows = _bands(bounds)
    else:
        clusters = _merge_continuations(T, _text_rows(T, lines, min_gap_y), cols)
        rows = _bands(_bounds_from_clusters(clusters, T.lo_y, T.hi_y)) if clusters else [[T.lo_y, T.hi_y]]
    borders = "grid" if (inner_v and inner_h) else ("hlines" if inner_h or (len(hl) >= 2 and not inner_v) else "none")
    if borders == "hlines" and not inner_h and len(hl) >= 2 and len(rows) < 2:
        borders = "none"
    return {"rows": rows or [[T.lo_y, T.hi_y]], "cols": cols or [[T.lo_x, T.hi_x]], "borders": borders,
            "inner_v": inner_v, "inner_h": inner_h}


def _struct_grid(T: _Table, struct: dict) -> dict | None:
    """Structure-model cells (groups of this block's lines) -> bands; None when the groups are too thin."""
    R, C = int(struct.get("rows") or 0), int(struct.get("cols") or 0)
    cells = [c for c in struct.get("cells") or [] if c.get("lines")]
    if R < 1 or C < 1 or not cells:
        return None
    ys: dict[int, list] = {}
    xs: dict[int, list] = {}
    for c in cells:
        bbs = [T.L[id(l)] for l in c["lines"] if id(l) in T.L]
        if not bbs:
            continue
        y0, y1 = min(b[1] for b in bbs), max(b[3] for b in bbs)
        x0, x1 = min(b[0] for b in bbs), max(b[2] for b in bbs)
        if int(c.get("rs", 1)) == 1:
            ys.setdefault(int(c["r"]), []).append((y0, y1))
        if int(c.get("cs", 1)) == 1:
            xs.setdefault(int(c["c"]), []).append((x0, x1))

    def bands_from(ext: dict, n: int, lo: float, hi: float):
        ks = sorted(ext)
        if len(ks) < 1:
            return None
        spans = [(min(a for a, _ in ext[k]), max(b for _, b in ext[k])) for k in ks]
        bounds = [lo] + [(spans[i][1] + spans[i + 1][0]) / 2 for i in range(len(spans) - 1)] + [hi]
        return _bands(bounds), ks
    ry, rx = bands_from(ys, R, T.lo_y, T.hi_y), bands_from(xs, C, T.lo_x, T.hi_x)
    if ry is None or rx is None:
        return None
    rows, rk = ry
    cols, ck = rx
    rmap, cmap = {k: i for i, k in enumerate(rk)}, {k: i for i, k in enumerate(ck)}
    spans = []
    for c in cells:
        r0, c0 = rmap.get(int(c["r"])), cmap.get(int(c["c"]))
        rs, cs = int(c.get("rs", 1)), int(c.get("cs", 1))
        if r0 is None or c0 is None:
            continue
        r1 = rmap.get(int(c["r"]) + rs - 1, r0)
        c1 = cmap.get(int(c["c"]) + cs - 1, c0)
        if r1 > r0 or c1 > c0:
            spans.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1, 1.0))
    return {"rows": rows, "cols": cols, "spans": spans}


def _place_spans(R: int, C: int, spans: list) -> dict:
    """spans [(r, c, rs, cs, score)] -> owner map {(r, c): (r0, c0, rs, cs)}; overlaps resolved by score."""
    owner: dict = {}
    for r0, c0, rs, cs, sc in sorted(spans, key=lambda s: -s[4]):
        rs, cs = max(1, min(rs, R - r0)), max(1, min(cs, C - c0))
        if r0 < 0 or c0 < 0 or r0 >= R or c0 >= C or (rs == 1 and cs == 1):
            continue
        keys = [(r, c) for r in range(r0, r0 + rs) for c in range(c0, c0 + cs)]
        if any(k in owner for k in keys):
            continue
        for k in keys:
            owner[k] = (r0, c0, rs, cs)
    return owner


def _line_cols(T: _Table, cols: list, ln: dict) -> list[int]:
    """Sorted distinct columns the line's words fall in; without words the bands its box overlaps by > 0.3 h."""
    ws = T.words(ln)
    if ws:
        return sorted(set(_band_of(cols, (a + b) / 2) for _, a, b, _ in ws))
    b = T.L[id(ln)]
    h = max(1.0, b[3] - b[1])
    ks = [i for i, (a, c) in enumerate(cols) if min(b[2], c) - max(b[0], a) > min(0.3 * h, 0.5 * (c - a))]
    return ks or [_band_of(cols, (b[0] + b[2]) / 2)]


def _detect_spans(T: _Table, grid: dict, vl: list[Rule], hl: list[Rule], model_spans: list) -> dict:
    """Spanning cells -> owner map. The model's spans first; then, in a table with vertical rules, a run of columns
    with no rule between them inside a row band merges when its text sits in at most one of the columns or crosses
    them; without vertical rules only a line centred over the bands it covers (nothing else of its row there) spans;
    with horizontal rules, two stacked cells merge where the rule between them is missing and the text crosses the
    boundary or is centred over the pair with the other cell empty."""
    rows, cols = grid["rows"], grid["cols"]
    R, C = len(rows), len(cols)
    has_v, has_h = bool(grid.get("inner_v")), bool(grid.get("inner_h"))
    tol_x, tol_y = max(4.0, _RULE_TOL * T.w), max(4.0, _RULE_TOL * T.h)
    spans = list(model_spans)
    by_row: dict[int, list] = {}
    lcols: dict[int, list] = {}
    for l in T.lines:
        b = T.L[id(l)]
        by_row.setdefault(_band_of(rows, (b[1] + b[3]) / 2), []).append(l)
        lcols[id(l)] = _line_cols(T, cols, l)
    for r in range(R):
        lns = by_row.get(r, [])
        y0, y1 = rows[r]
        if has_v:
            runs, run = [], [0]
            for k in range(C - 1):
                rule = _rule_at(vl, cols[k][1], tol_x)
                if rule is not None and rule.coverage(y0, y1) >= 0.5:
                    runs.append(run); run = [k + 1]
                else:
                    run.append(k + 1)
            runs.append(run)
            for rn in runs:
                if len(rn) < 2:
                    continue
                inside = [l for l in lns if set(lcols[id(l)]) & set(rn)]
                if not inside:
                    continue
                occupied = {k for l in inside for k in lcols[id(l)] if k in rn}
                crossing = any(len([k for k in lcols[id(l)] if k in rn]) >= 2 for l in inside)
                if crossing or len(occupied) <= 1:
                    spans.append((r, rn[0], 1, len(rn), 0.5))
        else:
            for l in lns:
                ks = lcols[id(l)]
                if len(ks) < 2:
                    continue
                c0, c1 = ks[0], ks[-1]
                b = T.L[id(l)]
                sx0, sx1 = cols[c0][0], cols[c1][1]
                centred = abs((b[0] + b[2]) / 2 - (sx0 + sx1) / 2) < 0.15 * (sx1 - sx0)
                others = [o for o in lns if o is not l and any(c0 <= k <= c1 for k in lcols[id(o)])]
                if centred and not others:
                    spans.append((r, c0, 1, c1 - c0 + 1, 0.5))
    if has_h:
        for c in range(C):
            x0, x1 = cols[c]
            for r in range(R - 1):
                rule = _rule_at(hl, rows[r][1], tol_y)
                if rule is None or rule.coverage(x0, x1) >= 0.5:
                    continue
                top = [l for l in by_row.get(r, []) if c in lcols[id(l)]]
                bot = [l for l in by_row.get(r + 1, []) if c in lcols[id(l)]]
                crossing = any(min(T.L[id(l)][3], rows[r + 1][1]) - max(T.L[id(l)][1], rows[r + 1][0]) > 0.25 * (T.L[id(l)][3] - T.L[id(l)][1]) for l in top) \
                    or any(min(T.L[id(l)][3], rows[r][1]) - max(T.L[id(l)][1], rows[r][0]) > 0.25 * (T.L[id(l)][3] - T.L[id(l)][1]) for l in bot)
                centred = False
                if (top and not bot) or (bot and not top):
                    lns_ = top or bot
                    cy = (min(T.L[id(l)][1] for l in lns_) + max(T.L[id(l)][3] for l in lns_)) / 2
                    my0, my1 = rows[r][0], rows[r + 1][1]
                    centred = abs(cy - (my0 + my1) / 2) < 0.2 * (my1 - my0)
                if crossing or centred:
                    spans.append((r, c, 2, 1, 0.4))
    return _place_spans(R, C, spans)


def _split_crossing_lines(T: _Table, grid: dict, owner: dict, read_boxes) -> tuple[list, int, list]:
    """Lines whose words lie in several cells are cut at the cell boundaries: with word spans / character positions
    directly, else by re-recognising the pieces (``read_boxes``, batched). -> (new line list, number split, pending
    re-recognition jobs [(line, pieces boxes)])."""
    rows, cols = grid["rows"], grid["cols"]
    out, n_split, jobs = [], 0, []

    def cell_key(r, c):
        o = owner.get((r, c))
        return (o[0], o[1]) if o else (r, c)

    for l in T.lines:
        b = T.L[id(l)]
        r = _band_of(rows, (b[1] + b[3]) / 2)
        ws = T.words(l)
        keys = [cell_key(r, _band_of(cols, (a + c) / 2)) for _, a, c, _ in ws] if ws else [cell_key(r, k) for k in _line_cols(T, cols, l)]
        if len(set(keys)) <= 1:
            out.append(l)
            continue
        h = max(1.0, b[3] - b[1])
        if ws:  # group consecutive words by cell
            groups: list[list] = []
            for w, k in zip(ws, keys):
                if groups and groups[-1][0] == k:
                    groups[-1][1].append(w)
                else:
                    groups.append([k, [w]])
            spans_all = l.get("spans") or []
            for k, grp in groups:
                texts = [w[0] for w in grp]
                x0 = min(w[1] for w in grp) + T.ox
                x1 = max(w[2] for w in grp) + T.ox
                sp = [{"text": w[0], "x0": round(w[1] + T.ox, 1), "x1": round(w[2] + T.ox, 1), "bold": bool(w[3])} for w in grp] if spans_all else []
                out.append(_piece(l, " ".join(texts).strip(), max(l["bbox"][0], x0), min(l["bbox"][2], x1), sp))
            n_split += 1
            continue
        if read_boxes is None:
            out.append(l)
            continue
        # no positions: cut the box at the column boundaries and read every piece
        ks = sorted(set(k for k in _line_cols(T, cols, l)))
        pieces = []
        for k in ks:
            cx0, cx1 = max(b[0], cols[k][0]), min(b[2], cols[k][1])
            if cx1 - cx0 >= 0.5 * h:
                pieces.append([cx0 + T.ox, l["bbox"][1], cx1 + T.ox, l["bbox"][3]])
        if len(pieces) < 2:
            out.append(l)
            continue
        new = [_piece(l, "", p[0], p[2], [], 0.0) for p in pieces]
        jobs.append((new, pieces))
        out.extend(new)
        n_split += 1
    return out, n_split, jobs


def _family(l: dict) -> str | None:
    f = l.get("font") or {}
    return f.get("family") if float(f.get("confidence") or 0.0) >= 0.5 else None


def _header_rows(T: _Table, grid: dict, cells: list, hl: list[Rule]) -> int:
    """1 when the first row is bold, shaded, set in another typeface than the body or separated by a heavier (or
    the only) rule; the number of text rows of the first ruled band when that band is a multi-row header block."""
    rows, cols = grid["rows"], grid["cols"]
    R, C = len(rows), len(cols)
    if R < 2:
        return 0
    n_head = max(1, min(int(getattr(T, "first_band_rows", 1)), R - 1))
    head = [c for c in cells if c["r"] < n_head and c["rs"] == 1]
    filled0 = [c for c in head if c["_lines"] and c["r"] == 0]
    if len(filled0) < min(2, C):
        return 0
    body = [c for c in cells if c["r"] >= n_head and c["_lines"]]
    bold0 = all(c["bold"] for c in filled0)
    bold = bold0 and not (bool(body) and all(c["bold"] for c in body))
    y0, y1 = rows[0][0], rows[n_head - 1][1]
    p0 = _paper_level(T.gray, T.ink, [T.lo_x, y0, T.hi_x, y1])
    p_rest = _paper_level(T.gray, T.ink, [T.lo_x, rows[n_head][0], T.hi_x, rows[-1][1]])
    shaded = p0 is not None and p_rest is not None and p_rest - p0 >= 10
    fam0 = {_family(l) for c in filled0 for l in c["_lines"]}
    fam_body = [_family(l) for c in body for l in c["_lines"] if _family(l)]
    face = bool(fam_body) and None not in fam0 and len(fam0) == 1 and next(iter(fam0)) != max(set(fam_body), key=fam_body.count) \
        and fam_body.count(max(set(fam_body), key=fam_body.count)) >= 0.7 * len(fam_body)
    heavier = False
    tol_y = max(4.0, _RULE_TOL * T.h)
    rule = _rule_at(hl, y1, tol_y)
    if rule is not None:
        others = [r.thick for r in hl if r is not rule and T.lo_y + tol_y < r.pos < T.hi_y - tol_y]
        if others:
            heavier = rule.thick >= 1.5 * float(np.median(others)) and rule.thick >= 2.0
        elif len(hl) <= 3:  # the only inner rule is the one under the header
            heavier = True
    return n_head if (bold or shaded or face or heavier) else 0


def _cells_from_grid(T: _Table, grid: dict, owner: dict) -> list[dict]:
    rows, cols = grid["rows"], grid["cols"]
    R, C = len(rows), len(cols)
    cells, seen = [], set()
    for r in range(R):
        for c in range(C):
            o = owner.get((r, c))
            if o is not None:
                if (o[0], o[1]) in seen:
                    continue
                seen.add((o[0], o[1]))
                r0, c0, rs, cs = o
            else:
                r0, c0, rs, cs = r, c, 1, 1
            box = [cols[c0][0], rows[r0][0], cols[c0 + cs - 1][1], rows[r0 + rs - 1][1]]
            cells.append({"r": r0, "c": c0, "rs": rs, "cs": cs, "_box": box, "_lines": []})
    cells.sort(key=lambda d: (d["r"], d["c"]))
    return cells


def _assign_lines(T: _Table, cells: list, grid: dict, owner: dict) -> int:
    """Every line to the cell holding its centre (the owner of that grid position); -> lines placed."""
    rows, cols = grid["rows"], grid["cols"]
    at = {(c["r"], c["c"]): c for c in cells}
    n = 0
    for l in T.lines:
        b = T.L[id(l)]
        r, c = _band_of(rows, (b[1] + b[3]) / 2), _band_of(cols, (b[0] + b[2]) / 2)
        o = owner.get((r, c))
        cell = at.get((o[0], o[1]) if o else (r, c))
        if cell is not None:
            cell["_lines"].append(l); n += 1
    for cell in cells:
        cell["_lines"] = sort_lines(cell["_lines"], key=lambda l: T.L[id(l)])
    return n


def _finish_cells(T: _Table, cells: list, header_rows: int) -> None:
    rule_px = 3.0
    for cell in cells:
        lns = [l for l in cell["_lines"]]
        x0, y0, x1, y1 = cell["_box"]
        texted = [l for l in lns if (l.get("text") or "").strip()]
        runs = paragraph_runs(texted) if texted else []
        text = "".join(r["text"] for r in runs).strip()
        if lns:
            lx0, lx1 = min(T.L[id(l)][0] for l in lns), max(T.L[id(l)][2] for l in lns)
            left, right = lx0 - x0, x1 - lx1
            w = max(1.0, x1 - x0)
            if left < 0.12 * w and right < 0.12 * w:
                align = "l"
            elif abs(left - right) < 0.25 * max(left, right, 1.0):
                align = "c"
            elif right < left:
                align = "r"
            else:
                align = "l"
        else:
            align = "l"
        if not text and _ink_fraction(T.ink, cell["_box"], rule_px + 2) < 0.002:
            conf = 1.0
        elif texted:
            conf = float(np.mean([float(l.get("confidence", 0.0)) for l in texted]))
        else:
            conf = 0.0  # ink but nothing read
        cell.update(text=text, align=align, bold=bool(texted) and all(bool(l.get("bold")) for l in texted),
                    header=cell["r"] < header_rows, confidence=round(conf, 4))


def build_table(block: dict, bgr: np.ndarray, ink: np.ndarray, model_boxes: list | None = None, model_id: str | None = None,
                struct_table: dict | None = None, read_boxes=None, pad_x: float = 0.25) -> dict:
    """Structure of one ``table`` block (page pixels in, page pixels out). Mutates ``block["lines"]`` (crossing lines
    split into pieces, lines re-ordered cell by cell) and ``block["paras"]`` (one paragraph per non-empty cell).
    ``model_boxes``: [{"cls", "score", "box" (page px)}] of the TSR model; ``struct_table``: {"rows", "cols",
    "cells": [{"r", "c", "rs", "cs", "lines": [line dicts]}]} of the structure transformer; ``read_boxes(boxes)
    -> [(text, confidence)]`` re-recognises page boxes. Returns ``block["table"]``."""
    T = _Table(block, bgr, ink, pad_x)
    if T.w < 8 or T.h < 8:
        tab = {"rows": [[block["box"][1], block["box"][3]]], "cols": [[block["box"][0], block["box"][2]]], "header_rows": 0,
               "borders": "none", "cells": [{"r": 0, "c": 0, "rs": 1, "cs": 1, "box": list(block["box"]), "header": False,
                                             "text": "", "_lines": list(block["lines"]), "align": "l", "bold": False, "confidence": 0.0}],
               "source": "rules", "confidence": 0.0}
        block["table"] = tab
        return tab
    hl, vl = find_ruling_lines(T.ink)
    grid = _build_rules_grid(T, hl, vl)
    source, model_spans = "rules", []
    # 1. the TSR model's bands
    if model_boxes:
        mb = [{"cls": b["cls"], "score": float(b["score"]), "box": T.to_crop(b["box"])} for b in model_boxes]
        rows = _model_bands([b for b in mb if b["cls"] == "row"], 1, T.lo_y, T.hi_y, hl, T.h)
        cols = _model_bands([b for b in mb if b["cls"] == "column"], 0, T.lo_x, T.hi_x, vl, T.w)
        if rows is not None or cols is not None:
            source = model_id or "tsr_model"
            if rows is not None:
                grid["rows"] = rows
            if cols is not None:
                grid["cols"] = cols
            grid["model_axes"] = {"rows": rows is not None, "cols": cols is not None}
            R, C = len(grid["rows"]), len(grid["cols"])
            for b in sorted(mb, key=lambda b: -b["score"]):
                if b["cls"] not in ("spanning_cell", "projected_row_header"):
                    continue
                x0, y0, x1, y1 = b["box"]
                rs_ = [i for i, (a, c) in enumerate(grid["rows"]) if min(y1, c) - max(y0, a) >= 0.5 * (c - a)]
                cs_ = [i for i, (a, c) in enumerate(grid["cols"]) if min(x1, c) - max(x0, a) >= 0.5 * (c - a)]
                if b["cls"] == "projected_row_header":
                    cs_ = list(range(C))
                if rs_ and cs_ and (len(rs_) > 1 or len(cs_) > 1):
                    model_spans.append((rs_[0], cs_[0], rs_[-1] - rs_[0] + 1, cs_[-1] - cs_[0] + 1, 1.0 + b["score"]))
            heads = [b for b in mb if b["cls"] == "column_header"]
            grid["model_header"] = max((sum(1 for a, c in grid["rows"] if b["box"][1] <= (a + c) / 2 <= b["box"][3]) for b in heads), default=None)
    # 2. the structure transformer's cells for borderless tables
    if source == "rules" and struct_table and grid["borders"] == "none":
        sg = _struct_grid(T, struct_table)
        if sg is not None and len(sg["rows"]) * len(sg["cols"]) >= 2:
            grid["rows"], grid["cols"] = sg["rows"], sg["cols"]
            model_spans = [(r, c, rs, cs, 1.0) for r, c, rs, cs, _ in sg["spans"]]
            source = "struct_v3"
    owner = _detect_spans(T, grid, vl, hl, model_spans)
    T.lines, n_split, jobs = _split_crossing_lines(T, grid, owner, read_boxes)
    if jobs and read_boxes is not None:
        boxes = [p for _, pcs in jobs for p in pcs]
        try:
            results = list(read_boxes(boxes))
        except Exception:
            results = []
        if len(results) == len(boxes):
            i = 0
            for new, pcs in jobs:
                for ln in new:
                    t, cf = results[i]; i += 1
                    ln["text"], ln["confidence"] = (t or "").strip(), float(cf)
    T.L = {id(l): T.to_crop(l["bbox"]) for l in T.lines}
    cells = _cells_from_grid(T, grid, owner)
    _assign_lines(T, cells, grid, owner)
    for cell in cells:  # a bold flag per cell before the header test
        texted = [l for l in cell["_lines"] if (l.get("text") or "").strip()]
        cell["bold"] = bool(texted) and all(bool(l.get("bold")) for l in texted)
    header_rows = _header_rows(T, grid, cells, hl)
    if grid.get("model_header") is not None:
        header_rows = int(grid["model_header"])
    _finish_cells(T, cells, header_rows)
    # the block's lines in cell order; one paragraph per non-empty cell
    ordered = [l for cell in cells for l in cell["_lines"]]
    seen = {id(l) for l in ordered}
    block["lines"] = ordered + [l for l in T.lines if id(l) not in seen]
    block["paras"] = [{"lines": list(cell["_lines"]), "bullet": None, "anchor": cell["_lines"][0]["bbox"][0], "indent_level": 0}
                      for cell in cells if cell["_lines"]]
    fit = 1.0 - (n_split / max(1, len(block["lines"]))) if block["lines"] else 1.0
    base = {"grid": 0.9, "hlines": 0.75, "none": 0.6}[grid["borders"]]
    if source != "rules":
        base = 0.85
    tab = {"rows": [[round(a + T.oy, 1), round(b + T.oy, 1)] for a, b in grid["rows"]],
           "cols": [[round(a + T.ox, 1), round(b + T.ox, 1)] for a, b in grid["cols"]],
           "header_rows": header_rows, "borders": grid["borders"],
           "cells": [{"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": T.to_page(c["_box"]), "header": c["header"],
                      "text": c["text"], "_lines": c["_lines"], "align": c["align"], "bold": c["bold"], "confidence": c["confidence"]}
                     for c in cells],
           "source": source, "confidence": round(base * (0.5 + 0.5 * fit), 3),
           "rules": {"h": len(hl), "v": len(vl)}, "lines_split": n_split}
    block["table"] = tab
    return tab


# ================================================================================================= page level

GRID_KEEP = {"stamp", "signature", "logo", "qr_code", "barcode", "header", "footer", "page_number", "title"}


def _gaps_clusters(vals: list, gap: float) -> list[list[float]]:
    out: list = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def find_grid_tables(ink: np.ndarray, blocks: list) -> list[list[float]]:
    """Ruled tables the layout model missed (a photographed spreadsheet / form read as text or figure): connected
    groups of horizontal + vertical ruling lines whose crossings form >= 3 distinct columns and >= 3 distinct rows of
    crossings (a ruled notebook page with one margin line, a framed box or an underline do not). Tilt-tolerant:
    crossings are clustered by gaps, not by exact positions. -> page boxes [x0, y0, x1, y1] that no layout ``table``
    block already covers. The caller turns them into table blocks."""
    H, W = ink.shape[:2]
    if H < 50 or W < 50:
        return []
    bw = (ink > 0).astype(np.uint8)
    kh, kv = max(15, int(round(0.025 * W))), max(15, int(round(0.02 * H)))
    hor = cv2.morphologyEx(cv2.dilate(bw, np.ones((3, 1), np.uint8)), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    ver = cv2.morphologyEx(cv2.dilate(bw, np.ones((1, 3), np.uint8)), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)))
    k5 = np.ones((5, 5), np.uint8)
    hd, vd = cv2.dilate(hor, k5), cv2.dilate(ver, k5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hd | vd, 8)
    xn, xlab, xst, xcen = cv2.connectedComponentsWithStats(hd & vd, 8)
    tables_now = [_box4(b) for b in blocks if (b.get("cls") or "") == "table"]
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i][:4])
        if w < 0.12 * W or h < 0.04 * H or w * h < 0.015 * W * H:
            continue
        pts = [xcen[j] for j in range(1, xn) if lab[int(xcen[j][1]), int(xcen[j][0])] == i]
        if len(pts) < 9:
            continue
        cols = [c for c in _gaps_clusters([p[0] for p in pts], max(6.0, 0.04 * w)) if len(c) >= 2]
        rows = [r for r in _gaps_clusters([p[1] for p in pts], max(6.0, 0.03 * h)) if len(r) >= 2]
        if len(cols) < 3 or len(rows) < 3:
            continue
        box = [float(x), float(y), float(x + w), float(y + h)]
        if any(_overlap(box, t) >= 0.3 or _overlap(t, box) >= 0.5 for t in tables_now):
            continue
        out.append(box)
    return out


def _overlap(a: list, b: list) -> float:
    """Share of ``a``'s area inside ``b``."""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))



DESKEW_MIN_DEG = 0.7     # tables whose text lines lean more than this are straightened before the structure pass
DESKEW_MAX_DEG = 20.0


def table_skew(block: dict) -> float:
    """Median lean (degrees, image coordinates: positive = the line falls to the right) of the block's long text
    lines, from the top edge of their quads; 0.0 when there are fewer than 3 usable lines."""
    angs, ws = [], []
    for l in block.get("lines", []):
        q = l.get("quad")
        if q is None:
            continue
        q = np.asarray(q, np.float32).reshape(-1, 2)
        if q.shape[0] != 4:
            continue
        dx, dy = float(q[1, 0] - q[0, 0]), float(q[1, 1] - q[0, 1])
        h = float(np.hypot(q[3, 0] - q[0, 0], q[3, 1] - q[0, 1]))
        w = float(np.hypot(dx, dy))
        if w < 2.5 * max(1.0, h) or dx <= 0:
            continue
        angs.append(np.degrees(np.arctan2(dy, dx)))
        ws.append(w)
    if len(angs) < 3:
        return 0.0
    tilted = [i for i, a in enumerate(angs) if abs(a) >= 0.5]
    if len(tilted) >= max(3, 0.3 * len(angs)):  # axis-aligned quads are re-drawn boxes (snapped / fallback lines)
        angs, ws = [angs[i] for i in tilted], [ws[i] for i in tilted]
    order = np.argsort(angs)
    a, w = np.asarray(angs)[order], np.asarray(ws)[order]
    return float(a[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])


def _tf(M: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    return p @ M[:, :2].T + M[:, 2]


def _bbox_of(pts: np.ndarray) -> list[float]:
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def _build_deskewed(b: dict, bgr: np.ndarray, ink: np.ndarray, angle: float, tsr, st, read_boxes, pad_x: float) -> dict:
    """``build_table`` on a straightened copy of the page around the table (rotation by ``angle`` about the block
    centre), results mapped back to page pixels: cell boxes = bounding boxes of the rotated-back cells, row / column
    bands taken along the table's centre lines, split line pieces get rotated-back quads, the original line dicts
    are kept (by identity) everywhere else."""
    H, W = ink.shape[:2]
    x0, y0, x1, y1 = b["box"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)  # rotates the leaning content back to horizontal
    Mi = cv2.invertAffineTransform(M)
    rbgr = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rink = cv2.warpAffine(ink, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    # the table rectangle inscribed in the axis-aligned block box (the box holds the leaning rectangle)
    t = np.radians(abs(angle))
    cs_, sn = np.cos(t), np.sin(t)
    BW, BH = x1 - x0, y1 - y0
    den = cs_ * cs_ - sn * sn
    w_in = (BW * cs_ - BH * sn) / den if den > 1e-3 else BW
    h_in = (BH * cs_ - BW * sn) / den if den > 1e-3 else BH
    w_in, h_in = (w_in if w_in > 0.3 * BW else BW), (h_in if h_in > 0.3 * BH else BH)
    rbox = [cx - w_in / 2, cy - h_in / 2, cx + w_in / 2, cy + h_in / 2]
    back: dict = {}
    rlines = []
    for l in b["lines"]:
        nl = dict(l)
        q = np.asarray(l.get("quad") if l.get("quad") is not None else
                       [[l["bbox"][0], l["bbox"][1]], [l["bbox"][2], l["bbox"][1]], [l["bbox"][2], l["bbox"][3]], [l["bbox"][0], l["bbox"][3]]], np.float32)
        rq = _tf(M, q)
        nl["quad"], nl["bbox"] = rq.astype(np.float32), _bbox_of(rq)
        if l.get("spans"):
            ymid = (l["bbox"][1] + l["bbox"][3]) / 2
            sp = []
            for s_ in l["spans"]:
                s2 = dict(s_)
                if s_.get("x0") is not None and s_.get("x1") is not None:
                    s2["x0"] = float(_tf(M, [s_["x0"], ymid])[0, 0])
                    s2["x1"] = float(_tf(M, [s_["x1"], ymid])[0, 0])
                sp.append(s2)
            nl["spans"] = sp
        rbox = [min(rbox[0], nl["bbox"][0]), min(rbox[1], nl["bbox"][1]), max(rbox[2], nl["bbox"][2]), max(rbox[3], nl["bbox"][3])]
        back[id(nl)] = l
        rlines.append(nl)
    rbox = [max(0.0, rbox[0]), max(0.0, rbox[1]), min(float(W), rbox[2]), min(float(H), rbox[3])]
    rb = {**b, "box": rbox, "lines": rlines}
    model_boxes, model_id = None, None
    if tsr is not None:
        try:
            model_boxes = tsr.detect(rbgr, rbox)
            model_id = getattr(getattr(tsr, "spec", None), "id", None) or "tsr_model"
        except Exception:
            model_boxes = None
    rst = None
    if st is not None:
        by_orig = {id(back[id(nl)]): nl for nl in rlines}
        rst = {**st, "cells": [{**c, "lines": [by_orig[id(x)] for x in c.get("lines", []) if id(x) in by_orig]} for c in st["cells"]]}
    rread = None
    if read_boxes is not None:
        def rread(boxes):  # pieces cut in the straightened frame are read from the page along the leaning quad
            return read_boxes([_tf(Mi, [[a, c], [e, c], [e, d], [a, d]]).tolist() for a, c, e, d in boxes])
    tab = build_table(rb, rbgr, rink, model_boxes, model_id, rst, rread, pad_x)

    def orig(l):
        o = back.get(id(l))
        if o is not None:
            return o
        if not l.get("_deskewed_back"):  # a new piece: rotate it back to the page
            q = _tf(Mi, l["quad"])
            l["quad"], l["bbox"] = q.astype(np.float32), _bbox_of(q)
            if l.get("spans"):
                ymid = float(np.mean(q[:, 1]))
                for s_ in l["spans"]:
                    if s_.get("x0") is not None:
                        s_["x0"] = round(float(_tf(Mi, [s_["x0"], ymid])[0, 0]), 1)
                        s_["x1"] = round(float(_tf(Mi, [s_["x1"], ymid])[0, 0]), 1)
            l["_deskewed_back"] = True
        return l

    b["lines"] = [orig(l) for l in rb["lines"]]
    for l in b["lines"]:
        l.pop("_deskewed_back", None)
    b["paras"] = [{**p, "lines": [orig(l) for l in p["lines"]], "anchor": orig(p["lines"][0])["bbox"][0] if p["lines"] else p.get("anchor")}
                  for p in rb.get("paras", [])]
    rcx, rcy = (rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2
    tab["rows"] = [[round(float(_tf(Mi, [rcx, a])[0, 1]), 1), round(float(_tf(Mi, [rcx, c])[0, 1]), 1)] for a, c in tab["rows"]]
    tab["cols"] = [[round(float(_tf(Mi, [a, rcy])[0, 0]), 1), round(float(_tf(Mi, [c, rcy])[0, 0]), 1)] for a, c in tab["cols"]]
    for c in tab["cells"]:
        a, d, e, f = c["box"]
        c["box"] = [round(v, 1) for v in _bbox_of(_tf(Mi, [[a, d], [e, d], [e, f], [a, f]]))]
        c["_lines"] = [orig(l) for l in c["_lines"]]
    tab["deskew_deg"] = round(angle, 2)
    b["table"] = tab
    return tab


def apply_tables(bgr: np.ndarray, ink: np.ndarray, blocks: list, tsr=None, struct_tables: list | None = None,
                 read_boxes=None, pad_x: float = 0.25, deskew: bool = True) -> dict:
    """Run ``build_table`` on every table block of the page. ``tsr``: the loaded table-structure stage (detect(image,
    box) -> boxes with class names) or None; ``struct_tables``: the structure transformer's tables (cells as groups
    of line dicts, page px boxes). Returns counters for the timings."""
    stats = {"tables": 0, "cells": 0, "lines_split": 0, "source": {}}
    for b in blocks:
        if b["cls"] != "table":
            continue
        model_boxes, model_id = None, None
        skew = table_skew(b) if deskew else 0.0
        st = None
        if struct_tables:
            ids = {id(l) for l in b["lines"]}
            best, best_n = None, 0
            for t in struct_tables:
                n = sum(1 for c in t.get("cells", []) for l in c.get("lines", []) if id(l) in ids)
                if n > best_n:
                    best, best_n = t, n
            if best is not None and best_n >= 2:
                st = {"rows": best["rows"], "cols": best["cols"],
                      "cells": [{**c, "lines": [l for l in c.get("lines", []) if id(l) in ids]} for c in best["cells"]]}
        if deskew and DESKEW_MIN_DEG <= abs(skew) <= DESKEW_MAX_DEG:
            tab = _build_deskewed(b, bgr, ink, skew, tsr, st, read_boxes, pad_x)
            stats["deskewed"] = stats.get("deskewed", 0) + 1
        else:
            if tsr is not None:
                try:
                    model_boxes = tsr.detect(bgr, b["box"])
                    model_id = getattr(getattr(tsr, "spec", None), "id", None) or "tsr_model"
                except Exception:
                    model_boxes = None
            tab = build_table(b, bgr, ink, model_boxes, model_id, st, read_boxes, pad_x)
        stats["tables"] += 1
        stats["cells"] += len(tab["cells"])
        stats["lines_split"] += tab.get("lines_split", 0)
        stats["source"][tab["source"]] = stats["source"].get(tab["source"], 0) + 1
    return stats


def render_table(tab: dict, index_of: dict) -> dict:
    """The response form of ``block["table"]``: cell line dicts -> indices into the rendered ``lines[]``."""
    cells = []
    for c in tab["cells"]:
        cells.append({"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": [round(float(v), 1) for v in c["box"]],
                      "header": bool(c["header"]), "text": c["text"], "lines": [index_of[id(l)] for l in c.get("_lines", []) if id(l) in index_of],
                      "align": c["align"], "bold": bool(c["bold"]), "confidence": round(float(c["confidence"]), 4)})
    return {"rows": tab["rows"], "cols": tab["cols"], "header_rows": tab["header_rows"], "borders": tab["borders"],
            "cells": cells, "source": tab["source"], "confidence": tab["confidence"]}
