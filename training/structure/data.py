"""Training data for the structure model -> per-line features and labels.

A page is a set of text lines (box + text + bold + bullet). The model gets them in a canonical geometric order
(top-to-bottom, left-to-right by centre) and predicts for every line: role, whether it starts a new paragraph, its
list level, whether it starts a list item, and its successor in reading order (pointer).

Two kinds of pages:
* GT pages (docgen / struct_v2 manifests with "struct"): the generator's own line boxes. Markers drawn as separate
  lines are merged into their text line (the pipeline attaches bullets too). For v2 features, training simulates
  what the pipeline does on phone photos (``simulate_pipeline``): curl / rotation of the boxes, lines split in two,
  short lines missed, markers left as tiny lines or glued into the text, recognition noise.
* pipe pages (training.structure.pipe_extract): the REAL pipeline's lines on rendered + photographed pages, matched to
  the generator's labels (-100 = unknown line, ignored by the loss).

Feature sets: v1 (23 features, struct_v1) and v2 (FEATURES_V2: v1 + neighbour-relative indent / gaps, local margins,
row mates, marker kind). ``feat_version`` selects; the ONNX metadata records it so the backend computes the same."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np

ROLES = ["text", "title", "section_header", "list_item", "caption", "header", "footer", "page_number", "footnote",
         "table", "form_field", "other"]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
# docgen layout class id -> role (for lines a template drew directly)
CLS_TO_ROLE = {0: "text", 1: "title", 2: "section_header", 3: "list_item", 4: "table", 6: "caption", 7: "header", 8: "footer",
               9: "page_number", 10: "footnote", 20: "form_field", 18: "text"}
KINDS = ["document", "sheet", "mixed"]          # v3: page-level document kind (docs/TABLES_CONTRACT.md §2.1)
KIND_ID = {k: i for i, k in enumerate(KINDS)}
TABLE_ROLE = ROLE_ID["table"]
MAX_LEVEL = 3
N_CHAR_IDS = 6
CHAR_BUCKETS = 2048
MAX_LINES = 256
BULLET_RE = re.compile(r"^\s*([•\-–—○o◦▪■➢►✓*·]|\d{1,3}[.)]|[០-៩]{1,3}[.)]|[a-zA-Z][.)]|[ក-អ][.)]|[ivx]{1,4}[.)])\s")
NUM_BULLET_RE = re.compile(r"^\(?(?:\d{1,3}|[០-៩]{1,3}|[a-zA-Z]|[ក-អ]|[ivx]{1,4})[.)]$")
TERMINAL_RE = re.compile(r"[។.:៖!?]\s*$")
KHMER_RE = re.compile(r"[ក-៿]")
DIGIT_RE = re.compile(r"[0-9០-៩]")
LATIN_RE = re.compile(r"[A-Za-z]")
UPPER_RE = re.compile(r"[A-Z]")
DOTS_RE = re.compile(r"[.…_]{4,}")

FEATURES = ["x0", "y0", "x1", "y1", "w", "h_rel", "cx", "cy", "left_off", "right_gap", "khmer", "digit", "latin", "upper",
            "len", "starts_marker", "ends_terminal", "bold", "bullet", "size_rel", "gap_above", "gap_below", "n_lines"]
N_FEAT = len(FEATURES)
FEATURES_V2 = FEATURES + ["dx_above", "dx_below", "gap_above_col", "gap_below_col", "above_right_short", "w_above_rel",
                          "local_left", "local_right", "row_left", "row_right", "bullet_num", "bullet_glyph", "dots", "x0_rank",
                          "dx_above1", "dy_above1"]
N_FEAT_V2 = len(FEATURES_V2)


def n_feat(version: int) -> int:
    return N_FEAT_V2 if version >= 2 else N_FEAT


def _hash_char(ch: str) -> int:
    return (ord(ch) * 2654435761) % (CHAR_BUCKETS - 1) + 1  # 0 = padding


def char_ids(text: str) -> list[int]:
    t = text.strip()
    head, tail = t[:3], t[-3:] if len(t) >= 3 else t
    ids = [_hash_char(c) for c in head] + [0] * (3 - len(head)) + [_hash_char(c) for c in tail] + [0] * (3 - len(tail))
    return ids[:N_CHAR_IDS]


def line_features(boxes: np.ndarray, texts: list[str], bold: np.ndarray, bullets: np.ndarray, W: float, H: float) -> np.ndarray:
    """v1: boxes [N,4] page px (x0,y0,x1,y1) -> [N, N_FEAT] float32."""
    n = len(boxes)
    h = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    med_h = float(np.median(h)) if n else 1.0
    left = float(np.percentile(boxes[:, 0], 5)) if n else 0.0
    right = float(np.percentile(boxes[:, 2], 95)) if n else W
    order = np.argsort(boxes[:, 1] + boxes[:, 3])
    gap_above = np.zeros(n, np.float32); gap_below = np.zeros(n, np.float32)
    for k, i in enumerate(order):  # vertical gap to the nearest line above / below (any x)
        if k > 0:
            j = order[k - 1]; gap_above[i] = (boxes[i, 1] - boxes[j, 3]) / med_h
        if k < n - 1:
            j = order[k + 1]; gap_below[i] = (boxes[j, 1] - boxes[i, 3]) / med_h
    f = np.zeros((n, N_FEAT), np.float32)
    for i in range(n):
        x0, y0, x1, y1 = boxes[i]
        t = texts[i]
        L = max(1, len(t.strip()))
        f[i] = [x0 / W, y0 / H, x1 / W, y1 / H, (x1 - x0) / W, h[i] / med_h, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                (x0 - left) / W, (right - x1) / W, len(KHMER_RE.findall(t)) / L, len(DIGIT_RE.findall(t)) / L, len(LATIN_RE.findall(t)) / L,
                len(UPPER_RE.findall(t)) / L, math.log1p(L) / 5.0, 1.0 if BULLET_RE.match(t + " ") else 0.0, 1.0 if TERMINAL_RE.search(t) else 0.0,
                float(bold[i]), float(bullets[i]), min(4.0, h[i] / med_h), np.clip(gap_above[i], -2, 6), np.clip(gap_below[i], -2, 6), min(1.0, n / MAX_LINES)]
    return f


def line_features_v2(boxes: np.ndarray, texts: list[str], bold: np.ndarray, bullets: list, W: float, H: float) -> np.ndarray:
    """v2: v1 features + neighbour-relative geometry. ``bullets``: per line the bullet string ("" / None = none, "-",
    "•", "3." ...). Neighbour above/below = nearest line vertically that overlaps horizontally (same column)."""
    n = len(boxes)
    bflag = np.array([1.0 if b else 0.0 for b in bullets], np.float32)
    f1 = line_features(boxes, texts, bold, bflag, W, H)
    if n == 0:
        return np.zeros((0, N_FEAT_V2), np.float32)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    h = np.maximum(1.0, y1 - y0)
    mh = float(np.median(h))
    cy = (y0 + y1) / 2
    right = float(np.percentile(x1, 95))
    # horizontal overlap matrix and vertical relations
    ov = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    vov = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    hmin = np.minimum(h[:, None], h[None, :])
    same_row = vov > 0.5 * hmin
    np.fill_diagonal(same_row, False)
    col = (ov > 0.0) & ~same_row
    np.fill_diagonal(col, False)
    above = col & (cy[None, :] < cy[:, None])   # [i, j]: j above i
    below = col & (cy[None, :] > cy[:, None])
    INF = 1e9
    dy_up = np.where(above, cy[:, None] - cy[None, :], INF)
    dy_dn = np.where(below, cy[None, :] - cy[:, None], INF)
    ja = dy_up.argmin(1); has_a = dy_up.min(1) < INF
    jb = dy_dn.argmin(1); has_b = dy_dn.min(1) < INF
    # any line above (nearest by centre, regardless of x) - for the "row above" relation
    any_up = np.where((cy[None, :] < cy[:, None] - 0.5 * hmin), cy[:, None] - cy[None, :], INF)
    np.fill_diagonal(any_up, INF)
    j1 = any_up.argmin(1); has_1 = any_up.min(1) < INF
    # local margins: lines whose centre is within +-6 line heights
    near = np.abs(cy[:, None] - cy[None, :]) < 6 * mh
    loc_left = np.where(near, x0[None, :], INF).min(1)
    loc_right = np.where(near, x1[None, :], -INF).max(1)
    # x0 rank among distinct left edges of the page (tab stops), coarse
    edges = np.sort(x0)
    stops = [edges[0]]
    for e in edges[1:]:
        if e - stops[-1] > 0.6 * mh:
            stops.append(e)
    stops = np.array(stops)
    f2 = np.zeros((n, N_FEAT_V2 - N_FEAT), np.float32)
    for i in range(n):
        a, b_, o = int(ja[i]), int(jb[i]), int(j1[i])
        bt = bullets[i] or ""
        t = texts[i]
        f2[i] = [np.clip((x0[i] - x0[a]) / mh, -8, 8) if has_a[i] else 0.0,
                 np.clip((x0[b_] - x0[i]) / mh, -8, 8) if has_b[i] else 0.0,
                 np.clip((y0[i] - y1[a]) / mh, -2, 8) if has_a[i] else 8.0,
                 np.clip((y0[b_] - y1[i]) / mh, -2, 8) if has_b[i] else 8.0,
                 np.clip((right - x1[a]) / mh, -2, 20) / 4 if has_a[i] else 0.0,
                 np.clip((x1[a] - x0[a]) / max(1.0, x1[i] - x0[i]), 0, 8) / 4 if has_a[i] else 0.0,
                 np.clip((x0[i] - loc_left[i]) / mh, 0, 30) / 6,
                 np.clip((loc_right[i] - x1[i]) / mh, 0, 60) / 10,
                 1.0 if (same_row[i] & (x1 < x0[i] + 0.5 * mh)).any() else 0.0,
                 1.0 if (same_row[i] & (x0 > x1[i] - 0.5 * mh)).any() else 0.0,
                 1.0 if NUM_BULLET_RE.match(bt) else 0.0,
                 1.0 if bt and not NUM_BULLET_RE.match(bt) else 0.0,
                 1.0 if DOTS_RE.search(t) else 0.0,
                 min(6, int((stops < x0[i] - 0.3 * mh).sum())) / 6,
                 np.clip((x0[i] - x0[o]) / mh, -8, 8) if has_1[i] else 0.0,
                 np.clip((cy[i] - cy[o]) / mh, 0, 8) if has_1[i] else 8.0]
    return np.concatenate([f1, f2], 1)


def default_table_labels(roles) -> dict:
    """v3 table labels when a page has no cell annotations: lines of table role (or unknown role) have unknown cell
    structure (-100, ignored by the loss); every other line is outside any table (-1)."""
    r = np.asarray(roles, np.int64)
    tid = np.where((r == TABLE_ROLE) | (r < 0), -100, -1).astype(np.int64)
    return {"tid": tid, "row": tid.copy(), "col": tid.copy(), "cell": tid.copy()}


def table_labels_from_record(rec: dict, orig_to_kept: dict, roles) -> dict:
    """docgen v5 ``tables`` (contract §5) -> per kept line: table id, row, col, cell id (-1 outside, -100 unknown)."""
    lab = default_table_labels(roles)
    for ti, tb in enumerate(rec.get("tables") or []):
        for ci, cell in enumerate(tb.get("cells") or []):
            for lid in cell.get("line_ids") or []:
                k = orig_to_kept.get(int(lid))
                if k is None:
                    continue
                lab["tid"][k] = ti; lab["row"][k] = int(cell["r"]); lab["col"][k] = int(cell["c"]); lab["cell"][k] = ti * 10000 + ci
    return lab


def kind_label_from_record(rec: dict) -> int:
    k = KIND_ID.get(rec.get("doc_kind") or "", -1)
    if k >= 0:
        return k
    return KIND_ID["mixed"] if any(o[0] == 4 for o in rec.get("objs") or []) else KIND_ID["document"]


class Page:
    """Lines of one page with labels. ``rank`` = reading position (-100 = unknown); GT pages are in reading order
    (rank = index). ``bullets`` = bullet string per line ("" = none). Label arrays may hold -100 (ignored).
    v3: ``table`` = {tid, row, col, cell} int arrays per line (-1 outside tables, -100 unknown) and ``kind_label``
    (index into KINDS, -100 unknown)."""

    def __init__(self, boxes, texts, bold, bullets, roles, first, levels, W, H, rank=None, blabel=None, markers=None, kind="gt",
                 table=None, kind_label=-100):
        self.boxes, self.texts, self.bold, self.bullets = boxes, texts, bold, bullets
        self.roles, self.first, self.levels, self.W, self.H = roles, first, levels, W, H
        self.rank = rank if rank is not None else np.arange(len(texts))
        self.blabel = blabel if blabel is not None else np.array([1 if b else 0 for b in bullets], np.int64)
        self.markers = markers or {}   # GT pages: line index -> (marker text, marker box)
        self.kind = kind
        self.table = table if table is not None else default_table_labels(roles)
        self.kind_label = int(kind_label)

    def __len__(self):
        return len(self.texts)


def _quad_box(q):
    xs, ys = q[0::2], q[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _resolve_role(role: str, box, objs) -> str:
    if role != "other":
        return role
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, area = None, float("inf")
    for cls, x0, y0, x1, y1 in objs:
        if x0 <= cx <= x1 and y0 <= cy <= y1 and (x1 - x0) * (y1 - y0) < area:
            best, area = cls, (x1 - x0) * (y1 - y0)
    return CLS_TO_ROLE.get(best, "text") if best is not None else "text"


def _is_bold_font(fn: str) -> bool:
    return "Bold" in fn or "Black" in fn or "Moul" in fn or "Muol" in fn or "bold" in fn


def page_from_record(rec: dict) -> Page | None:
    st = rec.get("struct") or []
    lines, texts, fonts = rec["lines"], rec["line_texts"], rec.get("line_fonts") or [""] * len(rec["lines"])
    if not st or len(st) != len(lines) or len(lines) < 2:
        return None
    W, H = float(rec["w"]), float(rec["h"])
    # merge markers into the first text line of their paragraph
    marker_of: dict[int, tuple] = {}
    for q, t, s in zip(lines, texts, st):
        if s["marker"]:
            marker_of.setdefault(s["para"], (t, _quad_box(q)))
    boxes, txts, bold, bullets, roles, first, levels, markers = [], [], [], [], [], [], [], {}
    seen_first: set[int] = set()
    orig_to_kept: dict[int, int] = {}
    for li, (q, t, fn, s) in enumerate(zip(lines, texts, fonts, st)):
        if s["marker"]:
            continue
        orig_to_kept[li] = len(txts)
        b = _quad_box(q)
        is_first = s["first"] or (s["para"] not in seen_first)
        seen_first.add(s["para"])
        role = _resolve_role(s["role"], b, rec.get("objs") or [])
        bt = ""
        if is_first and role == "list_item":
            if s["para"] in marker_of:
                bt = marker_of[s["para"]][0].strip()
                markers[len(txts)] = marker_of[s["para"]]
            else:  # glued marker: "-text" (the pipeline splits it off)
                m = re.match(r"^\s*([-–—•·▪]|\(?(?:\d{1,3}|[០-៩]{1,3}|[a-zA-Z]|[ក-អ])[.)])\s*", t)
                if m and len(t) - m.end() >= 2:
                    bt = m.group(1) if not m.group(1) in "-–—" else "-"
                    t = t[m.end():]
                else:
                    bt = "•"
        boxes.append(b); txts.append(t); bold.append(1.0 if _is_bold_font(fn) else 0.0)
        bullets.append(bt)
        roles.append(ROLE_ID[role])
        first.append(1 if is_first else 0); levels.append(min(MAX_LEVEL, int(s["level"])))
    if len(txts) < 2:
        return None
    return Page(np.array(boxes, np.float32), txts, np.array(bold, np.float32), bullets,
                np.array(roles, np.int64), np.array(first, np.int64), np.array(levels, np.int64), W, H, markers=markers,
                table=table_labels_from_record(rec, orig_to_kept, roles), kind_label=kind_label_from_record(rec))


def load_pages(manifest: Path, limit: int | None = None) -> list[Page]:
    out = []
    with open(manifest, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and len(out) >= limit:
                break
            p = page_from_record(json.loads(line))
            if p is not None:
                out.append(p)
    return out


def page_from_pipe(rec: dict) -> Page | None:
    L = rec["lines"]
    if len(L) < 2 or sum(1 for r in rec["rank"] if r >= 0) < 2:
        return None
    boxes = np.array([l["bbox"] for l in L], np.float32)
    table = {k: np.array(rec[k], np.int64) for k in ("tid", "row", "col", "cell")} if "tid" in rec else None
    return Page(boxes, [l.get("text", "") for l in L], np.array([1.0 if l.get("bold") else 0.0 for l in L], np.float32),
                [l.get("bullet") or "" for l in L], np.array(rec["role"], np.int64), np.array(rec["first"], np.int64),
                np.array(rec["level"], np.int64), float(rec["W"]), float(rec["H"]), rank=np.array(rec["rank"], np.int64),
                blabel=np.array(rec["bullet"], np.int64), kind="pipe", table=table, kind_label=int(rec.get("kind", -100)))


def load_pipe_pages(paths: list[Path], limit: int | None = None) -> list[Page]:
    out = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if limit and len(out) >= limit:
                    return out
                try:
                    pg = page_from_pipe(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    continue  # a shard being written
                if pg is not None:
                    out.append(pg)
    return out


# ------------------------------------------------------------------------------------------------ pipeline simulation
NOISE_CHARS = list("ាិីឹឺុូួើឿៀេែៃោៅំះ់៉៊់៍្ ") + list("aeinorstl., ")


def _text_noise(t: str, rng: random.Random, p: float) -> str:
    if not t or rng.random() > p:
        return t
    s = list(t)
    for _ in range(rng.randint(1, max(1, len(s) // 25))):
        k = rng.randrange(len(s))
        r = rng.random()
        if r < 0.4:
            s[k] = rng.choice(NOISE_CHARS)
        elif r < 0.7 and len(s) > 3:
            del s[k]
        else:
            s.insert(k, rng.choice(NOISE_CHARS))
    return "".join(s)


def simulate_pipeline(page: Page, rng: random.Random):
    """GT page -> (boxes, texts, bold, bullets, roles, first, levels, blabel, rank) as the pipeline would hand them
    over on a phone photo: curl + rotation (axis-aligned boxes of warped lines), lines split in two pieces, short lines
    missed, clause numbers left as tiny lines of their own or glued into the text, bullets missed, OCR noise."""
    n = len(page)
    W, H = page.W, page.H
    out = []  # (box, text, bold, bullet, role, first, level, blabel, rank_key)
    hmed = float(np.median(page.boxes[:, 3] - page.boxes[:, 1])) if n else 10.0
    for i in range(n):
        b = page.boxes[i].astype(np.float64).copy()
        t, bt = page.texts[i], page.bullets[i]
        role, fi, lv, bl = int(page.roles[i]), int(page.first[i]), int(page.levels[i]), int(page.blabel[i])
        h = max(1.0, b[3] - b[1])
        if (b[2] - b[0]) < 3 * h and rng.random() < 0.06:
            continue  # short line missed
        if bt and i in page.markers and rng.random() < 0.12:  # the marker stayed a line of its own
            mt, mb = page.markers[i]
            out.append((np.array(mb, np.float64), mt.strip(), page.bold[i], "", role, fi, lv, bl, (i, -1)))
            fi, bl, bt = 0, 0, ""
        elif bt and rng.random() < 0.07:  # bullet not found at all
            bt = ""
        elif bt and NUM_BULLET_RE.match(bt) and rng.random() < 0.03:  # number misread as a dot
            bt = "•"
        if rng.random() < 0.04 and (b[2] - b[0]) > 8 * h:  # split in two pieces (a wide word gap)
            cut = b[0] + (b[2] - b[0]) * rng.uniform(0.3, 0.75)
            g = rng.uniform(0.2, 1.2) * h
            k = int(len(t) * (cut - b[0]) / max(1.0, b[2] - b[0]))
            out.append((np.array([b[0], b[1], cut - g / 2, b[3]]), t[:k].rstrip(), page.bold[i], bt, role, fi, lv, bl, (i, 0)))
            out.append((np.array([cut + g / 2, b[1], b[2], b[3]]), t[k:].lstrip(), page.bold[i], "", role, 0, lv, 0, (i, 1)))
            continue
        out.append((b, t, page.bold[i], bt, role, fi, lv, bl, (i, 0)))
    if len(out) < 2:
        return None
    boxes = np.stack([o[0] for o in out]).astype(np.float64)
    # geometry: curl (y offset as a smooth function of x), rotation, perspective-ish x scaling with y
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    if rng.random() < 0.6:
        amp = H * rng.uniform(0.002, 0.012) * rng.choice([-1, 1]); ph = rng.uniform(0, math.pi); fr = rng.uniform(0.6, 1.4)
        for k in range(len(boxes)):  # a curled line: its box grows by the bow over its width
            xs = np.linspace(boxes[k, 0], boxes[k, 2], 5)
            dy = amp * np.sin(math.pi * xs / W * fr + ph)
            boxes[k, 1] += dy.min(); boxes[k, 3] += dy.max()
    ang = math.radians(rng.choice([0.0, rng.uniform(-2.5, 2.5), rng.uniform(-6, 6)]))
    if ang:
        c, s = math.cos(ang), math.sin(ang)
        P = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1) - [W / 2, H / 2]
        R = P @ np.array([[c, s], [-s, c]]) + [W / 2, H / 2]
        boxes = np.concatenate([R.min(1), R.max(1)], 1)
    if rng.random() < 0.5:  # keystone: the far edge of the page is narrower
        k_ = rng.uniform(-0.08, 0.08)
        f = 1 + k_ * ((boxes[:, 1] + boxes[:, 3]) / 2 / H - 0.5)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - W / 2) * f[:, None] + W / 2
    jit = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.08, boxes.shape) * hmed
    boxes = boxes + jit
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 2); boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 2)
    texts = [_text_noise(o[1], rng, 0.3) for o in out]
    order = sorted(range(len(out)), key=lambda k: out[k][8])
    rank = np.zeros(len(out), np.int64)
    for r, k in enumerate(order):
        rank[k] = r
    return (boxes.astype(np.float32), texts, np.array([o[2] for o in out], np.float32), [o[3] for o in out],
            np.array([o[4] for o in out], np.int64), np.array([o[5] for o in out], np.int64), np.array([o[6] for o in out], np.int64),
            np.array([o[7] for o in out], np.int64), rank, np.array([o[8][0] for o in out], np.int64))  # last: source line index


def make_example(page: Page, rng: random.Random, train: bool, feat_version: int = 1, sim_p: float = 0.0) -> dict:
    """-> features [N,F], chars [N,6], labels (role, first, level, bullet, succ) with lines in geometric order.
    Training augmentation: photo-like crop (lines outside a window vanish), box jitter, random line drops; with
    ``sim_p`` GT pages first go through simulate_pipeline."""
    boxes, texts, bold, bullets = page.boxes.copy(), list(page.texts), page.bold.copy(), list(page.bullets)
    roles, first, levels, blabel, rank = page.roles.copy(), page.first.copy(), page.levels.copy(), page.blabel.copy(), page.rank.copy()
    tab = {k: v.copy() for k, v in page.table.items()}
    W, H = page.W, page.H
    if train and page.kind == "gt" and rng.random() < sim_p:
        sim = simulate_pipeline(page, rng)
        if sim is not None:
            boxes, texts, bold, bullets, roles, first, levels, blabel, rank, src = sim
            tab = {k: v[src] for k, v in tab.items()}
    idx = list(range(len(texts)))
    if train:
        if rng.random() < 0.25:  # a cut-off photo: keep a window of the page
            fx0, fx1 = sorted([rng.uniform(0, 0.3), rng.uniform(0.7, 1.0)])
            fy0, fy1 = sorted([rng.uniform(0, 0.35), rng.uniform(0.65, 1.0)])
            x0, x1, y0, y1 = fx0 * W, fx1 * W, fy0 * H, fy1 * H
            idx = [i for i in idx if x0 <= (boxes[i, 0] + boxes[i, 2]) / 2 <= x1 and y0 <= (boxes[i, 1] + boxes[i, 3]) / 2 <= y1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(x0, x1); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(y0, y1)
            boxes[:, [0, 2]] -= x0; boxes[:, [1, 3]] -= y0; W, H = x1 - x0, y1 - y0
        if rng.random() < 0.15:
            idx = [i for i in idx if rng.random() > 0.04]
        if page.kind == "gt" or rng.random() < 0.5:
            h = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
            boxes = boxes + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.06 if page.kind == "pipe" else 0.12, boxes.shape) * h[:, None]
    if len(idx) > MAX_LINES:
        start = rng.randrange(0, len(idx) - MAX_LINES + 1) if train else 0
        idx = sorted(idx, key=lambda i: (boxes[i, 1] + boxes[i, 3]))[start:start + MAX_LINES]
    if len(idx) < 2:
        idx = list(range(min(len(texts), MAX_LINES)))
    sel = np.array(idx)
    order = sorted(range(len(sel)), key=lambda k: ((boxes[sel[k], 1] + boxes[sel[k], 3]) / 2, boxes[sel[k], 0]))
    sel = sel[order]
    n = len(sel)
    # successor: next KNOWN line in reading order among the kept lines; unknown lines (rank < 0) get -100
    known = [k for k in range(n) if rank[sel[k]] >= 0]
    known.sort(key=lambda k: rank[sel[k]])
    succ = np.full(n, -100, np.int64)
    for a_, b_ in zip(known, known[1:]):
        succ[a_] = b_
    if known:
        succ[known[-1]] = n  # end of page
    fl = first[sel].copy()
    tsel = [texts[int(i)] for i in sel]
    if feat_version >= 2:
        feats = line_features_v2(boxes[sel], tsel, bold[sel], [bullets[int(i)] for i in sel], W, H)
    else:
        feats = line_features(boxes[sel], tsel, bold[sel], np.array([1.0 if bullets[int(i)] else 0.0 for i in sel], np.float32), W, H)
    chars = np.array([char_ids(t) for t in tsel], np.int64)
    return {"feats": feats, "chars": chars, "role": roles[sel], "first": fl, "level": levels[sel],
            "bullet": blabel[sel].astype(np.int64), "succ": succ,
            "tid": tab["tid"][sel], "row": tab["row"][sel], "col": tab["col"][sel], "cell": tab["cell"][sel], "kind": page.kind_label}


def features_for_pipeline(lines: list[dict], W: float, H: float, feat_version: int = 1):
    """Backend helper: pipeline line dicts (bbox, text, bold, bullet) -> (feats, chars, geometric order indices)."""
    boxes = np.array([l["bbox"] for l in lines], np.float32)
    order = sorted(range(len(lines)), key=lambda k: ((boxes[k, 1] + boxes[k, 3]) / 2, boxes[k, 0]))
    b = boxes[order]
    texts = [lines[k].get("text", "") for k in order]
    bold = np.array([1.0 if lines[k].get("bold") else 0.0 for k in order], np.float32)
    if feat_version >= 2:
        f = line_features_v2(b, texts, bold, [lines[k].get("bullet") or "" for k in order], W, H)
    else:
        f = line_features(b, texts, bold, np.array([1.0 if lines[k].get("bullet") else 0.0 for k in order], np.float32), W, H)
    return f, np.array([char_ids(t) for t in texts], np.int64), order
