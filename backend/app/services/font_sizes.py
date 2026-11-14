"""Page-consistent font sizes and typefaces from the per-line font model outputs (font v2 / v3: ``font.em_px``).

Why: a phone photo is not a scan. Per-line em sizes are noisy (a list number "3." cut as its own line, a dotted
leader line, a crop that took in an emblem) and the photo's scale changes across the page (perspective: the far edge
of a tilted page is smaller in pixels, a curled page bends lines). Taking one median per block and snapping the
blocks to levels "within 10 % of the smallest" then splits the body text of one document into two or three sizes.

What (``page_sizes``, called from ``page_structure.measure_fonts``):
 1. line weight: chars that count (no spaces / dots / dashes); lines of < 3 such chars, number-only lines (a list
    number) and dotted leader lines barely vote;
 2. scale field: a robust (Huber IRLS) fit log(em) = a + b x + c y over the page's body lines (the dominant size
    +-20 %), accepted only when it is sizeable (>= 4 % across the body), agrees in direction with the same fit on the
    ink core heights (an independent geometric cue) and keeps the correction within 0.75-1.33; every line's em is then
    expressed at the scale of the body's centroid (``font.em_norm``) - sizes stay in photo pixels on average;
 3. block size: weighted median of its lines' normalised em;
 4. size levels: 1-D agglomerative clustering in log space (closest pair first, a level spans <= 15 %), level =
    weighted median of its blocks -> blocks of one style get exactly one size; no chaining from the smallest size;
 5. typeface: body-level blocks whose own vote is weak take the page's body family (the probabilities of all
    body lines summed), so one paragraph is not "Khmer OS System" between "Battambang" ones.
"""

from __future__ import annotations

import math
import re

import numpy as np

_CORE = re.compile(r"[\s.\-_…·:៖,;'\"()\[\]/|]+")
_NUMBERISH = re.compile(r"^[\d០-៩a-zA-Zក-អ]{0,2}[.)]?$|^[\d០-៩.,:/()\-]+$")
LEVEL_SPAN = math.log(1.15)
MAX_CORR = (0.75, 1.33)


def line_weight(ln: dict) -> float:
    t = (ln.get("text") or "").strip()
    core = _CORE.sub("", t)
    n = len(core)
    if n < 3 or _NUMBERISH.match(core) or _NUMBERISH.match(t):
        return 0.03
    w = min(n, 30) / 30.0
    if t:
        dots = sum(t.count(c) for c in ".…_") / len(t)
        if dots > 0.3:
            w *= 0.25
    return max(0.03, w)


def _wmedian(v, w) -> float:
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][min(len(v) - 1, int(np.searchsorted(c, 0.5 * c[-1])))])


def _irls(X, y, w, delta=0.05, it=8):
    beta = np.linalg.lstsq(X * w[:, None] ** 0.5, y * w ** 0.5, rcond=None)[0]
    for _ in range(it):
        r = y - X @ beta
        hw = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(1e-9, np.abs(r)))
        ww = w * hw
        beta = np.linalg.lstsq(X * ww[:, None] ** 0.5, y * ww ** 0.5, rcond=None)[0]
    return beta


def scale_field(items: list[dict], W: int, H: int) -> tuple[callable, dict]:
    """items: {"x","y","em","core","w"} -> (factor(x, y) that maps a line's em to the body-centroid scale, info)."""
    ident = (lambda x, y: 1.0), {"applied": False}
    rel = [it for it in items if it["w"] >= 0.3 and it["em"] > 0]
    if len(rel) < 8:
        return ident[0], {**ident[1], "reason": "few lines"}
    m = _wmedian([math.log(it["em"]) for it in rel], [it["w"] for it in rel])
    body = [it for it in rel if abs(math.log(it["em"]) - m) < 0.2]
    if len(body) < 8:
        return ident[0], {**ident[1], "reason": "few body lines"}
    xs = np.array([it["x"] / W - 0.5 for it in body]); ys = np.array([it["y"] / H - 0.5 for it in body])
    xspan, yspan = float(np.ptp(xs)), float(np.ptp(ys))
    if yspan < 0.25 and xspan < 0.25:
        return ident[0], {**ident[1], "reason": "body too compact"}
    w = np.array([it["w"] for it in body])
    X = np.stack([np.ones_like(xs), xs if xspan >= 0.2 else np.zeros_like(xs), ys if yspan >= 0.2 else np.zeros_like(ys)], 1)
    le = np.log([it["em"] for it in body])
    b = _irls(X, le, w)
    change = abs(b[1]) * xspan + abs(b[2]) * yspan
    info = {"applied": False, "grad": [round(float(b[1]), 4), round(float(b[2]), 4)], "change": round(float(change), 4), "n_body": len(body)}
    if change < math.log(1.04):
        return ident[0], {**info, "reason": "flat"}
    cores = [it["core"] for it in body]
    if all(c and c > 0 for c in cores):
        bc = _irls(X, np.log(cores), w, delta=0.08)
        g, gc = np.array(b[1:]), np.array(bc[1:])
        info["grad_core"] = [round(float(v), 4) for v in gc]
        if float(g @ gc) <= 0 or float(np.linalg.norm(gc)) < 0.3 * float(np.linalg.norm(g)):
            return ident[0], {**info, "reason": "ink core disagrees"}
    x0, y0 = float(np.average(xs, weights=w)), float(np.average(ys, weights=w))
    bx, by = float(X[:, 1].any() and b[1]), float(X[:, 2].any() and b[2])

    def factor(x, y):
        f = math.exp(-(bx * (x / W - 0.5 - x0) + by * (y / H - 0.5 - y0)))
        return min(MAX_CORR[1], max(MAX_CORR[0], f))
    info.update(applied=True, reason="ok")
    return factor, info


def levels(vals: list[float], wts: list[float]) -> list[int]:
    """1-D agglomerative clustering of log sizes: merge the closest adjacent clusters while the merged span stays
    within LEVEL_SPAN. -> cluster id per value."""
    if not vals:
        return []
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    cl = [[i] for i in order]
    lv = [math.log(max(1e-3, v)) for v in vals]

    def center(c):
        return _wmedian([lv[i] for i in c], [wts[i] for i in c])
    while len(cl) > 1:
        best, bi = None, -1
        for k in range(len(cl) - 1):
            span = lv[cl[k + 1][-1]] - lv[cl[k][0]]
            if span > LEVEL_SPAN:
                continue
            gap = center(cl[k + 1]) - center(cl[k])
            if best is None or gap < best:
                best, bi = gap, k
        if bi < 0:
            break
        cl[bi] = cl[bi] + cl[bi + 1]
        del cl[bi + 1]
    out = [0] * len(vals)
    for k, c in enumerate(cl):
        for i in c:
            out[i] = k
    return out


def page_sizes(blocks: list, text_classes: set, W: int, H: int) -> dict:
    """Sets b["font_px"] / b["font_src"] for text blocks whose lines carry ``font.em_px``; returns debug info.
    Blocks without model sizes are left to the caller (ink fallback). Also harmonises body typefaces."""
    tb = [b for b in blocks if b["cls"] in text_classes and b["lines"]]
    items = []
    for b in tb:
        for ln in b["lines"]:
            f = ln.get("font") or {}
            if f.get("em_px"):
                x0, y0, x1, y1 = ln["bbox"]
                items.append({"ln": ln, "b": b, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "em": float(f["em_px"]),
                              "core": ln.get("core_h"), "w": line_weight(ln) * (1.0 if b["cls"] != "handwriting" else 0.1)})
    if not items:
        return {"lines": 0}
    factor, info = scale_field(items, W, H)
    for it in items:
        it["en"] = it["em"] * factor(it["x"], it["y"])
        it["ln"]["font"]["em_norm"] = round(it["en"], 2)
    by_block: dict[int, list] = {}
    for it in items:
        by_block.setdefault(id(it["b"]), []).append(it)
    sized, vals, wts = [], [], []
    for b in tb:
        its = by_block.get(id(b))
        if not its:
            continue
        v = _wmedian([it["en"] for it in its], [it["w"] for it in its])
        b["font_px_line"] = round(v, 1)
        b["font_src"] = "model"
        sized.append(b); vals.append(v); wts.append(sum(it["w"] for it in its))
    lab = levels(vals, wts)
    lv_val = {}
    for k in set(lab):
        idx = [i for i, l in enumerate(lab) if l == k]
        lv_val[k] = (_wmedian([vals[i] for i in idx], [wts[i] for i in idx]), sum(wts[i] for i in idx))
    for b, k in zip(sized, lab):
        b["font_px"] = round(lv_val[k][0], 1)
    body_k = max(lv_val, key=lambda k: lv_val[k][1]) if lv_val else None
    info["levels"] = sorted(round(v[0], 1) for v in lv_val.values())
    info["body_px"] = round(lv_val[body_k][0], 1) if body_k is not None else None
    # ---- typeface: the page body family for weakly-voted body-level blocks
    names: dict[int, str] = {}
    acc = None
    pos = {id(b): i for i, b in enumerate(sized)}
    for it, in_body in ((it, lab[pos[id(it["b"])]] == body_k) for it in items if id(it["b"]) in pos):
        f = it["ln"]["font"]
        p = f.get("probs")
        if p is None:
            continue
        p = np.asarray(p, float)
        names.setdefault(int(p.argmax()), f.get("family"))
        if in_body and it["w"] >= 0.3:
            acc = p * it["w"] if acc is None else acc + p * it["w"]
    if acc is not None:
        bf = int(acc.argmax())
        info["body_family"] = names.get(bf)
        if names.get(bf):
            for b, k in zip(sized, lab):
                if k != body_k or not b.get("font"):
                    continue
                its = [it for it in by_block.get(id(b), []) if it["ln"]["font"].get("probs") is not None]
                if not its:
                    continue
                bp = sum(np.asarray(it["ln"]["font"]["probs"], float) * max(0.03, it["w"]) for it in its)
                if int(bp.argmax()) != bf and bp[bf] >= 0.35 * bp.max():
                    b["font"]["family_model"] = b["font"].get("family")
                    b["font"]["family"] = names[bf]
                    b["font"]["confidence"] = round(float(bp[bf] / bp.sum()), 4)
    return info
