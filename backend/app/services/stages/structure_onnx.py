"""Structure model (training/structure): the page's text lines -> role, paragraph start, list level, bullet and
reading order (successor pointer) per line. Learned replacement for the geometric rules in doc_structure /
page_structure; the rules stay as the fallback and for what the model does not cover (stamps, fonts, colours).

Contract: inputs feats [1,N,F] float32 (training.structure.data.line_features), chars [1,N,6] int64; outputs softmax
role [1,N,R], first [1,N,2], level [1,N,L], bullet [1,N,2] and successor logits succ [1,N,N+1]."""

from __future__ import annotations

import json
import math
import re

import numpy as np

from app.services.stages.base import Stage
from app.services.stages.ort_utils import session

# ---- feature extraction (kept identical to training/structure/data.py; vendored so the backend never imports training/)
CHAR_BUCKETS, N_CHAR_IDS = 2048, 6
MAX_LINES = 256
BULLET_RE = re.compile(r"^\s*([•\-–—○o◦▪■➢►✓*·]|\d{1,3}[.)]|[០-៩]{1,3}[.)]|[a-zA-Z][.)]|[ក-អ][.)]|[ivx]{1,4}[.)])\s")
TERMINAL_RE = re.compile(r"[។.:៖!?]\s*$")
KHMER_RE, DIGIT_RE, LATIN_RE, UPPER_RE = re.compile(r"[ក-៿]"), re.compile(r"[0-9០-៩]"), re.compile(r"[A-Za-z]"), re.compile(r"[A-Z]")
N_FEAT = 23
N_FEAT_V2 = 39  # v2 = v1 + 16 neighbour-relative features (training/structure/data.py FEATURES_V2)
NUM_BULLET_RE = re.compile(r"^\(?(?:\d{1,3}|[០-៩]{1,3}|[a-zA-Z]|[ក-អ]|[ivx]{1,4})[.)]$")
DOTS_RE = re.compile(r"[.…_]{4,}")


def _hash_char(ch: str) -> int:
    return (ord(ch) * 2654435761) % (CHAR_BUCKETS - 1) + 1


def char_ids(text: str) -> list[int]:
    t = text.strip()
    head, tail = t[:3], t[-3:] if len(t) >= 3 else t
    ids = [_hash_char(c) for c in head] + [0] * (3 - len(head)) + [_hash_char(c) for c in tail] + [0] * (3 - len(tail))
    return ids[:N_CHAR_IDS]


def line_features(boxes: np.ndarray, texts: list[str], bold: np.ndarray, bullets: np.ndarray, W: float, H: float) -> np.ndarray:
    n = len(boxes)
    h = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    med_h = float(np.median(h)) if n else 1.0
    left = float(np.percentile(boxes[:, 0], 5)) if n else 0.0
    right = float(np.percentile(boxes[:, 2], 95)) if n else W
    order = np.argsort(boxes[:, 1] + boxes[:, 3])
    gap_above = np.zeros(n, np.float32); gap_below = np.zeros(n, np.float32)
    for k, i in enumerate(order):
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
                float(bold[i]), float(bullets[i]), min(4.0, h[i] / med_h), float(np.clip(gap_above[i], -2, 6)), float(np.clip(gap_below[i], -2, 6)), min(1.0, n / MAX_LINES)]
    return f


def line_features_v2(boxes: np.ndarray, texts: list[str], bold: np.ndarray, bullets: list, W: float, H: float) -> np.ndarray:
    """v2 features (vendored from training/structure/data.py line_features_v2): v1 + neighbour-relative geometry.
    ``bullets``: per line the bullet string ("" = none)."""
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
    ov = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    vov = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    hmin = np.minimum(h[:, None], h[None, :])
    same_row = vov > 0.5 * hmin
    np.fill_diagonal(same_row, False)
    col = (ov > 0.0) & ~same_row
    np.fill_diagonal(col, False)
    above = col & (cy[None, :] < cy[:, None])
    below = col & (cy[None, :] > cy[:, None])
    INF = 1e9
    dy_up = np.where(above, cy[:, None] - cy[None, :], INF)
    dy_dn = np.where(below, cy[None, :] - cy[:, None], INF)
    ja = dy_up.argmin(1); has_a = dy_up.min(1) < INF
    jb = dy_dn.argmin(1); has_b = dy_dn.min(1) < INF
    any_up = np.where((cy[None, :] < cy[:, None] - 0.5 * hmin), cy[:, None] - cy[None, :], INF)
    np.fill_diagonal(any_up, INF)
    j1 = any_up.argmin(1); has_1 = any_up.min(1) < INF
    near = np.abs(cy[:, None] - cy[None, :]) < 6 * mh
    loc_left = np.where(near, x0[None, :], INF).min(1)
    loc_right = np.where(near, x1[None, :], -INF).max(1)
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


def decode_order(succ_logits: np.ndarray, boxes: np.ndarray | None = None) -> list[int]:
    N = succ_logits.shape[0]
    P = succ_logits - succ_logits.max(1, keepdims=True)
    P = np.exp(P); P /= P.sum(1, keepdims=True)
    incoming = P[:, :N].sum(0)
    used = np.zeros(N, bool)
    order: list[int] = []
    while len(order) < N:
        cand = np.where(~used)[0]
        key = incoming[cand] + (boxes[cand, 1] / 1e6 if boxes is not None else cand / 1e6)
        cur = int(cand[np.argmin(key)])
        used[cur] = True; order.append(cur)
        while len(order) < N:
            row = P[cur].copy(); row[:N][used] = -1
            nxt = int(np.argmax(row))
            if nxt == N or row[nxt] <= 0:
                break
            used[nxt] = True; order.append(nxt); cur = nxt
    return order


class StructureClassifier(Stage):
    stage = "structure"


class StructureOnnx(StructureClassifier):
    def __init__(self, spec, threads: int = 4, device: str = "auto"):
        self.spec = spec
        self.sess = session(spec.path, threads, device)
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.roles = json.loads(meta.get("roles", "[]"))
        if not self.roles:
            raise ValueError(f"structure model {spec.path} has no 'roles' metadata")
        self.feat_version = int(meta.get("feat_version", "1"))
        self.has_tables = meta.get("tables", "0") == "1"      # v3: kind / intab / pair outputs
        self.kinds = json.loads(meta.get("kinds", '["document", "sheet", "mixed"]'))
        # sanity: the graph's feature width must match the version we compute
        want = N_FEAT_V2 if self.feat_version >= 2 else N_FEAT
        got = self.sess.get_inputs()[0].shape[-1]
        if isinstance(got, int) and got != want:
            raise ValueError(f"structure model {spec.path}: feats width {got} but feat_version {self.feat_version} needs {want}")

    def info(self) -> dict:
        return {**super().info(), "roles": self.roles, "feat_version": self.feat_version, "tables": self.has_tables}

    def predict(self, lines: list[dict], W: float, H: float, extra: dict | None = None) -> list[dict]:
        """lines: pipeline dicts with bbox / text / bold / bullet -> one dict per line (same order as ``lines``):
        role, role_p, first, first_p, level, bullet, order (0 = first in reading order).
        v3 models also fill ``extra`` (when given): kind {"kind", "confidence", "probs"}, intab [p per line] and
        tables = decoded cell grids (table_decode.decode_tables; line indices refer to ``lines``)."""
        if len(lines) < 2:
            return [{"role": "text", "role_p": 0.0, "first": True, "first_p": 1.0, "level": 0, "bullet": bool(l.get("bullet")), "order": i} for i, l in enumerate(lines)]
        boxes = np.array([l["bbox"] for l in lines], np.float32)
        geo = sorted(range(len(lines)), key=lambda k: ((boxes[k, 1] + boxes[k, 3]) / 2, boxes[k, 0]))
        if len(geo) > MAX_LINES:
            geo = geo[:MAX_LINES]
        b = boxes[geo]
        texts = [lines[k].get("text", "") for k in geo]
        bold = np.array([1.0 if lines[k].get("bold") else 0.0 for k in geo], np.float32)
        if self.feat_version >= 2:
            feats = line_features_v2(b, texts, bold, [lines[k].get("bullet") or "" for k in geo], W, H)
        else:
            feats = line_features(b, texts, bold, np.array([1.0 if lines[k].get("bullet") else 0.0 for k in geo], np.float32), W, H)
        chars = np.array([char_ids(t) for t in texts], np.int64)
        outs = self.sess.run(None, {"feats": feats[None], "chars": chars[None]})
        role, first, level, bullet, succ = outs[:5]
        if extra is not None and self.has_tables and len(outs) >= 8:
            from app.services.table_decode import decode_tables
            kind_p, intab, pair = outs[5][0], outs[6][0], outs[7][0]
            ki = int(kind_p.argmax())
            extra["kind"] = {"kind": self.kinds[ki], "confidence": float(kind_p[ki]), "probs": {k: float(p) for k, p in zip(self.kinds, kind_p)}}
            extra["intab"] = {geo[k]: float(intab[k, 1]) for k in range(len(geo))}
            tabs = decode_tables(intab[:, 1], pair, b)
            for t in tabs:  # back to the caller's line indices
                t["lines"] = [geo[k] for k in t["lines"]]
                for c in t["cells"]:
                    c["lines"] = [geo[k] for k in c["lines"]]
            extra["tables"] = tabs
        order = decode_order(succ[0], b)
        rank = {geo[k]: r for r, k in enumerate(order)}
        out = [None] * len(lines)
        for k, idx in enumerate(geo):
            ri = int(role[0, k].argmax())
            out[idx] = {"role": self.roles[ri], "role_p": float(role[0, k, ri]), "first": bool(first[0, k, 1] >= 0.5), "first_p": float(first[0, k, 1]),
                        "level": int(level[0, k].argmax()), "bullet": bool(bullet[0, k, 1] >= 0.5), "order": rank[idx]}
        nxt = len(geo)
        for i in range(len(lines)):  # lines beyond MAX_LINES: keep after the rest
            if out[i] is None:
                out[i] = {"role": "text", "role_p": 0.0, "first": True, "first_p": 0.5, "level": 0, "bullet": bool(lines[i].get("bullet")), "order": nxt}; nxt += 1
        return out
