"""Text-line detection with a DB (PP-OCR style) segmentation model in ONNX.

Port of Khm-Dataset-Creation det/textdet.py (DBDetector + boxes_from_prob + drop_contained).
Contract: input  x    [N, 3, H, W] BGR/255 normalised with the ImageNet mean/std, H and W multiples of 32
          output prob [N, 1, H, W] text-kernel probability.
Post-processing: threshold -> contours -> min-area rectangles -> box score (mean prob inside) >= box_thresh ->
unclip by the DB distance A*r/L (pyclipper polygon offset, falling back to rectangle maths) -> quads in input pixels.
Tuned defaults (det/web/server.py DET_KW, DET_SCALE): unclip_ratio 1.6, v_extra 1.0, long side 1600..2200.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.services.stages.base import TextLineDetector
from app.services.stages.ort_utils import session

try:
    import pyclipper
except ImportError:  # pragma: no cover - optional
    pyclipper = None

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def order_points(pts):
    pts = np.asarray(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1)[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def prep(bgr: np.ndarray, long_min: int = 1600, long_max: int = 2200) -> np.ndarray:
    """Resize so the long side lies in [long_min, long_max] (upscale at most 3x), both sides multiples of 32."""
    h, w = bgr.shape[:2]
    long = max(h, w)
    s = min(long_min / long, 3.0) if long < long_min else (long_max / long if long > long_max else 1.0)
    nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def to_tensor(bgr_u8: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(((bgr_u8.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None])


def unclip_rect(rect, ratio: float, v_extra: float):
    """Grow a cv2.minAreaRect by the DB unclip distance d = A*ratio/L; returns a new rect."""
    (cx, cy), (rw, rh), ang = rect
    d = (rw * rh) * ratio / (2 * (rw + rh))
    if pyclipper is not None:
        k = 16.0  # pyclipper works on integers: offset at 1/16 px precision
        pco = pyclipper.PyclipperOffset()
        pco.AddPath([tuple(p) for p in np.round(cv2.boxPoints(rect) * k).astype(np.int64).tolist()],
                    pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        grown = pco.Execute(d * k)
        if grown:
            (cx, cy), (rw2, rh2), ang = cv2.minAreaRect(np.array(grown[0], np.float32) / k)
        else:
            rw2, rh2 = rw + 2 * d, rh + 2 * d
    else:
        rw2, rh2 = rw + 2 * d, rh + 2 * d
    if rw2 >= rh2:  # text runs along the long side: optional extra room for stacked Khmer vowels / subscripts
        rh2 *= v_extra
    else:
        rw2 *= v_extra
    return (cx, cy), (rw2, rh2), ang


def drop_contained(items, frac=0.55, tiny=0.08):
    """Remove fragments (stray marks, stacked vowels) mostly inside a much bigger box."""
    areas = [cv2.contourArea(b) for b, _ in items]
    keep = []
    for i in sorted(range(len(items)), key=lambda k: -areas[k]):
        inside = False
        for j in keep:
            inter, _ = cv2.intersectConvexConvex(items[i][0], items[j][0])
            if areas[i] > 0 and inter / areas[i] > frac and areas[i] < tiny * areas[j]:
                inside = True
                break
        if not inside:
            keep.append(i)
    return [items[i] for i in sorted(keep)]


def boxes_from_prob(prob, h0, w0, thresh=0.3, box_thresh=0.5, unclip_ratio=1.6, v_extra=1.0, min_size=5,
                    max_candidates=1500):
    """prob [h,w] -> [(quad float32 [4,2] tl,tr,br,bl in the h0 x w0 image, score)]"""
    mask = (prob > thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sy, sx = h0 / prob.shape[0], w0 / prob.shape[1]
    out = []
    for c in cnts[:max_candidates]:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        if min(rect[1]) < 3:
            continue
        box = cv2.boxPoints(rect)
        x0, y0 = max(int(box[:, 0].min()), 0), max(int(box[:, 1].min()), 0)
        x1, y1 = int(np.ceil(box[:, 0].max())), int(np.ceil(box[:, 1].max()))
        sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(sub, [(box - [x0, y0]).astype(np.int32)], 1)
        region = prob[y0:y1 + 1, x0:x1 + 1]
        if region.size == 0 or sub.sum() == 0:
            continue
        msk = sub[:region.shape[0], :region.shape[1]] > 0
        if not msk.any():
            continue
        score = float(region[msk].mean())
        if score < box_thresh:
            continue
        nb = cv2.boxPoints(unclip_rect(rect, unclip_ratio, v_extra))
        nb[:, 0] = np.clip(nb[:, 0] * sx, 0, w0 - 1)
        nb[:, 1] = np.clip(nb[:, 1] * sy, 0, h0 - 1)
        nb = order_points(nb)
        if min(np.linalg.norm(nb[1] - nb[0]), np.linalg.norm(nb[3] - nb[0])) < min_size:
            continue
        out.append((nb, score))
    return drop_contained(out)



def ink_coverage(img: np.ndarray, quads, max_side: int = 1200) -> float:
    """Share of glyph-sized ink (Otsu connected components, weighted by area) whose centre lies inside a detected
    line. 1.0 when the image holds too little ink to judge (small crops)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if min(g.shape[:2]) < 16:
        return 1.0  # a strip, not a page
    s = min(1.0, max_side / max(g.shape[:2]))
    if s < 1:
        g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, st, cen = cv2.connectedComponentsWithStats(b, 8)
    H, W = g.shape
    st, cen = st[1:], cen[1:]
    ok = (st[:, 4] >= 3) & (st[:, 3] < 0.04 * H) & (st[:, 2] < 0.1 * W)
    if ok.sum() < 20:
        return 1.0
    m = np.zeros((H, W), np.uint8)
    for q in quads:
        cv2.fillPoly(m, [(np.asarray(q) * s).astype(np.int32)], 1)
    c = cen[ok].astype(int)
    w = st[ok, 4].astype(float)
    inside = m[c[:, 1].clip(0, H - 1), c[:, 0].clip(0, W - 1)] > 0
    return float((w * inside).sum() / w.sum())

class DBOnnxDetector(TextLineDetector):
    DEFAULTS = dict(thresh=0.3, box_thresh=0.5, unclip_ratio=1.6, v_extra=1.0, long_min=1600, long_max=2200,
                    retry_cov=0.6)

    def __init__(self, spec, threads: int = 4, device: str = "auto", **defaults):
        self.spec = spec
        self.sess = session(spec.path, threads, device)
        self.inp = self.sess.get_inputs()[0].name
        self.defaults = {**self.DEFAULTS, **(spec.params or {}), **{k: v for k, v in defaults.items() if v is not None}}

    def info(self) -> dict:
        return {**super().info(), "params": self.defaults}

    def prob(self, bgr, long_min, long_max):
        return self.sess.run(None, {self.inp: to_tensor(prep(bgr, long_min, long_max))})[0][0, 0]

    def detect(self, image: np.ndarray, **kw) -> list[tuple[np.ndarray, float]]:
        p = {**self.defaults, **{k: v for k, v in kw.items() if v is not None and k in self.DEFAULTS}}
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        def run(lo, hi):
            prob = self.prob(image, int(lo), int(hi))
            return boxes_from_prob(prob, *image.shape[:2], thresh=p["thresh"], box_thresh=p["box_thresh"],
                                   unclip_ratio=p["unclip_ratio"], v_extra=p["v_extra"])
        out = run(p["long_min"], p["long_max"])
        if not p.get("retry_cov"):
            return out
        # Self-check: on soft low-resolution scans the detector can collapse at its default scale (a 619x876 letter:
        # 4 lines at 1.83x, 27 at 2x). If most glyph-sized ink lies outside the detected lines, retry larger.
        cov = ink_coverage(image, [q for q, _ in out])
        if cov >= p["retry_cov"]:
            return out
        L = max(image.shape[:2])
        base = min(p["long_min"] / L, 3.0) if L < p["long_min"] else (p["long_max"] / L if L > p["long_max"] else 1.0)
        for f in (1.25, 1.6):
            side = L * min(base * f, 4.0)
            o2 = run(side, side)
            c2 = ink_coverage(image, [q for q, _ in o2])
            if c2 > cov + 0.05:
                out, cov = o2, c2
            if cov >= p["retry_cov"]:
                break
        return out
