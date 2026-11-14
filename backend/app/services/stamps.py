"""Ink stamps by colour. Official Cambodian stamps are red ink pressed over the text; the layout model misses faint or
partial ones and, when it finds one, the text under it is read through red strokes.

* ``red_ink_mask``   saturated red pixels (HSV).
* ``find_stamps``    closed red blobs the size of a stamp, roughly round -> "stamp" blocks (unless the layout already
                     has a stamp / logo there, which is then marked as red ink).
* ``clean_red``      the page with the red ink inpainted inside the stamp boxes: recognition crops come from it, so
                     the text under the stamp is read as if the stamp were not there.
* ``stamp_layer``    the stamp alone (red ink on white) for the crop shown in the rebuilt document.
"""

from __future__ import annotations

import cv2
import numpy as np

STAMP_CLASSES = ("stamp", "logo", "signature")


def red_ink_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (((h <= 10) | (h >= 165)) & (s >= 70) & (v >= 60)).astype(np.uint8) * 255


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def find_stamps(bgr: np.ndarray, blocks: list, block_factory, red: np.ndarray | None = None) -> list[dict]:
    """-> new "stamp" blocks (source "heuristic", ink "red"). Existing stamp/logo blocks overlapping a red blob get
    ``ink = "red"`` and their box grown to the blob."""
    H, W = bgr.shape[:2]
    if red is None:
        red = red_ink_mask(bgr)
    if red.sum() < 255 * 0.0002 * W * H:
        return []
    k = max(7, int(W / 60)) | 1
    closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_side, out = 0.06 * W, []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_side or h < min_side or w > 0.6 * W or h > 0.6 * W:
            continue
        # outer outline filled (a ring stamp is hollow inside): a round / square stamp fills >= 55 % of its box
        if not 0.6 <= w / max(1, h) <= 1.7 or cv2.contourArea(c) < 0.55 * w * h:
            continue
        m = max(3.0, 0.03 * max(w, h))
        box = [max(0.0, x - m), max(0.0, y - m), min(float(W), x + w + m), min(float(H), y + h + m)]
        hit = next((b for b in blocks if b["cls"] in STAMP_CLASSES and _iou(b["box"], box) > 0.2), None)
        if hit is not None:
            hit["ink"] = "red"
            hit["box"] = [min(hit["box"][0], box[0]), min(hit["box"][1], box[1]), max(hit["box"][2], box[2]), max(hit["box"][3], box[3])]
            if hit["cls"] == "logo":
                hit["cls"], hit["relabel"] = "stamp", "logo"
            continue
        b = block_factory("stamp", box, 0.5, "red_stamp", source="heuristic")
        b["ink"] = "red"
        out.append(b)
    return out


def _grown(box, W, H, f=0.08):
    x0, y0, x1, y1 = box
    gx, gy = f * (x1 - x0), f * (y1 - y0)
    return max(0, int(x0 - gx)), max(0, int(y0 - gy)), min(W, int(np.ceil(x1 + gx))), min(H, int(np.ceil(y1 + gy)))


def clean_red(bgr: np.ndarray, blocks: list, red: np.ndarray | None = None) -> np.ndarray | None:
    """Page copy with the red ink made invisible inside every red-stamp box, or None when there is no red stamp.

    Red ink is brighter in the red channel than black text, but on a photo or scan a saturated stamp is only mid
    grey there (R ~ 120-180), so the recogniser would still see the ring. Inside the stamp box, on the (dilated) red
    mask, the red channel is therefore re-stretched per stamp: an Otsu split of the mask's red values separates
    "stamp only" (high R -> white) from "text under the stamp" (low R, black ink absorbs red -> stays dark); values
    in between fade linearly. Pixels outside the mask keep their grey (inpainting would smear the crossed strokes;
    whitening by the mask alone would cut them)."""
    boxes = [b["box"] for b in blocks if b.get("ink") == "red"]
    if not boxes:
        return None
    H, W = bgr.shape[:2]
    if red is None:
        red = red_ink_mask(bgr)
    out = bgr.copy()
    for box in boxes:
        x0, y0, x1, y1 = _grown(box, W, H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        roi = bgr[y0:y1, x0:x1]
        g = _unstamp(roi, red[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = np.repeat(g[:, :, None], 3, axis=2)
    return out


RED_FLOOR = 40  # red-channel value of solid black ink; the stretch maps [floor, otsu] -> [black, white]


def _unstamp(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grey image of ``roi`` with the red ink of ``mask`` faded out (see clean_red)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    md = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0  # anti-aliased pink edges of the strokes too
    if md.sum() < 20:
        return roi[:, :, 2]
    r = roi[:, :, 2]
    t, _ = cv2.threshold(r[md].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(t), RED_FLOOR + 40.0)
    faded = 255.0 * np.clip((r.astype(np.float32) - RED_FLOOR) / (t - RED_FLOOR), 0.0, 1.0)
    return np.where(md, faded, gray.astype(np.float32)).clip(0, 255).astype(np.uint8)


def stamp_layer(bgr: np.ndarray, box, red: np.ndarray | None = None) -> np.ndarray:
    """The stamp alone: red ink kept, everything else white (the text under it does not belong to the stamp)."""
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(W, int(np.ceil(box[2]))), min(H, int(np.ceil(box[3])))
    if red is None:
        red = red_ink_mask(bgr)
    m = cv2.dilate(red[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) > 0
    out = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    out[m] = bgr[y0:y1, x0:x1][m]
    return out


def figures_full_of_text(blocks: list, W: int, H: int) -> int:
    """A layout "figure" that is mostly detected text lines (a footer / letterhead block the model took for a picture)
    becomes a text block so its lines are read. -> count."""
    n = 0
    for b in blocks:
        if b["cls"] != "figure" or len(b["lines"]) < 2:
            continue
        x0, y0, x1, y1 = b["box"]
        area = max(1.0, (x1 - x0) * (y1 - y0))
        cov = sum((l["bbox"][2] - l["bbox"][0]) * (l["bbox"][3] - l["bbox"][1]) for l in b["lines"]) / area
        wide = (x1 - x0) >= 5 * (y1 - y0)
        if cov >= 0.3 or (wide and len(b["lines"]) >= 2 and cov >= 0.15):
            b["cls"], b["relabel"], n = "text", "figure", n + 1
    return n
