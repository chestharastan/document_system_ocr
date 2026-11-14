"""Bold words inside recognised lines.

The recogniser (CTC) tells where along the line crop each character was emitted, so a line splits into words with
pixel extents. A word's weight is measured on the crop's ink: how much of its ink survives a small erosion (thin
regular strokes vanish, bold strokes stay) and how dark its core rows are. Both are compared with the page's typical
word (most words are regular), so the test adapts to the scan resolution and the typeface.

Output per line: ``spans`` [{text, x0, x1, bold}] in page pixels and ``bold`` (every word bold).

``mark_bold_model`` re-decides each word with the font model's weight head (training/font: regular / bold, 98 % on
synthetic crops, trained on words and short phrases): the erosion test above is a stand-in that depends on the scan
resolution; the model reads the stroke shape itself. The heuristic spans stay when the font stage is missing."""

from __future__ import annotations

import cv2
import numpy as np

MIN_ER, ER_GAIN, DARK_GAIN = 0.10, 2.5, 0.05


def word_spans(text: str, pos: list[float], crop_w: int, crop_h: int) -> list[tuple[str, int, int]]:
    """(word, x0, x1) in crop pixels, each word padded by ~0.6 of its mean character advance."""
    out, start = [], 0
    for i, ch in enumerate(text + " "):
        if ch != " ":
            continue
        if i > start and start < len(pos) and i - 1 < len(pos):
            x0, x1 = pos[start] * crop_w, pos[i - 1] * crop_w
            cw = (x1 - x0) / (i - start - 1) if i - start > 1 else 0.5 * crop_h
            out.append((text[start:i], int(max(0, x0 - 0.6 * cw)), int(min(crop_w, x1 + 0.6 * cw + 1))))
        start = i + 1
    return out


def word_measures(gray: np.ndarray, thr: float, k: int) -> tuple[float, float] | None:
    """(erosion survival, core darkness) of the ink in a word crop, or None when there is (almost) no ink."""
    ink = (gray <= thr).astype(np.uint8)  # <=: Otsu on a two-valued image returns the darker value itself
    n = int(ink.sum())
    if n < 10:
        return None
    er = float(cv2.erode(ink, np.ones((k, k), np.uint8)).sum()) / n
    rows = ink.sum(1)
    core = np.where(rows >= 0.35 * rows.max())[0]
    dark = float((255 - gray[core[0]:core[-1] + 1].astype(np.float32)).mean() / 255.0)
    return er, dark


def measure_lines(items: list[tuple[dict, np.ndarray]]) -> None:
    """items: (line dict with text / char_pos / bbox, its recognition crop BGR or gray). Sets ``ln["_words"]`` =
    [(word, x0, x1, er, dark)] in crop pixels."""
    for ln, crop in items:
        g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = g.shape[:2]
        text, pos = ln.get("text", ""), ln.get("char_pos") or []
        if not text.strip() or len(pos) != len(text) or w < 4:
            ln["_words"] = []
            continue
        thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = 3 if h <= 60 else (int(round(h / 20)) | 1)
        words = []
        for word, x0, x1 in word_spans(text, pos, w, h):
            m = word_measures(g[:, x0:x1], thr, k) if x1 - x0 > 3 else None
            words.append((word, x0, x1, *(m or (None, None))))
        ln["_words"], ln["_crop_w"] = words, w


def mark_bold(lines: list[dict], pad_x: float) -> dict:
    """Page-level decision. ``pad_x`` = recognition crop margin in line heights (crop x -> page x). Sets per line
    ``spans`` (page pixels) and ``bold``; returns the thresholds used."""
    er_all = [w[3] for ln in lines for w in ln.get("_words", []) if w[3] is not None and len(w[0]) >= 2]
    dk_all = [w[4] for ln in lines for w in ln.get("_words", []) if w[4] is not None and len(w[0]) >= 2]
    if len(er_all) < 3:
        for ln in lines:
            ln["spans"], ln["bold"] = [], False
            ln.pop("_words", None); ln.pop("_crop_w", None)
        return {}
    med_er, med_dk = float(np.median(er_all)), float(np.median(dk_all))
    er_thr, dk_thr = max(MIN_ER, ER_GAIN * med_er), med_dk + DARK_GAIN
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        h = max(1.0, y1 - y0)
        words = ln.pop("_words", [])
        crop_w = ln.pop("_crop_w", 0)
        if not words or not crop_w:
            ln["spans"], ln["bold"] = [], False
            continue
        scale = (x1 - x0 + 2 * pad_x * h) / max(1, crop_w)
        off = x0 - pad_x * h
        spans = []
        for word, wx0, wx1, er, dk in words:
            # thick strokes AND darker than the page's words; clearly thicker strokes alone are enough (a bold word of
            # thin letters such as "IJKL" is not much darker)
            bold = er is not None and er >= er_thr and (dk >= dk_thr or er >= 1.5 * er_thr)
            spans.append({"text": word, "x0": round(off + wx0 * scale, 1), "x1": round(off + wx1 * scale, 1), "bold": bool(bold),
                          "er": None if er is None else round(er, 3), "dk": None if dk is None else round(dk, 3)})
        ln["spans"] = spans
        measured = [s for s, w in zip(spans, words) if w[3] is not None]
        ln["bold"] = bool(measured) and all(s["bold"] for s in measured)
    return {"er": round(er_thr, 3), "dark": round(dk_thr, 3), "med_er": round(med_er, 3), "med_dk": round(med_dk, 3)}


def mark_bold_model(items: list[tuple[dict, np.ndarray]], classify, pad_x: float, thresholds: dict | None = None,
                    sure_p: float = 0.99, min_p: float = 0.5, short_p: float = 0.7, er_gain: float = 1.25) -> int:
    """items: (line with ``spans`` in page pixels, its recognition crop). ``classify(crops) -> [{weight, weight_confidence}]``
    is the font stage. Each word crop (the span, padded by 0.15 line heights) is classified. A word is bold when the
    model is sure (P(bold) >= sure_p), or when it leans bold (>= min_p; >= short_p for words under 3 characters) AND
    the word's strokes are also thicker and darker than the page's typical word (``thresholds`` from mark_bold:
    erosion survival >= er_gain x median, darkness >= median) - digits and low-resolution words look heavy to the
    model alone. Updates ``span["bold"]`` / ``line["bold"]``; returns the number of words re-decided."""
    th = thresholds or {}
    med_er, med_dk = th.get("med_er"), th.get("med_dk")
    crops, owners = [], []
    for ln, crop in items:
        spans = ln.get("spans") or []
        if not spans:
            continue
        x0, y0, x1, y1 = ln["bbox"]
        h = max(1.0, y1 - y0)
        cw = crop.shape[1]
        scale = cw / max(1.0, x1 - x0 + 2 * pad_x * h)      # page px -> crop px
        off = x0 - pad_x * h
        for si, s in enumerate(spans):
            cx0 = int(max(0, (s["x0"] - off - 0.15 * h) * scale))
            cx1 = int(min(cw, (s["x1"] - off + 0.15 * h) * scale))
            if cx1 - cx0 >= 3:
                crops.append(crop[:, cx0:cx1]); owners.append((ln, si))
    if not crops:
        return 0
    for (ln, si), r in zip(owners, classify(crops)):
        p_bold = r["weight_confidence"] if r["weight"] == "bold" else 1.0 - r["weight_confidence"]
        s = ln["spans"][si]
        leaning = p_bold >= (short_p if len(s["text"].strip()) < 3 else min_p)
        thick = (med_er is None or s.get("er") is None or s["er"] >= er_gain * med_er) and (med_dk is None or s.get("dk") is None or s["dk"] >= med_dk)
        s["bold"] = bool(p_bold >= sure_p or (leaning and thick))
        s["p_bold"] = round(float(p_bold), 3)
    for ln, _ in items:
        if ln.get("spans"):
            ln["bold"] = all(s["bold"] for s in ln["spans"])
    return len(crops)
