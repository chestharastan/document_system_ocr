"""The document pipeline:  layout -> text-line detection -> recognition -> (post-correction) -> reading order.

1. layout     regions in the v2 taxonomy; a "figure" covering most of the page is a known layout mistake (dense tables)
              and is ignored; near-identical boxes of different classes are de-duplicated.
2. textdet    one page pass; then (DET_CASCADE) every text region in which the page pass found (almost) nothing is
              cropped and detected again, keeping only new, non-duplicate lines (det/cascade.py).
              Each line goes to the smallest region containing its centre; lines outside every region are grouped
              into "orphan" text blocks.
3. rec        only lines of TEXT_CLASSES blocks (tables included) are read, in one batched call;
              non-text classes (figure, photo, signature, stamp, logo, qr_code, barcode, checkbox, handwriting,
              formula) get an image crop instead.
4. postcorrect  PostCorrector over all lines of the page (identity by default).
5. order      column-aware XY-cut over the blocks, lines in rows top-to-bottom / left-to-right.

Response blocks keep the v1 shape (id, type, label, reading_order, box, lines, crop, text) - ``type`` is the v1 UI type
(title/paragraph/header/caption/image) - and add cls, cls_id, color, score, recognized, source, lines[].confidence/quad.
"""

from __future__ import annotations

import logging
import os
import math
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.services.bold import mark_bold, measure_lines, mark_bold_model
from app.services.document import build_document, paragraph_runs
from app.services.doc_structure import (block_text, find_bullets, ink_mask, is_bullet_glyph, merge_bullet_lines,
                                        merge_row_overlaps, paragraph_text, refine_bullet_boxes, reread_ink_markers,
                                        split_text_marker, strip_leading_bullet, structure_blocks)
from app.services.page_structure import (find_decorations, find_rules, measure_fonts, page_layout, relabel_by_position,
                                         relabel_logos, snap_crop_boxes, snap_line_boxes, split_figures, text_colors, SPLIT_CLASSES)
from app.services.image_io import read_image
from app.services.model_pool import ModelPool
from app.services.reading_order import xy_cut
from app.services.structure_model import apply_structure, order_blocks, regroup
from app.services.stamps import STAMP_CLASSES, clean_red, find_stamps, red_ink_mask, stamp_layer
from app.services.tables import GRID_KEEP, _overlap, apply_tables, decide_kind, find_grid_tables, render_table  # v3 tables
from app.services.stages.base import LineResult
from app.taxonomy import CLS, COLORS, LABELS, TEXT_CLASSES, class_from_request, legacy_type

LAYOUT_PAD = 2.0  # px tolerance when testing whether a line centre lies in a region
log = logging.getLogger("app.pipeline")
ENGINES = ("rules", "model", "llm")  # v3 tables: structure engines (docs/TABLES_CONTRACT.md §1.3, §2.4)


# ------------------------------------------------------------------------------------------- v3 tables: engine
def llm_status() -> dict:
    """The LLM engine's status (services/llm_providers.py, written by the llm-engine agent); a safe default when the
    module is missing or fails, so /pipeline/info always answers."""
    try:
        from app.services.llm_providers import llm_status as _status
        st = dict(_status() or {})
    except Exception:
        st = {}
    return {"configured": bool(st.get("configured", False)), "provider": st.get("provider"), "model": st.get("model"),
            "reachable": bool(st.get("reachable", False)), **{k: v for k, v in st.items() if k not in ("configured", "provider", "model", "reachable")}}


def resolve_engine(pool, llm: dict | None = None) -> tuple[str, str]:
    """-> (requested, effective) structure engine. requested = the POST /pipeline/settings value or "auto";
    effective: llm only when the LLM engine is configured, model only when a structure model is loaded, else rules."""
    requested = getattr(pool, "structure_engine", None) or "auto"
    has_model = getattr(pool, "structure", None) is not None
    if requested == "llm":
        llm = llm if llm is not None else llm_status()
        if llm.get("configured"):
            return requested, "llm"
    if requested in ("llm", "model", "auto"):
        return requested, "model" if has_model else "rules"
    return requested, "rules"


def engine_info(pool) -> dict:
    """The engine keys of GET /pipeline/info and POST /pipeline/settings."""
    llm = llm_status()
    requested, effective = resolve_engine(pool, llm)
    return {"structure_engine": effective, "structure_engine_requested": requested, "llm": llm,
            "table_model": getattr(pool, "active", {}).get("table")}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _bbox(q: np.ndarray) -> list[float]:
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def _quad(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _inside(c, box, tol=0.0):
    x0, y0, x1, y1 = box
    return x0 - tol <= c[0] <= x1 + tol and y0 - tol <= c[1] <= y1 + tol


def _bullet_or_text(text: str, ln: dict) -> str:
    """A near-square little box read as a single o/0/- is a list bullet (the recogniser has no class for the ring glyph).
    Dashes keep their glyph so the list style survives; everything else becomes the dot."""
    if is_bullet_glyph(text, ln):
        return "-" if text in ("-", "–", "—") else "•"
    return text


def crop_quad(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-rectify a line quad (tl,tr,br,bl) into an upright crop (det/textdet.py crop_box).
    The quad is clipped to the page first, so the crop can never be larger than the page."""
    H, W = bgr.shape[:2]
    q = np.nan_to_num(np.asarray(quad, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # grow the quad along its own axes: detector boxes sit tight on the first/last glyph and a tight crop loses
    # a leading "(" or "A"; the recogniser was trained with a little margin and ignores blank space (CTC)
    hv = (q[3] - q[0] + q[2] - q[1]) / 2.0
    h = float(np.linalg.norm(hv))
    if h > 0:
        ux = (q[1] - q[0] + q[2] - q[3]) / 2.0
        ux = ux / max(1e-6, float(np.linalg.norm(ux)))
        uy = hv / h
        px, py = settings.rec_pad_x * h, settings.rec_pad_y * h
        q = np.stack([q[0] - ux * px - uy * py, q[1] + ux * px - uy * py, q[2] + ux * px + uy * py, q[3] - ux * px + uy * py]).astype(np.float32)
    q[:, 0] = q[:, 0].clip(0, W - 1)
    q[:, 1] = q[:, 1].clip(0, H - 1)
    w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    w, h = min(max(w, 4), max(W, 4)), min(max(h, 4), max(H, 4))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


class PipelineService:
    """Stateless per request; models live in the shared ModelPool."""

    def __init__(self, pool: ModelPool):
        self.pool = pool

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _read(image_path) -> np.ndarray:
        """Decode with the same rules as the upload (alpha on white, 8-bit, long side capped)."""
        return read_image(image_path)

    def _snapshot(self, layout: bool, textdet: bool, rec: bool):
        """Resolve the stage implementations ONCE at request start: a model switch during the request neither
        changes the models used half-way nor what the response reports."""
        self._impl = {}
        if layout:
            self._impl["layout"] = self.pool.layout
        if textdet:
            self._impl["textdet"] = self.pool.textdet
        if rec:
            self._impl["rec"] = self.pool.recognizer
            font = getattr(self.pool, "font", None)  # optional stage (stub pools in tests have none)
            if font is not None:
                self._impl["font"] = font
            structure = getattr(self.pool, "structure", None)
            if structure is not None:
                self._impl["structure"] = structure
        # v3 tables: the structure engine of this request and the optional table-structure (TSR) stage
        self.structure_engine = resolve_engine(self.pool)[1]
        if self.structure_engine == "rules":
            self._impl.pop("structure", None)  # pure rules: no learned structure, no learned table structure
        elif rec:
            table = getattr(self.pool, "table", None)
            if table is not None:
                self._impl["table"] = table
        active = dict(self.pool.active)
        self._models = {}
        for stage in ("layout", "textdet", "rec", "font", "structure", "table"):
            impl = self._impl.get(stage)
            spec = getattr(impl, "spec", None)
            self._models[stage] = getattr(spec, "id", None) or (active.get(stage) if impl is not None else None)
        self._models["postcorrect"] = self.pool.postcorrector.name

    @staticmethod
    def _clip_box(box, W, H):
        """User box {x, y, width, height} -> [x0, y0, x1, y1] clipped to the page, or None (non-finite / < 2 px)."""
        try:
            x0, y0 = float(box.get("x", 0)), float(box.get("y", 0))
            x1, y1 = x0 + float(box.get("width", 0)), y0 + float(box.get("height", 0))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return None
        x0, x1 = min(max(x0, 0.0), float(W)), min(max(x1, 0.0), float(W))
        y0, y1 = min(max(y0, 0.0), float(H)), min(max(y1, 0.0), float(H))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [x0, y0, x1, y1]

    @staticmethod
    def _block(cls, box, score=1.0, label="", source="layout") -> dict:
        return {"cls": cls, "box": [float(v) for v in box], "score": float(score), "label": label or cls,
                "source": source, "lines": []}

    def _is_page_sized(self, b: dict, W: int, H: int) -> bool:
        x0, y0, x1, y1 = b["box"]
        return b["cls"] == "figure" and (x1 - x0) * (y1 - y0) > settings.page_sized_figure * W * H

    def _clean_regions(self, regions, W, H):
        """-> (blocks used by the pipeline, ignored region dicts)"""
        blocks, ignored = [], []
        for r in sorted(regions, key=lambda r: -r.score):
            b = self._block(r.cls, r.box, r.score, r.label)
            if self._is_page_sized(b, W, H):
                ignored.append({**r.as_dict(), "reason": "page_sized_figure"})
                continue
            if any(_iou(b["box"], k["box"]) > settings.layout_dedupe_iou for k in blocks):
                ignored.append({**r.as_dict(), "reason": "duplicate"})
                continue
            blocks.append(b)
        return blocks, ignored

    # ------------------------------------------------------------------ stage 2
    def _detect_lines(self, bgr, blocks) -> list[tuple[np.ndarray, float]]:
        """Page pass + layout-guided re-detection of text regions the page pass missed (det/cascade.py)."""
        det = self._impl["textdet"]
        H, W = bgr.shape[:2]
        out = list(det.detect(bgr))
        if settings.det_hw_pass:
            out = self._handwriting_pass(bgr, blocks, det, out)
        if not settings.det_cascade:
            return out
        kept = [_bbox(q) for q, _ in out]
        for b in blocks:
            # text regions the page pass missed, and figures / logos: they often hold text (an emblem with the
            # document title under it) that the page pass skipped; split_figures sorts picture from text later
            if b["cls"] not in TEXT_CLASSES and b["cls"] not in SPLIT_CLASSES:
                continue
            x0, y0, x1, y1 = b["box"]
            area = max(1.0, (x1 - x0) * (y1 - y0))
            inside = [k for k in kept if _inside(((k[0] + k[2]) / 2, (k[1] + k[3]) / 2), b["box"])]
            if inside and sum((k[2] - k[0]) * (k[3] - k[1]) for k in inside) / area >= 0.3:
                continue  # the page pass already found the text of this region
            self._redetect_region(bgr, b["box"], det, out, kept)
        for b in blocks:  # the text around a picture (the title under an emblem) is found at the picture's own scale
            if b["cls"] in SPLIT_CLASSES and b.get("ink") != "red":
                self._redetect_region(bgr, b["box"], det, out, kept, grow=0.5)
        clean = getattr(self, "_clean", None)
        if clean is not None:  # text under a red stamp: detect again on the page with the stamp inpainted away
            for b in blocks:
                if b.get("ink") == "red":
                    self._redetect_region(clean, b["box"], det, out, kept, grow=0.25, merge_rows=True)
        return out

    @staticmethod
    def _redetect_region(img, box, det, out, kept, grow=0.04, merge_rows=False):
        """Detect lines inside ``box`` (grown by ``grow``) of ``img`` and add the new ones (centre inside the box, not
        overlapping a kept line by > 30 % of their area) to ``out`` / ``kept``. With ``merge_rows`` a new line on the
        same row as a kept line that it touches replaces it by the union of the two (the page pass saw only the part of
        the line outside the stamp; the clean pass sees the rest)."""
        H, W = img.shape[:2]
        long_min, long_max = settings.det_long_min, settings.det_long_max
        x0, y0, x1, y1 = box
        mx, my = max(6.0, grow * (x1 - x0)), max(6.0, grow * (y1 - y0))
        cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        cx1, cy1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
        if cx1 - cx0 < 24 or cy1 - cy0 < 12:
            return
        lm = int(min(long_min, 2.0 * max(cx1 - cx0, cy1 - cy0)))  # enlarge the crop at most 2x
        for q, sc in det.detect(img[cy0:cy1, cx0:cx1], long_min=lm, long_max=max(lm, long_max)):
            q = q + np.array([cx0, cy0], np.float32)
            if not _inside(q.mean(0), box, max(0.02, grow) * max(x1 - x0, y1 - y0)):
                continue
            qb = _bbox(q)
            qa = max(1e-6, (qb[2] - qb[0]) * (qb[3] - qb[1]))
            if merge_rows:
                same_row = [i for i, k in enumerate(kept)
                            if min(qb[3], k[3]) - max(qb[1], k[1]) >= 0.6 * min(qb[3] - qb[1], k[3] - k[1])
                            and min(qb[2], k[2]) - max(qb[0], k[0]) > -0.5 * (qb[3] - qb[1])]
                if same_row:
                    for i in same_row:
                        k = kept[i]
                        qb = [min(qb[0], k[0]), min(qb[1], k[1]), max(qb[2], k[2]), max(qb[3], k[3])]
                    for i in sorted(same_row, reverse=True):
                        del kept[i]
                        del out[i]
                    out.append((_quad(*qb), sc))
                    kept.append(qb)
                    continue
            if any(min(qb[2], k[2]) - max(qb[0], k[0]) > 0 and min(qb[3], k[3]) - max(qb[1], k[1]) > 0 and
                   (min(qb[2], k[2]) - max(qb[0], k[0])) * (min(qb[3], k[3]) - max(qb[1], k[1])) / qa > 0.3
                   for k in kept):
                continue
            out.append((q, sc))
            kept.append(qb)

    @staticmethod
    def _handwriting_pass(bgr, blocks, det, out):
        """Handwriting on phone photos is written large: the DB detector (trained on pages 800..1900 px long) sees whole
        lines only when the page is ~1600 px long, while the page pass runs at 1600..2200 px for small print and breaks
        the strokes into fragments. So when the layout found handwriting anywhere, the whole page is detected once more
        at DET_HW_LONG px and the two passes are merged: where boxes overlap the larger box wins (a whole line beats its
        fragments). Lines outside every layout block become orphan text blocks later, so handwriting the layout model
        missed is still read."""
        if not any(b["cls"] == "handwriting" for b in blocks):
            return out
        H, W = bgr.shape[:2]
        long = min(settings.det_hw_long, max(H, W))
        out = list(out)
        for q, sc in det.detect(bgr, long_min=long, long_max=long):
            qb = _bbox(q)
            qa = max(1e-6, (qb[2] - qb[0]) * (qb[3] - qb[1]))
            overlapped = []
            blocked = False
            for i, (q2, _) in enumerate(out):
                kb = _bbox(q2)
                ka = max(1e-6, (kb[2] - kb[0]) * (kb[3] - kb[1]))
                iw, ih = min(qb[2], kb[2]) - max(qb[0], kb[0]), min(qb[3], kb[3]) - max(qb[1], kb[1])
                if iw > 0 and ih > 0 and iw * ih / min(qa, ka) > 0.3:
                    if ka >= qa:
                        blocked = True  # an existing box at least as large covers it: keep the page pass
                        break
                    overlapped.append(i)
            if blocked:
                continue
            out = [x for i, x in enumerate(out) if i not in overlapped] + [(q, sc)]
        return out

    @staticmethod
    def _merge_paragraphs(blocks: list) -> list:
        """The layout model often returns one "text" box per printed line. Stack vertically adjacent text blocks into
        paragraphs: same class "text", gap below the upper block < 0.7 line heights, horizontal overlap >= half the
        narrower block, the lower block not indented (an indented first line starts a new paragraph), and the two
        left-aligned or of similar width (a centred title stays on its own)."""
        def lh(b):
            hs = [l["bbox"][3] - l["bbox"][1] for l in b["lines"]] if b["lines"] else []
            return float(np.median(hs)) if hs else max(1.0, (b["box"][3] - b["box"][1]))
        text = [b for b in blocks if b["cls"] == "text"]
        other = [b for b in blocks if b["cls"] != "text"]
        text.sort(key=lambda b: (b["box"][1], b["box"][0]))
        def left(b):  # left edge from the detected lines (accurate) rather than the layout box (often cut)
            return min(l["bbox"][0] for l in b["lines"]) if b["lines"] else b["box"][0]
        def top_left(b):
            return min(b["lines"], key=lambda l: l["bbox"][1])["bbox"][0] if b["lines"] else b["box"][0]
        first_x0 = {id(b): top_left(b) for b in text}  # left edge of each block's FIRST line (survives merging)
        changed = True
        while changed and len(text) > 1:
            changed = False
            for i in range(len(text) - 1):
                a = text[i]
                for j in range(i + 1, len(text)):
                    b = text[j]
                    ax0, ay0, ax1, ay1 = a["box"]; bx0, by0, bx1, by1 = b["box"]
                    ax0, bx0 = left(a), left(b)
                    h = min(lh(a), lh(b))
                    if by0 - ay1 > 0.7 * h or by0 < ay0:
                        continue
                    ov = min(ax1, bx1) - max(ax0, bx0)
                    wa, wb = ax1 - ax0, bx1 - bx0
                    if ov < 0.5 * min(wa, wb):
                        continue
                    if first_x0[id(b)] - ax0 > 0.8 * h:  # indented first line below -> new paragraph
                        continue
                    # the upper block's last line ended early (a reference number, a heading, the end of a
                    # paragraph): what follows is not its continuation
                    a_last = max(a["lines"], key=lambda l: l["bbox"][3]) if a["lines"] else None
                    if a_last is not None and a_last["bbox"][2] < bx1 - 3.0 * h and len(b["lines"]) >= 2:
                        continue
                    # a centred title / short centred line is not part of the paragraph under or above it:
                    # the two must be left-aligned, or of similar width
                    if abs(bx0 - ax0) > 0.8 * h and min(wa, wb) < 0.6 * max(wa, wb):
                        continue
                    a["box"] = [min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)]
                    a["lines"] = a["lines"] + b["lines"]
                    a["score"] = max(a["score"], b["score"])
                    if a["source"] == "orphan" and b["source"] != "orphan":
                        a["source"] = b["source"]
                    del text[j]
                    changed = True
                    break
                if changed:
                    break
        return other + text

    @staticmethod
    def _assign(lines, blocks) -> list:
        """Attach each line to the smallest block containing its centre; returns the orphan lines."""
        orphans = []
        for q, sc in lines:
            c = q.mean(0)
            best, area = None, float("inf")
            for b in blocks:
                if b["cls"] in STAMP_CLASSES:
                    continue  # a stamp / signature lies OVER the text: the lines under it belong to the paragraph
                x0, y0, x1, y1 = b["box"]
                a = (x1 - x0) * (y1 - y0)
                if a < area and _inside(c, b["box"], LAYOUT_PAD):
                    best, area = b, a
            ln = {"quad": q, "score": float(sc), "bbox": _bbox(q)}
            if best is None:
                orphans.append(ln)
            else:
                best["lines"].append(ln)
        return orphans

    def _group_orphans(self, orphans) -> list[dict]:
        """Lines outside every region -> text blocks (a line joins a group whose box, grown by 0.8 line heights
        vertically and 1.5 horizontally, it touches)."""
        groups: list[dict] = []
        for ln in sorted(orphans, key=lambda l: (l["bbox"][1], l["bbox"][0])):
            x0, y0, x1, y1 = ln["bbox"]
            h = max(1.0, y1 - y0)
            for g in groups:
                gx0, gy0, gx1, gy1 = g["box"]
                if x0 <= gx1 + 1.5 * h and x1 >= gx0 - 1.5 * h and y0 <= gy1 + 0.8 * h and y1 >= gy0 - 0.8 * h:
                    # the group's lowest line ended well before where this line ends: that line closed a
                    # paragraph (a title, a reference number, a short last line) - this one starts a new group
                    last = max(g["lines"], key=lambda l: l["bbox"][3])
                    if last["bbox"][2] < x1 - 3.0 * h and y0 >= last["bbox"][3] - 0.5 * h:
                        continue
                    g["lines"].append(ln)
                    g["box"] = [min(gx0, x0), min(gy0, y0), max(gx1, x1), max(gy1, y1)]
                    break
            else:
                g = self._block("text", ln["bbox"], ln["score"], "orphan_lines", source="orphan")
                g["lines"].append(ln)
                groups.append(g)
        return groups

    # ------------------------------------------------------------------ stage 3 + 4
    def _recognize(self, bgr, blocks, recognize: bool, timings: dict):
        todo = [(b, ln) for b in blocks if b["cls"] in TEXT_CLASSES for ln in b["lines"]]
        for b in blocks:
            for ln in b["lines"]:
                ln.setdefault("text", "")
                ln.setdefault("confidence", 0.0)
        t = time.perf_counter()
        if recognize and todo:
            src = self._clean if getattr(self, "_clean", None) is not None else bgr
            crops = [crop_quad(src, ln["quad"]) for _, ln in todo]
            rec = self._impl["rec"]
            if hasattr(rec, "recognize_ex"):  # character positions -> words -> bold words
                results = rec.recognize_ex(crops)
            else:
                results = [(t_, c_, None) for t_, c_ in rec.recognize(crops)]
            for (_, ln), (text, conf, pos) in zip(todo, results):
                raw = text
                ln["text"], ln["confidence"] = _bullet_or_text(text.strip(), ln), float(conf)
                if pos is not None and ln["text"] == raw:  # positions line up with the text only when untouched
                    ln["char_pos"] = list(pos)
        timings["rec_ms"] = _ms(t)
        timings["lines_recognized"] = len(todo) if recognize else 0

        t = time.perf_counter()
        if recognize and todo and settings.detect_bold:
            printed = [(ln, crops[i]) for i, (b, ln) in enumerate(todo) if b["cls"] != "handwriting" and ln.get("char_pos")]
            measure_lines(printed)
            timings["bold_thresholds"] = mark_bold([ln for ln, _ in printed], settings.rec_pad_x)
            timings["bold_words"] = sum(1 for ln, _ in printed for s_ in ln.get("spans", []) if s_["bold"])
        timings["bold_ms"] = _ms(t)

        t = time.perf_counter()
        font = self._impl.get("font")
        if recognize and todo and font is not None:  # printed lines only: handwriting has no typeface
            sel = [(i, b, ln) for i, (b, ln) in enumerate(todo) if b["cls"] != "handwriting" and len(ln.get("text", "")) >= 2]
            for (i, b, ln), res in zip(sel, font.classify([crops[i] for i, _, _ in sel])):
                ln["font"] = res
            for b in blocks:
                self._block_font(b)
            if settings.detect_bold and settings.bold_model:  # bold words: the model's weight head per word
                printed = [(ln, crops[i]) for i, b, ln in sel if ln.get("spans")]
                timings["bold_words_model"] = mark_bold_model(printed, font.classify, settings.rec_pad_x, timings.get("bold_thresholds"))
                timings["bold_words"] = sum(1 for ln, _ in printed for s_ in ln.get("spans", []) if s_["bold"])
        timings["font_ms"] = _ms(t)

        t = time.perf_counter()
        if recognize and todo:
            results = [LineResult(text=ln["text"], confidence=ln["confidence"], quad=ln["quad"], block_cls=b["cls"])
                       for b, ln in todo]
            ctx = {"page_width": bgr.shape[1], "page_height": bgr.shape[0],
                   "blocks": [{"cls": b["cls"], "box": b["box"]} for b in blocks]}
            fixed = self.pool.postcorrector.correct(results, ctx)
            if len(fixed) != len(results):
                raise RuntimeError("PostCorrector must return one result per line")
            for (_, ln), r in zip(todo, fixed):
                ln["text"], ln["confidence"] = r.text, float(r.confidence)
                if r.extra:
                    ln["post"] = r.extra
        timings["postcorrect_ms"] = _ms(t)

    def _read_boxes(self, boxes: list) -> list:
        """Recognise axis-aligned page boxes (x0, y0, x1, y1) -> texts (used for list-number marks)."""
        return [t for t, _ in self._read_boxes_conf(boxes)]

    def _read_boxes_conf(self, boxes: list) -> list:
        """v3 tables: recognise axis-aligned page boxes (or 4-point quads) -> [(text, confidence)] (cell pieces of a table line that has
        no character positions; batched by the recogniser)."""
        src = self._clean if getattr(self, "_clean", None) is not None else self._bgr
        crops = [crop_quad(src, np.float32(b).reshape(4, 2) if np.size(b) == 8 else  # a quad (a deskewed table's piece)
                           np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])) for b in boxes]
        return [(t, float(c)) for t, c in self._impl["rec"].recognize(crops)]

    @staticmethod
    def _block_font(b: dict) -> None:
        """One typeface per block: the family probabilities of its lines averaged with the line width as weight;
        weight / slant by weighted vote."""
        fl = [ln for ln in b["lines"] if ln.get("font")]
        if not fl:
            return
        ws = np.array([max(1.0, ln["bbox"][2] - ln["bbox"][0]) for ln in fl])
        probs = np.average(np.stack([ln["font"]["probs"] for ln in fl]), axis=0, weights=ws)
        fam_i = int(probs.argmax())
        first = fl[0]["font"]
        families = None
        for ln in fl:  # the family list is on the stage; recover the name through the per-line argmax
            if int(np.argmax(ln["font"]["probs"])) == fam_i:
                families = ln["font"]["family"]
                break
        bold = sum(w for ln, w in zip(fl, ws) if ln["font"]["weight"] == "bold") > 0.5 * ws.sum()
        italic = sum(w for ln, w in zip(fl, ws) if ln["font"]["slant"] == "italic") > 0.5 * ws.sum()
        b["font"] = {"family": families or first["family"], "confidence": round(float(probs[fam_i]), 4),
                     "weight": "bold" if bold else "regular", "slant": "italic" if italic else "upright"}

    # ------------------------------------------------------------------ output
    def _crop_png(self, bgr, box, crop_dir, prefix, name_prefix):
        h, w = bgr.shape[:2]
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        name = f"{name_prefix}_{uuid.uuid4().hex[:8]}.png"
        cv2.imwrite(str(Path(crop_dir) / name), bgr[y0:y1, x0:x1])
        return {"url": f"{prefix}{name}", "file": name}

    def _render(self, bgr, blocks, crop_dir, prefix, recognize):
        H, W = bgr.shape[:2]
        out = []
        for order, b in enumerate(order_blocks(blocks, lambda bs: xy_cut(bs, W, H)), start=1):
            cls = b["cls"]
            x0, y0, x1, y1 = max(0.0, b["box"][0]), max(0.0, b["box"][1]), min(W, b["box"][2]), min(H, b["box"][3])
            text_cls = cls in TEXT_CLASSES
            blk = {
                "id": f"b{order}", "type": legacy_type(cls), "label": cls, "cls": cls, "cls_id": CLS.get(cls, -1),
                "display": LABELS.get(cls, cls), "color": _hex(COLORS.get(cls, (120, 120, 120))),
                "model_label": b["label"], "score": round(b["score"], 4), "source": b["source"],
                "reading_order": order, "box": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
                "lines": [], "crop": None, "text": "", "recognized": bool(text_cls and recognize),
                "font_px": b.get("font_px"), "font_src": b.get("font_src"), "align": b.get("align", "left"), "relabel": b.get("relabel"),
                "font": b.get("font"), "text_color": b.get("text_color"),
            }
            if not text_cls:
                src = stamp_layer(bgr, [x0, y0, x1, y1], self._red) if b.get("ink") == "red" else bgr
                box_src = [0, 0, src.shape[1], src.shape[0]] if b.get("ink") == "red" else [x0, y0, x1, y1]
                blk["crop"] = self._crop_png(src, box_src, crop_dir, prefix, cls)
                if b.get("ink"):
                    blk["ink"] = b["ink"]
                out.append(blk)
                continue
            rendered: dict[int, dict] = {}
            index_of: dict[int, int] = {}  # v3 tables: id(line) -> index into blk["lines"]
            for i, ln in enumerate(b["lines"], start=1):
                lx0, ly0, lx1, ly1 = ln["bbox"]
                lx0, ly0, lx1, ly1 = max(0.0, lx0), max(0.0, ly0), min(float(W), lx1), min(float(H), ly1)
                if lx1 - lx0 < 2 or ly1 - ly0 < 2:
                    continue
                line = {"box": {"x": lx0, "y": ly0, "width": lx1 - lx0, "height": ly1 - ly0},
                        "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in ln["quad"]],
                        "text": ln.get("text", ""), "confidence": round(ln.get("confidence", 0.0), 4),
                        "score": round(ln["score"], 4), "reading_order": i,
                        "indent_level": int(ln.get("indent_level", 0)), "indent": int(ln.get("indent", 0)),
                        "tab": bool(ln.get("tab", False)), "bullet": ln.get("bullet"),
                        "bullet_box": [round(float(v), 1) for v in ln["bullet_box"]] if ln.get("bullet_box") else None,
                        "baseline": round(float(ln["baseline"]), 1) if ln.get("baseline") is not None else None,
                        "core_h": round(float(ln["core_h"]), 1) if ln.get("core_h") is not None else None,
                        "font": ({k: v for k, v in ln["font"].items() if k != "probs"} if ln.get("font") else None),
                        "bold": bool(ln.get("bold", False)), "spans": ln.get("spans") or [],
                        "text_color": ln.get("text_color")}
                if "post" in ln:
                    line["post"] = ln["post"]
                blk["lines"].append(line)
                rendered[id(ln)] = line
                index_of[id(ln)] = len(blk["lines"]) - 1
            if b.get("table"):  # v3 tables: the cell grid (contract §2.2), cell lines as indices into blk["lines"]
                blk["table"] = render_table(b["table"], index_of)
            blk["text"] = block_text(blk["lines"])
            blk["bold"] = bool(blk["lines"]) and all(l["bold"] for l in blk["lines"] if l["text"])
            blk["paragraphs"] = []
            for p in b.get("paras", []):
                pl = [rendered[id(ln)] for ln in p["lines"] if id(ln) in rendered]
                if pl and any(l["text"] for l in pl):
                    levels = [l["indent_level"] for l in pl]
                    blk["paragraphs"].append({"indent_level": int(p.get("indent_level", 0)), "bullet": p.get("bullet"),
                                              "text": paragraph_text(pl), "lines": [l["reading_order"] for l in pl],
                                              "runs": paragraph_runs(pl),
                                              "first_line_indent": len(pl) > 1 and levels[0] > min(levels[1:])})
            out.append(blk)
        return out

    def _finish(self, bgr, blocks, crop_dir, prefix, recognize, timings, t_start, extra=None, regroup_blocks=False):
        # regroup_blocks: layout-detected blocks may be rebuilt from the structure model; user-drawn regions never are
        H, W = bgr.shape[:2]
        self._bgr = bgr
        t = time.perf_counter()
        gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ink = getattr(self, "_ink", None)
        if ink is None or ink.shape != gray.shape:
            ink = ink_mask(gray)
        self._ink = None
        if settings.snap_lines:  # text line boxes grown to the glyphs the detector cut
            timings["lines_snapped"] = snap_line_boxes(ink, blocks, TEXT_CLASSES)
        if settings.snap_crops:  # figure / logo / stamp boxes grown to the ink they cut through
            timings["crops_snapped"] = snap_crop_boxes(ink, blocks, TEXT_CLASSES)
            # a weak layout "figure" around a signature grows into the stamp above it only when snapped: absorb again
            blocks = absorb_duplicate_figures(blocks)
            timings["figures_absorbed"] = timings.get("figures_absorbed", 0) + absorb_duplicate_figures.last
        if settings.relabel_by_position:
            relabel_logos(blocks, W, H)
        text_blocks = [b for b in blocks if b["cls"] in TEXT_CLASSES]
        timings["bullets_found"] = find_bullets(bgr, text_blocks, gray=gray, ink=ink) if settings.detect_bullets else 0
        timings["bullets_ms"] = _ms(t)
        if settings.merge_dup_lines:  # one printed line detected twice as overlapping pieces -> one box, read once
            for b in text_blocks:
                b["lines"] = merge_row_overlaps(b["lines"])
        self._recognize(bgr, blocks, recognize, timings)
        t = time.perf_counter()
        if recognize:  # a bullet the detector returned as a tiny line of its own joins the text line of its row
            if settings.reread_markers:  # an ink mark that is really a clause number ("2.") is read, not drawn as a dot
                timings["markers_reread"] = reread_ink_markers(text_blocks, self._read_boxes)
            for b in text_blocks:
                b["lines"] = merge_bullet_lines(b["lines"])
                for ln in b["lines"]:
                    strip_leading_bullet(ln)
                    if settings.split_text_markers and b["cls"] not in ("table", "form_field"):
                        split_text_marker(ln)  # "1. text" / "-text" read as one line: the marker becomes the bullet
            refine_bullet_boxes(ink, text_blocks)
            if settings.detect_decorations:  # ornaments / rules the detector returned as text lines
                blocks = find_decorations(ink, blocks, TEXT_CLASSES, self._block)
                blocks += find_rules(ink, blocks, TEXT_CLASSES, self._block)  # rule lines outside every block
                timings["decorations"] = sum(1 for b in blocks if b["cls"] == "decoration")
        if settings.relabel_by_position:
            relabel_by_position(blocks, W, H)
        structure_blocks(blocks, TEXT_CLASSES)
        model = self._impl.get("structure")
        if recognize and model is not None:  # learned roles / paragraphs / levels / order override the rules where confident
            t_s = time.perf_counter()
            lines = [ln for b in blocks if b["cls"] in TEXT_CLASSES and b["cls"] not in ("table", "form_field", "handwriting") for ln in b["lines"] if ln.get("text")]
            if len(lines) >= 2:
                preds = {id(ln): p for ln, p in zip(lines, model.predict(lines, W, H))}
                timings["structure"] = apply_structure(blocks, preds, TEXT_CLASSES)
                if regroup_blocks and settings.structure_regroup:  # paragraphs rebuilt across layout boxes (a list box that swallowed the
                    # next clause, a box overlapping another one, a justified line cut in two at a wide gap)
                    timings["structure"].update(regroup(blocks, preds, TEXT_CLASSES, self._block, only_tangled=True))
                for b in blocks:  # a heading cut out of a paragraph block gets its own typeface vote
                    if b.get("split_role"):
                        b.pop("font", None)
                        self._block_font(b)
            timings["structure_ms"] = _ms(t_s)
        # v3 tables: table structure (TSR model > structure-transformer v3 cells > rules) on every table block
        t_t = time.perf_counter()
        struct_tables, self._kind_prior = None, None
        if recognize and model is not None and getattr(model, "has_tables", False):
            all_lines = [ln for b in blocks if b["cls"] in TEXT_CLASSES for ln in b["lines"] if ln.get("text")]
            if len(all_lines) >= 2:
                extra: dict = {}
                try:
                    model.predict(all_lines, W, H, extra)
                except Exception as exc:  # an older stage signature / a decoding failure: the rules stand
                    log.warning("structure model table heads failed: %s", exc)
                    extra = {}
                self._kind_prior = extra.get("kind")
                struct_tables = [{**t, "cells": [{**c, "lines": [all_lines[i] for i in c.get("lines", []) if 0 <= i < len(all_lines)]}
                                                 for c in t.get("cells", [])]} for t in extra.get("tables", [])]
        timings["tables"] = apply_tables(bgr, ink, blocks, self._impl.get("table"), struct_tables,
                                         self._read_boxes_conf if recognize else None, settings.rec_pad_x)
        timings["table_ms"] = _ms(t_t)
        timings["font_sizes"] = measure_fonts(ink, blocks, TEXT_CLASSES)
        if settings.detect_colors:
            text_colors(bgr, ink, blocks, TEXT_CLASSES)
        layout = page_layout(blocks, W, H, TEXT_CLASSES)
        out_blocks = self._render(bgr, blocks, crop_dir, prefix, recognize)
        timings["order_ms"] = _ms(t)
        timings["total_ms"] = _ms(t_start)
        res = {"width": W, "height": H, "blocks": out_blocks, "layout": layout,
               "document": build_document(out_blocks, layout, W, H), "timings": timings,
               "models": dict(self._models),
               "device": self.pool.device}
        if extra:
            res.update(extra)
        # v3 tables: the page kind (contract §2.1) and the engine that produced the structure; the LLM engine rewrites
        # the whole result when it is selected and available
        res["doc_kind"] = decide_kind(out_blocks, W, H, getattr(self, "_kind_prior", None))
        res["structure_engine"] = self.structure_engine
        if self.structure_engine == "llm":
            res = self._apply_llm(res)
        return res

    def _apply_llm(self, res: dict) -> dict:
        """v3 tables: services/llm_structure.apply_llm_structure(res) -> rewritten result (kind, elements, tables, order).
        The module belongs to the llm-engine agent; when it is missing or fails the model / rules result stands and
        ``structure_engine`` says so (``llm_error`` carries the reason)."""
        fallback = "model" if self._impl.get("structure") is not None else "rules"
        try:
            from app.services.llm_structure import apply_llm_structure, submit_llm_job, sync_timeout_s
        except Exception as exc:
            res["structure_engine"], res["llm_error"] = fallback, f"LLM engine not available: {type(exc).__name__}"
            return res
        # v3 llm (llm-engine agent): a laptop LLM needs minutes per page, so by default the request returns the phone-
        # tier result at once and the LLM refinement runs as a background job (res["llm_job"], poll GET /llm/jobs/<id>);
        # LLM_SYNC=1 keeps the synchronous call, limited to LLM_SYNC_TIMEOUT_S (default 60 s)
        if os.environ.get("LLM_SYNC", "0") not in ("1", "true", "yes"):
            try:
                job = submit_llm_job(res)
                res["structure_engine"] = fallback
                res["llm_job"] = {"id": job["id"], "status": job["status"], "poll": f"/llm/jobs/{job['id']}"}
            except Exception as exc:
                res["structure_engine"], res["llm_error"] = fallback, f"{type(exc).__name__}: {exc}"[:300]
            return res
        try:
            out = apply_llm_structure(res, timeout_s=sync_timeout_s(), wait_if_busy=False)
        except Exception as exc:
            log.warning("LLM structure engine failed, keeping the %s result: %s", fallback, exc)
            res["structure_engine"], res["llm_error"] = fallback, f"{type(exc).__name__}: {exc}"[:300]
            return res
        if not isinstance(out, dict) or "blocks" not in out:
            res["structure_engine"], res["llm_error"] = fallback, "LLM engine returned no result"
            return res
        if (out.get("timings") or {}).get("llm_error"):  # apply_llm_structure failed softly: the input result stands
            out["structure_engine"], out["llm_error"] = fallback, out["timings"]["llm_error"]
            return out
        out["structure_engine"] = "llm"
        return out

    @staticmethod
    def _fallback_whole_block(b, line_h: float | None = None):
        """A text block in which no line was detected: read the whole block as one line. For layout regions only
        when the block is line-shaped (w >= 2h and h <= 2x the median detected line height); ``line_h=None`` means
        always (user-drawn regions)."""
        x0, y0, x1, y1 = b["box"]
        w, h = x1 - x0, y1 - y0
        if b["cls"] not in TEXT_CLASSES or b["lines"] or w < 4 or h < 4:
            return
        if line_h is None or (w >= 2 * h and (line_h <= 0 or h <= 2 * line_h)):
            b["lines"].append({"quad": _quad(x0, y0, x1, y1), "score": 0.0, "bbox": [x0, y0, x1, y1]})

    @staticmethod
    def _drop_empty_containers(blocks, ignored):
        """A text region without lines of its own that encloses other regions which got the lines (a paragraph box
        around per-line boxes) is redundant: drop it."""
        keep = []
        for b in blocks:
            if b["cls"] in TEXT_CLASSES and not b["lines"]:
                x0, y0, x1, y1 = b["box"]
                inner = [o for o in blocks if o is not b and o["lines"] and
                         _inside(((o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2), b["box"])]
                if inner:
                    ignored.append({"cls": b["cls"], "label": b["label"], "score": round(b["score"], 4),
                                    "box": [round(v, 1) for v in b["box"]], "reason": "container"})
                    continue
            keep.append(b)
        return keep

    def _add_grid_tables(self, blocks, ignored, boxes):
        """integration_qa: each ruled grid the layout model missed becomes a ``table`` block (source "rules"); layout
        regions lying mostly inside it (text / list / figure pieces of the table) are dropped as "in_grid_table";
        stamps, signatures, logos, headers, footers and titles stay."""
        for box in boxes:
            keep = []
            for b in blocks:
                if b["cls"] not in GRID_KEEP and _overlap(b["box"], box) >= 0.6:
                    ignored.append({"cls": b["cls"], "label": b["label"], "score": round(b["score"], 4),
                                    "box": [round(v, 1) for v in b["box"]], "reason": "in_grid_table"})
                    continue
                keep.append(b)
            blocks = keep + [self._block("table", box, 1.0, "table", source="rules")]
        return blocks

    def _find_stamps(self, bgr, blocks, timings):
        """Red ink stamps by colour (new blocks + marking of layout stamps); keeps the page with the red ink inpainted
        (self._clean) for detection / recognition under the stamps and the red mask (self._red) for the stamp crops."""
        self._clean, self._red = None, None
        if not settings.detect_stamps:
            timings["stamps_found"] = 0
            return
        self._red = red_ink_mask(bgr)
        new = find_stamps(bgr, blocks, self._block, self._red)
        blocks += new
        timings["stamps_found"] = len(new)
        self._clean = clean_red(bgr, blocks, self._red)

    # ------------------------------------------------------------------ public API
    def process_image(self, image_path: str, crop_dir: str, crop_url_prefix: str = "", recognize: bool = True) -> dict:
        t_start = time.perf_counter()
        self._snapshot(layout=True, textdet=True, rec=recognize)
        bgr = self._read(image_path)
        H, W = bgr.shape[:2]
        timings: dict = {}
        self._ink = None

        t = time.perf_counter()
        regions = self._impl["layout"].detect(bgr)
        blocks, ignored = self._clean_regions(regions, W, H)
        timings["layout_ms"] = _ms(t)
        if settings.grid_tables:  # integration_qa: ruled tables the layout model missed become table blocks
            t = time.perf_counter()
            gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self._ink = ink_mask(gray)
            blocks = self._add_grid_tables(blocks, ignored, find_grid_tables(self._ink, blocks))
            timings["grid_tables_ms"] = _ms(t)
        t = time.perf_counter()
        self._find_stamps(bgr, blocks, timings)
        timings["stamps_ms"] = _ms(t)

        t = time.perf_counter()
        lines = self._detect_lines(bgr, blocks)
        blocks = drop_background_figures(bgr, blocks, lines, ignored)
        timings["background_figures"] = drop_background_figures.last
        orphans = self._assign(lines, blocks)
        if settings.split_figures:  # a figure holding text lines: picture keeps the box, the lines become text
            gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if getattr(self, "_ink", None) is None or self._ink.shape != gray.shape:
                self._ink = ink_mask(gray)
            nonpaper = gray < 235  # a photo or a coloured seal is not paper even where it is not "ink"
            if bgr.ndim == 3:
                nonpaper |= cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1] > 40
            released = split_figures(self._ink, blocks, nonpaper)
            orphans += released
            kept = [_bbox(q) for q, _ in lines]
            for b in blocks:  # the page pass treated the figure as a picture: find the text around the picture at its scale
                if b.get("split_from"):
                    extra: list = []
                    self._redetect_region(bgr, b["box"], self._impl["textdet"], extra, kept, grow=0.5)
                    for q, sc in extra:
                        if not _inside(q.mean(0), b["box"], LAYOUT_PAD):
                            orphans.append({"quad": q, "score": float(sc), "bbox": _bbox(q)})
                            released.append(None)
            timings["figure_lines_released"] = len(released)
        for b in blocks:
            if b["cls"] not in TEXT_CLASSES:
                b["lines"] = []  # lines inside signatures, stamps, figures ... are not text to read
        blocks = absorb_duplicate_figures(blocks)
        timings["figures_absorbed"] = getattr(absorb_duplicate_figures, "last", 0)
        blocks = self._drop_empty_containers(blocks, ignored)
        line_h = float(np.median([_bbox(q)[3] - _bbox(q)[1] for q, _ in lines])) if lines else 0.0
        for b in blocks:
            self._fallback_whole_block(b, line_h)
        blocks += self._group_orphans(orphans)
        if settings.merge_paragraphs:
            blocks = self._merge_paragraphs(blocks)
        timings["textdet_ms"] = _ms(t)
        timings["lines_detected"] = len(lines)

        return self._finish(bgr, blocks, crop_dir, crop_url_prefix, recognize, timings, t_start,
                            {"ignored_regions": ignored}, regroup_blocks=True)

    def process_regions(self, image_path: str, regions: list, crop_dir: str, crop_url_prefix: str = "",
                        recognize: bool = True) -> dict:
        """User-edited regions: skip layout, detect + recognise lines inside them. Boxes are clipped to the page."""
        t_start = time.perf_counter()
        self._snapshot(layout=False, textdet=True, rec=recognize)
        bgr = self._read(image_path)
        H, W = bgr.shape[:2]
        timings: dict = {"layout_ms": 0.0}
        blocks = []
        for r in regions:
            box = self._clip_box(r.get("box") or {}, W, H)
            if box is not None:
                blocks.append(self._block(class_from_request(r.get("type")), box, source="user"))
        self._find_stamps(bgr, blocks, timings)
        t = time.perf_counter()
        lines = self._detect_lines(bgr, blocks) if blocks else []
        self._assign(lines, blocks)  # lines outside the user's regions are dropped
        for b in blocks:
            if b["cls"] not in TEXT_CLASSES:
                b["lines"] = []
            self._fallback_whole_block(b)
        timings["textdet_ms"] = _ms(t)
        timings["lines_detected"] = len(lines)
        return self._finish(bgr, blocks, crop_dir, crop_url_prefix, recognize, timings, t_start)

    def process_lines(self, image_path: str, blocks_in: list, crop_dir: str, crop_url_prefix: str = "",
                      recognize: bool = True) -> dict:
        """User-edited line boxes: recognise exactly these lines. Block and line boxes are clipped to the page."""
        t_start = time.perf_counter()
        self._snapshot(layout=False, textdet=False, rec=recognize)
        bgr = self._read(image_path)
        H, W = bgr.shape[:2]
        timings: dict = {"layout_ms": 0.0, "textdet_ms": 0.0}
        blocks = []
        for blk in blocks_in:
            cls = class_from_request(blk.get("type"))
            lines = []
            if cls in TEXT_CLASSES:
                for ln in blk.get("lines", []):
                    lb = self._clip_box(ln, W, H)
                    if lb is not None:
                        lines.append({"quad": _quad(*lb), "score": 1.0, "bbox": lb})
            box = self._clip_box(blk.get("box") or {}, W, H)
            if box is None and lines:  # block box unusable: wrap its lines
                box = [min(l["bbox"][0] for l in lines), min(l["bbox"][1] for l in lines),
                       max(l["bbox"][2] for l in lines), max(l["bbox"][3] for l in lines)]
            if box is None:
                continue
            b = self._block(cls, box, source="user")
            b["lines"] = lines
            blocks.append(b)
        self._find_stamps(bgr, blocks, timings)
        return self._finish(bgr, blocks, crop_dir, crop_url_prefix, recognize, timings, t_start)


SPECIFIC_PICTURES = ("stamp", "qr_code", "barcode", "logo", "signature")


def absorb_duplicate_figures(blocks: list, min_cover: float = 0.5) -> list:
    """A generic "figure" box that mostly covers a stamp / QR code / logo / signature block is the same picture found
    twice (the Document view drew the stamp twice and each QR code twice). The specific block absorbs the figure: it
    grows to the union (a stamp + the signature under it stay one picture) and the figure is dropped."""
    out, n = [], 0
    specific = [b for b in blocks if b["cls"] in SPECIFIC_PICTURES]
    for b in blocks:
        if b["cls"] == "figure" and not b.get("lines") and specific:
            fx = b["box"]; fa = max(1e-6, (fx[2] - fx[0]) * (fx[3] - fx[1]))
            best, cover = None, 0.0
            for o in specific:
                ox = o["box"]
                iw = min(fx[2], ox[2]) - max(fx[0], ox[0]); ih = min(fx[3], ox[3]) - max(fx[1], ox[1])
                if iw > 0 and ih > 0:
                    oa = max(1e-6, (ox[2] - ox[0]) * (ox[3] - ox[1]))
                    c = iw * ih / min(fa, oa)
                    if c > cover:
                        best, cover = o, c
            if best is not None and cover >= min_cover:
                ox = best["box"]
                best["box"] = [min(fx[0], ox[0]), min(fx[1], ox[1]), max(fx[2], ox[2]), max(fx[3], ox[3])]
                n += 1
                continue
        out.append(b)
    absorb_duplicate_figures.last = n
    return out


BACKGROUND_CLASSES = ("figure", "photo")


def drop_background_figures(bgr: np.ndarray, blocks: list, lines: list, ignored: list, min_outside: float = 0.7,
                            max_paper: float = 0.4) -> list:
    """A phone photo of a sheet lying on a desk: the layout model calls a strip of the desk (wood grain) along the
    image border a "figure". Such a block touches the image border, lies (>= min_outside of its area) outside the
    box of all detected text lines, holds no text line, and is mostly not paper (paper = bright and unsaturated);
    it goes to ``ignored`` with reason "background". A real picture of the document lies inside the text area or on
    paper, and a scan has a margin between the picture and the border."""
    drop_background_figures.last = 0
    H, W = bgr.shape[:2]
    boxes = [_bbox(q) for q, _ in lines]
    if len(boxes) < 3:
        return blocks
    tx0, ty0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    tx1, ty1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    hsv = None
    keep = []
    for b in blocks:
        x0, y0, x1, y1 = b["box"]
        border = min(x0, y0, W - x1, H - y1) <= 0.015 * max(W, H)
        if b["cls"] not in BACKGROUND_CLASSES or not border or x1 - x0 < 4 or y1 - y0 < 4:
            keep.append(b)
            continue
        area = (x1 - x0) * (y1 - y0)
        ix = max(0.0, min(x1, tx1) - max(x0, tx0)) * max(0.0, min(y1, ty1) - max(y0, ty0))
        has_line = any(_inside(((l[0] + l[2]) / 2, (l[1] + l[3]) / 2), b["box"]) for l in boxes)
        if has_line or 1.0 - ix / area < min_outside:
            keep.append(b)
            continue
        if hsv is None:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if bgr.ndim == 3 else None
        if hsv is not None:
            r = hsv[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))]
            paper = float(np.mean((r[:, :, 2] > 170) & (r[:, :, 1] < 40))) if r.size else 1.0
        else:
            r = bgr[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))]
            paper = float(np.mean(r > 200)) if r.size else 1.0
        if paper > max_paper:
            keep.append(b)
            continue
        ignored.append({"cls": b["cls"], "label": b["label"], "score": round(b["score"], 4),
                        "box": [round(v, 1) for v in b["box"]], "reason": "background"})
        drop_background_figures.last += 1
    return keep


drop_background_figures.last = 0
