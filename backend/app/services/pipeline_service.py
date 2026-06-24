# app/services/pipeline_service.py
from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image

from app.services.model_pool import ModelPool

# Your existing constants
LAYOUT_PAD = 8
LABEL_TO_TYPE = {
    "title": "title",
    "plain text": "paragraph",
    "abandon": "header",
    "figure": "image",
    "figure_caption": "caption",
    "table_caption": "caption",
}
TEXT_TYPES = {"title", "paragraph", "header", "caption"}
RECOGNIZE_TYPES = {"title", "paragraph", "caption"}
KEEP_LABELS = set(LABEL_TO_TYPE)
TYPE_TO_LABEL = {
    "title": "title",
    "paragraph": "plain text",
    "header": "abandon",
    "image": "figure",
    "caption": "figure_caption",
}


class PipelineService:
    """Encapsulates the full OCR pipeline: layout → detection → recognition."""
    
    def __init__(self, pool: ModelPool):
        self.pool = pool
    
    # ------------------------------------------------------------------ #
    # Box post-processing (from your existing pipeline.py)
    # ------------------------------------------------------------------ #
    def _merge_and_filter_boxes(self, boxes):
        """Fuse overlapping blocks."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[1])
        merged = []
        while boxes:
            cx1, cy1, cx2, cy2, clabel = boxes.pop(0)
            for i, (mx1, my1, mx2, my2, mlabel) in enumerate(merged):
                if min(cx2, mx2) > max(cx1, mx1) and min(cy2, my2) > max(cy1, my1):
                    merged[i] = (
                        min(cx1, mx1), min(cy1, my1), max(cx2, mx2), max(cy2, my2),
                        "title" if "title" in (mlabel, clabel) else mlabel,
                    )
                    break
            else:
                merged.append((cx1, cy1, cx2, cy2, clabel))
        return merged

    def _overlap_ratio(self, a0, a1, b0, b1):
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1.0, min(a1 - a0, b1 - b0))
        return overlap / denom

    def _same_line_duplicate(self, a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ah, bh = max(1.0, ay1 - ay0), max(1.0, by1 - by0)
        center_close = abs(((ay0 + ay1) / 2) - ((by0 + by1) / 2)) <= min(ah, bh) * 0.65
        x_overlap = self._overlap_ratio(ax0, ax1, bx0, bx1)
        y_overlap = self._overlap_ratio(ay0, ay1, by0, by1)
        return x_overlap >= 0.55 and center_close

    def _dedupe_lines(self, boxes):
        kept = []
        for box in sorted(boxes, key=lambda b: (b[2] - b[0], b[3] - b[1]), reverse=True):
            if any(self._same_line_duplicate(box, e) for e in kept):
                continue
            kept.append(box)
        return sorted(kept, key=lambda ln: (ln[1], ln[0]))

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def _layout_blocks(self, image_path):
        """Run doclayout, filter, merge, sort."""
        res = self.pool.layout.predict(source=str(image_path), imgsz=1024, conf=0.25)[0]
        raw = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = res.names[int(box.cls[0])]
            if label in KEEP_LABELS:
                raw.append((x1, y1, x2, y2, label))
        blocks = self._merge_and_filter_boxes(raw)
        return sorted(blocks, key=lambda b: (b[1], b[0]))

    def _page_lines(self, image_path):
        """Detect all text lines on the full page."""
        res = self.pool.det(str(image_path), conf=self.pool.det_conf, verbose=False)[0]
        raw = [tuple(map(float, box.xyxy[0].tolist())) for box in res.boxes]
        return sorted(raw, key=lambda ln: (ln[1], ln[0]))

    def _lines_in_block(self, page_lines, bx1, by1, bx2, by2):
        inside = [
            ln for ln in page_lines
            if bx1 <= (ln[0] + ln[2]) / 2 <= bx2 and by1 <= (ln[1] + ln[3]) / 2 <= by2
        ]
        return sorted(self._dedupe_lines(inside), key=lambda ln: ln[1])

    def _recover_orphan_lines(self, blocks, page_lines, page_w, page_h):
        """Rescue lines the layout model missed."""
        if not page_lines:
            return blocks
        padded = [
            (max(0, x1 - LAYOUT_PAD), max(0, y1 - LAYOUT_PAD),
             min(page_w, x2 + LAYOUT_PAD), min(page_h, y2 + LAYOUT_PAD))
            for (x1, y1, x2, y2, _label) in blocks
        ]
        orphans = []
        for ln in page_lines:
            cx, cy = (ln[0] + ln[2]) / 2, (ln[1] + ln[3]) / 2
            if not any(bx1 <= cx <= bx2 and by1 <= cy <= by2
                       for (bx1, by1, bx2, by2) in padded):
                orphans.append(ln)
        if not orphans:
            return blocks

        orphans.sort(key=lambda l: (l[1], l[0]))
        groups = []
        for ln in orphans:
            h = max(1.0, ln[3] - ln[1])
            if groups and (ln[1] - groups[-1][-1][3]) <= h * 0.8:
                groups[-1].append(ln)
            else:
                groups.append([ln])

        recovered = []
        for g in groups:
            gx1 = min(l[0] for l in g)
            gy1 = min(l[1] for l in g)
            gx2 = max(l[2] for l in g)
            gy2 = max(l[3] for l in g)
            is_heading = len(g) == 1 and (gx2 - gx1) < page_w * 0.5
            label = "title" if is_heading else "plain text"
            recovered.append((int(gx1), int(gy1), int(gx2), int(gy2), label))

        return sorted(blocks + recovered, key=lambda b: (b[1], b[0]))

    def _crop_png(self, page_bgr, x1, y1, x2, y2, crop_dir, prefix):
        h, w = page_bgr.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        crop = page_bgr[y1:y2, x1:x2]
        if not crop.size:
            return None
        name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        cv2.imwrite(str(Path(crop_dir) / name), crop)
        return name

    def _wrap_crop(self, name, prefix):
        if not name:
            return None
        return {"url": f"{prefix}{name}", "file": name}

    def _apply_paragraph_tab_rule(self, block):
        lines = block.get("lines") or []
        for line in lines:
            line["indent"] = 0
            line["tab"] = False
        if block.get("type") != "paragraph" or not lines:
            return

        heights = sorted(
            max(0.0, float(line["box"].get("height", 0.0)))
            for line in lines
            if line.get("box")
        )
        if not heights:
            return
        threshold = max(4.0, heights[len(heights) // 2] * 0.5)
        block_left = float(block["box"]["x"])

        for line in lines:
            free_left = float(line["box"]["x"]) - block_left
            if free_left > threshold:
                line["indent"] = int(round(free_left))
                line["tab"] = True

    def _fill_block_lines(self, block, lines, page_bgr, page_w, page_h, recog):
        line_order = 0
        for (lx0, ly0, lx1, ly1) in lines:
            px0, py0 = max(0.0, lx0), max(0.0, ly0)
            px1, py1 = min(float(page_w), lx1), min(float(page_h), ly1)
            if px1 - px0 < 2 or py1 - py0 < 2:
                continue
            text = ""
            if recog is not None:
                crop = page_bgr[int(py0):int(py1), int(px0):int(px1)]
                if crop.size:
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    text = recog.recognize(pil).strip()
            line_order += 1
            block["lines"].append({
                "box": {"x": px0, "y": py0,
                        "width": px1 - px0, "height": py1 - py0},
                "text": text,
                "reading_order": line_order,
            })
        self._apply_paragraph_tab_rule(block)
        block["text"] = "\n".join(
            ("\t" if line.get("tab") else "") + line["text"]
            for line in block["lines"]
            if line["text"]
        )

    def _build_blocks(self, blocks, page_bgr, page_lines, page_w, page_h,
                      crop_dir, crop_url_prefix, recog):
        out_blocks = []
        order = 0
        for (x1, y1, x2, y2, label) in blocks:
            order += 1
            gx1, gy1 = max(0, x1), max(0, y1)
            gx2, gy2 = min(page_w, x2), min(page_h, y2)
            bx1, by1 = max(0, x1 - LAYOUT_PAD), max(0, y1 - LAYOUT_PAD)
            bx2, by2 = min(page_w, x2 + LAYOUT_PAD), min(page_h, y2 + LAYOUT_PAD)
            btype = LABEL_TO_TYPE.get(label, "paragraph")

            block = {
                "id": f"b{order}",
                "type": btype,
                "label": label,
                "reading_order": order,
                "box": {"x": float(gx1), "y": float(gy1),
                        "width": float(gx2 - gx1), "height": float(gy2 - gy1)},
                "lines": [],
                "crop": None,
                "text": "",
            }

            if btype in ("image", "header"):
                block["crop"] = self._wrap_crop(
                    self._crop_png(page_bgr, gx1, gy1, gx2, gy2, crop_dir,
                                  "hdr" if btype == "header" else "fig"),
                    crop_url_prefix,
                )
                out_blocks.append(block)
                continue

            lines = self._lines_in_block(page_lines, bx1, by1, bx2, by2)
            if not lines:
                lines = [(gx1, gy1, gx2, gy2)]

            block_recog = recog if btype in RECOGNIZE_TYPES else None
            self._fill_block_lines(block, lines, page_bgr, page_w, page_h, block_recog)
            out_blocks.append(block)

        return out_blocks

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_image(self, image_path: str, crop_dir: str,
                     crop_url_prefix: str = "", recognize: bool = True) -> dict:
        image_path = Path(image_path)
        page_bgr = cv2.imread(str(image_path))
        if page_bgr is None:
            raise FileNotFoundError(image_path)
        page_h, page_w = page_bgr.shape[:2]

        blocks = self._layout_blocks(image_path)
        page_lines = self._page_lines(image_path) if blocks else []
        blocks = self._recover_orphan_lines(blocks, page_lines, page_w, page_h)

        recog = self.pool.recognizer if recognize else None
        out_blocks = self._build_blocks(blocks, page_bgr, page_lines, page_w, page_h,
                                        crop_dir, crop_url_prefix, recog)
        return {"width": page_w, "height": page_h, "blocks": out_blocks}

    def process_regions(self, image_path: str, regions: list, crop_dir: str,
                        crop_url_prefix: str = "", recognize: bool = True) -> dict:
        image_path = Path(image_path)
        page_bgr = cv2.imread(str(image_path))
        if page_bgr is None:
            raise FileNotFoundError(image_path)
        page_h, page_w = page_bgr.shape[:2]

        boxes = []
        for r in regions:
            box = r.get("box", {})
            x1 = int(round(float(box.get("x", 0))))
            y1 = int(round(float(box.get("y", 0))))
            x2 = x1 + int(round(float(box.get("width", 0))))
            y2 = y1 + int(round(float(box.get("height", 0))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            label = TYPE_TO_LABEL.get(r.get("type", "paragraph"), "plain text")
            boxes.append((x1, y1, x2, y2, label))
        boxes.sort(key=lambda b: (b[1], b[0]))

        page_lines = self._page_lines(image_path) if boxes else []
        recog = self.pool.recognizer if recognize else None
        out_blocks = self._build_blocks(boxes, page_bgr, page_lines, page_w, page_h,
                                        crop_dir, crop_url_prefix, recog)
        return {"width": page_w, "height": page_h, "blocks": out_blocks}

    def process_lines(self, image_path: str, blocks_in: list, crop_dir: str,
                      crop_url_prefix: str = "", recognize: bool = True) -> dict:
        image_path = Path(image_path)
        page_bgr = cv2.imread(str(image_path))
        if page_bgr is None:
            raise FileNotFoundError(image_path)
        page_h, page_w = page_bgr.shape[:2]

        recog = self.pool.recognizer if recognize else None
        ordered = sorted(
            blocks_in,
            key=lambda b: (float(b.get("box", {}).get("y", 0)),
                           float(b.get("box", {}).get("x", 0))),
        )

        out_blocks = []
        order = 0
        for blk in ordered:
            order += 1
            box = blk.get("box", {})
            x1 = int(round(float(box.get("x", 0))))
            y1 = int(round(float(box.get("y", 0))))
            x2 = x1 + int(round(float(box.get("width", 0))))
            y2 = y1 + int(round(float(box.get("height", 0))))
            gx1, gy1 = max(0, x1), max(0, y1)
            gx2, gy2 = min(page_w, x2), min(page_h, y2)
            btype = blk.get("type", "paragraph")
            label = TYPE_TO_LABEL.get(btype, "plain text")

            block = {
                "id": f"b{order}",
                "type": btype,
                "label": label,
                "reading_order": order,
                "box": {"x": float(gx1), "y": float(gy1),
                        "width": float(gx2 - gx1), "height": float(gy2 - gy1)},
                "lines": [],
                "crop": None,
                "text": "",
            }

            if btype == "image":
                block["crop"] = self._wrap_crop(
                    self._crop_png(page_bgr, gx1, gy1, gx2, gy2, crop_dir, "fig"),
                    crop_url_prefix,
                )
                out_blocks.append(block)
                continue

            line_boxes = []
            for ln in blk.get("lines", []):
                lx = float(ln.get("x", 0))
                ly = float(ln.get("y", 0))
                line_boxes.append((lx, ly, lx + float(ln.get("width", 0)),
                                   ly + float(ln.get("height", 0))))
            line_boxes.sort(key=lambda b: (b[1], b[0]))

            block_recog = recog if btype in RECOGNIZE_TYPES else None
            self._fill_block_lines(block, line_boxes, page_bgr, page_w, page_h, block_recog)
            out_blocks.append(block)

        return {"width": page_w, "height": page_h, "blocks": out_blocks}