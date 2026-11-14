"""Shared helpers for the external-dataset converters (training/data/ext_*.py).

Canonical page manifest record (one JSON per line in <out>/<split>/manifest.jsonl, image under <out>/<split>/images/):
    {"image": "images/x.jpg", "w", "h", "source", "doc_type", "lang",
     "lines": [[x0,y0,x1,y1,x2,y2,x3,y3], ...], "line_texts": [...],   # only when real transcriptions exist
     "objs": [[v2_cls_id, x0, y0, x1, y1], ...], "lines_ok": bool, "objs_ok": bool,
     optional: "partial_classes": [v2 ids that ARE exhaustively labelled], "eval_classes": [...], "nc": true}
Images: JPEG q88, long side <= 1600 (never upscaled).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from training.taxonomy import CLASSES, COLORS

LONG_SIDE = 1600
JPEG_Q = 88


def decode(buf: bytes) -> np.ndarray:
    """bytes -> BGR uint8 image (grayscale / RGBA / palette handled by OpenCV)."""
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable image")
    return img


def fit(img: np.ndarray, long_side: int = LONG_SIDE) -> tuple[np.ndarray, float]:
    """Downscale so max(h, w) <= long_side. Returns (image, scale applied to coordinates)."""
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def save_jpg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])


def r1(v) -> float:
    return round(float(v), 1)


def clip_box(b, W, H):
    x0, y0, x1, y1 = b
    x0, x1 = sorted((max(0.0, min(W, x0)), max(0.0, min(W, x1))))
    y0, y1 = sorted((max(0.0, min(H, y0)), max(0.0, min(H, y1))))
    return [x0, y0, x1, y1]


def obj(cls_id: int, box, W, H, min_side: float = 2.0):
    """[cls, x0, y0, x1, y1] clipped to the image, or None if degenerate."""
    x0, y0, x1, y1 = clip_box(box, W, H)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return [int(cls_id), r1(x0), r1(y0), r1(x1), r1(y1)]


def quad_from_box(x0, y0, x1, y1):
    return [r1(x0), r1(y0), r1(x1), r1(y0), r1(x1), r1(y1), r1(x0), r1(y1)]


class Manifest:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w")
        self.n = 0

    def add(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.n += 1

    def close(self):
        self.f.close()


def draw_preview(img_path: Path, rec: dict, out: Path, show_lines: bool = True) -> None:
    """Boxes in taxonomy COLORS with class names; line quads in thin grey-red."""
    img = cv2.imread(str(img_path))
    if show_lines:
        for q in rec.get("lines", []):
            pts = np.array(q, np.float32).reshape(4, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, (60, 60, 230), 1)
    # side fields (not taxonomy objects): thin lines, labelled in small grey text
    for key, col in (("input_boxes", (150, 150, 150)), ("sig_fields", (120, 0, 120)), ("unlinked_questions", (0, 140, 255))):
        for x0, y0, x1, y1 in rec.get(key, []):
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 1)
            cv2.putText(img, key[:5], (int(x0) + 1, int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
    for c, x0, y0, x1, y1 in rec.get("objs", []):
        name = CLASSES[c]
        r, g, b = COLORS[name]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (b, g, r), 2)
        cv2.putText(img, name, (int(x0) + 2, max(12, int(y0) + 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (b, g, r), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def stats(manifest: Path) -> dict:
    from collections import Counter
    pages, lines, cls = 0, 0, Counter()
    for ln in open(manifest):
        r = json.loads(ln)
        pages += 1
        lines += len(r.get("lines", []))
        cls.update(CLASSES[o[0]] for o in r.get("objs", []))
    return {"pages": pages, "lines": lines, "objs": dict(cls.most_common())}
