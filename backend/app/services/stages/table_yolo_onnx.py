"""Table-structure recognition (TSR) with an ultralytics YOLO (v8/11 detect head) exported to ONNX, kind ``tsr_yolo_onnx``.

The model sees one table crop (the layout ``table`` box padded by 2 % of its size, long side 800, letterboxed) and
predicts PubTables-1M style boxes: ``row`` (full table width), ``column`` (full height), ``column_header`` (one box
over all header rows), ``spanning_cell`` (one per cell with rs > 1 or cs > 1) and ``projected_row_header``
(docs/TABLES_CONTRACT.md §5). Export / decode contract as the layout stage (services/stages/layout_yolo_onnx.py):
input [1, 3, S, S] RGB/255, output [1, 4 + nc, A] (cx, cy, w, h) in input pixels + per-class scores, class-aware NMS
in numpy. Class names come from the ONNX ``names`` metadata, else the manifest ``classes``, else TSR_CLASSES.
``detect`` returns boxes in page pixels, so services/tables.py merges them with the ruling lines directly.
"""

from __future__ import annotations

import numpy as np

from app.services.stages.base import Stage
from app.services.stages.layout_yolo_onnx import letterbox, nms, parse_names
from app.services.stages.ort_utils import session

TSR_CLASSES = ["row", "column", "column_header", "spanning_cell", "projected_row_header"]
CROP_PAD = 0.02  # the table box is padded by this fraction of its size before the crop (as the training crops)


class TableStructureDetector(Stage):
    stage = "table"

    def __init__(self, spec, threads: int = 4, device: str = "auto", conf: float = 0.3, iou: float = 0.5, max_det: int = 300):
        self.spec = spec
        self.sess = session(spec.path, threads, device)
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        shp = inp.shape
        if isinstance(shp[2], int) and isinstance(shp[3], int):
            self.size = (shp[2], shp[3])
        else:
            s = spec.input_size or [800, 800]
            self.size = (int(s[0]), int(s[-1]))
        self.names = parse_names(self.sess.get_modelmeta().custom_metadata_map) or spec.classes or list(TSR_CLASSES)
        p = spec.params or {}
        self.conf, self.iou, self.max_det = float(p.get("conf", conf)), float(p.get("iou", iou)), int(p.get("max_det", max_det))

    def info(self) -> dict:
        return {**super().info(), "input_size": list(self.size), "model_classes": self.names}

    @staticmethod
    def crop_box(box, W: int, H: int) -> tuple[int, int, int, int]:
        """The table box padded by CROP_PAD of its size, clipped to the page -> integer (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = (float(v) for v in box)
        px, py = CROP_PAD * (x1 - x0), CROP_PAD * (y1 - y0)
        cx0, cy0 = max(0, int(np.floor(x0 - px))), max(0, int(np.floor(y0 - py)))
        cx1, cy1 = min(W, int(np.ceil(x1 + px))), min(H, int(np.ceil(y1 + py)))
        return cx0, cy0, cx1, cy1

    def detect(self, image: np.ndarray, box, conf: float | None = None) -> list[dict]:
        """BGR page + table box [x0, y0, x1, y1] (page px) -> [{"cls", "score", "box": [x0, y0, x1, y1] page px}]."""
        conf = self.conf if conf is None else conf
        H, W = image.shape[:2]
        cx0, cy0, cx1, cy1 = self.crop_box(box, W, H)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            return []
        crop = image[cy0:cy1, cx0:cx1]
        if crop.ndim == 2:
            crop = np.repeat(crop[:, :, None], 3, axis=2)
        img, r, (px, py) = letterbox(crop, self.size)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0]  # [4+nc, A]
        if out.shape[0] > out.shape[1]:
            out = out.T
        nc = out.shape[0] - 4
        scores_all = out[4:4 + nc]
        cls = scores_all.argmax(0)
        score = scores_all[cls, np.arange(out.shape[1])]
        m = score >= conf
        if not m.any():
            return []
        cx, cy, bw, bh = out[0, m], out[1, m], out[2, m], out[3, m]
        cls, score = cls[m], score[m]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        keep = nms(boxes + cls[:, None].astype(np.float32) * 7680.0, score, self.iou)[: self.max_det]
        boxes = (boxes[keep] - [px, py, px, py]) / r
        ch, cw = crop.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, cw) + cx0
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ch) + cy0
        res = []
        for b, c, s in zip(boxes.tolist(), cls[keep].tolist(), score[keep].tolist()):
            name = self.names[c] if c < len(self.names) else str(c)
            res.append({"cls": name, "score": float(s), "box": [float(v) for v in b]})
        return res
