"""Layout detection with an ultralytics YOLO (v8/11 detect head) exported to ONNX, NMS in numpy.

Export:  YOLO("layout_best.pt").export(format="onnx", imgsz=1024, opset=17, simplify=False, nms=False)
Contract: input  images [1, 3, S, S] RGB/255, letterboxed (grey 114 padding, centred)
          output        [1, 4 + nc, A]  (cx, cy, w, h) in input pixels + per-class sigmoid scores
Class names come from the ONNX metadata ``names`` (ultralytics writes it), else from the manifest ``classes``, else the
v2 taxonomy; they are mapped onto the v2 taxonomy (app/taxonomy.py resolve_class), so the 12-class v1 model works.
"""

from __future__ import annotations

import ast

import cv2
import numpy as np

from app.services.stages.base import LayoutDetector, Region
from app.services.stages.ort_utils import session
from app.taxonomy import CLASSES, resolve_class


def letterbox(bgr: np.ndarray, size: tuple[int, int]):
    """-> (padded RGB uint8 [S_h, S_w, 3], scale, (pad_x, pad_y)), same arithmetic as ultralytics LetterBox(auto=False)."""
    h, w = bgr.shape[:2]
    th, tw = size
    r = min(th / h, tw / w)
    nw, nh = max(1, min(tw, int(round(w * r)))), max(1, min(th, int(round(h * r))))  # never 0 (5000x2 strips)
    dw, dh = (tw - nw) / 2, (th - nh) / 2
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    bottom, right = th - nh - top, tw - nw - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img[:, :, ::-1], r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy NMS. boxes [N,4] xyxy -> kept indices (by descending score)."""
    x0, y0, x1, y1 = boxes.T
    areas = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        iw = np.maximum(0, np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]))
        ih = np.maximum(0, np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]))
        inter = iw * ih
        ov = inter / np.maximum(1e-9, areas[i] + areas[rest] - inter)
        order = rest[ov <= iou]
    return keep


def parse_names(meta: dict) -> list[str] | None:
    raw = meta.get("names")
    if not raw:
        return None
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(d, dict):
        return [str(d[k]) for k in sorted(d, key=int)]
    if isinstance(d, (list, tuple)):
        return [str(x) for x in d]
    return None


class YoloOnnxLayout(LayoutDetector):
    def __init__(self, spec, threads: int = 4, device: str = "auto", conf: float = 0.3, iou: float = 0.5,
                 max_det: int = 300):
        self.spec = spec
        self.sess = session(spec.path, threads, device)
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        shp = inp.shape
        if isinstance(shp[2], int) and isinstance(shp[3], int):
            self.size = (shp[2], shp[3])
        else:
            s = spec.input_size or [1024, 1024]
            self.size = (int(s[0]), int(s[-1]))
        self.names = parse_names(self.sess.get_modelmeta().custom_metadata_map) or spec.classes or list(CLASSES)
        self.conf, self.iou, self.max_det = conf, iou, max_det

    def info(self) -> dict:
        return {**super().info(), "input_size": list(self.size), "model_classes": self.names}

    def detect(self, image: np.ndarray, conf: float | None = None) -> list[Region]:
        conf = self.conf if conf is None else conf
        H, W = image.shape[:2]
        img, r, (px, py) = letterbox(image, self.size)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0]  # [4+nc, A]
        if out.shape[0] > out.shape[1]:  # some exports are [A, 4+nc]
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
        # class-aware NMS (ultralytics default agnostic=False): offset boxes per class
        keep = nms(boxes + cls[:, None].astype(np.float32) * 7680.0, score, self.iou)[: self.max_det]
        boxes = (boxes[keep] - [px, py, px, py]) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        regions = []
        for b, c, s in zip(boxes.tolist(), cls[keep].tolist(), score[keep].tolist()):
            name = self.names[c] if c < len(self.names) else str(c)
            regions.append(Region(cls=resolve_class(name, b, W, H), score=float(s), box=b, label=name))
        return regions
