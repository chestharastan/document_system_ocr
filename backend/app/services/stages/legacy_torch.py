"""Legacy / PyTorch stage implementations, kept behind the registry. All heavy imports are lazy, so the ONNX-only
install (no torch, ultralytics, doclayout_yolo) never touches them unless such a model id is selected.

kinds:  doclayout_yolo_pt   layout   DocLayout-YOLO (YOLOv10) .pt, DocStructBench/DocLayNet names mapped onto v2
        ultralytics_pt      layout   any ultralytics YOLO .pt (e.g. the v1 12-class layout_best.pt)
        ultralytics_lines   textdet  ultralytics YOLO line/word detector .pt (the old v1 detection/best.pt)
        torch_resnet_bilstm rec      ResNet-BiLSTM-CTC .pth + char.json (the old v1 recogniser)
"""

from __future__ import annotations

import numpy as np

from app.services.stages.base import LayoutDetector, LineRecognizer, Region, TextLineDetector
from app.taxonomy import resolve_class


def _torch_device(device: str) -> str:
    import torch
    return "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"


class UltralyticsPtLayout(LayoutDetector):
    def __init__(self, spec, device: str = "auto", conf: float = 0.3, iou: float = 0.5, max_det: int = 300,
                 imgsz: int | None = None, **_):
        from ultralytics import YOLO
        self.spec = spec
        self.model = YOLO(str(spec.path))
        self.device = _torch_device(device)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.imgsz = imgsz or int((spec.input_size or [1024])[0])

    def _predict(self, image, conf):
        return self.model.predict(image, imgsz=self.imgsz, conf=conf, iou=self.iou, device=self.device,
                                  verbose=False, max_det=self.max_det)[0]

    def detect(self, image: np.ndarray, conf: float | None = None) -> list[Region]:
        H, W = image.shape[:2]
        r = self._predict(image, self.conf if conf is None else conf)
        names = r.names
        out = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.cls.cpu().numpy().astype(int).tolist(),
                           r.boxes.conf.cpu().numpy().tolist()):
            out.append(Region(cls=resolve_class(names[c], b, W, H), score=float(s), box=b, label=str(names[c])))
        return out


class DocLayoutYoloPt(UltralyticsPtLayout):
    def __init__(self, spec, device: str = "auto", conf: float = 0.25, iou: float = 0.5, max_det: int = 300,
                 imgsz: int | None = None, **_):
        from doclayout_yolo import YOLOv10
        self.spec = spec
        self.model = YOLOv10(str(spec.path))
        self.device = _torch_device(device)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.imgsz = imgsz or int((spec.input_size or [1120])[0])


class UltralyticsLineDetector(TextLineDetector):
    def __init__(self, spec, device: str = "auto", conf: float = 0.25, **_):
        from ultralytics import YOLO
        self.spec = spec
        self.model = YOLO(str(spec.path))
        self.device = _torch_device(device)
        self.conf = conf

    def detect(self, image: np.ndarray, **kw) -> list[tuple[np.ndarray, float]]:
        r = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()):
            out.append((np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), float(s)))
        return out


class TorchResNetRecognizer(LineRecognizer):
    def __init__(self, spec, device: str = "auto", **_):
        import torch
        from app.services.stages.legacy_resnet import KhmerRecognizer
        self.spec = spec
        self.torch = torch
        self.impl = KhmerRecognizer(str(spec.path), str(spec.vocab_path) if spec.vocab_path else None)

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        from PIL import Image
        out = []
        for c in crops:
            img = Image.fromarray(c[..., ::-1] if c.ndim == 3 else c)
            out.append(self.impl.recognize_with_conf(img))
        return out
