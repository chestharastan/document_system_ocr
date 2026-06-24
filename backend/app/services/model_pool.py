# app/services/model_pool.py
from __future__ import annotations
import torch
from app.config import settings, BASE_DIR
from app.services.model_registry import DET_MODELS, RECOG_MODELS

class ModelPool:
    def __init__(self):
        self._layout = None
        self._det = None
        self._recog = None
        self._recog_key = None
        self._det_key = None
        self.active_det_key = "line_level"
        self.active_recog_key = settings.recog_model
        self.det_conf = settings.det_conf
    
    @property
    def layout(self):
        if self._layout is None:
            from doclayout_yolo import YOLOv10
            self._layout = YOLOv10(str(settings.layout_weights))
        return self._layout
    
    @property
    def det(self):
        if self._det is None or self._det_key != self.active_det_key:
            from ultralytics import YOLO
            self._det = YOLO(str(DET_MODELS[self.active_det_key].path))
            self._det_key = self.active_det_key
        return self._det
    
    @property
    def recognizer(self):
        key = self.active_recog_key
        if self._recog is None or self._recog_key != key:
            from app.services.recognizer import KhmerRecognizer  # ✅ Correct import
            meta = RECOG_MODELS[key]
            self._recog = KhmerRecognizer(str(meta.path), str(meta.charset))
            self._recog_key = key
        return self._recog
    
    def switch_det(self, model_id: str):
        if model_id not in DET_MODELS:
            raise ValueError(f"Unknown detection model: {model_id}")
        self.active_det_key = model_id
        self._det = None
    
    def switch_recog(self, model_id: str):
        if model_id not in RECOG_MODELS:
            raise ValueError(f"Unknown recognition model: {model_id}")
        self.active_recog_key = model_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._recog = None
    
    def warmup(self):
        _ = self.layout
        _ = self.det
        _ = self.recognizer
    
    def info(self) -> dict:
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "warm": self._recog is not None,
            "active_model": self.active_recog_key,
            "active_det_model": self.active_det_key,
            "models": []  # Populate from registries
        }