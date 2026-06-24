from pathlib import Path
from dataclasses import dataclass
from app.config import BASE_DIR

@dataclass
class ModelMeta:
    label: str
    path: Path
    charset: Path | None = None

MODELS_DIR = BASE_DIR / "models"

DET_MODELS: dict[str, ModelMeta] = {
    "line_level": ModelMeta(
        label="Line Level (yolo11s) - Khmer",
        path=MODELS_DIR / "detection" / "v1" / "yolo11s_khmer.pt",  # <-- your model
    ),
    "word_group": ModelMeta(
        label="Word Group (yolov8)",
        path=MODELS_DIR / "detection" / "word_group" / "best.pt",
    ),
}

RECOG_MODELS: dict[str, ModelMeta] = {
    "resnet_bilstm_ctc_300k": ModelMeta(
        label="ResNet BiLSTM CTC 300k",
        path=MODELS_DIR / "recognition" / "resnet_bilstm_ctc_300k",
        charset=MODELS_DIR / "recognition" / "resnet_bilstm_ctc_300k" / "char.json",
    ),
}