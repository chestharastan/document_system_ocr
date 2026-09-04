from pathlib import Path
from dataclasses import dataclass
from app.config import settings

@dataclass
class ModelMeta:
    label: str
    path: Path
    charset: Path | None = None

MODEL_ROOT = settings.model_root_path

DET_MODELS: dict[str, ModelMeta] = {
    "line_level": ModelMeta(
        label="Line Level (yolo11s) - Khmer",
        path=settings.det_weights_path
        or MODEL_ROOT / "v1" / "detection" / "best.pt",
    ),
    "word_group": ModelMeta(
        label="Word Group (yolov8)",
        path=MODEL_ROOT / "v1" / "detection" / "word_group" / "best.pt",
    ),
}

RECOG_MODELS: dict[str, ModelMeta] = {
    "resnet_bilstm_ctc": ModelMeta(
        label="ResNet BiLSTM CTC",
        path=settings.recog_dir_path
        or MODEL_ROOT / "v1" / "recognition",
        charset=settings.recog_charset_path
        or MODEL_ROOT / "v1" / "recognition" / "char.json",
    ),
}
