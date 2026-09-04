from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Backend root = directory containing app/, storage/, etc.
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
PROJECT_ROOT = BASE_DIR.parent


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _project_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else PROJECT_ROOT / value


class Settings(BaseSettings):
    # Model root
    model_root: Path = Field(default=PROJECT_ROOT / ".model", alias="MODEL_ROOT")

    # Layout
    layout_weights: Path = Field(
        default=Path(".model/v1/layout/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"),
        alias="LAYOUT_WEIGHTS",
    )
    layout_imgsz: int = Field(default=1120, alias="LAYOUT_IMGSZ")
    layout_conf: float = Field(default=0.25, alias="LAYOUT_CONF")
    
    # Detection
    det_weights: Path | None = Field(default=None, alias="DET_WEIGHTS")
    det_model: str = Field(default="line_level", alias="DET_MODEL")
    det_conf: float = Field(default=0.25, alias="DET_CONF")
    det_group_lines: bool = Field(default=True, alias="DET_GROUP_LINES")
    det_line_y_overlap: float = Field(default=0.35, alias="DET_LINE_Y_OVERLAP")
    det_line_y_tolerance: float = Field(default=0.55, alias="DET_LINE_Y_TOLERANCE")
    det_line_x_gap_ratio: float = Field(default=1.75, alias="DET_LINE_X_GAP_RATIO")
    det_line_x_gap_min: float = Field(default=36.0, alias="DET_LINE_X_GAP_MIN")
    
    # Recognition
    recog_dir: Path | None = Field(default=None, alias="RECOG_DIR")
    recog_model: str = Field(default="resnet_bilstm_ctc", alias="RECOG_MODEL")
    recog_charset: Path | None = Field(default=None, alias="RECOG_CHARSET")
    
    # Storage
    page_dir: Path = BASE_DIR / "storage" / "pages"
    crop_dir: Path = BASE_DIR / "storage" / "crops"
    crop_url_prefix: str = ""

    # CORS
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: str = Field(default="*", alias="CORS_ALLOW_METHODS")
    cors_allow_headers: str = Field(default="*", alias="CORS_ALLOW_HEADERS")

    @property
    def cors_origin_list(self) -> list[str]:
        return _csv_list(self.cors_origins)

    @property
    def cors_method_list(self) -> list[str]:
        return _csv_list(self.cors_allow_methods)

    @property
    def cors_header_list(self) -> list[str]:
        return _csv_list(self.cors_allow_headers)

    @field_validator("det_weights", "recog_dir", "recog_charset", mode="before")
    @classmethod
    def empty_path_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def model_root_path(self) -> Path:
        path = _project_path(self.model_root)
        if path is None:
            raise ValueError("MODEL_ROOT cannot be empty")
        return path

    @property
    def layout_weights_path(self) -> Path:
        path = _project_path(self.layout_weights)
        if path is None:
            raise ValueError("LAYOUT_WEIGHTS cannot be empty")
        return path

    @property
    def det_weights_path(self) -> Path | None:
        return _project_path(self.det_weights)

    @property
    def recog_dir_path(self) -> Path | None:
        return _project_path(self.recog_dir)

    @property
    def recog_charset_path(self) -> Path | None:
        return _project_path(self.recog_charset)
    
    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"

settings = Settings()
