from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

# Project root = directory containing app/, models/, storage/, etc.
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/

class Settings(BaseSettings):
    # Layout
    layout_weights: Path = Field(default=..., alias="LAYOUT_WEIGHTS")
    
    # Detection
    det_weights: Path | None = Field(default=None, alias="DET_WEIGHTS")
    det_conf: float = Field(default=0.25, alias="DET_CONF")
    
    # Recognition
    recog_dir: Path | None = Field(default=None, alias="RECOG_DIR")
    recog_model: str = Field(default="resnet_bilstm_ctc", alias="RECOG_MODEL")
    recog_charset: Path | None = Field(default=None, alias="RECOG_CHARSET")
    
    # Storage
    crop_dir: Path = BASE_DIR / "storage" / "crops"
    crop_url_prefix: str = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()