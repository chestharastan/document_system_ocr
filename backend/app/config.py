from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Backend root = directory containing app/, storage/, etc.
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
PROJECT_ROOT = BASE_DIR.parent


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def project_path(value: Path | str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    value = Path(value)
    return value if value.is_absolute() else PROJECT_ROOT / value


class Settings(BaseSettings):
    """All knobs can be set in backend/.env or the environment. Paths are relative to the project root."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore",
                                      populate_by_name=True)

    # ---- model registry -------------------------------------------------------------------------------------
    model_root: Path = Field(default=Path(".model"), alias="MODEL_ROOT")
    # manifest of the drop-in models (see README "Models"); relative paths inside it are relative to its folder
    model_manifest: Path = Field(default=Path(".model/v2/models.json"), alias="MODEL_MANIFEST")
    # active model id per stage; empty = the manifest's "defaults" (or the first available model of that stage)
    layout_model: str = Field(default="", alias="LAYOUT_MODEL")
    textdet_model: str = Field(default="", alias="TEXTDET_MODEL")
    rec_model: str = Field(default="", alias="REC_MODEL")
    font_model: str = Field(default="", alias="FONT_MODEL")  # stage 3b: font family / weight per line (optional)
    structure_model: str = Field(default="", alias="STRUCTURE_MODEL")  # stage 4: roles / paragraphs / levels / reading order (optional)
    postcorrect: str = Field(default="none", alias="POSTCORRECT")  # future 4th stage; only "none" ships
    verify_sha256: bool = Field(default=True, alias="VERIFY_SHA256")  # warn (not fail) on a checksum mismatch
    warmup: bool = Field(default=True, alias="WARMUP")  # load all active models at startup

    # ---- onnxruntime ------------------------------------------------------------------------------------------
    ort_device: str = Field(default="auto", alias="ORT_DEVICE")  # auto | cpu | cuda
    ort_threads: int = Field(default=4, alias="ORT_THREADS")

    # ---- stage 1: layout ------------------------------------------------------------------------------------------
    layout_conf: float = Field(default=0.3, alias="LAYOUT_CONF")
    layout_iou: float = Field(default=0.5, alias="LAYOUT_IOU")
    layout_max_det: int = Field(default=300, alias="LAYOUT_MAX_DET")
    layout_dedupe_iou: float = Field(default=0.7, alias="LAYOUT_DEDUPE_IOU")  # cross-class duplicate removal
    page_sized_figure: float = Field(default=0.5, alias="PAGE_SIZED_FIGURE")  # figure > this page fraction = ignored

    # ---- stage 2: text-line detection (DB); defaults = det/web/server.py DET_KW + DET_SCALE ------------------------
    det_thresh: float = Field(default=0.3, alias="DET_THRESH")
    det_box_thresh: float = Field(default=0.5, alias="DET_BOX_THRESH")
    det_unclip_ratio: float = Field(default=1.6, alias="DET_UNCLIP_RATIO")
    det_v_extra: float = Field(default=1.0, alias="DET_V_EXTRA")
    det_long_min: int = Field(default=1600, alias="DET_LONG_MIN")
    det_long_max: int = Field(default=2200, alias="DET_LONG_MAX")
    det_cascade: bool = Field(default=True, alias="DET_CASCADE")
    det_hw_pass: bool = Field(default=True, alias="DET_HW_PASS")  # detect handwriting blocks again at the DET_HW_LONG page scale
    det_hw_long: int = Field(default=1600, alias="DET_HW_LONG")
    merge_paragraphs: bool = Field(default=True, alias="MERGE_PARAGRAPHS")  # stack per-line text blocks into paragraphs  # re-detect inside text regions the page pass missed

    # ---- stage 3: recognition ---------------------------------------------------------------------------------------
    rec_pad_x: float = Field(default=0.25, alias="REC_PAD_X")  # line crop margin left/right, in line heights
    rec_pad_y: float = Field(default=0.08, alias="REC_PAD_Y")  # line crop margin top/bottom, in line heights
    rec_batch: int = Field(default=16, alias="REC_BATCH")

    # ---- document structure -----------------------------------------------------------------------------------------
    detect_bullets: bool = Field(default=True, alias="DETECT_BULLETS")  # find the "-" / "o" list marks the detector drops
    # folder served at /demo (the one-page demo by default; app/static_new = the KhmerOCR app, run it on another port)
    demo_dir: Path = Field(default=BASE_DIR / "app" / "static", alias="DEMO_DIR")
    # folder of .ttf files the demo page loads (@font-face) to draw the recognised page in the detected typeface
    demo_fonts_dir: Path = Field(default=Path("/home/hello/ocr/work/fonts"), alias="DEMO_FONTS_DIR")
    detect_colors: bool = Field(default=True, alias="DETECT_COLORS")  # ink colour per line / word (text_color hex)
    detect_bold: bool = Field(default=True, alias="DETECT_BOLD")  # bold words from stroke weight (needs a CTC recogniser)
    bold_model: bool = Field(default=True, alias="BOLD_MODEL")  # per-word bold from the font model's weight head (else erosion test)
    use_structure_model: bool = Field(default=True, alias="USE_STRUCTURE_MODEL")  # learned structure on top of the rules when registered
    merge_dup_lines: bool = Field(default=True, alias="MERGE_DUP_LINES")  # overlapping detections of one line -> one box
    reread_markers: bool = Field(default=True, alias="REREAD_MARKERS")  # ink marks read by the recogniser ("2." is not a dot)
    split_text_markers: bool = Field(default=True, alias="SPLIT_TEXT_MARKERS")  # "1. text" / "-text": marker -> bullet
    structure_regroup: bool = Field(default=True, alias="STRUCTURE_REGROUP")  # paragraphs rebuilt across layout blocks from the structure model
    detect_fonts: bool = Field(default=True, alias="DETECT_FONTS")  # run the font classifier when a model is registered
    split_figures: bool = Field(default=True, alias="SPLIT_FIGURES")  # text lines inside a figure box leave it as text
    grid_tables: bool = Field(default=True, alias="GRID_TABLES")  # ruled grids the layout missed become table blocks
    detect_stamps: bool = Field(default=True, alias="DETECT_STAMPS")  # red ink stamps by colour; text under them read on an inpainted page
    detect_decorations: bool = Field(default=True, alias="DETECT_DECORATIONS")  # ornament / rule lines -> "decoration" crops
    snap_crops: bool = Field(default=True, alias="SNAP_CROPS")  # grow figure / logo / stamp boxes to the ink they cut
    snap_lines: bool = Field(default=True, alias="SNAP_LINES")  # grow text line boxes to cut glyphs
    relabel_by_position: bool = Field(default=True, alias="RELABEL_BY_POSITION")  # letterhead -> header, logo, title, footer

    # ---- legacy torch stack (only used when a legacy model id is selected) ------------------------------------------
    legacy_layout_imgsz: int = Field(default=1120, alias="LAYOUT_IMGSZ")
    legacy_det_conf: float = Field(default=0.25, alias="DET_CONF")

    # ---- input limits ----------------------------------------------------------------------------------------------
    max_upload_mb: float = Field(default=40.0, alias="MAX_UPLOAD_MB")         # per file -> 413
    max_files: int = Field(default=50, alias="MAX_FILES")                     # per /process request -> 413
    max_image_side: int = Field(default=3600, alias="MAX_IMAGE_SIDE")         # longer pages are downscaled on upload
    max_image_pixels: int = Field(default=150_000_000, alias="MAX_IMAGE_PIXELS")  # decompression-bomb guard -> 422

    # ---- storage ------------------------------------------------------------------------------------------------------
    page_dir: Path = BASE_DIR / "storage" / "pages"
    crop_dir: Path = BASE_DIR / "storage" / "crops"

    # ---- CORS ----------------------------------------------------------------------------------------------------------
    cors_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000", alias="CORS_ORIGINS")
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

    @property
    def model_root_path(self) -> Path:
        return project_path(self.model_root)

    @property
    def manifest_path(self) -> Path:
        return project_path(self.model_manifest)


settings = Settings()
