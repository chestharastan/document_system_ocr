"""FastAPI API server for the Khmer document OCR system.

Endpoints (consumed by the Next.js frontend at http://localhost:3000)
  GET  /api/health           -> readiness (and whether models are warm)
  POST /api/process          -> analyse one OR many uploaded images
  GET  /storage/...          -> saved page images + figure crops

A POST to /api/process accepts multipart `files` (one or several images). Each
image is saved, run through the layout -> detection -> recognition pipeline, and
returned with its served page-image URL, dimensions, and the staged block
structure the frontend animates and rebuilds into an editable document.

Run inside the conda env that has doclayout_yolo + the Khmer OCR stack:
  /home/thareah/miniconda3/envs/thaocr/bin/python -m uvicorn server:app --port 8200
(see run.sh).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
import pipeline

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
PAGES_DIR = STORAGE_DIR / "pages"
CROPS_DIR = STORAGE_DIR / "crops"
for d in (PAGES_DIR, CROPS_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

app = FastAPI(title="Khmer Document OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_method_list,
    allow_headers=settings.cors_header_list,
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "warm": pipeline._LAYOUT is not None
        and pipeline._DET is not None
        and pipeline._RECOG is not None
        and pipeline._RECOG_DIR_LOADED == pipeline.RECOG_DIR,
    }


@app.get("/api/models")
def models():
    """Report which models the pipeline is using (for the UI model dropdown)."""
    return pipeline.model_info()


class ModelRequest(BaseModel):
    model_id: str


@app.post("/api/models")
def set_model(req: ModelRequest):
    """Switch the active recognition model."""
    try:
        return pipeline.set_recognition_model(req.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/det-models")
def set_det_model(req: ModelRequest):
    """Switch the active detection model."""
    try:
        return pipeline.set_detection_model(req.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class ReprocessRequest(BaseModel):
    page_id: str


@app.post("/api/reprocess")
def reprocess(req: ReprocessRequest):
    """Re-run the full pipeline on a previously uploaded page (by page_id)."""
    matches = list(PAGES_DIR.glob(f"{req.page_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Unknown page_id")
    page_path = matches[0]
    crop_subdir = CROPS_DIR / req.page_id
    crop_subdir.mkdir(parents=True, exist_ok=True)
    try:
        result = pipeline.process_image(
            str(page_path),
            crop_dir=str(crop_subdir),
            crop_url_prefix=f"/storage/crops/{req.page_id}/",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    return {
        "page_id": req.page_id,
        "width": result["width"],
        "height": result["height"],
        "blocks": result["blocks"],
    }


@app.post("/api/process")
async def process(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded")

    pages = []
    for index, upload in enumerate(files):
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ext or upload.filename}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
            )

        page_id = uuid.uuid4().hex
        page_path = PAGES_DIR / f"{page_id}{ext}"
        page_path.write_bytes(await upload.read())

        crop_subdir = CROPS_DIR / page_id
        crop_subdir.mkdir(parents=True, exist_ok=True)

        try:
            result = pipeline.process_image(
                str(page_path),
                crop_dir=str(crop_subdir),
                crop_url_prefix=f"/storage/crops/{page_id}/",
            )
        except Exception as exc:  # surface a clean error to the UI
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

        pages.append({
            "page_id": page_id,
            "index": index,
            "filename": upload.filename,
            "image_url": f"/storage/pages/{page_path.name}",
            "width": result["width"],
            "height": result["height"],
            "blocks": result["blocks"],
        })

    return {"pages": pages}


class RegionBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Region(BaseModel):
    type: str
    box: RegionBox


class RegionsRequest(BaseModel):
    page_id: str
    regions: list[Region]


class LineBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class BlockWithLines(BaseModel):
    type: str
    box: RegionBox
    lines: list[LineBox] = []


class LinesRequest(BaseModel):
    page_id: str
    blocks: list[BlockWithLines]


@app.post("/api/regions")
def recognize_regions(req: RegionsRequest):
    """Re-run detection + recognition for manually edited regions of a page.

    The page image was already saved at upload time; we look it up by page_id
    and re-process only the supplied boxes, returning the same block structure
    the frontend uses so the analysis/document views can rebuild from the edits.
    """
    matches = list(PAGES_DIR.glob(f"{req.page_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Unknown page_id")
    page_path = matches[0]

    crop_subdir = CROPS_DIR / req.page_id
    crop_subdir.mkdir(parents=True, exist_ok=True)

    try:
        result = pipeline.process_regions(
            str(page_path),
            [r.model_dump() for r in req.regions],
            crop_dir=str(crop_subdir),
            crop_url_prefix=f"/storage/crops/{req.page_id}/",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    return result


@app.post("/api/lines")
def recognize_lines(req: LinesRequest):
    """Re-recognise a page from hand-edited text-line boxes (line editor)."""
    matches = list(PAGES_DIR.glob(f"{req.page_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Unknown page_id")
    page_path = matches[0]

    crop_subdir = CROPS_DIR / req.page_id
    crop_subdir.mkdir(parents=True, exist_ok=True)

    try:
        result = pipeline.process_lines(
            str(page_path),
            [b.model_dump() for b in req.blocks],
            crop_dir=str(crop_subdir),
            crop_url_prefix=f"/storage/crops/{req.page_id}/",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    return result


# Page images and figure crops are served from here.
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
