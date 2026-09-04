from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.dependencies import get_pipeline
from app.schemas.pipeline import ProcessImageResponse, ProcessRegionsRequest, ProcessLinesRequest

router = APIRouter()  # No prefix

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _find_page_path(page_id: str) -> Path:
    from app.config import settings

    matches = list(settings.page_dir.glob(f"{page_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Unknown page_id")
    return matches[0]


def _model_dump(value):
    return value.model_dump() if hasattr(value, "model_dump") else value

@router.post("/process")
async def process_files(
    files: list[UploadFile] = File(...),
    recognize: bool = Form(True),
    pipeline=Depends(get_pipeline),
):
    """Upload images and run full OCR pipeline."""
    from app.config import settings

    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded")

    settings.page_dir.mkdir(parents=True, exist_ok=True)
    settings.crop_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for index, file in enumerate(files):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ext or file.filename}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
            )

        page_id = uuid.uuid4().hex
        page_path = settings.page_dir / f"{page_id}{ext}"
        page_path.write_bytes(await file.read())

        crop_subdir = settings.crop_dir / page_id
        crop_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = pipeline.process_image(
                str(page_path),
                crop_dir=str(crop_subdir),
                crop_url_prefix=f"/storage/crops/{page_id}/",
                recognize=recognize
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

        results.append({
            "page_id": page_id,
            "index": index,
            "filename": file.filename,
            "image_url": f"/storage/pages/{page_path.name}",
            "width": result["width"],
            "height": result["height"],
            "blocks": result["blocks"],
        })
    
    return {"pages": results}  # Frontend expects { pages: [...] }

@router.post("/reprocess")
async def reprocess_page(
    data: dict,  # { page_id: str }
    pipeline=Depends(get_pipeline),
):
    """Re-run OCR on a previously uploaded page."""
    from app.config import settings

    page_id = data.get("page_id")
    if not page_id:
        raise HTTPException(status_code=422, detail="page_id is required")

    page_path = _find_page_path(page_id)
    crop_subdir = settings.crop_dir / page_id
    crop_subdir.mkdir(parents=True, exist_ok=True)
    try:
        return pipeline.process_image(
            str(page_path),
            crop_dir=str(crop_subdir),
            crop_url_prefix=f"/storage/crops/{page_id}/",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

@router.post("/regions")
async def recognize_regions(
    data: ProcessRegionsRequest,
    pipeline=Depends(get_pipeline),
):
    """Re-run detection + recognition on edited regions."""
    from app.config import settings

    page_path = _find_page_path(data.page_id)
    crop_subdir = settings.crop_dir / data.page_id
    crop_subdir.mkdir(parents=True, exist_ok=True)
    try:
        return pipeline.process_regions(
            str(page_path),
            [_model_dump(region) for region in data.regions],
            crop_dir=str(crop_subdir),
            crop_url_prefix=f"/storage/crops/{data.page_id}/",
            recognize=data.recognize
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

@router.post("/lines")
async def recognize_lines(
    data: ProcessLinesRequest,
    pipeline=Depends(get_pipeline),
):
    """Re-recognize from hand-edited line boxes."""
    from app.config import settings

    page_path = _find_page_path(data.page_id)
    crop_subdir = settings.crop_dir / data.page_id
    crop_subdir.mkdir(parents=True, exist_ok=True)
    try:
        return pipeline.process_lines(
            str(page_path),
            [_model_dump(block) for block in data.blocks],
            crop_dir=str(crop_subdir),
            crop_url_prefix=f"/storage/crops/{data.page_id}/",
            recognize=data.recognize
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
