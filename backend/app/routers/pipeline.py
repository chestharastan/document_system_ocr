from fastapi import APIRouter, Depends, UploadFile, File, Form
from app.dependencies import get_pipeline
from app.schemas.pipeline import ProcessImageResponse, ProcessRegionsRequest, ProcessLinesRequest

router = APIRouter()  # No prefix

@router.post("/process")
async def process_files(
    files: list[UploadFile] = File(...),
    recognize: bool = Form(True),
    pipeline=Depends(get_pipeline),
):
    """Upload images and run full OCR pipeline."""
    import tempfile
    from pathlib import Path
    from app.config import settings
    
    results = []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        result = pipeline.process_image(
            tmp_path,
            crop_dir=str(settings.crop_dir),
            crop_url_prefix=settings.crop_url_prefix,
            recognize=recognize
        )
        Path(tmp_path).unlink(missing_ok=True)
        results.append(result)
    
    return {"pages": results}  # Frontend expects { pages: [...] }

@router.post("/reprocess")
async def reprocess_page(
    data: dict,  # { page_id: str }
    pipeline=Depends(get_pipeline),
):
    """Re-run OCR on a previously uploaded page."""
    # You'll need to store original images and retrieve by page_id
    # For now, placeholder:
    raise NotImplementedError("Reprocess requires image storage")

@router.post("/regions")
async def recognize_regions(
    data: ProcessRegionsRequest,
    pipeline=Depends(get_pipeline),
):
    """Re-run detection + recognition on edited regions."""
    from app.config import settings
    return pipeline.process_regions(
        data.image_path,
        data.regions,
        crop_dir=str(settings.crop_dir),
        crop_url_prefix=settings.crop_url_prefix,
        recognize=data.recognize
    )

@router.post("/lines")
async def recognize_lines(
    data: ProcessLinesRequest,
    pipeline=Depends(get_pipeline),
):
    """Re-recognize from hand-edited line boxes."""
    from app.config import settings
    return pipeline.process_lines(
        data.image_path,
        data.blocks,
        crop_dir=str(settings.crop_dir),
        crop_url_prefix=settings.crop_url_prefix,
        recognize=data.recognize
    )