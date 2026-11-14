import logging
import re
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import BASE_DIR, PROJECT_ROOT, settings
from app.dependencies import get_pipeline
from app.schemas.pipeline import ProcessLinesRequest, ProcessRegionsRequest, ReprocessRequest
from app.services.image_io import ImageDecodeError, decode_image

log = logging.getLogger("app.pipeline")

router = APIRouter()  # No prefix

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PAGE_ID_RE = re.compile(r"[0-9a-f]{32}")


def _page_paths(page_id) -> tuple[Path, Path, str]:
    """page_id -> (stored page image, its crop folder, crop URL prefix). page_id must be a uuid4 hex as issued by
    /process (no paths, no glob characters) and the resolved paths must stay inside the storage folders."""
    if not isinstance(page_id, str) or not PAGE_ID_RE.fullmatch(page_id):
        raise HTTPException(status_code=422, detail="page_id must be the 32-character id returned by /process")
    pages_root = settings.page_dir.resolve()
    crops_root = settings.crop_dir.resolve()
    matches = [p for p in (settings.page_dir / f"{page_id}{ext}" for ext in sorted(ALLOWED_EXTS)) if p.is_file()]
    if not matches or matches[0].resolve().parent != pages_root:
        raise HTTPException(status_code=404, detail="Unknown page_id")
    crop_subdir = (settings.crop_dir / page_id).resolve()
    if crop_subdir.parent != crops_root:
        raise HTTPException(status_code=404, detail="Unknown page_id")
    crop_subdir.mkdir(parents=True, exist_ok=True)
    return matches[0], crop_subdir, f"/storage/crops/{page_id}/"


def _model_dump(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def _safe_message(exc: Exception) -> str:
    """Error text for the client without absolute server paths."""
    msg = str(exc)
    for root in sorted({str(settings.page_dir), str(settings.crop_dir), str(BASE_DIR), str(PROJECT_ROOT)},
                       key=len, reverse=True):
        msg = msg.replace(root, "<server>")
    return f"{type(exc).__name__}: {msg}"[:300]


async def _run(fn, *args, **kwargs):
    """Run a blocking pipeline call off the event loop; turn failures into a clean error for the UI."""
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except HTTPException:
        raise
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Could not read the stored page: {exc}")
    except Exception as exc:
        log.exception("processing failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {_safe_message(exc)}")


async def _read_upload(file: UploadFile) -> bytes:
    """Read one upload, refusing anything above MAX_UPLOAD_MB with 413 (read in chunks, never more than the cap)."""
    limit = int(settings.max_upload_mb * 2**20)
    chunks, n = [], 0
    while chunk := await file.read(1 << 20):
        n += len(chunk)
        if n > limit:
            raise HTTPException(status_code=413,
                                detail=f"'{file.filename}' is larger than {settings.max_upload_mb:g} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _store_upload(buf: bytes, ext: str, filename: str) -> tuple[str, Path]:
    """Decode + validate before anything is written; store the original bytes, or the normalised image (alpha
    composited / downscaled / 8-bit) as PNG when decoding changed the pixels."""
    try:
        bgr, changed = decode_image(buf)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Could not read '{filename}' as an image: {exc}")
    page_id = uuid.uuid4().hex
    if changed:
        page_path = settings.page_dir / f"{page_id}.png"
        if not cv2.imwrite(str(page_path), bgr):
            raise HTTPException(status_code=500, detail="Could not store the page")
    else:
        page_path = settings.page_dir / f"{page_id}{ext}"
        page_path.write_bytes(buf)
    return page_id, page_path


@router.post("/process")
async def process_files(
    files: list[UploadFile] = File(...),
    recognize: bool = Form(True),
    pipeline=Depends(get_pipeline),
):
    """Upload images and run full OCR pipeline."""
    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded")
    if len(files) > settings.max_files:
        raise HTTPException(status_code=413, detail=f"At most {settings.max_files} files per request")

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
        buf = await _read_upload(file)
        page_id, page_path = await run_in_threadpool(_store_upload, buf, ext, file.filename or "")
        del buf
        crop_subdir = settings.crop_dir / page_id
        crop_subdir.mkdir(parents=True, exist_ok=True)

        result = await _run(
            pipeline.process_image,
            str(page_path),
            crop_dir=str(crop_subdir),
            crop_url_prefix=f"/storage/crops/{page_id}/",
            recognize=recognize,
        )

        # v1 keys first; timings / models / ignored_regions are v2 extras
        results.append({
            "page_id": page_id,
            "index": index,
            "filename": file.filename,
            "image_url": f"/storage/pages/{page_path.name}",
            **result,
        })

    return {"pages": results}  # Frontend expects { pages: [...] }


@router.post("/reprocess")
async def reprocess_page(data: ReprocessRequest, pipeline=Depends(get_pipeline)):
    """Re-run OCR on a previously uploaded page."""
    page_path, crop_subdir, prefix = _page_paths(data.page_id)
    result = await _run(pipeline.process_image, str(page_path), crop_dir=str(crop_subdir), crop_url_prefix=prefix)
    return {"page_id": data.page_id, "image_url": f"/storage/pages/{page_path.name}", **result}


@router.post("/regions")
async def recognize_regions(data: ProcessRegionsRequest, pipeline=Depends(get_pipeline)):
    """Re-run detection + recognition on edited regions."""
    page_path, crop_subdir, prefix = _page_paths(data.page_id)
    return await _run(
        pipeline.process_regions,
        str(page_path),
        [_model_dump(region) for region in data.regions],
        crop_dir=str(crop_subdir),
        crop_url_prefix=prefix,
        recognize=data.recognize,
    )


@router.post("/lines")
async def recognize_lines(data: ProcessLinesRequest, pipeline=Depends(get_pipeline)):
    """Re-recognize from hand-edited line boxes."""
    page_path, crop_subdir, prefix = _page_paths(data.page_id)
    return await _run(
        pipeline.process_lines,
        str(page_path),
        [_model_dump(block) for block in data.blocks],
        crop_dir=str(crop_subdir),
        crop_url_prefix=prefix,
        recognize=data.recognize,
    )
