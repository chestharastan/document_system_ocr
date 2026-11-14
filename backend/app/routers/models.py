# app/routers/models.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_model_pool
from app.services.model_pool import ModelPool
from app.services.pipeline_service import engine_info  # v3 tables

router = APIRouter()


class ModelRequest(BaseModel):
    model_id: str


class StageModelRequest(BaseModel):
    stage: str  # layout | textdet | rec | font | structure | table
    model_id: str


async def _switch(pool: ModelPool, stage: str, model_id: str) -> dict:
    """Loads the new model off the event loop; a model that fails to load is refused (422), the old one stays."""
    try:
        await run_in_threadpool(pool.switch, stage, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return pool.info()


@router.get("/models")
async def fetch_models(pool: ModelPool = Depends(get_model_pool)):
    return pool.info()


@router.post("/models")
async def set_recognition_model(data: ModelRequest, pool: ModelPool = Depends(get_model_pool)):
    """v1 API: switch the recognition model."""
    return await _switch(pool, "rec", data.model_id)


@router.post("/det-models")
async def set_detection_model(data: ModelRequest, pool: ModelPool = Depends(get_model_pool)):
    """v1 API: switch the text-line detection model."""
    return await _switch(pool, "textdet", data.model_id)


@router.post("/layout-models")
async def set_layout_model(data: ModelRequest, pool: ModelPool = Depends(get_model_pool)):
    return await _switch(pool, "layout", data.model_id)


@router.post("/pipeline/models")
async def set_stage_model(data: StageModelRequest, pool: ModelPool = Depends(get_model_pool)):
    """Switch the active model of any stage (layout | textdet | rec | font | structure | table); loaded lazily on the next request."""
    return await _switch(pool, data.stage, data.model_id)


@router.get("/pipeline/info")
async def pipeline_info(pool: ModelPool = Depends(get_model_pool)):
    """Configured stages, active + available models (manifest entries), runtime details, post-corrector; v3: the
    structure engine (rules | model | llm) and the LLM engine's status (docs/TABLES_CONTRACT.md §2.4)."""
    return {**pool.pipeline_info(), **engine_info(pool)}


# ---------------------------------------------------------------------------------------------- v3 tables: settings
class PipelineSettings(BaseModel):
    structure_engine: str | None = None  # rules | model | llm | auto


@router.post("/pipeline/settings")
async def set_pipeline_settings(data: PipelineSettings, pool: ModelPool = Depends(get_model_pool)):
    """Switch the structure engine for the process: rules (geometry only), model (phone-tier structure + table
    models, the default when a structure model is loaded), llm (falls back to model / rules when the LLM engine is
    not configured or fails), auto. Answers with the effective engine and the LLM status."""
    if data.structure_engine is not None:
        v = data.structure_engine.strip().lower()
        if v not in ("rules", "model", "llm", "auto", ""):
            raise HTTPException(status_code=422, detail="structure_engine must be one of rules, model, llm, auto")
        pool.structure_engine = None if v in ("auto", "") else v
    return engine_info(pool)


@router.get("/health")
async def health(pool: ModelPool = Depends(get_model_pool)):
    return {"status": "ok", "warm": pool.info()["warm"]}
