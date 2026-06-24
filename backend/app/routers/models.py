# app/routers/models.py
from fastapi import APIRouter, Depends
from app.dependencies import get_model_pool
from app.services.model_pool import ModelPool

router = APIRouter()

@router.get("/models")
async def fetch_models(pool: ModelPool = Depends(get_model_pool)):
    return pool.info()

@router.post("/models")
async def set_recognition_model(
    data: dict,
    pool: ModelPool = Depends(get_model_pool)
):
    pool.switch_recog(data["model_id"])
    return pool.info()

@router.post("/det-models")
async def set_detection_model(
    data: dict,
    pool: ModelPool = Depends(get_model_pool)
):
    pool.switch_det(data["model_id"])
    return pool.info()