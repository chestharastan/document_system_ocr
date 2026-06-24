from fastapi import APIRouter, Depends
from app.dependencies import get_model_pool
from app.services.model_pool import ModelPool

router = APIRouter()  # No prefix — we'll mount at root in main.py

@router.get("/models")
async def fetch_models(pool: ModelPool = Depends(get_model_pool)):
    """Return all available models and current state."""
    return pool.info()

@router.post("/models")
async def set_recognition_model(
    data: dict,  # { model_id: str }
    pool: ModelPool = Depends(get_model_pool)
):
    """Switch recognition model."""
    pool.switch_recog(data["model_id"])
    return pool.info()

@router.post("/det-models")
async def set_detection_model(
    data: dict,  # { model_id: str }
    pool: ModelPool = Depends(get_model_pool)
):
    """Switch detection model."""
    pool.switch_det(data["model_id"])
    return pool.info()