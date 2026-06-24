from fastapi import Request
from app.services.model_pool import ModelPool

def get_model_pool(request: Request) -> ModelPool:
    """Get the shared ModelPool from app state."""
    return request.app.state.models

def get_pipeline(request: Request):
    """Get a PipelineService instance with the shared model pool."""
    from app.services.pipeline_service import PipelineService
    return PipelineService(get_model_pool(request))