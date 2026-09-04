from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, settings
from app.services.model_pool import ModelPool
from app.routers import models, pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelPool()
    app.state.models.warmup()
    yield

app = FastAPI(title="Khmer OCR API", lifespan=lifespan)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_method_list,
    allow_headers=settings.cors_header_list,
)

# Mount routers at ROOT to match frontend URLs
app.include_router(models.router)
app.include_router(pipeline.router)
(BASE_DIR / "storage").mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=BASE_DIR / "storage"), name="storage")
