from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=["http://localhost:3000"],  # Your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers at ROOT to match frontend URLs
app.include_router(models.router)
app.include_router(pipeline.router)