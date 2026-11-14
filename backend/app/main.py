import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import export, models, pipeline
from app.services.model_pool import ModelPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelPool()
    if settings.warmup:
        app.state.models.warmup()
    yield


app = FastAPI(title="Khmer OCR API", version="2.0.0", lifespan=lifespan)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_method_list,
    allow_headers=settings.cors_header_list,
)

@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """422 without echoing the raw input (it can be non-JSON-serialisable, e.g. Infinity, or huge)."""
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Refuse oversized requests from Content-Length before the body is parsed (per-file caps are in /process)."""
    limit = settings.max_upload_mb * max(1, settings.max_files) * 2**20
    try:
        n = int(request.headers.get("content-length") or 0)
    except ValueError:
        n = 0
    if n > limit:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


# Mount routers at ROOT to match frontend URLs
app.include_router(models.router)
app.include_router(pipeline.router)
app.include_router(export.router)
from app.routers import llm as llm_router; app.include_router(llm_router.router)  # v3 llm
settings.page_dir.mkdir(parents=True, exist_ok=True)
settings.crop_dir.mkdir(parents=True, exist_ok=True)
# served separately so PAGE_DIR / CROP_DIR can live anywhere without exposing their parent folders
@app.middleware("http")
async def no_cache_demo(request: Request, call_next):
    """The demo page changes often: never let the browser cache it."""
    response = await call_next(request)
    if request.url.path.startswith("/demo"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/demo/")


app.mount("/demo", StaticFiles(directory=str(settings.demo_dir), html=True), name="demo")
if settings.demo_fonts_dir and Path(settings.demo_fonts_dir).is_dir():
    app.mount("/demo-fonts", StaticFiles(directory=str(settings.demo_fonts_dir)), name="demo_fonts")
app.mount("/storage/pages", StaticFiles(directory=settings.page_dir), name="pages")
app.mount("/storage/crops", StaticFiles(directory=settings.crop_dir), name="crops")
