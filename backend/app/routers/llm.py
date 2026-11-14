"""LLM structure engine endpoints (contract §2.4, v3 llm):
    GET  /llm/status            provider / model / configured / reachable / busy
    POST /llm/structure         body = one full process result (or {"page": result}).
                                ?async=1 (or body "async": true): queue a background job -> 202 {"id", "status", ...}
                                (the web app uses this: a 14B model on a laptop needs minutes per page).
                                Without it: synchronous, limited to LLM_SYNC_TIMEOUT_S (default 60 s) -> the rewritten
                                page result; 503 when no provider is configured / reachable / the server is busy,
                                502 when the model failed or ran out of time.
    GET  /llm/jobs/{id}         {"id", "status": queued|running|done|failed, "position", "elapsed_s", "error",
                                 "stage", "progress" (0..1), "lines_done", "lines_total",
                                 "result" (status done: the rewritten page result)}; 404 for an unknown / expired job
    GET  /llm/jobs              the recent jobs without results."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.services import llm_providers as lp
from app.services.llm_providers import llm_status
from app.services.llm_structure import JOBS, apply_llm_structure, submit_llm_job, sync_timeout_s

router = APIRouter(prefix="/llm", tags=["llm"])


def _busy():
    prov = lp.get_provider()
    return prov.busy() if prov is not None else None


@router.get("/status")
async def status():
    st = await run_in_threadpool(llm_status)
    if st.get("configured") and st.get("reachable"):
        st["busy"] = await run_in_threadpool(_busy)
    return st


@router.post("/structure")
async def structure(body: dict = Body(...), async_: bool = Query(False, alias="async")):
    """Body = the process result itself, or {"page": <process result>, ...} (the web app sends both shapes at once:
    the page spread at the top level and again under "page")."""
    want_async = async_ or bool(body.pop("async", False))
    result = body.get("page") if isinstance(body.get("page"), dict) else body
    if not isinstance(result.get("blocks"), list) or not result.get("width") or not result.get("height"):
        raise HTTPException(status_code=422, detail="body must be a process result (width, height, blocks[]) or {page: result}")
    result.pop("page", None)  # never echo a stale nested copy of the page
    st = await run_in_threadpool(llm_status)
    if not st["configured"]:
        raise HTTPException(status_code=503, detail="LLM engine not configured (LLM_PROVIDER)")
    if not st["reachable"]:
        raise HTTPException(status_code=503, detail=f"LLM endpoint not reachable ({st.get('provider')} / {st.get('model')})")
    if want_async:
        job = await run_in_threadpool(submit_llm_job, result)
        return JSONResponse(status_code=202, content={**job, "poll": f"/llm/jobs/{job['id']}"})
    out = await run_in_threadpool(apply_llm_structure, result, sync_timeout_s(), False)
    err = (out.get("timings") or {}).get("llm_error")
    if err:
        code = 503 if err.startswith("LLM busy") else 502
        raise HTTPException(status_code=code, detail=f"LLM structure failed: {err}")
    out["structure_engine"] = "llm"
    return out


@router.get("/jobs/{job_id}")
async def job(job_id: str):
    j = JOBS.public(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="unknown or expired LLM job")
    return j


@router.get("/jobs")
async def jobs():
    return JOBS.list()
