"""QA reproductions (reviewer). Every test here asserts the CORRECT behaviour, so a failing test = a confirmed defect.

Run:  CUDA_VISIBLE_DEVICES= ../../../khm_repo_clone/det_venv/bin/python -m pytest -q tests/test_qa_backend.py
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.reading_order import xy_cut
from app.services.stages.layout_yolo_onnx import letterbox


# --------------------------------------------------------------------------- helpers
def _upload(client, path, name=None):
    with open(path, "rb") as f:
        return client.post("/process", files=[("files", (name or Path(path).name, f, "image/png"))])


class _StubRec:
    def recognize(self, crops):
        return [("x", 1.0) for _ in crops]


class _StubDet:
    def detect(self, image, **kw):
        return []


class _StubPool:
    """Model-free pool: no lines detected, recogniser returns 'x'."""
    textdet = _StubDet()
    recognizer = _StubRec()
    active = {"layout": "l", "textdet": "d", "rec": "r"}
    device = "cpu"

    class postcorrector:  # noqa: N801
        name = "none"

        @staticmethod
        def correct(lines, ctx):
            return lines


def _record_warp(monkeypatch):
    """Replace cv2.warpPerspective in the pipeline module with a recorder that refuses to allocate."""
    import app.services.pipeline_service as ps
    seen = []

    def fake(src, M, dsize, **kw):
        seen.append(tuple(int(v) for v in dsize))
        return np.full((min(dsize[1], 64), min(dsize[0], 64), 3), 255, np.uint8)

    monkeypatch.setattr(ps.cv2, "warpPerspective", fake)
    return seen


# --------------------------------------------------------------------------- security: page_id
def test_qa_page_id_path_traversal_reads_files_outside_storage(client, synthetic_page, tmp_path):
    """/reprocess (also /regions, /lines) globs settings.page_dir / f"{page_id}.*" and creates crop_dir / page_id
    without validating page_id: '../..' escapes storage, the server OCRs an arbitrary image on disk and creates
    directories / writes crop PNGs outside storage/crops."""
    from app.config import settings
    secret_dir = tmp_path / "outside_storage"
    secret_dir.mkdir()
    secret = secret_dir / "secret_doc.png"
    secret.write_bytes(Path(synthetic_page).read_bytes())
    page_id = os.path.relpath(secret.with_suffix(""), settings.page_dir)
    assert page_id.startswith("..")
    r = client.post("/reprocess", json={"page_id": page_id})
    escaped_crop_dir = (settings.crop_dir / page_id).resolve()
    created_outside = escaped_crop_dir.exists() and settings.crop_dir.resolve() not in escaped_crop_dir.parents
    assert r.status_code in (400, 404, 422), f"traversal page_id accepted: {r.status_code}, " \
        f"text={r.json().get('blocks', [{}])[0].get('text', '')[:40]!r}, crop dir created outside={created_outside}"


def test_qa_page_id_glob_wildcard_matches_other_pages(client, synthetic_page):
    """page_id is used as a glob pattern: '*' (or 'a*') resolves to some other user's page."""
    _upload(client, synthetic_page)
    r = client.post("/reprocess", json={"page_id": "*"})
    assert r.status_code in (400, 404, 422), f"wildcard page_id accepted -> {r.status_code} {r.json().get('image_url')}"


# --------------------------------------------------------------------------- uploads
def test_qa_upload_has_no_size_limit_and_bad_content_is_500(client):
    """POST /process reads the whole upload into memory and writes it to disk with no size cap; a non-image with an
    allowed extension is stored, then fails inside the pipeline -> 500 (not 413/422) and the file stays on disk."""
    from app.config import settings
    before = set(settings.page_dir.glob("*"))
    blob = os.urandom(45 * 2**20)  # 45 MB of garbage, the reference server caps bodies at 40 MB
    r = client.post("/process", files=[("files", ("big.png", blob, "image/png"))])
    left = [p for p in set(settings.page_dir.glob("*")) - before if p.stat().st_size >= len(blob)]
    assert r.status_code in (413, 422), f"got {r.status_code}: {r.text[:120]}; garbage file left on disk: {bool(left)}"


def test_qa_huge_image_not_downscaled(client, tmp_path):
    """No MAX_SIDE cap (reference det/web/server.py decode() caps the long side at 3600): an 9000x9000 PNG of a few
    hundred KB is decoded at full size (243 MB BGR); 30000x30000 would be 2.7 GB per copy."""
    p = tmp_path / "huge.png"
    cv2.imwrite(str(p), np.full((9000, 9000, 3), 255, np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert p.stat().st_size < 2 * 2**20
    r = _upload(client, p)
    assert r.status_code == 200
    page = r.json()["pages"][0]
    assert max(page["width"], page["height"]) <= 3600, f"processed at {page['width']}x{page['height']}"


def test_qa_rgba_transparent_png_becomes_black(client, synthetic_page, tmp_path):
    """cv2.imread(IMREAD_COLOR) drops alpha. A transparent PNG (background RGBA 0,0,0,0, black opaque text - the
    usual export of 'transparent' scans/screenshots) becomes an all-black page and nothing is read.
    The reference decode() composites alpha onto white."""
    g = cv2.cvtColor(cv2.imread(str(synthetic_page)), cv2.COLOR_BGR2GRAY)
    alpha = (255 - g).astype(np.uint8)            # ink opaque, paper transparent
    rgba = np.zeros((*g.shape, 4), np.uint8)
    rgba[..., 3] = alpha                           # RGB = 0 everywhere
    p = tmp_path / "transparent.png"
    cv2.imwrite(str(p), rgba)
    r = _upload(client, p)
    assert r.status_code == 200, r.text
    lines = [l for b in r.json()["pages"][0]["blocks"] for l in b["lines"] if l["text"].strip()]
    assert len(lines) >= 6, f"only {len(lines)} lines read from the transparent PNG (page decoded as black)"


# --------------------------------------------------------------------------- crash paths
def test_qa_letterbox_extreme_aspect_ratio_crashes():
    """letterbox(): nh = round(h * r) can be 0 (e.g. a 2 x 5000 strip) -> cv2.resize assertion -> HTTP 500."""
    img = np.full((2, 5000, 3), 255, np.uint8)
    try:
        out, r, pad = letterbox(img, (1024, 1024))
    except cv2.error as e:
        pytest.fail(f"letterbox crashed on a 5000x2 image: {str(e).splitlines()[0]}")
    assert out.shape[:2] == (1024, 1024)


def test_qa_extreme_aspect_upload_returns_500(client, tmp_path):
    p = tmp_path / "strip.png"
    cv2.imwrite(str(p), np.full((5000, 2, 3), 255, np.uint8))
    r = _upload(client, p)
    assert r.status_code == 200, f"{r.status_code}: {r.text[:160]}"


def test_qa_regions_infinite_box_is_500(client, synthetic_page):
    """Region boxes are not validated (finite, inside the page): width=Infinity on a blank area reaches the
    whole-block fallback and crop_quad -> int(round(inf)) OverflowError -> 500 instead of 422."""
    pid = _upload(client, synthetic_page).json()["pages"][0]["page_id"]
    body = ('{"page_id": "%s", "regions": [{"type": "paragraph", '
            '"box": {"x": 100, "y": 1500, "width": Infinity, "height": 100}}]}' % pid)
    r = client.post("/regions", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code in (200, 422), f"{r.status_code}: {r.text[:160]}"


# --------------------------------------------------------------------------- DoS: unbounded crops
def test_qa_lines_unclipped_box_allocates_unbounded_crop(monkeypatch, tmp_path):
    """/lines: user line boxes are neither clipped to the page nor size-limited before crop_quad ->
    warpPerspective allocates w*h*3 bytes. A 60000x60000 'line' = 10.8 GB -> OOM-kills the server."""
    from app.services.pipeline_service import PipelineService
    seen = _record_warp(monkeypatch)
    page = tmp_path / "p.png"
    cv2.imwrite(str(page), np.full((1000, 800, 3), 255, np.uint8))
    blocks = [{"type": "paragraph", "box": {"x": 0, "y": 0, "width": 800, "height": 1000},
               "lines": [{"x": 0, "y": 0, "width": 60000, "height": 60000}]}]
    PipelineService(_StubPool()).process_lines(str(page), blocks, str(tmp_path))
    assert seen and max(w * h for w, h in seen) <= 800 * 1000, f"crop sizes requested: {seen}"


def test_qa_regions_fallback_allocates_unbounded_crop(monkeypatch, tmp_path):
    """/regions: a user region with no detected lines is read as one line (_fallback_whole_block) using the
    unclipped user box -> the same unbounded warpPerspective allocation."""
    from app.services.pipeline_service import PipelineService
    seen = _record_warp(monkeypatch)
    page = tmp_path / "p.png"
    cv2.imwrite(str(page), np.full((1000, 800, 3), 255, np.uint8))
    regions = [{"type": "paragraph", "box": {"x": 0, "y": 0, "width": 60000, "height": 60000}}]
    PipelineService(_StubPool()).process_regions(str(page), regions, str(tmp_path))
    assert seen and max(w * h for w, h in seen) <= 800 * 1000, f"crop sizes requested: {seen}"


# --------------------------------------------------------------------------- reading order
def test_qa_xy_cut_two_columns_with_aligned_paragraph_gap():
    """2-column page whose paragraph breaks line up across the columns: the horizontal cut is tried first and
    wins, so the order becomes L1, R1, L2, R2 instead of L1, L2, R1, R2 (same algorithm as the reference)."""
    W, H = 1240, 1754
    blocks = [
        {"id": "title", "box": [100, 80, 1140, 160]},
        {"id": "L1", "box": [100, 200, 600, 700]}, {"id": "L2", "box": [100, 730, 600, 1600]},
        {"id": "R1", "box": [640, 200, 1140, 700]}, {"id": "R2", "box": [640, 730, 1140, 1600]},
    ]
    order = [b["id"] for b in xy_cut(blocks, W, H)]
    assert order == ["title", "L1", "L2", "R1", "R2"], order


# --------------------------------------------------------------------------- model switching
def test_qa_response_models_not_snapshotted(client, synthetic_page, monkeypatch):
    """The response's "models" field is read from pool.active AFTER processing, so a POST /models that lands while
    a page is being processed makes the response claim a model that did not produce its text."""
    from app.main import app
    pool = app.state.models
    used = pool.active["rec"]
    other = next(m["id"] for m in pool.info()["models"] if m["stage_key"] == "rec" and m["id"] != used)
    real = pool.recognizer
    orig = real.recognize

    def recognize_then_switch(crops):
        out = orig(crops)
        pool.switch("rec", other)  # what a concurrent POST /models does
        return out

    monkeypatch.setattr(real, "recognize", recognize_then_switch)
    try:
        page = _upload(client, synthetic_page).json()["pages"][0]
    finally:
        monkeypatch.undo()
        pool.switch("rec", used)
    assert page["models"]["rec"] == used, f"text produced by {used} but response says {page['models']['rec']}"


# --------------------------------------------------------------------------- concurrency (expected to PASS)
def test_qa_concurrent_requests_are_consistent(client, synthetic_page):
    """Shared onnxruntime sessions under 4 concurrent /process calls give the same text as a sequential call."""
    ref = "\n".join(b["text"] for b in _upload(client, synthetic_page).json()["pages"][0]["blocks"])
    out, errs = [], []

    def go():
        try:
            r = _upload(client, synthetic_page)
            out.append("\n".join(b["text"] for b in r.json()["pages"][0]["blocks"]))
        except Exception as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(out) == 4 and all(o == ref for o in out)
