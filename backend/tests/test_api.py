"""API smoke tests with the real (v1) models from the manifest on a synthetic page rendered with PIL."""

from __future__ import annotations

from app.taxonomy import CLASSES


def _upload(client, path):
    with open(path, "rb") as f:
        r = client.post("/process", files=[("files", (path.name, f, "image/png"))])
    assert r.status_code == 200, r.text
    return r.json()


def test_models_and_info(client):
    info = client.get("/models").json()
    assert {"device", "warm", "models", "active_model", "active_det_model"} <= set(info)
    stages = {m["stage"] for m in info["models"]}
    assert {"Layout", "Detection", "Recognition"} <= stages
    pi = client.get("/pipeline/info").json()
    assert {"layout", "textdet", "rec", "font", "structure", "postcorrect"} <= set(pi["stages"])  # v3 adds the optional "table" stage
    assert pi["stages"]["postcorrect"]["active"] == "none"
    assert pi["stages"]["layout"]["runtime"]["input_size"] == [1024, 1024]


def test_switch_unknown_model_is_422(client):
    assert client.post("/models", json={"model_id": "nope"}).status_code == 422
    assert client.post("/pipeline/models", json={"stage": "layout", "model_id": "nope"}).status_code == 422


def test_process_synthetic_page(client, synthetic_page):
    data = _upload(client, synthetic_page)
    page = data["pages"][0]
    assert {"page_id", "width", "height", "blocks", "image_url"} <= set(page)
    assert page["width"] == 1240 and page["height"] == 1754
    blocks = page["blocks"]
    assert blocks, "no blocks"
    for b in blocks:
        assert {"id", "type", "label", "reading_order", "box", "lines", "crop", "text"} <= set(b)
        assert b["cls"] in CLASSES
        assert b["type"] in {"title", "paragraph", "header", "caption", "image"}
    text_blocks = [b for b in blocks if b["recognized"]]
    lines = [l for b in text_blocks for l in b["lines"]]
    assert len(lines) >= 6, f"only {len(lines)} lines"
    assert sum(1 for l in lines if l["text"].strip()) >= 6
    assert all(0.0 <= l["confidence"] <= 1.0 for l in lines)
    joined = "\n".join(b["text"] for b in text_blocks)
    assert "ក្រសួង" in joined or "កម្ពុជា" in joined, joined
    assert [b["reading_order"] for b in blocks] == list(range(1, len(blocks) + 1))
    t = page["timings"]
    assert all(k in t for k in ("layout_ms", "textdet_ms", "rec_ms", "postcorrect_ms", "total_ms"))
    # image crops are served
    for b in blocks:
        if b["crop"]:
            assert client.get(b["crop"]["url"]).status_code == 200
    assert client.get(page["image_url"]).status_code == 200


def test_reprocess_regions_lines(client, synthetic_page):
    page = _upload(client, synthetic_page)["pages"][0]
    pid = page["page_id"]
    r = client.post("/reprocess", json={"page_id": pid})
    assert r.status_code == 200 and r.json()["blocks"]

    regions = [{"type": "paragraph", "box": {"x": 120, "y": 310, "width": 1000, "height": 270}},
               {"type": "image", "box": {"x": 150, "y": 1000, "width": 500, "height": 380}}]
    r = client.post("/regions", json={"page_id": pid, "regions": regions})
    assert r.status_code == 200, r.text
    blocks = r.json()["blocks"]
    assert len(blocks) == 2
    para = next(b for b in blocks if b["type"] == "paragraph")
    assert len(para["lines"]) >= 3 and para["text"].strip()
    assert next(b for b in blocks if b["type"] == "image")["crop"]

    lines_payload = [{"type": "text", "box": {"x": 120, "y": 310, "width": 1000, "height": 80},
                      "lines": [{"x": 140, "y": 320, "width": 800, "height": 55}]}]
    r = client.post("/lines", json={"page_id": pid, "blocks": lines_payload})
    assert r.status_code == 200, r.text
    b = r.json()["blocks"][0]
    assert len(b["lines"]) == 1 and b["lines"][0]["text"].strip()

    assert client.post("/reprocess", json={"page_id": "doesnotexist"}).status_code == 422  # malformed id
    assert client.post("/reprocess", json={"page_id": "0" * 32}).status_code == 404     # well-formed, unknown
    assert client.post("/reprocess", json={"page_id": 123}).status_code == 422
    assert client.post("/reprocess", json={"page_id": ["a"]}).status_code == 422


def test_export_docx_from_processed_page(client, synthetic_page):
    import io
    import docx
    with open(synthetic_page, "rb") as fh:
        res = client.post("/process", files={"files": ("page.png", fh, "image/png")})
    assert res.status_code == 200
    page = res.json()["pages"][0]
    out = client.post("/export/docx", json={"pages": [page], "title": "test page"})
    assert out.status_code == 200
    assert out.headers["content-type"].startswith("application/vnd.openxmlformats")
    d = docx.Document(io.BytesIO(out.content))
    texts = [p.text for p in d.paragraphs if p.text.strip()]
    assert texts, "the document has paragraphs"
    assert any(l["text"] in " ".join(texts) for b in page["blocks"] for l in b["lines"] if l["text"])
    bad = client.post("/export/docx", json={"pages": []})
    assert bad.status_code == 422
