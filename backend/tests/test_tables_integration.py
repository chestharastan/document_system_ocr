"""integration_qa: v3 tables regressions found end to end (docs/TABLES_CONTRACT.md) - engine status import, LLM soft
failure fallback, doc kind for small sheet tables, grid-table detection (no false tables on ruled paper), deskewed
table structure, sliver bands, xlsx / csv / docx exports of a table page."""

from __future__ import annotations

import io

import cv2
import numpy as np

from app.services import tables as T


# ------------------------------------------------------------------ helpers
def _grid_image(W=1200, H=900, x0=150, y0=200, cols=(0, 200, 420, 640, 860), rows=8, rh=50, angle=0.0):
    img = np.full((H, W), 255, np.uint8)
    x1, y1 = x0 + cols[-1], y0 + rows * rh
    for r in range(rows + 1):
        cv2.line(img, (x0, y0 + r * rh), (x1, y0 + r * rh), 0, 2)
    for c in cols:
        cv2.line(img, (x0 + c, y0), (x0 + c, y1), 0, 2)
    lines = []
    for r in range(rows):
        for ci in range(len(cols) - 1):
            bx0, by0 = x0 + cols[ci] + 15, y0 + r * rh + 15
            bx1, by1 = x0 + cols[ci + 1] - 40, y0 + (r + 1) * rh - 15
            for gx in range(bx0, bx1 - 8, 14):  # "characters": small separate blobs (no long horizontal ink)
                cv2.rectangle(img, (gx, by0 + 3), (gx + 8, by1 - 3), 0, -1)
            lines.append([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])
    box = [x0, y0, x1, y1]
    if angle:
        M = cv2.getRotationMatrix2D(((x0 + x1) / 2, (y0 + y1) / 2), -angle, 1.0)  # leans by +angle (falls to the right)
        img = cv2.warpAffine(img, M, (W, H), borderValue=255)
        lines = [(np.float32(q) @ M[:, :2].T + M[:, 2]).tolist() for q in lines]
        corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ M[:, :2].T + M[:, 2]
        box = [float(corners[:, 0].min()), float(corners[:, 1].min()), float(corners[:, 0].max()), float(corners[:, 1].max())]
    return img, box, lines


def _block(box, quads):
    lines = []
    for i, q in enumerate(quads):
        q = np.float32(q)
        lines.append({"quad": q, "bbox": [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())],
                      "text": f"w{i}", "confidence": 0.99, "score": 1.0})
    return {"cls": "table", "box": [float(v) for v in box], "lines": lines}


# ------------------------------------------------------------------ engine status
def test_pipeline_llm_status_reads_the_provider_module(monkeypatch):
    """GET /pipeline/info reported llm.configured=false for a reachable llama-server: llm_status was imported from
    llm_structure (where it does not exist) and the exception was swallowed."""
    import app.services.llm_providers as P
    from app.services import pipeline_service as S
    monkeypatch.setattr(P, "llm_status", lambda: {"configured": True, "provider": "openai", "model": "m", "reachable": True})
    st = S.llm_status()
    assert st["configured"] is True and st["reachable"] is True and st["model"] == "m"


def test_llm_soft_failure_reports_the_fallback_engine(monkeypatch):
    import app.services.llm_structure as L
    from app.services.pipeline_service import PipelineService

    def soft_fail(res, *a, **k):
        res.setdefault("timings", {})["llm_error"] = "timeout after 120s"
        return res
    monkeypatch.setattr(L, "apply_llm_structure", soft_fail)
    monkeypatch.setenv("LLM_SYNC", "1")   # llm-engine: the default is now a background job (test_llm_structure covers it)
    p = PipelineService.__new__(PipelineService)
    p._impl = {"structure": object()}
    out = p._apply_llm({"blocks": [], "timings": {}, "structure_engine": "llm"})
    assert out["structure_engine"] == "model" and "timeout" in out["llm_error"]


# ------------------------------------------------------------------ doc kind
def _kind_blocks(t_area, text_area, W=1000, H=1000):
    tb = {"cls": "table", "box": [0, 0, W, H * t_area], "table": {"rows": [[0, 1]] * 6, "cols": [[0, 1]] * 4, "confidence": 0.9}}
    blocks = [tb]
    if text_area:
        blocks.append({"cls": "text", "box": [0, H * 0.9, W, H * (0.9 + text_area)], "lines": [{}]})
    return blocks


def test_small_table_alone_on_the_page_is_a_sheet():
    assert T.decide_kind(_kind_blocks(0.3, 0.01), 1000, 1000)["kind"] == "sheet"
    assert T.decide_kind(_kind_blocks(0.3, 0.12), 1000, 1000)["kind"] == "mixed"
    assert T.decide_kind(_kind_blocks(0.12, 0.01), 1000, 1000)["kind"] == "mixed"
    assert T.decide_kind([{"cls": "text", "box": [0, 0, 900, 900], "lines": [{}]}], 1000, 1000)["kind"] == "document"


# ------------------------------------------------------------------ grid detection
def test_find_grid_tables_finds_a_ruled_grid_and_skips_covered_ones():
    img, box, _ = _grid_image()
    ink = (img < 128).astype(np.uint8) * 255
    found = T.find_grid_tables(ink, [])
    assert len(found) == 1
    assert all(abs(a - b) < 15 for a, b in zip(found[0], box))
    assert T.find_grid_tables(ink, [{"cls": "table", "box": box}]) == []


def test_find_grid_tables_ignores_ruled_notebook_paper():
    img = np.full((1400, 1000), 255, np.uint8)
    for y in range(150, 1350, 40):
        cv2.line(img, (40, y), (960, y), 0, 2)
    cv2.line(img, (120, 100), (120, 1380), 0, 2)  # margin line
    cv2.rectangle(img, (300, 20), (700, 90), 0, 2)  # a framed header box
    ink = (img < 128).astype(np.uint8) * 255
    assert T.find_grid_tables(ink, []) == []


# ------------------------------------------------------------------ deskew
def test_tilted_table_keeps_its_rows():
    img, box, quads = _grid_image(angle=6.0)
    blk = _block(box, quads)
    assert 5.0 < T.table_skew(blk) < 7.0
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ink = (img < 128).astype(np.uint8) * 255
    stats = T.apply_tables(bgr, ink, [blk])
    tab = blk["table"]
    assert stats.get("deskewed") == 1 and tab["deskew_deg"] > 5
    assert len(tab["rows"]) == 8 and len(tab["cols"]) == 4, (len(tab["rows"]), len(tab["cols"]))
    # every original line object is in exactly one cell, cell boxes are page px inside the block box (+ margin)
    ids = [id(l) for c in tab["cells"] for l in c["_lines"]]
    assert len(ids) == len(set(ids)) == 32 and {id(l) for l in blk["lines"]} >= set(ids)
    for c in tab["cells"]:
        assert c["box"][0] >= box[0] - 30 and c["box"][2] <= box[2] + 30


def test_straight_table_not_deskewed():
    img, box, quads = _grid_image()
    blk = _block(box, quads)
    assert abs(T.table_skew(blk)) < 0.1
    stats = T.apply_tables(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (img < 128).astype(np.uint8) * 255, [blk])
    assert "deskewed" not in stats and len(blk["table"]["rows"]) == 8


def test_sliver_bands_are_merged():
    bands = [[0, 4], [4, 200], [200, 400], [400, 600], [600, 603]]
    assert T._drop_slivers(bands) == [[0, 200], [200, 400], [400, 603]]
    assert T._drop_slivers([[0, 30], [30, 200], [200, 400]]) == [[0, 30], [30, 200], [200, 400]]


# ------------------------------------------------------------------ exports
def _table_page():
    cells = [{"r": 0, "c": 0, "rs": 1, "cs": 2, "header": True, "text": "Item / Qty", "align": "l", "bold": True, "runs": [{"text": "Item / Qty", "bold": True}], "low": False},
             {"r": 0, "c": 2, "rs": 1, "cs": 1, "header": True, "text": "Price", "align": "r", "bold": True, "runs": [{"text": "Price", "bold": True}], "low": False},
             {"r": 1, "c": 0, "rs": 1, "cs": 1, "header": False, "text": "ទឹក", "align": "l", "bold": False, "runs": [{"text": "ទឹក", "bold": False}], "low": False},
             {"r": 1, "c": 1, "rs": 1, "cs": 1, "header": False, "text": "១២", "align": "r", "bold": False, "runs": [{"text": "១២", "bold": False}], "low": False},
             {"r": 1, "c": 2, "rs": 1, "cs": 1, "header": False, "text": "1,234.50", "align": "r", "bold": False, "runs": [{"text": "1,234.50", "bold": False}], "low": False}]
    bcells = [{**{k: c[k] for k in ("r", "c", "rs", "cs", "header", "text", "align", "bold")}, "box": [100, 100, 200, 140], "lines": [], "confidence": 0.9} for c in cells]
    el = {"type": "table", "rows": 2, "cols": 3, "header_rows": 1, "borders": "grid", "cells": cells, "col_widths": [0.3, 0.3, 0.4],
          "block": "b0", "box": [100, 100, 600, 80], "style": {"font_px": 20}, "low": False}
    blk = {"id": "b0", "cls": "table", "type": "paragraph", "label": "table", "box": {"x": 100, "y": 100, "width": 600, "height": 80},
           "lines": [], "text": "", "table": {"rows": [[100, 140], [140, 180]], "cols": [[100, 300], [300, 500], [500, 700]], "header_rows": 1,
                                              "borders": "grid", "cells": bcells, "source": "rules", "confidence": 0.9}}
    return {"width": 1000, "height": 1400, "blocks": [blk], "layout": {}, "timings": {},
            "document": {"width": 1000, "height": 1400, "margins": {}, "elements": [el]},
            "doc_kind": {"kind": "sheet", "confidence": 0.9, "source": "rules", "tables": 1, "table_area": 0.5, "text_area": 0.0}}


def test_export_xlsx_csv_docx_of_a_table_page():
    import openpyxl
    import docx
    from fastapi.testclient import TestClient
    from app.main import app
    page = _table_page()
    with TestClient(app) as c:
        r = c.post("/export/xlsx", json={"pages": [page], "title": "t", "mode": "auto"})
        assert r.status_code == 200, r.text
        ws = openpyxl.load_workbook(io.BytesIO(r.content)).worksheets[0]
        assert [str(m) for m in ws.merged_cells.ranges] == ["A1:B1"]
        assert ws["A1"].value == "Item / Qty" and ws["A1"].font.b
        assert ws["B2"].value == 12 and ws["C2"].value == 1234.5 and ws["A2"].value == "ទឹក"
        r = c.post("/export/csv", json={"pages": [page]})
        assert r.status_code == 200 and r.content.startswith("﻿".encode())
        assert r.content.decode("utf-8-sig").splitlines()[1] == "ទឹក,១២,\"1,234.50\""
        assert c.post("/export/csv", json={"pages": [page], "table_index": 3}).status_code == 422
        r = c.post("/export/docx", json={"pages": [page], "title": "t"})
        assert r.status_code == 200
        t = docx.Document(io.BytesIO(r.content)).tables[0]
        assert len(t.rows) == 2 and len(t.columns) == 3 and t.cell(1, 2).text == "1,234.50"


def _text_page():
    return {"width": 1000, "height": 1400, "layout": {}, "timings": {}, "document": {"elements": []},
            "doc_kind": {"kind": "document", "confidence": 0.95, "source": "rules", "tables": 0, "table_area": 0.0, "text_area": 0.4},
            "blocks": [
                {"id": "b1", "cls": "text", "reading_order": 1, "box": {"x": 0, "y": 300, "width": 900, "height": 200},
                 "paragraphs": [{"text": "First paragraph."}, {"text": "=1+1 is not a formula"}], "lines": [], "text": ""},
                {"id": "b0", "cls": "title", "reading_order": 0, "box": {"x": 0, "y": 100, "width": 900, "height": 80},
                 "lines": [{"text": "របាយការណ៍"}, {"text": "ប្រចាំឆ្នាំ"}], "text": ""},
                {"id": "b2", "cls": "figure", "reading_order": 2, "box": {"x": 0, "y": 600, "width": 300, "height": 300}, "lines": [], "text": ""}]}


def test_export_xlsx_page_without_tables_holds_its_text():
    import openpyxl
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        r = c.post("/export/xlsx", json={"pages": [_text_page()], "title": "letter"})
        assert r.status_code == 200, r.text
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        ws = wb.worksheets[0]
        assert [ws.cell(row=i, column=1).value for i in range(1, 5)] == ["របាយការណ៍ប្រចាំឆ្នាំ", "First paragraph.", "=1+1 is not a formula", None]
        assert ws["A3"].data_type == "s"
        assert ws["A1"].font.name == "Khmer OS"
        # mixed job: the table page keeps its table sheet, the text page gets a text sheet
        r = c.post("/export/xlsx", json={"pages": [_table_page(), _text_page()], "title": "job"})
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        assert wb.sheetnames == ["P1", "P2"] and wb["P1"]["C2"].value == 1234.5 and wb["P2"]["A2"].value == "First paragraph."
