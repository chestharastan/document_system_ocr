"""Model-free unit tests: taxonomy, NMS, reading order, DB post-processing, post-corrector."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from app import taxonomy
from app.services.reading_order import sort_lines, xy_cut
from app.services.stages.layout_yolo_onnx import letterbox, nms, parse_names
from app.services.stages.postcorrect import IdentityPostCorrector, build_postcorrector
from app.services.stages.textdet_db_onnx import boxes_from_prob


def test_taxonomy_matches_training_copy():
    src = Path(__file__).resolve().parents[2] / "training" / "taxonomy.py"
    if not src.exists():
        pytest.skip("training/taxonomy.py not present")
    spec = importlib.util.spec_from_file_location("training_taxonomy", src)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    assert taxonomy.CLASSES == t.CLASSES
    assert taxonomy.TEXT_CLASSES == t.TEXT_CLASSES
    assert taxonomy.COLORS == t.COLORS


def test_resolve_v1_names():
    assert taxonomy.resolve_class("text") == "text"
    assert taxonomy.resolve_class("header_footer", [0, 10, 100, 40], 1000, 1400) == "header"
    assert taxonomy.resolve_class("header_footer", [0, 1300, 100, 1340], 1000, 1400) == "footer"
    assert taxonomy.resolve_class("qr_barcode", [0, 0, 100, 100]) == "qr_code"
    assert taxonomy.resolve_class("qr_barcode", [0, 0, 300, 60]) == "barcode"
    assert taxonomy.resolve_class("plain text") == "text"
    assert taxonomy.legacy_type("signature") == "image"
    assert taxonomy.legacy_type("section_header") == "title"
    assert taxonomy.class_from_request("paragraph") == "text"
    assert taxonomy.class_from_request("stamp") == "stamp"


def test_parse_names_and_letterbox():
    assert parse_names({"names": "{0: 'text', 1: 'title'}"}) == ["text", "title"]
    img, r, (px, py) = letterbox(np.zeros((500, 1000, 3), np.uint8), (1024, 1024))
    assert img.shape == (1024, 1024, 3) and abs(r - 1.024) < 1e-6 and px == 0 and py == 256


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_xy_cut_two_columns():
    items = [{"box": [0, 0, 400, 50], "n": "title"},
             {"box": [0, 100, 180, 500], "n": "left"}, {"box": [220, 100, 400, 300], "n": "right_top"},
             {"box": [220, 320, 400, 500], "n": "right_bottom"}]
    assert [i["n"] for i in xy_cut(items[::-1], 400, 500)] == ["title", "left", "right_top", "right_bottom"]


def test_sort_lines_rows():
    lines = [{"bbox": [200, 0, 300, 20], "n": 2}, {"bbox": [0, 2, 100, 22], "n": 1}, {"bbox": [0, 40, 100, 60], "n": 3}]
    assert [l["n"] for l in sort_lines(lines)] == [1, 2, 3]


def test_db_postprocess_finds_a_bar():
    prob = np.zeros((128, 256), np.float32)
    prob[50:62, 20:220] = 0.9
    out = boxes_from_prob(prob, 256, 512)
    assert len(out) == 1
    q, s = out[0]
    assert s > 0.8 and q[:, 0].min() < 45 and q[:, 0].max() > 435  # scaled x2 and unclipped


def test_postcorrect_identity():
    pc = build_postcorrector("none")
    assert isinstance(pc, IdentityPostCorrector)
    with pytest.raises(ValueError):
        build_postcorrector("nope")


def test_page_sized_figure_and_duplicates_are_ignored():
    from app.services.pipeline_service import PipelineService
    from app.services.stages.base import Region
    ps = PipelineService(pool=None)
    regs = [Region("figure", 0.9, [0, 0, 1000, 1300]), Region("text", 0.8, [100, 100, 500, 200]),
            Region("list", 0.5, [102, 101, 500, 200])]
    blocks, ignored = ps._clean_regions(regs, 1000, 1400)
    assert [b["cls"] for b in blocks] == ["text"]
    assert sorted(i["reason"] for i in ignored) == ["duplicate", "page_sized_figure"]
