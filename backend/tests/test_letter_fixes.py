"""Regressions from a low-resolution Khmer letter (2026-11-13): duplicate stamp/QR pictures, ❖ bullets read as
"*"/"+", wrapped lines of such items split into paragraphs, detector collapse on soft small scans."""
import cv2
import numpy as np

from app.services.doc_structure import split_paragraphs, split_text_marker
from app.services.pipeline_service import absorb_duplicate_figures
from app.services.stages.textdet_db_onnx import ink_coverage


def _blk(cls, box, lines=None):
    return {"cls": cls, "box": list(box), "lines": lines or []}


def test_figure_over_stamp_is_absorbed_into_stamp():
    stamp = _blk("stamp", [358, 334, 527, 515])
    fig = _blk("figure", [353, 404, 543, 538])      # the stamp plus the signature under it
    other = _blk("figure", [56, 418, 143, 506])      # a separate picture stays
    out = absorb_duplicate_figures([stamp, fig, other])
    assert out == [stamp, other]
    assert stamp["box"] == [353, 334, 543, 538]


def test_figure_inside_qr_is_absorbed():
    qr = _blk("qr_code", [203, 421, 282, 498])
    fig = _blk("figure", [218, 428, 282, 498])
    assert absorb_duplicate_figures([qr, fig]) == [qr]


def test_star_and_plus_read_bullets_become_markers():
    for t in ("* វគ្គបណ្តុះបណ្តាល C111", "+ វគ្គបណ្តុះបណ្តាល", "❖ item text"):
        ln = {"text": t, "bbox": [105, 107, 560, 125]}
        assert split_text_marker(ln) and ln["bullet"] and not ln["text"].startswith(("*", "+", "❖"))
        assert 105 < ln["text_x0"] < 160
    for t in ("2+3 = 5", "*note", "+ 5 points"):
        ln = {"text": t, "bbox": [0, 0, 100, 10]}
        assert not split_text_marker(ln)


def test_wrapped_line_of_text_marker_item_stays_in_item():
    first = {"text": "* វគ្គបណ្តុះបណ្តាល C111, C112, MTESOL", "bbox": [105, 107, 560, 125]}
    split_text_marker(first)
    first["text_x0"] = 122.0
    wrap = {"text": "នៅថ្ងៃទី០១ ខែកញ្ញា ឆ្នាំ២០២៦", "bbox": [121, 128, 360, 146]}
    paras = split_paragraphs([first, wrap], tol=9.0, col_right=560)
    assert len(paras) == 1 and paras[0]["bullet"] and len(paras[0]["lines"]) == 2


def test_ink_coverage_flags_missed_text():
    img = np.full((800, 600, 3), 255, np.uint8)
    for y in range(100, 700, 30):
        for x in range(60, 540, 12):
            cv2.rectangle(img, (x, y), (x + 6, y + 10), (0, 0, 0), -1)
    lines = [np.array([[55, y - 3], [545, y - 3], [545, y + 14], [55, y + 14]], np.float32) for y in range(100, 700, 30)]
    assert ink_coverage(img, lines) > 0.95
    assert ink_coverage(img, lines[:3]) < 0.3
    assert ink_coverage(np.full((5000, 2, 3), 255, np.uint8), []) == 1.0
