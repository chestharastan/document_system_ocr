"""Font stage plumbing that needs no model: window preprocessing and the per-block vote."""

from __future__ import annotations

import numpy as np

from app.services.pipeline_service import PipelineService
from app.services.stages.font_cnn_onnx import H, W, windows


def test_windows_pad_short_and_split_long_lines():
    short = np.full((40, 200, 3), 230, np.uint8)
    ws = windows(short)
    assert len(ws) == 1 and ws[0].shape == (H, W)
    assert ws[0][0, 0] == 230  # padded with the border colour
    long = np.full((20, 1500), 200, np.uint8)  # 2400 px wide at height 32 -> 3 windows
    ws = windows(long)
    assert len(ws) == 3 and all(w.shape == (H, W) for w in ws)


def test_block_font_vote_weights_by_line_width():
    def ln(x1, fam_idx, weight="regular", slant="upright"):
        probs = np.zeros(3, np.float32); probs[fam_idx] = 0.9; probs[(fam_idx + 1) % 3] = 0.1
        return {"bbox": [0.0, 0.0, float(x1), 20.0],
                "font": {"family": ["A", "B", "C"][fam_idx], "weight": weight, "slant": slant, "probs": probs}}
    b = {"lines": [ln(900, 0, "bold"), ln(100, 1), ln(100, 1)]}
    PipelineService._block_font(b)
    assert b["font"]["family"] == "A" and b["font"]["weight"] == "bold" and b["font"]["slant"] == "upright"
    assert 0.5 < b["font"]["confidence"] < 0.9
    empty = {"lines": [{"bbox": [0, 0, 10, 10]}]}
    PipelineService._block_font(empty)
    assert "font" not in empty


def test_mark_bold_model_decides_words_with_the_weight_head():
    """Two words in a line; a fake font stage calls the first bold. Word crops are cut from the recognition crop
    (page -> crop pixels through the rec padding); short words need a higher probability."""
    from app.services.bold import mark_bold_model
    ln = {"bbox": [100.0, 50.0, 300.0, 70.0], "text": "ក ខខខ",
          "spans": [{"text": "ក", "x0": 100.0, "x1": 120.0, "bold": False}, {"text": "ខខខ", "x0": 130.0, "x1": 300.0, "bold": True}]}
    crop = np.full((20, 210, 3), 255, np.uint8)  # 200 px + 2 * 0.25 * 20 padding
    seen = []

    def classify(crops):
        seen.extend(c.shape[1] for c in crops)
        return [{"weight": "bold", "weight_confidence": 0.6}, {"weight": "regular", "weight_confidence": 0.95}]
    n = mark_bold_model([(ln, crop)], classify, pad_x=0.25)
    assert n == 2 and len(seen) == 2 and seen[1] > seen[0]
    assert ln["spans"][0]["bold"] is False  # 0.6 < 0.7 for a 1-character word
    assert ln["spans"][1]["bold"] is False and ln["bold"] is False
    ln["spans"][0]["text"] = "កកក"
    mark_bold_model([(ln, crop)], classify, pad_x=0.25)
    assert ln["spans"][0]["bold"] is True and ln["spans"][0]["p_bold"] == 0.6
