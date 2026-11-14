"""Bold words from stroke weight (app/services/bold.py) and the recogniser's character positions."""

from __future__ import annotations

import cv2
import numpy as np

from app.services import bold as B


def _line_crop(words, h=40, thick=(), gap=14, adv=14):
    """A fake line crop: Hershey characters placed ``adv`` px apart, thickness 1 (regular) or 2 (bold)."""
    w = sum(adv * len(t) for t in words) + gap * (len(words) + 1)
    img = np.full((h, w), 240, np.uint8)
    x, pos, text = gap, [], ""
    for wi, t in enumerate(words):
        for ci, ch in enumerate(t):
            cx = x + ci * adv
            cv2.putText(img, ch.upper(), (cx, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 20, 2 if wi in thick else 1, cv2.LINE_AA)
            pos.append((cx + adv / 2) / w)
            text += ch
        x += adv * len(t)
        if wi < len(words) - 1:
            text += " "
            pos.append((x + gap / 2) / w)
            x += gap
    return img, text, pos


def test_word_spans_follow_character_positions():
    img, text, pos = _line_crop(["abc", "de"])
    spans = B.word_spans(text, pos, img.shape[1], img.shape[0])
    assert [s[0] for s in spans] == ["abc", "de"]
    assert spans[0][1] < spans[0][2] <= spans[1][1] < spans[1][2] <= img.shape[1]


def test_bold_words_are_found_relative_to_the_page():
    lines = []
    for thick in [(), (1,), (), (0, 1, 2), ()]:
        img, text, pos = _line_crop(["abcd", "efgh", "ijkl"], thick=thick)
        lines.append(({"text": text, "char_pos": pos, "bbox": [100.0, 50.0, 100.0 + img.shape[1], 90.0]}, img))
    B.measure_lines(lines)
    thr = B.mark_bold([ln for ln, _ in lines], pad_x=0.0)
    assert thr and thr["er"] >= B.MIN_ER
    flags = [[s["bold"] for s in ln["spans"]] for ln, _ in lines]
    assert flags == [[False] * 3, [False, True, False], [False] * 3, [True] * 3, [False] * 3]
    assert [ln["bold"] for ln, _ in lines] == [False, False, False, True, False]
    # spans are in page pixels, left to right inside the line box
    sp = lines[1][0]["spans"]
    assert 100.0 <= sp[0]["x0"] < sp[0]["x1"] <= sp[1]["x0"] < sp[2]["x1"] <= lines[1][0]["bbox"][2] + 1


def test_all_regular_page_has_no_bold_and_missing_positions_are_skipped():
    lines = []
    for _ in range(4):
        img, text, pos = _line_crop(["abcd", "efgh"])
        lines.append(({"text": text, "char_pos": pos, "bbox": [0.0, 0.0, float(img.shape[1]), 40.0]}, img))
    img, text, _ = _line_crop(["xy"])
    lines.append(({"text": text, "char_pos": [], "bbox": [0.0, 0.0, float(img.shape[1]), 40.0]}, img))
    B.measure_lines(lines)
    B.mark_bold([ln for ln, _ in lines], pad_x=0.25)
    assert not any(s["bold"] for ln, _ in lines for s in ln["spans"])
    assert lines[-1][0]["spans"] == [] and lines[-1][0]["bold"] is False
