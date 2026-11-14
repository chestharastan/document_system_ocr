"""Red stamps by colour, the cleaned page under them, the stamp-only crop, text-filled figures (app/services/stamps.py)."""

from __future__ import annotations

import cv2
import numpy as np

from app.services import stamps as S

W, H = 800, 1000


def _factory(cls, box, score=1.0, label="", source="layout"):
    return {"cls": cls, "box": [float(v) for v in box], "score": float(score), "label": label or cls, "source": source, "lines": []}


def _page_with_stamp(cx=500, cy=700, r=80):
    img = np.full((H, W, 3), 255, np.uint8)
    for y in range(600, 760, 40):  # black text lines, two of them under the stamp
        for x in range(380, 720, 5):
            cv2.line(img, (x, y), (x + 1, y + 18), (20, 20, 20), 1)
    cv2.circle(img, (cx, cy), r, (40, 40, 210), 5)          # red ring (BGR)
    cv2.circle(img, (cx, cy), r - 25, (40, 40, 210), 3)
    cv2.putText(img, "STAMP", (cx - 45, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 210), 2)
    return img


def test_find_stamps_adds_a_block_and_marks_layout_stamps():
    img = _page_with_stamp()
    blocks = [_factory("text", [380, 590, 720, 780])]
    new = S.find_stamps(img, blocks, _factory)
    assert len(new) == 1 and new[0]["cls"] == "stamp" and new[0]["ink"] == "red"
    x0, y0, x1, y1 = new[0]["box"]
    assert x0 <= 420 and y0 <= 620 and x1 >= 580 and y1 >= 780
    # a layout stamp already there: marked red, no second block; a layout logo there: becomes a stamp
    lay = [_factory("stamp", [430, 630, 570, 770])]
    assert S.find_stamps(img, lay, _factory) == [] and lay[0]["ink"] == "red"
    logo = [_factory("logo", [430, 630, 570, 770])]
    S.find_stamps(img, logo, _factory)
    assert logo[0]["cls"] == "stamp" and logo[0]["relabel"] == "logo"


def test_no_stamp_on_a_plain_page_or_for_thin_red_text():
    img = np.full((H, W, 3), 255, np.uint8)
    assert S.find_stamps(img, [], _factory) == []
    cv2.putText(img, "RED HEADING TEXT", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 210), 2)  # wide, not round
    assert S.find_stamps(img, [], _factory) == []


def test_clean_red_keeps_text_and_removes_the_stamp():
    img = _page_with_stamp()
    blocks = S.find_stamps(img, [], _factory)
    clean = S.clean_red(img, blocks)
    assert clean is not None and clean.shape == img.shape
    red = S.red_ink_mask(img) > 0
    assert S.red_ink_mask(clean).sum() == 0                      # no red ink left
    assert clean[red].mean() > 200                               # stamp strokes are paper now
    text = (img.min(axis=2) < 60) & ~red                         # black text pixels
    diff = np.abs(clean[text].astype(int) - img[text].astype(int))
    assert diff.mean() < 3 and diff.max() <= 40  # strokes kept (anti-aliased edges may shift a little)
    assert S.clean_red(img, [_factory("text", [0, 0, 10, 10])]) is None  # no red block -> None


def test_stamp_layer_is_red_on_white():
    img = _page_with_stamp()
    layer = S.stamp_layer(img, [400, 600, 600, 800])
    assert layer.shape == (200, 200, 3)
    dark_before = (img[600:800, 400:600].max(axis=2) < 60).sum()
    assert (layer.max(axis=2) < 60).sum() < 0.1 * dark_before  # the black text is gone (remnants under the strokes)
    assert (S.red_ink_mask(layer) > 0).sum() > 500       # the red strokes are there


def test_figures_full_of_text_become_text():
    def ln(x0, y0, w, h=16):
        return {"bbox": [float(x0), float(y0), float(x0 + w), float(y0 + h)]}
    footer = _factory("figure", [20, 940, 780, 60]); footer["lines"] = [ln(30, 945, 300), ln(30, 970, 250), ln(500, 945, 200)]
    emblem = _factory("figure", [0, 0, 800, 220]); emblem["lines"] = [ln(300, 150, 200), ln(280, 180, 240)]
    picture = _factory("figure", [100, 300, 500, 600]); picture["lines"] = []
    blocks = [footer, emblem, picture]
    assert S.figures_full_of_text(blocks, W, H) == 1
    assert footer["cls"] == "text" and footer["relabel"] == "figure"
    assert emblem["cls"] == "figure" and picture["cls"] == "figure"


def test_unstamp_fades_a_photo_grade_red_ring_but_keeps_black_text():
    """On a photo the stamp's red channel is only mid grey (R ~150): the plain red channel left a grey ring that the
    recogniser read as strokes. The per-stamp stretch sends stamp-only pixels to white and keeps the black text."""
    img = np.full((60, 200, 3), 235, np.uint8)
    cv2.circle(img, (100, 30), 26, (40, 40, 150), 4)           # dark-red ring (BGR), R = 150 only
    cv2.line(img, (10, 30), (190, 30), (20, 20, 20), 3)         # black text stroke crossing the ring
    red = S.red_ink_mask(img)
    g = S._unstamp(img, red)
    ring_only = (red > 0) & ~(np.abs(np.arange(60)[:, None] - 30) <= 2)   # ring pixels away from the stroke
    assert np.median(g[ring_only]) > 230                                 # faded to paper
    assert g[30, 100] < 80 and g[30, 20] < 80                            # the stroke, inside and outside the ring
    assert img[30, 100, 2] < 100                                         # (it really crosses the ring)
