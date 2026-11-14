"""Bullets, tab stops and paragraphs (app/services/doc_structure.py) on synthetic pages - no models needed."""

from __future__ import annotations

import cv2
import numpy as np

from app.services import doc_structure as D

H_LINE = 20  # line height of the synthetic text lines


def _page(w=600, h=400):
    return np.full((h, w, 3), 255, np.uint8)


def _text_line(img, x0, y0, width, h=H_LINE):
    """Fake printed text: dense dark strokes with gaps (what the ink mask of a Khmer line looks like)."""
    rng = np.random.default_rng(x0 + y0)
    for x in range(x0, x0 + width, 4):
        cv2.line(img, (x, y0 + 3 + int(rng.integers(0, 3))), (x + 1, y0 + h - 3), (20, 20, 20), 1)
    return {"bbox": [float(x0), float(y0), float(x0 + width), float(y0 + h)], "text": "x" * (width // 8), "score": 1.0}


def _block(lines, cls="text"):
    xs0 = min(l["bbox"][0] for l in lines); ys0 = min(l["bbox"][1] for l in lines)
    xs1 = max(l["bbox"][2] for l in lines); ys1 = max(l["bbox"][3] for l in lines)
    return {"cls": cls, "box": [xs0, ys0, xs1, ys1], "score": 1.0, "label": cls, "source": "layout", "lines": lines}


# ------------------------------------------------------------------------------------------------ bullets

def test_find_bullets_ring_and_dash():
    img = _page()
    plain = _text_line(img, 60, 40, 400)
    ring = _text_line(img, 90, 80, 380)
    cv2.circle(img, (76, 90), 3, (20, 20, 20), 1)                    # "o" bullet 0.35 h, 11 px gap
    dash = _text_line(img, 90, 120, 380)
    cv2.rectangle(img, (72, 129), (78, 130), (20, 20, 20), -1)         # "-" bullet 7x2 px
    b = _block([plain, ring, dash])
    assert D.find_bullets(img, [b]) == 2
    assert "bullet" not in plain
    assert ring["bullet"] == "•" and dash["bullet"] == "-"
    assert ring["bullet_box"][2] < 90 and ring["bullet_box"][0] >= 72
    assert dash["bullet_box"][3] - dash["bullet_box"][1] <= 3


def test_find_bullets_ignores_rules_and_neighbours():
    img = _page()
    # a notebook rule running from the margin under the text: not a bullet
    ruled = _text_line(img, 120, 40, 400)
    cv2.line(img, (30, 52), (560, 52), (120, 120, 120), 2)
    # the last word of a left column (a detected line) sits in the window of the right column's line
    right = _text_line(img, 300, 100, 250)
    left_word = _text_line(img, 240, 100, 40)
    # a tall vertical stroke (margin line fragment) left of a line
    tall = _text_line(img, 120, 160, 400)
    cv2.line(img, (100, 160), (100, 178), (20, 20, 20), 1)
    # faint bleed-through dot
    faint = _text_line(img, 120, 220, 400)
    cv2.circle(img, (104, 230), 3, (215, 215, 215), -1)
    blocks = [_block([ruled, tall, faint]), _block([right]), _block([left_word])]
    assert D.find_bullets(img, blocks) == 0
    assert not any(ln.get("bullet") for b in blocks for ln in b["lines"])


def test_find_bullets_skips_tables_and_tiny_lines():
    img = _page()
    cell = _text_line(img, 90, 40, 200)
    cv2.circle(img, (76, 50), 3, (20, 20, 20), 1)
    assert D.find_bullets(img, [_block([cell], cls="table")]) == 0
    tiny = {"bbox": [90.0, 80.0, 100.0, 100.0], "text": "", "score": 1.0}  # a bullet-sized box is not a text line
    cv2.circle(img, (76, 90), 3, (20, 20, 20), 1)
    assert D.find_bullets(img, [_block([tiny])]) == 0


def test_merge_bullet_lines_attaches_detected_marks():
    mark = {"bbox": [100.0, 102.0, 108.0, 110.0], "text": "•", "score": 1.0}
    dash = {"bbox": [100.0, 142.0, 108.0, 146.0], "text": "-", "score": 1.0}
    item = {"bbox": [120.0, 100.0, 400.0, 118.0], "text": "item", "score": 1.0}
    item2 = {"bbox": [120.0, 135.0, 400.0, 153.0], "text": "item2", "score": 1.0}
    far = {"bbox": [100.0, 302.0, 108.0, 310.0], "text": "•", "score": 1.0}  # nothing on its row: stays a line
    out = D.merge_bullet_lines([mark, item, dash, item2, far])
    assert [ln["text"] for ln in out] == ["item", "item2", "•"]
    assert item["bullet"] == "•" and item["bullet_box"] == [100.0, 102.0, 108.0, 110.0]
    assert item2["bullet"] == "-"


def test_bullet_glyph_and_strip():
    assert D.is_bullet_glyph("o", {"bbox": [0, 0, 8, 8]})
    assert not D.is_bullet_glyph("o", {"bbox": [0, 0, 30, 8]})  # a wide box reading "o" is text
    assert not D.is_bullet_glyph("ok", {"bbox": [0, 0, 8, 8]})
    ln = {"bullet": "-", "text": "- word"}
    D.strip_leading_bullet(ln)
    assert ln["text"] == "word"
    ln = {"bullet": None, "text": "- word"}
    D.strip_leading_bullet(ln)
    assert ln["text"] == "- word"


# ------------------------------------------------------------------------------------------------ structure

def _ln(x0, y0, x1, text="t", bullet=None):
    ln = {"bbox": [float(x0), float(y0), float(x1), float(y0 + H_LINE)], "text": text, "score": 1.0}
    if bullet:
        ln["bullet"] = bullet
    return ln


def test_structure_khmer_first_line_indent_and_lists():
    """margin 70; Khmer paragraph indented first line at 100; dash items at 100 with hanging text; ring sub-items
    at 130; a heading back at the margin."""
    heading = _block([_ln(70, 10, 300, "A)")])
    para = _block([_ln(100, 40, 560, "p1"), _ln(70, 64, 560, "p2"), _ln(70, 88, 400, "p3")])
    items = _block([_ln(100, 120, 560, "i1", "-"), _ln(100, 144, 560, "i1b"), _ln(100, 168, 300, "i2", "-"),
                    _ln(130, 192, 400, "s1", "•"), _ln(130, 216, 400, "s2", "•")])
    para2 = _block([_ln(100, 250, 560, "q1"), _ln(70, 274, 560, "q2")])
    blocks = [heading, para, items, para2]
    D.structure_blocks(blocks, {"text"})
    levels = {ln["text"]: ln["indent_level"] for b in blocks for ln in b["lines"]}
    assert levels == {"A)": 0, "p1": 1, "p2": 0, "p3": 0, "i1": 1, "i1b": 1, "i2": 1, "s1": 2, "s2": 2, "q1": 1, "q2": 0}
    assert [(p["indent_level"], p["bullet"], len(p["lines"])) for p in para["paras"]] == [(1, None, 3)]
    assert [(p["indent_level"], p["bullet"], len(p["lines"])) for p in items["paras"]] == \
        [(1, "-", 2), (1, "-", 1), (2, "•", 1), (2, "•", 1)]
    assert [(p["indent_level"], len(p["lines"])) for p in para2["paras"]] == [(1, 2)]
    assert heading["lines"][0]["tab"] is False and para["lines"][0]["tab"] is True
    assert para["lines"][0]["indent"] == 30


def test_structure_short_line_and_gap_start_paragraphs():
    b = _block([_ln(70, 10, 300, "ends early"), _ln(70, 34, 560, "new para"), _ln(70, 58, 560, "cont"),
                _ln(70, 120, 560, "after gap")])
    D.structure_blocks([b], {"text"})
    assert [[ln["text"] for ln in p["lines"]] for p in b["paras"]] == [["ends early"], ["new para", "cont"], ["after gap"]]
    assert all(ln["indent_level"] == 0 for ln in b["lines"])


def test_structure_two_columns_have_their_own_margins():
    left = _block([_ln(50, 10, 280, "l1"), _ln(50, 34, 280, "l2")])
    right = _block([_ln(320, 10, 560, "r1"), _ln(350, 34, 560, "r2")])  # r2 indented within its column
    D.structure_blocks([left, right], {"text"})
    assert [ln["indent_level"] for ln in left["lines"]] == [0, 0]
    assert [ln["indent_level"] for ln in right["lines"]] == [0, 1]


def test_structure_sorts_rows_and_skips_tables():
    b = _block([_ln(70, 40, 300, "second"), _ln(70, 10, 300, "first")])
    t = _block([_ln(400, 40, 500, "c2"), _ln(400, 10, 500, "c1")], cls="table")
    D.structure_blocks([b, t], {"text", "table"})
    assert [ln["text"] for ln in b["lines"]] == ["first", "second"]
    assert [p["lines"][0]["text"] for p in t["paras"]] == ["c1", "c2"]  # one paragraph per row, level 0
    assert all(ln["indent_level"] == 0 for ln in t["lines"])


def test_text_rendering():
    lines = [{"text": "ក", "indent_level": 1, "bullet": "-"}, {"text": "ខ", "indent_level": 1},
             {"text": "", "indent_level": 0}, {"text": "A", "indent_level": 2, "bullet": "•"}]
    assert D.block_text(lines) == "\t- ក\n\tខ\n\t\t• A"
    assert D.join_wrapped("ភាសា", "ខ្មែរ") == "ភាសាខ្មែរ"
    assert D.join_wrapped("Khmer", "text") == "Khmer text"
    assert D.join_wrapped("ភាសា", "RELC") == "ភាសា RELC"
    assert D.paragraph_text([{"text": "ក"}, {"text": "ខ"}, {"text": "2027"}]) == "កខ 2027"
