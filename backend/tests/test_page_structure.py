"""Decorations, crop-box snapping, position relabelling, font metrics and page layout (app/services/page_structure.py)."""

from __future__ import annotations

import cv2
import numpy as np

from app.services import page_structure as P
from app.services.doc_structure import ink_mask
from app.taxonomy import TEXT_CLASSES

W, H = 800, 1100


def _page():
    return np.full((H, W, 3), 255, np.uint8)


def _text_line(img, x0, y0, width, h=24, text="ក" * 10, conf=0.98):
    rng = np.random.default_rng(x0 * 7 + y0)
    for x in range(x0, x0 + width, 4):
        top = y0 + 6 + int(rng.integers(0, 3))
        cv2.line(img, (x, top), (x + 1, y0 + h - 5), (20, 20, 20), 1)
        if rng.random() < 0.3:  # a vowel sign above / a subscript below
            cv2.line(img, (x, y0 + 1), (x, y0 + 4), (20, 20, 20), 1)
    return {"bbox": [float(x0), float(y0), float(x0 + width), float(y0 + h)], "text": text, "confidence": conf, "score": 1.0}


def _block(cls, lines, box=None, score=0.9):
    if box is None:
        box = [min(l["bbox"][0] for l in lines), min(l["bbox"][1] for l in lines),
               max(l["bbox"][2] for l in lines), max(l["bbox"][3] for l in lines)]
    return {"cls": cls, "box": [float(v) for v in box], "score": score, "label": cls, "source": "layout", "lines": lines}


def _factory(cls, box, score=1.0, label="", source="layout"):
    return {"cls": cls, "box": [float(v) for v in box], "score": float(score), "label": label or cls, "source": source, "lines": []}


def _ornament(img, cx, y, half=60):
    """A mirror-symmetric divider: thin rule with a few blobs, symmetric about cx."""
    cv2.line(img, (cx - half, y), (cx + half, y), (40, 40, 40), 1)
    for dx, r in ((0, 5), (18, 3), (34, 2)):
        cv2.circle(img, (cx - dx, y), r, (20, 20, 20), -1)
        cv2.circle(img, (cx + dx, y), r, (20, 20, 20), -1)


# ------------------------------------------------------------------------------------------------ decorations

def test_ornament_becomes_decoration_and_text_stays():
    img = _page()
    body = [_text_line(img, 100, 300 + i * 36, 600) for i in range(4)]
    _ornament(img, W // 2, 200)
    orn = {"bbox": [W / 2 - 66.0, 193.0, W / 2 + 66.0, 207.0], "text": ".w12e7", "confidence": 0.45, "score": 0.5}
    blocks = [_block("text", body), _block("text", [orn])]
    ink = ink_mask(img)
    out = P.find_decorations(ink, blocks, TEXT_CLASSES, _factory)
    kinds = sorted(b["cls"] for b in out)
    assert kinds == ["decoration", "text"]
    assert len([b for b in out if b["cls"] == "text"][0]["lines"]) == 4


def test_low_confidence_text_line_is_not_a_decoration():
    img = _page()
    ln = _text_line(img, 100, 300, 600, conf=0.4)  # badly read but a real line: asymmetric ink, full height
    body = [_text_line(img, 100, 340 + i * 36, 600) for i in range(3)]
    ink = ink_mask(img)
    assert not P.is_decoration(ink, ln, 24.0, W, H)
    for l in body:
        assert not P.is_decoration(ink, l, 24.0, W, H)


# ------------------------------------------------------------------------------------------------ crop boxes

def test_snap_crop_box_covers_the_cut_emblem():
    img = _page()
    cv2.circle(img, (300, 300), 100, (30, 60, 200), 6)  # ring 200..400
    cv2.circle(img, (300, 300), 60, (30, 60, 200), -1)
    _text_line(img, 100, 430, 600)  # a text line just under it must not be swallowed
    ink = ink_mask(img)
    nb = P.snap_crop_box(ink, [215.0, 212.0, 385.0, 392.0], W, H)  # box clips the ring on every side
    assert nb[0] <= 196 and nb[1] <= 196 and nb[2] >= 404 and nb[3] >= 404
    assert nb[3] < 430  # not grown into the text below


def test_relabel_logo_and_positions():
    lines = [_text_line(_page(), 150, 500 + i * 36, 500) for i in range(5)]
    body = _block("text", lines)
    motto = _block("text", [_text_line(_page(), 500, 60, 220), _text_line(_page(), 480, 95, 260)])
    letterhead = _block("text", [_text_line(_page(), 80, 250, 300)])
    title = _block("text", [_text_line(_page(), 320, 440, 160)])  # centred, just above the body
    emblem = _block("figure", [], box=[120, 100, 300, 290])
    pageno = _block("text", [_text_line(_page(), 385, 1010, 30, text="១/២")])
    footer = _block("text", [_text_line(_page(), 80, 1040, 500, text="Tel 012"), _text_line(_page(), 80, 1068, 500, text="x")])
    first_item = _block("text", [_text_line(_page(), 150, 464, 500)])  # directly above the body: NOT a header
    blocks = [body, motto, letterhead, title, emblem, pageno, footer]
    P.relabel_logos(blocks, W, H)
    P.relabel_by_position(blocks, W, H)
    assert emblem["cls"] == "logo" and emblem["relabel"] == "figure"
    assert motto["cls"] == "header" and letterhead["cls"] == "header"
    assert title["cls"] == "title"
    assert pageno["cls"] == "page_number" and footer["cls"] == "footer"
    assert body["cls"] == "text"
    blocks2 = [body, first_item]
    P.relabel_by_position(blocks2, W, H)
    assert first_item["cls"] == "text"


# ------------------------------------------------------------------------------------------------ fonts + layout

def test_measure_fonts_one_size_per_block_and_page_levels():
    img = _page()
    a = _block("text", [_text_line(img, 100, 300 + i * 36, 600) for i in range(4)])
    b = _block("text", [_text_line(img, 100, 500 + i * 36, 600) for i in range(3)])
    big = _block("text", [_text_line(img, 200, 100, 400, h=48)])
    blocks = [a, b, big]
    P.measure_fonts(ink_mask(img), blocks, TEXT_CLASSES)
    assert a["font_px"] == b["font_px"]  # same print size -> same level
    assert big["font_px"] > 1.5 * a["font_px"]
    for ln in a["lines"]:
        assert 10 <= ln["core_h"] <= 20 and ln["bbox"][1] < ln["baseline"] <= ln["bbox"][3]


def test_page_layout_margins_zones_align():
    img = _page()
    body = _block("text", [_text_line(img, 100, 300 + i * 36, 600) for i in range(4)])
    centred = _block("title", [_text_line(img, 330, 250, 140)])
    right = _block("text", [_text_line(img, 500, 1000, 200)])
    head = _block("header", [_text_line(img, 100, 60, 300)])
    foot = _block("footer", [_text_line(img, 100, 1050, 300)])
    blocks = [body, centred, right, head, foot]
    lay = P.page_layout(blocks, W, H, TEXT_CLASSES)
    m = lay["margins"]
    assert 95 <= m["left"] <= 105 and 95 <= m["right"] <= 105
    assert lay["zones"]["header"] == [60.0, 84.0] and lay["zones"]["footer"][0] == 1050.0
    assert body["align"] == "left" and centred["align"] == "center" and right["align"] == "right"


def test_text_colors_blue_header_black_body_and_spans():
    img = _page()
    blue = _block("header", [_text_line(img, 100, 60, 300)])
    for l in blue["lines"]:  # recolour the header strokes blue (BGR)
        x0, y0, x1, y1 = (int(v) for v in l["bbox"])
        reg = img[y0:y1, x0:x1]
        reg[(reg.min(axis=2) < 60)] = (160, 60, 20)
    body = _block("text", [_text_line(img, 100, 300 + i * 36, 600) for i in range(3)])
    body["lines"][0]["spans"] = [{"text": "a", "x0": 100.0, "x1": 300.0, "bold": True}, {"text": "b", "x0": 320.0, "x1": 700.0, "bold": False}]
    blocks = [blue, body]
    P.text_colors(img, ink_mask(img), blocks, TEXT_CLASSES)
    r, g, b = (int(blue["text_color"][i:i + 2], 16) for i in (1, 3, 5))
    assert b > r + 60  # blue
    assert body["text_color"] == "#1a1a1a" and body["lines"][1]["text_color"] == "#1a1a1a"
    assert all(sp["text_color"] == "#1a1a1a" for sp in body["lines"][0]["spans"])


def _blk(cls, box, lines=()):
    return {"cls": cls, "box": [float(v) for v in box], "score": 0.9, "label": cls, "source": "layout",
            "lines": [{"bbox": [float(v) for v in l], "quad": None, "score": 0.9} for l in lines]}


def test_split_figures_releases_text_between_two_logos_and_splits_the_pictures():
    """A loose figure box around an emblem, a second small logo and the letterhead text between them: the text leaves,
    the box shrinks to the emblem, the small logo becomes a block of its own (press release letterhead)."""
    img = np.full((500, 700), 255, np.uint8)
    cv2.circle(img, (330, 230), 90, 0, -1)                     # emblem
    cv2.rectangle(img, (120, 420), (250, 460), 0, -1)         # small logo bottom-left
    for y in (350, 380):                                       # two text lines under the emblem
        cv2.putText(img, "ROYAL GOVERNMENT OF CAMBODIA", (130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    ink = ink_mask(img)
    fig = _blk("figure", (90, 130, 600, 480), lines=[(128, 332, 530, 356), (128, 362, 530, 386)])
    blocks = [fig]
    released = P.split_figures(ink, blocks)
    assert len(released) == 2
    assert fig["cls"] == "figure" and not fig["lines"] and fig.get("split_from")
    x0, y0, x1, y1 = fig["box"]
    assert 220 <= x0 <= 245 and 125 <= y0 <= 145 and 415 <= x1 <= 440 and 315 <= y1 <= 335   # the emblem only
    assert len(blocks) == 2 and blocks[1]["cls"] == "figure" and blocks[1]["source"] == "split"
    bx = blocks[1]["box"]
    assert bx[0] <= 120 and bx[2] >= 250 and bx[1] <= 420 and bx[3] >= 460


def test_split_figures_logo_gives_back_the_title_under_it_and_drops_duplicates():
    img = np.full((400, 600), 255, np.uint8)
    cv2.circle(img, (300, 100), 60, 0, -1)                     # emblem
    cv2.putText(img, "ROYAL DECREE", (210, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    ink = ink_mask(img)
    logo = _blk("logo", (200, 30, 400, 240), lines=[(208, 198, 392, 226)])
    exact = _blk("logo", (238, 38, 362, 162))                  # the same emblem, tight, from the layout model too
    blocks = [logo, exact]
    released = P.split_figures(ink, blocks)
    assert len(released) == 1 and released[0]["bbox"][1] >= 190
    assert blocks == [exact]                                   # the loose duplicate is gone


def test_split_figures_keeps_chart_text():
    img = np.full((300, 300), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (260, 260), 0, 3)
    cv2.putText(img, "12%", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    ink = ink_mask(img)
    fig = _blk("figure", (30, 30, 270, 270), lines=[(118, 128, 180, 156)])
    blocks = [fig]
    assert P.split_figures(ink, blocks) == [] and len(fig["lines"]) == 1 and len(blocks) == 1


def test_find_rules_page_line_between_body_and_footer_but_not_ruled_paper():
    img = np.full((600, 500), 255, np.uint8)
    cv2.line(img, (20, 480), (480, 480), 0, 2)
    ink = ink_mask(img)
    body = _blk("text", (40, 100, 460, 400), lines=[(40, 100, 460, 120)])
    foot = _blk("footer", (40, 520, 300, 560), lines=[(40, 520, 300, 540)])
    made = P.find_rules(ink, [body, foot], TEXT_CLASSES, lambda cls, box, score, label, source: {"cls": cls, "box": box, "source": source})
    assert len(made) == 1 and made[0]["cls"] == "decoration" and abs(made[0]["box"][1] - 478) <= 2
    for y in range(60, 600, 40):                               # ruled paper: many lines -> none
        cv2.line(img, (0, y), (500, y), 0, 1)
    assert P.find_rules(ink_mask(img), [], TEXT_CLASSES, lambda *a, **k: {}) == []


def test_split_figures_emblem_touching_the_title_is_still_a_picture():
    img = np.full((400, 600), 255, np.uint8)
    cv2.circle(img, (300, 110), 70, 0, -1)                     # emblem, bottom at y=180
    cv2.putText(img, "ROYAL DECREE", (210, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 3)   # title top ~ y=185
    cv2.line(img, (300, 180), (300, 190), 0, 3)                # a ribbon of the emblem touches the title
    ink = ink_mask(img)
    logo = _blk("logo", (200, 20, 400, 240), lines=[(205, 183, 395, 212)])
    blocks = [logo]
    released = P.split_figures(ink, blocks)
    assert len(released) == 1 and logo["cls"] == "logo" and not logo["lines"]
    assert logo["box"][3] <= 200                               # shrunk to the emblem


def test_split_figures_leaves_a_photo_with_text_on_it_alone():
    img = np.full((300, 400, 3), 255, np.uint8)
    img[20:280, 20:380] = (90, 120, 160)                        # a photo
    cv2.putText(img, "CAPTION", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nonpaper = (gray < 235) | (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1] > 40)
    fig = _blk("figure", (10, 10, 390, 290), lines=[(98, 228, 260, 256)])
    blocks = [fig]
    assert P.split_figures(ink_mask(gray), blocks, nonpaper) == [] and len(fig["lines"]) == 1 and len(blocks) == 1


def test_snap_line_boxes_recovers_cut_glyphs_but_not_underlines():
    img = np.full((200, 600), 255, np.uint8)
    cv2.putText(img, "ROYAL DECREE", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)   # ~ x 100..390, y 32..62
    cv2.line(img, (60, 75), (540, 75), 0, 2)                                            # underline far below
    cv2.putText(img, "second line", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    ink = ink_mask(img)
    cut = _blk("title", (100, 30, 370, 64), lines=[(110, 34, 345, 62)])                 # first / last glyph cut through
    other = _blk("text", (100, 100, 320, 135), lines=[(100, 106, 320, 134)])
    n = P.snap_line_boxes(ink, [cut, other], TEXT_CLASSES)
    x0, y0, x1, y1 = cut["lines"][0]["bbox"]
    assert n >= 1 and x1 >= 352 and x0 <= 103          # grew over the cut glyphs
    assert y1 <= 70                                      # the underline did not drag it down
    assert cut["lines"][0]["quad"].shape == (4, 2)


def test_measure_fonts_prefers_model_em_size_and_snaps_levels():
    """Lines carrying the font model's em size give the block its size (font_src 'model'); blocks without it fall back
    to the ink core; sizes within 10 % snap to one page level so paragraphs of one style render alike."""
    from app.services.page_structure import measure_fonts
    ink = np.zeros((400, 600), np.uint8)
    cv2.rectangle(ink, (50, 50), (550, 70), 255, -1)
    cv2.rectangle(ink, (50, 100), (550, 120), 255, -1)
    cv2.rectangle(ink, (50, 200), (550, 240), 255, -1)

    def ln(y0, y1, em=None):
        d = {"bbox": [50.0, float(y0), 550.0, float(y1)], "text": "កខគ"}
        if em:
            d["font"] = {"family": "Khmer OS", "em_px": em}
        return d
    def blk(box, lines):
        return {**_blk("text", box), "lines": lines}
    a = blk((50, 45, 550, 75), [ln(45, 75, 14.2)])
    b = blk((50, 95, 550, 125), [ln(95, 125, 15.1), ln(95, 125, 14.0), ln(95, 125, 40.0)])  # one odd line
    c = blk((50, 195, 550, 245), [ln(195, 245)])  # no model size -> ink core x K
    measure_fonts(ink, [a, b, c], {"text"})
    assert a["font_src"] == "model" and b["font_src"] == "model" and c["font_src"] == "ink"
    assert a["font_px"] == b["font_px"]  # 14.2 and the trimmed 15.1 snap to one level
    assert c["font_px"] > 1.8 * a["font_px"]


def test_structure_model_splits_a_heading_glued_to_a_paragraph():
    """Layout glued 'ឃ) heading' to the last line of the paragraph above; the structure model is sure the second line
    is a section header -> it becomes its own block (own typeface vote, own size), the paragraph keeps its line."""
    from app.services.structure_model import apply_structure
    l1 = {"bbox": [60.0, 500.0, 560.0, 520.0], "text": "អាហារូបករណ៍ តាមរយៈការផ្ញើអ៊ីម៉ែល", "font": {"family": "Khmer OS Siemreap", "probs": np.array([0.9, 0.1])}}
    l2 = {"bbox": [60.0, 530.0, 400.0, 552.0], "text": "ឃ) កាលបរិច្ឆេទ", "font": {"family": "Khmer OS Muol Light", "probs": np.array([0.05, 0.95])}}
    b = {**_blk("text", (60, 500, 560, 552)), "lines": [l1, l2], "paras": [{"lines": [l1, l2], "bullet": None, "anchor": 60.0, "indent_level": 0}]}
    blocks = [b]
    preds = {id(l1): {"order": 0, "role": "text", "role_p": 1.0, "first_p": 0.1, "level": 0, "first": False, "bullet": False},
             id(l2): {"order": 1, "role": "section_header", "role_p": 1.0, "first_p": 1.0, "level": 0, "first": True, "bullet": False}}
    stats = apply_structure(blocks, preds, {"text", "section_header"})
    assert stats["blocks_split"] == 1 and len(blocks) == 2
    assert blocks[0]["cls"] == "text" and blocks[0]["lines"] == [l1]
    assert blocks[1]["cls"] == "section_header" and blocks[1]["lines"] == [l2] and blocks[1]["source"] == "split_role"
    assert blocks[1]["box"][1] == 530.0 and "font" not in blocks[1]
    # no split when the model is unsure
    l3 = {"bbox": [60.0, 600.0, 560.0, 620.0], "text": "x"}; l4 = {"bbox": [60.0, 630.0, 400.0, 652.0], "text": "y"}
    c = {**_blk("text", (60, 600, 560, 652)), "lines": [l3, l4], "paras": []}
    preds2 = {id(l3): {**preds[id(l1)]}, id(l4): {**preds[id(l2)], "role_p": 0.8}}
    assert apply_structure([c], preds2, {"text", "section_header"})["blocks_split"] == 0


def test_structure_model_line_order_cannot_jump_rows():
    """The model put the footer's top line last; lines of a block are a vertical stack, so the geometric order wins.
    Side-by-side lines on one row may still follow the model."""
    from app.services.structure_model import apply_structure
    l1 = {"bbox": [100.0, 1204.0, 300.0, 1222.0], "text": "address"}
    l2 = {"bbox": [100.0, 1227.0, 300.0, 1245.0], "text": "district"}
    l3 = {"bbox": [100.0, 1249.0, 300.0, 1267.0], "text": "city"}
    b = {**_blk("footer", (100, 1204, 300, 1267)), "lines": [l1, l2, l3], "paras": []}
    pr = lambda o: {"order": o, "role": "footer", "role_p": 1.0, "first_p": 1.0, "level": 0, "first": True, "bullet": False}
    preds = {id(l1): pr(26), id(l2): pr(22), id(l3): pr(24)}
    stats = apply_structure([b], preds, {"footer"})
    assert [l["text"] for l in b["lines"]] == ["address", "district", "city"] and stats["line_order_kept"] == 1
    # two lines on one row: the model's order (right one first) is accepted
    r1 = {"bbox": [100.0, 50.0, 200.0, 70.0], "text": "left"}; r2 = {"bbox": [300.0, 51.0, 400.0, 71.0], "text": "right"}
    c = {**_blk("text", (100, 50, 400, 71)), "lines": [r1, r2], "paras": []}
    apply_structure([c], {id(r1): pr(2), id(r2): pr(1)}, {"text"})
    assert [l["text"] for l in c["lines"]] == ["right", "left"]
