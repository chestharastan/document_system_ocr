"""Phone-photo fixes (2026-11-14, user's folded contract photo 9f6c6554...): background figure, tangled blocks,
reading-order repair, speck in a justified gap."""
import numpy as np

from app.services.pipeline_service import drop_background_figures, _quad
from app.services.page_structure import _between_row_pieces
from app.services.structure_model import _left_beside, _repair_order, order_blocks, tangled_blocks


def _blk(cls, box, lines=()):
    return {"cls": cls, "box": list(map(float, box)), "score": 0.4, "label": cls, "source": "layout",
            "lines": [{"bbox": list(map(float, l))} for l in lines]}


def _page_with_wood(h=1000, w=800):
    img = np.full((h, w, 3), 245, np.uint8)
    img[:150] = (40, 110, 190)  # saturated brown desk strip on top (BGR)
    return img


def test_background_figure_on_desk_dropped():
    img = _page_with_wood()
    lines = [(_quad(100, 200 + 40 * i, 700, 230 + 40 * i), 0.9) for i in range(5)]
    wood = _blk("figure", [0, 0, 780, 140])
    real = _blk("figure", [200, 500, 600, 800])  # a picture on the paper, inside the page
    ignored: list = []
    keep = drop_background_figures(img, [wood, real], lines, ignored)
    assert keep == [real] and ignored[0]["reason"] == "background"


def test_paper_figure_at_border_kept():
    img = np.full((1000, 800, 3), 245, np.uint8)  # a white strip touching the border is paper, not desk
    lines = [(_quad(100, 200 + 40 * i, 700, 230 + 40 * i), 0.9) for i in range(5)]
    fig = _blk("figure", [0, 0, 780, 140])
    assert drop_background_figures(img, [fig], lines, []) == [fig]


def test_repair_order_fields_after_paragraph_above():
    para = _blk("text", [100, 400, 900, 480]); fields = _blk("text", [600, 700, 900, 800]); top = _blk("title", [100, 100, 900, 150])
    assert _repair_order([top, fields, para]) == [top, para, fields]
    left = _blk("text", [100, 500, 400, 600]); right = _blk("text", [600, 500, 900, 600])  # side by side: no constraint
    assert _repair_order([left, right]) == [left, right]


def test_logo_left_of_company_name_first():
    logo = _blk("logo", [200, 244, 380, 446]); name = _blk("header", [388, 228, 900, 360], [[388, 228, 900, 300]])
    assert _left_beside(logo["box"], name["box"])
    name["order_key"] = 0
    assert order_blocks([name, logo], lambda bs: bs)[0] is logo


def test_tangled_blocks():
    big = _blk("list", [250, 1000, 1260, 1210], [[283, 993, 1254, 1030], [285, 1100, 1266, 1140]])
    row = _blk("list", [285, 1065, 1260, 1102], [[285, 1069, 694, 1102], [721, 1065, 1261, 1098]])
    far = _blk("text", [250, 1400, 1260, 1450], [[250, 1400, 1260, 1450]])
    assert tangled_blocks([big, row, far]) == [big, row]


def test_speck_between_row_pieces():
    row = [{"bbox": [285, 1069, 694, 1102]}, {"bbox": [721, 1065, 1261, 1098]}]
    assert _between_row_pieces([691, 1075, 731, 1091], row)
    assert not _between_row_pieces([1300, 1075, 1340, 1091], row)
