"""Structured document (app/services/document.py): runs, lists, headings, first-line indents, footer columns."""

from __future__ import annotations

from app.services.document import build_document, paragraph_runs

W, H = 1000, 1400


def _line(order, text, level=0, bullet=None, spans=None, conf=0.95, bold=False):
    return {"reading_order": order, "text": text, "indent_level": level, "bullet": bullet, "spans": spans or [], "confidence": conf, "bold": bold}


def _block(bid, cls, order, lines, paras=None, box=(100, 100, 800, 60), align="left", font_px=20.0, bold=False, font=None, crop=None):
    return {"id": bid, "cls": cls, "reading_order": order, "lines": lines, "paragraphs": paras or [],
            "box": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]}, "align": align, "font_px": font_px, "bold": bold,
            "font": font or {"family": "Khmer OS Battambang", "weight": "regular", "slant": "upright", "confidence": 0.9}, "crop": crop, "text_color": "#1a1a1a"}


def test_paragraph_runs_join_khmer_without_space_and_merge_bold():
    lines = [_line(1, "ភាសា ខ្មែរ", spans=[{"text": "ភាសា", "bold": True}, {"text": "ខ្មែរ", "bold": False}]), _line(2, "ជាតិ"), _line(3, "RELC 2027")]
    runs = paragraph_runs(lines)
    assert runs[0] == {"text": "ភាសា", "bold": True}
    assert "".join(r["text"] for r in runs) == "ភាសា ខ្មែរជាតិ RELC 2027"
    assert all(not r["bold"] for r in runs[1:])


def test_build_document_structure():
    title = _block("b1", "title", 1, [_line(1, "សេចក្តីប្រកាសព័ត៌មាន")], [{"lines": [1], "indent_level": 0, "bullet": None}], box=(300, 100, 400, 40), align="center", font_px=28, font={"family": "Khmer OS Muol Light", "weight": "regular", "slant": "upright", "confidence": 0.95})
    heading = _block("b2", "text", 2, [_line(1, "ខ) លក្ខខណ្ឌ", bold=True)], [{"lines": [1], "indent_level": 0, "bullet": None}], box=(100, 160, 200, 30), bold=True)
    para = _block("b3", "text", 3, [_line(1, "ដំណាក់កាលទី១៖ មន្ត្រី", level=1), _line(2, "ពាក្យស្នើសុំ", 0), _line(3, "សមត្ថភាព។", 0)],
                  [{"lines": [1, 2, 3], "indent_level": 1, "bullet": None}], box=(100, 200, 800, 90))
    items = _block("b4", "text", 4, [_line(1, "ពាក្យស្នើសុំ", 2, "•"), _line(2, "លិខិតបញ្ជាក់", 2, "•")],
                   [{"lines": [1], "indent_level": 2, "bullet": "•"}, {"lines": [2], "indent_level": 2, "bullet": "•"}], box=(140, 300, 600, 60))
    items2 = _block("b5", "text", 5, [_line(1, "សញ្ញាបត្រ", 2, "•", conf=0.5)], [{"lines": [1], "indent_level": 2, "bullet": "•"}], box=(140, 362, 600, 30))
    logo = _block("b6", "logo", 6, [], box=(400, 20, 160, 160), align="center", crop={"url": "/storage/crops/x/logo.png", "file": "logo.png"})
    foot_l = _block("b7", "footer", 7, [_line(1, "អាសយដ្ឋាន")], [{"lines": [1], "indent_level": 0, "bullet": None}], box=(100, 1300, 300, 30), font_px=14)
    foot_r = _block("b8", "footer", 8, [_line(1, "Tel 012")], [{"lines": [1], "indent_level": 0, "bullet": None}], box=(600, 1302, 300, 30), align="right", font_px=14)
    doc = build_document([title, heading, para, items, items2, logo, foot_l, foot_r], {"margins": {"left": 100, "right": 100, "top": 20, "bottom": 70}}, W, H)
    types = [e["type"] for e in doc["elements"]]
    assert types == ["heading", "heading", "paragraph", "list", "figure", "footer_row"]
    h1, h3, p, lst, fig, foot = doc["elements"]
    assert h1["heading"] == 1 and h1["align"] == "center" and h1["style"]["heavy"] is True
    assert h3["heading"] == 3  # a short bold line is a heading
    assert p["first_line_indent"] is True and p["level"] == 0 and p["text"] == "ដំណាក់កាលទី១៖ មន្ត្រីពាក្យស្នើសុំសមត្ថភាព។"
    assert len(lst["items"]) == 3 and all(it["bullet"] == "•" and it["level"] == 2 for it in lst["items"]) and lst["items"][2]["low"] is True
    assert fig["crop"].endswith("logo.png") and 0 < fig["width_frac"] <= 1
    assert len(foot["columns"]) == 2 and foot["columns"][1]["align"] == "right" and foot["columns"][0]["elements"][0]["text"] == "អាសយដ្ឋាន"
    assert doc["margins"]["left"] == 100.0


def test_letterhead_columns_become_a_row():
    logo = _block("b1", "logo", 2, [], box=(230, 147, 212, 204), crop={"url": "/storage/crops/x/logo.png", "file": "logo.png"})
    motto = _block("b2", "header", 1, [_line(1, "ព្រះរាជាណាចក្រកម្ពុជា"), _line(2, "ជាតិ សាសនា ព្រះមហាក្សត្រ")],
                   [{"lines": [1], "indent_level": 0, "bullet": None}, {"lines": [2], "indent_level": 0, "bullet": None}], box=(560, 74, 349, 87), align="center")
    orn = _block("b3", "decoration", 3, [], box=(684, 178, 86, 12), crop={"url": "/storage/crops/x/deco.png", "file": "deco.png"})
    name = _block("b4", "header", 4, [_line(1, "រដ្ឋលេខាធិការដ្ឋាន"), _line(2, "លេខៈ ០៩", conf=0.4)],
                  [{"lines": [1], "indent_level": 0, "bullet": None}, {"lines": [2], "indent_level": 0, "bullet": None}], box=(107, 352, 457, 100))
    title = _block("b5", "title", 5, [_line(1, "សេចក្តីប្រកាសព័ត៌មាន")], [{"lines": [1], "indent_level": 0, "bullet": None}], box=(593, 522, 304, 45), align="center")
    body = _block("b6", "text", 6, [_line(1, "អត្ថបទ", 1), _line(2, "បន្ត", 0)], [{"lines": [1, 2], "indent_level": 1, "bullet": None}], box=(213, 587, 1100, 60))
    doc = build_document([motto, logo, orn, name, title, body], {"margins": {"left": 107, "right": 139, "top": 74, "bottom": 20}}, 1448, 2048)
    types = [e["type"] for e in doc["elements"]]
    assert types == ["row", "heading", "paragraph"]
    row = doc["elements"][0]
    assert len(row["columns"]) == 2
    left, right = row["columns"]
    assert [e["type"] for e in left["elements"]] == ["figure", "header", "header"]      # emblem over the ministry name
    assert [e["type"] for e in right["elements"]] == ["header", "header", "decoration"]  # motto over the ornament
    assert left["x0"] < right["x0"]
