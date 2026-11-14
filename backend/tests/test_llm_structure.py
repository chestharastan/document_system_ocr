"""LLM structure engine (app/services/llm_structure.py, llm_providers.py, routers/llm.py) with fake providers: no
network is ever touched (a fake provider returns deterministic JSON; httpx uses a MockTransport; the Anthropic client
is a stub)."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services import llm_providers as lp
from app.services import llm_structure as ls
from app.services.llm_providers import AnthropicProvider, LLMError, LLMProvider, LLMTruncated, OpenAICompatProvider, parse_json_text
from app.services.llm_structure import (SCHEMA, apply_llm_structure, page_lines, serialize_page, structure_lines, table_structure,
                                        truncate_text, validate_and_repair)

W, H = 1000, 1400


# ------------------------------------------------------------------------------------------------ fixtures
def _line(order, text, x0, y0, x1, y1, bold=False, bullet=None, conf=0.95):
    return {"box": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}, "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "text": text, "confidence": conf, "score": 0.9, "reading_order": order, "indent_level": 0, "indent": 0, "tab": False,
            "bullet": bullet, "bullet_box": None, "baseline": None, "core_h": None, "font": None, "bold": bold,
            "spans": [{"text": t, "bold": bold} for t in text.split(" ")] if text else [], "text_color": "#1a1a1a"}


def _block(bid, order, cls, lines, box, font_px=20.0, crop=None):
    return {"id": bid, "type": cls, "label": cls, "cls": cls, "cls_id": 0, "display": cls, "color": "#000000", "model_label": cls,
            "score": 0.9, "source": "layout", "reading_order": order, "box": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
            "lines": lines, "crop": crop, "text": "\n".join(l["text"] for l in lines), "recognized": bool(lines), "font_px": font_px,
            "font_src": "ink", "align": "left", "relabel": None, "font": {"family": "Khmer OS Battambang", "weight": "regular", "slant": "upright", "confidence": 0.9},
            "text_color": "#1a1a1a", "bold": bool(lines) and all(l["bold"] for l in lines),
            "paragraphs": [{"indent_level": 0, "bullet": l["bullet"], "text": l["text"], "lines": [l["reading_order"]], "runs": [], "first_line_indent": False} for l in lines]}


def make_result() -> dict:
    """A page: letterhead (2 lines) + logo, a title the layout model called text, a 3-line paragraph, two numbered
    items + a clause heading whose number the rules split off, a 2 x 3 table, a footer, an unrecognised block."""
    header = _block("b1", 1, "header", [_line(1, "ព្រះរាជាណាចក្រកម្ពុជា", 350, 40, 650, 70), _line(2, "ជាតិ សាសនា ព្រះមហាក្សត្រ", 330, 75, 670, 105)], (330, 40, 670, 105))
    logo = _block("b2", 2, "logo", [], (60, 30, 200, 170), crop={"url": "/storage/crops/x/logo.png", "file": "logo.png"})
    title = _block("b3", 3, "text", [_line(1, "សេចក្តីប្រកាសព័ត៌មាន", 300, 220, 700, 260, bold=True)], (300, 220, 700, 260), font_px=28.0)
    para = _block("b4", 4, "text", [_line(1, "ក្រសួងពាណិជ្ជកម្ម បានរៀបចំកិច្ចប្រជុំប្រចាំខែ", 140, 300, 860, 330),
                                    _line(2, "នៅថ្ងៃទី៥ ខែវិច្ឆិកា ឆ្នាំ២០២៤ រាជធានីភ្នំពេញ", 100, 335, 860, 365),
                                    _line(3, "សមិទ្ធិផលនេះឆ្លុះបញ្ចាំងពីការយកចិត្តទុកដាក់ខ្ពស់", 100, 370, 600, 400)], (100, 300, 860, 400))
    para["lines"][0]["indent_level"] = 1
    items = _block("b5", 5, "text", [_line(1, "ពាក្យស្នើសុំ និងឯកសារភ្ជាប់", 140, 430, 700, 460, bullet="១."),
                                     _line(2, "លិខិតបញ្ជាក់ពីអាជ្ញាធរ", 140, 465, 700, 495, bullet="២."),
                                     _line(3, "វិសាលភាព", 100, 520, 300, 550, bold=True, bullet="2.")], (100, 430, 700, 550))
    table = _block("b6", 6, "table", [_line(1, "ល.រ", 100, 800, 160, 825), _line(2, "ឈ្មោះ", 300, 800, 400, 825), _line(3, "ចំនួន", 620, 800, 700, 825),
                                      _line(4, "1", 110, 850, 125, 875), _line(5, "កៅអី", 300, 850, 380, 875), _line(6, "12", 670, 850, 700, 875)], (90, 790, 710, 885))
    footer = _block("b7", 7, "footer", [_line(1, "Tel 012 345 678", 350, 1320, 650, 1345)], (350, 1320, 650, 1345), font_px=14.0)
    empty = _block("b8", 8, "text", [_line(1, "", 100, 1000, 300, 1020)], (100, 1000, 300, 1020))
    blocks = [header, logo, title, para, items, table, footer, empty]
    return {"width": W, "height": H, "blocks": blocks, "layout": {"margins": {"left": 100, "right": 140, "top": 30, "bottom": 55}},
            "document": {"elements": []}, "timings": {"total_ms": 1.0}, "models": {}, "device": "cpu"}


FAKE_ANSWER = {"kind": "mixed", "els": [
    {"t": "header", "lv": 0, "b": "", "l": [1, 2], "c": []},
    {"t": "heading", "lv": 1, "b": "", "l": [3], "c": []},
    {"t": "para", "lv": 0, "b": "", "l": [4, 5, 6, 4, 999], "c": []},          # a duplicate and a bad id
    {"t": "list", "lv": 1, "b": "១.", "l": [7], "c": []},
    {"t": "list", "lv": 1, "b": "២.", "l": [8], "c": []},
    {"t": "heading", "lv": 2, "b": "", "l": [9], "c": []},
    {"t": "table", "lv": 0, "b": "", "l": [], "c": [
        {"r": 0, "c": 0, "rs": 1, "cs": 1, "h": True, "l": [10]}, {"r": 0, "c": 1, "rs": 1, "cs": 1, "h": True, "l": [11]},
        {"r": 0, "c": 2, "rs": 0, "cs": 1, "h": True, "l": [12]},                   # rs 0 -> 1
        {"r": 1, "c": 0, "rs": 1, "cs": 1, "h": False, "l": [13]}, {"r": 1, "c": 1, "rs": 1, "cs": 1, "h": False, "l": [14]},
        {"r": 1, "c": 2, "rs": 1, "cs": 1, "h": False, "l": [15]}]},
    # the footer (16) is forgotten by the model: the repair appends it as a paragraph
]}


class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-1"

    def __init__(self, answers=None, fn=None):
        super().__init__()
        self.answers, self.fn, self.calls = list(answers or []), fn, []

    def complete_json(self, system, user, schema, max_tokens):
        self.calls.append({"system": system, "user": user, "schema": schema, "max_tokens": max_tokens})
        self.last_usage = {"prompt_tokens": len(user) // 4, "completion_tokens": 50}
        if self.fn:
            return self.fn(user)
        return self.answers.pop(0)

    def _probe_once(self):
        return True


@pytest.fixture
def fake(monkeypatch):
    prov = FakeProvider([json.loads(json.dumps(FAKE_ANSWER))])
    monkeypatch.setattr(ls, "get_provider", lambda: prov)
    monkeypatch.setattr(lp, "get_provider", lambda: prov)
    return prov


# ------------------------------------------------------------------------------------------------ serialisation
def test_serialize_page_is_compact_and_ids_follow_block_order():
    res = make_result()
    lines = page_lines(res)
    assert [l["id"] for l in lines] == list(range(1, 17))            # the empty line of b8 is not sent
    assert lines[2]["cls"] == "text" and lines[2]["bold"] and lines[6]["bullet"] == "១."
    text = serialize_page(res)
    rows = text.splitlines()
    assert rows[0] == f"page {W}x{H}px, 16 lines"
    assert rows[1].startswith("blocks: b1 header [330 29 670 75]; b2 logo [60 21 200 121]")
    assert "L1 b1 r1 [350 29 650 50] s20 | ព្រះរាជាណាចក្រកម្ពុជា" in rows
    assert "L3 b3 r3 [300 157 700 186] s28 B | សេចក្តីប្រកាសព័ត៌មាន" in rows
    assert "L7 b5 r7 [140 307 700 329] s20 m:១. | ពាក្យស្នើសុំ និងឯកសារភ្ជាប់" in rows
    rb = ls.row_bands(lines)
    assert rb[10] == rb[11] == rb[12] and rb[13] == rb[14] == rb[15] == rb[10] + 1     # the table's two rows
    assert all(0 <= int(v) <= 1000 for r in rows[3:] for v in r.split("[")[1].split("]")[0].split())


def test_truncate_keeps_khmer_clusters_whole():
    t = "ក" + "្ក" * 100   # 201 chars of one cluster chain: coeng + consonant pairs must not be cut apart
    out = truncate_text(t, 10)
    assert out.endswith("…") and not out[:-1].endswith("្")
    assert truncate_text("short", 10) == "short"
    long_latin = "word " * 50
    assert len(truncate_text(long_latin)) == ls.TEXT_MAX + 1


def test_schema_is_strict_for_grammars():
    def check(node):
        for alt in node.get("anyOf", []):
            check(alt)
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node["required"]) == set(node["properties"])
            for v in node["properties"].values():
                check(v)
        if node.get("type") == "array":
            check(node["items"])
    check(SCHEMA)
    assert "qwen" not in ls.SYSTEM_PROMPT.lower() and "Khmer" in ls.SYSTEM_PROMPT


# ------------------------------------------------------------------------------------------------ validation / repair
def test_validate_and_repair_fixes_ids_cells_and_kind():
    ids = list(range(1, 17))
    boxes = {l["id"]: l["box"] for l in page_lines(make_result())}
    clean, stats = validate_and_repair(FAKE_ANSWER, ids, boxes)
    used = [i for e in clean["els"] for i in e["l"]] + [i for e in clean["els"] for c in e["c"] for i in c["l"]]
    assert sorted(used) == ids                                   # every line once, nothing invented
    assert stats["bad_ids"] == 1 and stats["dup_ids"] == 1 and stats["unused"] == 1
    assert clean["els"][-1] == {"t": "para", "lv": 0, "b": "", "l": [16], "c": []}
    table = next(e for e in clean["els"] if e["t"] == "table")
    assert all(c["rs"] >= 1 and c["cs"] >= 1 for c in table["c"]) and ls.grid_size(table["c"]) == (2, 3)
    assert clean["kind"] == "mixed"


def test_validate_and_repair_drops_bad_tables_and_overlaps():
    ids = [1, 2, 3, 4, 5]
    one_col = {"kind": "sheet", "els": [{"t": "table", "lv": 0, "b": "", "l": [], "c": [
        {"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "l": [1]}, {"r": 1, "c": 0, "rs": 1, "cs": 1, "h": False, "l": [2]}]}]}
    clean, stats = validate_and_repair(one_col, ids)
    assert stats["tables_dropped"] == 1 and clean["kind"] == "document"      # one column is not a table, no table -> document
    assert [e["t"] for e in clean["els"]] == ["para"] * 5 and [e["l"] for e in clean["els"]] == [[1], [2], [3], [4], [5]]
    overlap = {"kind": "document", "els": [{"t": "table", "lv": 0, "b": "", "l": [], "c": [
        {"r": 0, "c": 0, "rs": 1, "cs": 2, "h": True, "l": [1]}, {"r": 0, "c": 1, "rs": 1, "cs": 1, "h": True, "l": [2]},   # (0,1) covered by the span
        {"r": 1, "c": 0, "rs": 1, "cs": 1, "h": False, "l": [3]}, {"r": 1, "c": 1, "rs": 1, "cs": 1, "h": False, "l": [4]},
        {"r": -1, "c": 0, "rs": 1, "cs": 1, "h": False, "l": [5]}]}]}                                                  # out of range
    clean, stats = validate_and_repair(overlap, ids)
    table = clean["els"][0]
    assert table["t"] == "table" and stats["bad_cells"] == 2 and clean["kind"] == "mixed"
    assert [(c["r"], c["c"]) for c in table["c"]] == [(0, 0), (1, 0), (1, 1)]
    assert sorted(e["l"][0] for e in clean["els"][1:]) == [2, 5]            # their lines came back as paragraphs
    garbage, stats = validate_and_repair({"kind": "nope", "els": "x"}, [1, 2])
    assert garbage["kind"] == "document" and [e["l"] for e in garbage["els"]] == [[1], [2]]
    # lines a table puts in its own "l" become paragraphs; an id already owned by an earlier element is not duplicated
    stray = {"kind": "mixed", "els": [{"t": "para", "lv": 0, "b": "", "l": [5], "c": []},
                                      {"t": "table", "lv": 0, "b": "", "l": [5, 4], "c": [
                                          {"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "l": [1]}, {"r": 0, "c": 1, "rs": 1, "cs": 1, "h": False, "l": [2]},
                                          {"r": 1, "c": 0, "rs": 1, "cs": 1, "h": False, "l": [3]}]}]}
    clean, stats = validate_and_repair(stray, ids)
    assert [e["t"] for e in clean["els"]] == ["para", "table", "para"] and clean["els"][2]["l"] == [4]
    assert stats["dup_ids"] == 1 and sorted(i for e in clean["els"] for i in e["l"]) == [4, 5]


def test_table_structure_bands_cells_and_spans():
    boxes = [[100, 800, 160, 825], [300, 800, 400, 825], [620, 800, 700, 825], [110, 850, 125, 875], [300, 850, 380, 875], [670, 850, 700, 875], [300, 900, 400, 925]]
    texts = ["ល.រ", "ឈ្មោះ", "ចំនួន", "1", "កៅអី", "12", "សរុប"]
    cells = [{"r": 0, "c": 0, "rs": 1, "cs": 1, "h": True, "idx": [0]}, {"r": 0, "c": 1, "rs": 1, "cs": 1, "h": True, "idx": [1]},
             {"r": 0, "c": 2, "rs": 1, "cs": 1, "h": True, "idx": [2]}, {"r": 1, "c": 0, "rs": 1, "cs": 1, "h": False, "idx": [3]},
             {"r": 1, "c": 1, "rs": 1, "cs": 1, "h": False, "idx": [4]}, {"r": 1, "c": 2, "rs": 1, "cs": 1, "h": False, "idx": [5]},
             {"r": 2, "c": 0, "rs": 1, "cs": 3, "h": False, "idx": [6]}]
    t = table_structure(cells, boxes, texts, [True] * 3 + [False] * 4, [0.9] * 7)
    assert len(t["rows"]) == 3 and len(t["cols"]) == 3 and t["header_rows"] == 1 and t["source"] == "llm"
    assert t["rows"][0][1] == t["rows"][1][0] and 825 < t["rows"][0][1] < 850          # bands are contiguous, cut at the mid-gap
    assert t["cols"][0][1] == t["cols"][1][0] and 160 < t["cols"][0][1] < 300
    assert [(c["r"], c["c"]) for c in t["cells"]] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]  # spanned positions not repeated
    assert t["cells"][6]["cs"] == 3 and t["cells"][6]["text"] == "សរុប" and t["cells"][6]["lines"] == [6]
    assert t["cells"][0]["header"] and t["cells"][0]["bold"] and t["cells"][0]["box"] == [100, 800, 160, 825]
    assert t["cells"][5]["align"] == "r" and t["cells"][0]["align"] == "l"


def test_table_structure_fills_empty_cells():
    boxes = [[0, 0, 50, 10], [100, 0, 150, 10], [0, 20, 50, 30]]
    cells = [{"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "idx": [0]}, {"r": 0, "c": 1, "rs": 1, "cs": 1, "h": False, "idx": [1]},
             {"r": 1, "c": 0, "rs": 1, "cs": 1, "h": False, "idx": [2]}]
    t = table_structure(cells, boxes, ["a", "b", "c"])
    empty = t["cells"][3]
    assert (empty["r"], empty["c"], empty["text"], empty["lines"]) == (1, 1, "", [])
    assert empty["box"][0] == t["cols"][1][0] and empty["box"][1] == t["rows"][1][0]


# ------------------------------------------------------------------------------------------------ apply
def test_apply_rewrites_kind_blocks_table_and_document(fake):
    res = make_result()
    out = apply_llm_structure(res)
    assert out is res and "llm_error" not in out["timings"]
    assert out["doc_kind"]["kind"] == "mixed" and out["doc_kind"]["source"] == "llm" and out["doc_kind"]["tables"] == 1
    t = out["timings"]
    assert t["llm_model"] == "fake-1" and t["llm_calls"] == 1 and t["llm_tokens"] > 0 and t["llm_ms"] >= 0
    assert t["llm_repairs"]["unused"] == 1 and t["llm_lines"] == 16
    blocks = out["blocks"]
    assert [b["id"] for b in blocks] == [f"b{i}" for i in range(1, len(blocks) + 1)]
    assert [b["reading_order"] for b in blocks] == list(range(1, len(blocks) + 1))
    cls = [b["cls"] for b in blocks]
    assert cls[:2] == ["logo", "header"] or cls[:2] == ["header", "logo"]       # the crop block survives, placed by geometry
    assert cls[2:] == ["title", "text", "list", "section_header", "table", "text", "text"]   # footer line -> para (repair), empty block kept
    logo = next(b for b in blocks if b["cls"] == "logo")
    assert logo["crop"]["url"].endswith("logo.png")
    header = next(b for b in blocks if b["cls"] == "header")
    assert len(header["paragraphs"]) == 2 and [p["lines"] for p in header["paragraphs"]] == [[1], [2]]   # letterhead lines stay lines
    assert out["llm_structure"]["kind"] == "mixed" and out["llm_structure"]["els"][0]["l"] == [1, 2]
    title = next(b for b in blocks if b["cls"] == "title")
    assert title["relabel"] == "text" and title["source"] == "llm" and title["font_px"] == 28.0 and title["align"] == "center"
    para = blocks[3]
    assert len(para["lines"]) == 3 and para["paragraphs"][0]["lines"] == [1, 2, 3] and para["paragraphs"][0]["first_line_indent"] is True
    lst = blocks[4]
    assert [p["bullet"] for p in lst["paragraphs"]] == ["១.", "២."] and all(p["indent_level"] == 1 for p in lst["paragraphs"])
    head2 = blocks[5]
    assert head2["lines"][0]["text"] == "2. វិសាលភាព" and head2["lines"][0]["bullet"] is None   # the split-off number went back
    assert head2["lines"][0]["spans"][0] == {"text": "2.", "bold": True}
    table = blocks[6]
    assert table["cls"] == "table" and table["table"]["source"] == "llm"
    tb = table["table"]
    assert len(tb["rows"]) == 2 and len(tb["cols"]) == 3 and tb["header_rows"] == 1 and len(tb["cells"]) == 6
    assert tb["cells"][0]["text"] == "ល.រ" and tb["cells"][0]["lines"] == [0] and tb["cells"][0]["header"] is True
    assert tb["cells"][4]["text"] == "កៅអី" and tb["cells"][4]["lines"] == [4] and tb["cells"][4]["header"] is False
    assert tb["cells"][0]["box"] == [100, 800, 160, 825] and 790 <= table["box"]["y"] <= 800   # page pixels
    assert [l["reading_order"] for l in table["lines"]] == [1, 2, 3, 4, 5, 6]
    # the document is rebuilt from the new blocks
    els = out["document"]["elements"]
    types = [e["type"] for e in els]
    assert types[0] == "row"                                                     # letterhead: logo + motto columns
    assert types[1:5] == ["heading", "paragraph", "list", "heading"]
    assert els[1]["heading"] == 1 and els[4]["heading"] == 2 and els[4]["text"].startswith("2. ")
    assert els[2]["lines"] == [1, 2, 3] and els[2]["first_line_indent"] is True
    assert [it["bullet"] for it in els[3]["items"]] == ["១.", "២."]
    table_els = [e for e in els if e["type"] in ("table", "table_row")]
    assert table_els and table_els[0]["block"] == "b7"
    if table_els[0]["type"] == "table":                                          # once document.py emits contract §2.3 tables
        assert table_els[0]["rows"] == 2 and table_els[0]["cols"] == 3
    assert "Tel 012 345 678" in [e.get("text") for e in els]
    assert out["layout"]["margins"]["left"] > 0


def test_apply_without_provider_or_lines_is_a_no_op(monkeypatch):
    monkeypatch.setattr(ls, "get_provider", lambda: None)
    res = make_result()
    before = json.dumps(res["blocks"], sort_keys=True)
    out = apply_llm_structure(res)
    assert "not configured" in out["timings"]["llm_error"] and json.dumps(out["blocks"], sort_keys=True) == before
    assert "doc_kind" not in out
    tiny = {"width": 10, "height": 10, "blocks": [], "timings": {}}
    assert "fewer than 2" in apply_llm_structure(tiny)["timings"]["llm_error"]


def test_apply_keeps_result_when_the_model_fails(monkeypatch):
    class Failing(FakeProvider):
        def complete_json(self, *a, **k):
            raise LLMError("boom")
    monkeypatch.setattr(ls, "get_provider", lambda: Failing())
    res = make_result()
    out = apply_llm_structure(res)
    assert out["timings"]["llm_error"] == "boom" and out["timings"]["llm_model"] == "fake-1" and "doc_kind" not in out
    assert [b["id"] for b in out["blocks"]] == [f"b{i}" for i in range(1, 9)]


def test_truncated_answer_is_retried_with_a_bigger_budget():
    budgets = []

    class Trunc(FakeProvider):
        def complete_json(self, system, user, schema, max_tokens):
            budgets.append(max_tokens)
            self.last_usage = {"prompt_tokens": 10, "completion_tokens": max_tokens}
            if len(budgets) == 1:
                raise LLMTruncated("cut")
            return {"kind": "document", "els": [{"t": "para", "lv": 0, "b": "", "l": [1, 2], "c": []}]}
    lines = page_lines(make_result())[:2]
    answer, meta = structure_lines(lines, W, H, Trunc())
    assert budgets[1] == 2 * budgets[0] and meta["calls"] == 2 and answer["els"][0]["l"] == [1, 2]
    assert meta["tokens"]["completion_tokens"] == budgets[0] + budgets[1]


def test_long_pages_are_chunked_into_bands_at_gaps_and_merged():
    # 600 lines, a bigger gap after every 50th line: bands end at those gaps, no line is sent twice
    lines = [{"id": i + 1, "bi": 0, "li": i, "block": 1, "cls": "text", "box": [100, 10 * i + 20 * (i // 50), 900, 10 * i + 20 * (i // 50) + 8],
              "font_px": 8.0, "bold": False, "bullet": "", "text": f"line {i + 1}", "confidence": 1.0} for i in range(600)]
    seen = []

    def answer(user):
        ids = [int(r.split()[0][1:]) for r in user.splitlines() if r.startswith("L")]
        seen.append(ids)
        assert "part" in user.splitlines()[0]
        return {"kind": "sheet", "els": [{"t": "para", "lv": 0, "b": "", "l": ids[:len(ids) // 2] + [9999]},
                                         {"t": "para", "lv": 0, "b": "", "l": ids[len(ids) // 2:]}]}
    prov = FakeProvider(fn=answer)
    out, meta = structure_lines(lines, W, 6000, prov)
    assert meta["bands"] == len(seen) == meta["calls"] and meta["bands"] >= 4
    assert all(len(s) <= ls.MAX_LINES_PER_CALL and len(s) % 50 == 0 for s in seen)      # cut at the big gaps
    assert sorted(i for s in seen for i in s) == list(range(1, 601))                      # no overlap
    used = [i for e in out["els"] for i in e["l"]]
    assert used == list(range(1, 601))                                                    # merged in order, once each
    assert out["kind"] == "document"                                                      # no table survived -> document


def test_compact_table_rows_spans_and_padding():
    el = {"t": "table", "hr": 1, "rows": [[1, [2], 3], [[4], 0, [5]], [6, [7]]], "sp": [[1, 0, 1, 2], [0, 0, 1, 1]]}
    t = ls.table_from_rows(el)
    cells = {(c["r"], c["c"]): c for c in t["c"]}
    assert cells[(1, 0)]["cs"] == 2 and (1, 1) not in cells and cells[(0, 0)]["h"] and not cells[(1, 0)]["h"]
    assert cells[(2, 0)]["l"] == [6] and (2, 2) not in cells                               # short row: missing position
    # ids written into a covered position join the span's cell; a 1x1 "span" is ignored
    t2 = ls.table_from_rows({"t": "table", "hr": 0, "rows": [[[1], [2]], [[3], [4]]], "sp": [[0, 0, 1, 2]]})
    assert {(c["r"], c["c"]): c["l"] for c in t2["c"]} == {(0, 0): [1], (0, 1): [2], (1, 0): [3], (1, 1): [4]}   # bogus span ignored
    t3 = ls.table_from_rows({"t": "table", "hr": 0, "rows": [[1, 0], [3, 4]], "sp": [[0, 0, 1, 2]]})
    assert {(c["r"], c["c"]): (c["l"], c["cs"]) for c in t3["c"]} == {(0, 0): ([1], 2), (1, 0): ([3], 1), (1, 1): ([4], 1)}
    ans = {"kind": "sheet", "els": [{"t": "heading", "lv": 1, "b": "", "l": [8]}, el]}
    clean, stats = validate_and_repair(ans, list(range(1, 9)))
    tab = clean["els"][1]
    assert clean["kind"] == "sheet" and tab["t"] == "table" and ls.grid_size(tab["c"]) == (3, 3) and stats["unused"] == 0
    # an empty padding column / row the model added is trimmed
    pad = {"kind": "sheet", "els": [{"t": "table", "hr": 0, "rows": [[1, 0, 2], [[], 0, 0], [3, [], [4]]], "sp": []}]}
    clean, _ = validate_and_repair(pad, [1, 2, 3, 4])
    assert ls.grid_size(clean["els"][0]["c"]) == (2, 2)


def test_tables_cut_by_a_band_edge_are_joined():
    def ln(i, y, x):
        return {"id": i, "bi": 0, "li": i, "block": 1, "cls": "table", "box": [x, y, x + 50, y + 10], "font_px": 10.0, "bold": False,
                "bullet": "", "text": str(i), "confidence": 1.0}
    lines = [ln(1 + 2 * r, 20 * r, 100) for r in range(8)] + [ln(2 + 2 * r, 20 * r, 300) for r in range(8)]
    lines.sort(key=lambda l: l["id"])

    def answer(user):
        ids = sorted(int(r.split()[0][1:]) for r in user.splitlines() if r.startswith("L"))
        rows = [[a, [a + 1]] for a in ids if a % 2]
        return {"kind": "sheet", "els": [{"t": "table", "hr": 1, "rows": rows, "sp": []}]}
    old = ls.MAX_LINES_PER_CALL
    ls.MAX_LINES_PER_CALL = 6
    try:
        out, meta = structure_lines(lines, 1000, 1000, FakeProvider(fn=answer))
    finally:
        ls.MAX_LINES_PER_CALL = old
    assert meta["bands"] >= 3
    tables = [e for e in out["els"] if e["t"] == "table"]
    assert len(tables) == 1 and ls.grid_size(tables[0]["c"]) == (8, 2) and out["kind"] == "sheet"
    assert sum(1 for c in tables[0]["c"] if c["h"]) == 2                                  # only the first band's header


def test_page_budget_and_line_cap_raise(monkeypatch):
    lines = [{"id": i + 1, "bi": 0, "li": i, "block": 1, "cls": "text", "box": [100, 10 * i, 900, 10 * i + 8], "font_px": 8.0,
              "bold": False, "bullet": "", "text": "x", "confidence": 1.0} for i in range(400)]
    prov = FakeProvider(fn=lambda user: {"kind": "document", "els": []})
    with pytest.raises(LLMError, match="budget"):
        structure_lines(lines, W, 6000, prov, budget_s=-1)
    assert prov_calls(prov) == 1
    monkeypatch.setattr(ls, "MAX_PAGE_LINES", 100)
    with pytest.raises(LLMError, match="LLM_MAX_PAGE_LINES"):
        structure_lines(lines, W, 6000, prov)


def prov_calls(prov):
    return len(prov.calls)


def test_text_budget_shrinks_long_pages():
    assert ls.text_budget(10) == ls.TEXT_MAX and ls.text_budget(1000) == ls.TEXT_MIN
    lines = [{"id": i + 1, "block": 1, "box": [0, i, 10, i + 1], "font_px": 1, "text": "ក" * 200} for i in range(140)]
    rows = ls.serialize_lines(lines, 100, 200).splitlines()[2:]
    assert all(len(r.split(" | ", 1)[1]) <= ls.text_budget(140) + 1 for r in rows)


# ------------------------------------------------------------------------------------------------ providers
def _chat(content, finish="stop", usage=None):
    return {"choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20}}


def test_openai_provider_schema_request_and_repair_retry():
    requests = []
    replies = [_chat("Sure! here it is: not json at all"), _chat('```json\n{"kind": "document", "els": []}\n```')]

    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen3-14b"}]})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0))
    prov = OpenAICompatProvider("http://llm.test/v1", "qwen3-14b", api_key="k", transport=httpx.MockTransport(handler))
    assert prov.reachable() is True and prov.info() == {"provider": "openai", "model": "qwen3-14b", "base_url": "http://llm.test/v1"}
    out = prov.complete_json("sys", "user text", SCHEMA, 500)
    assert out == {"kind": "document", "els": []}
    assert len(requests) == 2 and prov.stats["repairs"] == 1 and prov.stats["calls"] == 2
    body = requests[0]
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA}}
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0 and body["max_tokens"] == 500
    assert body["messages"][0] == {"role": "system", "content": "sys"} and body["messages"][1]["content"] == "user text"
    assert requests[1]["messages"][2]["role"] == "assistant" and "not valid JSON" in requests[1]["messages"][3]["content"]
    assert prov.last_usage == {"prompt_tokens": 100, "completion_tokens": 20}


def test_openai_provider_errors():
    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            raise httpx.ConnectError("refused")
        body = json.loads(request.content)
        if body["max_tokens"] == 7:
            return httpx.Response(200, json=_chat('{"kind": "docu', finish="length"))
        if body["max_tokens"] == 8:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_chat("<think></think>{\"kind\": \"sheet\", \"els\": []}"))
    prov = OpenAICompatProvider("http://llm.test/v1", "m", transport=httpx.MockTransport(handler))
    assert prov.reachable() is False
    with pytest.raises(LLMTruncated):
        prov.complete_json("s", "u", SCHEMA, 7)
    with pytest.raises(LLMError):
        prov.complete_json("s", "u", SCHEMA, 8)
    assert prov.complete_json("s", "u", SCHEMA, 9) == {"kind": "sheet", "els": []}   # a leaked think block is tolerated
    assert prov.stats["truncated"] == 1 and prov.stats["failed"] == 1


def test_parse_json_text_tolerates_wrappers():
    assert parse_json_text('Answer:\n```json\n{"a": 1}\n```\nDone') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_text("[1, 2]")
    with pytest.raises(ValueError):
        parse_json_text("")


def test_anthropic_provider_uses_structured_output(monkeypatch):
    calls = []

    class Block:
        def __init__(self, text):
            self.type, self.text = "text", text

    class Msg:
        def __init__(self, text, stop="end_turn"):
            self.content, self.stop_reason, self.stop_details = [Block(text)], stop, None
            self.usage = type("U", (), {"input_tokens": 300, "output_tokens": 40})()

    class Stream:
        def __init__(self, msg):
            self.msg = msg

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return self.msg

    class Messages:
        def stream(self, **kw):
            calls.append(kw)
            return Stream(Msg('{"kind": "document", "els": []}'))

    class Models:
        def retrieve(self, model):
            return {"id": model}

    class Client:
        messages, models = Messages(), Models()

    prov = AnthropicProvider("claude-opus-5-5", api_key="x", client=Client())
    assert prov.reachable() is True and prov.info() == {"provider": "anthropic", "model": "claude-opus-5-5"}
    assert prov.complete_json("sys", "page", SCHEMA, 1234) == {"kind": "document", "els": []}
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 1234 and kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "page"}]
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert prov.last_usage == {"prompt_tokens": 300, "completion_tokens": 40}


def test_config_and_status(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_TIMEOUT_S", "7")
    lp.reset_provider()
    cfg = lp.llm_config()
    assert cfg == {"provider": "openai", "model": "qwen3-14b", "base_url": "http://127.0.0.1:8082/v1", "api_key": None, "timeout_s": 7.0,
                   "extra_body": {}, "grammar": "auto"}
    prov = lp.get_provider()
    assert isinstance(prov, OpenAICompatProvider) and prov.timeout_s == 7.0 and lp.get_provider() is prov   # cached
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    prov2 = lp.get_provider()
    assert isinstance(prov2, AnthropicProvider) and prov2.model == "claude-opus-5-5"          # a new config -> a new provider
    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert lp.get_provider() is None
    assert lp.llm_status() == {"configured": False, "provider": None, "model": None, "reachable": False}
    lp.reset_provider()


def test_unloaded_router_model_is_unreachable_and_apply_falls_back_fast(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "qwen3-14b", "status": {"value": "loaded"}},
                                                  {"id": "qwen2.5-7b", "status": {"value": "unloaded"}}]})
    monkeypatch.delenv("LLM_ALLOW_LOAD", raising=False)
    small = OpenAICompatProvider("http://llm.test/v1", "qwen2.5-7b", transport=httpx.MockTransport(handler))
    assert small.reachable() is False                       # never make a shared llama-server swap models
    assert OpenAICompatProvider("http://llm.test/v1", "qwen3-14b", transport=httpx.MockTransport(handler)).reachable() is True
    monkeypatch.setenv("LLM_ALLOW_LOAD", "1")
    assert OpenAICompatProvider("http://llm.test/v1", "qwen2.5-7b", transport=httpx.MockTransport(handler)).reachable() is True
    monkeypatch.setattr(ls, "get_provider", lambda: small)
    out = apply_llm_structure(make_result())
    assert "not reachable" in out["timings"]["llm_error"] and "doc_kind" not in out
    assert all(c.endswith("/models") for c in calls)        # no completion request was ever sent


def test_extra_body_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"top_k": 20}')
    lp.reset_provider()
    assert lp.get_provider().extra_body == {"top_k": 20}
    monkeypatch.setenv("LLM_EXTRA_BODY", "[1]")
    assert lp.llm_config()["extra_body"] == {}
    lp.reset_provider()


# ------------------------------------------------------------------------------------------------ router
def _app():
    from app.routers import llm as llm_router
    app = FastAPI()
    app.include_router(llm_router.router)
    return TestClient(app)


def test_router_status_unconfigured_and_structure_503(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    lp.reset_provider()
    c = _app()
    assert c.get("/llm/status").json() == {"configured": False, "provider": None, "model": None, "reachable": False}
    r = c.post("/llm/structure", json=make_result())
    assert r.status_code == 503 and "not configured" in r.json()["detail"]
    assert c.post("/llm/structure", json={"width": 1}).status_code == 422
    lp.reset_provider()


def test_router_structure_with_fake_provider(fake):
    c = _app()
    assert c.get("/llm/status").json() == {"configured": True, "provider": "fake", "model": "fake-1", "reachable": True, "busy": None}
    r = c.post("/llm/structure", json=make_result())
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["doc_kind"] == {"kind": "mixed", "confidence": pytest.approx(0.95), "source": "llm", "tables": 1,
                               "table_area": out["doc_kind"]["table_area"], "text_area": out["doc_kind"]["text_area"]}
    assert any(b.get("table") for b in out["blocks"]) and out["timings"]["llm_model"] == "fake-1"
    fake.calls.clear()
    fake.answers = []
    fake.fn = lambda user: (_ for _ in ()).throw(LLMError("down"))
    assert c.post("/llm/structure", json=make_result()).status_code == 502


def test_router_async_jobs(fake, monkeypatch):
    import time as _t
    jobs = ls.LLMJobs()
    monkeypatch.setattr(ls, "JOBS", jobs)
    from app.routers import llm as llm_router
    monkeypatch.setattr(llm_router, "JOBS", jobs)
    monkeypatch.setattr(llm_router, "submit_llm_job", lambda res: jobs.submit(json.loads(json.dumps(res))))
    c = _app()
    r = c.post("/llm/structure?async=1", json={**make_result(), "page_id": "p1"})
    assert r.status_code == 202, r.text
    j = r.json()
    assert j["status"] in ("queued", "running", "done") and j["poll"] == f"/llm/jobs/{j['id']}" and j["page_id"] == "p1"
    for _ in range(200):
        st = c.get(j["poll"]).json()
        if st["status"] in ("done", "failed"):
            break
        _t.sleep(0.02)
    assert st["status"] == "done" and st["result"]["doc_kind"]["source"] == "llm" and st["result"]["structure_engine"] == "llm"
    assert c.get("/llm/jobs").json()[0]["id"] == j["id"] and "result" not in c.get("/llm/jobs").json()[0]
    assert c.get("/llm/jobs/nope").status_code == 404
    # a failing model -> job failed with the reason, the worker survives
    fake.fn = lambda user: (_ for _ in ()).throw(LLMError("down"))
    j2 = c.post("/llm/structure", json={**make_result(), "async": True}).json()
    for _ in range(200):
        st = c.get(f"/llm/jobs/{j2['id']}").json()
        if st["status"] in ("done", "failed"):
            break
        _t.sleep(0.02)
    assert st["status"] == "failed" and st["error"] == "down"


def test_streamed_answer_and_deadline():
    sse = "".join(f"data: {json.dumps(e)}\n\n" for e in [
        {"choices": [{"index": 0, "delta": {"content": '{"kind":"document",'}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": '"els":[]}'}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 5}, "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0}}]) + "data: [DONE]\n\n"
    bodies = []

    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
    prov = OpenAICompatProvider("http://llm.test/v1", "m", transport=httpx.MockTransport(handler))
    assert prov.complete_json("s", "u", SCHEMA, 50) == {"kind": "document", "els": []}
    assert bodies[0]["stream"] is True and prov.last_usage["completion_tokens"] == 5
    assert prov.last_usage["server_timings"]["predicted_ms"] == 2.0
    with prov.deadline(-1):
        with pytest.raises(LLMError, match="deadline"):
            prov.complete_json("s", "u", SCHEMA, 50)


# ------------------------------------------------------------------------------------------------ teacher
def test_teacher_labels_follow_the_training_format():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if not (root / "training" / "structure" / "llm_label.py").exists():
        pytest.skip("training/ not present")
    sys.path.insert(0, str(root))
    from training.structure.data import ROLE_ID
    from training.structure.llm_label import answer_to_labels, lines_from_record
    res = make_result()
    lines, w, h, blocks = lines_from_record(res)
    assert (w, h, len(lines)) == (W, H, 16) and blocks is res["blocks"]
    answer, _ = validate_and_repair(FAKE_ANSWER, [l["id"] for l in lines])
    lab = answer_to_labels(answer, lines, answer["kind"])
    assert lab["role"][:3] == [ROLE_ID["header"], ROLE_ID["header"], ROLE_ID["title"]]
    assert lab["role"][3] == ROLE_ID["text"] and lab["role"][6] == ROLE_ID["list_item"] and lab["role"][8] == ROLE_ID["section_header"]
    assert lab["role"][9:15] == [ROLE_ID["table"]] * 6 and lab["role"][15] == ROLE_ID["text"]
    assert lab["first"] == [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert lab["bullet"] == [0] * 6 + [1, 1] + [0] * 8
    assert lab["level"][6] == 1 and lab["level"][0] == 0 and lab["level"][3] == -100     # list depth / fixed 0 / unknown
    assert sorted(lab["rank"]) == list(range(16)) and lab["rank"][:3] == [0, 1, 2] and lab["rank"][15] == 15
    assert len(lab["tables"]) == 1
    tab = lab["tables"][0]
    assert len(tab["rows"]) == 2 and len(tab["cols"]) == 3 and tab["header_rows"] == 1 and tab["sheet"] is False
    assert tab["cells"][0]["line_ids"] == [9] and tab["cells"][0]["text"] == "ល.រ" and tab["cells"][0]["header"] is True
    assert tab["cells"][4]["line_ids"] == [13] and tab["box"][1] < 800
    pipe = {"image": "x.jpg", "W": 100, "H": 50, "lines": [{"bbox": [1, 2, 30, 10], "text": "a", "bold": True, "bullet": "-"}, {"bbox": [1, 20, 30, 28], "text": "", "bold": False, "bullet": None}]}
    lines, w, h, blocks = lines_from_record(pipe)
    assert (w, h, blocks) == (100, 50, None) and len(lines) == 1 and lines[0]["bullet"] == "-" and lines[0]["bold"] is True


# ------------------------------------------------------------------------------------------------ teacher labels
def test_teacher_labels_have_v3_table_heads_and_load_as_pipe_pages():
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    pytest.importorskip("numpy")
    from training.structure import llm_label as L
    from training.structure.data import KIND_ID, ROLE_ID, page_from_pipe
    rec = {"image": "x.jpg", "W": 1000, "H": 1000, "lines": [
        {"bbox": [100, 50, 600, 80], "text": "Price list", "bold": True},
        {"bbox": [100, 100, 200, 120], "text": "Item"}, {"bbox": [400, 100, 500, 120], "text": "Price"},
        {"bbox": [100, 130, 200, 150], "text": "Pen"}, {"bbox": [400, 130, 500, 150], "text": "1.00"},
        {"bbox": [400, 155, 500, 175], "text": "(each)"}, {"bbox": [100, 300, 200, 320], "text": ""}]}
    ans = {"kind": "sheet", "els": [{"t": "heading", "lv": 1, "b": "", "l": [1]},
                                    {"t": "table", "hr": 1, "rows": [[2, 3], [4, [5, 6]]], "sp": []}]}
    out, why = L.label_record(rec, FakeProvider([ans]), 0.3)
    assert why is None and len(out["lines"]) == 6 and out["kind"] == KIND_ID["sheet"] and out["doc_kind"] == "sheet"
    assert out["role"] == [ROLE_ID["title"]] + [ROLE_ID["table"]] * 5
    assert out["tid"] == [-1, 0, 0, 0, 0, 0] and out["row"] == [-1, 0, 0, 1, 1, 1] and out["col"] == [-1, 0, 1, 0, 1, 1]
    assert out["cell"][4] == out["cell"][5] != out["cell"][3]
    t = out["tables"][0]
    assert t["header_rows"] == 1 and t["sheet"] and [c["line_ids"] for c in t["cells"]] == [[1], [2], [3], [4, 5]]
    page = page_from_pipe(json.loads(json.dumps(out)))
    assert page is not None and page.kind_label == KIND_ID["sheet"] and list(page.table["row"]) == out["row"]
    assert L.main(["--inputs", "/x/work/data/user_docs/a.jsonl", "--out", "/tmp/never.jsonl"]) == 2   # eval data refused


def test_compact_gbnf_grammar_for_llamacpp_servers(monkeypatch):
    g = lp.schema_to_gbnf(SCHEMA)
    assert g.splitlines()[0].startswith("root-kind ::=") and "root ::= \"{\" \"\\\"kind\\\"\" \":\" root-kind" in g
    assert " ws" not in g and "[ \\t\\n]" not in g                    # no whitespace anywhere in the grammar
    bodies = []

    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen3-14b", "owned_by": "llamacpp", "status": {"value": "loaded"}}]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_chat('{"kind":"document","els":[]}'))
    prov = OpenAICompatProvider("http://llm.test/v1", "qwen3-14b", transport=httpx.MockTransport(handler))
    prov.complete_json("s", "u", SCHEMA, 100)
    assert "grammar" in bodies[0] and "response_format" not in bodies[0]
    prov2 = OpenAICompatProvider("http://llm.test/v1", "qwen3-14b", transport=httpx.MockTransport(handler), grammar="schema")
    prov2.complete_json("s", "u", SCHEMA, 100)
    assert "response_format" in bodies[1] and "grammar" not in bodies[1]


def test_busy_single_slot_server_falls_back_at_once(monkeypatch):
    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen3-14b", "owned_by": "llamacpp", "status": {"value": "loaded"}}]})
        if request.url.path.endswith("/slots"):
            assert request.url.params.get("model") == "qwen3-14b"
            return httpx.Response(200, json=[{"id": 0, "is_processing": True}])
        raise AssertionError("no completion may be sent while the slot is busy")
    prov = OpenAICompatProvider("http://llm.test/v1", "qwen3-14b", transport=httpx.MockTransport(handler))
    assert prov.busy() is True
    monkeypatch.setattr(ls, "get_provider", lambda: prov)
    monkeypatch.delenv("LLM_WAIT_IF_BUSY", raising=False)
    out = apply_llm_structure(make_result())
    assert out["timings"]["llm_error"].startswith("LLM busy") and "doc_kind" not in out
    assert FakeProvider().busy() is None


def test_pipeline_hook_returns_phone_tier_at_once_and_queues_a_job(monkeypatch):
    from app.services.pipeline_service import PipelineService
    submitted = []
    monkeypatch.setattr(ls, "submit_llm_job", lambda res: submitted.append(res) or {"id": "j1", "status": "queued"})
    monkeypatch.delenv("LLM_SYNC", raising=False)
    p = PipelineService.__new__(PipelineService)
    p._impl = {"structure": None}
    res = {"blocks": [], "timings": {}, "structure_engine": "llm"}
    out = p._apply_llm(res)
    assert out is res and out["structure_engine"] == "rules" and out["llm_job"] == {"id": "j1", "status": "queued", "poll": "/llm/jobs/j1"}
    assert len(submitted) == 1
    # synchronous mode: short timeout, never queue behind a busy server
    seen = {}
    monkeypatch.setenv("LLM_SYNC", "1")
    monkeypatch.setenv("LLM_SYNC_TIMEOUT_S", "45")

    def fake_apply(res, timeout_s=None, wait_if_busy=None):
        seen.update(timeout_s=timeout_s, wait_if_busy=wait_if_busy)
        return {**res, "blocks": [], "timings": {}}
    monkeypatch.setattr(ls, "apply_llm_structure", fake_apply)
    out = p._apply_llm({"blocks": [], "timings": {}, "structure_engine": "llm"})
    assert seen == {"timeout_s": 45.0, "wait_if_busy": False} and out["structure_engine"] == "llm"


def test_truncated_answer_is_salvaged_when_it_places_most_lines():
    from app.services.llm_providers import parse_partial_json
    cut = '{"kind":"mixed","els":[{"t":"para","lv":0,"b":"","l":[1,2]},{"t":"table","hr":1,"rows":[[3,4],[5,6]],"sp":[]},{"t":"para","lv":0,"b":"","l":[7,'
    part = parse_partial_json(cut)
    assert part["kind"] == "mixed" and len(part["els"]) == 2 and part["els"][1]["rows"] == [[3, 4], [5, 6]]
    assert parse_partial_json('{"kind":"do') is None or parse_partial_json('{"kind":"do') == {}
    calls = []

    class Trunc(FakeProvider):
        def complete_json(self, system, user, schema, max_tokens):
            calls.append(max_tokens)
            self.last_usage = {"prompt_tokens": 1, "completion_tokens": max_tokens}
            raise LLMTruncated("cut", cut)
    lines = [{"id": i, "bi": 0, "li": i, "block": 1, "cls": "text", "box": [100 + 200 * (i % 2), 30 * i, 250 + 200 * (i % 2), 30 * i + 20],
              "font_px": 20.0, "bold": False, "bullet": "", "text": f"t{i}", "confidence": 1.0} for i in range(1, 9)]
    ans, meta = structure_lines(lines, 1000, 1000, Trunc())
    assert len(calls) == 1 and meta["salvaged"] == 1                         # 6 of 8 lines placed: no second call
    assert sorted(i for e in ans["els"] for i in e["l"] + [j for c in e["c"] for j in c["l"]]) == list(range(1, 9))


def test_geometric_regrid_fixes_ragged_rows_and_absorbs_row_labels():
    # 3 columns at x 100 / 400 / 700; the model wrote ragged rows (empty middle cell omitted), a table-only crop
    # where it also listed the row labels as paragraphs first, and a merged cell as repeated ids
    boxes = {}
    k = 0
    grid = {}
    for r in range(4):
        for c, x in enumerate((100, 400, 700)):
            if r == 2 and c == 1:
                continue                                  # empty cell
            k += 1
            boxes[k] = [x, 100 + 40 * r, x + 120, 125 + 40 * r]
            grid[(r, c)] = k
    labels = [grid[(r, 0)] for r in range(4)]
    rows = [[grid[(r, c)] for c in range(3) if (r, c) in grid and c > 0] for r in range(4)]   # model forgot column 0
    rows[2] = [grid[(2, 2)]]                                                                   # ragged: no 0 placeholder
    ans = {"kind": "mixed", "els": [{"t": "para", "lv": 0, "b": "", "l": [i]} for i in labels] +
           [{"t": "table", "hr": 1, "rows": rows, "sp": []}]}
    clean, stats = validate_and_repair(ans, list(boxes), boxes)
    tabs = [e for e in clean["els"] if e["t"] == "table"]
    assert len(tabs) == 1 and ls.grid_size(tabs[0]["c"]) == (4, 3) and stats["absorbed"] == 4
    pos = {(c["r"], c["c"]): c["l"] for c in tabs[0]["c"]}
    assert pos[(2, 2)] == [grid[(2, 2)]] and (2, 1) not in pos and pos[(3, 0)] == [grid[(3, 0)]]
    assert clean["kind"] == "sheet"                                                           # every line in the table
    t = ls.table_from_rows({"t": "table", "hr": 1, "rows": [[1, 1, 1], [2, 3, 4]], "sp": []})
    assert {(c["r"], c["c"]): (c["l"], c["cs"]) for c in t["c"]}[(0, 0)] == ([1], 3)         # repeated id = span


def test_streamed_degenerate_loop_is_stopped_and_salvaged():
    head = '{"kind":"document","els":[{"t":"para","lv":0,"b":"","l":[1,2]},{"t":"para","lv":0,"b":"","l":[3]}'
    loop = ',{"t":"para","lv":0,"b":"","l":[3]}'
    chunks = [head] + [loop] * 300 + [']}']
    sse = "".join(f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': c}, 'finish_reason': None}]})}\n\n" for c in chunks)

    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
    prov = OpenAICompatProvider("http://llm.test/v1", "m", transport=httpx.MockTransport(handler))
    lines = [{"id": i, "bi": 0, "li": i, "block": 1, "cls": "text", "box": [100, 30 * i, 500, 30 * i + 20], "font_px": 20.0,
              "bold": False, "bullet": "", "text": f"t{i}", "confidence": 1.0} for i in range(1, 5)]
    ans, meta = structure_lines(lines, 1000, 1000, prov)
    assert meta["salvaged"] == 1 and prov.stats["truncated"] == 1
    assert sorted(i for e in ans["els"] for i in e["l"]) == [1, 2, 3, 4]      # loop cut, line 4 added back
