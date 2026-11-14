"""Tests for training/table: eval_tsr (adjacency F1, TEDS-struct) and make_tsr_dataset (crops, YOLO labels, rectification).

    cd document_system_ocr && python -m pytest -q training/table/tests
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.table import eval_tsr as E
from training.table import make_tsr_dataset as M


# --------------------------------------------------------------------------- helpers
def grid_cells(R, C, header_rows=1, x0=0, y0=0, cw=50, rh=20, texts=True):
    cells = []
    for r in range(R):
        for c in range(C):
            cells.append({"r": r, "c": c, "rs": 1, "cs": 1, "box": [x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh],
                          "text": f"t{r}{c}" if texts else "", "header": r < header_rows})
    return cells


def table(R, C, **kw):
    return {"cells": grid_cells(R, C, **kw), "header_rows": kw.get("header_rows", 1)}


# --------------------------------------------------------------------------- eval_tsr
def test_tree_edit_distance_known():
    a = E.Node("table", [E.Node("tr", [E.Node(("td", 1, 1)), E.Node(("td", 1, 1))])])
    b = E.Node("table", [E.Node("tr", [E.Node(("td", 1, 1))])])
    assert E.tree_edit_distance(a, a) == 0
    assert E.tree_edit_distance(a, b) == 1  # one td deleted
    c = E.Node("table", [E.Node("tr", [E.Node(("td", 2, 1)), E.Node(("td", 1, 1))])])
    assert E.tree_edit_distance(a, c) == 1  # one relabel (colspan)
    assert E.tree_edit_distance(a, E.Node("table")) == 3


def test_identity_is_perfect():
    gt = [table(4, 3), table(3, 2)]
    summary, per = E.evaluate(gt, json.loads(json.dumps(gt)))
    assert summary["adjacency"]["f1"] == 1.0 and summary["teds_struct"] == 1.0 and summary["exact_match"] == 1.0
    assert all(t["adj_f1"] == 1.0 and t["teds_struct"] == 1.0 for t in per)


def test_merged_cell_is_penalised_but_partial():
    gt = table(4, 3)
    pred = json.loads(json.dumps(gt))
    # merge (1,0)+(1,1) into one spanning cell
    cells = [c for c in pred["cells"] if not (c["r"] == 1 and c["c"] == 1)]
    for c in cells:
        if c["r"] == 1 and c["c"] == 0:
            c["cs"] = 2
            c["box"][2] = 100
    pred["cells"] = cells
    s, per = E.evaluate([gt], [pred])
    assert 0.5 < s["adjacency"]["f1"] < 1.0
    assert 0.8 < s["teds_struct"] < 1.0
    assert s["exact_match"] == 0.0 and s["rows_acc"] == 1.0 and s["cols_acc"] == 1.0


def test_empty_prediction_scores_zero():
    s, per = E.evaluate([table(3, 3)], [None])
    assert s["adjacency"]["f1"] == 0.0 and s["teds_struct"] < 0.2 and s["tables"] == 1
    s2, _ = E.evaluate([table(3, 3)], [])  # shorter prediction list -> padded with empty tables
    assert s2["tables"] == 1 and s2["adjacency"]["recall"] == 0.0


def test_blank_cells_are_skipped_in_relations():
    cells = grid_cells(2, 3)
    cells[1]["text"] = ""  # (0,1) blank: (0,0) links to (0,2) directly
    rel = E.relations([dict(c) for c in cells], skip_blank=True)
    assert (0, 2, "h") in rel and (0, 1, "h") not in rel
    rel2 = E.relations([dict(c) for c in cells], skip_blank=False)
    assert (0, 1, "h") in rel2


def test_text_matching_and_loaders(tmp_path):
    gt = [table(3, 2)]
    pred = json.loads(json.dumps(gt))
    for c in pred[0]["cells"]:  # shifted boxes (bad IoU) but same texts
        c["box"] = [v + 40 for v in c["box"]]
    s_iou, _ = E.evaluate(gt, pred, match="iou")
    s_txt, _ = E.evaluate(gt, pred, match="text")
    assert s_iou["adjacency"]["f1"] < 1.0 and s_txt["adjacency"]["f1"] == 1.0
    # loader shapes: list of tables, {"tables": [...]}, pipeline result with blocks, jsonl
    (tmp_path / "a.json").write_text(json.dumps({"tables": gt}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps([{"blocks": [{"cls": "text"}, {"cls": "table", "table": pred[0]}]}]), encoding="utf-8")
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(t) for t in gt) + "\n", encoding="utf-8")
    assert len(E.load_tables(tmp_path / "a.json")) == 1 and len(E.load_tables(tmp_path / "b.json")) == 1 and len(E.load_tables(tmp_path / "c.jsonl")) == 1
    s = E.main(["--gt", str(tmp_path / "a.json"), "--pred", str(tmp_path / "b.json"), "--match", "text", "--json", str(tmp_path / "out.json")])
    assert s["adjacency"]["f1"] == 1.0 and (tmp_path / "out.json").exists()


# --------------------------------------------------------------------------- make_tsr_dataset
def _page_with_table(W=800, H=600, box=(100, 80, 700, 440), R=4, C=3):
    """White page with a drawn grid; returns (image RGB, table label dict in page px)."""
    img = np.full((H, W, 3), 255, np.uint8)
    x0, y0, x1, y1 = box
    xs = np.linspace(x0, x1, C + 1)
    ys = np.linspace(y0, y1, R + 1)
    for x in xs:
        cv2.line(img, (int(x), y0), (int(x), y1), (0, 0, 0), 2)
    for y in ys:
        cv2.line(img, (x0, int(y)), (x1, int(y)), (0, 0, 0), 2)
    cells = []
    for r in range(R):
        for c in range(C):
            if r == 2 and c == 1:
                continue  # covered by the 2-col span at (2,0)
            cs = 2 if (r == 2 and c == 0) else 1
            cells.append({"r": r, "c": c, "rs": 1, "cs": cs, "box": [xs[c], ys[r], xs[c + cs], ys[r + 1]], "header": r == 0, "text": f"c{r}{c}",
                          "align": "l", "bold": r == 0, "line_ids": [], "kind": "hdr" if r == 0 else "body"})
    t = {"box": list(map(float, box)), "rows": [[float(ys[i]), float(ys[i + 1])] for i in range(R)], "cols": [[float(xs[i]), float(xs[i + 1])] for i in range(C)],
         "header_rows": 1, "borders": "grid", "sheet": True, "cells": cells, "style": "grid", "excel": False}
    t["page"] = {"box": t["box"], "rows": t["rows"], "cols": t["cols"], "cell_boxes": [c["box"] for c in cells]}
    return img, t


def _write_split(root: Path, split: str, recs_imgs):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    with open(root / split / "manifest.jsonl", "w", encoding="utf-8") as f:
        for rec, img in recs_imgs:
            cv2.imwrite(str(root / split / rec["image"]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            f.write(json.dumps(rec) + "\n")


def test_make_tsr_dataset_flat_and_rotated(tmp_path):
    src = tmp_path / "src"
    img, t = _page_with_table()
    rec0 = {"image": "images/p0.jpg", "w": 800, "h": 600, "doc_type": "x", "lang": "en", "doc_kind": "sheet", "aug": {"degrade": False, "H": np.eye(3).reshape(-1).tolist()},
            "tables": [t], "objs": [], "lines": []}
    # rotated page: image warped by H, image-space labels = bounding boxes of the warped geometry
    ang = math.radians(8)
    Hm = np.array([[math.cos(ang), -math.sin(ang), 120], [math.sin(ang), math.cos(ang), 10], [0, 0, 1]])
    img_r = cv2.warpPerspective(img, Hm, (950, 750), borderValue=(40, 40, 40))

    def wb(b):
        P = cv2.perspectiveTransform(np.float64([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), Hm).reshape(-1, 2)
        return [float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())]
    t_r = json.loads(json.dumps(t))
    t_r["box"] = wb(t["box"])
    t_r["rows"] = [[wb([t["box"][0], a, t["box"][2], b])[1], wb([t["box"][0], a, t["box"][2], b])[3]] for a, b in t["rows"]]
    t_r["cols"] = [[wb([a, t["box"][1], b, t["box"][3]])[0], wb([a, t["box"][1], b, t["box"][3]])[2]] for a, b in t["cols"]]
    for c in t_r["cells"]:
        c["box"] = wb(c["box"])
    rec1 = {**rec0, "image": "images/p1.jpg", "w": 950, "h": 750, "aug": {"degrade": True, "mode": "rotate", "H": Hm.reshape(-1).tolist()}, "tables": [t_r]}
    _write_split(src, "train", [(rec0, img), (rec1, img_r)])
    out = tmp_path / "tsr"
    stats = M.main(["--src", str(src), "--out", str(out), "--splits", "train", "--long-side", "400", "--rectify", "always", "--preview", "2"])
    assert stats and stats[0]["crops"] == 2 and stats[0]["rectified"] == 1
    assert (out / "tsr.yaml").exists() and "projected_row_header" in (out / "tsr.yaml").read_text()
    labs = sorted((out / "train" / "labels").glob("*.txt"))
    assert len(labs) == 2
    for lab in labs:
        lines = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()]
        cls = [int(ln[0]) for ln in lines]
        assert cls.count(0) == 4 and cls.count(1) == 3 and cls.count(2) == 1 and cls.count(3) == 1 and cls.count(4) == 0
        for ln in lines:
            vals = [float(v) for v in ln[1:]]
            assert all(0.0 <= v <= 1.0 for v in vals)
        im = cv2.imread(str(out / "train" / "images" / (lab.stem + ".jpg")))
        assert max(im.shape[:2]) == 400
    # the rectified crop of the rotated page must carry the same (scaled) geometry as the flat one
    recs = [json.loads(ln) for ln in (out / "train" / "tables.jsonl").read_text().splitlines()]
    flat, rot = recs[0], recs[1]
    assert flat["rectified"] is False and rot["rectified"] is True
    assert np.allclose(np.array(flat["rows"]), np.array(rot["rows"]), atol=1.0) and np.allclose(np.array(flat["cols"]), np.array(rot["cols"]), atol=1.0)
    # and its pixels are an (almost) axis-aligned grid again: dark vertical rules at the labelled column boundaries
    im = cv2.imread(str(out / "train" / "images" / "p1_t0.jpg"), cv2.IMREAD_GRAYSCALE)
    x_rule = int(round(rot["cols"][1][0]))
    assert im[:, x_rule - 2:x_rule + 3].min() < 90
    # never-rectify keeps the bounding-band labels of the rotated page
    stats2 = M.main(["--src", str(src), "--out", str(tmp_path / "tsr2"), "--splits", "train", "--rectify", "never"])
    assert stats2[0]["rectified"] == 0 and stats2[0]["crops"] == 2
    assert (tmp_path / "tsr2" / "preview").exists() is False


def test_warp_info():
    ang, persp = M.warp_info(np.eye(3), 100, 100)
    assert ang == 0.0 and persp == 0.0
    a = math.radians(5)
    H = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    assert abs(M.warp_info(H, 100, 100)[0] - 5) < 1e-6


# --------------------------------------------------------------------------- docgen v5 labels (needs the fonts / old repo on this machine)
RES = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation")


@pytest.mark.skipif(not RES.exists(), reason="docgen resources not available")
def test_docgen_table_labels_consistent():
    import random
    from faker import Faker
    from training.docgen import generate as G
    from training.docgen import tables as T
    from training.docgen import blocks as B
    from training.docgen.canvas import Page
    from training.docgen.ctx import Ctx
    G.init_parent(0.5, 0.0, sheets=True)
    fake = Faker("en_US")
    n_tables = n_spans = 0
    for seed in range(12):
        rng = random.Random(seed)
        fake.seed_instance(seed)
        lang = "km" if seed % 2 else "en"
        pg = Page(1240, 1754, (255, 255, 255))
        ctx = Ctx(rng, lang, fake)
        y = B.b_table(pg, ctx, 100, 100, 1040, 1600, kind=rng.choice(["generic", "items", "statement"]), nrow=rng.randint(4, 12), caption=rng.random() < 0.5)
        assert y > 100
        assert pg.meta.get("tables") and len(pg.meta["tables"]) == 1
        lab = pg.meta["tables"][0]
        assert T.check_label(lab, len(pg.lines)) == []
        assert any(o[0] == 4 for o in pg.objs)  # the table layout box is still there
        n_tables += 1
        for c in lab["cells"]:
            n_spans += int(c["rs"] > 1 or c["cs"] > 1)
            for li in c["line_ids"]:  # printed lines sit inside their cell box (small tolerance for ink overshoot)
                q = pg.lines[li]
                assert min(q[0::2]) >= c["box"][0] - 3 and max(q[0::2]) <= c["box"][2] + 3
                assert min(q[1::2]) >= c["box"][1] - 8 and max(q[1::2]) <= c["box"][3] + 8
            assert (c["text"] == "") == (len(c["line_ids"]) == 0)
        # struct: every table line belongs to a "table" paragraph
        for c in lab["cells"]:
            for li in c["line_ids"]:
                assert pg.struct[li]["role"] == "table"
    assert n_tables == 12 and n_spans > 0
    # explicit data keeps its shape (no variety) - the application-form call site relies on it
    rng = random.Random(5)
    pg = Page(1240, 1754)
    ctx = Ctx(rng, "km", fake)
    hdr = ["ល.រ", "ឆ្នាំសិក្សា", "គ្រឹះស្ថាន", "សញ្ញាបត្រ"]
    B.b_table(pg, ctx, 100, 100, 1000, 1000, hdr=hdr, rows=[[ctx.d(i + 1), "", "", ""] for i in range(3)], al=["c", "l", "l", "l"], style="grid")
    lab = pg.meta["tables"][0]
    assert len(lab["rows"]) == 4 and len(lab["cols"]) == 4 and all(c["rs"] == 1 and c["cs"] == 1 for c in lab["cells"])
    assert pg.objs[-1][0] == 4


@pytest.mark.skipif(not RES.exists(), reason="docgen resources not available")
def test_sheet_pages_and_warped_records():
    import random
    from training.docgen import generate as G
    from training.docgen import tables as T
    from training.docgen import templates_sheet as S
    G.init_parent(0.5, 0.0, sheets=True)
    fake = G.G["fake"]
    kinds = list(S.BUILDERS)
    for i, kind in enumerate(kinds):
        rng = random.Random(100 + i)
        fake.seed_instance(100 + i)
        pg, ctx = S.sheet_page(rng, fake, "km" if i % 2 else "en", kind)
        assert pg.meta.get("doc_kind") == "sheet"
        assert pg.meta.get("tables"), kind
        lab = pg.meta["tables"][0]
        assert lab["sheet"] is True and T.check_label(lab, len(pg.lines)) == []
        assert len(lab["rows"]) >= 4
    # warp_tables: identity keeps the geometry, a rotation gives bounding boxes that contain the originals
    pg, ctx = S.sheet_page(random.Random(3), fake, "en", "budget")
    tabs = pg.meta["tables"]
    line_map = {i: i for i in range(len(pg.lines))}
    same = G.warp_tables(tabs, np.eye(3), pg.W, pg.H, line_map)
    assert len(same) == len(tabs)
    assert np.allclose(same[0]["box"], tabs[0]["box"], atol=0.1) and np.allclose(same[0]["rows"], tabs[0]["rows"], atol=0.1)
    assert same[0]["page"]["cell_boxes"][0] == [round(v, 1) for v in tabs[0]["cells"][0]["box"]]
    a = math.radians(4)
    Hm = np.array([[math.cos(a), -math.sin(a), 200], [math.sin(a), math.cos(a), 50], [0, 0, 1]])
    rot = G.warp_tables(tabs, Hm, pg.W + 400, pg.H + 400, line_map)
    b0, b1 = tabs[0]["box"], rot[0]["box"]
    assert (b1[2] - b1[0]) > (b0[2] - b0[0]) and (b1[3] - b1[1]) > (b0[3] - b0[1])
    # a table pushed out of the image is dropped
    far = G.warp_tables(tabs, np.array([[1, 0, 5000], [0, 1, 0], [0, 0, 1]], float), pg.W, pg.H, line_map)
    assert far == []
