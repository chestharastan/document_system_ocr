"""docgen v2 - synthetic Cambodian + US/English document pages with exact line quads, line texts and v2 layout boxes.

    python -m training.docgen.generate --out DIR --n N --seed S --workers W [--lang-mix [P]] [--degrade-p 0.6]

--doc-types A,B : only these doc types (e.g. km_exam_sheet,en_exam_worksheet; --lang-mix then splits km/en)
--lang-mix P : probability that a page comes from the Cambodian (Khmer) template families (default 0.5; the flag alone
               = 0.5). The rest are US/English families. Khmer families mix in English, some English ones mix in Khmer.
--with-sheets : also register the v5 full-page table doc types (templates_sheet.py). Off by default so that v3/v4 runs
               keep their doc-type stream; a sheet type named in --doc-types registers them too.
--table-only : v5 table set: sheet doc types + the table-bearing document types (TABLE_DOC_TYPES); a page without a
               cell-labelled table is re-drawn (same doc type, derived seed, up to 12 tries); the sheet weights are
               rescaled so that --sheet-share (default 0.6) of each family's pages are sheets.
Output: DIR/images/*.jpg + DIR/manifest.jsonl (one JSON record per page; see README.md in this package).
v5 records carry ``tables`` (cell-level labels, warped like ``objs``) and ``doc_kind`` (document | sheet | mixed).
Run from the repo root with the det_venv python. CPU only. Uses fork() workers (forkserver OOMs on py3.14 here).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from . import res  # noqa: E402  (puts the old repo on sys.path)
from ..taxonomy import CLS  # noqa: E402

KH = re.compile(r"[ក-៿]")
LAT = re.compile(r"[A-Za-z]")
G: dict = {}

# document doc types that draw a cell-labelled table (b_table / group_table) at least sometimes
TABLE_DOC_TYPES = ["km_invoice", "km_bank_statement", "km_application_form", "km_textbook", "km_gov_notice",
                   "us_invoice", "us_bank_statement", "academic_paper", "en_textbook"]


def init_parent(lang_mix: float, degrade_p: float, only: list[str] | None = None, sheets: bool = False, table_only: bool = False,
                sheet_share: float = 0.6):
    res.load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401
    import qrcode  # noqa: F401
    import barcode  # noqa: F401
    from faker import Faker
    from . import templates_en, templates_exam, templates_km, templates_sheet
    G["fake"] = Faker("en_US")
    G["km"] = {**templates_km.KM_TEMPLATES, **templates_exam.EXAM_KM}
    G["en"] = {**templates_en.EN_TEMPLATES, **templates_exam.EXAM_EN}
    sheet_names = set(templates_sheet.SHEET_KM) | set(templates_sheet.SHEET_EN)
    if sheets or table_only or (only and any(n in sheet_names for n in only)):
        G["km"] = {**G["km"], **templates_sheet.SHEET_KM}
        G["en"] = {**G["en"], **templates_sheet.SHEET_EN}
    if table_only and not only:
        only = TABLE_DOC_TYPES + sorted(sheet_names)
    if only:  # restrict to the named doc types (a family with none left falls back to the other family)
        bad = [n for n in only if n not in G["km"] and n not in G["en"]]
        if bad:
            raise SystemExit(f"unknown doc type(s): {bad}")
        G["km"] = {k: v for k, v in G["km"].items() if k in only}
        G["en"] = {k: v for k, v in G["en"].items() if k in only}
    if table_only:  # sheet weights -> sheet_share of each family's mass
        for fam in ("km", "en"):
            docs = sum(wt for n, (_, wt) in G[fam].items() if n not in sheet_names)
            shs = sum(wt for n, (_, wt) in G[fam].items() if n in sheet_names)
            if docs > 0 and shs > 0 and 0 < sheet_share < 1:
                k = docs * sheet_share / (1 - sheet_share) / shs
                G[fam] = {n: (f, wt * k if n in sheet_names else wt) for n, (f, wt) in G[fam].items()}
    G["lang_mix"] = lang_mix
    G["degrade_p"] = degrade_p
    G["table_only"] = table_only
    G.setdefault("jpeg_q", (85, 92))
    G.setdefault("max_long", 2200)


def page_lang(texts):
    k = sum(len(KH.findall(t)) for t in texts)
    l_ = sum(len(LAT.findall(t)) for t in texts)
    if k == 0 and l_ == 0:
        return "en"
    if k == 0:
        return "en"
    if l_ < 0.08 * (k + l_):
        return "km"
    return "mixed"


def _xf(P: np.ndarray, Hm: np.ndarray) -> np.ndarray:
    if not len(P):
        return P
    return cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)


def _quad(x0, y0, x1, y1):
    return np.float64([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def warp_tables(tables, Hm, w, h, line_map):
    """pg.meta["tables"] (page px) -> manifest ``tables`` (image px): boxes = bbox of the warped quad, row / column
    bands = bounding band of the warped band quad, line_ids remapped to the kept lines. Tables < 60 % visible are dropped."""
    out = []
    for t in tables:
        bx = t["box"]
        Tq = _xf(_quad(*bx), Hm)
        x0, y0 = Tq.min(0)
        x1, y1 = Tq.max(0)
        a0 = max(1e-6, (x1 - x0) * (y1 - y0))
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if cx1 - cx0 < 3 or cy1 - cy0 < 3 or (cx1 - cx0) * (cy1 - cy0) < 0.6 * a0:
            continue

        def band_y(b):
            P = _xf(_quad(bx[0], b[0], bx[2], b[1]), Hm)
            return [round(float(np.clip(P[:, 1].min(), 0, h)), 1), round(float(np.clip(P[:, 1].max(), 0, h)), 1)]

        def band_x(b):
            P = _xf(_quad(b[0], bx[1], b[1], bx[3]), Hm)
            return [round(float(np.clip(P[:, 0].min(), 0, w)), 1), round(float(np.clip(P[:, 0].max(), 0, w)), 1)]

        cells = []
        for c in t["cells"]:
            P = _xf(_quad(*c["box"]), Hm)
            cells.append({"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"],
                          "box": [round(float(np.clip(P[:, 0].min(), 0, w)), 1), round(float(np.clip(P[:, 1].min(), 0, h)), 1),
                                  round(float(np.clip(P[:, 0].max(), 0, w)), 1), round(float(np.clip(P[:, 1].max(), 0, h)), 1)],
                          "header": bool(c["header"]), "text": c["text"], "align": c["align"], "bold": bool(c["bold"]),
                          "line_ids": [line_map[i] for i in c["line_ids"] if i in line_map], "kind": c.get("kind", "body")})
        out.append({"box": [round(float(cx0), 1), round(float(cy0), 1), round(float(cx1), 1), round(float(cy1), 1)],
                    "rows": [band_y(b) for b in t["rows"]], "cols": [band_x(b) for b in t["cols"]], "header_rows": int(t["header_rows"]),
                    "borders": t["borders"], "sheet": bool(t.get("sheet")), "cells": cells, "style": t.get("style", ""), "excel": bool(t.get("excel")),
                    # unwarped geometry (page px before aug; aug.H maps it to the image): exact bands for rectified crops
                    "page": {"box": [round(float(v), 1) for v in bx], "rows": [[round(float(v), 1) for v in b] for b in t["rows"]],
                             "cols": [[round(float(v), 1) for v in b] for b in t["cols"]], "cell_boxes": [[round(float(v), 1) for v in c["box"]] for c in t["cells"]]}})
    return out


def make_page(job):
    idx, seed, out = job
    rng = random.Random(seed)
    fake = G["fake"]
    fake.seed_instance(seed)
    fam = "km" if rng.random() < G["lang_mix"] else "en"
    if not G[fam]:
        fam = "en" if fam == "km" else "km"
    tpls = G[fam]
    names = list(tpls)
    name = rng.choices(names, [tpls[n][1] for n in names])[0]
    try:
        pg, ctx = tpls[name][0](rng, fake)
        if G.get("table_only"):  # redraw the same doc type until it carries a cell-labelled table
            for attempt in range(1, 12):
                if pg.meta.get("tables"):
                    break
                s2 = seed * 31 + attempt * 7919
                rng = random.Random(s2)
                fake.seed_instance(s2)
                pg, ctx = tpls[name][0](rng, fake)
    except Exception as e:  # a template bug must not kill the run; it is reported in the log
        return {"error": f"{name}: {e!r}", "tb": traceback.format_exc(limit=4), "idx": idx}
    img = np.asarray(pg.img)
    H0, W0 = img.shape[:2]
    aug = {"degrade": False}
    Hm = np.eye(3)
    if rng.random() < pg.meta.get("degrade_p", G["degrade_p"]):
        from .degrade2 import apply
        prof = pg.meta.get("degrade_profile")
        if prof:  # template-specific capture profile; gets the label geometry so it never half-cuts a labelled line
            lq = pg.lines + pg.hw_lines + pg.formula_lines
            meta = {**pg.meta, "line_pts": np.float64(lq).reshape(-1, 4, 2) if lq else None,
                    "obj_pts": [np.float64(P) for _, P in pg.objs]}
            img, Hm, info = apply(img, rng, prof, meta)
        else:
            img, Hm, info = apply(img, rng)
        aug = {"degrade": True, **info}
    h, w = img.shape[:2]
    # final resolution: long side 1000..2200 (templates may narrow it via meta["long_side"])
    long0 = max(w, h)
    lo, hi = pg.meta.get("long_side", (1000, 2200))
    hi = min(hi, G.get("max_long", 2200))
    target = float(np.clip(long0 * rng.uniform(0.75, 1.2), min(lo, hi), hi))
    s = target / long0
    if abs(s - 1) > 0.01:
        img = cv2.resize(img, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        S = np.diag([s, s, 1.0])
        Hm = S @ Hm
    h, w = img.shape[:2]
    aug["H"] = [round(float(v), 6) for v in Hm.reshape(-1)]

    kept_idx: list[list[int]] = []

    def quads(qs, keep_texts=None):
        out_q, out_t, keep = [], [], []
        kept_idx.append(keep)
        for i, q in enumerate(qs):
            P = _xf(np.float64(q).reshape(4, 2), Hm)
            if P[:, 0].min() < -1 or P[:, 1].min() < -1 or P[:, 0].max() > w + 1 or P[:, 1].max() > h + 1:
                continue
            keep.append(i)
            P[:, 0] = P[:, 0].clip(0, w)
            P[:, 1] = P[:, 1].clip(0, h)
            out_q.append([round(float(v), 1) for v in P.reshape(-1)])
            if keep_texts is not None:
                out_t.append(keep_texts[i])
        return out_q, out_t

    lines, texts = quads(pg.lines, list(zip(pg.line_texts, pg.line_fonts)))
    hw, hw_t = quads(pg.hw_lines, pg.hw_texts)
    fl, _ = quads(pg.formula_lines)
    objs = []
    for cls, P in pg.objs:
        T = _xf(P, Hm)
        x0, y0 = T.min(0)
        x1, y1 = T.max(0)
        a0 = max(1e-6, (x1 - x0) * (y1 - y0))
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if cx1 - cx0 < 3 or cy1 - cy0 < 3 or (cx1 - cx0) * (cy1 - cy0) < 0.6 * a0:
            continue
        objs.append([int(cls), round(float(cx0), 1), round(float(cy0), 1), round(float(cx1), 1), round(float(cy1), 1)])
    line_map = {old: new for new, old in enumerate(kept_idx[0])}
    tables = warp_tables(pg.meta.get("tables", []), Hm, w, h, line_map)
    kind = pg.meta.get("doc_kind")
    if not kind:
        has_tab = any(len(t["rows"]) >= 2 and len(t["cols"]) >= 2 for t in tables) or any(o[0] == CLS["table"] for o in objs)
        kind = "mixed" if has_tab else "document"
    fn = f"dg_{idx:07d}.jpg"
    cv2.imwrite(str(Path(out) / "images" / fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(*G.get("jpeg_q", (85, 92)))])
    lt = [t for t, _ in texts]
    rec = {"image": f"images/{fn}", "w": w, "h": h, "source": "docgen", "doc_type": name, "lang": page_lang(lt + hw_t),
           "lines": lines, "line_texts": lt, "line_fonts": [f for _, f in texts], "hw_lines": hw, "hw_texts": hw_t, "formula_lines": fl,
           "objs": objs, "lines_ok": True, "objs_ok": True, "seed": seed, "aug": aug,
           "struct": [pg.struct[i] for i in kept_idx[0]] if len(pg.struct) == len(pg.lines) else [],
           "tables": tables, "doc_kind": kind}
    if pg.meta.get("watermark"):
        rec["watermark"] = pg.meta["watermark"]
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lang-mix", type=float, nargs="?", const=0.5, default=0.5)
    ap.add_argument("--degrade-p", type=float, default=0.6)
    ap.add_argument("--start", type=int, default=0, help="first page index (to extend a dataset)")
    ap.add_argument("--doc-types", default=None, help="comma list: only generate these doc types (e.g. km_exam_sheet,en_exam_worksheet)")
    ap.add_argument("--with-sheets", action="store_true", help="register the v5 sheet doc types (templates_sheet.py)")
    ap.add_argument("--table-only", action="store_true", help="v5 table set: sheets + table-bearing doc types, every page has a labelled table")
    ap.add_argument("--sheet-share", type=float, default=0.6, help="with --table-only: share of sheet pages per language family")
    ap.add_argument("--jpeg-quality", default="85,92", help="lo,hi JPEG quality range of the page images (disk budget)")
    ap.add_argument("--max-long-side", type=int, default=2200, help="cap of the final long side in px (disk budget)")
    a = ap.parse_args()
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ql, qh = [int(v) for v in a.jpeg_quality.split(",")]
    G["jpeg_q"] = (ql, qh)
    G["max_long"] = a.max_long_side
    init_parent(a.lang_mix, a.degrade_p, a.doc_types.split(",") if a.doc_types else None, sheets=a.with_sheets, table_only=a.table_only,
                sheet_share=a.sheet_share)
    print(f"resources loaded in {time.time() - t0:.1f}s", flush=True)
    jobs = [(i, a.seed * 1_000_003 + i, str(a.out)) for i in range(a.start, a.start + a.n)]
    recs, errs = [], 0
    t1 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, maxtasksperchild=400) as pool:
        for k, r in enumerate(pool.imap_unordered(make_page, jobs, chunksize=2)):
            if "error" in r:
                errs += 1
                if errs <= 10:
                    print("   skipped:", r["error"], "\n", r["tb"], flush=True)
            else:
                recs.append(r)
            if (k + 1) % 100 == 0:
                el = time.time() - t1
                print(f"{k + 1}/{a.n} pages  {el:.0f}s  {(k + 1) / el:.2f} pages/s  errors={errs}", flush=True)
    recs.sort(key=lambda r: r["image"])
    mode = "a" if a.start > 0 and (a.out / "manifest.jsonl").exists() else "w"
    with open(a.out / "manifest.jsonl", mode, encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    el = time.time() - t1
    print(f"done: {len(recs)} pages ({errs} errors) in {el:.0f}s = {len(recs) / max(el, 1e-6):.2f} pages/s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
