#!/usr/bin/env python3
"""D4LA (Apache-2.0 per ModelScope damo/D4LA) form/invoice/letter/memo pages -> work/data/ext/d4la/{train,test}.

    python -m training.data.ext_d4la convert --raw work/data/_raw/d4la_pq --out work/data/ext/d4la
    python -m training.data.ext_d4la remap   --out work/data/ext/d4la      # re-apply MAP to rec["src_objs"], offline

Source files: the parquet mirror tuandunghcmut/D4LA (data/{train,test}-*.parquet; one file downloaded at a time, rows of
the 4 types kept, file deleted). Verified against the per-file mirror juliozhao/D4LA (DocLayout-YOLO author):
identical train/test membership (2834/711 pages of these types), pixel-identical images (checked 2 PNGs), identical
quads and identical class ids (checked every box of 5 pages). Split = D4LA's own train/test.

CLASS IDS - VERIFIED VISUALLY. The two mirrors agree on ids but publish different name lists. The correct names are the
DocLayout-YOLO d4la.yaml order (below). The tuandunghcmut card's names (0 DocTitle, 1 ListText, 2 LetterHead, ...) are
WRONG for its own ids: e.g. id 2 sits on body paragraphs, id 17 on a crest logo, id 7 on "Dear Jayne,", id 24 on the
vertical Bates number. Confirmed by drawing boxes on probe pages and per-class crop sheets.

Mapping D4LA (id name) -> v2 class. Each row was checked on a 12-crop sheet of random test boxes (what it really holds):
     0 DocTitle    -> title           ("INVOICE", "MEMORANDUM"; some are company-name letterheads -> would be header)
     1 ParaTitle   -> section_header   2 ParaText   -> text              3 ListText  -> list (one bulleted/numbered item)
     4 RegionTitle -> section_header   5 Date       -> text (date line of a letter/memo/invoice)
     6 LetterHead  -> text  !! NOT header: D4LA "LetterHead" is the INSIDE ADDRESS (recipient name/company/street block)
     7 LetterDear  -> text ("Dear ...")
     8 LetterSign  -> text  (closing + ink signature + typed name/title; the ink signature itself is NOT boxed)
     9 Question    -> text (rare)     10 OtherText  -> text  (letterhead fragments, "CONFIDENTIAL" stamps, slogans, notes)
    11 RegionKV    -> form_field  (key: value groups: "TO:/FROM:/DATE:/RE:/cc:", invoice header fields, incl. handwritten values)
    12 RegionList  -> table  !! mostly borderless tables / columnar data (line items, figures by month, name/title columns,
                              a checkbox grid); not bulleted lists
    13 Abstract    -> text            14 Author     -> text
    15 TableName   -> caption         16 Table      -> table (ruled tables and boxed form grids; one boxed stamp seen)
    17 Figure      -> logo if the box ends in the top 25% of the page and covers < 6% of it (crop sheet: ~10/11 such boxes
                      are company logos, 1 a RECEIVED stamp); otherwise NOT mapped -> rec["ignore_boxes"], because lower
                      Figures are a grab-bag of logos, state seals, ink stamps (RECEIVED/BEST COPY/date), hanko, barcode,
                      signature, illustration: no single v2 class fits, so the trainer should ignore those regions.
    18 FigureName  -> caption         19 Equation   -> formula          20 Reference -> text
    21 Footer      -> footer          22 PageHeader -> header (letterhead text, fax transmission lines at the top)
    23 PageFooter  -> footer
    24 Number      -> footer  (always the IIT-CDIP Bates/document id printed vertically in the lower-right margin: a marginal
                               archive artifact, not body text and not a page number)
    25 Catalog     -> list (rare)     26 PageNumber -> page_number

Known unlabelled v2 classes visible on these pages (false negatives for training; use ignore/partial strategies):
  signature (ink inside LetterSign or unboxed), stamp (inside OtherText/Table/ignored Figure or unboxed), handwriting
  (notes; filled values inside RegionKV), checkbox (inside RegionList/RegionKV/Table), logo (when not a separate top
  Figure), barcode. Hence partial_classes = all v2 ids except {12,13,14,15,16,17,18,19}.
lines: none (no text annotation) -> lines_ok false.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from training.data.ext_common import Manifest, decode, fit, obj, r1, save_jpg, stats
from training.taxonomy import CLS

D4LA_NAMES = ["DocTitle", "ParaTitle", "ParaText", "ListText", "RegionTitle", "Date", "LetterHead", "LetterDear",
              "LetterSign", "Question", "OtherText", "RegionKV", "RegionList", "Abstract", "Author", "TableName", "Table",
              "Figure", "FigureName", "Equation", "Reference", "Footer", "PageHeader", "PageFooter", "Number", "Catalog",
              "PageNumber"]
MAP = {
    "DocTitle": "title", "ParaTitle": "section_header", "ParaText": "text", "ListText": "list",
    "RegionTitle": "section_header", "Date": "text", "LetterHead": "text", "LetterDear": "text", "LetterSign": "text",
    "Question": "text", "OtherText": "text", "RegionKV": "form_field", "RegionList": "table", "Abstract": "text",
    "Author": "text", "TableName": "caption", "Table": "table", "Figure": "figure", "FigureName": "caption",
    "Equation": "formula", "Reference": "text", "Footer": "footer", "PageHeader": "header", "PageFooter": "footer",
    "Number": "footer", "Catalog": "list", "PageNumber": "page_number",
}
LOGO_TOP, LOGO_AREA = 0.25, 0.06
TYPES = ("form", "invoice", "letter", "memo")
UNLABELLED = {CLS[c] for c in ("signature", "stamp", "logo", "qr_code", "barcode", "photo", "handwriting", "checkbox")}
PARTIAL = [i for i in range(len(CLS)) if i not in UNLABELLED]
REPO = "tuandunghcmut/D4LA"


def map_obj(sid, x0, y0, x1, y1, W, H):
    name = D4LA_NAMES[sid]
    c = MAP[name]
    if c is None:
        return None
    if name == "Figure":
        if y1 <= LOGO_TOP * H and (x1 - x0) * (y1 - y0) < LOGO_AREA * W * H:
            c = "logo"
        else:
            return "ignore"
    return obj(CLS[c], [x0, y0, x1, y1], W, H)


def mk_objs(src, W, H):
    """-> (objs, ignore_boxes)."""
    objs, ign = [], []
    for s in src:
        o = map_obj(s[0], *s[1:], W, H)
        if o == "ignore":
            ign.append(list(s[1:]))
        elif o:
            objs.append(o)
    return objs, ign


def convert(a):
    from huggingface_hub import HfFileSystem, hf_hub_download
    import pyarrow.parquet as pq
    files = sorted(f.split("/", 3)[-1] for f in HfFileSystem().ls(f"datasets/{REPO}/data", detail=False))
    mans = {s: Manifest(a.out / s / "manifest.jsonl") for s in ("train", "test")}
    for fn in files:
        split = Path(fn).name.split("-")[0]
        path = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=str(a.raw))
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            names = pf.read_row_group(rg, columns=["file_name"]).column(0).to_pylist()
            keep = [i for i, n in enumerate(names) if n.split("_")[0] in TYPES]
            if not keep:
                continue
            t = pf.read_row_group(rg).take(keep)
            for r in t.to_pylist():
                img = decode(r["image"]["bytes"])
                img, _ = fit(img)
                H, W = img.shape[:2]
                src = []
                for cid, q in zip(r["category_ids"], r["quads"]):
                    p = np.array(q, np.float64).reshape(-1, 2) * [W, H]
                    src.append([int(cid), r1(p[:, 0].min()), r1(p[:, 1].min()), r1(p[:, 0].max()), r1(p[:, 1].max())])
                stem = Path(r["file_name"]).stem
                name = f"{stem}.jpg"
                save_jpg(img, a.out / split / "images" / name)
                objs, ign = mk_objs(src, W, H)
                mans[split].add(dict(image=f"images/{name}", w=W, h=H, source="d4la", doc_type=stem.split("_")[0],
                                     lang="en", objs=objs, ignore_boxes=ign, lines_ok=False, objs_ok=True,
                                     partial_classes=PARTIAL, src_objs=src))
        del pf
        os.remove(path)
        print(fn, {s: m.n for s, m in mans.items()}, flush=True)
    for s, m in mans.items():
        m.close()
        print(s, json.dumps(stats(a.out / s / "manifest.jsonl")))


def remap(a):
    for s in ("train", "test"):
        p = a.out / s / "manifest.jsonl"
        recs = [json.loads(x) for x in open(p)]
        with open(p, "w") as f:
            for r in recs:
                r["objs"], r["ignore_boxes"] = mk_objs(r["src_objs"], r["w"], r["h"])
                r["partial_classes"] = PARTIAL
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(s, json.dumps(stats(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "remap"])
    ap.add_argument("--raw", type=Path, default=Path("/home/hello/ocr/docsys/work/data/_raw/d4la_pq"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    convert(a) if a.cmd == "convert" else remap(a)


if __name__ == "__main__":
    main()
