#!/usr/bin/env python3
"""FinTabNet (IBM, CDLA-Permissive-1.0) financial-report tables, structure from FinTabNet.c (Microsoft, CDLA-Permissive-2.0)
-> work/data/tsr_real_fintabnet.

    python -m training.data.ext_tables_fintabnet fetch-xml --raw work/data/ext/tables_fintabnet/raw [--cap-mb 1200]
    python -m training.data.ext_tables_fintabnet convert   --raw work/data/ext/tables_fintabnet/raw --out work/data/tsr_real_fintabnet
        [--max-val 1500 --max-test 1500]

Sources:
  * ds4sd/FinTabNet_OTSL parquet shards (--raw/data/{val,test}-*.parquet; shard name = official FinTabNet split): `image`
    = tight 72-dpi PNG crop of the table, `cells` = one entry per <td> with bbox [x0,y0,x1,y1,cls] (cls 2 non-empty,
    1 empty) in crop pixels and character tokens (text). The OTSL `html` has no <thead>, so it is NOT used for structure.
  * FinTabNet.c XML (PubTables-1M formulation: table, table row, table column, table column header, table spanning cell,
    table projected row header). `fetch-xml` streams the first --cap-mb of bsmock/FinTabNet.c FinTabNet.c-Structure.tar.gz
    (order inside the tar: val/*.xml, test/*.xml, words/, images/ ..., train/*.xml only after 3 GB of images, so train
    XML is out of reach) and keeps the XML of pages present in the local shards (--raw/fintabnet_c_xml/{val,test}/).
Join (`match_xml`): FinTabNet.c names are COMPANY_YEAR_page_N_table_K, OTSL names COMPANY.YEAR.page_N.pdf_IMGID.png, so
candidates share the page key; FinTabNet.c crops are rendered ~1.26x larger, so a candidate must have table-box/OTSL-image
scale sx, sy in [1.1, 1.45] with |sx-sy| < 6 %, and the XML grid mapped into the OTSL frame (x = (x_c - tx0) / sx) must
contain >= 85 % of the OTSL content boxes (their centre's grid cell holds >= 80 % of the box); the best candidate wins
when it beats the runner-up by >= 0.1. Unmatched tables (FinTabNet.c dropped ~7 % of FinTabNet, multi-table pages with
equal sizes) are skipped and counted. Structure (bands, header rows, PRH rows, spans) = the mapped XML; cell text /
content box = union of the OTSL cells whose centre falls in that grid cell. A 2 % white margin is added around the tight
crop so the contract crop (table box + 2 %) has the usual context.
Splits (document = company + year): official val shard -> train; official test shard -> val / test by sha1 of the
document until --max-val / --max-test tables, the remaining test-shard documents -> train. The official train shard is
not used (no FinTabNet.c XML reachable -> no header labels).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from training.data.ext_tables_common import TsrWriter, clean_text, grid_from_voc, parse_voc, union

TAR_URL = "https://huggingface.co/datasets/bsmock/FinTabNet.c/resolve/main/FinTabNet.c-Structure.tar.gz"
MARGIN = 0.02


def page_key_otsl(filename: str) -> str:
    """'FDX.2008.page_46.pdf_55235.png' -> 'FDX_2008_page_46'"""
    return filename.split(".pdf")[0].replace(".", "_")


def page_key_c(name: str) -> str:
    """'SBUX_2018_page_90_table_1' -> 'SBUX_2018_page_90'"""
    return name.rsplit("_table_", 1)[0]


def doc_of(filename: str) -> str:
    return "_".join(page_key_otsl(filename).split("_")[:2])


def shard_split(p: Path) -> str:
    return p.name.split("-")[0]


# --------------------------------------------------------------------------- fetch-xml
def fetch_xml(a):
    keys = set()
    for s in sorted((a.raw / "data").glob("*.parquet")):
        for fn in pq.read_table(s, columns=["filename"]).column(0).to_pylist():
            keys.add(page_key_otsl(fn))
    print("page keys in local shards:", len(keys))
    out = a.raw / "fintabnet_c_xml"
    out.mkdir(exist_ok=True)
    cap = a.cap_mb * 1024 * 1024
    proc = subprocess.Popen(["curl", "-sL", "-r", f"0-{cap - 1}", TAR_URL], stdout=subprocess.PIPE)
    kept, seen, dirs, t = 0, 0, Counter(), time.time()
    first_image = None
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|gz")
        for m in tf:
            seen += 1
            parts = m.name.split("/")
            if len(parts) >= 2:
                dirs[parts[1]] += 1
            if parts[1:2] == ["images"] and first_image is None:
                first_image = seen
            if m.isfile() and m.name.endswith(".xml") and len(parts) == 3:
                stem = Path(parts[2]).stem
                if page_key_c(stem) in keys:
                    d = out / parts[1]
                    d.mkdir(exist_ok=True)
                    (d / parts[2]).write_bytes(tf.extractfile(m).read())
                    kept += 1
            if seen % 20000 == 0:
                print(f"  {seen} entries, kept {kept}, dirs {dict(dirs)}", flush=True)
    except (tarfile.ReadError, EOFError) as e:
        print("stream ended:", repr(e)[:80])
    proc.stdout.close()
    proc.wait()
    rep = {"entries_seen": seen, "kept_xml": kept, "dirs": dict(dirs), "first_image_entry": first_image,
           "cap_mb": a.cap_mb, "seconds": round(time.time() - t)}
    json.dump(rep, open(out / "fetch_report.json", "w"), indent=1)
    print(json.dumps(rep))


# --------------------------------------------------------------------------- convert
def containment(flat, voc_grid, sx, sy, tx0, ty0):
    """Share of OTSL content boxes (mapped into the XML frame) that lie (>= 80 %) inside the grid cell under their centre."""
    cells, R, C, hdr, prh, rb, cb = voc_grid
    n = ok = 0
    for c in flat:
        bb = c["bbox"]
        if not bb or len(bb) < 5 or bb[4] != 2:
            continue
        x0, y0, x1, y1 = bb[0] * sx + tx0, bb[1] * sy + ty0, bb[2] * sx + tx0, bb[3] * sy + ty0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ri = next((i for i, (p, q) in enumerate(rb) if p <= cy <= q), None)
        ci = next((j for j, (p, q) in enumerate(cb) if p <= cx <= q), None)
        n += 1
        if ri is None or ci is None:
            continue
        ix = max(0.0, min(x1, cb[ci][1]) - max(x0, cb[ci][0]))
        iy = max(0.0, min(y1, rb[ri][1]) - max(y0, rb[ri][0]))
        if ix * iy >= 0.8 * (x1 - x0) * (y1 - y0):
            ok += 1
    return ok / n if n else 0.0, n


def match_xml(row, W, H, cands):
    """-> (stem, voc, grid, sx, sy, score) of the matching FinTabNet.c annotation or (None, reason)."""
    flat = [c for g in row["cells"] for c in g]
    scored = []
    for stem, voc in cands:
        if not voc["table"] or not voc["row"] or not voc["column"]:
            continue
        tx0, ty0, tx1, ty1 = voc["table"][0]
        sx, sy = (tx1 - tx0) / W, (ty1 - ty0) / H
        if not (1.1 <= sx <= 1.45 and abs(sx - sy) / sx < 0.06):
            continue
        grid = grid_from_voc(voc)
        score, n = containment(flat, grid, sx, sy, tx0, ty0)
        scored.append((score, stem, voc, grid, sx, sy))
    if not scored:
        return None, "no scale-consistent candidate"
    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    if best[0] < 0.85:
        return None, f"low containment {best[0]:.2f}"
    if len(scored) > 1 and best[0] - scored[1][0] < 0.1:
        return None, "ambiguous"
    return best, None


def build_record(row, img, match):
    score, stem, voc, grid, sx, sy = match
    cells, R, C, header_rows, prh_rows, rb, cb = grid
    tx0, ty0, tx1, ty1 = voc["table"][0]
    H, W = img.shape[:2]
    mx, my = int(round(MARGIN * W)) + 1, int(round(MARGIN * H)) + 1
    img = cv2.copyMakeBorder(img, my, my, mx, mx, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    def fx(x):
        return (x - tx0) / sx + mx

    def fy(y):
        return (y - ty0) / sy + my

    def fb(b):
        return [fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])]

    rbm = [[fy(a), fy(b)] for a, b in rb]
    cbm = [[fx(a), fx(b)] for a, b in cb]
    by_pos = {}
    for c in cells:
        c["box"] = fb(c["box"]) if c.get("box") else None
        c["_otsl"] = []
        for a in range(c["r"], c["r"] + c["rs"]):
            for b in range(c["c"], c["c"] + c["cs"]):
                by_pos[(a, b)] = c
    for oc in (c for g in row["cells"] for c in g):
        bb = oc["bbox"]
        if not bb or len(bb) < 5 or bb[4] != 2:
            continue
        x0, y0, x1, y1 = bb[0] + mx, bb[1] + my, bb[2] + mx, bb[3] + my
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ri = next((i for i, (p, q) in enumerate(rbm) if p <= cy <= q), None)
        ci = next((j for j, (p, q) in enumerate(cbm) if p <= cx <= q), None)
        if ri is None or ci is None:
            continue
        by_pos[(ri, ci)]["_otsl"].append(([x0, y0, x1, y1], clean_text(oc["tokens"])))
    for c in cells:
        parts = sorted(c.pop("_otsl"), key=lambda t: (round(t[0][1] / 4), t[0][0]))
        if parts:
            c["text"] = " ".join(t for _, t in parts if t)
            c["box"] = union([b for b, _ in parts])
    rec = {"table_box": fb(voc["table"][0]), "R": R, "C": C, "cells": cells, "header_rows": header_rows, "prh_rows": prh_rows,
           "row_bands": rbm, "col_bands": cbm, "spans": [fb(b) for b in voc["span"]], "match_score": round(score, 3), "c_name": stem}
    return img, rec


def select_docs(filenames, max_tables):
    per_doc = Counter(doc_of(f) for f in filenames)
    order = sorted(per_doc, key=lambda d: hashlib.sha1(d.encode()).hexdigest())
    keep, n = set(), 0
    for d in order:
        if n >= max_tables:
            break
        keep.add(d)
        n += per_doc[d]
    return keep


def convert(a):
    xml_by_key = defaultdict(list)
    for split in ("val", "test"):
        for p in sorted((a.raw / "fintabnet_c_xml" / split).glob("*.xml")):
            xml_by_key[page_key_c(p.stem)].append((p.stem, parse_voc(p.read_text())))
    print("FinTabNet.c xml pages:", len(xml_by_key))
    shards = {shard_split(s): s for s in (a.raw / "data").glob("*.parquet")}
    # our splits: official val shard -> train; official test shard docs -> val / test / train
    test_names = pq.read_table(shards["test"], columns=["filename"]).column(0).to_pylist()
    val_docs = select_docs(test_names, a.max_val)
    test_docs = select_docs([f for f in test_names if doc_of(f) not in val_docs], a.max_test)
    print("test shard docs -> val", len(val_docs), "test", len(test_docs))

    def our_split(src_split, fn):
        if src_split == "val":
            return "train"
        d = doc_of(fn)
        return "val" if d in val_docs else ("test" if d in test_docs else "train")

    w = TsrWriter(a.out, "fintabnet")
    reasons = Counter()
    for src_split in ("val", "test"):
        pf = pq.ParquetFile(shards[src_split])
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                fn = row["filename"]
                img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    reasons["undecodable"] += 1
                    continue
                H, W = img.shape[:2]
                cands = xml_by_key.get(page_key_otsl(fn), [])
                if not cands:
                    reasons["no FinTabNet.c xml for page"] += 1
                    continue
                match, why = match_xml(row, W, H, cands)
                if match is None:
                    reasons[why] += 1
                    continue
                img2, rec = build_record(row, img, match)
                rec.update(source="fintabnet", doc_id=doc_of(fn), src_split=src_split, src_name=fn, nc=False, lang="en")
                e = w.add(our_split(src_split, fn), Path(fn).stem.replace(".pdf", ""), img2, rec)
                reasons["converted" if e else "label error"] += 1
            print(f"{src_split} rg {rg + 1}/{pf.metadata.num_row_groups}: {dict(reasons)}", flush=True)
    s = w.close()
    s["match"] = dict(reasons)
    json.dump(s, open(a.out / "summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch-xml", "convert"])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cap-mb", type=int, default=1200)
    ap.add_argument("--max-val", type=int, default=1500)
    ap.add_argument("--max-test", type=int, default=1500)
    a = ap.parse_args()
    fetch_xml(a) if a.cmd == "fetch-xml" else convert(a)


if __name__ == "__main__":
    main()
