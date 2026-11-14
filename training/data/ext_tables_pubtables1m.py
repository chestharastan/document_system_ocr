#!/usr/bin/env python3
"""PubTables-1M (CDLA-Permissive-2.0, Microsoft / PubMed Central OA) table crops -> work/data/tsr_real_pubtables1m.

    python -m training.data.ext_tables_pubtables1m --raw work/data/ext/tables_pubtables1m/raw --out work/data/tsr_real_pubtables1m
        [--max-val 1500 --max-test 1500] [--max-train 0]

Two sources for the same tables, joined by file name (PMC<id>_table_<k>):
  * images + cell content: ds4sd/PubTables-1M_OTSL parquet shards (--raw/data/*.parquet; shard name gives the official
    split). `image` is the official structure crop (same pixel size as the XML <size>, checked), `html` = structure
    tokens with <thead> / rowspan / colspan, `cells` = one entry per <td> in order with bbox [x0,y0,x1,y1,cls]
    (cls 2 = non-empty, 1 = empty; empty-cell boxes are ignored, the grid box is used) and character tokens.
  * canonical structure: the official PASCAL-VOC XML (bsmock/pubtables-1m PubTables-1M-Structure_Annotations_{Train,Val,
    Test}.tar.gz, streamed once, only the needed files kept in --raw/xml_cache/). Objects: table, table row,
    table column, table column header, table spanning cell, table projected row header. Rows / columns tile the
    table (the dilated canonical bands), so they are used as the row / column bands 1:1; the column header box gives
    header_rows (rows whose centre lies inside it), spanning-cell boxes are used as the spanning_cell labels, projected
    row headers are the PRH rows.
The HTML grid (r, c, rs, cs from the tokens) must agree with the XML grid (R = #rows, C = #columns, every spanning
cell found at the same position); tables that disagree are skipped and counted (`summary.json` -> errors,
conversion_errors.txt). Split = official split (document-level, by PMC id). val/test are subsampled to --max-val /
--max-test tables by taking whole documents in sha1 order of the PMC id.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from training.data.ext_tables_common import TsrWriter, clean_text, grid_from_voc, html_grid, parse_voc

SPLIT_TAR = {"train": "PubTables-1M-Structure_Annotations_Train.tar.gz", "val": "PubTables-1M-Structure_Annotations_Val.tar.gz",
             "test": "PubTables-1M-Structure_Annotations_Test.tar.gz"}


def shard_split(p: Path) -> str:
    return p.name.split("-")[0]


def doc_of(filename: str) -> str:
    return filename.split("_table_")[0]


def xml_cache(raw: Path, needed: dict[str, set[str]]) -> dict[str, str]:
    """needed: split -> set of xml basenames. Streams each tar once; returns basename -> xml text (cached on disk)."""
    cache = raw / "xml_cache"
    cache.mkdir(exist_ok=True)
    out = {}
    for split, names in needed.items():
        todo = {n for n in names if not (cache / n).exists()}
        for n in names - todo:
            out[n] = (cache / n).read_text()
        if not todo:
            continue
        tar = raw / SPLIT_TAR[split]
        t = time.time()
        with tarfile.open(tar, "r:gz") as tf:
            for m in tf:
                base = m.name.split("/")[-1]
                if base in todo and m.isfile():
                    txt = tf.extractfile(m).read().decode("utf-8")
                    (cache / base).write_text(txt)
                    out[base] = txt
                    todo.discard(base)
                    if not todo:
                        break
        print(f"{split}: xml from {tar.name} in {time.time() - t:.0f}s, missing {len(todo)}", flush=True)
    return out


def select_docs(filenames: list[str], max_tables: int) -> set[str]:
    """Whole documents in sha1 order until max_tables tables are reached (0 = all)."""
    if not max_tables:
        return {doc_of(f) for f in filenames}
    per_doc = Counter(doc_of(f) for f in filenames)
    order = sorted(per_doc, key=lambda d: hashlib.sha1(d.encode()).hexdigest())
    keep, n = set(), 0
    for d in order:
        if n >= max_tables:
            break
        keep.add(d)
        n += per_doc[d]
    return keep


def table_record(row: dict, xml: str):
    """OTSL row + VOC xml -> (record, problem). Grid from the XML; text/content boxes from the OTSL cells."""
    voc = parse_voc(xml)
    cells, R, C, header_rows, prh_rows, rb, cb = grid_from_voc(voc)
    hcells, hR, hC, hhead = html_grid(row["html"])
    if (hR, hC) != (R, C):
        return None, f"grid mismatch html {hR}x{hC} vs xml {R}x{C}"
    by_pos = {(c["r"], c["c"]): c for c in cells}
    flat = [c for grp in row["cells"] for c in grp]
    if len(flat) != len(hcells):
        return None, f"{len(flat)} cell entries for {len(hcells)} <td>"
    for hc, oc in zip(hcells, flat):
        tgt = by_pos.get((hc["r"], hc["c"]))
        if tgt is None or (tgt["rs"], tgt["cs"]) != (hc["rs"], hc["cs"]):
            return None, f"span mismatch at {hc['r']},{hc['c']}: html {hc['rs']}x{hc['cs']} xml {(tgt['rs'], tgt['cs']) if tgt else None}"
        tgt["text"] = clean_text(oc["tokens"])
        bb = oc["bbox"]
        if bb and len(bb) >= 4 and (len(bb) < 5 or bb[4] == 2) and bb[2] > bb[0] and bb[3] > bb[1]:
            tgt["box"] = [float(v) for v in bb[:4]]  # content box (non-empty cells only)
    tb = voc["table"][0] if voc["table"] else [cb[0][0], rb[0][0], cb[-1][1], rb[-1][1]]
    rec = {"table_box": tb, "R": R, "C": C, "cells": cells, "header_rows": header_rows, "prh_rows": prh_rows,
           "row_bands": rb, "col_bands": cb, "spans": voc["span"], "html_header_rows": hhead}
    return rec, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-val", type=int, default=1500)
    ap.add_argument("--max-test", type=int, default=1500)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    shards = sorted((a.raw / "data").glob("*.parquet"))
    shards = [s for s in shards if shard_split(s) in a.splits.split(",")]
    print("shards:", [s.name for s in shards])
    # pass 1: filenames per split, document subsampling
    names = {}
    for s in shards:
        names.setdefault(shard_split(s), []).extend(pq.read_table(s, columns=["filename"]).column(0).to_pylist())
    keep_docs = {}
    for split, fns in names.items():
        mx = {"train": a.max_train, "val": a.max_val, "test": a.max_test}[split]
        keep_docs[split] = select_docs(fns, mx)
        print(split, "tables", len(fns), "docs", len({doc_of(f) for f in fns}), "-> keep docs", len(keep_docs[split]))
    needed = {split: {f.replace(".jpg", ".xml") for f in fns if doc_of(f) in keep_docs[split]} for split, fns in names.items()}
    xmls = xml_cache(a.raw, needed)
    # pass 2: convert
    w = TsrWriter(a.out, "pubtables1m")
    hdr_cmp = Counter()
    for s in shards:
        split = shard_split(s)
        pf = pq.ParquetFile(s)
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                fn = row["filename"]
                if doc_of(fn) not in keep_docs[split]:
                    continue
                xml = xmls.get(fn.replace(".jpg", ".xml"))
                if xml is None:
                    w.errors.append(f"{fn}: no xml")
                    continue
                rec, prob = table_record(row, xml)
                if prob:
                    w.errors.append(f"{fn}: {prob}")
                    continue
                img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    w.errors.append(f"{fn}: undecodable image")
                    continue
                hh = rec.pop("html_header_rows")
                hdr_cmp["html_thead==xml_header" if hh == rec["header_rows"] else f"differ({hh} vs {rec['header_rows']})"[:30]] += 1
                rec.update(source="pubtables1m", doc_id=doc_of(fn), src_split=split, src_name=fn, nc=False, lang="en")
                w.add(split, Path(fn).stem, img, rec)
            print(f"{s.name} rg {rg + 1}/{pf.metadata.num_row_groups}: {sum(len(v) for v in w.entries.values())} tables, {len(w.errors)} errors", flush=True)
    summary = w.close()
    summary["header_rows_html_vs_xml"] = dict(hdr_cmp.most_common(8))
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
