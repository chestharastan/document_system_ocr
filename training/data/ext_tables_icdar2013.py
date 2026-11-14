#!/usr/bin/env python3
"""ICDAR-2013.c (CDLA-Permissive-2.0; corrected ICDAR 2013 Table Competition, EU/US government PDFs) -> work/data/tsr_real_icdar2013.

    python -m training.data.ext_tables_icdar2013 --raw work/data/ext/tables_icdar2013/raw --out work/data/tsr_real_icdar2013

Source: https://huggingface.co/datasets/bsmock/ICDAR-2013.c ICDAR-2013.c-Structure.tar.gz (read straight from the tar):
images/<name>.jpg (padded table crops), val/<name>.xml + test/<name>.xml (PASCAL VOC, PubTables-1M formulation: table,
table row, table column, table column header, table spanning cell, table projected row header), words/<name>_words.json
(word boxes + text in crop pixels). Grid from the XML exactly like PubTables-1M (rows / columns tile the table, PRH rows
become one full-width cell); cell text = the words whose centre falls in the cell's grid box, in reading order. Split =
the source's val (100 tables) / test (158 tables); eval set only (small), doc_id = document stem before `_table_`.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

from training.data.ext_tables_common import TsrWriter, finish_table, grid_from_voc, parse_voc


def words_to_cells(cells, R, C, rb, cb, words):
    """Assign words (box + text) to cells by centre; text = words sorted by (line, x)."""
    grid = {}
    for c in cells:
        for a in range(c["r"], c["r"] + c["rs"]):
            for b in range(c["c"], c["c"] + c["cs"]):
                grid[(a, b)] = c
    per = {}
    for wd in words:
        x0, y0, x1, y1 = wd["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ri = next((i for i, (a, b) in enumerate(rb) if a <= cy <= b), None)
        ci = next((j for j, (a, b) in enumerate(cb) if a <= cx <= b), None)
        if ri is None or ci is None:
            continue
        per.setdefault(id(grid[(ri, ci)]), (grid[(ri, ci)], [])).__getitem__(1).append((round(cy / 6), x0, wd["text"]))
    for c, ws in per.values():
        c["text"] = " ".join(t for _, _, t in sorted(ws))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tf = tarfile.open(a.raw / "ICDAR-2013.c-Structure.tar.gz", "r:gz")
    members = {m.name: m for m in tf.getmembers() if m.isfile()}
    w = TsrWriter(a.out, "icdar2013c")
    for split in ("val", "test"):
        for name in sorted(n for n in members if n.startswith(f"ICDAR-2013.c-Structure/{split}/") and n.endswith(".xml")):
            stem = Path(name).stem
            xml = tf.extractfile(members[name]).read().decode("utf-8")
            voc = parse_voc(xml)
            img_m = members.get(f"ICDAR-2013.c-Structure/images/{stem}.jpg")
            if img_m is None:
                w.errors.append(f"{stem}: no image")
                continue
            img = cv2.imdecode(np.frombuffer(tf.extractfile(img_m).read(), np.uint8), cv2.IMREAD_COLOR)
            cells, R, C, header_rows, prh_rows, rb, cb = grid_from_voc(voc)
            wm = members.get(f"ICDAR-2013.c-Structure/words/{stem}_words.json")
            if wm is not None:
                words_to_cells(cells, R, C, rb, cb, json.loads(tf.extractfile(wm).read().decode("utf-8")))
            tb = voc["table"][0] if voc["table"] else [cb[0][0], rb[0][0], cb[-1][1], rb[-1][1]]
            rec = {"table_box": tb, "R": R, "C": C, "cells": cells, "header_rows": header_rows, "prh_rows": prh_rows,
                   "row_bands": rb, "col_bands": cb, "spans": voc["span"], "source": "icdar2013c",
                   "doc_id": stem.split("_table_")[0], "src_split": split, "src_name": stem, "nc": False, "lang": "en"}
            w.add(split, stem, img, rec)
    s = w.close()
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
