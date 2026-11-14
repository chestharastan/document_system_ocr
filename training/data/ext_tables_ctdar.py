#!/usr/bin/env python3
"""ICDAR 2019 cTDaR, modern track B2 test set (NO LICENCE PUBLISHED -> EVALUATION ONLY) -> work/data/tsr_real_ctdar (test).

    python -m training.data.ext_tables_ctdar --raw work/data/ext/tables_ctdar/raw --out work/data/tsr_real_ctdar

Source: https://github.com/cndplab-founder/ICDAR2019_cTDaR test/TRACKB2/cTDaR_t1*.jpg (100 modern document images,
~1200x1600 px scans / renders of forms, reports, invoices, some with handwriting) + test_ground_truth/TRACKB2/*.xml
(fetched file by file from raw.githubusercontent.com into --raw/img and --raw/xml). The archival track (cTDaR_t0*)
is not used. Ground truth: `<table><Coords points="x,y ..."/>` + `<cell start-row end-row start-col end-col>` with the
content polygon ("Bounding box of the content describes a cell region" for the modern set). Only cells with content
are listed, so uncovered grid positions are filled with empty cells; r/c are 0-based; no header information
(header_rows = None, no column_header label), no text. One record per table (several tables per page possible),
doc_id = image name, all `nc: true`, split = test.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from training.data.ext_tables_common import TsrWriter, check_grid, fill_missing, union


def poly_box(points: str):
    p = np.array([[float(v) for v in xy.split(",")] for xy in points.split()], np.float64)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())], p


def skew_deg(p: np.ndarray) -> float:
    """Angle of the longest polygon edge (table outline) in degrees, 0 = axis aligned."""
    best, ang = 0.0, 0.0
    for i in range(len(p)):
        d = p[(i + 1) % len(p)] - p[i]
        L = float(np.hypot(*d))
        if L > best:
            best, ang = L, float(np.degrees(np.arctan2(d[1], d[0])))
    ang = ((ang + 45) % 90) - 45
    return round(ang, 2)


def tables_of(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    out = []
    for t in root.findall("table"):
        tb, tp = poly_box(t.find("Coords").get("points"))
        cells = []
        for c in t.findall("cell"):
            r0, r1 = int(c.get("start-row")), int(c.get("end-row"))
            c0, c1 = int(c.get("start-col")), int(c.get("end-col"))
            box, _ = poly_box(c.find("Coords").get("points"))
            cells.append({"r": r0, "c": c0, "rs": r1 - r0 + 1, "cs": c1 - c0 + 1, "box": box, "text": ""})
        if not cells:
            continue
        R = max(c["r"] + c["rs"] for c in cells)
        C = max(c["c"] + c["cs"] for c in cells)
        added = fill_missing(cells, R, C)
        ub = union([c["box"] for c in cells])
        tb = [min(tb[0], ub[0]), min(tb[1], ub[1]), max(tb[2], ub[2]), max(tb[3], ub[3])]
        out.append({"table_box": tb, "R": R, "C": C, "cells": cells, "header_rows": None, "prh_rows": [],
                    "skew_deg": skew_deg(tp), "filled_empty": added})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    w = TsrWriter(a.out, "ctdar_modern")
    n_img, filled = 0, 0
    for xp in sorted((a.raw / "xml").glob("cTDaR_t1*.xml")):
        ip = a.raw / "img" / (xp.stem + ".jpg")
        img = cv2.imread(str(ip))
        if img is None:
            w.errors.append(f"{xp.stem}: no image")
            continue
        n_img += 1
        for k, rec in enumerate(tables_of(xp)):
            filled += rec.pop("filled_empty")
            bad = check_grid(rec["cells"], rec["R"], rec["C"])
            if bad:
                w.errors.append(f"{xp.stem} t{k}: {bad[0]}")
                continue
            rec.update(source="ctdar_modern", doc_id=xp.stem, src_split="test", src_name=xp.stem, nc=True, lang="en")
            w.add("test", f"{xp.stem}_t{k}", img, rec)
    s = w.close()
    s["images"] = n_img
    s["filled_empty_cells"] = filled
    json.dump(s, open(a.out / "summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
