"""Shared helpers for the real table-structure converters (training/data/ext_tables_*.py), docs/TABLES_CONTRACT.md §5-6.

Every set's parser yields one *table record* in source-image pixel coordinates:
    {"table_box": [x0,y0,x1,y1], "R": rows, "C": cols,
     "cells": [{"r","c","rs","cs","box": [x0,y0,x1,y1] | None, "text": str}],   # box = content box when the source has one
     "header_rows": int | None,          # None = the source does not mark header rows (no column_header label is written)
     "prh_rows": [row index, ...],       # projected row headers, only when the source marks them (PubTables-1M formulation)
     "row_bands": [[y0,y1],...] | None,  # canonical bands when the source provides them (PubTables-1M XML); derived otherwise
     "col_bands": [[x0,x1],...] | None,
     "spans": [[x0,y0,x1,y1], ...] | None,   # canonical spanning-cell boxes when the source provides them
     "doc_id": str, "source": str, ...extras}
`finish_table` turns it into the contract label: bands derived like docgen (training/docgen/tables.py::_bands with
ruled=False: the band boundary between two neighbours is the mid-gap between their ink, boundaries without ink are
interpolated, the first/last band touch the table box), cell grid box = intersection of its bands, column_header = one
box over the header rows, spanning_cell = grid box of every cell with rs > 1 or cs > 1.

Outputs per set (`TsrWriter`): work/data/tsr_real_<name>/{train,val,test}/{images,labels} (crop = table box padded by 2 %
of its width/height, long side resized to 800, JPEG q88; YOLO labels, classes TSR_NAMES) + tsr.yaml + cells_<split>.json =
list of {"image": "<split>/images/<name>.jpg", "table_box", "rows", "cols", "header_rows", "cells": [{r,c,rs,cs,box,text
[,header]}], "row_bands", "col_bands", "source", "doc_id", "nc"} with every coordinate in **crop pixels** (the image the
detector sees). Cell `box` = content box when the source has one, else the grid box; `grid_box` is always present.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

TSR_NAMES = ["row", "column", "column_header", "spanning_cell", "projected_row_header"]
PAD = 0.02
LONG_SIDE = 800
JPEG_Q = 88
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- small utilities
def r1(v) -> float:
    return round(float(v), 1)


def split_for(doc_id: str, train: float = 0.6, val: float = 0.1, salt: str = "tsr") -> str:
    """Deterministic document-level split from a hash of the document id."""
    h = int(hashlib.sha1(f"{salt}:{doc_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < train else ("val" if h < train + val else "test")


def clean_text(tokens) -> str:
    """OTSL / PubTabNet cell tokens (characters and HTML tags) -> plain text."""
    s = "".join(tokens) if not isinstance(tokens, str) else tokens
    return SPACE_RE.sub(" ", TAG_RE.sub("", s)).strip()


def union(boxes):
    b = np.asarray([x for x in boxes if x is not None], np.float64).reshape(-1, 4)
    if not len(b):
        return None
    return [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]


def clip(b, W, H):
    x0, y0, x1, y1 = b
    return [min(max(0.0, x0), W), min(max(0.0, y0), H), min(max(0.0, x1), W), min(max(0.0, y1), H)]


# --------------------------------------------------------------------------- HTML structure tokens -> grid
def html_grid(tokens):
    """PubTabNet-style structure tokens ('<thead>','<tr>','<td','  colspan="2"','>','</td>',...) -> (cells, R, C, header_rows).
    cells in <td> order: {"r","c","rs","cs","header"}; header = inside <thead>. header_rows = None when no <thead>."""
    cells, occ = [], set()
    r = -1
    c = 0
    in_head, saw_head, open_row = False, False, False
    head_rows = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "<thead>":
            in_head, saw_head = True, True
        elif t == "</thead>":
            in_head = False
        elif t == "<tr>" or (t == "</tr>" and not open_row):
            # an unmatched </tr> is an implicit empty row (the OTSL converter drops the <tr> of rows fully covered by rowspans)
            r += 1
            c = 0
            open_row = t == "<tr>"
            if in_head:
                head_rows = r + 1
        elif t == "</tr>":
            open_row = False
        elif t == "<td>" or t == "<td":
            rs = cs = 1
            if t == "<td":
                i += 1
                while i < len(tokens) and tokens[i] != ">":
                    m = re.search(r'(rowspan|colspan)="(\d+)"', tokens[i])
                    if m:
                        if m.group(1) == "rowspan":
                            rs = int(m.group(2))
                        else:
                            cs = int(m.group(2))
                    i += 1
            while (r, c) in occ:
                c += 1
            for a in range(r, r + rs):
                for b in range(c, c + cs):
                    occ.add((a, b))
            cells.append({"r": r, "c": c, "rs": rs, "cs": cs, "header": in_head})
            c += cs
        i += 1
    R = max((a for a, _ in occ), default=-1) + 1
    C = max((b for _, b in occ), default=-1) + 1
    return cells, R, C, (head_rows if saw_head else None)


# --------------------------------------------------------------------------- PASCAL VOC XML (PubTables-1M formulation)
VOC_NAMES = {"table": "table", "table row": "row", "table column": "column", "table column header": "header",
             "table spanning cell": "span", "table projected row header": "prh"}


def parse_voc(xml_text: str) -> dict:
    """-> {"size": (w, h), "table": [...], "row": [[x0,y0,x1,y1],...], "column": [...], "header": [...], "span": [...], "prh": [...]}"""
    root = ET.fromstring(xml_text)
    out = {v: [] for v in VOC_NAMES.values()}
    sz = root.find("size")
    out["size"] = (int(sz.find("width").text), int(sz.find("height").text)) if sz is not None else None
    out["filename"] = root.findtext("filename")
    for o in root.findall("object"):
        n = VOC_NAMES.get(o.findtext("name"))
        if n is None:
            continue
        bb = o.find("bndbox")
        out[n].append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
    out["row"].sort(key=lambda b: b[1])
    out["column"].sort(key=lambda b: b[0])
    return out


def _overlap_1d(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def grid_from_voc(voc: dict):
    """Rows/columns/spanning cells of a VOC annotation -> (cells, R, C, header_rows, prh_rows, row_bands, col_bands).
    A spanning box covers the grid positions whose bands overlap it by >= 50 % of the band; every other position is a
    1x1 cell. Header rows = rows whose band centre lies inside the column-header box; prh rows = rows whose band overlaps a
    projected-row-header box by >= 50 %."""
    rows, cols = voc["row"], voc["column"]
    R, C = len(rows), len(cols)
    rb = [[b[1], b[3]] for b in rows]
    cb = [[b[0], b[2]] for b in cols]
    occ = {}
    cells = []
    for sb in voc["span"]:
        rr = [i for i, (y0, y1) in enumerate(rb) if _overlap_1d(y0, y1, sb[1], sb[3]) >= 0.5 * max(1e-6, y1 - y0)]
        cc = [j for j, (x0, x1) in enumerate(cb) if _overlap_1d(x0, x1, sb[0], sb[2]) >= 0.5 * max(1e-6, x1 - x0)]
        if not rr or not cc:
            continue
        r0, r1_, c0, c1_ = min(rr), max(rr), min(cc), max(cc)
        if any((a, b) in occ for a in range(r0, r1_ + 1) for b in range(c0, c1_ + 1)):
            continue  # overlapping spans: keep the first, drop the rest (reported by the caller via check_grid)
        cell = {"r": r0, "c": c0, "rs": r1_ - r0 + 1, "cs": c1_ - c0 + 1, "box": list(sb), "text": ""}
        for a in range(r0, r1_ + 1):
            for b in range(c0, c1_ + 1):
                occ[(a, b)] = cell
        cells.append(cell)
    prh = []
    for pb in voc["prh"]:
        prh += [i for i, (y0, y1) in enumerate(rb) if _overlap_1d(y0, y1, pb[1], pb[3]) >= 0.5 * max(1e-6, y1 - y0)]
    prh = sorted(set(prh))
    for i in prh:  # a projected row header is one cell over the full width (PubTables-1M does not list it as a spanning cell)
        if all((i, b) not in occ for b in range(C)):
            cell = {"r": i, "c": 0, "rs": 1, "cs": C, "box": None, "text": ""}
            for b in range(C):
                occ[(i, b)] = cell
            cells.append(cell)
    for a in range(R):
        for b in range(C):
            if (a, b) not in occ:
                cell = {"r": a, "c": b, "rs": 1, "cs": 1, "box": None, "text": ""}
                occ[(a, b)] = cell
                cells.append(cell)
    cells.sort(key=lambda c: (c["r"], c["c"]))
    header_rows = None
    if voc["header"]:
        hb = union(voc["header"])
        hr = [i for i, (y0, y1) in enumerate(rb) if hb[1] <= (y0 + y1) / 2 <= hb[3]]
        header_rows = (max(hr) + 1) if hr else 0
    return cells, R, C, header_rows, prh, rb, cb


# --------------------------------------------------------------------------- bands + label
def derive_bands(n: int, lo: float, hi: float, ink):
    """n bands tiling [lo, hi]. ink[i] = (ink_lo, ink_hi) of the rs==1 / cs==1 cells with that index, or None.
    Boundary i (between band i-1 and i) = mid-gap of the neighbouring ink; unknown boundaries are interpolated."""
    b = [None] * (n + 1)
    b[0], b[n] = float(lo), float(hi)
    for i in range(1, n):
        p, q = ink[i - 1], ink[i]
        if p is not None and q is not None:
            b[i] = (p[1] + q[0]) / 2.0
    known = [i for i in range(n + 1) if b[i] is not None]
    for i in range(1, n):
        if b[i] is None:
            k0 = max(k for k in known if k < i)
            k1 = min(k for k in known if k > i)
            b[i] = b[k0] + (b[k1] - b[k0]) * (i - k0) / (k1 - k0)
    for i in range(1, n + 1):
        b[i] = max(b[i], b[i - 1])
    b = [min(max(v, lo), hi) for v in b]
    return [[b[i], b[i + 1]] for i in range(n)]


def ink_by_index(cells, R, C):
    row_ink, col_ink = [None] * R, [None] * C
    for c in cells:
        if not c.get("box"):
            continue
        x0, y0, x1, y1 = c["box"]
        if c["rs"] == 1:
            cur = row_ink[c["r"]]
            row_ink[c["r"]] = (y0, y1) if cur is None else (min(cur[0], y0), max(cur[1], y1))
        if c["cs"] == 1:
            cur = col_ink[c["c"]]
            col_ink[c["c"]] = (x0, x1) if cur is None else (min(cur[0], x0), max(cur[1], x1))
    return row_ink, col_ink


def check_grid(cells, R, C):
    """Problems of a cell grid (empty list = consistent): out-of-grid, overlaps, uncovered positions."""
    bad, seen = [], set()
    for c in cells:
        if c["rs"] < 1 or c["cs"] < 1 or c["r"] < 0 or c["c"] < 0 or c["r"] + c["rs"] > R or c["c"] + c["cs"] > C:
            bad.append(f"cell out of grid r={c['r']} c={c['c']} rs={c['rs']} cs={c['cs']} R={R} C={C}")
            continue
        for a in range(c["r"], c["r"] + c["rs"]):
            for b in range(c["c"], c["c"] + c["cs"]):
                if (a, b) in seen:
                    bad.append(f"overlap at {a},{b}")
                seen.add((a, b))
    if len(seen) != R * C:
        bad.append(f"grid not covered: {len(seen)} of {R * C}")
    return bad


def finish_table(rec: dict) -> dict:
    """Table record -> contract label (source-image pixels): adds row_bands/col_bands (derived when missing), grid_box and
    box for every cell, header flag, spans, prh boxes. Raises ValueError on an inconsistent grid."""
    R, C, cells = rec["R"], rec["C"], rec["cells"]
    bad = check_grid(cells, R, C)
    if bad or R < 1 or C < 1:
        raise ValueError("; ".join(bad) or "empty grid")
    tb = rec["table_box"]
    rb, cb = rec.get("row_bands"), rec.get("col_bands")
    if not rb or not cb:
        row_ink, col_ink = ink_by_index(cells, R, C)
        rb = rb or derive_bands(R, tb[1], tb[3], row_ink)
        cb = cb or derive_bands(C, tb[0], tb[2], col_ink)
    H = rec.get("header_rows")
    prh_rows = sorted(set(rec.get("prh_rows") or []))
    out_cells, spans = [], []
    for c in sorted(cells, key=lambda c: (c["r"], c["c"])):
        gb = [cb[c["c"]][0], rb[c["r"]][0], cb[c["c"] + c["cs"] - 1][1], rb[c["r"] + c["rs"] - 1][1]]
        oc = {"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": list(c["box"]) if c.get("box") else gb,
              "grid_box": gb, "text": c.get("text", "")}
        if H is not None:
            oc["header"] = c["r"] < H
        out_cells.append(oc)
        if c["rs"] > 1 or c["cs"] > 1:
            spans.append(gb)
    if rec.get("spans"):
        spans = [list(b) for b in rec["spans"]]
    lab = {"table_box": [float(v) for v in tb], "rows": R, "cols": C, "header_rows": H, "row_bands": rb, "col_bands": cb,
           "cells": out_cells, "spans": spans,
           "header_box": [tb[0], rb[0][0], tb[2], rb[H - 1][1]] if H else None,
           "prh_boxes": [[tb[0], rb[i][0], tb[2], rb[i][1]] for i in prh_rows if i < R]}
    for k in ("source", "doc_id", "nc", "lang", "variant", "skew_deg", "borders", "src_split", "src_name"):
        if k in rec:
            lab[k] = rec[k]
    return lab


# --------------------------------------------------------------------------- crop + YOLO + eval JSON
def crop_table(img: np.ndarray, table_box, pad: float = PAD, long_side: int = LONG_SIDE):
    """-> (crop, (ox, oy, s)): crop = table box padded by `pad` of its width/height per side (clipped to the image), long
    side resized to `long_side`. Source coords map to crop coords as ((x - ox) * s, (y - oy) * s)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = table_box
    pw, ph = pad * (x1 - x0), pad * (y1 - y0)
    cx0, cy0 = int(max(0, np.floor(x0 - pw))), int(max(0, np.floor(y0 - ph)))
    cx1, cy1 = int(min(W, np.ceil(x1 + pw))), int(min(H, np.ceil(y1 + ph)))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        raise ValueError("degenerate crop")
    crop = img[cy0:cy1, cx0:cx1]
    s = long_side / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return crop, (float(cx0), float(cy0), float(s))


def to_crop(b, tf):
    ox, oy, s = tf
    return [(b[0] - ox) * s, (b[1] - oy) * s, (b[2] - ox) * s, (b[3] - oy) * s]


def lab_to_crop(lab: dict, tf, W: int, H: int) -> dict:
    """Transform every coordinate of a finished label into crop pixels (clipped to the crop)."""
    ox, oy, s = tf
    out = dict(lab)
    out["table_box"] = [r1(v) for v in clip(to_crop(lab["table_box"], tf), W, H)]
    out["row_bands"] = [[r1(max(0.0, (a - oy) * s)), r1(min(H, (b - oy) * s))] for a, b in lab["row_bands"]]
    out["col_bands"] = [[r1(max(0.0, (a - ox) * s)), r1(min(W, (b - ox) * s))] for a, b in lab["col_bands"]]
    out["spans"] = [[r1(v) for v in clip(to_crop(b, tf), W, H)] for b in lab["spans"]]
    out["prh_boxes"] = [[r1(v) for v in clip(to_crop(b, tf), W, H)] for b in lab["prh_boxes"]]
    out["header_box"] = [r1(v) for v in clip(to_crop(lab["header_box"], tf), W, H)] if lab.get("header_box") else None
    cells = []
    for c in lab["cells"]:
        cc = dict(c)
        cc["box"] = [r1(v) for v in clip(to_crop(c["box"], tf), W, H)]
        cc["grid_box"] = [r1(v) for v in clip(to_crop(c["grid_box"], tf), W, H)]
        cells.append(cc)
    out["cells"] = cells
    return out


def yolo_lines(labc: dict, W: int, H: int) -> list[str]:
    """YOLO label lines (crop coords already) for the contract classes."""
    tb = labc["table_box"]
    boxes = [(0, [tb[0], a, tb[2], b]) for a, b in labc["row_bands"]]
    boxes += [(1, [a, tb[1], b, tb[3]]) for a, b in labc["col_bands"]]
    if labc.get("header_box"):
        boxes.append((2, labc["header_box"]))
    boxes += [(3, b) for b in labc["spans"]]
    boxes += [(4, b) for b in labc["prh_boxes"]]
    lines = []
    for cls, (x0, y0, x1, y1) in boxes:
        x0, y0, x1, y1 = clip([x0, y0, x1, y1], W, H)
        w, h = x1 - x0, y1 - y0
        if w < 1 or h < 1:
            continue
        lines.append(f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {w / W:.6f} {h / H:.6f}")
    return lines


def eval_entry(labc: dict, image_rel: str) -> dict:
    cells = []
    for c in labc["cells"]:
        e = {"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": c["box"], "grid_box": c["grid_box"], "text": c["text"]}
        if "header" in c:
            e["header"] = c["header"]
        cells.append(e)
    e = {"image": image_rel, "table_box": labc["table_box"], "rows": labc["rows"], "cols": labc["cols"],
         "header_rows": labc["header_rows"], "cells": cells, "row_bands": labc["row_bands"], "col_bands": labc["col_bands"],
         "source": labc.get("source", ""), "doc_id": labc.get("doc_id", ""), "nc": bool(labc.get("nc", False))}
    for k in ("lang", "variant", "skew_deg", "borders", "src_split", "src_name"):
        if k in labc:
            e[k] = labc[k]
    return e


class TsrWriter:
    """Writes one tsr_real_<name> set: images + YOLO labels per split, tsr.yaml, cells_<split>.json."""

    def __init__(self, out: Path, source: str):
        self.out = Path(out)
        self.source = source
        self.entries = {"train": [], "val": [], "test": []}
        self.errors = []
        for s in self.entries:
            (self.out / s / "images").mkdir(parents=True, exist_ok=True)
            (self.out / s / "labels").mkdir(parents=True, exist_ok=True)

    def add(self, split: str, name: str, img: np.ndarray, rec: dict) -> dict | None:
        """rec = table record (source pixels). Returns the eval entry or None (error recorded)."""
        try:
            lab = finish_table(rec)
            crop, tf = crop_table(img, lab["table_box"])
        except ValueError as e:
            self.errors.append(f"{name}: {e}")
            return None
        H, W = crop.shape[:2]
        labc = lab_to_crop(lab, tf, W, H)
        rel = f"{split}/images/{name}.jpg"
        cv2.imwrite(str(self.out / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        (self.out / split / "labels" / f"{name}.txt").write_text("\n".join(yolo_lines(labc, W, H)) + "\n")
        e = eval_entry(labc, rel)
        self.entries[split].append(e)
        return e

    def close(self) -> dict:
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(TSR_NAMES))
        (self.out / "tsr.yaml").write_text(f"# {self.source}: real table crops, docs/TABLES_CONTRACT.md §5\n"
                                           f"path: {self.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                                           f"names:\n{names}\n")
        summary = {}
        for s, ents in self.entries.items():
            with open(self.out / f"cells_{s}.json", "w") as f:
                json.dump(ents, f, ensure_ascii=False, separators=(",", ":"))
            summary[s] = stats(ents)
        summary["errors"] = len(self.errors)
        if self.errors:
            (self.out / "conversion_errors.txt").write_text("\n".join(self.errors) + "\n")
        with open(self.out / "summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        return summary


def stats(entries) -> dict:
    n = len(entries)
    if not n:
        return {"tables": 0}
    docs = {e.get("doc_id") for e in entries}
    cells = sum(len(e["cells"]) for e in entries)
    spans = sum(1 for e in entries for c in e["cells"] if c["rs"] > 1 or c["cs"] > 1)
    hdr_known = sum(1 for e in entries if e.get("header_rows") is not None)
    hdr = sum(1 for e in entries if e.get("header_rows"))
    return {"tables": n, "docs": len(docs), "cells": cells, "spanning_cells": spans, "header_known": hdr_known,
            "with_header": hdr, "rows_mean": round(sum(e["rows"] for e in entries) / n, 1),
            "cols_mean": round(sum(e["cols"] for e in entries) / n, 1)}


# --------------------------------------------------------------------------- previews
def draw_tsr_preview(img_path: Path, entry: dict, out: Path) -> None:
    """Rows (blue bands), columns (green), header (orange), spans (red), prh (purple), cells (thin grey + r,c)."""
    img = cv2.imread(str(img_path))
    ov = img.copy()
    for a, b in entry["row_bands"]:
        cv2.rectangle(ov, (int(entry["table_box"][0]), int(a)), (int(entry["table_box"][2]), int(b)), (230, 120, 40), -1)
    img = cv2.addWeighted(ov, 0.12, img, 0.88, 0)
    for i, (a, b) in enumerate(entry["row_bands"]):
        cv2.line(img, (int(entry["table_box"][0]), int(a)), (int(entry["table_box"][2]), int(a)), (230, 120, 40), 1)
        cv2.putText(img, f"r{i}", (max(0, int(entry["table_box"][0]) - 22), int((a + b) / 2) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 120, 40), 1, cv2.LINE_AA)
    for j, (a, b) in enumerate(entry["col_bands"]):
        cv2.line(img, (int(a), int(entry["table_box"][1])), (int(a), int(entry["table_box"][3])), (40, 170, 40), 1)
        cv2.putText(img, f"c{j}", (int((a + b) / 2) - 6, max(10, int(entry["table_box"][1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 170, 40), 1, cv2.LINE_AA)
    tb = entry["table_box"]
    cv2.rectangle(img, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), (0, 0, 0), 1)
    for c in entry["cells"]:
        x0, y0, x1, y1 = c["box"]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (150, 150, 150), 1)
        cv2.putText(img, f"{c['r']},{c['c']}", (int(x0) + 1, int(y0) + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (90, 90, 90), 1, cv2.LINE_AA)
        if c["rs"] > 1 or c["cs"] > 1:
            g = c["grid_box"]
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 0, 220), 2)
    if entry.get("header_rows"):
        y1 = entry["row_bands"][entry["header_rows"] - 1][1]
        cv2.rectangle(img, (int(tb[0]) + 2, int(tb[1]) + 2, int(tb[2] - tb[0]) - 4, int(y1 - tb[1]) - 4), (0, 140, 255), 2)
    for b in entry.get("prh_boxes", []):
        cv2.rectangle(img, (int(b[0]) + 4, int(b[1]) + 1), (int(b[2]) - 4, int(b[3]) - 1), (200, 0, 200), 2)
    txt = f"{entry.get('source','')} R={entry['rows']} C={entry['cols']} hdr={entry.get('header_rows')} {entry.get('doc_id','')[:28]}"
    cv2.putText(img, txt, (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def preview_set(set_dir: Path, out_dir: Path, n: int = 20, seed: int = 0, split: str | None = None) -> list[Path]:
    import random
    set_dir, out_dir = Path(set_dir), Path(out_dir)
    ents = []
    for s in (["train", "val", "test"] if split is None else [split]):
        p = set_dir / f"cells_{s}.json"
        if p.exists():
            ents += json.load(open(p))
    random.Random(seed).shuffle(ents)
    paths = []
    for i, e in enumerate(ents[:n]):
        # the eval JSON holds crop-pixel coordinates but not prh boxes: rebuild them from the label file classes
        e = dict(e)
        e["prh_boxes"] = prh_from_label(set_dir / e["image"].replace("/images/", "/labels/").replace(".jpg", ".txt"), e)
        out = out_dir / f"{i:02d}_{Path(e['image']).stem[:40]}.jpg"
        draw_tsr_preview(set_dir / e["image"], e, out)
        paths.append(out)
    return paths


def fill_missing(cells, R, C):
    """Add empty 1x1 cells at grid positions no cell covers (sources that only annotate non-empty cells)."""
    occ = set()
    for c in cells:
        for a in range(c["r"], c["r"] + c["rs"]):
            for b in range(c["c"], c["c"] + c["cs"]):
                occ.add((a, b))
    added = 0
    for a in range(R):
        for b in range(C):
            if (a, b) not in occ:
                cells.append({"r": a, "c": b, "rs": 1, "cs": 1, "box": None, "text": ""})
                added += 1
    cells.sort(key=lambda c: (c["r"], c["c"]))
    return added


def prh_from_label(label_path: Path, entry: dict):
    img = cv2.imread(str(Path(label_path).parent.parent / "images" / (Path(label_path).stem + ".jpg")))
    if img is None or not Path(label_path).exists():
        return []
    H, W = img.shape[:2]
    out = []
    for ln in Path(label_path).read_text().splitlines():
        p = ln.split()
        if len(p) == 5 and p[0] == "4":
            cx, cy, w, h = (float(v) for v in p[1:])
            out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return out
