"""Build the YOLO table-structure (TSR) dataset from docgen v5 manifests (docs/TABLES_CONTRACT.md §5).

    python -m training.table.make_tsr_dataset --src work/data/docgen_v5_tables --out work/data/tsr_v1
        [--splits train,val,test] [--long-side 800] [--pad 0.02] [--quality 88] [--rectify auto|never|always]
        [--skew-keep-p 0.25] [--skew-max-deg 3] [--min-rows 2] [--min-cols 2] [--max-per-split N] [--preview 20] [--seed 0]

One image per table crop (table box padded by --pad of its size, long side resized to --long-side), YOLO labels with
classes ``0 row, 1 column, 2 column_header, 3 spanning_cell, 4 projected_row_header`` (PubTables-1M formulation):
row boxes span the full table width, column boxes the full height, column_header is one box over all header rows,
spanning_cell one box per cell with rs > 1 or cs > 1, projected_row_header one box per row-group header cell (a cell
spanning every column outside the header; it also gets a spanning_cell box, as the contract defines that class by rs/cs).

Warped pages (docgen aug: rotation / perspective) store axis-aligned *bounding* bands, which overlap on rotated tables.
Each record also carries the unwarped geometry (``tables[].page``) and the page->image homography (``aug.H``), so by
default (``--rectify auto``) a table on a rotated / keystoned page is cropped **rectified** (warped back to page space
with the exact bands); slightly skewed tables (< --skew-max-deg) are kept as-is with probability --skew-keep-p so the
detector also sees a little residual skew. ``--rectify never`` keeps every crop as captured (bounding-band labels).

Outputs: OUT/{split}/images/*.jpg, OUT/{split}/labels/*.txt, OUT/{split}/tables.jsonl (one line per crop: source,
rectified flag, crop-space rows / cols / header_rows / cells [{r,c,rs,cs,box,text,header}] = the cell-level ground truth
for eval_tsr.py), OUT/tsr.yaml (ultralytics data file), OUT/preview/{split}/ (optional label drawings).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["row", "column", "column_header", "spanning_cell", "projected_row_header"]


def warp_info(H: np.ndarray, w: int, h: int):
    """(rotation in degrees, perspective strength in px over the image diagonal) of a 3x3 page->image homography."""
    Hn = H / H[2, 2]
    ang = float(np.degrees(np.arctan2(Hn[1, 0], Hn[0, 0])))
    persp = float((abs(Hn[2, 0]) + abs(Hn[2, 1])) * max(w, h))
    return ang, persp


def _clip01(v):
    return min(1.0, max(0.0, float(v)))


def yolo_line(cls: int, box, cw: int, ch: int):
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = max(0.0, min(x0, cw)), max(0.0, min(x1, cw))
    y0, y1 = max(0.0, min(y0, ch)), max(0.0, min(y1, ch))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return f"{cls} {_clip01((x0 + x1) / 2 / cw):.6f} {_clip01((y0 + y1) / 2 / ch):.6f} {_clip01((x1 - x0) / cw):.6f} {_clip01((y1 - y0) / ch):.6f}"


def crop_table(img: np.ndarray, rec: dict, t: dict, long_side: int, pad: float, rectify: bool):
    """-> (crop image, dict(rows, cols, box, cells(with crop boxes), header_rows, scale, rectified))."""
    h, w = img.shape[:2]
    if rectify and t.get("page"):
        geo = t["page"]
        bx = geo["box"]
        H = np.array(rec["aug"]["H"], np.float64).reshape(3, 3)
        rows, cols, cell_boxes = geo["rows"], geo["cols"], geo["cell_boxes"]
    else:
        bx = t["box"]
        H = np.eye(3)
        rows, cols, cell_boxes = t["rows"], t["cols"], [c["box"] for c in t["cells"]]
    bw, bh = bx[2] - bx[0], bx[3] - bx[1]
    if bw < 8 or bh < 8:
        return None, None
    px, py = pad * bw, pad * bh
    rx0, ry0, rx1, ry1 = bx[0] - px, bx[1] - py, bx[2] + px, bx[3] + py
    if not rectify:
        rx0, ry0, rx1, ry1 = max(0.0, rx0), max(0.0, ry0), min(float(w), rx1), min(float(h), ry1)
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw < 8 or rh < 8:
        return None, None
    s = long_side / max(rw, rh)
    cw, ch = max(8, int(round(rw * s))), max(8, int(round(rh * s)))
    A = np.array([[1 / s, 0, rx0], [0, 1 / s, ry0], [0, 0, 1]], np.float64)  # crop -> page (or image when not rectified)
    M = H @ A  # crop -> image
    crop = cv2.warpPerspective(img, M, (cw, ch), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    def tx(v):
        return (float(v) - rx0) * s

    def ty(v):
        return (float(v) - ry0) * s

    out = {"box": [tx(bx[0]), ty(bx[1]), tx(bx[2]), ty(bx[3])], "rows": [[ty(a), ty(b)] for a, b in rows], "cols": [[tx(a), tx(b)] for a, b in cols],
           "header_rows": int(t["header_rows"]), "scale": s, "rectified": bool(rectify), "cells": []}
    for c, cb in zip(t["cells"], cell_boxes):
        out["cells"].append({"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": [tx(cb[0]), ty(cb[1]), tx(cb[2]), ty(cb[3])],
                             "text": c.get("text", ""), "header": bool(c.get("header")), "kind": c.get("kind", "body")})
    return crop, out


def labels_for(g: dict, cw: int, ch: int):
    """YOLO label lines for one crop geometry."""
    bx = g["box"]
    C = len(g["cols"])
    lines = []
    for (y0, y1) in g["rows"]:
        ln = yolo_line(0, (bx[0], y0, bx[2], y1), cw, ch)
        if ln:
            lines.append(ln)
    for (x0, x1) in g["cols"]:
        ln = yolo_line(1, (x0, bx[1], x1, bx[3]), cw, ch)
        if ln:
            lines.append(ln)
    hr = g["header_rows"]
    if hr > 0 and hr <= len(g["rows"]):
        ln = yolo_line(2, (bx[0], g["rows"][0][0], bx[2], g["rows"][hr - 1][1]), cw, ch)
        if ln:
            lines.append(ln)
    for c in g["cells"]:
        if c["rs"] > 1 or c["cs"] > 1:
            ln = yolo_line(3, c["box"], cw, ch)
            if ln:
                lines.append(ln)
        if (c.get("kind") == "group" or (c["cs"] == C and C > 1 and not c["header"])) and c["r"] >= hr:
            ln = yolo_line(4, c["box"], cw, ch)
            if ln:
                lines.append(ln)
    return lines


def draw_preview(crop: np.ndarray, lines: list[str]):
    img = crop.copy()
    h, w = img.shape[:2]
    cols = {0: (255, 160, 60), 1: (60, 200, 255), 2: (0, 0, 255), 3: (255, 0, 255), 4: (0, 140, 255)}
    for ln in lines:
        k, cx, cy, bw, bh = ln.split()
        k = int(k)
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        x0, y0, x1, y1 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
        off = {0: 0, 1: 0, 2: 3, 3: 1, 4: 2}[k]
        cv2.rectangle(img, (x0 + off, y0 + off), (x1 - off, y1 - off), cols[k], 2 if k >= 2 else 1)
    return img


def build_split(src_dir: Path, out_dir: Path, split: str, a, rng: random.Random):
    man = src_dir / split / "manifest.jsonl"
    if not man.exists():
        print(f"[{split}] no manifest at {man}, skipped")
        return None
    (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    prev_dir = out_dir / "preview" / split
    if a.preview:
        prev_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(ln) for ln in open(man, encoding="utf-8") if ln.strip()]
    n_img, n_rect, n_skip = 0, 0, 0
    cls_count = Counter()
    n_prev = 0
    with open(out_dir / split / "tables.jsonl", "w", encoding="utf-8") as fo:
        for rec in recs:
            tables = rec.get("tables") or []
            if not tables:
                continue
            img = None
            for k, t in enumerate(tables):
                if len(t["rows"]) < a.min_rows or len(t["cols"]) < a.min_cols:
                    n_skip += 1
                    continue
                if img is None:
                    img = cv2.imread(str(src_dir / split / rec["image"]))
                    if img is None:
                        break
                h, w = img.shape[:2]
                Hm = np.array(rec.get("aug", {}).get("H", np.eye(3).reshape(-1).tolist()), np.float64).reshape(3, 3)
                ang, persp = warp_info(Hm, w, h)
                skewed = abs(ang) > 0.3 or persp > 0.01
                if a.rectify == "never" or not t.get("page"):
                    rect = False
                elif a.rectify == "always":
                    rect = skewed
                else:  # auto
                    rect = skewed and (abs(ang) > a.skew_max_deg or persp > 0.01 or rng.random() >= a.skew_keep_p)
                crop, g = crop_table(img, rec, t, a.long_side, a.pad, rect)
                if crop is None:
                    n_skip += 1
                    continue
                ch, cw = crop.shape[:2]
                lines = labels_for(g, cw, ch)
                if sum(1 for ln in lines if ln.startswith("0 ")) < a.min_rows or sum(1 for ln in lines if ln.startswith("1 ")) < a.min_cols:
                    n_skip += 1
                    continue
                stem = f"{Path(rec['image']).stem}_t{k}"
                cv2.imwrite(str(out_dir / split / "images" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                with open(out_dir / split / "labels" / f"{stem}.txt", "w") as fl:
                    fl.write("\n".join(lines) + ("\n" if lines else ""))
                cls_count.update(int(ln.split()[0]) for ln in lines)
                fo.write(json.dumps({"image": f"images/{stem}.jpg", "src_image": rec["image"], "src_table": k, "doc_type": rec.get("doc_type"),
                                     "lang": rec.get("lang"), "doc_kind": rec.get("doc_kind"), "w": cw, "h": ch, "rectified": g["rectified"],
                                     "scale": round(g["scale"], 5), "skew_deg": round(ang, 3), "style": t.get("style"), "borders": t.get("borders"),
                                     "sheet": t.get("sheet"), "box": [round(v, 1) for v in g["box"]], "rows": [[round(v, 1) for v in b] for b in g["rows"]],
                                     "cols": [[round(v, 1) for v in b] for b in g["cols"]], "header_rows": g["header_rows"],
                                     "cells": [{**c, "box": [round(v, 1) for v in c["box"]]} for c in g["cells"]]}, ensure_ascii=False) + "\n")
                n_img += 1
                n_rect += int(g["rectified"])
                if a.preview and n_prev < a.preview:
                    cv2.imwrite(str(prev_dir / f"{stem}.jpg"), draw_preview(crop, lines), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    n_prev += 1
                if a.max_per_split and n_img >= a.max_per_split:
                    break
            if a.max_per_split and n_img >= a.max_per_split:
                break
    stats = {"split": split, "crops": n_img, "rectified": n_rect, "skipped": n_skip, "boxes": {CLASSES[k]: v for k, v in sorted(cls_count.items())}}
    print(json.dumps(stats), flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="docgen v5 root with {split}/manifest.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--long-side", type=int, default=800)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--rectify", choices=["auto", "never", "always"], default="auto")
    ap.add_argument("--skew-keep-p", type=float, default=0.25, help="auto: keep a slightly skewed table unrectified with this probability")
    ap.add_argument("--skew-max-deg", type=float, default=3.0, help="auto: always rectify beyond this rotation")
    ap.add_argument("--min-rows", type=int, default=2)
    ap.add_argument("--min-cols", type=int, default=2)
    ap.add_argument("--max-per-split", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, help="draw this many label previews per split into OUT/preview")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    stats = []
    for split in a.splits.split(","):
        st = build_split(a.src, a.out, split.strip(), a, rng)
        if st:
            stats.append(st)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (a.out / "tsr.yaml").write_text(f"path: {a.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n{names}\n", encoding="utf-8")
    (a.out / "build_stats.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "splits": stats}, indent=1), encoding="utf-8")
    print(f"wrote {a.out / 'tsr.yaml'}")
    return stats


if __name__ == "__main__":
    main()
