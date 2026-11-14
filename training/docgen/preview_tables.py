"""Draw the v5 table labels (tables[]: rows / cols / cells / header rows / spans / line_ids) on pages, to check by eye.

    python -m training.docgen.preview_tables --manifest DIR/manifest.jsonl --out PREVIEW_DIR --n 30 [--seed 0] [--doc-type X]
                                             [--only-tables] [--max-side 1600]
Row bands: translucent alternating blue / yellow; column bands: thin cyan borders; header rows: translucent red;
cells: green boxes, spanning cells (rs>1 or cs>1) thick magenta with "rs x cs"; projected row header (group) rows:
orange; every line_id of a cell is tied to the cell by a small dot at the line centre; printed lines: thin green quads;
other layout boxes: grey. The top-left caption shows doc_type, lang, doc_kind, capture mode and per-table style /
borders / size. --only-tables keeps only pages with at least one table.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from ..taxonomy import CLASSES


def draw(rec, root: Path, max_side=1600):
    img = cv2.imread(str(root / rec["image"]))
    if img is None:
        return None
    over = img.copy()
    for c, x0, y0, x1, y1 in rec.get("objs", []):
        col = (30, 160, 110) if CLASSES[c] == "table" else (170, 170, 170)
        cv2.rectangle(img, (int(x0), int(y0)), (int(round(x1)), int(round(y1))), col[::-1], 2 if CLASSES[c] == "table" else 1)
    for q in rec.get("lines", []):
        cv2.polylines(img, [np.int32(np.round(np.float32(q).reshape(4, 2)))], True, (0, 200, 0), 1, cv2.LINE_AA)
    lines = rec.get("lines", [])
    for t in rec.get("tables", []):
        for i, (y0, y1) in enumerate(t["rows"]):
            col = (255, 200, 120) if i % 2 == 0 else (120, 220, 255)
            if i < t["header_rows"]:
                col = (90, 90, 230)
            cv2.rectangle(over, (int(t["box"][0]), int(y0)), (int(t["box"][2]), int(y1)), col, -1)
        for (x0, x1) in t["cols"]:
            cv2.rectangle(img, (int(x0), int(t["box"][1])), (int(x1), int(t["box"][3])), (200, 200, 0), 1)
        bx = t["box"]
        cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 0, 255), 3)
        for c in t["cells"]:
            x0, y0, x1, y1 = [int(v) for v in c["box"]]
            span = c["rs"] > 1 or c["cs"] > 1
            grp = c.get("kind") == "group"
            col = (0, 120, 255) if grp else ((255, 0, 255) if span else (0, 170, 0))
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), col, 2 if (span or grp) else 1)
            if span:
                cv2.putText(img, f"{c['rs']}x{c['cs']}", (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            for li in c.get("line_ids", []):
                if li < len(lines):
                    q = np.float32(lines[li]).reshape(4, 2)
                    lx, ly = int(q[:, 0].mean()), int(q[:, 1].mean())
                    cv2.circle(img, (lx, ly), 3, (0, 0, 200), -1)
                    cv2.line(img, (lx, ly), (cx, cy), (0, 0, 200), 1)
            if c.get("header"):
                cv2.putText(img, "H", (x1 - 12, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            if c["text"] == "" and not c.get("line_ids"):
                cv2.line(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (160, 160, 160), 1)
    img = cv2.addWeighted(over, 0.22, img, 0.78, 0)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    info = [f'{rec.get("doc_type")} {rec.get("lang")} kind={rec.get("doc_kind")} {rec.get("aug", {}).get("mode", "clean")}']
    for i, t in enumerate(rec.get("tables", [])):
        spans = sum(1 for c in t["cells"] if c["rs"] > 1 or c["cs"] > 1)
        empt = sum(1 for c in t["cells"] if not c["text"])
        info.append(f'T{i}: {t.get("style")}/{t["borders"]} {len(t["rows"])}x{len(t["cols"])} hdr={t["header_rows"]} spans={spans} empty={empt} sheet={t["sheet"]} excel={t.get("excel")}')
    for k, s_ in enumerate(info):
        cv2.putText(img, s_, (8, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--doc-type", default=None)
    ap.add_argument("--only-tables", action="store_true")
    a = ap.parse_args()
    recs = [json.loads(ln) for ln in open(a.manifest, encoding="utf-8") if ln.strip()]
    if a.doc_type:
        recs = [r for r in recs if r.get("doc_type") == a.doc_type]
    if a.only_tables:
        recs = [r for r in recs if r.get("tables")]
    random.Random(a.seed).shuffle(recs)
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in recs[: a.n]:
        im = draw(r, a.manifest.parent, a.max_side)
        if im is not None:
            cv2.imwrite(str(a.out / (Path(r["image"]).stem + "_tab.jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
    print(f"{n} previews -> {a.out}")


if __name__ == "__main__":
    main()
