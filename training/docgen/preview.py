"""Draw layout boxes (taxonomy colours + class names) and text lines on pages.

    python -m training.docgen.preview --manifest DIR/manifest.jsonl --out PREVIEW_DIR --n 40 [--seed 0] [--max-side 1400]
Printed lines: thin green quads; handwriting lines: orange; formula lines: magenta.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from ..taxonomy import CLASSES, COLORS


def draw(rec, root: Path, max_side=1400):
    img = cv2.imread(str(root / rec["image"]))
    if img is None:
        return None
    for key, col in (("lines", (0, 200, 0)), ("hw_lines", (0, 140, 255)), ("formula_lines", (255, 0, 255))):
        for q in rec.get(key, []):
            cv2.polylines(img, [np.int32(np.round(np.float32(q).reshape(4, 2)))], True, col, 1, cv2.LINE_AA)
    for c, x0, y0, x1, y1 in rec.get("objs", []):
        name = CLASSES[c]
        r, g, b = COLORS[name]
        cv2.rectangle(img, (int(x0), int(y0)), (int(round(x1)), int(round(y1))), (b, g, r), 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yt = int(y0) - 2 if y0 > th + 4 else int(y0) + th + 2
        cv2.rectangle(img, (int(x0), yt - th - 2), (int(x0) + tw + 2, yt + 2), (b, g, r), -1)
        cv2.putText(img, name, (int(x0) + 1, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    lab = f'{rec.get("doc_type")} {rec.get("lang")} {rec.get("aug", {}).get("mode", "clean")}'
    cv2.putText(img, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--doc-type", default=None)
    a = ap.parse_args()
    recs = [json.loads(ln) for ln in open(a.manifest, encoding="utf-8") if ln.strip()]
    if a.doc_type:
        recs = [r for r in recs if r.get("doc_type") == a.doc_type]
    random.Random(a.seed).shuffle(recs)
    a.out.mkdir(parents=True, exist_ok=True)
    for r in recs[: a.n]:
        im = draw(r, a.manifest.parent, a.max_side)
        if im is not None:
            cv2.imwrite(str(a.out / (Path(r["image"]).stem + "_prev.jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{min(a.n, len(recs))} previews -> {a.out}")


if __name__ == "__main__":
    main()
