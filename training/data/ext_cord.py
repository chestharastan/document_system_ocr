#!/usr/bin/env python3
"""CORD v2 (CC-BY-4.0) receipts -> canonical page manifest in work/data/ext/cord/{train,validation,test}.

    python -m training.data.ext_cord --raw work/data/_raw/cord --out work/data/ext/cord

Downloads one parquet at a time into --raw (naver-clova-ix/cord-v2 data/*.parquet), converts, deletes it.
Images (up to 4096 px PNG) are re-encoded to JPEG q88, long side 1600.

Labels:
  lines/line_texts: CORD `valid_line` = one semantic field (e.g. menu.nm, total.total_price) whose words carry a quad,
      text, is_key and row_id. A text line = the words of one valid_line on one row_id, further split where the
      horizontal gap between consecutive words > 1.5 x word height (so "Total:      28,000" -> 2 lines, matching how a
      line detector sees a receipt). Quad = cv2.minAreaRect of the word corners (handles slightly rotated photos),
      ordered tl,tr,br,bl. Text = words joined by a space.
      CORD only annotates the transaction part: store name/address/phone, footer slogans, dates and cashier ids are
      mostly NOT in valid_line (and some are blurred). So lines are NOT exhaustive -> lines_ok false; still useful for
      recognition crops (line_texts are real transcriptions) and for line detection with an ignore-outside strategy.
  objs: form_field (20) = union box of a valid_line that contains BOTH key words (is_key=1, e.g. "Total:", "Cash:")
      and value words. Lines with only a value (menu items) are not boxed. Only form_field is labelled ->
      partial_classes [20]; objs_ok true only in that partial sense. Logos, tables (menu), text blocks unlabelled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from training.data.ext_common import Manifest, decode, fit, obj, r1, save_jpg, stats
from training.taxonomy import CLS

FILES = {
    "train": [f"data/train-0000{i}-of-00004-{h}.parquet" for i, h in
              enumerate(["b4aaeceff1d90ecb", "7dbbe248962764c5", "688fe1305a55e5cc", "2d0cd200555ed7fd"])],
    "validation": ["data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"],
    "test": ["data/test-00000-of-00001-9c204eb3f4e11791.parquet"],
}


def wpts(w, s):
    q = w["quad"]
    return np.array([[q["x1"], q["y1"]], [q["x2"], q["y2"]], [q["x3"], q["y3"]], [q["x4"], q["y4"]]], np.float32) * s


def quad_of(pts):
    """min-area rect of points -> [tl, tr, br, bl] flattened."""
    box = cv2.boxPoints(cv2.minAreaRect(pts.reshape(-1, 2)))
    c = box.mean(0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    box = box[np.argsort(ang)]  # order: starts at top-left-ish (angle ~ -135deg) for image coords
    i = int(np.argmin(box.sum(1)))
    box = np.roll(box, -i, 0)
    return [r1(v) for v in box.reshape(-1)]


def page(gt, s):
    lines, texts, objs = [], [], []
    for vl in gt["valid_line"]:
        ws = [w for w in vl["words"] if w["text"].strip()]
        if not ws:
            continue
        # split by row, then by big gaps
        rows = {}
        for w in ws:
            rows.setdefault(w["row_id"], []).append(w)
        for row in rows.values():
            row.sort(key=lambda w: wpts(w, s)[:, 0].min())
            seg = [row[0]]
            for w in row[1:]:
                p, c = wpts(seg[-1], s), wpts(w, s)
                hgt = max(np.ptp(p[:, 1]), np.ptp(c[:, 1]), 1.0)
                if c[:, 0].min() - p[:, 0].max() > 1.5 * hgt:
                    lines.append(quad_of(np.concatenate([wpts(x, s) for x in seg])))
                    texts.append(" ".join(x["text"] for x in seg))
                    seg = [w]
                else:
                    seg.append(w)
            lines.append(quad_of(np.concatenate([wpts(x, s) for x in seg])))
            texts.append(" ".join(x["text"] for x in seg))
        keys = [w for w in ws if w["is_key"]]
        if keys and len(keys) < len(ws):
            p = np.concatenate([wpts(w, s) for w in ws])
            objs.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
    return lines, texts, objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    for split, files in FILES.items():
        out = a.out / split
        man = Manifest(out / "manifest.jsonl")
        for fn in files:
            path = hf_hub_download("naver-clova-ix/cord-v2", fn, repo_type="dataset", local_dir=str(a.raw))
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                for r in pf.read_row_group(rg).to_pylist():
                    gt = json.loads(r["ground_truth"])
                    img = decode(r["image"]["bytes"])
                    h0, w0 = img.shape[:2]
                    ms = gt["meta"]["image_size"]
                    if (ms["width"], ms["height"]) != (w0, h0):
                        print("size mismatch", gt["meta"], w0, h0)
                    img, s = fit(img)
                    H, W = img.shape[:2]
                    lines, texts, fb = page(gt, s)
                    objs = [o for o in (obj(CLS["form_field"], b, W, H) for b in fb) if o]
                    name = f"{split}_{gt['meta']['image_id']:04d}.jpg"
                    save_jpg(img, out / "images" / name)
                    man.add(dict(image=f"images/{name}", w=W, h=H, source="cord", doc_type="receipt", lang="en",
                                 lines=lines, line_texts=texts, objs=objs, lines_ok=False, objs_ok=True,
                                 partial_classes=[CLS["form_field"]]))
            del pf
            if not a.keep_raw:
                os.remove(path)
        man.close()
        print(split, json.dumps(stats(out / "manifest.jsonl")), flush=True)


if __name__ == "__main__":
    main()
