#!/usr/bin/env python3
"""Owner's labelling-tool export (images_table.csv + annotations.csv + images/) -> packed recognition lines.

    python -m training.data.labeltool_crops --src /run/media/hello/THAOCR/ocr-train-data --out ../work/data/rec_hand_real_owner

Boxes are axis-aligned (x, y, width, height) in original image pixels. Pages are split train/val/test by image id
(hash), so no page is in two splits. Lines whose text is empty, out of vocabulary, or implausibly short/long for the
box are skipped (counted in stats.json).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from training.data.line_crops import VSET, norm_text, crop_line


def split_of(image_id: str, val: float, test: float) -> str:
    h = int(hashlib.md5(f"owner-{image_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.08)
    ap.add_argument("--test", type=float, default=0.08)
    a = ap.parse_args()

    images = {r["id"]: r for r in csv.DictReader(open(a.src / "images_table.csv", encoding="utf-8"))}
    by_img = defaultdict(list)
    for r in csv.DictReader(open(a.src / "annotations.csv", encoding="utf-8")):
        by_img[r["image_id"]].append(r)

    stats = Counter()
    writers = {}
    for s in ("train", "val", "test"):
        (a.out / s).mkdir(parents=True, exist_ok=True)
        writers[s] = [open(a.out / s / "lines.bin", "wb"), open(a.out / s / "metadata.jsonl", "w", encoding="utf-8"), 0]
    pages = Counter()
    for iid, anns in sorted(by_img.items(), key=lambda kv: int(kv[0])):
        im = images.get(iid)
        if im is None:
            stats["no_image_row"] += len(anns)
            continue
        img = cv2.imread(str(a.src / "images" / im["filename"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["unreadable"] += len(anns)
            continue
        sx, sy = img.shape[1] / float(im["width"]), img.shape[0] / float(im["height"])
        sp = split_of(iid, a.val, a.test)
        pages[sp] += 1
        for r in anns:
            t = norm_text(r["text"].replace("\n", " "))
            if len(t) < 1:
                stats["empty"] += 1
                continue
            bad = {c for c in t if c not in VSET and c != " "}
            if bad:
                stats["oov"] += 1
                stats.update("oovchar:" + c for c in bad)
                continue
            x, y, w, h = (float(r[k]) for k in ("x", "y", "width", "height"))
            x, w, y, h = x * sx, w * sx, y * sy, h * sy
            if h < 8 or w < 8:
                stats["tiny"] += 1
                continue
            # chars per box-height unit: handwritten Khmer ~0.6-3 (codepoints incl. subscripts)
            ratio = len(t) / max(w / h, 0.3)
            if ratio < 0.25 or ratio > 8:
                stats["ratio_outlier"] += 1
                continue
            crop = crop_line(img, [x, y, x + w, y, x + w, y + h, x, y + h], pad_y=0.04, pad_x=0.02)
            if crop is None:
                stats["tiny"] += 1
                continue
            nh = 48
            nw = max(8, int(round(crop.shape[1] * nh / crop.shape[0])))
            crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            fb, fm, off = writers[sp]
            b = buf.tobytes()
            fb.write(b)
            fm.write(json.dumps({"offset": off, "nbytes": len(b), "text": t, "split": sp, "source": "owner_labeltool",
                                 "doc_type": "handwriting_real", "level": "real", "kind": "hand",
                                 "width": nw, "height": nh, "page": im["original_filename"],
                                 "project": im["project_id"], "ann_id": r["id"]}, ensure_ascii=False) + "\n")
            writers[sp][2] = off + len(b)
            stats["kept_" + sp] += 1
    for fb, fm, _ in writers.values():
        fb.close(), fm.close()
    out = {"pages": dict(pages), **dict(stats.most_common())}
    json.dump(out, open(a.out / "stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
