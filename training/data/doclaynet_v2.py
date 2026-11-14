#!/usr/bin/env python3
"""DocLayNet-base (CDLA-Permissive-1.0) -> canonical page manifest with the v2 taxonomy.

    python -m training.data.doclaynet_v2 --zip .../dataset_base.zip --out work/data/doclaynet_v2 --workers 8

Reads straight from the zip (never extracted). Per page: rendered from the PDF (long side 1280), v2 layout boxes
(training/taxonomy.py DOCLAYNET_MAP; header/footer boxes whose text is only a page number -> page_number;
text-less pictures that the annotation omits are recovered from the PDF), tight printed text-line quads + their text
from the PDF text layer (line_texts, used for English recognition crops).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import re
import sys
import zipfile
from pathlib import Path

import cv2

OLD = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation")
sys.path.insert(0, str(OLD))
import fitz  # noqa: E402
from det.pdf_pages import _iou, figure_boxes, render, text_lines  # noqa: E402

from training.taxonomy import CLS, DOCLAYNET_MAP  # noqa: E402

PAGENUM_RE = re.compile(r"^\W*(?:page|p\.|seite|pg\.?)?\s*[0-9ivxlcIVXLC]{1,5}(?:\s*(?:of|/|von|sur|de)\s*\d{1,5})?\W*$", re.I)


def job(a):
    zpath, names, split, out, long_side = a
    z = zipfile.ZipFile(zpath)
    recs = []
    for base in names:
        try:
            ann = json.loads(z.read(base.replace("/pdfs/", "/annotations/") + ".json"))
            doc = fitz.open(stream=z.read(base + ".pdf"), filetype="pdf")
            page = doc[0]
            if page.rotation != 0:
                continue
            img, s = render(page, long_side)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            quads, _, texts = text_lines(page, gray, s)
            H, W = img.shape[:2]
            cw, ch = ann["metadata"]["coco_width"], ann["metadata"]["coco_height"]
            boxes, btext = {}, {}
            for it in ann["form"]:
                if it["category"] not in DOCLAYNET_MAP:
                    continue
                boxes.setdefault(it["id_box"], (it["category"], it["box"]))
                btext[it["id_box"]] = btext.get(it["id_box"], "") + " " + (it.get("text") or "")
            objs = []
            for bid, (cat, (x, y, w, h)) in boxes.items():
                c = DOCLAYNET_MAP[cat]
                if c in ("header", "footer") and PAGENUM_RE.match(btext[bid].strip() or "x"):
                    c = "page_number"
                objs.append([CLS[c], x / cw * W, y / ch * H, (x + w) / cw * W, (y + h) / ch * H])
            have = [o for o in objs if o[0] in (CLS["figure"], CLS["table"])]
            for f in figure_boxes(page, s, W, H, []):
                if not any(_iou(f, o[1:5]) > 0.3 for o in have):
                    objs.append([CLS["figure"], *f])
            name = Path(base).name + ".jpg"
            cv2.imwrite(str(out / split / "images" / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            md = ann["metadata"]
            recs.append(dict(image=f"images/{name}", w=W, h=H, source="doclaynet", doc_type=md.get("doc_category", ""), lang="en",
                             lines=[[round(float(v), 1) for v in q] for q in quads], line_texts=texts,
                             objs=[[o[0]] + [round(float(v), 1) for v in o[1:]] for o in objs],
                             lines_ok=len(quads) >= 3, objs_ok=bool(objs)))
        except Exception as e:
            print(f"   {base[-12:]}: {e!r}"[:160], flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    allnames = z.namelist()
    ctx = mp.get_context("fork")
    for split in ("val", "test", "train"):
        pdfs = sorted(x[:-4] for x in allnames if f"/{split}/pdfs/" in x and x.endswith(".pdf"))
        random.Random(3).shuffle(pdfs)
        (a.out / split / "images").mkdir(parents=True, exist_ok=True)
        chunks = [pdfs[i::a.workers * 8] for i in range(a.workers * 8)]
        recs = []
        with ctx.Pool(a.workers) as pool:
            for r in pool.imap_unordered(job, [(str(a.zip), c, split, a.out, a.long_side) for c in chunks if c]):
                recs += r
        recs.sort(key=lambda r: r["image"])
        with open(a.out / split / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"doclaynet {split}: {len(recs)} pages, {sum(len(r['lines']) for r in recs)} lines, "
              f"{sum(len(r['objs']) for r in recs)} boxes", flush=True)


if __name__ == "__main__":
    main()
