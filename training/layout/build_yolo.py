#!/usr/bin/env python3
"""Assemble the v2 YOLO layout dataset (21 classes, training/taxonomy.py) from all sources.

    python -m training.layout.build_yolo --out work/data/layout_v2 [--docgen DIR ...] [--extra-manifest NAME=PATH ...]

Sources (train):
  * docgen v2 synthetic pages (canonical manifests, v2 ids)                       --docgen <dir with train/ val/>
  * DocLayNet v2 (real, 11 DocLayNet classes + page_number)                       --doclaynet
  * real Khmer PDF pages (silver heuristic labels, old 12-class ids -> v2)        --khm  (+ one degraded copy each)
  * real SignverOD / stamp scans from the v1 round-3 set (old ids -> v2)          --round3
  * any extra canonical v2 manifest (e.g. researched datasets)                    --extra-manifest name=path
  * degraded copies of N DocLayNet pages (scan/photo look, labels warped)
Validation (checkpoint selection only): docgen val, DocLayNet val, Khmer val. Test sets are never touched here.
Images are hard-linked when unchanged (no extra disk).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

OLD = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation")
sys.path.insert(0, str(OLD))
from det import degrade  # noqa: E402

from training.taxonomy import CLASSES, CLS  # noqa: E402

OLD_CLASSES = ["text", "title", "list", "table", "figure", "signature", "stamp", "logo", "qr_barcode", "handwriting",
               "header_footer", "formula"]
KHM_OLD_TO_V2 = {"text": "text", "title": "section_header", "list": "list", "table": "table", "figure": "figure",
                 "signature": "signature", "stamp": "stamp", "logo": "logo", "handwriting": "handwriting", "formula": "formula"}


def map_old(objs, W, H, title_as="section_header"):
    """old 12-class boxes -> v2 (qr vs barcode by aspect ratio, header vs footer by position)."""
    out = []
    for c, x0, y0, x1, y1 in objs:
        name = OLD_CLASSES[int(c)]
        if name == "qr_barcode":
            w, h = x1 - x0, y1 - y0
            name = "qr_code" if 0.6 < w / max(1, h) < 1.6 else "barcode"
        elif name == "header_footer":
            name = "header" if (y0 + y1) / 2 < H / 2 else "footer"
        elif name == "title":
            name = title_as
        else:
            name = KHM_OLD_TO_V2[name]
        out.append([CLS[name], x0, y0, x1, y1])
    return out


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_manifest(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.counts = {"train": 0, "val": 0}
        self.cls_n = {s: [0] * len(CLASSES) for s in self.counts}
        self.src_n = {}

    def emit(self, split, name, objs, W, H, src=None, img=None, tag=""):
        dst = self.out / "images" / split / f"{name}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if img is not None:
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            H, W = img.shape[:2]
        else:
            link(src, dst)
        lines = []
        for c, x0, y0, x1, y1 in objs:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            lines.append(f"{int(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            self.cls_n[split][int(c)] += 1
        lp = self.out / "labels" / split / f"{name}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(lines))
        self.counts[split] += 1
        self.src_n[(split, tag)] = self.src_n.get((split, tag), 0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docgen", type=Path, nargs="*", default=[])
    ap.add_argument("--doclaynet", type=Path, default=Path("/home/hello/ocr/docsys/work/data/doclaynet_v2"))
    ap.add_argument("--khm", type=Path, default=Path("/home/hello/ocr/khm_repo_clone/det_data/real/khm_pdf"))
    ap.add_argument("--round3", type=Path, default=Path("/home/hello/ocr/khm_repo_clone/det_data/round3"))
    ap.add_argument("--extra-manifest", nargs="*", default=[], help="name=path/to/manifest.jsonl (v2 ids, train)")
    ap.add_argument("--n-dln-degraded", type=int, default=1500)
    ap.add_argument("--val-docgen", type=int, default=300)
    ap.add_argument("--val-dln", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.out.exists():
        shutil.rmtree(a.out)
    w = Writer(a.out)

    def canon(root: Path, split: str, tag: str, n=None, old=False):
        recs = [r for r in read_manifest(root / split / "manifest.jsonl") if r.get("objs_ok", True)]
        random.Random(1).shuffle(recs)
        for r in recs[:n]:
            objs = map_old(r["objs"], r["w"], r["h"]) if old else r["objs"]
            w.emit(split, f"{tag}_{Path(r['image']).stem}", objs, r["w"], r["h"], src=root / split / r["image"], tag=tag)
        return recs[:n]

    # ---- validation
    for i, d in enumerate(a.docgen):
        canon(d, "val", f"gen{i}", a.val_docgen)
    canon(a.doclaynet, "val", "dln", a.val_dln)
    canon(a.khm, "val", "khm", None, old=True)

    # ---- train
    for i, d in enumerate(a.docgen):
        canon(d, "train", f"gen{i}")
    dln = canon(a.doclaynet, "train", "dln")
    khm = canon(a.khm, "train", "khm", None, old=True)
    for spec in a.extra_manifest:
        name, p = spec.split("=", 1)
        p = Path(p)
        for r in read_manifest(p):
            if not (r.get("objs_ok", True) and r["objs"]):
                continue
            stem = f"{name}_{Path(r['image']).stem}"
            if r.get("ignore_boxes"):  # ambiguous regions: paint them out with the paper colour (YOLO has no ignore label)
                img = cv2.imread(str(p.parent / r["image"]))
                paper = np.median(img.reshape(-1, 3), axis=0)
                for b in r["ignore_boxes"]:
                    x0, y0, x1, y1 = (int(round(v)) for v in b[-4:])
                    img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = paper
                w.emit("train", stem, r["objs"], 0, 0, img=img, tag=name)
            else:
                w.emit("train", stem, r["objs"], r["w"], r["h"], src=p.parent / r["image"], tag=name)
    # real signature / stamp scans (v1 round-3, YOLO labels with old ids)
    for img in sorted((a.round3 / "images/train").glob("*.jpg")):
        if not img.name.startswith(("sig_", "stamp_")):
            continue
        im = cv2.imread(str(img), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        H, W = im.shape[0] * 8, im.shape[1] * 8  # only the aspect matters for normalised labels
        objs = []
        for ln in (a.round3 / "labels/train" / (img.stem + ".txt")).read_text().split("\n"):
            if ln.strip():
                c, cx, cy, bw, bh = map(float, ln.split())
                objs.append([c, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H])
        w.emit("train", f"r3_{img.stem}", map_old(objs, W, H), W, H, src=img, tag="real_sig_stamp")
    print("clean:", w.counts, flush=True)

    # ---- degraded copies of real pages (Khmer all, DocLayNet subset)
    rng.shuffle(dln)
    for tag, root, r, old in [("khm", a.khm, r, True) for r in khm] + [("dln", a.doclaynet, r, False) for r in dln[: a.n_dln_degraded]]:
        img = cv2.cvtColor(cv2.imread(str(root / "train" / r["image"])), cv2.COLOR_BGR2RGB)
        objs = map_old(r["objs"], r["w"], r["h"]) if old else r["objs"]
        d, _, objs = degrade.apply(img, np.zeros((0, 4, 2), np.float32), [list(o) for o in objs], rng)
        w.emit("train", f"{tag}_{Path(r['image']).stem}_deg", objs, 0, 0, img=cv2.cvtColor(d, cv2.COLOR_RGB2BGR), tag=tag + "_deg")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (a.out / "layout.yaml").write_text(f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    stats = {"counts": w.counts, "by_source": {f"{k[0]}/{k[1]}": v for k, v in sorted(w.src_n.items())},
             "boxes": {s: dict(zip(CLASSES, w.cls_n[s])) for s in w.cls_n}}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
