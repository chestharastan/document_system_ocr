"""Build the layout fine-tune set (v3): docgen v3 pages (22 classes incl. decoration) + a hard-linked subset of the
round-2 YOLO set so the old classes / real pages are not forgotten.

    python -m training.layout.build_finetune --docgen /home/hello/ocr/docsys/work/data/docgen_v3 \
        --base /home/hello/ocr/docsys/work/data/layout_v2_r2 --out /home/hello/ocr/docsys/work/data/layout_v3 \
        --base-train 8000 --base-val 600
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import yaml

NAMES = ["text", "title", "section_header", "list", "table", "figure", "caption", "header", "footer", "page_number", "footnote",
         "formula", "signature", "stamp", "logo", "qr_code", "barcode", "photo", "handwriting", "checkbox", "form_field", "decoration"]


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def add_docgen(split_dir: Path, out: Path, split: str, prefix: str) -> int:
    n = 0
    with open(split_dir / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("objs_ok", True):
                continue
            img = split_dir / r["image"]
            if not img.exists():
                continue
            name = f"{prefix}_{Path(r['image']).stem}"
            link(img, out / "images" / split / f"{name}{img.suffix}")
            W, H = float(r["w"]), float(r["h"])
            rows = []
            for cls, x0, y0, x1, y1 in r["objs"]:
                if cls >= len(NAMES) or x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                rows.append(f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            n += 1
    return n


def add_base(base: Path, out: Path, split: str, k: int, rng: random.Random) -> int:
    imgs = sorted((base / "images" / split).iterdir())
    rng.shuffle(imgs)
    n = 0
    for img in imgs[:k]:
        lab = base / "labels" / split / (img.stem + ".txt")
        if not lab.exists():
            continue
        link(img, out / "images" / split / img.name)
        link(lab, out / "labels" / split / lab.name)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docgen", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-train", type=int, default=8000)
    ap.add_argument("--base-val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for s in ("train", "val"):
        (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    stats = {"docgen_train": add_docgen(a.docgen / "train", a.out, "train", "dg3"), "docgen_val": add_docgen(a.docgen / "val", a.out, "val", "dg3"),
             "base_train": add_base(a.base, a.out, "train", a.base_train, rng), "base_val": add_base(a.base, a.out, "val", a.base_val, rng)}
    (a.out / "layout.yaml").write_text(yaml.safe_dump({"path": str(a.out), "train": "images/train", "val": "images/val",
                                                        "names": {i: n for i, n in enumerate(NAMES)}}, sort_keys=False))
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
