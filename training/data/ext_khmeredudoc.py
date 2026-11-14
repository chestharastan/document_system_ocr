#!/usr/bin/env python3
"""KhmerEduDoc (CC-BY-NC-4.0, NON-COMMERCIAL, EVAL ONLY) test_setA / test_setB -> work/data/eval_nc/khmeredudoc/<split>.

    python -m training.data.ext_khmeredudoc --raw work/data/_raw/khmeredudoc/data --out work/data/eval_nc/khmeredudoc

The dataset has NO boxes of any kind (no layout, no line quads). Its only label is a human-verified page transcription
in reading order (tables as HTML, math as LaTeX). So: lines_ok false, objs_ok false, no lines/objs, and the page text
goes into rec["page_text"] for end-to-end page CER of the 3-stage pipeline (strip HTML tags / LaTeX before scoring).
Extra fields: book_id, page, content_type (setA: text_only | text_with_latex), label_source (always "human" here).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from training.data.ext_common import Manifest, decode, fit, save_jpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for split in ("test_setA", "test_setB"):
        out = a.out / split
        man = Manifest(out / "manifest.jsonl")
        for i, r in enumerate(pq.read_table(a.raw / f"{split}-00000-of-00001.parquet").to_pylist()):
            img, _ = fit(decode(r["image"]["bytes"]))
            H, W = img.shape[:2]
            name = f"{split}_{i:03d}.jpg"
            save_jpg(img, out / "images" / name)
            man.add(dict(image=f"images/{name}", w=W, h=H, source="khmeredudoc", doc_type="textbook", lang="km", nc=True,
                         objs=[], lines_ok=False, objs_ok=False, page_text=r["text"], book_id=r["book_id"] or "",
                         page=r["page"], content_type=r["content_type"] or "", label_source=r["label_source"]))
        man.close()
        print(split, man.n)


if __name__ == "__main__":
    main()
