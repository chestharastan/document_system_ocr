#!/usr/bin/env python3
"""Recogniser v2 trainer = the proven CRNN-CTC trainer of Khm-Dataset-Creation (scripts/train.py) plus:
  * mix part kind "packed": <path>/<split>/lines.bin + metadata.jsonl (offset/nbytes/text), written by training.data.line_crops
  * extra architectures (training.rec.archs) registered into khmerocr.model.ARCHS
All other flags are the old trainer's (python -m training.rec.train --help).
"""
from __future__ import annotations

import json
import random
import runpy
import sys
from pathlib import Path

OLD = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation")
sys.path.insert(0, str(OLD))

import khmerocr.mixed_dataset as md  # noqa: E402
import khmerocr.model as km  # noqa: E402

_add_folder = md.add_folder


def add_packed(ds, spec, root, seed, group=40000):
    base = (root / spec["path"]).resolve() / spec.get("split", "train")
    rows = [json.loads(l) for l in open(base / "metadata.jsonl", encoding="utf-8")]
    if spec.get("sources"):
        rows = [r for r in rows if r.get("source") in spec["sources"]]
    random.Random(seed).shuffle(rows)
    if spec.get("n"):
        rows = rows[: spec["n"]]
    buf = (base / "lines.bin").read_bytes()
    tag = f"packed:{base.parent.name}"
    for s in range(0, len(rows), group):
        recs = []
        for r in rows[s : s + group]:
            text = md.normalize_khmer(r["text"])
            if not text or not all(c in ds.codec.idx for c in text):
                continue
            recs.append((buf[r["offset"] : r["offset"] + r["nbytes"]], text, md._width(r["width"], r.get("height", 48))))
        ds.add(recs, tag)


def dispatch(ds, spec, root, seed, *a, **k):
    if spec.get("kind") == "packed":
        return add_packed(ds, spec, root, seed)
    return _add_folder(ds, spec, root, seed, *a, **k)


md.add_folder = dispatch

# wider variant: same CNN/BiLSTM design, more channels (~8M params)
km.ARCHS["L"] = dict(c=(64, 128, 256, 320), proj=384, hidden=288, layers=2)

if __name__ == "__main__":
    import os
    if float(os.environ.get("STAMP_AUG", "0") or 0) > 0:  # stamp-over-text overlays on the training crops
        from training.rec.stamp_aug import patch_mixed_lines
        patch_mixed_lines(float(os.environ["STAMP_AUG"]))
        print(f"[aug] stamp overlays on {float(os.environ['STAMP_AUG']):.0%} of the training crops", flush=True)
    sys.argv[0] = str(OLD / "scripts/train.py")
    runpy.run_path(str(OLD / "scripts/train.py"), run_name="__main__")
