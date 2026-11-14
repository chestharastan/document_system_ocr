#!/usr/bin/env python3
"""CER/WER of ONNX recognisers on line folders (metadata.jsonl + images).

    python -m training.rec.eval_folder --model v1=path/rec.onnx --folder name=work/data/rec_doclaynet/val [--limit 3000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

OLD = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation")
sys.path.insert(0, str(OLD))
from det.recognizer import Recognizer  # noqa: E402


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(preds, refs):
    ce = sum(lev(p, r) for p, r in zip(preds, refs))
    cn = sum(len(r) for r in refs)
    we = sum(lev(p.split(), r.split()) for p, r in zip(preds, refs))
    wn = sum(len(r.split()) for r in refs)
    exact = sum(p == r for p, r in zip(preds, refs))
    return dict(cer=100 * ce / max(1, cn), wer=100 * we / max(1, wn), line_acc=100 * exact / max(1, len(refs)), n=len(refs))


def load(folder: Path, limit: int, seed=0):
    rows = [json.loads(l) for l in open(folder / "metadata.jsonl", encoding="utf-8")]
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    if (folder / "lines.bin").exists():  # packed format (training.data.line_crops)
        import numpy as np
        buf = (folder / "lines.bin").read_bytes()
        ims = [cv2.imdecode(np.frombuffer(buf[r["offset"]:r["offset"] + r["nbytes"]], np.uint8), cv2.IMREAD_GRAYSCALE) for r in rows]
    else:
        ims = [cv2.imread(str(folder / r["file_name"]), cv2.IMREAD_GRAYSCALE) for r in rows]
    return ims, [r["text"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--folder", action="append", required=True)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    res = {}
    for fspec in a.folder:
        fname, fpath = fspec.split("=", 1)
        ims, refs = load(Path(fpath), a.limit)
        for mspec in a.model:
            mname, mpath = mspec.split("=", 1)
            rec = Recognizer(mpath, str(Path(mpath).parent / "vocab.json"))
            preds = [p if isinstance(p, str) else p[0] for p in rec.recognize(ims)]
            s = score(preds, refs)
            res.setdefault(fname, {})[mname] = s
            print(f"{fname:20s} {mname:12s} CER {s['cer']:6.2f}  WER {s['wer']:6.2f}  line-acc {s['line_acc']:5.1f}  n={s['n']}", flush=True)
            for p, r in list(zip(preds, refs))[: a.show]:
                if p != r:
                    print(f"    ref: {r}\n    hyp: {p}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
