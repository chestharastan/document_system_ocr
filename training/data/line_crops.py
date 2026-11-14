#!/usr/bin/env python3
"""Page manifests (lines + line_texts) -> packed recognition lines (<out>/<split>/lines.bin + metadata.jsonl with offset/nbytes, height-48 gray JPEG q85).

    python -m training.data.line_crops --manifest work/data/doclaynet_v2/val/manifest.jsonl --out work/data/rec_doclaynet --split val

Text is normalised (typographic quotes/dashes/ligatures -> ASCII, Khmer via normalize_khmer) and lines with a character
outside the recogniser vocabulary are skipped (counted). Read with training.rec.data (kind "packed").
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

OLD = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation")
sys.path.insert(0, str(OLD))
from khmerocr.normalize import normalize_khmer  # noqa: E402

VOCAB = json.load(open(OLD / "runs/crnn_m/export/vocab.json"))["itos"]
VSET = set(VOCAB[1:])
TR = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'", "′": "'",
                    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-", " ": " ", " ": " ",
                    " ": " ", "·": "•", "●": "•", "▪": "•", "×": "x"})
LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "…": "..."}


def norm_text(t: str) -> str:
    t = t.translate(TR)
    for k, v in LIG.items():
        t = t.replace(k, v)
    if any("ក" <= c <= "៿" for c in t):
        t = normalize_khmer(t)
    t = " ".join(t.split())
    return t


def crop_line(img, quad, pad_y=0.12, pad_x=0.04):
    xs, ys = quad[0::2], quad[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    h = y1 - y0
    H, W = img.shape[:2]
    X0, X1 = int(max(0, x0 - pad_x * h - 1)), int(min(W, x1 + pad_x * h + 1))
    Y0, Y1 = int(max(0, y0 - pad_y * h)), int(min(H, y1 + pad_y * h + 1))
    if X1 - X0 < 4 or Y1 - Y0 < 6:
        return None
    return img[Y0:Y1, X0:X1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-per-page", type=int, default=0)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", choices=["printed", "hand", "both"], default="printed",
                    help="printed = lines/line_texts (default), hand = hw_lines/hw_texts, both")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = a.out / a.split
    out.mkdir(parents=True, exist_ok=True)
    meta = open(out / "metadata.jsonl", "w", encoding="utf-8")
    binf = open(out / "lines.bin", "wb")  # packed JPEGs: no per-file block overhead
    off = 0
    n = skipped_oov = skipped_geo = 0
    for man in a.manifest:
        root = man.parent
        for ln in open(man, encoding="utf-8"):
            r = json.loads(ln)
            quads, texts, kinds = [], [], []
            for kind, qk, tk in (("printed", "lines", "line_texts"), ("hand", "hw_lines", "hw_texts")):
                q, t = r.get(qk) or [], r.get(tk) or []
                if a.which in (kind, "both") and t and len(t) == len(q):
                    quads += q
                    texts += t
                    kinds += [kind] * len(t)
            if not texts:
                continue
            img = cv2.imread(str(root / r["image"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            idx = list(range(len(texts)))
            if a.max_per_page:
                rng.shuffle(idx)
                idx = idx[: a.max_per_page]
            for i in idx:
                t = norm_text(texts[i])
                if len(t) < a.min_chars:
                    continue
                if any(c not in VSET for c in t):
                    skipped_oov += 1
                    continue
                c = crop_line(img, quads[i])
                if c is None:
                    skipped_geo += 1
                    continue
                h, w = c.shape
                nw = max(8, int(round(w * 48 / h)))
                if nw > 2400 or nw < 4 * len(t) * 0.5:
                    skipped_geo += 1
                    continue
                c = cv2.resize(c, (nw, 48), interpolation=cv2.INTER_AREA if h > 48 else cv2.INTER_CUBIC)
                jpg = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                binf.write(jpg)
                meta.write(json.dumps(dict(offset=off, nbytes=len(jpg), text=t, split=a.split, source=r.get("source", ""),
                                           doc_type=r.get("doc_type", ""), level=r.get("source", ""), kind=kinds[i], width=nw, height=48),
                                      ensure_ascii=False) + "\n")
                off += len(jpg)
                n += 1
                if a.max_lines and n >= a.max_lines:
                    break
            if a.max_lines and n >= a.max_lines:
                break
    meta.close()
    binf.close()
    print(f"{out}: {n} lines, skipped {skipped_oov} OOV, {skipped_geo} geometry")


if __name__ == "__main__":
    main()
