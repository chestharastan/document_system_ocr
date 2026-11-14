#!/usr/bin/env python3
"""Render N preview pages (taxonomy COLORS, class names, line quads) from a canonical manifest.

    python -m training.data.ext_preview --split-dir work/data/ext/d4la/train --out work/reviews/ext_previews/d4la -n 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from training.data.ext_common import draw_preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lines", action="store_true")
    a = ap.parse_args()
    recs = [json.loads(x) for x in open(a.split_dir / "manifest.jsonl")]
    random.Random(a.seed).shuffle(recs)
    for i, r in enumerate(recs[: a.n]):
        draw_preview(a.split_dir / r["image"], r, a.out / f"{i}_{Path(r['image']).stem[:40]}.jpg", show_lines=not a.no_lines)
        print(a.out / f"{i}_{Path(r['image']).stem[:40]}.jpg")


if __name__ == "__main__":
    main()
