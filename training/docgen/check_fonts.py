"""Verify that every printed line_text renders with its recorded font without missing glyphs.

    python -m training.docgen.check_fonts --manifest DIR/manifest.jsonl
Checks (per line): fontTools cmap has every non-space char; HarfBuzz shaping gives no .notdef and no inserted dotted
circle (broken Khmer cluster). Prints counts and the first failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontTools.ttLib import TTFont

from .res import checker, font_pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    a = ap.parse_args()
    paths = {Path(p).name: p for pool in font_pools().values() for p in pool}
    cmaps = {}
    n = bad = 0
    fails = []
    for ln in open(a.manifest, encoding="utf-8"):
        r = json.loads(ln)
        for t, fn in zip(r["line_texts"], r.get("line_fonts", [])):
            n += 1
            p = paths.get(fn)
            if p is None:
                bad += 1
                fails.append((r["image"], t, fn, "unknown font"))
                continue
            if p not in cmaps:
                cmaps[p] = set(TTFont(p, lazy=True).getBestCmap())
            miss = [c for c in set(t) if c not in " ​" and ord(c) not in cmaps[p]]
            prob = None if miss else checker(p).problem(t)
            if miss or prob:
                bad += 1
                fails.append((r["image"], t, fn, f"missing {miss}" if miss else prob))
        if len(r["line_texts"]) != len(r["lines"]):
            fails.append((r["image"], "", "", "lines/line_texts length mismatch"))
            bad += 1
    print(f"checked {n} printed lines: {bad} problems")
    for f in fails[:20]:
        print("  ", f)


if __name__ == "__main__":
    main()
