"""Dataset statistics for a docgen (or any v2) manifest.

    python -m training.docgen.stats --manifest DIR/manifest.jsonl
Prints per-class box counts and % of pages containing the class, per-doc_type and per-lang page counts, line counts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..taxonomy import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    a = ap.parse_args()
    recs = [json.loads(ln) for ln in open(a.manifest, encoding="utf-8") if ln.strip()]
    n = len(recs)
    boxes, pages = Counter(), Counter()
    for r in recs:
        cs = [o[0] for o in r.get("objs", [])]
        boxes.update(cs)
        pages.update(set(cs))
    print(f"pages: {n}")
    print(f"{'id':>3} {'class':<15} {'boxes':>7} {'pages':>6} {'pages%':>7}")
    for i, c in enumerate(CLASSES):
        print(f"{i:>3} {c:<15} {boxes[i]:>7} {pages[i]:>6} {100 * pages[i] / max(n, 1):>6.1f}%")
    print("\ndoc_type:")
    for k, v in sorted(Counter(r.get("doc_type") for r in recs).items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:>6} {100 * v / n:>5.1f}%")
    print("\nlang:", dict(Counter(r.get("lang") for r in recs)))
    print("degraded:", sum(1 for r in recs if r.get("aug", {}).get("degrade")), "  modes:", dict(Counter(r.get("aug", {}).get("mode", "clean") for r in recs)))
    nl = sum(len(r.get("lines", [])) for r in recs)
    print(f"printed lines: {nl} ({nl / max(n, 1):.1f}/page)  hw lines: {sum(len(r.get('hw_lines', [])) for r in recs)}  "
          f"formula lines: {sum(len(r.get('formula_lines', [])) for r in recs)}")
    sizes = [max(r['w'], r['h']) for r in recs]
    if sizes:
        print(f"long side: min {min(sizes)} median {sorted(sizes)[len(sizes) // 2]} max {max(sizes)}")


if __name__ == "__main__":
    main()
