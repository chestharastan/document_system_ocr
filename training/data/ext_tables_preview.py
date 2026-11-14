#!/usr/bin/env python3
"""Draw N converted table crops of a tsr_real_<name> set with rows / columns / header / spans / cells overlaid.

    python -m training.data.ext_tables_preview --set work/data/tsr_real_pubtables1m --out work/reviews/ext_previews/tables_pubtables1m -n 20 [--split test]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from training.data.ext_tables_common import preview_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    for p in preview_set(a.set, a.out, a.n, a.seed, a.split):
        print(p)


if __name__ == "__main__":
    main()
