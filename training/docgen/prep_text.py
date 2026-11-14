"""Build the English sentence pool for docgen from public-domain Project Gutenberg books.

    python -m training.docgen.prep_text --raw /home/hello/ocr/docsys/work/data/docgen_text/raw \
        --out /home/hello/ocr/docsys/work/data/docgen_text/en_sentences.txt.gz

Strips the Gutenberg header/footer licence boilerplate, joins wrapped lines into paragraphs, splits into sentences,
keeps 25..260 char sentences of plain printable text. Output: one paragraph per line, sentences separated by TAB
(so consecutive sentences can be re-joined into coherent paragraphs).
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.I)
SENT = re.compile(r"(?<=[.!?])[\"'”’)]?\s+(?=[A-Z“\"'(])")
OK = re.compile(r"^[A-Za-z0-9 .,;:!?'\"()\-–—’‘“”&%$/]+$")


def paragraphs(text: str):
    m = START.search(text)
    if m:
        text = text[m.end():]
    m = END.search(text)
    if m:
        text = text[:m.start()]
    text = text.replace("\r", "")
    for block in re.split(r"\n\s*\n", text):
        p = " ".join(ln.strip() for ln in block.split("\n")).strip()
        p = p.replace("_", "").replace("--", "—")
        p = re.sub(r"\s+", " ", p)
        if len(p) < 60 or p.isupper() or p.lower().startswith(("chapter", "book ", "letter ", "illustration", "[")):
            continue
        yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n_par = n_sent = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for p in sorted(a.raw.glob("*.txt")):
            for par in paragraphs(p.read_text(encoding="utf-8", errors="ignore")):
                s = [x.strip() for x in SENT.split(par)]
                s = [x for x in s if 25 <= len(x) <= 260 and OK.match(x)]
                if len(s) >= 1:
                    f.write("\t".join(s) + "\n")
                    n_par += 1
                    n_sent += len(s)
    print(f"{n_par} paragraphs, {n_sent} sentences -> {a.out}")


if __name__ == "__main__":
    main()
