#!/usr/bin/env python3
"""FUNSD (NON-COMMERCIAL, EVAL ONLY) test split -> canonical page manifest in work/data/eval_nc/funsd/test.

    python -m training.data.ext_funsd --zip work/data/_raw/funsd/dataset.zip --out work/data/eval_nc/funsd

Source: https://guillaumejaume.github.io/FUNSD/dataset.zip (original, has entity linking; the nielsr/funsd HF mirror
has no links). Read straight from the zip.

Labels:
  lines/line_texts: every FUNSD word is annotated, so lines are exhaustive. Words of one entity are grouped into
      text lines (vertical-overlap clustering, then left-to-right); quad = axis-aligned word union. lines_ok true.
  objs: form_field (20) = union box of a `question` entity and all `answer` entities linked to it.
      Questions without any linked answer (blank fields whose input area is not annotated) are NOT boxed, they are
      listed in rec["unlinked_questions"] (boxes) so an evaluator can ignore detections there.
      FUNSD `header` (form titles / section headings) and `other` are not mapped. Checkboxes, logos, signatures,
      handwriting present on the pages are unlabelled -> partial_classes [20].
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from training.data.ext_common import Manifest, decode, fit, obj, quad_from_box, save_jpg, stats
from training.taxonomy import CLS


def group_lines(words):
    """words: [(box[x0,y0,x1,y1], text)] -> list of (box, text) lines."""
    ws = sorted([w for w in words if w[1].strip()], key=lambda w: (w[0][1] + w[0][3]) / 2)
    lines = []
    for b, t in ws:
        h = b[3] - b[1]
        for L in lines:
            lb = L["box"]
            ov = min(lb[3], b[3]) - max(lb[1], b[1])
            if ov > 0.5 * min(h, lb[3] - lb[1]):
                L["w"].append((b, t))
                L["box"] = [min(lb[0], b[0]), min(lb[1], b[1]), max(lb[2], b[2]), max(lb[3], b[3])]
                break
        else:
            lines.append({"box": list(b), "w": [(b, t)]})
    out = []
    for L in lines:
        L["w"].sort(key=lambda w: w[0][0])
        out.append((L["box"], " ".join(t for _, t in L["w"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="testing_data")
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    split = "test" if a.split == "testing_data" else "train"
    out = a.out / split
    man = Manifest(out / "manifest.jsonl")
    anns = sorted(n for n in z.namelist() if n.startswith(f"dataset/{a.split}/annotations/") and n.endswith(".json"))
    for an in anns:
        stem = Path(an).stem
        img = decode(z.read(f"dataset/{a.split}/images/{stem}.png"))
        img, s = fit(img)
        H, W = img.shape[:2]
        form = json.loads(z.read(an))["form"]
        by_id = {e["id"]: e for e in form}
        lines, texts = [], []
        for e in form:
            for b, t in group_lines([([v * s for v in w["box"]], w["text"]) for w in e["words"]]):
                lines.append(quad_from_box(*b))
                texts.append(t)
        objs, unlinked = [], []
        for e in form:
            if e["label"] != "question":
                continue
            ans = set()
            for p, q in e.get("linking", []):
                other = q if p == e["id"] else p
                if other in by_id and by_id[other]["label"] == "answer":
                    ans.add(other)
            boxes = [e["box"]] + [by_id[i]["box"] for i in ans]
            ub = [min(b[0] for b in boxes) * s, min(b[1] for b in boxes) * s, max(b[2] for b in boxes) * s, max(b[3] for b in boxes) * s]
            o = obj(CLS["form_field"], ub, W, H)
            if o is None:
                continue
            (objs if ans else unlinked).append(o if ans else o[1:])
        name = f"{stem}.jpg"
        save_jpg(img, out / "images" / name)
        man.add(dict(image=f"images/{name}", w=W, h=H, source="funsd", doc_type="form", lang="en", nc=True,
                     lines=lines, line_texts=texts, objs=objs, lines_ok=True, objs_ok=True,
                     partial_classes=[CLS["form_field"]], unlinked_questions=unlinked))
    man.close()
    print(json.dumps(stats(out / "manifest.jsonl")))


if __name__ == "__main__":
    main()
