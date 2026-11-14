#!/usr/bin/env python3
"""CommonForms (Apache-2.0) one shard -> canonical page manifest in work/data/ext/commonforms/valid.

    python -m training.data.ext_commonforms --raw work/data/_raw/commonforms --out work/data/ext/commonforms

Source: jbarrow/CommonForms data/valid-00000-of-00012.parquet (1,517 pages rendered from Common Crawl PDFs that have
fillable AcroForm widgets). Annotations = the PDF widgets: category 0 Text (input box), 1 CheckBox (checkbox / radio),
2 Signature (signature *field*, usually empty). bbox is COCO xywh in page pixels.

Mapping decision:
  CheckBox  -> checkbox (19), stored in objs.
  Text      -> NOT a form_field box. Our form_field = printed label + its input area; the widget covers only the blank
               input area and the dataset has no text layer, so the label (left of / above / in the previous table cell)
               cannot be attached reliably from pixels. Stored raw in rec["input_boxes"] ([x0,y0,x1,y1]) so a trainer
               can use them as ignore regions and an evaluator can count a predicted form_field as correct if it
               contains an input box.
  Signature -> NOT signature (19 is ink; these are empty sign-here fields). Stored raw in rec["sig_fields"].
Partial labelling: only checkbox is labelled -> partial_classes [19], eval_classes [19], partial true. All other
  content (text, titles, tables, logos, ...) is unlabelled: train with an ignore strategy for the other classes.
  Caveat (seen in previews): only *fillable* checkboxes are annotated; a printed "[ ]" glyph that is not a widget is
  not boxed, so checkbox recall labels can have some false negatives.
lines: none (no text layer) -> lines_ok false.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from training.data.ext_common import Manifest, decode, fit, obj, r1, save_jpg, stats
from training.taxonomy import CLS

FILE = "data/valid-00000-of-00012.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("jbarrow/CommonForms", FILE, repo_type="dataset", local_dir=str(a.raw))
    out = a.out / "valid"
    man = Manifest(out / "manifest.jsonl")
    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(rg).to_pylist():
            img = decode(r["image"]["bytes"])
            h0, w0 = img.shape[:2]
            img, s = fit(img)
            H, W = img.shape[:2]
            # bbox is in (width, height) of the record; rescale defensively if the decoded image differs
            sx, sy = s * w0 / r["width"], s * h0 / r["height"]
            o = r["objects"]
            objs, inputs, sigs = [], [], []
            for (x, y, w, h), c in zip(o["bbox"], o["category"]):
                b = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
                if c == 1:
                    ob = obj(CLS["checkbox"], b, W, H, min_side=1.0)
                    if ob:
                        objs.append(ob)
                else:
                    (inputs if c == 0 else sigs).append([r1(v) for v in b])
            name = f"cf_{r['id']:07d}.jpg"
            save_jpg(img, out / "images" / name)
            man.add(dict(image=f"images/{name}", w=W, h=H, source="commonforms", doc_type="form", lang="und",
                         objs=objs, lines_ok=False, objs_ok=True, partial=True,
                         partial_classes=[CLS["checkbox"]], eval_classes=[CLS["checkbox"]],
                         input_boxes=inputs, sig_fields=sigs, src_file=r["file_name"]))
    man.close()
    del pf
    if not a.keep_raw:
        os.remove(path)
    print(json.dumps(stats(out / "manifest.jsonl")))


if __name__ == "__main__":
    main()
