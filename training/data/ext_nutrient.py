#!/usr/bin/env python3
"""nutrientdocs/form-field-v1-benchmark (CC-BY-4.0, "Attribute to Nutrient") 5 row groups -> work/data/ext/nutrient_eval/test.

    python -m training.data.ext_nutrient --out work/data/ext/nutrient_eval --row-groups 0 30 60 90 118

Reads only the selected row groups of data/test-00000-of-00001.parquet over HTTP (HfFileSystem, nothing cached).
fields = [{box: [x,y,w,h] page px, category: Text | ChoiceButton | Signature}], variant empty | filled | handwritten
(empty pages are real blank forms; filled/handwritten values are synthetic).

Mapping (same decision as ext_commonforms.py):
  ChoiceButton -> checkbox (19) in objs.
  Text         -> rec["input_boxes"]: input area only, no label, so NOT a form_field box (our form_field = label + input).
                  Use for form_field eval as "a predicted form_field that contains / overlaps an input box is a hit", or as
                  ignore regions in training.
  Signature    -> rec["sig_fields"] (sign-here fields; in the handwritten variant they may contain synthetic ink, but
                  the box is the field, not the ink -> not mapped to signature).
partial_classes / eval_classes [19]. Handwritten values are not boxed as handwriting (18). No lines.
Extra fields: variant, page_id, src (source tag), license_class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from training.data.ext_common import Manifest, decode, fit, obj, r1, save_jpg, stats
from training.taxonomy import CLS

REPO_FILE = "datasets/nutrientdocs/form-field-v1-benchmark/data/test-00000-of-00001.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--row-groups", type=int, nargs="+", default=[0, 30, 60, 90, 118])
    a = ap.parse_args()
    from huggingface_hub import HfFileSystem
    pf = pq.ParquetFile(HfFileSystem().open(REPO_FILE, block_size=8 << 20))
    out = a.out / "test"
    man = Manifest(out / "manifest.jsonl")
    for rg in a.row_groups:
        for r in pf.read_row_group(rg).to_pylist():
            img = decode(r["image"]["bytes"])
            h0, w0 = img.shape[:2]
            img, s = fit(img)
            H, W = img.shape[:2]
            sx, sy = s * w0 / r["width"], s * h0 / r["height"]
            objs, inputs, sigs = [], [], []
            for f in r["fields"]:
                x, y, w, h = f["box"]
                b = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
                if f["category"] == "ChoiceButton":
                    ob = obj(CLS["checkbox"], b, W, H, min_side=1.0)
                    if ob:
                        objs.append(ob)
                elif f["category"] == "Text":
                    inputs.append([r1(v) for v in b])
                else:
                    sigs.append([r1(v) for v in b])
            name = f"{r['page_id']}.jpg".replace("/", "_")
            save_jpg(img, out / "images" / name)
            man.add(dict(image=f"images/{name}", w=W, h=H, source="nutrientdocs", doc_type="form", lang="und",
                         objs=objs, lines_ok=False, objs_ok=True, partial=True,
                         partial_classes=[CLS["checkbox"]], eval_classes=[CLS["checkbox"]],
                         input_boxes=inputs, sig_fields=sigs, variant=r["variant"], page_id=r["page_id"],
                         src=r["source"], license_class=r["license_class"]))
        print("row group", rg, man.n, flush=True)
    man.close()
    print(json.dumps(stats(out / "manifest.jsonl")))


if __name__ == "__main__":
    main()
