"""Fill the classes a partially-labelled real set does not annotate, using a teacher layout model.

    python -m training.layout.pseudo_label --weights runs/layout_r1/weights/best.pt \
        --manifest data/ext/d4la/train/manifest.jsonl [--conf 0.5]

Writes manifest_pl.jsonl next to the input (same image paths). Rules:
  * human boxes are kept exactly; human-labelled classes (`partial_classes`) are never added by the teacher;
  * a teacher box is added only if its class is NOT in partial_classes, conf >= --conf (per-class override for
    the small/rare classes via --conf-small), it does not overlap a human box (IoU > 0.5 or >70% of the teacher
    box covered), and its centre is not inside an ignore_box;
  * after filling, the page counts as fully labelled (partial_classes removed, pseudo=true recorded).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from ultralytics import YOLO

from training.taxonomy import CLASSES

SMALL = {"page_number", "checkbox", "qr_code", "barcode", "logo", "stamp", "signature"}


def iou_cover(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, aa + bb - inter), inter / max(1e-6, aa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--conf-small", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=1024)
    a = ap.parse_args()
    m = YOLO(a.weights)
    for man in a.manifest:
        recs = [json.loads(l) for l in open(man) if l.strip()]
        added = collections.Counter()
        out = open(man.with_name("manifest_pl.jsonl"), "w")
        for i in range(0, len(recs), 16):
            chunk = recs[i:i + 16]
            res = m.predict([str(man.parent / r["image"]) for r in chunk], imgsz=a.imgsz, conf=min(a.conf, a.conf_small),
                            verbose=False, half=True)
            for r, p in zip(chunk, res):
                labelled = set(r.get("partial_classes", range(len(CLASSES))))
                human = [o[1:5] for o in r["objs"]]
                ign = r.get("ignore_boxes") or []
                new = []
                for b, c, s in zip(p.boxes.xyxy.tolist(), p.boxes.cls.tolist(), p.boxes.conf.tolist()):
                    c = int(c)
                    if c in labelled or s < (a.conf_small if CLASSES[c] in SMALL else a.conf):
                        continue
                    if any(iou_cover(b, h)[0] > 0.5 or iou_cover(b, h)[1] > 0.7 for h in human):
                        continue
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if any(g[-4] <= cx <= g[-2] and g[-3] <= cy <= g[-1] for g in ign):
                        continue
                    new.append([c] + [round(v, 1) for v in b])
                    added[CLASSES[c]] += 1
                r = dict(r, objs=r["objs"] + new, pseudo=True, n_pseudo=len(new))
                r.pop("partial_classes", None)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.close()
        print(man, len(recs), "pages; added", dict(added.most_common()), flush=True)


if __name__ == "__main__":
    main()
