"""Export a trained layout detector for the browser / backend and (optionally) validate it.

    python -m training.layout.export_web --weights work/runs/layout_n_v4/weights/best.pt --out web/models/layout/layout_v4_n_22cls.onnx \
        --data work/data/layout_v4/layout.yaml [--int8] [--val]

- ONNX opset 17, static [1,3,1024,1024], batch 1, class names in the `names` metadata (same contract as layout_v3).
- --int8 also writes <out>_int8.onnx (onnxruntime dynamic quantisation of the MatMul/Conv weights) and reports sizes.
- --val runs ultralytics validation of the .pt on the data yaml (GPU) and prints per-class mAP50 / mAP50-95.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    if a.val and a.data:
        r = model.val(data=str(a.data), imgsz=a.imgsz, batch=8, device=a.device, plots=False, verbose=False)
        names = model.names
        per = {names[int(c)]: {"map50": round(float(r.box.ap50[i]), 4), "map": round(float(r.box.ap[i]), 4), "p": round(float(r.box.p[i]), 4), "r": round(float(r.box.r[i]), 4)}
               for i, c in enumerate(r.box.ap_class_index)}
        summary = {"map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4), "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4), "per_class": per}
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        (a.out.parent / (a.out.stem + ".val.json")).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    path = model.export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False, batch=1, device="cpu")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, a.out)
    print("onnx", a.out, round(os.path.getsize(a.out) / 1048576, 2), "MB")
    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = a.out.with_name(a.out.stem + "_int8.onnx")
        quantize_dynamic(str(a.out), str(q), weight_type=QuantType.QUInt8, per_channel=False, reduce_range=False)
        print("int8", q, round(os.path.getsize(q) / 1048576, 2), "MB")


if __name__ == "__main__":
    main()
