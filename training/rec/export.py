#!/usr/bin/env python3
"""Export a recogniser checkpoint (arch S/M/L) to ONNX fp32 + int8 with vocab.json: the old exporter, with the L arch registered.
    python -m training.rec.export --ckpt work/runs/rec_v2_M/best.pt --out work/runs/rec_v2_M/export
"""
from __future__ import annotations
import runpy, sys
from pathlib import Path
OLD = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation"); sys.path.insert(0, str(OLD))
import khmerocr.model as km  # noqa: E402
km.ARCHS["L"] = dict(c=(64, 128, 256, 320), proj=384, hidden=288, layers=2)
if __name__ == "__main__":
    if "--quantize" not in sys.argv: sys.argv.append("--quantize")
    sys.argv[0] = str(OLD / "scripts/export_onnx.py")
    runpy.run_path(str(OLD / "scripts/export_onnx.py"), run_name="__main__")
