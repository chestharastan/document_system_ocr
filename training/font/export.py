"""Export best.pt -> ONNX (dynamic batch, fixed 1 x H x W input: 32 x 256 cnn, 40 x 320 v3best; softmax outputs + em-ratio size) with the label lists in the ONNX
metadata, so the backend stage needs no side file.

    python -m training.font.export --run /home/hello/ocr/docsys/work/runs/font_v1 --out .model/v2/font/font_cnn_v1.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

from training.font.fonts import SLANTS, WEIGHTS
from training.font.model import build


class WithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        f, w, s, r = self.m(x)
        return torch.softmax(f, 1), torch.softmax(w, 1), torch.softmax(s, 1), torch.exp(r).unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    ck = torch.load(Path(a.run) / a.ckpt, map_location="cpu", weights_only=False)
    names = ck["names"]
    model, (H, W) = build(ck.get("arch", "cnn"), len(names), ck.get("width"))  # v1 / v2 checkpoints: cnn, 32 x 256
    model.load_state_dict(ck["model"]); model.eval()
    wrapped = WithSoftmax(model).eval()  # the wrapper must be in eval mode too, else the check below runs dropout
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 1, H, W)
    torch.onnx.export(wrapped, x, str(out), input_names=["image"], output_names=["family", "weight", "slant", "size"],
                      dynamic_axes={"image": {0: "batch"}, "family": {0: "batch"}, "weight": {0: "batch"}, "slant": {0: "batch"}, "size": {0: "batch"}},
                      opset_version=17, dynamo=False)
    m = onnx.load(str(out))
    for k, v in {"families": json.dumps(names, ensure_ascii=False), "weights": json.dumps(WEIGHTS), "slants": json.dumps(SLANTS),
                 "input": f"gray [N,1,{H},{W}] (x/255-0.5)/0.5", "size": "em ratio: CSS font px = size * crop height (px) before resizing", "val": json.dumps(ck.get("val", {}), ensure_ascii=False), "arch": ck.get("arch", "cnn")}.items():
        e = m.metadata_props.add(); e.key, e.value = k, v
    onnx.save(m, str(out))
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    xs = np.random.rand(3, 1, H, W).astype(np.float32)
    ref = [t.detach().numpy() for t in wrapped(torch.from_numpy(xs))]
    got = s.run(None, {"image": xs})
    err = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    print(f"exported {out} ({out.stat().st_size / 1e6:.2f} MB), {len(names)} families, max |torch-onnx| = {err:.2e}")


if __name__ == "__main__":
    main()
