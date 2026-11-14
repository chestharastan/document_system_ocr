"""Evaluate a font checkpoint (v1 / v2 / v3) on the synthetic val set of its input size and the page-crop val set.

    python -m training.font.eval_v3 --ckpt /home/hello/ocr/docsys/work/runs/font_v2/best.pt --synth /home/hello/ocr/docsys/work/runs/font_v2/val_v2.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from training.font.model import build, count_params
from training.font.train_v3 import PageVal, eval_pages, eval_synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    names = ck["names"]
    model, (H, W) = build(ck.get("arch", "cnn"), len(names), ck.get("width"))
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    d = np.load(a.synth)
    es = eval_synth(model, (d["x"], d["y"], d["r"]), dev, names)
    ep = eval_pages(model, PageVal(H, W), dev, names)
    res = {"ckpt": a.ckpt, "params_M": round(count_params(model) / 1e6, 3), "input": [H, W], "synth": es, "pages": ep}
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
