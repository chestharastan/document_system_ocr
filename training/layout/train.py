#!/usr/bin/env python3
"""Train the layout detector (Ultralytics YOLO11) on the dataset built by training/layout/build_yolo.py.

    python -m training.layout.train --data work/data/layout_v2/layout.yaml --out work/runs --weights <init.pt> --epochs 30

NOTE: Ultralytics (code and pretrained weights) is AGPL-3.0. Fine for local use; if you ship this inside a closed
product or offer it as a network service you must comply with the AGPL or buy an Enterprise licence. The dataset format
(YOLO txt) is standard, so the same data trains any other detector.
Caps this process's GPU memory (--vram-gb) so a shared laptop GPU stays usable. Resumable: --resume.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--vram-gb", type=float, default=5.0)
    ap.add_argument("--name", default="layout")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training set per epoch (speed benchmarks)")
    ap.add_argument("--device", default="0", help="GPU index, or cpu for a smoke test")
    ap.add_argument("--lr0", type=float, default=0.001, help="peak learning rate (use ~3e-4 to fine-tune a trained model)")
    ap.add_argument("--warmup", type=float, default=1.0, help="warm-up epochs")
    ap.add_argument("--hours", type=float, default=0, help="time budget: the LR schedule is fitted to this many hours (overrides --epochs)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from ultralytics import YOLO

    if a.vram_gb and a.device != "cpu":
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    if a.resume:
        YOLO(str(a.out / a.name / "weights/last.pt")).train(resume=True)
        return
    model = YOLO(a.weights)
    model.train(data=str(a.data), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, device=a.device,
                project=str(a.out.resolve()), name=a.name, exist_ok=True, pretrained=True, optimizer="AdamW", lr0=a.lr0, cos_lr=True,
                warmup_epochs=a.warmup, close_mosaic=3, mosaic=0.6, mixup=0.0, degrees=2.0, translate=0.08, scale=0.35, shear=1.0, perspective=0.0003,
                fliplr=0.0, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, plots=False, patience=100, fraction=a.fraction, amp=True,
                cache=False, save_period=1, val=True, verbose=True, **({"time": a.hours} if a.hours else {}))


if __name__ == "__main__":
    main()
