#!/usr/bin/env python3
"""Train the phone-size table-structure detector (YOLO11n) on table crops built by training/table/make_tsr_dataset.py
(classes: row, column, column_header, spanning_cell, projected_row_header — docs/TABLES_CONTRACT.md §5).

    python -m training.table.train_tsr --data work/data/tsr_v1/tsr.yaml --out work/runs --name tsr_n_v1 --weights yolo11n.pt

Rows and columns are long thin boxes, so: no mosaic (it cuts rows), no flips, tiny rotation, mild scale jitter, and a
square 800-px letterbox (table crops are wide; 800 keeps thin rows >= 6 px). Fits a shared GPU: --vram-gb caps the
process; --device cpu works for smoke tests. Resumable with --resume. Ultralytics is AGPL-3.0 (see layout/train.py).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--vram-gb", type=float, default=1.3)
    ap.add_argument("--name", default="tsr_n")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=0, help="time budget in hours (overrides --epochs)")
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
                warmup_epochs=a.warmup, mosaic=0.0, close_mosaic=0, mixup=0.0, degrees=1.0, translate=0.04, scale=0.2, shear=0.5,
                perspective=0.0002, fliplr=0.0, flipud=0.0, hsv_h=0.015, hsv_s=0.4, hsv_v=0.35, erasing=0.0, plots=False, patience=100,
                fraction=a.fraction, amp=True, cache=False, save_period=1, val=True, verbose=True, rect=False,
                **({"time": a.hours} if a.hours else {}))


if __name__ == "__main__":
    main()
