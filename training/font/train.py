"""Train the font classifier on synthetic line crops generated on the fly.

    python -m training.font.train --out /home/hello/ocr/docsys/work/runs/font_v1 --epochs 30 --steps 400 --batch 256

Logs one JSON line per epoch ("[epoch] {...}") like training/rec/train.py; keeps best.pt (highest family accuracy
on a fixed synthetic validation set, 8,000 crops from the held-out news split) and last.pt."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from training.font.fonts import SLANTS, WEIGHTS
from training.font.gen import Catalog, Texts, make_sample
from training.font.model import FontNet, count_params


class Stream(IterableDataset):
    def __init__(self, steps: int, batch: int, seed: int):
        self.n, self.seed = steps * batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + wid + int(time.time() * 1e3) % 100000)
        texts, cat = Texts("train"), Catalog()
        for _ in range(self.n // nw):
            g, fi, wi, si, r = make_sample(texts, cat, rng, train=True)
            yield torch.from_numpy(g).unsqueeze(0), fi, wi, si, float(np.log(r))


class Fixed(Dataset):
    def __init__(self, path: Path, n: int):
        if path.exists():
            d = np.load(path)
            self.x, self.y, self.r = d["x"], d["y"], d["r"]
        else:
            rng = random.Random(123)
            texts, cat = Texts("val"), Catalog()
            xs, ys, rs = [], [], []
            for _ in range(n):
                g, fi, wi, si, r = make_sample(texts, cat, rng, train=False)
                xs.append(g); ys.append((fi, wi, si)); rs.append(np.log(r))
            self.x, self.y, self.r = np.stack(xs), np.array(ys, np.int64), np.array(rs, np.float32)
            np.savez_compressed(path, x=self.x, y=self.y, r=self.r)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]).unsqueeze(0), int(self.y[i, 0]), int(self.y[i, 1]), int(self.y[i, 2]), float(self.r[i])


def norm(x):
    return (x.float() / 255.0 - 0.5) / 0.5


@torch.no_grad()
def evaluate(model, loader, dev, names):
    model.eval()
    ok = np.zeros(3); n = 0
    per = np.zeros((len(names), 2))
    size_err = []
    for x, f, w, s, r in loader:
        lf, lw, ls, lr = model(norm(x.to(dev)))
        pf, pw, ps = lf.argmax(1).cpu(), lw.argmax(1).cpu(), ls.argmax(1).cpu()
        ok += [(pf == f).sum().item(), (pw == w).sum().item(), (ps == s).sum().item()]
        size_err.append((torch.exp(lr.float().cpu() - r.float()) - 1).abs().numpy())  # relative size error
        n += len(f)
        for t, p in zip(f.numpy(), pf.numpy()):
            per[t, 0] += 1; per[t, 1] += (t == p)
    acc = ok / max(1, n)
    per_family = {names[i]: round(100 * per[i, 1] / per[i, 0], 1) for i in range(len(names)) if per[i, 0]}
    se = np.concatenate(size_err) if size_err else np.zeros(1)
    return {"family_acc": acc[0], "weight_acc": acc[1], "slant_acc": acc[2], "per_family": per_family,
            "size_rel_err_median": float(np.median(se)), "size_rel_err_mean": float(se.mean()), "size_within_10pct": float((se <= 0.10).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--val-n", type=int, default=8000)
    ap.add_argument("--init", default="")
    ap.add_argument("--size-w", type=float, default=4.0, help="weight of the size regression loss (log em ratio, smooth L1)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat = Catalog()
    names = cat.names
    (out / "labels.json").write_text(json.dumps({"families": names, "weights": WEIGHTS, "slants": SLANTS}, ensure_ascii=False, indent=1))
    model = FontNet(len(names), width=a.width).to(dev)
    if a.init:  # v1 checkpoints have no size head
        missing, unexpected = model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"], strict=False)
        print(f"[init] {a.init}: missing {missing}, unexpected {unexpected}", flush=True)
    print(f"[info] {len(names)} families, {count_params(model) / 1e6:.2f} M params, device {dev}", flush=True)
    val = DataLoader(Fixed(out / "val_v2.npz", a.val_n), batch_size=512, num_workers=2)
    train = DataLoader(Stream(a.steps, a.batch, seed=1), batch_size=a.batch, num_workers=a.workers,
                       persistent_workers=True, prefetch_factor=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best = -1.0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, f, w, s, r in train:
            x = norm(x.to(dev, non_blocking=True)); f, w, s, r = f.to(dev), w.to(dev), s.to(dev), r.float().to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                lf, lw, ls, lr = model(x)
                loss = (F.cross_entropy(lf, f, label_smoothing=0.05) + 0.5 * F.cross_entropy(lw, w) + 0.5 * F.cross_entropy(ls, s)
                        + a.size_w * F.smooth_l1_loss(lr.float(), r, beta=0.05))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if nb < a.steps - 1:
                sched.step()
            tot += loss.item(); nb += 1
            if nb >= a.steps:
                break
        ev = evaluate(model, val, dev, names)
        rec = {"epoch": ep, "loss": tot / max(1, nb), "val_family_acc": round(100 * ev["family_acc"], 2),
               "val_weight_acc": round(100 * ev["weight_acc"], 2), "val_slant_acc": round(100 * ev["slant_acc"], 2),
               "val_size_err_median_pct": round(100 * ev["size_rel_err_median"], 2), "val_size_within_10pct": round(100 * ev["size_within_10pct"], 1),
               "minutes": round((time.time() - t0) / 60, 1), "lr": opt.param_groups[0]["lr"]}
        if ep % 5 == 0 or ep == a.epochs:
            rec["per_family"] = ev["per_family"]
        print("[epoch] " + json.dumps(rec, ensure_ascii=False), flush=True)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        state = {"model": model.state_dict(), "names": names, "width": a.width, "epoch": ep, "val": rec}
        torch.save(state, out / "last.pt")
        score = ev["family_acc"] + ev["size_within_10pct"]  # both heads matter for the page
        if score > best:
            best = score
            torch.save(state, out / "best.pt")
    print(f"[done] best val (family acc + size-within-10%) {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
