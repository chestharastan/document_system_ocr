"""Train font v3 (family, weight, slant, size) on realistic page crops + the on-the-fly synthetic generator.

Sources (mixed per sample, ``--real-p``):
  * page crops (training/font/pagedata.py): docgen pages, photo-degraded, cut by the real DB detector + snap +
    crop_quad, labels incl. em ratio in the photo's own pixels; family-balanced sampling (1/sqrt(freq)); crop-margin
    augmentation (pad / trim top-bottom with the em ratio label adjusted);
  * synthetic single lines (training/font/gen.py make_sample, rendered at the model's input size).
Validation every epoch: the synthetic set (val_synth.npz, seed 123 like v2) and the page-crop val set (all crops,
inference-style windows: up to 3 windows per line averaged), incl. within-paragraph size consistency.
One JSON line per epoch in <out>/log.jsonl (v2 keys + page_* keys); best.pt = best page score.

    python -m training.font.train_v3 --arch v3best --out /home/hello/ocr/docsys/work/runs/font_v3_best --epochs 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from training.font import gen
from training.font.fonts import SLANTS, WEIGHTS
from training.font.model import build, count_params

PAGES = Path("/home/hello/ocr/docsys/work/data/font_pages")
SH = 40  # stored crop height (pagedata.SH)


# --------------------------------------------------------------------------------------------- page crops
def load_index(d: Path) -> list[dict]:
    rows = []
    for f in sorted(d.glob("shard_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("w", 0) >= 4 and r.get("ch", 0) >= 4 and 0.15 <= r["r"] <= 4.0:
                    rows.append(r)
    return rows


class Shards:
    def __init__(self, d: Path):
        self.d, self.mm = d, {}

    def get(self, r) -> np.ndarray:
        m = self.mm.get(r["shard"])
        if m is None or r["off"] + SH * r["w"] > len(m):  # the shard grows while pagedata is still writing
            m = self.mm[r["shard"]] = np.memmap(self.d / (r["shard"] + ".bin"), np.uint8, "r")
        if r["off"] + SH * r["w"] > len(m):
            return None  # index line written before its pixels were flushed
        return np.asarray(m[r["off"]:r["off"] + SH * r["w"]]).reshape(SH, r["w"])


def to_input(g: np.ndarray, H: int) -> np.ndarray:
    if g.shape[0] != H:
        nw = max(8, int(round(g.shape[1] * H / g.shape[0])))
        g = cv2.resize(g, (nw, H), interpolation=cv2.INTER_AREA if g.shape[0] > H else cv2.INTER_CUBIC)
    return g


def windows(g: np.ndarray, H: int, W: int) -> list[np.ndarray]:
    """Inference windows (same as the backend stage): pad a short line to W, cut a long one in up to 3 windows."""
    g = to_input(g, H)
    nw = g.shape[1]
    if nw <= W:
        fill = int(np.median(np.concatenate([g[:, :2].ravel(), g[:, -2:].ravel()])))
        pad = W - nw
        return [cv2.copyMakeBorder(g, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=fill)]
    n = min(3, int(np.ceil(nw / W)))
    return [g[:, x:x + W] for x in (int(round(k * (nw - W) / max(1, n - 1))) if n > 1 else 0 for k in range(n))]


def aug_real(g: np.ndarray, r: float, rng: random.Random, H: int, W: int):
    """Train-time view of a stored crop: margin jitter (label adjusted), photometric, random window. -> (img, ratio)"""
    h, w = g.shape
    if rng.random() < 0.5:  # vertical margin: trim or pad top / bottom (detector variance)
        t, b = (int(round(rng.uniform(-0.10, 0.12) * h)) for _ in range(2))
        if t < 0:
            g = g[-t:]
        if b < 0:
            g = g[:g.shape[0] + b]
        if t > 0 or b > 0:
            fill = int(np.median(np.concatenate([g[:2].ravel(), g[-2:].ravel()])))
            g = cv2.copyMakeBorder(g, max(0, t), max(0, b), 0, 0, cv2.BORDER_CONSTANT, value=fill)
        r = r * h / max(1, g.shape[0])
    if rng.random() < 0.3:  # extra resolution loss
        f = rng.uniform(0.45, 0.85)
        s = cv2.resize(g, (max(4, int(g.shape[1] * f)), max(4, int(g.shape[0] * f))), interpolation=cv2.INTER_AREA)
        g = cv2.resize(s, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    g = to_input(g, H)
    x = g.astype(np.float32)
    if rng.random() < 0.6:
        x = x * rng.uniform(0.75, 1.2) + rng.uniform(-25, 25)
    if rng.random() < 0.3:
        x += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 6), x.shape)
    g = x.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.04:
        g = 255 - g  # light text on dark
    nw = g.shape[1]
    if nw >= W:
        x0 = rng.randint(0, nw - W)
        g = g[:, x0:x0 + W]
    else:
        fill = int(np.median(np.concatenate([g[:, :2].ravel(), g[:, -2:].ravel()])))
        pad = W - nw
        left = rng.randint(0, pad)
        g = cv2.copyMakeBorder(g, 0, 0, left, pad - left, cv2.BORDER_CONSTANT, value=fill)
    return g, r


class Mix(IterableDataset):
    def __init__(self, n: int, H: int, W: int, real_p: float, seed: int):
        self.n, self.H, self.W, self.real_p, self.seed = n, H, W, real_p, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 7919 + wid * 104729 + int(time.time() * 1e3) % 1000003)
        gen.H, gen.W = self.H, self.W  # the synthetic generator renders at this model's input size
        texts, cat = gen.Texts("train"), gen.Catalog()
        sh = Shards(PAGES / "train")
        nprs = np.random.default_rng(rng.randrange(1 << 30))
        rows, cum = [], None
        for k in range(self.n // nw):
            if k % 20000 == 0:  # re-read the index: page data may still be growing
                rows = load_index(PAGES / "train")
                cnt = defaultdict(int)
                for r in rows:
                    cnt[r["fam"]] += 1
                wts = np.array([1.0 / math.sqrt(cnt[r["fam"]]) for r in rows])
                cum = np.cumsum(wts / wts.sum()) if len(rows) else None
            if rows and rng.random() < self.real_p:
                r = rows[min(len(rows) - 1, int(np.searchsorted(cum, nprs.random())))]
                g0 = sh.get(r)
                if g0 is None:
                    continue
                g, ratio = aug_real(g0, r["r"], rng, self.H, self.W)
                yield torch.from_numpy(g).unsqueeze(0), int(r["fam"]), int(r["wt"]), int(r["sl"]), float(np.log(ratio)), 1
            else:
                g, fi, wi, si, ratio = gen.make_sample(texts, cat, rng, train=True)
                yield torch.from_numpy(g).unsqueeze(0), fi, wi, si, float(np.log(ratio)), 0


# --------------------------------------------------------------------------------------------- evaluation
def norm(x):
    return (x.float() / 255.0 - 0.5) / 0.5


def synth_val(path: Path, n: int, H: int, W: int):
    if path.exists():
        d = np.load(path)
        return d["x"], d["y"], d["r"]
    rng = random.Random(123)
    gen.H, gen.W = H, W
    texts, cat = gen.Texts("val"), gen.Catalog()
    xs, ys, rs = [], [], []
    for _ in range(n):
        g, fi, wi, si, r = gen.make_sample(texts, cat, rng, train=False)
        xs.append(g); ys.append((fi, wi, si)); rs.append(np.log(r))
    x, y, r = np.stack(xs), np.array(ys, np.int64), np.array(rs, np.float32)
    np.savez_compressed(path, x=x, y=y, r=r)
    return x, y, r


@torch.no_grad()
def predict(model, xs: np.ndarray, dev, bs=512):
    """xs uint8 [N,H,W] -> probs family, weight, slant, ratio (numpy)."""
    model.eval()
    out = [[], [], [], []]
    for i in range(0, len(xs), bs):
        x = norm(torch.from_numpy(xs[i:i + bs]).unsqueeze(1).to(dev))
        lf, lw, ls, lr = model(x)
        out[0].append(torch.softmax(lf.float(), 1).cpu().numpy()); out[1].append(torch.softmax(lw.float(), 1).cpu().numpy())
        out[2].append(torch.softmax(ls.float(), 1).cpu().numpy()); out[3].append(torch.exp(lr.float()).cpu().numpy())
    return [np.concatenate(o) for o in out]


def eval_synth(model, val, dev, names):
    x, y, r = val
    pf, pw, ps, pr = predict(model, x, dev)
    se = np.abs(pr / np.exp(r) - 1)
    per = {}
    for i, n in enumerate(names):
        m = y[:, 0] == i
        if m.any():
            per[n] = round(100 * float((pf[m].argmax(1) == i).mean()), 1)
    return {"family_acc": float((pf.argmax(1) == y[:, 0]).mean()), "weight_acc": float((pw.argmax(1) == y[:, 1]).mean()),
            "slant_acc": float((ps.argmax(1) == y[:, 2]).mean()), "size_med": float(np.median(se)), "size_w10": float((se <= 0.1).mean()), "per_family": per}


class PageVal:
    """All page-val crops cut into inference windows once (per input size)."""

    def __init__(self, H: int, W: int, limit: int = 30000):
        rows = load_index(PAGES / "val")
        random.Random(5).shuffle(rows)
        self.rows = rows[:limit]
        sh = Shards(PAGES / "val")
        wins, self.owner = [], []
        for r in self.rows:
            ws = windows(sh.get(r), H, W)
            self.owner.append(list(range(len(wins), len(wins) + len(ws))))
            wins += ws
        self.x = np.stack(wins) if wins else np.zeros((0, H, W), np.uint8)


def eval_pages(model, pv: PageVal, dev, names) -> dict:
    if not pv.rows:
        return {}
    pf, pw, ps, pr = predict(model, pv.x, dev)
    fam, wt, sl, err, em_pred = [], [], [], [], []
    groups = defaultdict(list)
    per = defaultdict(lambda: [0, 0])
    for r, idx in zip(pv.rows, pv.owner):
        f, w, s, ratio = pf[idx].mean(0), pw[idx].mean(0), ps[idx].mean(0), float(pr[idx].mean())
        em = ratio * r["ch"]
        if r["fam"] >= 0:
            ok = int(f.argmax()) == r["fam"]
            fam.append(ok); per[r["fam"]][0] += 1; per[r["fam"]][1] += ok
        wt.append(int(w.argmax()) == r["wt"]); sl.append(int(s.argmax()) == r["sl"])
        err.append(abs(em / r["em"] - 1))
        if r["src"] == "det" and r["n"] >= 3:
            groups[(r["page"], r["para"])].append((em, r["em"]))
    err = np.array(err)
    cv_pred, cv_gt, para_err = [], [], []
    for v in groups.values():
        if len(v) >= 2:
            p, g = np.array(v).T
            cv_pred.append(p.std() / p.mean()); cv_gt.append(g.std() / g.mean())
            para_err.append(abs(np.median(p) / np.median(g) - 1))
    return {"family_acc": float(np.mean(fam)), "weight_acc": float(np.mean(wt)), "slant_acc": float(np.mean(sl)),
            "size_med": float(np.median(err)), "size_w10": float((err <= 0.1).mean()),
            "para_cv": float(np.mean(cv_pred)) if cv_pred else None, "para_cv_gt": float(np.mean(cv_gt)) if cv_gt else None,
            "para_size_med": float(np.median(para_err)) if para_err else None, "n": len(pv.rows),
            "per_family": {names[k]: round(100 * v[1] / v[0], 1) for k, v in sorted(per.items())}}


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="v3best", choices=["v3best", "cnn"])
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real-p", type=float, default=0.7)
    ap.add_argument("--size-w", type=float, default=4.0)
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat = gen.Catalog()
    names = cat.names
    model, (H, W) = build(a.arch, len(names), a.width)
    model.to(dev)
    (out / "labels.json").write_text(json.dumps({"families": names, "weights": WEIGHTS, "slants": SLANTS, "arch": a.arch, "input": [H, W]}, ensure_ascii=False, indent=1))
    if a.init:
        sd = torch.load(a.init, map_location=dev, weights_only=False)["model"]
        own = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(sd, strict=False)
        print(f"[init] {a.init}: loaded {len(sd)}/{len(own)} tensors", flush=True)
    print(f"[info] arch {a.arch} input {H}x{W}, {len(names)} families, {count_params(model) / 1e6:.3f} M params, device {dev}", flush=True)
    t0 = time.time()
    sval = synth_val(out / "val_synth.npz", 8000, H, W)
    pv = PageVal(H, W)
    print(f"[info] val: synthetic {len(sval[0])}, page crops {len(pv.rows)} ({len(pv.x)} windows), {time.time() - t0:.0f}s", flush=True)
    train = DataLoader(Mix(int(a.steps * a.batch * a.epochs * 1.1) + 10000, H, W, a.real_p, a.seed), batch_size=a.batch, num_workers=a.workers,
                       persistent_workers=True, prefetch_factor=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=5e-4 if a.arch == "v3best" else 1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps + 10, pct_start=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best = -1.0
    it = iter(train)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = {"loss": 0.0, "fam": 0.0, "size": 0.0}
        for _ in range(a.steps):
            x, f, w, s, r, real = next(it)
            x = norm(x.to(dev, non_blocking=True)); f, w, s, r = f.to(dev), w.to(dev), s.to(dev), r.float().to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                lf, lw, ls, lr = model(x)
                lfam = F.cross_entropy(lf.float(), f, label_smoothing=0.05, ignore_index=-1)
                lsize = F.smooth_l1_loss(lr.float(), r, beta=0.05)
                loss = lfam + 0.6 * F.cross_entropy(lw.float(), w) + 0.6 * F.cross_entropy(ls.float(), s) + a.size_w * lsize
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot["loss"] += loss.item(); tot["fam"] += lfam.item(); tot["size"] += lsize.item()
        es, ep_ = eval_synth(model, sval, dev, names), eval_pages(model, pv, dev, names)
        rec = {"epoch": ep, "loss": tot["loss"] / a.steps, "loss_family": tot["fam"] / a.steps, "loss_size": tot["size"] / a.steps,
               "val_family_acc": round(100 * es["family_acc"], 2), "val_weight_acc": round(100 * es["weight_acc"], 2),
               "val_slant_acc": round(100 * es["slant_acc"], 2), "val_size_err_median_pct": round(100 * es["size_med"], 2),
               "val_size_within_10pct": round(100 * es["size_w10"], 1),
               "page_family_acc": round(100 * ep_["family_acc"], 2), "page_weight_acc": round(100 * ep_["weight_acc"], 2),
               "page_slant_acc": round(100 * ep_["slant_acc"], 2), "page_size_err_median_pct": round(100 * ep_["size_med"], 2),
               "page_size_within_10pct": round(100 * ep_["size_w10"], 1),
               "page_para_size_cv_pct": round(100 * ep_["para_cv"], 2) if ep_.get("para_cv") is not None else None,
               "page_para_size_err_median_pct": round(100 * ep_["para_size_med"], 2) if ep_.get("para_size_med") is not None else None,
               "minutes": round((time.time() - t0) / 60, 1), "lr": opt.param_groups[0]["lr"]}
        if ep % 5 == 0 or ep == a.epochs:
            rec["per_family"] = es["per_family"]
            rec["page_per_family"] = ep_["per_family"]
        print("[epoch] " + json.dumps(rec, ensure_ascii=False), flush=True)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        state = {"model": model.state_dict(), "names": names, "arch": a.arch, "width": a.width, "input": [H, W], "epoch": ep, "val": rec}
        torch.save(state, out / "last.pt")
        # page crops are what the pipeline sees: family + size + weight on them, synthetic family as a tie-breaker
        score = ep_["family_acc"] + ep_["size_w10"] + 0.5 * ep_["weight_acc"] + 0.25 * ep_["slant_acc"] + 0.25 * es["family_acc"]
        if score > best:
            best = score
            torch.save(state, out / "best.pt")
        try:
            from importlib import util
            spec = util.spec_from_file_location("progress", "/home/hello/ocr/docsys/work/font_v3/progress.py")
            pm = util.module_from_spec(spec); spec.loader.exec_module(pm)
            tag = out.name
            pm.write(None, None, {f"{tag}_epoch": f"{ep}/{a.epochs}", f"{tag}_page_family": rec["page_family_acc"], f"{tag}_page_size_w10": rec["page_size_within_10pct"],
                                  f"{tag}_page_size_med": rec["page_size_err_median_pct"], f"{tag}_page_bold": rec["page_weight_acc"],
                                  f"{tag}_page_italic": rec["page_slant_acc"], f"{tag}_synth_family": rec["val_family_acc"]})
        except Exception as e:  # progress is cosmetic
            print("[progress]", e, flush=True)
    print(f"[done] best score {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
