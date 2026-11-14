"""Train the structure model on docgen pages with "struct" labels.

    python -m training.structure.train --train DIR/train/manifest.jsonl --val DIR/val/manifest.jsonl \
        --out /home/hello/ocr/docsys/work/runs/struct_v1 --epochs 40

Logs "[epoch] {...}" JSON lines; keeps best.pt by a composite score (paragraph-start F1 + role acc + successor acc)."""

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
from torch.utils.data import DataLoader, Dataset

from training.structure.data import ROLES, load_pages, load_pipe_pages, make_example
from training.structure.model import StructureNet, count_params, decode_order


class Pages(Dataset):
    def __init__(self, pages, train: bool, seed: int = 0, epoch_mult: int = 1, feat_version: int = 1, sim_p: float = 0.0):
        self.pages, self.train, self.seed, self.mult = pages, train, seed, epoch_mult
        self.fv, self.sim_p = feat_version, sim_p

    def __len__(self):
        return len(self.pages) * self.mult

    def __getitem__(self, i):
        rng = random.Random(self.seed * 7919 + i + (int(time.time() * 1e3) % 10007 if self.train else 0))
        return make_example(self.pages[i % len(self.pages)], rng, self.train, self.fv, self.sim_p if self.train else 0.0)


LINE_LABELS = ("role", "first", "level", "bullet", "succ", "tid", "row", "col", "cell")


def collate(batch):
    N = max(len(b["role"]) for b in batch)
    B = len(batch)
    feats = torch.zeros(B, N, batch[0]["feats"].shape[1]); chars = torch.zeros(B, N, batch[0]["chars"].shape[1], dtype=torch.long)
    pad = torch.ones(B, N, dtype=torch.bool)
    labels = {k: torch.full((B, N), -100, dtype=torch.long) for k in LINE_LABELS}
    labels["kind"] = torch.full((B,), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["role"])
        feats[i, :n] = torch.from_numpy(b["feats"]); chars[i, :n] = torch.from_numpy(b["chars"]); pad[i, :n] = False
        for k in LINE_LABELS:
            v = torch.from_numpy(np.asarray(b.get(k, np.full(n, -100)), np.int64))
            if k == "succ":
                v = torch.where(v >= n, torch.full_like(v, N), v)  # end-of-page -> index N of the padded matrix
            labels[k][i, :n] = v
        labels["kind"][i] = int(b.get("kind", -100))
    return feats, chars, pad, labels


def pair_targets(lab: dict, dev):
    """-> targets [B,N,N,3] (same row / column / cell) and the valid-pair mask: both lines labelled, at least one inside a
    table, i != j. Pairs of two lines outside every table are skipped (trivial, and they would dominate the loss)."""
    tid, row, col, cell = (lab[k].to(dev) for k in ("tid", "row", "col", "cell"))
    known, intab = tid != -100, tid >= 0
    same_t = (tid[:, :, None] == tid[:, None, :]) & intab[:, :, None] & intab[:, None, :]
    tgt = torch.stack([same_t & (row[:, :, None] == row[:, None, :]), same_t & (col[:, :, None] == col[:, None, :]),
                       same_t & (cell[:, :, None] == cell[:, None, :])], -1).float()
    N = tid.shape[1]
    valid = known[:, :, None] & known[:, None, :] & (intab[:, :, None] | intab[:, None, :]) & ~torch.eye(N, dtype=torch.bool, device=dev)[None]
    return tgt, valid


PAIR_POS_W = [1.0, 1.0, 3.0]


def table_losses(outs, lab: dict, dev):
    """v3 heads: kind CE, in-table CE (lines with known labels), pairwise BCE over valid pairs."""
    kind, intab, pair = outs[5], outs[6], outs[7]
    tid = lab["tid"].to(dev)
    lk = F.cross_entropy(kind, lab["kind"].to(dev), ignore_index=-100) if (lab["kind"] >= 0).any() else kind.sum() * 0
    it_t = torch.where(tid == -100, torch.full_like(tid, -100), (tid >= 0).long())
    li = F.cross_entropy(intab.transpose(1, 2), it_t, ignore_index=-100) if (it_t >= 0).any() else intab.sum() * 0
    tgt, valid = pair_targets(lab, dev)
    if valid.any():
        lp = F.binary_cross_entropy_with_logits(pair, tgt, pos_weight=torch.tensor(PAIR_POS_W, device=dev), reduction="none")
        lp = (lp * valid[..., None]).sum() / (valid.sum() * 3).clamp(min=1)
    else:
        lp = pair.sum() * 0
    return lk, li, lp


@torch.no_grad()
def evaluate_tables(model, loader, dev):
    model.eval()
    kind_ok = kind_n = 0
    it = [0, 0, 0]                       # tp, fp, fn for in-table
    pr = [[0, 0, 0] for _ in range(3)]   # per relation tp, fp, fn
    for feats, chars, pad, lab in loader:
        outs = model(feats.to(dev), chars.to(dev), pad.to(dev))
        kind, intab, pair = outs[5].cpu(), outs[6].cpu(), outs[7].cpu()
        yk = lab["kind"]; m = yk >= 0
        kind_ok += int((kind.argmax(-1)[m] == yk[m]).sum()); kind_n += int(m.sum())
        tid = lab["tid"]; known = tid != -100
        p = intab.argmax(-1) == 1; y = tid >= 0
        it[0] += int((p & y & known).sum()); it[1] += int((p & ~y & known).sum()); it[2] += int((~p & y & known).sum())
        tgt, valid = pair_targets(lab, torch.device("cpu"))
        pp = pair > 0
        for r in range(3):
            t = tgt[..., r] > 0.5; q = pp[..., r]
            pr[r][0] += int((q & t & valid).sum()); pr[r][1] += int((q & ~t & valid).sum()); pr[r][2] += int((~q & t & valid).sum())

    def f1(tp, fp, fn):
        return 2 * tp / max(1, 2 * tp + fp + fn)
    return {"kind_acc": kind_ok / max(1, kind_n), "intab_f1": f1(*it), "row_f1": f1(*pr[0]), "col_f1": f1(*pr[1]), "cell_f1": f1(*pr[2])}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tot = {"role": [0, 0], "level": [0, 0], "bullet": [0, 0], "succ": [0, 0]}
    tp = fp = fn = 0
    kendall, pages = 0.0, 0
    for feats, chars, pad, lab in loader:
        feats, chars, pad = feats.to(dev), chars.to(dev), pad.to(dev)
        role, first, level, bullet, succ = model(feats, chars, pad)[:5]
        m = ~pad
        for name, logit in (("role", role), ("level", level), ("bullet", bullet)):
            pred = logit.argmax(-1).cpu(); y = lab[name]
            ok = (pred == y) & (y >= 0); tot[name][0] += int(ok.sum()); tot[name][1] += int((y >= 0).sum())
        pf = first.argmax(-1).cpu(); yf = lab["first"]; valid = yf >= 0
        tp += int(((pf == 1) & (yf == 1)).sum()); fp += int(((pf == 1) & (yf == 0) & valid).sum()); fn += int(((pf == 0) & (yf == 1)).sum())
        ps = succ.argmax(-1).cpu(); ys = lab["succ"]
        ok = (ps == ys) & (ys >= 0); tot["succ"][0] += int(ok.sum()); tot["succ"][1] += int((ys >= 0).sum())
        # order quality: Kendall tau between decoded order and the truth, per page
        sl = succ.cpu().numpy(); ysn = ys.numpy()
        for b in range(feats.shape[0]):
            n = int(m[b].sum())
            if n < 3:
                continue
            logits = np.concatenate([sl[b, :n, :n], sl[b, :n, -1:]], axis=1)
            order = decode_order(logits)
            known = [i for i in range(n) if ysn[b, i] != -100]
            targets = set(int(v) for v in ysn[b, :n].tolist() if 0 <= v < n)
            cur = [i for i in known if i not in targets]
            if not known:
                continue
            # reconstruct true order from successor labels (unknown lines, -100, are not part of it)
            start = cur[0] if cur else known[0]
            seq, seen = [], set()
            while start is not None and start not in seen and len(seq) < n:
                seq.append(start); seen.add(start); nxt = int(ysn[b, start]); start = nxt if 0 <= nxt < n else None
            rank_t = {i: r for r, i in enumerate(seq)}
            rank_p = {i: r for r, i in enumerate(order)}
            if len(seq) < 3:
                continue
            pairs = [(i, j) for i in seq for j in seq if i < j]
            if pairs:
                conc = sum(1 for i, j in pairs if (rank_t[i] < rank_t[j]) == (rank_p[i] < rank_p[j]))
                kendall += 2 * conc / len(pairs) - 1; pages += 1
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"role_acc": tot["role"][0] / max(1, tot["role"][1]), "level_acc": tot["level"][0] / max(1, tot["level"][1]),
            "bullet_acc": tot["bullet"][0] / max(1, tot["bullet"][1]), "succ_acc": tot["succ"][0] / max(1, tot["succ"][1]),
            "first_f1": 2 * prec * rec / max(1e-9, prec + rec), "kendall": kendall / max(1, pages)}


def _progress(agent: str = "structure", **kw):
    """Live progress for the team page (work/progress/<agent>.json, atomic write)."""
    import os
    P = Path(f"/home/hello/ocr/docsys/work/progress/{agent}.json")
    try:
        d = json.loads(P.read_text())
    except Exception:  # noqa: BLE001
        d = {"agent": agent, "runs": [], "metrics": {}, "findings": []}
    for k, v in kw.items():
        if k == "metrics":
            d.setdefault("metrics", {}).update(v)
        else:
            d[k] = v
    d["agent"] = agent; d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = P.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1)); os.replace(tmp, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="comma list of GT manifests (docgen format with struct)")
    ap.add_argument("--val", required=True, help="GT val manifest (synthetic val)")
    ap.add_argument("--pipe-train", default="", help="comma list of pipe_extract jsonl files (re-read every epoch: may still grow)")
    ap.add_argument("--pipe-val", default="", help="comma list of pipe_extract jsonl files (photo-like held-out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--d", type=int, default=160); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=0)
    ap.add_argument("--feat-version", type=int, default=1); ap.add_argument("--pair", action="store_true")
    ap.add_argument("--sim-p", type=float, default=0.0, help="share of GT pages run through simulate_pipeline")
    ap.add_argument("--gt-per-epoch", type=int, default=6000, help="GT pages sampled per epoch")
    ap.add_argument("--pipe-mult", type=int, default=2, help="each pipe page is seen this many times per epoch")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--init", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--tables", action="store_true", help="v3 heads: page kind, line in table, same row / column / cell pairs")
    ap.add_argument("--table-w", type=float, default=1.0, help="weight of the v3 losses")
    ap.add_argument("--progress-agent", default="structure", help="work/progress/<name>.json written per epoch")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.name or out.name
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_pages = []
    for m in a.train.split(","):
        if m:
            tr_pages += load_pages(Path(m), a.limit)
    va_pages = load_pages(Path(a.val), 400)
    pipe_tr = [Path(p) for p in a.pipe_train.split(",") if p]
    pipe_va = load_pipe_pages([Path(p) for p in a.pipe_val.split(",") if p], 400)
    print(f"[info] {len(tr_pages)} GT train pages, {len(va_pages)} GT val, {len(pipe_va)} pipe val, device {dev}", flush=True)
    ff = a.ff or 2 * a.d
    model = StructureNet(d=a.d, layers=a.layers, heads=a.heads, ff=ff, feat_version=a.feat_version, pair=a.pair, tables=a.tables).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        print("[info] init from", a.init, model.load_state_dict(ck["model"], strict=False), flush=True)
    print(f"[info] {count_params(model) / 1e6:.2f} M params", flush=True)
    vl = DataLoader(Pages(va_pages, False, 2, feat_version=a.feat_version), batch_size=a.batch, shuffle=False, num_workers=2, collate_fn=collate)
    pl = DataLoader(Pages(pipe_va, False, 3, feat_version=a.feat_version), batch_size=a.batch, shuffle=False, num_workers=2, collate_fn=collate) if pipe_va else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    est_pipe = len(load_pipe_pages(pipe_tr)) if pipe_tr else 0
    steps_per_epoch = (min(a.gt_per_epoch, len(tr_pages)) + a.pipe_mult * max(est_pipe, 3000 if pipe_tr else 0)) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch + 10, pct_start=0.06)
    first_w = torch.tensor([1.0, 2.0], device=dev)  # paragraph starts are the minority
    best, t0, step = -1.0, time.time(), 0
    rng = random.Random(17)
    for ep in range(1, a.epochs + 1):
        pipe_pages = load_pipe_pages(pipe_tr) if pipe_tr else []
        gt = rng.sample(tr_pages, min(a.gt_per_epoch, len(tr_pages)))
        ds = torch.utils.data.ConcatDataset([Pages(gt, True, ep, feat_version=a.feat_version, sim_p=a.sim_p)] +
                                            ([Pages(pipe_pages, True, ep + 1000, epoch_mult=a.pipe_mult, feat_version=a.feat_version)] if pipe_pages else []))
        tl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True)
        model.train(); tot = 0.0; nb = 0
        for feats, chars, pad, lab in tl:
            if step >= a.epochs * steps_per_epoch:
                break
            feats, chars, pad = feats.to(dev), chars.to(dev), pad.to(dev)
            outs = model(feats, chars, pad)
            role, first, level, bullet, succ = outs[:5]
            loss = (F.cross_entropy(role.transpose(1, 2), lab["role"].to(dev), ignore_index=-100)
                    + 1.5 * F.cross_entropy(first.transpose(1, 2), lab["first"].to(dev), weight=first_w, ignore_index=-100)
                    + F.cross_entropy(level.transpose(1, 2), lab["level"].to(dev), ignore_index=-100)
                    + 0.5 * F.cross_entropy(bullet.transpose(1, 2), lab["bullet"].to(dev), ignore_index=-100)
                    + 1.5 * F.cross_entropy(succ.transpose(1, 2), lab["succ"].to(dev), ignore_index=-100))
            if a.tables:
                lk, li, lp = table_losses(outs, lab, dev)
                loss = loss + a.table_w * (0.5 * lk + 0.5 * li + 2.0 * lp)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
            step += 1
            tot += loss.item(); nb += 1
        ev = evaluate(model, vl, dev)
        if a.tables:
            ev.update(evaluate_tables(model, vl, dev))
        rec = {"epoch": ep, "loss": round(tot / max(1, nb), 4), **{k: round(v, 4) for k, v in ev.items()}}
        tscore = (0.5 * (ev["row_f1"] + ev["col_f1"] + ev["cell_f1"]) + 0.5 * ev["kind_acc"]) if a.tables else 0.0
        if pl is not None:
            evp = evaluate(model, pl, dev)
            if a.tables:
                evp.update(evaluate_tables(model, pl, dev))
                tscore += 0.5 * (evp["row_f1"] + evp["col_f1"] + evp["cell_f1"]) + 0.5 * evp["kind_acc"]
            rec.update({"photo_" + k: round(v, 4) for k, v in evp.items()})
            score = evp["first_f1"] + evp["role_acc"] + evp["succ_acc"] + evp["kendall"] + 0.5 * (ev["first_f1"] + ev["role_acc"] + ev["succ_acc"] + ev["kendall"])
        else:
            score = ev["first_f1"] + ev["role_acc"] + ev["succ_acc"] + ev["kendall"]
        score += tscore
        rec.update({"train_pages": len(gt) + len(pipe_pages), "pipe_pages": len(pipe_pages), "lr": round(sched.get_last_lr()[0], 7),
                    "minutes": round((time.time() - t0) / 60, 1)})
        print("[epoch] " + json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        state = {"model": model.state_dict(), "d": a.d, "layers": a.layers, "heads": a.heads, "ff": ff, "feat_version": a.feat_version,
                 "pair": a.pair, "tables": a.tables, "roles": ROLES, "epoch": ep, "val": rec}
        torch.save(state, out / "last.pt")
        if score > best:
            best = score; torch.save(state, out / "best.pt")
        try:
            el = (time.time() - t0) / ep
            _progress(agent=a.progress_agent, phase="train_best" if "best" in name else ("train_v3" if a.tables else "train_small"),
                      message=f"{name}: epoch {ep}/{a.epochs}, {len(pipe_pages)} real-pipeline pages + {len(gt)} GT pages this epoch",
                      metrics={name: {k: rec[k] for k in rec if k in ("epoch", "loss", "role_acc", "first_f1", "succ_acc", "kendall",
                                                                        "kind_acc", "intab_f1", "row_f1", "col_f1", "cell_f1",
                                                                        "photo_role_acc", "photo_first_f1", "photo_succ_acc", "photo_kendall",
                                                                        "photo_row_f1", "photo_col_f1", "photo_cell_f1", "photo_kind_acc")}},
                      eta=f"{name}: ~{el * (a.epochs - ep) / 60:.0f} min left")
        except Exception as e:  # noqa: BLE001
            print("[warn] progress", e, flush=True)
    print(f"[done] best composite {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
