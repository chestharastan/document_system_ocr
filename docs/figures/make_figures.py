"""Figures for docs/ThaOCR_Book.md, drawn only from the project's own run logs and data.

    /home/hello/ocr/khm_repo_clone/det_venv/bin/python docs/figures/make_figures.py

Style: light chart surface, hairline solid grid, 2 px lines, ringed markers, rounded bar ends. Categorical slots 1-3
(#2a78d6 blue, #eb6834 orange, #1baf7a aqua) and the ordinal pair #86b6ef -> #1c5cab were checked with the dataviz
validator (light mode). Every chart's numbers are also printed as tables in the book.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = Path(__file__).resolve().parent
REPO = OUT.parent.parent
RUNS = Path("/home/hello/ocr/docsys/work/runs")
DATA = Path("/home/hello/ocr/docsys/work/data")
V1_DET = Path("/home/hello/ocr/khm_repo_clone/det_runs")
V1_REC = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation/runs")

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT, DARK = "#86b6ef", "#1c5cab"
DPI = 100            # design resolution: 1 px here = 1 px on screen
SAVE_DPI = 200       # saved at 2x for sharp rendering


def pt(px: float) -> float:
    """pixels at the design DPI -> points (matplotlib line widths and marker sizes are in points)"""
    return px * 72.0 / DPI


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"], "font.size": 9.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "figure.dpi": DPI,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "legend.frameon": False, "legend.fontsize": 9, "axes.unicode_minus": False,
})


def style(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(pt(1))
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=pt(1), linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5)


def header(fig, title, subtitle=None, y=0.985):
    fig.text(0.012, y, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.012, y - 0.045 * (6.0 / fig.get_figheight()), subtitle, ha="left", va="top", fontsize=9, color=INK2, wrap=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=SAVE_DPI)
    plt.close(fig)
    print("wrote", OUT / name)


def line(ax, x, y, color, label=None, markers=False):
    ax.plot(x, y, color=color, linewidth=pt(2), solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    if markers:
        ax.plot(x, y, linestyle="none", marker="o", markersize=pt(8), markerfacecolor=color, markeredgecolor=SURF,
                markeredgewidth=pt(2), zorder=4)


def dot(ax, x, y, color, size=9, z=5):
    ax.plot([x], [y], linestyle="none", marker="o", markersize=pt(size), markerfacecolor=color, markeredgecolor=SURF,
            markeredgewidth=pt(2), zorder=z)


def end_labels(ax, items, min_gap, x_pad):
    """items: [(x_end, y_end, text)]; labels right of the line ends, nudged apart with a thin leader when moved"""
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for x, y, t in items:
        yy = y if not placed or y - placed[-1] >= min_gap else placed[-1] + min_gap
        placed.append(yy)
        if abs(yy - y) > 1e-9:
            ax.plot([x, x + x_pad * 0.6], [y, yy], color=AXIS, linewidth=pt(1), zorder=2, clip_on=False)
        ax.text(x + x_pad, yy, t, va="center", ha="left", fontsize=8.5, color=INK2, clip_on=False)


def hbars(ax, labels, values, colors, fmt, xmax, value_pad):
    """Ranked horizontal bars: <= 24 px thick, 4 px rounded data end, square at the baseline, value at the tip."""
    n = len(values)
    ax.set_xlim(0, xmax)
    ax.set_ylim(n - 0.5, -0.5)
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    xu = xmax / bb.width                 # x data units per pixel
    yu = n / bb.height                   # y data units per pixel
    h = min(0.62, 24 * yu)
    r = 4 * xu
    for i, (v, c) in enumerate(zip(values, colors)):
        if v <= 0:
            continue
        rr = min(r, v / 2)
        ax.add_patch(FancyBboxPatch((0, i - h / 2), v, h, boxstyle=f"round,pad=0,rounding_size={rr}",
                                    mutation_aspect=yu / xu, linewidth=0, facecolor=c, zorder=3))
        ax.add_patch(Rectangle((0, i - h / 2), max(0.0, v - rr), h, linewidth=0, facecolor=c, zorder=3))
        ax.text(v + value_pad, i, fmt(v), va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.8, color=INK2)


# ------------------------------------------------------------------------------------------------ data readers
def yolo_curve(run_dir: Path):
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(run_dir / "results.csv"))]
    ep = [int(float(r["epoch"])) for r in rows]
    m = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    return ep, m


def jsonl(p: Path):
    return [json.loads(l) for l in open(p) if l.strip()]


# ------------------------------------------------------------------------------------------------ figures
def fig_layout_curves():
    runs = [(V1_DET / "layout", "v1 round 1 · YOLO11s 960 px · 12 classes"),
            (RUNS / "layout_r1", "v2 round 1 · YOLO11s · 21 classes"),
            (RUNS / "layout_r2", "v2 round 2 · YOLO11s · 21 classes"),
            (RUNS / "layout_n", "v2 nano · YOLO11n · 21 classes"),
            (RUNS / "layout_n_v3", "v3 nano · YOLO11n · 22 classes"),
            (RUNS / "layout_n_v4", "v4 nano · YOLO11n · 22 classes")]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.09, hspace=0.42, wspace=0.08)
    header(fig, "Layout detector: validation mAP50-95 per epoch, every training run",
           "Each run is scored on its own validation set, so compare the shape of a curve, not heights across panels. "
           "The dot marks the epoch kept as best.pt.")
    for ax, (d, title) in zip(axes.flat, runs):
        ep, m = yolo_curve(d)
        style(ax)
        line(ax, ep, m, S1)
        k = max(range(len(m)), key=lambda i: m[i])
        dot(ax, ep[k], m[k], DARK)
        ax.annotate(f"{m[k]:.3f} (epoch {ep[k]})", (ep[k], m[k]), xytext=(0, -16), textcoords="offset points",
                    ha="center" if 0.15 < k / max(1, len(m) - 1) < 0.85 else ("right" if k > 0 else "left"), fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
        ax.set_ylim(0.40, 0.90)
        ax.set_xlim(0, max(ep) + 1)
        ax.set_xlabel("epoch", fontsize=8.5, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("mAP50-95", fontsize=8.5, color=MUTED)
    save(fig, "fig_layout_training_curves.png")


def fig_layout_v3_v4():
    R = RUNS / "layout_n_v4"
    sets = [("On the v4 validation set (887 pages, 500 of them new v4-style pages)", json.load(open(R / "val_v3_on_v4set.json")), json.load(open(R / "val_v4.json"))),
            ("On the older v3 validation set (1,100 pages)", json.load(open(R / "val_v3_on_v3set.json")), json.load(open(R / "val_v4_on_v3set.json")))]
    a3, a4 = sets[0][1]["per_class"], sets[0][2]["per_class"]
    classes = sorted(a4, key=lambda c: a4[c]["map"] - a3[c]["map"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.4), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.83, bottom=0.07, wspace=0.06)
    header(fig, "Layout v3 versus v4, mAP50-95 per class",
           "Light dot = v3, dark dot = v4. Classes are ordered by the change on the v4 set. "
           "v4 gains a lot on the new Khmer official-document pages and stays level with v3 on the older set.")
    for ax, (title, s3, s4) in zip(axes, sets):
        style(ax, grid_axis="x")
        ax.spines["bottom"].set_visible(False)
        p3, p4 = s3["per_class"], s4["per_class"]
        for i, c in enumerate(classes):
            x3, x4 = p3[c]["map"], p4[c]["map"]
            ax.plot([x3, x4], [i, i], color=AXIS, linewidth=pt(2), solid_capstyle="round", zorder=2)
            dot(ax, x3, i, LIGHT, 9, 4)
            dot(ax, x4, i, DARK, 9, 5)
        ax.set_xlim(0.35, 1.0)
        ax.set_ylim(len(classes) - 0.5, -0.8)
        ax.set_title(f"{title}\nall classes: v3 {s3['map50_95']:.3f}, v4 {s4['map50_95']:.3f}", loc="left", fontsize=9.5, color=INK, pad=8, linespacing=1.5)
        ax.set_xlabel("mAP50-95", fontsize=8.5, color=MUTED)
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels(classes, fontsize=8.8, color=INK2)
    for c in ("handwriting", "header", "text"):
        i = classes.index(c)
        d = a4[c]["map"] - a3[c]["map"]
        axes[0].annotate(f"+{d:.2f}", (a4[c]["map"], i), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    h1, = axes[1].plot([], [], linestyle="none", marker="o", markersize=pt(9), markerfacecolor=LIGHT, markeredgecolor=SURF, label="layout v3")
    h2, = axes[1].plot([], [], linestyle="none", marker="o", markersize=pt(9), markerfacecolor=DARK, markeredgecolor=SURF, label="layout v4")
    fig.legend(handles=[h1, h2], loc="upper left", bbox_to_anchor=(0.006, 0.935), ncol=2, handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig_layout_v3_v4_classes.png")


def fig_rec_lineage():
    chain = [("v1", V1_REC / "crnn_m" / "log.jsonl"), ("expA", RUNS / "rec_expA" / "log.jsonl"), ("v2", RUNS / "rec_v2_M" / "log.jsonl"),
             ("v2.1", RUNS / "rec_v2_M_ft" / "log.jsonl"), ("v2.3", RUNS / "rec_v2_M_stamp2" / "log.jsonl"), ("v2.4", RUNS / "rec_v2_M_stamp3" / "log.jsonl")]
    xs, v01, dev, tha, bounds = [], [], [], [], []
    x = 0
    for name, p in chain:
        rows = jsonl(p)
        bounds.append((x + 0.5, name, len(rows)))
        for r in rows:
            x += 1
            xs.append(x)
            v01.append(r["val_by_set"].get("V01", r["val_cer"]))
            dev.append(r["real_dev_cer"])
            tha.append(r["val_by_set"].get("hand_real_thareah"))
    # the shipped default v2.3 = rec_v2_M_stamp2 best.pt = its epoch 2
    start_v23 = [b for b in bounds if b[1] == "v2.3"][0][0] - 0.5
    x_def = start_v23 + 2
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.6), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.865, bottom=0.07, hspace=0.32)
    header(fig, "Recogniser lineage: validation error after every training epoch",
           "Each version starts from the previous best checkpoint. v2.2 (a side branch from v2.1) is left out. "
           "The dark dot is the shipped default, v2.3 epoch 2.")
    panels = [(axes[0], v01, "Printed Khmer lines, synthetic V01 (CER %)", None),
              (axes[1], dev, "Real signboard crops, dev split, 2,200 crops (CER %)", None),
              (axes[2], tha, "Real Khmer handwriting, Thareah val, 291 lines (CER %), measured from v2.1 on", None)]
    for ax, ys, title, _ in panels:
        style(ax)
        pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
        line(ax, [p[0] for p in pts], [p[1] for p in pts], S1, markers=True)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=22 if ax is axes[0] else 6)
        for b in bounds[1:]:
            ax.axvline(b[0], color=GRID, linewidth=pt(1), zorder=1)
        yv = [p[1] for p in pts if abs(p[0] - x_def) < 1e-9]
        if yv:
            dot(ax, x_def, yv[0], DARK, 10, 6)
            ax.annotate(f"{yv[0]:.2f}", (x_def, yv[0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.22 * (hi - lo))
    for i, b in enumerate(bounds):
        nxt = bounds[i + 1][0] if i + 1 < len(bounds) else xs[-1] + 0.5
        axes[0].text((b[0] + nxt) / 2, 1.02, f"{b[1]} ({b[2]} ep)", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK2)
    axes[2].set_xlabel("cumulative training epochs along the lineage", fontsize=8.5, color=MUTED)
    axes[2].set_xlim(0.3, xs[-1] + 0.7)
    save(fig, "fig_rec_lineage.png")


def fig_textdet():
    h = json.load(open(RUNS / "textdet_v2" / "history.json"))
    it = [r["iter"] for r in h]
    sets = [("synth_f1", "synthetic pages", S1), ("doclaynet_f1", "DocLayNet (real)", S2), ("khm_pdf_f1", "Khmer PDF pages (real)", S3)]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.07, right=0.83, top=0.74, bottom=0.13)
    header(fig, "Text-line detector v2: F1 at IoU 0.5 on held-out pages during fine-tuning",
           "60 pages per set, the trainer's fixed post-processing. Iteration 0 scores the starting weights (the v1 robust checkpoint). "
           "best.pt is the checkpoint with the best mean F1.")
    style(ax)
    best = max(h, key=lambda r: r["mean_f1"])
    ax.axvline(best["iter"], color=AXIS, linewidth=pt(1), zorder=1)
    ax.text(best["iter"], 0.985, f"best.pt: iteration {best['iter']:,}, mean F1 {best['mean_f1']:.3f}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color=INK2)
    ends = []
    for key, name, c in sets:
        ys = [r[key] for r in h]
        line(ax, it, ys, c, label=name, markers=True)
        ends.append((it[-1], ys[-1], f"{name} {ys[-1]:.3f}"))
    end_labels(ax, ends, 0.022, 600)
    ax.set_xlim(-500, it[-1] + 500)
    ax.set_ylim(0.58, 0.95)
    ax.set_xlabel("training iteration (batch 8, 640 px crops)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("F1", fontsize=8.5, color=MUTED)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0.0)
    save(fig, "fig_textdet_v2_f1.png")


def fig_font():
    rows = jsonl(RUNS / "font_v2" / "log.jsonl")
    per = rows[-1]["per_family"]
    items = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    fig.subplots_adjust(left=0.19, right=0.95, top=0.885, bottom=0.06)
    header(fig, "Font classifier v2: family accuracy per family (%)",
           "8,000 fixed synthetic validation crops, last logged epoch (24, overall 92.0 %); the kept checkpoint, epoch 20, has 92.3 %. "
           "Khmer OS and Khmer OS System are the weak pair the project notes record as one design.")
    style(ax, grid_axis="x")
    ax.spines["bottom"].set_visible(False)
    hbars(ax, [k for k, _ in items], [v for _, v in items], [S1] * len(items), lambda v: f"{v:.1f}", 108, 0.8)
    ax.set_xticks([0, 25, 50, 75, 100])
    save(fig, "fig_font_family_accuracy.png")


def fig_structure():
    rows = jsonl(RUNS / "struct_v1" / "log.jsonl")
    ep = [r["epoch"] for r in rows]
    series = [("role_acc", "role accuracy", S1), ("first_f1", "paragraph-start F1", S2), ("succ_acc", "successor accuracy (reading order)", S3)]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.13)
    header(fig, "Structure model v1: validation metrics per epoch",
           "400 docgen v3 validation pages. Reading order is the slowest head to learn; it starts near zero and ends at 0.964.")
    style(ax)
    for key, name, c in series:
        ys = [r[key] for r in rows]
        line(ax, ep, ys, c, label=f"{name}: {ys[-1]:.3f} at epoch {ep[-1]}")
        dot(ax, ep[-1], ys[-1], c, 8, 5)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, ep[-1] + 0.5)
    ax.set_xlabel("epoch", fontsize=8.5, color=MUTED)
    ax.legend(loc="lower right")
    save(fig, "fig_structure_training.png")


def fig_layout_sources():
    s = json.load(open(DATA / "layout_v2_r2" / "stats.json"))["by_source"]
    names = {"gen0": "docgen v2 synthetic pages", "dln": "DocLayNet (real)", "real_sig_stamp": "real signature / stamp scans",
             "gen1": "docgen exam answer sheets", "d4la": "D4LA (real US scans)", "dln_deg": "DocLayNet, degraded copies",
             "commonforms": "CommonForms (real, checkboxes)", "khm": "real Khmer PDF pages", "khm_deg": "real Khmer PDF, degraded copies"}
    items = sorted(((names.get(k.split("/")[1], k), v) for k, v in s.items() if k.startswith("train/")), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.24, right=0.95, top=0.80, bottom=0.08)
    header(fig, "Layout training set round 2 (layout_v2_r2): training pages by source",
           f"{sum(v for _, v in items):,} training pages in total. This set trained layout v2 (s and n) and is replayed in v3 and v4.")
    style(ax, grid_axis="x")
    ax.spines["bottom"].set_visible(False)
    hbars(ax, [k for k, _ in items], [v for _, v in items], [S1] * len(items), lambda v: f"{int(v):,}", 13800, 150)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    save(fig, "fig_layout_r2_sources.png")


def fig_rec_mix():
    def n_meta(p):
        return sum(1 for _ in open(p))
    v2 = json.load(open("/home/hello/ocr/dataset_v2/stats.json"))["splits"]["train"]["levels"]
    st = {d: json.load(open(f"/home/hello/ocr/{d}/stats.json"))["splits"]["train"]["n"] for d in ("dataset_nf_mix", "dataset_hard", "dataset_nf_hard")}
    rec = lambda d: n_meta(DATA / d / "train" / "metadata.jsonl")  # noqa: E731
    items = [("synthetic print sections S01-S08 (sampled)", 200000, False),
             ("synthetic scene / heavy / medium lines (dataset_v2)", v2["scene"] + v2["heavy"] + v2["medium"], False),
             ("synthetic clean lines (dataset_v2, sampled)", 25000, False),
             ("synthetic lines, new fonts only (nf_mix)", st["dataset_nf_mix"], False),
             ("synthetic steep-tilt signs (hard)", st["dataset_hard"], False),
             ("synthetic steep-tilt signs, new fonts (nf_hard)", st["dataset_nf_hard"], False),
             ("synthetic 30-40 word lines (long, sampled)", 10000, False),
             ("docgen v2 printed lines", rec("rec_docgen"), False),
             ("docgen exam handwriting lines", rec("rec_exam_hand"), False),
             ("docgen form handwriting values", rec("rec_docgen_hand"), False),
             ("DocLayNet English lines (real)", rec("rec_doclaynet"), True),
             ("Thareah handwriting lines (real) x8", 8 * rec("rec_hand_real_thareah"), True),
             ("SoyVitou handwriting snippets (real) x4", 4 * rec("rec_hand_real_soyvitou"), True)]
    items.sort(key=lambda t: t[1], reverse=True)
    total = sum(v for _, v, _ in items)
    real = sum(v for _, v, r in items if r)
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    fig.subplots_adjust(left=0.33, right=0.95, top=0.84, bottom=0.07)
    header(fig, "Recogniser training mix final_v2_hand_real: lines drawn per epoch, by source",
           f"{total:,} line slots per epoch before the charset filter; {real:,} ({100 * real / total:.1f} %) come from real images, "
           "and only 32,684 of those are real Khmer (handwriting, oversampled).")
    style(ax, grid_axis="x")
    ax.spines["bottom"].set_visible(False)
    hbars(ax, [k for k, _, _ in items], [v for _, v, _ in items], [S2 if r else S1 for _, _, r in items], lambda v: f"{int(v):,}", 390000, 3000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v / 1000)}K" if v else "0"))
    h1 = Rectangle((0, 0), 1, 1, facecolor=S1, linewidth=0, label="synthetic")
    h2 = Rectangle((0, 0), 1, 1, facecolor=S2, linewidth=0, label="real images")
    fig.legend(handles=[h1, h2], loc="upper right", bbox_to_anchor=(0.95, 0.915), ncol=2, handlelength=1.0, handleheight=0.8)
    save(fig, "fig_rec_training_mix.png")


def compose_pages():
    """Labelled previews of generated pages (training.docgen.preview), three per figure."""
    py = sys.executable
    picks = {"fig_docgen_v2_exam.jpg": [("docgen_v2/val", "km_official_letter", 316, "docgen v2 · km_official_letter"),
                                        ("docgen_v2/val", "us_invoice", 443, "docgen v2 · us_invoice (photo capture)"),
                                        ("docgen_exam/val", "km_exam_sheet", 39, "docgen exam · km_exam_sheet")],
             "fig_docgen_v3_v4.jpg": [("docgen_v3/val", "km_royal_decree", 325, "docgen v3 · km_royal_decree"),
                                      ("docgen_v4/val", "km_gov_notice", 313, "docgen v4 · km_gov_notice"),
                                      ("docgen_v4/val", "km_letter_v4", 197, "docgen v4 · km_letter_v4 (photo capture)")]}
    with tempfile.TemporaryDirectory() as tmp:
        for name, cells in picks.items():
            ims = []
            for split, dt, idx, cap in cells:
                out = Path(tmp) / f"{dt}_{idx}"
                subprocess.run([py, "-m", "training.docgen.preview", "--manifest", str(DATA / split / "manifest.jsonl"), "--out", str(out),
                                "--n", "3", "--seed", "7", "--max-side", "1100", "--doc-type", dt], cwd=REPO, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
                f = out / f"dg_{idx:07d}_prev.jpg"
                im = Image.open(f).convert("RGB")
                im.thumbnail((640, 860))
                ims.append((im, cap))
            H = max(im.height for im, _ in ims) + 44
            W = sum(im.width for im, _ in ims) + 24 * (len(ims) + 1)
            sheet = Image.new("RGB", (W, H), SURF)
            d = ImageDraw.Draw(sheet)
            try:
                font = ImageFont.truetype("/usr/share/fonts/noto/NotoSans-Regular.ttf", 20)
            except OSError:
                font = ImageFont.load_default()
            x = 24
            for im, cap in ims:
                sheet.paste(im, (x, 8))
                d.text((x, im.height + 14), cap, fill=INK2, font=font)
                x += im.width + 24
            sheet.save(OUT / name, quality=82, optimize=True)
            print("wrote", OUT / name, sheet.size)


if __name__ == "__main__":
    fig_layout_curves()
    fig_layout_v3_v4()
    fig_rec_lineage()
    fig_textdet()
    fig_font()
    fig_structure()
    fig_layout_sources()
    fig_rec_mix()
    compose_pages()
