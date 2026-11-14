"""Realistic font-stage training data: line crops cut from full synthetic pages the way the backend cuts them.

Pages come from the docgen templates (training/docgen: letters, contracts, invoices, forms, exams ... in Khmer and
English) with their photo / scan degradation (homography, lighting, shadow, blur, JPEG), plus page curl, a random
camera distance (global rescale) and phone-camera noise added here. ``Page.text`` is patched (source-level, docgen is
not modified) to
  * record every printed line's font file, font size (px), synthetic bold (stroke) and synthetic italic (shear),
  * remap the template's Khmer body / heading fonts to any family of training/font/fonts.py (one choice per page and
    original font, so a page stays typographically consistent and every family appears in page context),
  * set whole paragraphs in Word-style synthetic bold or italic now and then (Khmer has no italic faces).
The rendered page is then run through the backend's own text-line chain: DB detector (.model/v2/textdet) ->
``snap_line_boxes`` -> ``crop_quad`` (rec margins), and each detection is matched to the ground-truth lines it covers
(label only when unambiguous). A fraction of the ground-truth quads is also cut with detector-like jitter.

Size label = em px in the photo's own pixels: font px x local linear scale of the page homography (and rescale) at
the line, so ``em_ratio = em_px / crop height`` is what the pipeline needs, and lines of one paragraph share their em.

Output (per worker shard): <out>/shard_XX.bin (gray crops, height 40, width <= 960, row-major uint8) +
<out>/shard_XX.jsonl (one line per crop: offset, width, labels, page / paragraph ids, crop height, text ...).

    python -m training.font.pagedata --out /home/hello/ocr/docsys/work/data/font_pages/train --n 6000 --seed 11 --workers 16
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
for p in (str(REPO), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.font.fonts import families  # noqa: E402

SH = 40          # stored crop height
SW_MAX = 960     # stored crop width cap (3 windows of 256 at height 32 = 768 -> 960 at 40)
DET_ONNX = REPO / ".model/v2/textdet/db_v2.onnx"
KHM = re.compile(r"[ក-៿᧠-᧿]")

FAM = families()
FAM_NAMES = [n for n, _ in FAM]
KM_BODY = ["Khmer OS", "Khmer OS System", "Khmer OS Battambang", "Khmer OS Content", "Khmer OS Siemreap", "Noto Sans Khmer",
           "Noto Serif Khmer", "Khmer Digital", "Hanuman", "Kantumruy Pro", "Nokora", "Suwannaphum", "Preahvihear",
           "Odor Mean Chey", "Chenla", "Khmer OS Fasthand", "Khmer OS Freehand"]
# documents in the wild: the Khmer OS family dominates -> weight them up when remapping
KM_BODY_W = [3, 2, 4, 3, 3, 1.5, 1, 1.2, 1, 1, 1, 1, 0.8, 0.8, 0.6, 0.4, 0.4]
KM_HEAD = ["Khmer OS Muol", "Khmer OS Muol Light", "Khmer OS Muol Pali", "Khmer OS Bokor", "Khmer OS Metal Chrieng",
           "Koulen", "Dangrek", "Bayon", "Angkor", "Taprom"]
KM_HEAD_W = [3, 3, 1.5, 1, 1, 1.2, 1, 1, 1, 1]


def _fname_index() -> dict[str, tuple[int, str]]:
    """font file name -> (family index, weight) for the catalogue files and docgen's own files."""
    out = {}
    for i, (_, files) in enumerate(FAM):
        for p, w in files:
            out[p.name] = (i, w)
    # docgen files with other names (same designs)
    alias = {"Battambang-Regular.ttf": "Khmer OS Battambang", "Content-Regular.ttf": "Khmer OS Content",
             "Siemreap-Regular.ttf": "Khmer OS Siemreap", "KhmerOSsys.ttf": "Khmer OS System",
             "NotoSansKhmer-Regular.ttf": "Noto Sans Khmer", "Moul-Regular.ttf": "Khmer OS Muol",
             "Moulpali-Regular.ttf": "Khmer OS Muol Pali", "Bokor-Regular.ttf": "Khmer OS Bokor", "Metal-Regular.ttf": "Khmer OS Metal Chrieng"}
    for fn, fam in alias.items():
        if fn not in out and fam in FAM_NAMES:
            out[fn] = (FAM_NAMES.index(fam), "bold" if "Bold" in fn else "regular")
    return out


FNAME = _fname_index()


def label_of(path: str) -> tuple[int, str, str]:
    """-> (family index or -1, weight, slant) from a font file path."""
    fn = path.rsplit("/", 1)[-1]
    weight = "bold" if re.search(r"Bold|Black|Heavy|SemiBold|ExtraBold", fn) else "regular"
    slant = "italic" if "Italic" in fn else "upright"
    if fn in FNAME:
        fi, w = FNAME[fn]
        return fi, (w if w == "bold" else weight), slant
    if re.match(r"(Liberation|Noto)Serif(-|Display-)", fn) or fn.startswith("LiberationSerif"):
        return FAM_NAMES.index("Latin serif"), weight, slant
    if re.match(r"(Liberation|Noto)Sans-", fn):
        return FAM_NAMES.index("Latin sans"), weight, slant
    if fn.startswith("Battambang-") or fn.startswith("Content-") or fn.startswith("Suwannaphum-") or fn.startswith("KhmerDigital-"):
        base = {"Battambang": "Khmer OS Battambang", "Content": "Khmer OS Content", "Suwannaphum": "Suwannaphum", "KhmerDigital": "Khmer Digital"}[fn.split("-")[0]]
        return FAM_NAMES.index(base), weight, slant
    if fn.startswith("NotoSansKhmer") or fn.startswith("NotoSerifKhmer"):
        return FAM_NAMES.index("Noto Sans Khmer" if "Sans" in fn else "Noto Serif Khmer"), weight, slant
    return -1, weight, slant


# --------------------------------------------------------------------------------------------- Page.text patch
ST = SimpleNamespace(rng=random.Random(0), remap={}, para={}, lines={})


def _patch_canvas():
    from training.docgen import canvas, res
    src = textwrap.dedent(inspect.getsource(canvas.Page.text))
    src = src.replace("stroke=0, bound=None, marker=False):", "stroke=0, bound=None, marker=False, shear=0.0):", 1)
    anchor = "    a = np.asarray(m)\n"
    assert anchor in src, "docgen canvas.Page.text changed: update training/font/pagedata.py"
    src = src.replace(anchor, anchor + "    if shear:\n        a = cv2.warpAffine(a, np.float32([[1, -shear, shear * (pad + size * 0.8)], [0, 1, 0]]), (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)\n", 1)
    src = src.replace("def text(", "def _text_sh(", 1)
    ns: dict = {}
    exec(compile(src, canvas.__file__, "exec"), canvas.__dict__, ns)
    raw = ns["_text_sh"]

    def text(self, x, y, text, path, size, fill=(0, 0, 0), align="l", angle=0.0, kind="print", squeeze=1.0, stroke=0, bound=None, marker=False):
        if kind != "print" or not path or not text or not text.strip():
            return raw(self, x, y, text, path, size, fill, align, angle, kind, squeeze, stroke, bound, marker)
        rng = ST.rng
        fi, weight, slant = label_of(path)
        name = FAM_NAMES[fi] if fi >= 0 else ""
        # per-page font remap (one replacement per original file, kept for the page)
        if path not in ST.remap:
            new = path
            if rng.random() < 0.75 and name in KM_BODY and weight == "regular":
                fam = rng.choices(KM_BODY, KM_BODY_W)[0]
                files = [p for p, w in FAM[FAM_NAMES.index(fam)][1] if w == "regular"]
                new = str(rng.choice(files)) if files else path
            elif rng.random() < 0.75 and name in KM_HEAD:
                fam = rng.choices(KM_HEAD, KM_HEAD_W)[0]
                new = str(rng.choice(FAM[FAM_NAMES.index(fam)][1])[0])
            ST.remap[path] = new
        new = ST.remap[path]
        if new != path and res.font_ok(new, text):
            path = new
            fi, weight, slant = label_of(path)
            name = FAM_NAMES[fi] if fi >= 0 else ""
        # Word-style synthetic bold / italic, decided per paragraph
        key = self.cur["para"] if self.cur is not None else ("solo", len(self.lines))
        if key not in ST.para:
            heavy = name in KM_HEAD
            ST.para[key] = (rng.random() < (0.0 if heavy or weight == "bold" else 0.11),
                            rng.random() < (0.0 if slant == "italic" else (0.09 if KHM.search(text) else 0.05)),
                            rng.uniform(0.16, 0.30))
        sb, si, sh = ST.para[key]
        st = stroke
        if sb and not stroke:
            st = max(1, int(round(int(size) / 38.0)))
        n0 = len(self.lines)
        q = raw(self, x, y, text, path, size, fill, align, angle, kind, squeeze, st, bound, marker, shear=sh if si else 0.0)
        if q is not None and len(self.lines) > n0:
            self.struct[-1]["_fm"] = {"fam": fi, "w": 1 if (weight == "bold" or st > 0) else 0, "s": 1 if (slant == "italic" or si) else 0,
                                      "size": float(max(6, int(size))), "sq": float(squeeze), "file": path.rsplit("/", 1)[-1],
                                      "c": [float(np.mean(q[0::2])), float(np.mean(q[1::2]))]}
        return q

    canvas.Page.text = text


# --------------------------------------------------------------------------------------------- geometry helpers
def local_scale(Hm: np.ndarray, x: float, y: float, d: float = 4.0) -> float:
    P = cv2.perspectiveTransform(np.float64([[[x, y]], [[x + d, y]], [[x, y + d]]]), Hm).reshape(3, 2)
    J = np.stack([(P[1] - P[0]) / d, (P[2] - P[0]) / d], 1)
    return float(math.sqrt(abs(np.linalg.det(J))))


def curl(img: np.ndarray, rng: random.Random):
    """Page curl / wave: smooth vertical displacement varying along x (and a little along y). -> (img, fwd(pts))."""
    h, w = img.shape[:2]
    A = rng.uniform(0.004, 0.018) * h
    f = rng.uniform(0.5, 1.5)
    ph = rng.uniform(0, 2 * math.pi)
    B = rng.uniform(0, 0.006) * w

    def dy(x, y):
        return A * np.sin(math.pi * f * x / w + ph) * (0.6 + 0.4 * y / h)

    def dx(x, y):
        return B * np.sin(math.pi * y / h + ph)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = (xx - dx(xx, yy)).astype(np.float32)
    my = (yy - dy(xx, yy)).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def fwd(P):
        P = np.asarray(P, np.float64)
        return np.stack([P[:, 0] + dx(P[:, 0], P[:, 1]), P[:, 1] + dy(P[:, 0], P[:, 1])], 1)
    return out, fwd


def phone_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    rs = np.random.default_rng(rng.getrandbits(32))
    out = img.astype(np.float32)
    if rng.random() < 0.5:  # sensor noise (luma + chroma), stronger in shadows
        lum = out.mean(2, keepdims=True) / 255.0
        out += rs.normal(0, rng.uniform(2, 7), out.shape) * (1.3 - 0.6 * lum)
    if rng.random() < 0.35:  # demosaic / sharpening halo
        bl = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = out + rng.uniform(0.3, 0.9) * (out - bl)
    if rng.random() < 0.3:  # warm / cool white balance
        out *= np.float32([rng.uniform(0.85, 1.05), rng.uniform(0.95, 1.02), rng.uniform(0.95, 1.12)])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 92)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return out


# --------------------------------------------------------------------------------------------- worker
W_ = SimpleNamespace(det=None, tmp=None, bin=None, idx=None, off=0)


def _worker_init(out: str, tag: str):
    from app.services.stages.textdet_db_onnx import DBOnnxDetector
    spec = SimpleNamespace(path=str(DET_ONNX), params={}, id="db_v2", stage="textdet", kind="db_onnx", backend="onnxruntime")
    W_.det = DBOnnxDetector(spec, threads=1, device="cpu")
    W_.tmp = tempfile.mkdtemp(prefix="fontpg_", dir=out)
    os.makedirs(W_.tmp + "/images", exist_ok=True)
    wid = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    W_.bin = open(Path(out) / f"shard_{tag}{wid:02d}.bin", "ab")
    W_.idx = open(Path(out) / f"shard_{tag}{wid:02d}.jsonl", "a", encoding="utf-8")
    W_.off = W_.bin.tell()
    W_.name = f"shard_{tag}{wid:02d}"


def _store(gray: np.ndarray, rng: random.Random, meta: dict):
    h, w = gray.shape[:2]
    nw = max(8, int(round(w * SH / max(1, h))))
    g = cv2.resize(gray, (nw, SH), interpolation=cv2.INTER_AREA if h > SH else cv2.INTER_CUBIC)
    if nw > SW_MAX:
        x = rng.randint(0, nw - SW_MAX)
        g = g[:, x:x + SW_MAX]
        nw = SW_MAX
    g = np.ascontiguousarray(g, np.uint8)
    W_.bin.write(g.tobytes())
    meta.update(shard=W_.name, off=W_.off, w=int(nw))
    W_.off += g.size
    W_.idx.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n")


def make(job):
    idx, seed = job
    from training.docgen import generate
    from app.services.doc_structure import ink_mask
    from app.services.page_structure import snap_line_boxes
    from app.services.pipeline_service import crop_quad
    rng = random.Random(seed ^ 0x5F0F)
    ST.rng, ST.remap, ST.para = rng, {}, {}
    try:
        rec = generate.make_page((idx, seed, W_.tmp))
        if "error" in rec:
            return {"page": idx, "error": rec["error"][:200]}
        fpath = Path(W_.tmp) / rec["image"]
        img = cv2.imread(str(fpath))  # BGR, like the backend
        fpath.unlink(missing_ok=True)
        struct = rec.get("struct") or []
        if img is None or len(struct) != len(rec["lines"]):
            return {"page": idx, "error": "no struct"}
        Hm = np.float64(rec["aug"]["H"]).reshape(3, 3)
        gts = []
        for q, s, t in zip(rec["lines"], struct, rec["line_texts"]):
            fm = s.get("_fm")
            if not fm:
                continue
            em = fm["size"] * local_scale(Hm, *fm["c"])
            gts.append({"q": np.float64(q).reshape(4, 2), "em": em, "fam": fm["fam"], "w": fm["w"], "s": fm["s"], "para": s["para"],
                        "text": t, "role": s.get("role", "")})
        if len(gts) < 2:
            return {"page": idx, "error": "few lines"}
        # extra photo effects: curl, camera distance, phone noise
        if rng.random() < 0.35:
            img, fwd = curl(img, rng)
            for g in gts:
                g["q"] = fwd(g["q"])
        if rng.random() < 0.5:
            s = rng.uniform(0.55, 1.0)
            h, w = img.shape[:2]
            if max(h, w) * s >= 900:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                for g in gts:
                    g["q"] = g["q"] * s
                    g["em"] *= s
        if rng.random() < 0.6:
            img = phone_noise(img, rng)
        H, W = img.shape[:2]
        # backend text-line chain
        dets = W_.det.detect(img)
        blk = {"cls": "text", "lines": [{"quad": np.asarray(q, np.float32), "bbox": [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]} for q, _ in dets]}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        snap_line_boxes(ink_mask(gray), [blk], {"text"})
        GB = np.array([[g["q"][:, 0].min(), g["q"][:, 1].min(), g["q"][:, 0].max(), g["q"][:, 1].max()] for g in gts])
        n_det = n_gt = 0
        page_id = f"{seed}"
        for ln in blk["lines"]:
            x0, y0, x1, y1 = ln["bbox"]
            iy = np.minimum(y1, GB[:, 3]) - np.maximum(y0, GB[:, 1])
            ix = np.minimum(x1, GB[:, 2]) - np.maximum(x0, GB[:, 0])
            gh = np.maximum(1, GB[:, 3] - GB[:, 1])
            dh = max(1.0, y1 - y0)
            cand = np.where((iy >= 0.5 * np.minimum(gh, dh)) & (ix > 0.3 * gh))[0]
            # the line must really lie on the GT quad (a rotated watermark / stamp text has a huge bbox) and have a
            # comparable height (true quad height, not the bbox)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ok = []
            for c in cand:
                q = gts[c]["q"]
                th = max(1.0, float(np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)
                gc = q.mean(0)
                inside = cv2.pointPolygonTest(q.astype(np.float32), (float(cx), float(min(max(cy, q[:, 1].min()), q[:, 1].max()))), True) >= -0.3 * th \
                    or (x0 <= gc[0] <= x1 and y0 <= gc[1] <= y1)
                if inside and 0.6 * th <= dh <= 2.6 * th:
                    ok.append(c)
            cand = np.array(ok, int)
            if not len(cand) or rng.random() > 0.6:
                continue
            ixs = ix[cand]
            main = cand[int(np.argmax(ixs))]
            g0 = gts[main]
            same = all(gts[c]["fam"] == g0["fam"] and gts[c]["w"] == g0["w"] and gts[c]["s"] == g0["s"] and abs(gts[c]["em"] / g0["em"] - 1) < 0.04 for c in cand)
            if not same and ixs.max() < 0.85 * ixs.sum():
                continue
            crop = crop_quad(img, ln["quad"])
            ch = crop.shape[0]
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _store(g, rng, {"page": page_id, "para": g0["para"], "gt": int(main), "src": "det", "fam": g0["fam"], "wt": g0["w"], "sl": g0["s"],
                            "em": round(g0["em"], 3), "ch": int(ch), "r": round(g0["em"] / max(1, ch), 5), "n": len(g0["text"]),
                            "text": g0["text"][:48], "role": g0["role"], "y": round(float((y0 + y1) / 2 / H), 4), "x": round(float((x0 + x1) / 2 / W), 4),
                            "PW": W, "PH": H, "multi": int(len(cand) > 1)})
            n_det += 1
        # ground-truth quads with detector-like jitter (lines the detector missed or cut differently)
        for k, gt in enumerate(gts):
            if rng.random() > 0.2:
                continue
            q = gt["q"].astype(np.float32).copy()
            h = float(np.linalg.norm(q[3] - q[0]))
            ux = (q[1] - q[0]) / max(1e-6, float(np.linalg.norm(q[1] - q[0])))
            uy = (q[3] - q[0]) / max(1e-6, h)
            t, b = rng.uniform(-0.05, 0.3) * h, rng.uniform(-0.05, 0.3) * h
            l, r = rng.uniform(0, 0.5) * h, rng.uniform(0, 0.5) * h
            q = np.stack([q[0] - ux * l - uy * t, q[1] + ux * r - uy * t, q[2] + ux * r + uy * b, q[3] - ux * l + uy * b]).astype(np.float32)
            crop = crop_quad(img, q)
            ch = crop.shape[0]
            _store(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), rng, {"page": page_id, "para": gt["para"], "gt": k, "src": "gt", "fam": gt["fam"], "wt": gt["w"], "sl": gt["s"],
                   "em": round(gt["em"], 3), "ch": int(ch), "r": round(gt["em"] / max(1, ch), 5), "n": len(gt["text"]), "text": gt["text"][:48], "role": gt["role"],
                   "y": round(float(q[:, 1].mean() / H), 4), "x": round(float(q[:, 0].mean() / W), 4), "PW": W, "PH": H, "multi": 0})
            n_gt += 1
        W_.bin.flush(); W_.idx.flush()
        return {"page": idx, "det": n_det, "gt": n_gt, "lines": len(gts), "mode": rec["aug"].get("mode")}
    except Exception as e:
        return {"page": idx, "error": f"{e!r}", "tb": traceback.format_exc(limit=3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--degrade-p", type=float, default=0.85)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-tasks", type=int, default=0, help="recycle workers after N pages (bounds the per-worker memory leak)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from training.docgen import generate
    generate.init_parent(0.62, a.degrade_p)
    _patch_canvas()
    jobs = [(i, a.seed * 1_000_003 + i) for i in range(a.start, a.start + a.n)]
    ctx = mp.get_context("fork")
    t0 = time.time(); stats = {"det": 0, "gt": 0, "err": 0}
    with ctx.Pool(a.workers, initializer=_worker_init, initargs=(str(out), a.tag), maxtasksperchild=a.max_tasks or None) as pool:
        for k, r in enumerate(pool.imap_unordered(make, jobs, chunksize=2)):
            if "error" in r:
                stats["err"] += 1
                if stats["err"] <= 8:
                    print("  err", r.get("error"), r.get("tb", ""), flush=True)
            else:
                stats["det"] += r["det"]; stats["gt"] += r["gt"]
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"{k + 1}/{a.n} pages {el:.0f}s {(k + 1) / el:.2f} p/s crops det={stats['det']} gt={stats['gt']} errors={stats['err']}", flush=True)
    for d in out.glob("fontpg_*"):
        shutil.rmtree(d, ignore_errors=True)
    print(f"done {a.n} pages in {time.time() - t0:.0f}s: {stats}", flush=True)


if __name__ == "__main__":
    main()
