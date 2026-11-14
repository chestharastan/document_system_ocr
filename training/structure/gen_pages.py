"""struct_v2 page generator: docgen templates (contracts / company letters + the existing letter families) rendered and
photographed like a phone does it: perspective, curl / fold (smooth non-linear displacement), desk background, cut-off
edges, uneven light + shadow, blur, JPEG. Writes docgen-format manifests (lines = warped quads, struct labels) that
training.structure.pipe_extract then runs through the REAL pipeline (layout + text detection + recognition + bullets).

    python -m training.structure.gen_pages --out DIR --n N --seed S --workers 8 [--contract-p 0.7]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from training.docgen import res  # noqa: E402

G: dict = {}


def init(contract_p: float):
    res.load()
    from faker import Faker
    from training.docgen import templates_contract as TC, templates_km as TK, templates_km_v4 as TK4, templates_en as TE  # noqa: F401
    G["fake"] = Faker("en_US")
    G["contract"] = {**TC.CONTRACT_KM, **TC.CONTRACT_EN}
    other = {k: v for k, v in {**TK.KM_TEMPLATES, **TE.EN_TEMPLATES}.items()
             if k in ("km_gov_notice", "km_letter_v4", "km_official_letter", "km_press_release", "us_letter", "km_textbook", "en_textbook")}
    G["other"] = other
    G["contract_p"] = contract_p


# ----------------------------------------------------------------------------------------------- phone photo
def _smooth_field(H, W, rs, cells, amp):
    g = rs.normal(0, 1, (cells, cells)).astype(np.float32)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC) * amp


def phone_photo(img: np.ndarray, rng: random.Random):
    """-> (photo, fwd(points Nx2 page px -> photo px), info). Non-linear: curl + fold + perspective + crop."""
    from det.degrade import make_background  # old repo (sys.path via res)
    rs = np.random.default_rng(rng.getrandbits(32))
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    fx = []
    # paper tint / fade
    if rng.random() < 0.7:
        out = out * np.array([rng.uniform(0.9, 1.0), rng.uniform(0.9, 1.0), rng.uniform(0.82, 1.0)], np.float32)
    # ---- non-linear page deformation (in page coordinates): dst->src displacement D(x, y)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Dx = np.zeros((H, W), np.float32); Dy = np.zeros((H, W), np.float32)
    if rng.random() < 0.75:  # curl: lines bow (vertical displacement varying along x), edges lift
        a = H * rng.uniform(0.002, 0.012) * rng.choice([-1, 1])
        ph = rng.uniform(0, np.pi)
        Dy += a * np.sin(np.pi * xx / W * rng.uniform(0.6, 1.4) + ph) * (0.6 + 0.4 * yy / H)
        fx.append("curl")
    if rng.random() < 0.5:  # horizontal fold (letter folded in three): a soft step + slight bend at the crease
        yf = H * rng.choice([1 / 3, 2 / 3, 0.5]) + rng.uniform(-0.03, 0.03) * H
        s = rng.uniform(0.004, 0.012) * H
        t = 1 / (1 + np.exp(-(yy - yf) / (H * 0.01)))
        Dy += s * t * rng.choice([-1, 1]) * (0.5 + xx / W)
        Dx += s * 0.6 * t * rng.uniform(-1, 1)
        fx.append("fold")
    if rng.random() < 0.5:  # low-frequency wobble
        Dx += _smooth_field(H, W, rs, 4, W * 0.002)
        Dy += _smooth_field(H, W, rs, 4, H * 0.002)
    if fx:
        out = cv2.remap(out, xx + Dx, yy + Dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def page_fwd(P):  # src -> dst by fixed point: dst = src - D(dst)
        Q = P.copy()
        for _ in range(4):
            xi = np.clip(Q[:, 0], 0, W - 1).astype(np.int32); yi = np.clip(Q[:, 1], 0, H - 1).astype(np.int32)
            Q = P - np.stack([Dx[yi, xi], Dy[yi, xi]], 1)
        return Q

    # ---- perspective onto a desk
    mode = rng.choices(["photo", "photo_close", "scanlike"], [0.55, 0.3, 0.15])[0]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    if mode == "scanlike":
        s, ang, jit = 1.0, rng.uniform(-2.5, 2.5), 0.005
        Wc, Hc = int(W * 1.06), int(H * 1.06)
    elif mode == "photo":
        s, ang, jit = rng.uniform(0.6, 0.85), rng.uniform(-6, 6), rng.uniform(0.01, 0.06)
        Wc, Hc = int(W * s * rng.uniform(1.1, 1.35)), int(H * s * rng.uniform(1.08, 1.25))
    else:  # close-up: the page fills (and leaves) the frame -> cut-off edges
        s, ang, jit = rng.uniform(0.8, 1.0), rng.uniform(-4, 4), rng.uniform(0.01, 0.05)
        Wc, Hc = int(W * s * rng.uniform(0.86, 1.02)), int(H * s * rng.uniform(0.86, 1.02))
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = ((src - [W / 2, H / 2]) * s) @ R.T
    q = q + np.array([[rs.uniform(-jit, jit) * W * s, rs.uniform(-jit, jit) * H * s] for _ in range(4)])
    dst = (q + [Wc / 2 + rng.uniform(-0.04, 0.04) * Wc, Hc / 2 + rng.uniform(-0.04, 0.04) * Hc]).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    warped = cv2.warpPerspective(out, Hm, (Wc, Hc), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.ones((H, W), np.float32), Hm, (Wc, Hc), flags=cv2.INTER_LINEAR)[..., None]
    bg = make_background(Hc, Wc, rng, rs)
    canvas = bg * (1 - mask) + warped * mask
    # ---- light, shadow (a hand / phone shadow at the bottom), blur, noise, JPEG
    yy2, xx2 = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    light = 1.0 + rng.uniform(-0.15, 0.15) * (xx2 / Wc - 0.5) * 2 + rng.uniform(-0.15, 0.15) * (yy2 / Hc - 0.5) * 2
    if rng.random() < 0.6:
        sh = np.zeros((Hc, Wc), np.float32)
        y0 = rng.uniform(0.55, 0.95) * Hc
        pts = np.array([[rng.uniform(-0.2, 0.5) * Wc, Hc * 1.1], [rng.uniform(0.2, 0.8) * Wc, y0], [rng.uniform(0.7, 1.3) * Wc, y0 + rng.uniform(-0.2, 0.2) * Hc], [Wc * 1.2, Hc * 1.1]], np.int32)
        cv2.fillPoly(sh, [pts], 1.0)
        light *= 1 - cv2.GaussianBlur(sh, (0, 0), rng.uniform(25, 80)) * rng.uniform(0.15, 0.45)
        fx.append("shadow")
    canvas = canvas * np.clip(light, 0.4, 1.2)[..., None]
    if rng.random() < 0.6:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.4, 1.3))
    canvas = canvas + rs.normal(0, rng.choice([1, 2, 4]), canvas.shape).astype(np.float32)
    photo = np.clip(canvas, 0, 255).astype(np.uint8)
    # final size: phone photo long side 1400..2300
    long0 = max(Wc, Hc)
    sc = float(np.clip(long0 * rng.uniform(0.9, 1.3), 1400, 2300)) / long0
    photo = cv2.resize(photo, (int(Wc * sc), int(Hc * sc)), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    if rng.random() < 0.7:
        ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 90)])
        photo = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if ok else photo

    def fwd(P):
        P = page_fwd(np.asarray(P, np.float64).reshape(-1, 2))
        P = cv2.perspectiveTransform(P.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        return P * sc

    return photo, fwd, {"mode": mode, "effects": fx}


def _line_quad(fwd, q, n=6):
    """Warp a line quad by sampling its top and bottom edges (a curled line is not a parallelogram any more);
    -> min-area-rect quad of the warped edge points (what a text detector would return)."""
    P = np.float64(q).reshape(4, 2)
    t = np.linspace(0, 1, n)[:, None]
    top = P[0] + (P[1] - P[0]) * t; bot = P[3] + (P[2] - P[3]) * t
    W_ = fwd(np.concatenate([top, bot]))
    r = cv2.minAreaRect(W_.astype(np.float32))
    box = cv2.boxPoints(r)
    # order: tl, tr, br, bl by angle around the centre, starting at the top-left
    c = box.mean(0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    box = box[np.argsort(ang)]  # -pi.. : tl(-135), tr(-45), br(45), bl(135)
    return box


def make_page(job):
    idx, seed, out = job
    rng = random.Random(seed)
    fake = G["fake"]; fake.seed_instance(seed)
    tpls = G["contract"] if rng.random() < G["contract_p"] else G["other"]
    names = list(tpls)
    name = rng.choices(names, [tpls[n][1] for n in names])[0]
    try:
        pg, ctx = tpls[name][0](rng, fake)
        img = np.asarray(pg.img)
        photo, fwd, info = phone_photo(img, rng)
    except Exception as e:
        return {"error": f"{name}: {e!r}", "tb": traceback.format_exc(limit=5), "idx": idx}
    h, w = photo.shape[:2]
    lines, texts, fonts, st = [], [], [], []
    if len(pg.struct) != len(pg.lines):
        return {"error": f"{name}: struct/lines mismatch", "tb": "", "idx": idx}
    for q, t, f, s in zip(pg.lines, pg.line_texts, pg.line_fonts, pg.struct):
        Q = _line_quad(fwd, q)
        x0, y0 = Q.min(0); x1, y1 = Q.max(0)
        if x1 < 2 or y1 < 2 or x0 > w - 2 or y0 > h - 2:
            continue  # out of frame
        vis = (min(w, x1) - max(0, x0)) * (min(h, y1) - max(0, y0)) / max(1e-6, (x1 - x0) * (y1 - y0))
        s = dict(s)
        s["cut"] = bool(vis < 0.98)
        if vis < 0.4:
            continue
        Q[:, 0] = Q[:, 0].clip(0, w); Q[:, 1] = Q[:, 1].clip(0, h)
        lines.append([round(float(v), 1) for v in Q.reshape(-1)]); texts.append(t); fonts.append(f); st.append(s)
    objs = []
    for cls, P in pg.objs:
        T = fwd(P)
        x0, y0 = T.min(0); x1, y1 = T.max(0)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if cx1 - cx0 >= 3 and cy1 - cy0 >= 3:
            objs.append([int(cls), round(float(cx0), 1), round(float(cy0), 1), round(float(cx1), 1), round(float(cy1), 1)])
    fn = f"sp_{idx:07d}.jpg"
    cv2.imwrite(str(Path(out) / "images" / fn), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"image": f"images/{fn}", "w": w, "h": h, "source": "struct_v2", "doc_type": name, "lang": ctx.lang, "lines": lines,
            "line_texts": texts, "line_fonts": fonts, "objs": objs, "seed": seed, "aug": info, "struct": st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--contract-p", type=float, default=0.7); ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    init(a.contract_p)
    jobs = [(i, a.seed * 1_000_003 + i, str(a.out)) for i in range(a.start, a.start + a.n)]
    recs, errs, t1 = [], 0, time.time()
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=300) as pool:
        for k, r in enumerate(pool.imap_unordered(make_page, jobs, chunksize=2)):
            if "error" in r:
                errs += 1
                if errs <= 10:
                    print("   skipped:", r["error"], "\n", r["tb"], flush=True)
            else:
                recs.append(r)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{a.n} pages {time.time() - t1:.0f}s errors={errs}", flush=True)
    recs.sort(key=lambda r: r["image"])
    with open(a.out / "manifest.jsonl", "a" if a.start else "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"done: {len(recs)} pages ({errs} errors) in {time.time() - t1:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
