"""Capture simulation for docgen pages, label-exact.

apply(img_rgb_uint8, rng) -> (out_img, Hm 3x3 float64, info dict)
Every geometric change is ONE homography Hm (page px -> output px); the caller maps line quads and object point
sets through Hm. Photometric effects (stains, folds, fax, blur, JPEG, noise, lighting) never move pixels.
Based on the old det/degrade.py (scan / phone photo / lighting / shadow) plus stains, folds, fax, low-res phone,
JPEG artefacts and small whole-page rotations.
apply(..., profile="exam_phone", meta=page.meta + label geometry) -> apply_exam_phone (handwritten exam sheets).
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from det.degrade import _noise_field, make_background  # old repo (on sys.path via res)


def _stains(out, rng, rs):
    H, W = out.shape[:2]
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.04, 0.16) * min(W, H)
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - cx) ** 2 + ((yy - cy) * rng.uniform(0.7, 1.3)) ** 2) / r
        n = _noise_field(H, W, rs, 6) * 0.5
        if rng.random() < 0.5:  # coffee ring
            m = np.exp(-((dist + n - 1.0) ** 2) / 0.006) * 0.6 + (dist + n < 1.0) * 0.15
        else:  # blotch
            m = np.clip(1.2 - dist - n, 0, 1) ** 0.6 * 0.45
        col = np.array(rng.choice([(150, 110, 60), (170, 140, 90), (120, 120, 110), (180, 160, 120)]), np.float32) / 255
        a = (m * rng.uniform(0.3, 0.8))[..., None].astype(np.float32)
        out = out * (1 - a * (1 - col))
    return out


def _fold(out, rng):
    H, W = out.shape[:2]
    vertical = rng.random() < 0.5
    n = rng.choice([1, 1, 2])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for k in range(n):
        if vertical:
            p = W * rng.uniform(0.3, 0.7) if n == 1 else W * (k + 1) / (n + 1)
            d = xx - p + (yy - H / 2) * rng.uniform(-0.02, 0.02)
        else:
            p = H * rng.uniform(0.3, 0.7) if n == 1 else H * (k + 1) / (n + 1)
            d = yy - p + (xx - W / 2) * rng.uniform(-0.02, 0.02)
        s = rng.uniform(1.5, 4)
        crease = -np.exp(-(d / s) ** 2) * rng.uniform(0.15, 0.35) + np.exp(-((d - s * 2) / (s * 1.5)) ** 2) * 0.08
        side = np.where(d > 0, rng.uniform(-0.06, 0.0), 0.0)
        out = out * (1 + crease + side)[..., None]
        vertical = not vertical if rng.random() < 0.5 else vertical
    return out


def apply(img: np.ndarray, rng: random.Random, profile: str | None = None, meta: dict | None = None):
    if profile == "exam_phone":
        if rng.random() < 0.92:
            return apply_exam_phone(img, rng, meta or {})
        # a few scans / flat captures of the same sheets: fall through to the generic pipeline
    rs = np.random.default_rng(rng.getrandbits(32))
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    fx: list[str] = []

    # ---- flat-page effects
    if rng.random() < 0.12:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = np.clip((g - 110) * rng.uniform(1.3, 2.2) + 140, 0, 255)
        out = np.repeat(g[..., None], 3, 2)
        fx.append("photocopy")
    if rng.random() < 0.6:
        tint = np.array([rng.uniform(0.92, 1.0), rng.uniform(0.92, 1.0), rng.uniform(0.84, 1.0)], np.float32)
        out = out * tint * (0.94 + 0.06 * _noise_field(H, W, rs, 6)[..., None])
        fx.append("paper")
    if rng.random() < 0.15:
        out = _stains(out, rng, rs)
        fx.append("stain")
    if rng.random() < 0.18:
        out = _fold(out, rng)
        fx.append("fold")
    if rng.random() < 0.25:
        out = 255 - (255 - out) * rng.uniform(0.6, 0.9)
        fx.append("faded")
    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
        fx.append("blur")

    # ---- geometry (one homography)
    mode = rng.choices(["scan", "photo", "rotate", "flat"], [0.42, 0.3, 0.13, 0.15])[0]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    if mode == "flat":
        Wc, Hc, dst = W, H, src.copy()
    elif mode in ("scan", "rotate"):
        m = rng.choice([0, 0, 12, 30]) if mode == "scan" else rng.randint(20, 60)
        ang = np.deg2rad(rng.choice([0, rng.uniform(-1.2, 1.2), rng.uniform(-3.0, 3.0)]) if mode == "scan" else rng.uniform(-6, 6))
        # enlarge the canvas so the rotated page fits
        Wc = int(abs(W * np.cos(ang)) + abs(H * np.sin(ang))) + 2 * m
        Hc = int(abs(W * np.sin(ang)) + abs(H * np.cos(ang))) + 2 * m
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        dst = ((src - [W / 2, H / 2]) @ R.T + [Wc / 2, Hc / 2]).astype(np.float32)
    else:
        s = rng.uniform(0.55, 0.85)
        Wc, Hc = int(W * s * rng.uniform(1.12, 1.4)), int(H * s * rng.uniform(1.08, 1.25))
        ang = np.deg2rad(rng.uniform(-10, 10))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        q = ((src - [W / 2, H / 2]) * s) @ R.T
        jit = rng.uniform(0.02, 0.08)
        q = q + np.array([[rs.uniform(-jit, jit) * W * s, rs.uniform(-jit, jit) * H * s] for _ in range(4)])
        dst = (q + [Wc / 2 + rng.uniform(-0.03, 0.03) * Wc, Hc / 2 + rng.uniform(-0.03, 0.03) * Hc]).astype(np.float32)
        # keep the page inside the frame
        lo, hi = dst.min(0), dst.max(0)
        shift = np.maximum(0, 4 - lo) - np.maximum(0, hi - [Wc - 4, Hc - 4])
        dst = (dst + shift).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    if mode != "flat":
        warped = cv2.warpPerspective(out, Hm, (Wc, Hc), flags=cv2.INTER_AREA if mode == "photo" else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpPerspective(np.ones((H, W), np.float32), Hm, (Wc, Hc), flags=cv2.INTER_LINEAR)[..., None]
        if mode == "photo":
            bg = make_background(Hc, Wc, rng, rs)
        else:
            bg = np.ones((Hc, Wc, 3), np.float32) * np.float32(rng.choice([(20, 20, 20), (45, 45, 45), (235, 235, 235), (255, 255, 255), (90, 90, 90), (200, 190, 170)]))
        canvas = bg * (1 - mask) + warped * mask
        fx.append(mode)
    else:
        canvas = out

    # ---- lighting / shadow
    if mode in ("scan", "photo") or rng.random() < 0.2:
        yy, xx = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
        gx, gy = rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18)
        light = 1.0 + gx * (xx / Wc - 0.5) * 2 + gy * (yy / Hc - 0.5) * 2
        if mode == "photo":
            dd = np.sqrt(((xx / Wc - 0.5) ** 2 + (yy / Hc - 0.5) ** 2)) * rng.uniform(0.1, 0.45)
            light = light * (1 - dd)
        if rng.random() < 0.35:
            sh = np.zeros((Hc, Wc), np.float32)
            p0 = np.array([rng.uniform(0, Wc), rng.uniform(0, Hc)])
            p1 = p0 + np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)]) * max(Wc, Hc)
            cv2.fillPoly(sh, [np.array([p0, p1, [p1[0] + rng.uniform(-Wc, Wc) * 0.5, p1[1] + Hc * 0.6],
                                        [p0[0] + rng.uniform(-Wc, Wc) * 0.5, p0[1] + Hc * 0.6]], np.int32)], 1.0)
            light = light * (1 - cv2.GaussianBlur(sh, (0, 0), rng.uniform(20, 70)) * rng.uniform(0.1, 0.4))
            fx.append("shadow")
        canvas = canvas * np.clip(light, 0.4, 1.25)[..., None]
        if mode == "photo" and rng.random() < 0.5:
            canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.5, 1.6))
    canvas = canvas + rs.normal(0, rng.choice([0, 2, 3, 6]), canvas.shape).astype(np.float32)
    outimg = np.clip(canvas, 0, 255).astype(np.uint8)

    # ---- capture / transmission
    r = rng.random()
    if r < 0.07:  # fax: bitonal, low vertical resolution, streaks
        g = cv2.cvtColor(outimg, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (Wc // 2, Hc // 2), interpolation=cv2.INTER_AREA)
        thr = rng.randint(120, 175)
        b = np.where(g.astype(np.int16) + rs.normal(0, 8, g.shape) > thr, 255, 0).astype(np.uint8)
        for _ in range(rng.randint(0, 4)):
            y0 = rng.randrange(b.shape[0])
            b[y0:y0 + rng.randint(1, 2), :] = rng.choice([0, 255])
        b = cv2.resize(b, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        outimg = np.repeat(b[..., None], 3, 2)
        fx.append("fax")
    elif r < 0.22 and mode != "photo":  # low-resolution phone capture (photo mode is already downscaled)
        f = rng.uniform(0.35, 0.6)
        small = cv2.resize(outimg, (max(8, int(Wc * f)), max(8, int(Hc * f))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 70)])
        small = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if ok else small
        outimg = cv2.resize(small, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
        fx.append("lowres")
    elif r < 0.4:
        ok, buf = cv2.imencode(".jpg", outimg, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 65)])
        if ok:
            outimg = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            fx.append("jpeg")
    return outimg, Hm, {"mode": mode, "effects": fx}


# =============================================================================================== exam_phone profile
def _paper_bg(Hc, Wc, rng, rs):
    """What is around / under the sheet in a close phone photo: desk, cloth, dark, or other sheets of the stack."""
    r = rng.random()
    if r < 0.6:  # other answer sheets underneath (light paper, faint rules)
        base = np.float32(rng.choice([(205, 210, 220), (225, 228, 232), (190, 195, 205), (215, 212, 205)]))
        bg = np.ones((Hc, Wc, 3), np.float32) * base
        sp = rng.uniform(25, 40)
        off = rng.uniform(0, sp)
        for y in np.arange(off, Hc, sp):
            bg[int(y):int(y) + 1] *= 0.85
        bg *= (0.9 + 0.1 * _noise_field(Hc, Wc, rs, 8))[..., None]
        return bg
    if r < 0.8:  # desk / dark surface, muted
        bg = make_background(Hc, Wc, rng, rs)
        g = bg.mean(2, keepdims=True)
        return g + (bg - g) * rng.uniform(0.2, 0.7)
    return make_background(Hc, Wc, rng, rs)


def _show_through(out, img, rng, rs):
    """Mirrored reverse-side ink (writing on the back of the sheet), blurred and faint. Unlabelled."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = np.clip((235 - g) / 235, 0, 1)
    dark = cv2.flip(dark, 1)
    H, W = dark.shape
    M = np.float32([[1, 0, rng.uniform(-40, 40)], [0, 1, rng.uniform(-60, 60)]])
    dark = cv2.warpAffine(dark, M, (W, H))
    dark = cv2.GaussianBlur(dark, (0, 0), rng.uniform(1.2, 3.0))
    a = rng.uniform(0.05, 0.16)
    tint = np.float32(rng.choice([(0.85, 0.88, 1.0), (1.0, 1.0, 1.0), (0.9, 0.9, 0.95)]))
    return out * (1 - (a * dark)[..., None] * (1 - tint * 0.6))


def apply_exam_phone(img: np.ndarray, rng: random.Random, meta: dict):
    """Close-up phone photo of a sheet on a desk / stack: page (almost) fills a 3:4 frame, mild perspective and
    rotation, edges / top header possibly cut (only at meta['safe_cuts'] gaps so no text line is cut in half),
    grey-blue paper tone, show-through, lighting gradient + vignette + soft shadows, defocus, sharpening halo,
    noise, JPEG. One homography Hm (page px -> output px)."""
    rs = np.random.default_rng(rng.getrandbits(32))
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    fx: list[str] = ["exam_phone"]

    # ---- flat-page photometric effects
    if rng.random() < 0.7:
        out = _show_through(out, img, rng, rs)
        fx.append("show_through")
    g = rng.uniform(0.76, 0.95)
    tr = rng.uniform(0.9, 0.98)
    tone = np.float32([g * tr, g * rng.uniform(max(tr, 0.94), 1.0), g * rng.uniform(1.0, 1.06)])  # grey-blue, never purple
    out = out * tone * (0.95 + 0.05 * _noise_field(H, W, rs, 5)[..., None])
    if rng.random() < 0.1:
        out = _fold(out, rng)
        fx.append("fold")
    if rng.random() < 0.08:
        out = _stains(out, rng, rs)
        fx.append("stain")

    # ---- geometry: frame 3:4, page roughly fills it
    Wc = int(W * rng.uniform(0.9, 1.0))
    Hc = int(Wc * 4 / 3)
    cuts = [c for c in meta.get("safe_cuts", []) if 0 < c < H * 0.35]
    line_pts = meta.get("line_pts")  # [N,4,2] labelled line quads (page px)
    obj_pts = meta.get("obj_pts") or []  # labelled object point sets

    def frame_ok(Hm, y_cut):
        """No labelled line may be half inside the frame; no object may be 2..60 % visible (generate.py drops
        objects < 60 % visible, so they must be either clearly in or completely out)."""
        if line_pts is not None and len(line_pts):
            P = cv2.perspectiveTransform(line_pts.reshape(-1, 1, 2), Hm).reshape(-1, 4, 2)
            ins = (P[..., 0] >= 1) & (P[..., 0] <= Wc - 1) & (P[..., 1] >= 1) & (P[..., 1] <= Hc - 1)
            if (ins.any(1) & ~ins.all(1)).any():
                return False
        for O in obj_pts:
            T = cv2.perspectiveTransform(O.reshape(-1, 1, 2), Hm).reshape(-1, 2)
            x0, y0 = T.min(0)
            x1, y1 = T.max(0)
            a0 = max(1e-6, (x1 - x0) * (y1 - y0))
            vis = max(0, min(Wc, x1) - max(0, x0)) * max(0, min(Hc, y1) - max(0, y0)) / a0
            if 0.02 < vis < 0.62:
                return False
        return True

    Hm = None
    for attempt in range(40):
        last = attempt == 39
        y_cut = float(rng.choice(cuts)) if (cuts and rng.random() < 0.35 and not last) else 0.0
        src = np.float32([[0, y_cut], [W, y_cut], [W, H], [0, H]])
        vis_h = H - y_cut
        # close-up: page about as wide as the frame (A4 is taller than 3:4, so top/bottom margins may leave the frame)
        s = (Wc / W) * rng.uniform(0.94, 1.1) if not last else min(Wc / W, Hc / vis_h) * 0.9
        ang = np.deg2rad(rng.uniform(-2.5, 2.5)) if not last else 0.0
        k = rng.uniform(-0.05, 0.05)  # phone tilt: top or bottom edge narrower
        kx = rng.uniform(-0.03, 0.03)
        R_ = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        q = ((src - [W / 2, y_cut + vis_h / 2]) * s) @ R_.T
        q[0, 0] += k * W * s
        q[1, 0] -= k * W * s
        q[0, 1] += kx * vis_h * s
        q[3, 1] -= kx * vis_h * s
        q += rs.uniform(-0.012, 0.012, (4, 2)) * [W * s, vis_h * s]
        dst = q + [Wc / 2 + rng.uniform(-0.04, 0.04) * Wc, Hc / 2]
        if y_cut > 0:  # the cut gap becomes the top edge of the frame (rows above it are outside)
            dst[:, 1] -= max(dst[0, 1], dst[1, 1]) + 2
        else:  # page top edge: a bit inside the frame .. well above it
            dst[:, 1] += (rng.uniform(-0.08, 0.03) if not last else 0.03) * Hc - min(dst[0, 1], dst[1, 1])
        dst = dst.astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        if last or frame_ok(Hm, y_cut):
            break
    if y_cut > 0:
        fx.append("top_cut")
    fx.append(f"fit{attempt}")
    warped = cv2.warpPerspective(out, Hm, (Wc, Hc), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.ones((H, W), np.float32), Hm, (Wc, Hc), flags=cv2.INTER_LINEAR)[..., None]
    bg = _paper_bg(Hc, Wc, rng, rs)
    canvas = bg * (1 - mask) + warped * mask
    # thin shadow line along page edges on the background
    if rng.random() < 0.6:
        edge = cv2.GaussianBlur(mask[..., 0], (0, 0), rng.uniform(3, 8))
        canvas = canvas * (1 - (np.clip(edge - mask[..., 0], 0, 1) * rng.uniform(0.2, 0.5))[..., None])

    # ---- lighting
    yy, xx = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    gx, gy = rng.uniform(-0.2, 0.2), rng.uniform(-0.22, 0.22)
    light = 1.0 + gx * (xx / Wc - 0.5) * 2 + gy * (yy / Hc - 0.5) * 2
    dd = np.sqrt((xx / Wc - rng.uniform(0.35, 0.65)) ** 2 + (yy / Hc - rng.uniform(0.35, 0.65)) ** 2)
    light = light * (1 - dd * rng.uniform(0.05, 0.35))
    for _ in range(rng.choice([0, 1, 1, 2])):  # soft shadows (phone / hand / head)
        sh = np.zeros((Hc, Wc), np.float32)
        side = rng.choice(["t", "b", "l", "r", "c"])
        if side == "c":
            c0 = (rng.uniform(0.2, 0.8) * Wc, rng.uniform(0.2, 0.8) * Hc)
            cv2.ellipse(sh, (int(c0[0]), int(c0[1])), (int(rng.uniform(0.15, 0.4) * Wc), int(rng.uniform(0.1, 0.3) * Hc)),
                        rng.uniform(0, 180), 0, 360, 1.0, -1)
        else:
            d0 = rng.uniform(0.05, 0.3)
            if side == "t":
                sh[: int(d0 * Hc)] = 1
            elif side == "b":
                sh[int((1 - d0) * Hc):] = 1
            elif side == "l":
                sh[:, : int(d0 * Wc)] = 1
            else:
                sh[:, int((1 - d0) * Wc):] = 1
        light = light * (1 - cv2.GaussianBlur(sh, (0, 0), rng.uniform(30, 90)) * rng.uniform(0.12, 0.4))
        fx.append("shadow")
    exp = rng.uniform(0.92, 1.15)
    canvas = canvas * np.clip(light * exp, 0.35, 1.3)[..., None]
    # white balance
    wb = np.float32([rng.uniform(0.92, 1.0), rng.uniform(0.97, 1.02), rng.uniform(0.98, 1.08)])
    wb[0] = min(wb[0], wb[1])  # cool / neutral, never purple
    canvas = canvas * wb
    if rng.random() < 0.15:  # glare spot
        c0 = (rng.uniform(0.2, 0.8) * Wc, rng.uniform(0.2, 0.8) * Hc)
        g = np.exp(-(((xx - c0[0]) / (Wc * rng.uniform(0.1, 0.25))) ** 2 + ((yy - c0[1]) / (Hc * rng.uniform(0.08, 0.2))) ** 2))
        canvas = canvas + (255 - canvas) * (g * rng.uniform(0.2, 0.5))[..., None]
        fx.append("glare")

    # ---- optics
    if rng.random() < 0.85:
        sig = rng.uniform(0.5, 1.4)
        bl = cv2.GaussianBlur(canvas, (0, 0), sig)
        if rng.random() < 0.4:  # depth-of-field: one side sharper
            t = (xx / Wc if rng.random() < 0.5 else yy / Hc)
            t = t if rng.random() < 0.5 else 1 - t
            wgt = np.clip(t * rng.uniform(1.0, 2.0) - 0.2, 0, 1)[..., None]
            bl2 = cv2.GaussianBlur(canvas, (0, 0), sig + rng.uniform(0.8, 1.8))
            bl = bl * (1 - wgt) + bl2 * wgt
            fx.append("dof")
        canvas = bl
        fx.append("blur")
    if rng.random() < 0.5:  # phone sharpening halo
        bl = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(1.5, 3))
        canvas = canvas + (canvas - bl) * rng.uniform(0.3, 0.9)
        fx.append("sharpen")
    canvas = canvas + rs.normal(0, rng.uniform(1.5, 4.5), canvas.shape).astype(np.float32)
    outimg = np.clip(canvas, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", outimg, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 85)])
        if ok:
            outimg = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            fx.append("jpeg")
    return outimg, Hm, {"mode": "exam_phone", "effects": fx}
