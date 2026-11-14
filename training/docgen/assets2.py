"""Procedural assets for docgen v2 (all synthetic, no real people / real emblems).

Reuses the old generator's makers (det/assets.py: signatures, stamps, QR, barcode, chart) and adds:
synthetic portrait silhouettes (photo class), procedural emblems/logos, parameterised stamps, map/diagram figures,
math formulas (matplotlib mathtext, bundled DejaVu/STIX/CM fonts).
"""

from __future__ import annotations

import io
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .res import ASSETS, font, font_ok
from det import assets as A  # old repo (on sys.path via res)

# scene pictures from the old asset pool WITHOUT people / logos / document scans (checked by eye, see DATA_SOURCES.md)
SCENES = ("KHM_AGR_002_p0_7147 KHM_AGR_002_p0_7148 KHM_AGR_002_p149_1953 KHM_AGR_002_p156_2068 KHM_AGR_002_p241_3196 "
          "KHM_AGR_003_p47_140 KHM_AGR_003_p67_189 KHM_AGR_003_p96_255 KHM_EDU_002_p10_429 KHM_EDU_002_p122_749 "
          "KHM_EDU_002_p122_750 KHM_EDU_002_p29_505 KHM_EDU_002_p31_512 KHM_EDU_002_p31_513 KHM_EDU_002_p95_681 "
          "KHM_EDU_003_p12_89 KHM_EDU_003_p12_91 KHM_EDU_003_p18_117 KHM_EDU_003_p20_126 KHM_EDU_003_p24_144 "
          "KHM_EDU_003_p29_269 KHM_EDU_003_p34_293 KHM_EDU_003_p34_296 KHM_EDU_003_p67_390 KHM_EDU_003_p77_425 "
          "KHM_EDU_003_p77_427 KHM_EDU_005_p38_316 KHM_EDU_005_p38_317 KHM_EDU_005_p43_326 KHM_EDU_005_p43_328 "
          "KHM_EDU_005_p68_386 KHM_EDU_005_p68_387 KHM_EDU_005_p68_389 KHM_NGO_001_p5_83").split()
SCENE_PATHS = [str(ASSETS / "photos" / f"{s}.jpg") for s in SCENES]

INKS = A.INKS
STAMP_INKS = A.STAMP_INKS
SKIN = [(224, 188, 160), (198, 150, 115), (172, 124, 92), (140, 98, 70), (236, 204, 180), (120, 84, 60)]
HAIR = [(20, 18, 18), (40, 30, 25), (60, 45, 35), (15, 15, 20), (90, 70, 50), (130, 130, 130)]


def crop_alpha(img: Image.Image, thr=12) -> Image.Image:
    return A._crop_alpha(img, thr, pad=0)


# --------------------------------------------------------------------------- portrait (photo class)
def make_portrait(rng: random.Random, w: int, h: int) -> Image.Image:
    """Synthetic ID-photo silhouette: plain backdrop, shoulders, neck, head, hair, simple features. Not a real face."""
    S = 2
    W, H = w * S, h * S
    bg = rng.choice([(255, 255, 255), (230, 236, 245), (200, 220, 245), (90, 140, 210), (235, 235, 235), (190, 40, 50), (245, 245, 240)])
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    if rng.random() < 0.5:  # backdrop gradient
        g = np.linspace(0, 1, H)[:, None, None]
        a = np.asarray(im, np.float32) * (1 - 0.18 * g)
        im = Image.fromarray(a.astype(np.uint8))
        d = ImageDraw.Draw(im)
    skin, hair = rng.choice(SKIN), rng.choice(HAIR)
    shirt = rng.choice([(30, 40, 70), (240, 240, 240), (20, 20, 20), (60, 70, 90), (120, 30, 40), (200, 200, 210), (40, 80, 60)])
    cx = W * rng.uniform(0.46, 0.54)
    hw, hh = W * rng.uniform(0.2, 0.26), H * rng.uniform(0.22, 0.27)
    hy = H * rng.uniform(0.36, 0.42)
    # shoulders / body
    d.ellipse([cx - W * 0.55, H * 0.72, cx + W * 0.55, H * 1.45], fill=shirt)
    if rng.random() < 0.6:  # collar
        d.polygon([(cx - W * 0.12, H * 0.74), (cx, H * 0.86), (cx + W * 0.12, H * 0.74)], fill=(245, 245, 245))
        if rng.random() < 0.5:
            d.polygon([(cx - W * 0.025, H * 0.8), (cx + W * 0.025, H * 0.8), (cx + W * 0.04, H), (cx - W * 0.04, H)], fill=rng.choice([(150, 20, 30), (20, 40, 90), (60, 60, 60)]))
    d.rectangle([cx - hw * 0.45, hy + hh * 0.6, cx + hw * 0.45, H * 0.8], fill=tuple(int(c * 0.93) for c in skin))
    # hair back
    style = rng.random()
    if style < 0.4:  # long hair
        d.ellipse([cx - hw * 1.18, hy - hh * 1.12, cx + hw * 1.18, hy + hh * 1.5], fill=hair)
    d.ellipse([cx - hw, hy - hh, cx + hw, hy + hh], fill=skin)
    d.ellipse([cx - hw * 1.18, hy - hh * 0.25, cx - hw * 0.92, hy + hh * 0.25], fill=skin)  # ears
    d.ellipse([cx + hw * 0.92, hy - hh * 0.25, cx + hw * 1.18, hy + hh * 0.25], fill=skin)
    # hair top
    d.chord([cx - hw * 1.06, hy - hh * 1.12, cx + hw * 1.06, hy + hh * rng.uniform(0.1, 0.5)], 180, 360, fill=hair)
    # features (very simple, cartoon-like)
    ey = hy + hh * 0.05
    for sx in (-1, 1):
        ex = cx + sx * hw * 0.38
        d.ellipse([ex - hw * 0.1, ey - hh * 0.045, ex + hw * 0.1, ey + hh * 0.045], fill=(40, 30, 30))
        d.line([ex - hw * 0.16, ey - hh * 0.16, ex + hw * 0.14, ey - hh * 0.17], fill=hair, width=max(2, int(W * 0.012)))
    d.line([cx, ey + hh * 0.1, cx - hw * 0.06, ey + hh * 0.35], fill=tuple(int(c * 0.8) for c in skin), width=max(2, int(W * 0.01)))
    d.arc([cx - hw * 0.3, ey + hh * 0.3, cx + hw * 0.3, ey + hh * 0.62], 20, 160, fill=(120, 60, 60), width=max(2, int(W * 0.012)))
    if rng.random() < 0.2:  # glasses
        for sx in (-1, 1):
            ex = cx + sx * hw * 0.38
            d.ellipse([ex - hw * 0.22, ey - hh * 0.15, ex + hw * 0.22, ey + hh * 0.15], outline=(30, 30, 30), width=max(2, int(W * 0.01)))
    im = im.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    a = np.asarray(im, np.float32) + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(1, 6), (h, w, 3))
    if rng.random() < 0.25:
        a = np.repeat(a.mean(2, keepdims=True), 3, 2)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- logos / emblems
LOGO_COLS = [(200, 30, 40), (20, 90, 170), (20, 130, 80), (230, 150, 20), (90, 40, 140), (30, 30, 30), (0, 130, 150), (180, 140, 30), (25, 50, 110)]


def make_emblem(rng: random.Random, size: int) -> Image.Image:
    """Generic official-looking emblem: rays / wreath / tiered tower / lotus. Procedural, not a real coat of arms."""
    S = 3
    D = size * S
    im = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = D / 2
    col = rng.choice([(190, 150, 40), (30, 60, 130), (20, 20, 20), (160, 30, 40), (40, 100, 60)])
    col2 = rng.choice([(30, 60, 130), (190, 150, 40), (200, 40, 40), (255, 255, 255)])
    kind = rng.choice(["rays", "wreath", "tower", "lotus", "seal"])
    if kind in ("rays", "seal"):
        n = rng.randint(16, 36)
        for i in range(n):
            a = 2 * math.pi * i / n
            r0, r1 = D * 0.28, D * 0.48
            d.polygon([(c + r0 * math.cos(a - 0.06), c + r0 * math.sin(a - 0.06)), (c + r1 * math.cos(a), c + r1 * math.sin(a)),
                       (c + r0 * math.cos(a + 0.06), c + r0 * math.sin(a + 0.06))], fill=col)
        d.ellipse([c - D * 0.3, c - D * 0.3, c + D * 0.3, c + D * 0.3], fill=col2 if kind == "rays" else (255, 255, 255, 255), outline=col, width=D // 40)
        d.ellipse([c - D * 0.2, c - D * 0.2, c + D * 0.2, c + D * 0.2], outline=col, width=D // 50)
    if kind in ("wreath", "seal"):
        for side in (-1, 1):
            for i in range(9):
                a = math.radians(110 + i * 17) if side < 0 else math.radians(70 - i * 17)
                px, py = c + D * 0.38 * math.cos(a), c + D * 0.38 * math.sin(a) * -1 + D * 0.05
                d.ellipse([px - D * 0.05, py - D * 0.025, px + D * 0.05, py + D * 0.025], fill=(40, 120, 50) if kind == "wreath" else col)
    if kind in ("tower", "seal", "wreath"):  # tiered tower (generic temple silhouette)
        tw = D * rng.uniform(0.18, 0.26)
        base = c + D * 0.18
        for i in range(5):
            ww = tw * (1 - i * 0.17)
            d.rectangle([c - ww / 2, base - (i + 1) * D * 0.06, c + ww / 2, base - i * D * 0.06], fill=col if i % 2 == 0 else col2)
        d.polygon([(c - tw * 0.15, base - 5 * D * 0.06), (c, base - 5 * D * 0.06 - D * 0.1), (c + tw * 0.15, base - 5 * D * 0.06)], fill=col)
        if rng.random() < 0.6:
            for sx in (-1, 1):
                x = c + sx * tw * 0.9
                for i in range(3):
                    ww = tw * 0.45 * (1 - i * 0.25)
                    d.rectangle([x - ww / 2, base - (i + 1) * D * 0.045, x + ww / 2, base - i * D * 0.045], fill=col)
        d.rectangle([c - tw * 1.4, base, c + tw * 1.4, base + D * 0.04], fill=col)
    if kind == "lotus":
        for i in range(7):
            a = math.radians(-90 + (i - 3) * 22)
            px, py = c + D * 0.18 * math.cos(a), c + D * 0.05 + D * 0.18 * math.sin(a)
            d.ellipse([px - D * 0.09, py - D * 0.2, px + D * 0.09, py + D * 0.08], fill=col if i % 2 else col2, outline=col, width=D // 80)
        d.rectangle([c - D * 0.25, c + D * 0.2, c + D * 0.25, c + D * 0.26], fill=col)
    if rng.random() < 0.5:  # ribbon below
        d.polygon([(c - D * 0.42, c + D * 0.32), (c + D * 0.42, c + D * 0.32), (c + D * 0.36, c + D * 0.42), (c - D * 0.36, c + D * 0.42)], fill=col2 if col2 != (255, 255, 255) else col)
    return crop_alpha(im.resize((size, size), Image.LANCZOS))


def make_corp_logo(rng: random.Random, fonts: list[str], size: int, name: str | None = None) -> Image.Image:
    """Company mark: geometric shape(s) + optional monogram. Wordmarks are drawn as page text by the caller."""
    # (the old A.make_logo may add an unlabelled wordmark -> not used; wordmarks are drawn as labelled page text)
    return crop_alpha(_shape_logo(rng, fonts, size))


def _shape_logo(rng, fonts, size):
    S = 3
    D = size * S
    im = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c1, c2 = rng.sample(LOGO_COLS, 2)
    k = rng.choice(["rings", "hex", "triangles", "square", "swoosh", "bars", "mono"])
    if k == "rings":
        for i in range(rng.randint(2, 3)):
            r = D * (0.45 - i * 0.12)
            d.ellipse([D / 2 - r, D / 2 - r, D / 2 + r, D / 2 + r], outline=c1 if i % 2 == 0 else c2, width=D // 14)
    elif k == "hex":
        pts = [(D / 2 + D * 0.46 * math.cos(math.pi / 3 * i + math.pi / 6), D / 2 + D * 0.46 * math.sin(math.pi / 3 * i + math.pi / 6)) for i in range(6)]
        d.polygon(pts, fill=c1)
        pts2 = [(D / 2 + D * 0.24 * math.cos(math.pi / 3 * i + math.pi / 6), D / 2 + D * 0.24 * math.sin(math.pi / 3 * i + math.pi / 6)) for i in range(6)]
        d.polygon(pts2, fill=c2)
    elif k == "triangles":
        d.polygon([(D * 0.05, D * 0.9), (D * 0.45, D * 0.1), (D * 0.7, D * 0.9)], fill=c1)
        d.polygon([(D * 0.35, D * 0.9), (D * 0.7, D * 0.3), (D * 0.95, D * 0.9)], fill=c2)
    elif k == "square":
        d.rounded_rectangle([D * 0.05, D * 0.05, D * 0.95, D * 0.95], radius=D // 6, fill=c1)
        d.ellipse([D * 0.3, D * 0.3, D * 0.7, D * 0.7], fill=(255, 255, 255))
    elif k == "swoosh":
        d.ellipse([D * 0.05, D * 0.25, D * 0.95, D * 0.75], outline=c1, width=D // 12)
        d.ellipse([D * 0.35, D * 0.05, D * 0.65, D * 0.35], fill=c2)
    elif k == "bars":
        for i in range(4):
            hgt = D * (0.3 + 0.18 * i)
            d.rectangle([D * (0.06 + i * 0.23), D * 0.95 - hgt, D * (0.22 + i * 0.23), D * 0.95], fill=c1 if i % 2 == 0 else c2)
    else:
        d.ellipse([D * 0.03, D * 0.03, D * 0.97, D * 0.97], fill=c1)
    if k in ("square", "mono", "hex") and rng.random() < 0.7:
        mono = "".join(rng.choice("ABCDEFGHKLMNPRSTW") for _ in range(rng.choice([1, 1, 2])))
        cand = [p for p in fonts if font_ok(p, mono)] or fonts
        f = font(rng.choice(cand), int(D * (0.42 if len(mono) == 1 else 0.3)))
        bb = f.getbbox(mono)
        d.text((D / 2 - (bb[0] + bb[2]) / 2, D / 2 - (bb[1] + bb[3]) / 2), mono, font=f, fill=c2 if k != "square" else c1)
    return im.resize((size, size), Image.LANCZOS)


# --------------------------------------------------------------------------- stamps
def make_round_stamp(rng: random.Random, fonts: list[str], d: int, top: str, bottom: str | None, centre: list[str]) -> Image.Image:
    S = 3
    D = d * S
    c = D / 2
    mask = Image.new("L", (D, D), 0)
    dr = ImageDraw.Draw(mask)
    R = D / 2 - 3 * S
    th = max(2, int(rng.uniform(0.014, 0.035) * D))
    dr.ellipse([c - R, c - R, c + R, c + R], outline=255, width=th)
    if rng.random() < 0.4:
        r2 = R - th * 2.2
        dr.ellipse([c - r2, c - r2, c + r2, c + r2], outline=255, width=max(1, th // 2))
    inner = R * rng.uniform(0.55, 0.7)
    dr.ellipse([c - inner, c - inner, c + inner, c + inner], outline=255, width=max(2, int(th * rng.uniform(0.5, 1.0))))
    ring = (R + inner) / 2 - th
    fs = (R - inner) * rng.uniform(0.42, 0.58)
    cand = [f for f in fonts if font_ok(f, top)]
    if cand:
        A._arc_text(mask, top, font(rng.choice(cand), int(fs)), c, c, ring, -90, False)
    if bottom:
        cand = [f for f in fonts if font_ok(f, bottom)]
        if cand:
            A._arc_text(mask, bottom, font(rng.choice(cand), int(fs * 0.85)), c, c, ring, 90, True)
    if centre and rng.random() < 0.7:
        for j, t in enumerate(centre[:2]):
            cand = [f for f in fonts if font_ok(f, t)]
            if cand:
                f = font(rng.choice(cand), int(inner * 0.26))
                dr.text((c - f.getlength(t) / 2, c - inner * 0.42 + j * inner * 0.48), t, font=f, fill=255)
    else:
        rr = inner * 0.55
        pts = [(c + (rr if i % 2 == 0 else rr * 0.4) * math.cos(math.pi / 5 * i - math.pi / 2),
                c + (rr if i % 2 == 0 else rr * 0.4) * math.sin(math.pi / 5 * i - math.pi / 2)) for i in range(10)]
        dr.polygon(pts, fill=255)
    m = A._ink_texture(np.asarray(mask), rng, rng.uniform(0.2, 0.85))
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 0.9), (d, d), interpolation=cv2.INTER_AREA)
    return A._rotate(A._rgba(m, rng.choice(STAMP_INKS), rng.uniform(0.65, 1.0)), rng.uniform(0, 360))


def make_rect_stamp(rng: random.Random, fonts: list[str], w: int, lines: list[str]) -> Image.Image | None:
    S = 3
    fs = int(w * S * rng.uniform(0.1, 0.15))
    cand = [p for p in fonts if all(font_ok(p, t) for t in lines)]
    if not cand:
        return None
    f = font(rng.choice(cand), fs)
    tw = max(f.getlength(t) for t in lines)
    W = int(max(w * S, tw + fs * 1.4))
    lh = int(fs * 1.5)
    H = lh * len(lines) + 2 * int(fs * 0.6)
    mask = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(mask)
    th = max(2, int(fs * 0.12))
    dr.rounded_rectangle([2, 2, W - 3, H - 3], radius=int(fs * rng.uniform(0, 0.5)), outline=255, width=th)
    if rng.random() < 0.4:
        dr.rounded_rectangle([2 + th * 2, 2 + th * 2, W - 3 - th * 2, H - 3 - th * 2], radius=2, outline=255, width=max(1, th // 2))
    for j, t in enumerate(lines):
        dr.text((W / 2 - f.getlength(t) / 2, int(fs * 0.5) + j * lh), t, font=f, fill=255)
    m = A._ink_texture(np.asarray(mask), rng, rng.uniform(0.2, 0.8))
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 0.8), (int(W / S), int(H / S)), interpolation=cv2.INTER_AREA)
    return A._rotate(A._rgba(m, rng.choice(STAMP_INKS), rng.uniform(0.65, 1.0)), rng.uniform(-14, 14))


def real_ink(paths: list[str], rng: random.Random, inks=None) -> Image.Image | None:
    """Real signature/stamp crop as an RGBA ink layer. ~15% of the signature and ~60% of the stamp assets store a
    near-white RGB under their alpha (invisible when multiplied onto paper) -> those are re-inked with a random ink
    colour so that every labelled box has visible ink."""
    try:
        im = Image.open(rng.choice(paths)).convert("RGBA")
    except Exception:
        return None
    a = np.asarray(im).copy()
    m = a[..., 3] > 100
    if not m.any():
        return None
    if a[..., :3][m].mean() > 150:
        col = np.array(rng.choice(inks or (STAMP_INKS + INKS)), np.uint8)
        a[..., :3] = col
        im = Image.fromarray(a, "RGBA")
    return im


# --------------------------------------------------------------------------- codes
def make_qr(rng: random.Random, size: int, data: str | None = None, color=(0, 0, 0)) -> Image.Image:
    import qrcode
    q = qrcode.QRCode(box_size=4, border=0, error_correction=rng.choice([qrcode.constants.ERROR_CORRECT_L, qrcode.constants.ERROR_CORRECT_M]))
    q.add_data(data or "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789/:.-") for _ in range(rng.randint(12, 90))))
    q.make(fit=True)
    img = q.make_image(fill_color=color, back_color="white").convert("RGB")
    return img.resize((size, size), Image.NEAREST)


def make_barcode(rng: random.Random, w: int, h: int, code: str | None = None) -> Image.Image:
    """Bars only (no quiet zone, no text) -> box is exactly the bars."""
    import barcode
    from barcode.writer import ImageWriter
    code = code or "".join(rng.choice("0123456789") for _ in range(rng.choice([8, 10, 12, 14])))
    bc = barcode.get("code128", code, writer=ImageWriter())
    img = bc.render({"write_text": False, "module_height": 10, "module_width": 0.25, "quiet_zone": 0.5, "dpi": 150}).convert("L")
    a = np.asarray(img)
    ys, xs = np.nonzero(a < 128)
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    return img.resize((w, h), Image.NEAREST).convert("RGB")


def make_pdf417ish(rng: random.Random, w: int, h: int) -> Image.Image:
    """Stacked 2D code look (like the back of driving licences) - labelled qr_code (2D code)."""
    rows = rng.randint(10, 24)
    cols = rng.randint(30, 60)
    a = np.ones((rows, cols), np.uint8) * 255
    r = np.random.default_rng(rng.getrandbits(32))
    a[:, 3:-3] = np.where(r.random((rows, cols - 6)) < 0.5, 0, 255)
    a[:, :2] = 0
    a[:, -2:] = 0
    return Image.fromarray(a).resize((w, h), Image.NEAREST).convert("RGB")


# --------------------------------------------------------------------------- figures
def make_chart(rng: random.Random, w: int, h: int) -> Image.Image:
    return A.make_chart(rng, w, h)


def make_scene(rng: random.Random, w: int, h: int) -> Image.Image:
    if rng.random() < 0.8:
        try:
            im = Image.open(rng.choice(SCENE_PATHS)).convert("RGB")
            s = max(w / im.width, h / im.height)
            im = im.resize((max(w, int(im.width * s + 1)), max(h, int(im.height * s + 1))), Image.LANCZOS)
            x, y = rng.randint(0, im.width - w), rng.randint(0, im.height - h)
            return im.crop((x, y, x + w, y + h))
        except Exception:
            pass
    return A.make_photo(rng, w, h, None)


def make_diagram(rng: random.Random, w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    bw, bh = w // 4, h // 5
    boxes = []
    n = rng.randint(3, 6)
    for i in range(n):
        bx0 = int((w - bw) * (i / max(1, n - 1))) if rng.random() < 0.5 else rng.randint(0, w - bw)
        by0 = rng.randint(0, h - bh)
        fillc = rng.choice([(255, 255, 255), (220, 235, 250), (250, 235, 210), (225, 245, 225)])
        if rng.random() < 0.3:
            d.ellipse([bx0, by0, bx0 + bw, by0 + bh], outline=(40, 40, 40), width=2, fill=fillc)
        else:
            d.rectangle([bx0, by0, bx0 + bw, by0 + bh], outline=(40, 40, 40), width=2, fill=fillc)
        boxes.append((bx0, by0))
    for a, b in zip(boxes, boxes[1:]):
        d.line([a[0] + bw // 2, a[1] + bh // 2, b[0] + bw // 2, b[1] + bh // 2], fill=(80, 80, 80), width=2)
    d.rectangle([0, 0, w - 1, h - 1], outline=(150, 150, 150), width=1)
    return im


def make_map(rng: random.Random, w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h), rng.choice([(205, 225, 245), (240, 240, 235)]))
    d = ImageDraw.Draw(im)
    r = np.random.default_rng(rng.getrandbits(32))
    cx, cy = w / 2, h / 2
    n = 40
    rad = np.clip(min(w, h) * 0.42 * (1 + 0.25 * np.cumsum(r.normal(0, 0.12, n)) * 0.3), min(w, h) * 0.2, min(w, h) * 0.49)
    pts = [(cx + rad[i] * math.cos(2 * math.pi * i / n) * w / min(w, h) * 0.9, cy + rad[i] * math.sin(2 * math.pi * i / n)) for i in range(n)]
    d.polygon(pts, fill=(235, 230, 205), outline=(90, 90, 90))
    pal = [(170, 210, 160), (120, 180, 120), (60, 140, 80), (220, 220, 150), (240, 200, 140)]
    for _ in range(rng.randint(4, 9)):
        px, py = cx + r.normal(0, w * 0.15), cy + r.normal(0, h * 0.15)
        rr = r.uniform(0.05, 0.14) * min(w, h)
        poly = [(px + rr * r.uniform(0.6, 1.2) * math.cos(a), py + rr * r.uniform(0.6, 1.2) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)[:-1]]
        d.polygon(poly, fill=pal[int(r.integers(0, len(pal)))], outline=(110, 110, 110))
    d.line([(w * 0.1, h * 0.2), (w * 0.4, h * 0.45), (w * 0.55, h * 0.8)], fill=(60, 110, 200), width=3)
    d.rectangle([0, 0, w - 1, h - 1], outline=(60, 60, 60), width=2)
    return im


FORMULAS = [
    r"$E = mc^2$", r"$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$", r"$\sum_{i=1}^{n} x_i = n\bar{x}$", r"$\int_0^1 f(x)\,dx \approx \frac{1}{N}\sum_{k=1}^{N} f(x_k)$",
    r"$\sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$", r"$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$", r"$a^2 + b^2 = c^2$",
    r"$\frac{\partial L}{\partial w} = -\frac{1}{m}\sum_{j} (y_j - \hat{y}_j) x_j$", r"$F = G\frac{m_1 m_2}{r^2}$", r"$V = \frac{4}{3}\pi r^3$",
    r"$\lim_{x \to 0} \frac{\sin x}{x} = 1$", r"$A = P\left(1 + \frac{r}{n}\right)^{nt}$", r"$y = \beta_0 + \beta_1 x + \varepsilon$",
    r"$\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$", r"$e^{i\pi} + 1 = 0$", r"$f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(a)}{k!}(x-a)^k$",
    r"$\mathrm{CO_2} + \mathrm{H_2O} \rightarrow \mathrm{H_2CO_3}$", r"$S = \frac{n}{2}(a_1 + a_n)$", r"$\Delta G = \Delta H - T\Delta S$",
    r"$\hat{\theta} = \arg\max_{\theta} \prod_{i} p(x_i|\theta)$", r"$\mathbf{A}\mathbf{x} = \lambda \mathbf{x}$", r"$v = v_0 + at$",
    r"$\binom{n}{k} = \frac{n!}{k!(n-k)!}$", r"$\log_a(xy) = \log_a x + \log_a y$", r"$\cos^2\theta + \sin^2\theta = 1$",
]


def _rand_formula(rng):
    v = rng.choice("xyzabtnk")
    c = [rng.randint(2, 9) for _ in range(4)]
    return rng.choice([
        rf"${c[0]}{v}^2 + {c[1]}{v} - {c[2]} = 0$", rf"$\frac{{{c[0]}{v} + {c[1]}}}{{{c[2]}}} = {c[3]}$",
        rf"$\sqrt{{{c[0]}{v} + {c[1]}}} \leq {c[2]}$", rf"${v}_{{n+1}} = {c[0]}{v}_n + {c[1]}$",
        rf"$\int_{{0}}^{{{c[0]}}} {v}^{{{c[1]}}}\,d{v} = \frac{{{c[0]}^{{{c[1] + 1}}}}}{{{c[1] + 1}}}$"])


def make_formula(rng: random.Random, height_px: int, color=(0, 0, 0)) -> Image.Image | None:
    """Rendered with matplotlib mathtext (fonts bundled with matplotlib: DejaVu (Bitstream Vera licence), STIX (OFL), CM (BaKoMa))."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties
    s = rng.choice(FORMULAS) if rng.random() < 0.7 else _rand_formula(rng)
    fs = rng.choice(["cm", "stix", "dejavuserif", "dejavusans", "stixsans"])
    buf = io.BytesIO()
    try:
        with matplotlib.rc_context({"mathtext.fontset": fs}):
            mathtext.math_to_image(s, buf, prop=FontProperties(size=height_px * 0.55), dpi=100, format="png",
                                   color=tuple(c / 255 for c in color))
    except Exception:
        return None
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")
    a = np.asarray(im)
    alpha = a[..., 3].copy()
    if alpha.min() > 250 or alpha.max() == 0:  # opaque white background -> ink = darkness
        g = a[..., :3].astype(np.float32).mean(2)
        alpha = np.clip((255 - g) * 1.15, 0, 255).astype(np.uint8)
    rgb = np.zeros_like(a[..., :3])
    rgb[:] = color
    return crop_alpha(Image.fromarray(np.dstack([rgb, alpha]), "RGBA"))
