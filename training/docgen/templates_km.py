"""Cambodian document families (Khmer, often with English mixed in). All content synthetic; card designs generic."""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from . import blocks as B
from .canvas import Page, bbox, to_pts
from .ctx import KH_PROV, KH_PROV_EN, MINISTRIES, SUB_UNITS, KH_TITLES_LETTER, Ctx
from .res import font

KINGDOM = ["ព្រះរាជាណាចក្រកម្ពុជា", "ជាតិ សាសនា ព្រះមហាក្សត្រ"]
KH_PRODUCTS_EN = ["Fish Amok", "Lok Lak", "Num Banh Chok", "Kuy Teav", "Bai Sach Chrouk", "Iced Coffee", "Mango Sticky Rice", "Spring Rolls",
                  "Green Papaya Salad", "Lemongrass Chicken", "Fried Rice", "Pork Noodle Soup"]
PAPERS = [(255, 255, 255)] * 6 + [(252, 252, 250), (248, 245, 235), (245, 245, 245), (250, 247, 240)]
TINTS = [(236, 244, 250), (250, 244, 230), (238, 248, 238), (250, 238, 240), (244, 240, 250), (255, 250, 225)]


def new_page(rng, kind="a4", colour_p=0.12):
    W, H = {"a4": (1240, 1754), "letter": (1275, 1650), "a4l": (1754, 1240), "a5": (874, 1240)}[kind]
    paper = rng.choice(TINTS) if rng.random() < colour_p else rng.choice(PAPERS)
    pg = Page(W, H, paper)
    pg.meta["paper"] = paper
    return pg


def ornament(pg, cx, y, w, col):
    d = pg.d
    d.line([cx - w / 2, y, cx - 8, y], fill=col, width=2)
    d.line([cx + 8, y, cx + w / 2, y], fill=col, width=2)
    d.polygon([(cx, y - 7), (cx + 7, y), (cx, y + 7), (cx - 7, y)], fill=col)
    for s in (-1, 1):
        d.ellipse([cx + s * w / 2 - 4, y - 4, cx + s * w / 2 + 4, y + 4], outline=col, width=2)
    return (cx - w / 2 - 4, y - 7, cx + w / 2 + 5, y + 8)


def motto(pg, ctx, cx, y, size=None):
    rng = ctx.rng
    sz = int(size or ctx.size * rng.uniform(1.05, 1.25))
    p = ctx.font("moul", KINGDOM[0] + KINGDOM[1])
    col = ctx.fg if rng.random() < 0.8 else ctx.accent
    pg.begin("header")
    q1 = pg.text(cx, y, KINGDOM[0], p, sz, col, align="c")
    pg.end()
    pg.begin("header")
    q2 = pg.text(cx, y + sz * 1.9, KINGDOM[1], p, int(sz * 0.92), col, align="c")
    pg.end()
    qs = [q for q in (q1, q2) if q]
    if not qs:
        return y
    b = bbox(qs)
    pg.add("header", qs)
    orn = B.b_ornament(pg, ctx, cx, b[3] + sz * 0.8, sz * rng.uniform(4, 7), col) if rng.random() < 0.8 else None
    return (orn[3] if orn else b[3]) + sz * 0.5


def org_block(pg, ctx, x, y, logo_p=0.6, ref=True):
    """Emblem + ministry/sub-unit names + reference no. (top-left letterhead)."""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.95, 1.1))
    yy = y
    m = rng.choice(MINISTRIES)
    names = [m[0]] + ([rng.choice(SUB_UNITS)] if rng.random() < 0.6 else [])
    p = ctx.font("moul", "".join(names))
    w_text = max(font(p, sz).getlength(t) for t in names) if p else 300
    cx = x + w_text / 2
    if rng.random() < logo_p:
        ls = int(sz * rng.uniform(3, 4.4))
        B.b_logo(pg, ctx, cx - ls / 2, yy, ls, kind="emblem")
        yy += ls + sz * 0.4
    qs = []
    for i, t in enumerate(names):
        pg.begin("header")
        q = pg.text(cx, yy, t, p, int(sz * (1 if i == 0 else 0.9)), ctx.fg, align="c")
        pg.end()
        if q:
            qs.append(q)
            yy = bbox([q])[3] + sz * 0.6
    if ref:
        t = f"លេខ{rng.choice(['៖', ': ', ''])}{ctx.d(rng.randint(1, 2999))}{rng.choice(['', ' '])}{rng.choice(['អយក', 'សហវ', 'ប្រក', 'សជណ', 'អ.ក', 'ក.ប'])}{rng.choice(['.', '/'])}{rng.choice(['ចប', 'នស', 'អប', 'ស.ណ', 'វ.ក'])}"
        pg.begin("header")
        q = pg.text(cx, yy + sz * 0.2, t, ctx.font("body", t), int(sz * 0.9), ctx.fg, align="c")
        pg.end()
        if q:
            qs.append(q)
            yy = bbox([q])[3]
    if qs:
        pg.add("header", qs)
    return yy + sz * 0.6


def date_line(pg, ctx, x, y, w, align="r", lunar=None):
    rng = ctx.rng
    sz = int(ctx.size * 0.95)
    lines = []
    lunar = rng.random() < 0.45 if lunar is None else lunar
    if lunar:
        lines.append(f"ថ្ងៃ{rng.choice(['ច័ន្ទ', 'អង្គារ', 'ពុធ', 'ព្រហស្បតិ៍', 'សុក្រ'])} {ctx.d(rng.randint(1, 15))}{rng.choice(['កើត', 'រោច'])} ខែ{rng.choice(['ពិសាខ', 'ជេស្ឋ', 'អាសាឍ', 'ស្រាពណ៍', 'ភទ្របទ', 'អស្សុជ', 'កត្តិក', 'មិគសិរ'])} ឆ្នាំ{rng.choice(['រោង', 'ម្សាញ់', 'មមី', 'មមែ', 'វក', 'រកា'])} {rng.choice(['ឆស័ក', 'សប្តស័ក', 'អដ្ឋស័ក'])} ព.ស.{ctx.d(rng.randint(2560, 2571))}")
    lines.append(f"{rng.choice(['រាជធានីភ្នំពេញ', 'ខេត្តសៀមរាប', 'ខេត្តបាត់ដំបង', 'ខេត្តកំពត'])} ថ្ងៃទី{ctx.d(rng.randint(1, 28))} ខែ{rng.choice(['មករា', 'កុម្ភៈ', 'មីនា', 'មេសា', 'ឧសភា', 'មិថុនា', 'កក្កដា', 'សីហា', 'កញ្ញា', 'តុលា', 'វិច្ឆិកា', 'ធ្នូ'])} ឆ្នាំ{ctx.d(rng.randint(2018, 2027))}")
    qs, yy = [], y
    p = ctx.font("body", "".join(lines))
    for t in lines:
        pg.begin("text")
        q = pg.text(x + w if align == "r" else x + w / 2, yy, t, p, sz, ctx.fg, align=align)
        pg.end()
        if q:
            qs.append(q)
            yy = bbox([q])[3] + sz * 0.6
    if qs:
        pg.add("text", qs)
    return yy + sz * 0.3


# =========================================================================== 1. official letter
def km_official_letter(rng, fake):
    """Official letter family: the classic centred letterhead or the press-release / statement layout."""
    return km_press_release(rng, fake) if rng.random() < 0.5 else km_official_letter_classic(rng, fake)


def km_official_letter_classic(rng, fake):
    pg = new_page(rng, "a4", 0.08)
    ctx = Ctx(rng, "km", fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.07, 0.11))
    x, w = m, W - 2 * m
    y0 = int(H * rng.uniform(0.03, 0.05))
    if rng.random() < 0.06:
        B.watermark(pg, ctx)
    ym = motto(pg, ctx, x + w * rng.uniform(0.6, 0.68) if rng.random() < 0.3 else x + w / 2, y0)
    y = org_block(pg, ctx, x, ym - ctx.size * rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        y = date_line(pg, ctx, x, y - ctx.size * 2.5, w)
    title = rng.choice(KH_TITLES_LETTER)
    y = B.b_heading(pg, ctx, x, y + ctx.size * 0.5, w, cls="title", text=title, size=ctx.size * rng.uniform(1.3, 1.8), align="c", role="moul")
    if rng.random() < 0.25:
        o = B.b_ornament(pg, ctx, x + w / 2, y + ctx.size * 0.2, ctx.size * rng.uniform(4, 8))
        y = (o[3] if o else y) + ctx.size * 0.5
    if rng.random() < 0.7:  # subject / reference lines
        for pre in rng.sample(["កម្មវត្ថុ៖ ", "យោង៖ ", "សូមគោរពជូន "], rng.randint(1, 2)):
            y = B.b_para(pg, ctx, x, y, w, n=rng.randint(1, 3), text=pre + ctx.para(160), indent=0)
    ymax = H * rng.uniform(0.62, 0.72)
    y = B.fill_flow(pg, ctx, x, y, w, ymax, [("para", 6), ("list", 1.2), ("table", 0.5)], max_blocks=6)
    if rng.random() < 0.5:
        y = date_line(pg, ctx, x + w * 0.4, y, w * 0.6, "c")
    two = rng.random() < 0.2
    if two:
        B.b_sign(pg, ctx, x, y, w * 0.42, stamp_p=0.3)
        y = B.b_sign(pg, ctx, x + w * 0.55, y, w * 0.42, stamp_p=0.6)
    else:
        y = B.b_sign(pg, ctx, x + w * rng.uniform(0.45, 0.6), y, w * 0.4, stamp_p=0.65)
    if rng.random() < 0.5 and y < H - 200:  # copy-to list (bottom left)
        sz = int(ctx.size * 0.85)
        q = pg.text(x, y, "ចម្លងជូន៖", ctx.font("bold", "ចម្លងជូន៖"), sz, ctx.fg)
        if q:
            pg.add("text", [q])
            B.b_list(pg, ctx, x, bbox([q])[3] + sz * 0.4, w * 0.5, H - 120, items=[rng.choice(["ខុទ្ទកាល័យ", "ឯកសារ-កាលប្បវត្តិ", "ដូចប្រការ៣", "នាយកដ្ឋានពាក់ព័ន្ធ", "ក្រសួងពាក់ព័ន្ធ"]) + (" «ដើម្បីជ្រាប»" if rng.random() < 0.5 else "") for _ in range(rng.randint(2, 4))], size=sz, style="dash")
    if rng.random() < 0.35:
        B.b_footer(pg, ctx, x, H - m * 0.6, w, page_no=rng.random() < 0.5)
    return pg, ctx


# =========================================================================== 2/3. ID-card-like and passport-like
def guilloche(pg, x0, y0, x1, y1, rng, col):
    d = pg.d
    n = rng.randint(6, 14)
    for k in range(n):
        amp = (y1 - y0) * rng.uniform(0.03, 0.12)
        per = rng.uniform(80, 260)
        ph = rng.uniform(0, 6.28)
        yb = y0 + (y1 - y0) * (k + 0.5) / n
        pts = [(xx, yb + amp * math.sin(xx / per * 6.28 + ph)) for xx in np.linspace(x0, x1, 120)]
        d.line(pts, fill=col, width=1)


CARD_TITLES = [("អត្តសញ្ញាណប័ណ្ណ", "IDENTITY CARD"), ("ប័ណ្ណសមាជិក", "MEMBER CARD"), ("ប័ណ្ណបុគ្គលិក", "STAFF ID CARD"),
               ("ប័ណ្ណនិស្សិត", "STUDENT CARD"), ("ប័ណ្ណបើកបរ", "DRIVING LICENCE"), ("ប័ណ្ណស្នាក់នៅ", "RESIDENCE CARD"), ("ប័ណ្ណសុខភាព", "HEALTH CARD")]


def mrz_name(lat):
    fam, giv = lat.split(" ", 1)
    return f"{fam}<<{giv.replace(' ', '<')}"


def card_front(pg, ctx, x, y, cw, ch):
    rng = ctx.rng
    d = pg.d
    tint = rng.choice([(215, 232, 245), (238, 228, 210), (222, 240, 225), (245, 230, 235), (232, 232, 240), (250, 250, 250)])
    d.rounded_rectangle([x, y, x + cw, y + ch], radius=int(ch * 0.05), fill=tint, outline=(120, 120, 120), width=1)
    guilloche(pg, x + 6, y + 6, x + cw - 6, y + ch - 6, rng, tuple(max(0, c - 22) for c in tint))
    band = rng.choice([(30, 60, 130), (150, 30, 40), (20, 100, 70), None])
    bh = int(ch * 0.2)
    if band:
        d.rounded_rectangle([x, y, x + cw, y + bh], radius=int(ch * 0.05), fill=band)
        d.rectangle([x, y + bh // 2, x + cw, y + bh], fill=band)
    tc = (255, 255, 255) if band else ctx.fg
    bnd = (x + 4, y + 4, x + cw - 4, y + ch - 4)
    ls = int(bh * 0.85)
    B.b_logo(pg, ctx, x + ch * 0.04, y + (bh - ls) / 2, ls, kind="emblem")
    t_km, t_en = rng.choice(CARD_TITLES)
    hs = int(bh * 0.3)
    qs = []
    if rng.random() < 0.6:
        q = pg.text(x + cw / 2, y + bh * 0.08, KINGDOM[0], ctx.font("moul", KINGDOM[0]), hs, tc, align="c", bound=bnd)
        qs.append(q)
        q2 = pg.text(x + cw / 2, y + bh * 0.52, f"{t_km} {t_en}", ctx.font("bold", f"{t_km} {t_en}"), int(hs * 0.95), tc, align="c", bound=bnd)
        if q2:
            pg.add("title", [q2])
    else:
        q2 = pg.text(x + cw / 2, y + bh * 0.25, f"{t_km} / {t_en}", ctx.font("bold", f"{t_km} / {t_en}"), int(hs * 1.15), tc, align="c", bound=bnd)
        if q2:
            pg.add("title", [q2])
    if any(qs):
        pg.add("header", qs)
    # photo
    pw, ph = int(cw * 0.25), int(cw * 0.25 * 1.3)
    left = rng.random() < 0.7
    px = x + ch * 0.05 if left else x + cw - pw - ch * 0.05
    py = y + bh + (ch - bh - ph) * rng.uniform(0.2, 0.5)
    B.b_photo(pg, ctx, px, py, pw, ph)
    fx = px + pw + cw * 0.04 if left else x + ch * 0.06
    fw = (x + cw - ch * 0.04 - fx) if left else (px - fx - cw * 0.03)
    km, (lat, km2) = ctx.kh_name()[0], ctx.kh_name()[1]
    sz = int(ch * rng.uniform(0.042, 0.052))
    fields = [("លេខ / No.", ctx.d(rng.randint(100000000, 999999999))), ("ឈ្មោះ", km2), ("Name", lat),
              ("ថ្ងៃខែឆ្នាំកំណើត / DOB", ctx.d(f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1950, 2006)}")),
              ("ភេទ / Sex", rng.choice(["ប្រុស M", "ស្រី F"])), ("កម្ពស់ / Height", ctx.d(f"{rng.randint(145, 190)}") + " cm"),
              ("ទីកន្លែងកំណើត", rng.choice(KH_PROV)), ("អាសយដ្ឋាន", " ".join(ctx.address()[1:])),
              ("សុពលភាព / Expiry", ctx.d(f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2027, 2036)}"))]
    keep = fields[:3] + rng.sample(fields[3:], rng.randint(3, 5))
    yy = y + bh + ch * 0.04
    stacked = rng.random() < 0.35
    for lab, val in keep:
        if yy > y + ch - sz * 2.4:
            break
        lp = ctx.font("body", lab)
        vp = ctx.font("bold", val)
        if stacked:
            q1 = pg.text(fx, yy, lab, lp, int(sz * 0.8), (80, 80, 90), bound=bnd)
            q2 = B.fit_text(pg, fx, yy + sz * 1.05, val, vp, sz, fw, ctx.fg)
            yy += sz * 2.5
        else:
            q1 = pg.text(fx, yy, lab + "៖", lp, int(sz * 0.85), (80, 80, 90), bound=bnd)
            if not q1:
                continue
            vx = bbox([q1])[2] + sz * 0.4
            q2 = B.fit_text(pg, vx, yy - sz * 0.05, val, vp, sz, fx + fw - vx, ctx.fg)
            yy += sz * 1.75
        if q1 and q2:
            pg.add("form_field", [q1, q2])
        elif q1 or q2:
            pg.add("text", [q1 or q2])
    if rng.random() < 0.7:  # holder signature
        sig = B.make_signature(ctx)
        k = min(cw * 0.22 / sig.width, ch * 0.12 / sig.height, 1.0)
        sig = sig.resize((max(6, int(sig.width * k)), max(6, int(sig.height * k))))
        pg.ink(sig, (px + (pw - sig.width) / 2, min(py + ph + ch * 0.02, y + ch - sig.height - 6)), "signature")
    if rng.random() < 0.35:
        st = B.make_stamp(ctx, d=ch * 0.3)
        pg.ink(st, (px + pw * rng.uniform(0.4, 0.8), py + ph * rng.uniform(0.5, 0.8)), "stamp")


def card_back(pg, ctx, x, y, cw, ch):
    rng = ctx.rng
    d = pg.d
    tint = rng.choice([(225, 236, 245), (240, 232, 215), (230, 242, 230), (240, 240, 240)])
    d.rounded_rectangle([x, y, x + cw, y + ch], radius=int(ch * 0.05), fill=tint, outline=(120, 120, 120), width=1)
    guilloche(pg, x + 6, y + 6, x + cw - 6, y + ch * 0.55, rng, tuple(max(0, c - 20) for c in tint))
    bnd = (x + 4, y + 4, x + cw - 4, y + ch - 4)
    sz = int(ch * 0.045)
    yy = y + ch * 0.06
    for lab, val in rng.sample([("អាសយដ្ឋាន / Address", " ".join(ctx.address()[1:])), ("ស្នាមសម្គាល់ / Marks", rng.choice(["ប្រជ្រុយលើថ្ពាល់ឆ្វេង", "គ្មាន", "ស្លាកស្នាមលើដៃស្តាំ"])),
                                ("ថ្ងៃចេញ / Issued", ctx.d(f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2015, 2026)}"))], 2):
        q1 = pg.text(x + ch * 0.06, yy, lab + "៖", ctx.font("body", lab + "៖"), int(sz * 0.85), (70, 70, 80), bound=bnd)
        q2 = B.fit_text(pg, x + ch * 0.06, yy + sz * 1.2, val, ctx.font("bold", val), sz, cw * 0.5, ctx.fg)
        if q1 and q2:
            pg.add("form_field", [q1, q2])
        elif q1 or q2:
            pg.add("text", [q1 or q2])
        yy += sz * 3
    if rng.random() < 0.5:
        bw = cw * rng.uniform(0.3, 0.45)
        B.b_barcode(pg, ctx, x + cw - bw - ch * 0.06, y + ch * 0.08, bw, ch * 0.12, digits=False)
    else:
        from . import assets2 as X
        im = X.make_pdf417ish(rng, int(cw * 0.4), int(ch * 0.16))
        pg.paste(im, (x + cw - cw * 0.4 - ch * 0.06, y + ch * 0.08), "qr_code")
    # MRZ (TD1, 3 x 30)
    lat = ctx.kh_name()[1][0]
    no = "".join(rng.choice("0123456789") for _ in range(9))
    l1 = f"ID{'KHM' if rng.random() < 0.5 else 'XXA'}{no}{rng.randint(0, 9)}".ljust(30, "<")
    l2 = f"{rng.randint(50, 99)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}{rng.randint(0, 9)}{rng.choice('MF')}{rng.randint(27, 36)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}{rng.randint(0, 9)}KHM".ljust(29, "<") + str(rng.randint(0, 9))
    l3 = mrz_name(lat).ljust(30, "<")[:30]
    p = ctx.font("mono", l1 + l2 + l3)
    f = font(p, 10)
    msz = int(10 * (cw * 0.88) / f.getlength(l1))
    qs = []
    yy = y + ch - msz * 1.25 * 3 - ch * 0.05
    for t in (l1, l2, l3):
        q = pg.text(x + cw / 2, yy, t, p, msz, (20, 20, 20), align="c", bound=bnd)
        if q:
            qs.append(q)
        yy += msz * 1.25
    if qs:
        pg.add("text", qs)


def card_page(rng, fake, drawer_list, card_wh=(1010, 640)):
    """Either a capture of the card alone (table background) or a photocopy on A4 (front + back)."""
    ctx = Ctx(rng, "km", fake)
    cw, ch = card_wh
    if rng.random() < 0.55:
        mx, my = int(cw * rng.uniform(0.04, 0.15)), int(ch * rng.uniform(0.05, 0.2))
        pg = Page(cw + 2 * mx, ch + 2 * my, rng.choice([(70, 70, 75), (150, 120, 90), (230, 230, 230), (40, 50, 60), (200, 190, 170)]))
        drawer_list[0](pg, ctx, mx, my, cw, ch)
    else:
        pg = new_page(rng, "a4", 0.0)
        s = rng.uniform(0.75, 1.0)
        cw2, ch2 = int(cw * s), int(ch * s)
        x = int((pg.W - cw2) / 2 + rng.uniform(-60, 60))
        y = int(pg.H * rng.uniform(0.06, 0.15))
        for i, fn in enumerate(drawer_list[:2] if len(drawer_list) > 1 else drawer_list):
            fn(pg, ctx, x, y, cw2, ch2)
            y += ch2 + int(rng.uniform(40, 140))
        if rng.random() < 0.5 and y < pg.H - 300:
            t = rng.choice(["ច្បាប់ចម្លងត្រឹមត្រូវតាមច្បាប់ដើម", "បានឃើញ និងបញ្ជាក់ថា ត្រឹមត្រូវ", "CERTIFIED TRUE COPY"])
            q = pg.text(pg.W / 2, y, t, ctx.font("bold", t), ctx.size, ctx.fg, align="c")
            if q:
                pg.add("text", [q])
                B.b_sign(pg, ctx, pg.W * 0.5, bbox([q])[3] + 20, pg.W * 0.35, stamp_p=0.8)
    return pg, ctx


def km_id_card(rng, fake):
    r = rng.random()
    lst = [card_front, card_back] if r < 0.45 else ([card_front] if r < 0.85 else [card_back])
    return card_page(rng, fake, lst)


def passport_page(pg, ctx, x, y, cw, ch):
    rng = ctx.rng
    d = pg.d
    tint = rng.choice([(236, 232, 245), (228, 240, 236), (245, 236, 226), (232, 238, 248)])
    d.rectangle([x, y, x + cw, y + ch], fill=tint, outline=(150, 150, 160))
    guilloche(pg, x + 4, y + 4, x + cw - 4, y + ch * 0.7, rng, tuple(max(0, c - 18) for c in tint))
    bnd = (x + 4, y + 4, x + cw - 4, y + ch - 4)
    sz = int(ch * 0.032)
    hdr = rng.choice([("លិខិតឆ្លងដែន", "PASSPORT"), ("លិខិតឆ្លងដែនការទូត", "TRAVEL DOCUMENT"), ("ឯកសារធ្វើដំណើរ", "TRAVEL DOCUMENT")])
    t = f"{hdr[0]} / {hdr[1]}" if rng.random() < 0.6 else hdr[1]
    q = pg.text(x + cw / 2, y + ch * 0.03, t, ctx.font("bold", t), int(sz * 1.4), ctx.fg, align="c", bound=bnd)
    if q:
        pg.add("title", [q])
    t2 = rng.choice(["ព្រះរាជាណាចក្រកម្ពុជា / KINGDOM OF CAMBODIA", "KINGDOM OF CAMBODIA", "ព្រះរាជាណាចក្រកម្ពុជា"])
    q2 = pg.text(x + cw / 2, y + ch * 0.03 + sz * 2.4, t2, ctx.font("bold", t2), sz, ctx.fg, align="c", bound=bnd)
    if q2:
        pg.add("header", [q2])
    B.b_logo(pg, ctx, x + cw - ch * 0.17, y + ch * 0.03, int(ch * 0.13), kind="emblem")
    pw, ph = int(cw * 0.25), int(cw * 0.25 * 1.28)
    px, py = x + cw * 0.04, y + ch * 0.2
    B.b_photo(pg, ctx, px, py, pw, ph, frame=False)
    km, (lat, km2) = ctx.kh_name()[0], ctx.kh_name()[1]
    fam, giv = lat.split(" ", 1)
    no = rng.choice("NPKD") + "".join(rng.choice("0123456789") for _ in range(7))
    dob = f"{rng.randint(1, 28):02d} {rng.choice(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])} {rng.randint(1955, 2008)}"
    exp = f"{rng.randint(1, 28):02d} {rng.choice(['JAN', 'MAR', 'JUN', 'SEP', 'DEC'])} {rng.randint(2027, 2036)}"
    fields = [("ប្រភេទ / Type", "P"), ("លេខលិខិតឆ្លងដែន / Passport No.", no), ("នាមត្រកូល / Surname", fam), ("នាមខ្លួន / Given names", giv),
              ("ឈ្មោះ", km2), ("សញ្ជាតិ / Nationality", "CAMBODIAN"), ("ថ្ងៃខែឆ្នាំកំណើត / Date of birth", dob), ("ភេទ / Sex", rng.choice("MF")),
              ("ទីកន្លែងកំណើត / Place of birth", rng.choice(KH_PROV_EN).upper()), ("ថ្ងៃផុតកំណត់ / Date of expiry", exp)]
    fx, fy = px + pw + cw * 0.04, max(py - sz * 0.3, (bbox([q2])[3] + sz * 0.8) if q2 else py)
    colw = (x + cw - fx - cw * 0.03) / 2
    for i, (lab, val) in enumerate(fields):
        cxx = fx + (i % 2) * colw
        cyy = fy + (i // 2) * sz * 3.0
        if cyy > y + ch * 0.68:
            break
        q1 = pg.text(cxx, cyy, lab, ctx.font("body", lab), int(sz * 0.72), (70, 70, 90), bound=(cxx, y, cxx + colw - 6, y + ch))
        q2 = pg.text(cxx, cyy + sz * 1.05, val, ctx.font("bold", val), sz, ctx.fg, bound=(cxx, y, cxx + colw - 6, y + ch))
        if q1 and q2:
            pg.add("form_field", [q1, q2])
        elif q1 or q2:
            pg.add("text", [q1 or q2])
    sig = B.make_signature(ctx)
    k = min(cw * 0.2 / sig.width, ch * 0.07 / sig.height, 1.0)
    sig = sig.resize((max(6, int(sig.width * k)), max(6, int(sig.height * k))))
    pg.ink(sig, (px + (pw - sig.width) / 2, py + ph + ch * 0.01), "signature")
    l1 = f"P<{'KHM' if rng.random() < 0.5 else 'UTO'}{mrz_name(lat)}".ljust(44, "<")[:44]
    l2 = f"{no}{rng.randint(0, 9)}KHM{rng.randint(50, 99)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}{rng.randint(0, 9)}{rng.choice('MF')}{rng.randint(27, 36)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}{rng.randint(0, 9)}".ljust(42, "<") + f"{rng.randint(0, 9)}{rng.randint(0, 9)}"
    p = ctx.font("mono", l1 + l2)
    msz = int(10 * (cw * 0.92) / font(p, 10).getlength(l1))
    d.rectangle([x + 2, y + ch * 0.76, x + cw - 2, y + ch - 2], fill=tuple(min(255, c + 8) for c in tint))
    qs = []
    for i, t in enumerate((l1, l2)):
        q = pg.text(x + cw / 2, y + ch * 0.8 + i * msz * 1.45, t, p, msz, (15, 15, 15), align="c", bound=bnd)
        if q:
            qs.append(q)
    if qs:
        pg.add("text", qs)
    if rng.random() < 0.3:
        st = B.make_stamp(ctx, d=ch * 0.3)
        pg.ink(st, (x + cw * rng.uniform(0.3, 0.7), y + ch * rng.uniform(0.1, 0.4)), "stamp")


def km_passport(rng, fake):
    return card_page(rng, fake, [passport_page], card_wh=(1250, 880))


# =========================================================================== 4. certificate
def km_certificate(rng, fake):
    land = rng.random() < 0.6
    pg = new_page(rng, "a4l" if land else "a4", 0.3)
    ctx = Ctx(rng, "km", fake)
    W, H = pg.W, pg.H
    d = pg.d
    bc = ctx.accent
    k = int(min(W, H) * rng.uniform(0.025, 0.04))
    d.rectangle([k, k, W - k, H - k], outline=bc, width=rng.randint(5, 12))
    d.rectangle([k + 16, k + 16, W - k - 16, H - k - 16], outline=bc, width=2)
    if rng.random() < 0.5:
        for cx, cy in ((k + 16, k + 16), (W - k - 16, k + 16), (k + 16, H - k - 16), (W - k - 16, H - k - 16)):
            d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=bc)
    m = int(min(W, H) * 0.1)
    x, w = m, W - 2 * m
    y = motto(pg, ctx, W / 2, k + 40)
    ls = int(ctx.size * rng.uniform(3, 4.5))
    if rng.random() < 0.8:
        B.b_logo(pg, ctx, x + rng.uniform(0, 30), k + 50, ls, kind="emblem")
    org = rng.choice(MINISTRIES)[0] if rng.random() < 0.5 else rng.choice(["សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ", "វិទ្យាល័យ ព្រះស៊ីសុវត្ថិ", "សាលាបច្ចេកទេស ស្រុកខ្មែរ", "មជ្ឈមណ្ឌលបណ្តុះបណ្តាល អង្គរ"])
    q = pg.text(W / 2, y, org, ctx.font("moul", org), int(ctx.size * 1.0), ctx.fg, align="c")
    if q:
        pg.add("header", [q])
        y = bbox([q])[3] + ctx.size
    title = rng.choice(["វិញ្ញាបនបត្រ", "សញ្ញាបត្រ", "លិខិតសរសើរ", "បណ្ណសរសើរ", "វិញ្ញាបនបត្របញ្ជាក់ការសិក្សា"])
    en = rng.choice(["CERTIFICATE", "CERTIFICATE OF COMPLETION", "DIPLOMA", "CERTIFICATE OF APPRECIATION"])
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=title, size=ctx.size * rng.uniform(2.0, 2.8), align="c", role="moul", fill=bc)
    if rng.random() < 0.6:
        y = B.b_heading(pg, ctx, x, y - ctx.size * 0.3, w, cls="title", text=en, size=ctx.size * 1.3, align="c", role="latin_bold", fill=bc)
    photo = rng.random() < 0.4
    if photo:
        pw = int(ctx.size * 4.5)
        B.b_photo(pg, ctx, W - m - pw, y, pw, int(pw * 1.33))
    km, (lat, _) = ctx.kh_name()
    lines = [f"បញ្ជាក់ថា {rng.choice(['លោក', 'កញ្ញា', 'អ្នកនាង', 'និស្សិត'])} {km} ({lat})",
             f"ថ្ងៃខែឆ្នាំកំណើត {ctx.date()} នៅ{rng.choice(KH_PROV)}",
             ctx.para(rng.randint(80, 200))]
    yy = y
    for i, t in enumerate(lines):
        yy = B.b_para(pg, ctx, x + (0 if not photo else 0), yy, w - (ctx.size * 6 if photo else 0), n=3, text=t, align="c", indent=0,
                      size=ctx.size * (1.15 if i == 0 else 1.0))
    if rng.random() < 0.5:
        B.b_qr(pg, ctx, x, H - m - 150, rng.randint(100, 150))
    y = max(yy, H * 0.62)
    y = date_line(pg, ctx, x + w * 0.5, y, w * 0.5, "c", lunar=rng.random() < 0.3)
    if rng.random() < 0.5:
        B.b_sign(pg, ctx, x, y, w * 0.38, H - k - 30, stamp_p=0.4)
    B.b_sign(pg, ctx, x + w * 0.58, y, w * 0.4, H - k - 30, stamp_p=0.85)
    return pg, ctx


# =========================================================================== 5. invoice / receipt (KHQR-like)
def pay_qr_block(pg, ctx, x, y, size):
    """Generic payment-QR card: coloured header band + merchant + amount + QR (only the QR is qr_code)."""
    rng = ctx.rng
    d = pg.d
    col = rng.choice([(200, 30, 40), (20, 60, 140), (10, 120, 90)])
    cw = int(size * 1.25)
    hh = int(size * 0.22)
    if y + hh + size * 1.6 > pg.H:
        return y
    d.rounded_rectangle([x, y, x + cw, y + hh + size * 1.45], radius=12, fill=(255, 255, 255), outline=(200, 200, 200))
    d.rounded_rectangle([x, y, x + cw, y + hh], radius=12, fill=col)
    t = rng.choice(["SCAN & PAY", "QR PAY", "ស្កេនដើម្បីបង់ប្រាក់", "PAY HERE"])
    q = pg.text(x + cw / 2, y + hh * 0.18, t, ctx.font("bold", t), int(hh * 0.5), (255, 255, 255), align="c")
    qs = [q] if q else []
    m = ctx.company()
    q2 = pg.text(x + size * 0.1, y + hh + size * 0.05, m, ctx.font("body", m), int(size * 0.075), ctx.fg, bound=(x + 2, y, x + cw - 2, pg.H))
    if q2:
        qs.append(q2)
    amt = ctx.money(1, 300)
    q3 = pg.text(x + size * 0.1, y + hh + size * 0.17, amt, ctx.font("bold", amt), int(size * 0.1), ctx.fg, bound=(x + 2, y, x + cw - 2, pg.H))
    if q3:
        qs.append(q3)
    if qs:
        pg.add("text", qs)
    qy = y + hh + size * 0.38
    d.line([x + 10, qy - size * 0.06, x + cw - 10, qy - size * 0.06], fill=(200, 200, 200), width=1)
    B.b_qr(pg, ctx, x + (cw - size) / 2, qy, size, data="00020101021229" + "".join(rng.choice("0123456789ABCDEF") for _ in range(rng.randint(40, 110))))
    return qy + size + 20


def totals(pg, ctx, x, y, w, sub=None):
    rng = ctx.rng
    sz = int(ctx.size * 0.95)
    sub = sub or rng.uniform(10, 5000)
    vat = sub * 0.1
    if ctx.lang != "km":
        rate = rng.choice([6.0, 7.25, 8.25, 8.875, 9.5])
        vat = sub * rate / 100
    rows = [(ctx.label("សរុប", "Subtotal", 0.5), f"{sub:,.2f}"),
            (ctx.label("អាករលើតម្លៃបន្ថែម ១០%", "VAT 10%", 0.4) if ctx.lang == "km" else rng.choice([f"Sales Tax ({rate}%)", f"Tax {rate}%"]), f"{vat:,.2f}")]
    if rng.random() < 0.4:
        rows.append((ctx.label("បញ្ចុះតម្លៃ", "Discount", 0.5), f"-{sub * 0.05:,.2f}"))
    rows.append((ctx.label("សរុបរួម", "Grand Total", 0.6), f"${sub + vat:,.2f}"))
    if rng.random() < 0.4 and ctx.lang == "km":
        rows.append(("ជារៀល / KHR", ctx.d(f"{int((sub + vat) * 4100):,}") + " ៛"))
    qs = []
    yy = y
    for i, (a, b) in enumerate(rows):
        role = "bold" if i == len(rows) - 1 or rng.random() < 0.2 else "body"
        q2 = pg.text(x + w, yy, b, ctx.font(role, b), sz, ctx.fg, align="r")
        vx = bbox([q2])[0] if q2 else x + w * 0.7
        q1 = B.fit_text(pg, x + w * 0.15, yy, a, ctx.font(role, a), sz, vx - (x + w * 0.15) - 15, ctx.fg)
        if q1 is None and q2 is not None:  # keep text <-> label consistent: drop nothing, box the value alone
            pass
        qs += [q for q in (q1, q2) if q]
        yy += sz * 1.7
    if qs:
        pg.add("text", qs)
    return yy + sz * 0.5


def km_invoice(rng, fake):
    ctx = Ctx(rng, "km", fake)
    receipt = rng.random() < 0.3
    if receipt:
        return thermal_receipt(rng, fake, ctx)
    pg = new_page(rng, "a4", 0.1)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.06, 0.09))
    x, w = m, W - 2 * m
    y = int(H * 0.04)
    if rng.random() < 0.5:
        B.watermark(pg, ctx) if rng.random() < 0.15 else None
    ls = int(ctx.size * rng.uniform(3, 4.5))
    lb = B.b_logo(pg, ctx, x, y, ls, kind="corp")
    if lb:
        ls = max(ls, int(lb[2] - x))
    comp = ctx.company()
    en_comp = fake.company() + rng.choice([" Co., Ltd.", " (Cambodia) Ltd.", " Trading"])
    hq = []
    q = pg.text(x + ls + 24, y, comp, ctx.font("moul" if rng.random() < 0.5 else "bold", comp), int(ctx.size * 1.2), ctx.accent)
    hq.append(q)
    yy = (bbox([q])[3] if q else y + ctx.size) + 8
    for t in [en_comp.upper(), " ".join(ctx.address()[:2]), f"Tel: {ctx.phone()}  VATTIN: K{rng.randint(1, 9)}{rng.randint(10, 99)}-{rng.randint(100000000, 999999999)}"]:
        q = pg.text(x + ls + 24, yy, t, ctx.font("body", t), int(ctx.size * 0.78), ctx.fg, bound=(0, 0, W - m + 5, H))
        if q:
            hq.append(q)
            yy = bbox([q])[3] + 6
    pg.add("header", hq)
    y = max(yy, y + ls) + ctx.size
    pg.d.line([x, y, x + w, y], fill=ctx.accent, width=3)
    title = rng.choice(["វិក្កយបត្រ", "វិក្កយបត្រអាករ", "បង្កាន់ដៃទទួលប្រាក់", "វិក្កយបត្រ / INVOICE", "TAX INVOICE / វិក្កយបត្រអាករ", "សក្ខីប័ត្រ"])
    y = B.b_heading(pg, ctx, x, y + ctx.size * 0.6, w, cls="title", text=title, size=ctx.size * 1.6, align="c", role="moul" if rng.random() < 0.5 else "bold")
    # customer + invoice meta
    yl = yr = y
    form_like = rng.random() < 0.35
    left = [("អតិថិជន / Customer", ctx.kh_name()[0] if rng.random() < 0.5 else ctx.company()), ("អាសយដ្ឋាន", ctx.address()[2]), ("ទូរស័ព្ទ", ctx.phone())]
    right = [("លេខវិក្កយបត្រ / No.", f"INV-{ctx.d(rng.randint(10000, 99999))}"), ("កាលបរិច្ឆេទ / Date", ctx.date())]
    for lab, val in left:
        if form_like:
            yl = B.b_field(pg, ctx, x, yl, w * 0.55, label=lab, value=val, vkind=rng.choice(["hw", "print"]), style="dots")
        else:
            t = f"{lab}៖ {val}"
            q = pg.text(x, yl, t, ctx.font("body", t), ctx.size, ctx.fg, bound=(0, 0, x + w * 0.58, H))
            if q:
                pg.add("text", [q])
            yl += ctx.size * 1.8
    for lab, val in right:
        t = f"{lab}៖ {val}"
        q = pg.text(x + w, yr, t, ctx.font("body", t), ctx.size, ctx.fg, align="r", bound=(x + w * 0.58, 0, W, H))
        if q:
            pg.add("text", [q])
        yr += ctx.size * 1.8
    y = max(yl, yr) + ctx.size * 0.5
    y = B.b_table(pg, ctx, x, y, w, H * 0.72, kind="items", nrow=rng.randint(3, 12), style=rng.choice(["grid", "grid", "zebra", "hlines"]))
    y = totals(pg, ctx, x + w * 0.4, y, w * 0.6)
    if rng.random() < 0.4:
        y = B.b_check_group(pg, ctx, x, y, w * 0.6, q="វិធីទូទាត់", opts=["សាច់ប្រាក់", "ផ្ទេរតាមធនាគារ", "QR"])
    ybot = y
    if rng.random() < 0.65:
        ybot = pay_qr_block(pg, ctx, x, y + 10, rng.randint(170, 240))
    if rng.random() < 0.6:
        st = B.make_stamp(ctx, d=rng.uniform(170, 240))
        pg.ink(st, (x + w * rng.uniform(0.35, 0.6), y + rng.uniform(0, 80)), "stamp")
    sy = max(y, H * 0.75) if ybot < H * 0.75 else y
    if rng.random() < 0.7 and sy < H - 260:
        B.b_sign(pg, ctx, x + w * 0.62, sy, w * 0.36, role=rng.choice(["អ្នកលក់", "ហត្ថលេខាអ្នកលក់", "Authorized Signature"]), stamp_p=0.2)
        if rng.random() < 0.5:
            B.b_sign(pg, ctx, x + w * 0.3 if ybot > sy else x, sy, w * 0.3, role=rng.choice(["អ្នកទិញ", "អតិថិជន", "Customer"]), stamp_p=0.0)
    if rng.random() < 0.5:
        t = rng.choice(["សូមអរគុណ! Thank you!", "ទំនិញទិញហើយមិនអាចប្តូរវិញបានទេ", "សូមពិនិត្យទំនិញមុនចាកចេញ"])
        q = pg.text(W / 2, H - m * 0.9, t, ctx.font("body", t), int(ctx.size * 0.85), ctx.fg, align="c")
        pg.add("footer", [q])
    return pg, ctx


def thermal_receipt(rng, fake, ctx):
    """Narrow receipt (Khmer or US): mono/sans font, item rows (table), totals, barcode/QR."""
    W = rng.randint(560, 680)
    H = rng.randint(1400, 2000)
    paper = rng.choice([(250, 250, 248), (245, 245, 240), (255, 255, 255), (240, 238, 232)])
    pg = Page(W, H, paper)
    sz = int(rng.uniform(19, 24))
    ctx.size = sz
    ctx.fg = rng.choice([(30, 30, 30), (60, 60, 70), (90, 90, 100)])  # faded thermal print
    m = int(W * 0.06)
    x, w = m, W - 2 * m
    y = m
    if rng.random() < 0.6:
        ls = int(sz * rng.uniform(3, 4.5))
        B.b_logo(pg, ctx, W / 2 - ls / 2, y, ls, kind="corp")
        y += ls + 10
    name = ctx.company() if ctx.lang == "km" else fake.company().upper()
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=name, size=sz * 1.3, align="c", role="bold", max_lines=2)
    hq = []
    for t in ctx.address()[-2:] + [("Tel: " + ctx.phone())]:
        q = pg.text(W / 2, y, t, ctx.font("mono" if ctx.lang == "en" else "body", t), int(sz * 0.85), ctx.fg, align="c", bound=(2, 0, W - 2, H))
        if q:
            hq.append(q)
            y = bbox([q])[3] + 6
    if hq:
        pg.add("header", hq)
    y += sz * 0.5
    dash = "-" * 60
    mp = ctx.font("mono", dash)

    def rule(yy):
        q = pg.text(x, yy, dash[: int(w / font(mp, sz).getlength("-"))], mp, sz, ctx.fg, kind=None)
        return yy + sz * 1.2

    meta = [f"{ctx.label('វិក្កយបត្រ', 'Receipt', 0.5)} #{rng.randint(1000, 999999)}", f"{ctx.date()} {rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}",
            f"{ctx.label('អ្នកគិតលុយ', 'Cashier', 0.5)}: {fake.first_name()}"]
    mq = []
    for t in meta:
        q = pg.text(x, y, t, ctx.font("mono" if ctx.lang == "en" else "body", t), int(sz * 0.9), ctx.fg, bound=(2, 0, W - 2, H))
        if q:
            mq.append(q)
            y = bbox([q])[3] + 6
    if mq:
        pg.add("text", mq)
    y = rule(y + 4)
    # items: name left / price right; maybe qty line
    iq = []
    sub = 0.0
    for _ in range(rng.randint(3, 12)):
        if y > H * 0.62:
            break
        if ctx.lang == "km":
            from .ctx import KH_PRODUCTS
            it = rng.choice(KH_PRODUCTS) if rng.random() < 0.7 else fake.word().upper()
        else:
            it = rng.choice([fake.word().upper() + " " + fake.word().upper(), fake.color_name().upper() + " " + rng.choice(["TEE", "MUG", "SOCKS", "CABLE"]), rng.choice(["MILK 2%", "BREAD WHT", "EGGS LG 12CT", "COFFEE", "BANANAS", "CHICKEN BRST", "PAPER TOWEL"])])
        qn = rng.randint(1, 4)
        pr = rng.uniform(0.5, 40)
        sub += pr * qn
        t1 = it[:22]
        t2 = f"{pr * qn:,.2f}"
        f1 = ctx.font("mono" if ctx.lang == "en" else "body", t1)
        q1 = pg.text(x, y, t1, f1, sz, ctx.fg, bound=(2, 0, x + w * 0.7, H))
        q2 = pg.text(x + w, y, t2, ctx.font("mono", t2), sz, ctx.fg, align="r")
        iq += [q for q in (q1, q2) if q]
        y += sz * 1.45
        if qn > 1 and rng.random() < 0.6:
            t3 = f"  {qn} @ {pr:,.2f}"
            q3 = pg.text(x, y, t3, ctx.font("mono", t3), int(sz * 0.9), ctx.fg)
            if q3:
                iq.append(q3)
            y += sz * 1.35
    if iq:
        pg.add("table", iq)
    y = rule(y + 4)
    tq = []
    for lab, val in [(ctx.label("សរុប", "SUBTOTAL", 0.5), f"{sub:,.2f}"), (ctx.label("ពន្ធ", "TAX", 0.5), f"{sub * 0.08:,.2f}"),
                     (ctx.label("សរុបរួម", "TOTAL", 0.5), f"${sub * 1.08:,.2f}"), (rng.choice(["CASH", "VISA ****" + str(rng.randint(1000, 9999)), "KHQR"]), f"{sub * 1.08:,.2f}")]:
        q1 = pg.text(x, y, lab, ctx.font("bold" if "TOTAL" in lab else "body", lab), sz, ctx.fg)
        q2 = pg.text(x + w, y, val, ctx.font("mono", val), sz, ctx.fg, align="r")
        tq += [q for q in (q1, q2) if q]
        y += sz * 1.5
    if tq:
        pg.add("text", tq)
    y += sz
    if rng.random() < 0.7:
        y = B.b_barcode(pg, ctx, x + w * 0.1, y, w * 0.8, rng.randint(60, 90)) + sz
    if rng.random() < 0.5 and y + 220 < H:
        s = rng.randint(140, 200)
        B.b_qr(pg, ctx, W / 2 - s / 2, y, s)
        y += s + sz
    t = rng.choice(["THANK YOU!", "សូមអរគុណ!", "PLEASE COME AGAIN", "Thank you for shopping with us"]) if ctx.lang == "km" else rng.choice(["THANK YOU!", "PLEASE COME AGAIN", "Thank you for shopping with us", "RETURNS WITHIN 30 DAYS WITH RECEIPT"])
    q = pg.text(W / 2, y, t, ctx.font("bold", t), sz, ctx.fg, align="c", bound=(2, 0, W - 2, H - 4))
    if q:
        pg.add("footer", [q])
    pg.meta["thermal"] = True
    return pg, ctx


# =========================================================================== 6. bank statement
def bank_header(pg, ctx, x, y, w, bank, en):
    rng = ctx.rng
    ls = int(ctx.size * rng.uniform(2.6, 3.6))
    B.b_logo(pg, ctx, x, y, ls, kind="corp", wordmark=bank if rng.random() < 0.6 else None)
    t2 = rng.choice([en, ctx.phone(), "www." + ctx.fake.domain_name()])
    q = pg.text(x + w, y + ls * 0.25, t2, ctx.font("body", t2), int(ctx.size * 0.8), (90, 90, 90), align="r")
    if q:
        pg.add("header", [q])
    return y + ls + ctx.size


def km_bank_statement(rng, fake):
    pg = new_page(rng, "a4", 0.05)
    ctx = Ctx(rng, "km", fake)
    return _bank_statement(pg, ctx, rng.choice(["ធនាគារ អង្គរ", "ធនាគារ មេគង្គ ពាណិជ្ជ", "ធនាគារ ស្រុកខ្មែរ", "គ្រឹះស្ថានមីក្រូហិរញ្ញវត្ថុ បាយ័ន"]),
                           rng.choice(["ANGKOR BANK PLC.", "MEKONG COMMERCIAL BANK", "SROK KHMER BANK", "BAYON MFI PLC."]))


def _bank_statement(pg, ctx, bank, en):
    rng, fake = ctx.rng, ctx.fake
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.06, 0.09))
    x, w = m, W - 2 * m
    y = bank_header(pg, ctx, x, int(H * 0.035), w, bank, en)
    title = ctx.label("របាយការណ៍គណនី", "Account Statement", 0.6) if ctx.lang == "km" else rng.choice(["Account Statement", "Monthly Statement", "Checking Account Statement"])
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=title, size=ctx.size * 1.5, align=rng.choice(["l", "c"]))
    # customer address (left) + account summary (right)
    nm = ctx.kh_name(True)[0] if (ctx.lang == "km" and rng.random() < 0.5) else ctx.name()
    yl = y
    aq = []
    for t in [nm] + ctx.address():
        q = pg.text(x, yl, t, ctx.font("body", t), ctx.size, ctx.fg, bound=(0, 0, x + w * 0.5, H))
        if q:
            aq.append(q)
            yl = bbox([q])[3] + ctx.size * 0.45
    if aq:
        pg.add("text", aq)
    rows = [(ctx.label("លេខគណនី", "Account No.", 0.5), ctx.d(f"{rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(100, 999)}")),
            (ctx.label("រយៈពេល", "Period", 0.5), f"{ctx.date()} - {ctx.date()}"),
            (ctx.label("រូបិយប័ណ្ណ", "Currency", 0.5), rng.choice(["USD", "KHR"])),
            (ctx.label("សមតុល្យដើមគ្រា", "Opening balance", 0.5), f"{rng.uniform(100, 20000):,.2f}")]
    yr = y
    rq = []
    for a, b in rows:
        q2 = B.fit_text(pg, x + w, yr, b, ctx.font("bold", b), int(ctx.size * 0.9), w * 0.3, ctx.fg, align="r")
        bx0 = bbox([q2])[0] if q2 else x + w * 0.8
        q1 = B.fit_text(pg, x + w * 0.52, yr, a, ctx.font("body", a), int(ctx.size * 0.9), bx0 - (x + w * 0.52) - 12, ctx.fg)
        rq += [q for q in (q1, q2) if q]
        yr += ctx.size * 1.6
    if rq:
        if rng.random() < 0.5:
            b = bbox(rq)
            pg.d.rectangle([b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10], outline=(150, 150, 150))
            pg.add("table", (b[0] - 10, b[1] - 10, b[2] + 11, b[3] + 11))
        else:
            pg.add("text", rq)
    y = max(yl, yr) + ctx.size
    fn_top = H - m * 0.9
    if rng.random() < 0.5:
        fn_top = B.b_footnotes(pg, ctx, x, H - m * 1.2, w, n=rng.randint(1, 2))
    y = B.b_table(pg, ctx, x, y, w, fn_top - ctx.size, kind="statement", nrow=rng.randint(8, 30), style=rng.choice(["hlines", "zebra", "grid", "none"]),
                  size=ctx.size * 0.82, col_w=[1.1, 3.2, 1.1, 1.1, 1.2])
    if rng.random() < 0.3 and y < fn_top - 200:
        B.b_qr(pg, ctx, x, y, 120)
    B.b_footer(pg, ctx, x, H - m * 0.55, w, page_no=rng.random() < 0.85, text=rng.random() < 0.5)
    return pg, ctx


# =========================================================================== 7. application form
def km_application_form(rng, fake):
    pg = new_page(rng, "a4", 0.08)
    ctx = Ctx(rng, "km", fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.07, 0.1))
    x, w = m, W - 2 * m
    y = int(H * 0.035)
    if rng.random() < 0.7:
        y = motto(pg, ctx, W / 2, y)
    if rng.random() < 0.5:
        y = org_block(pg, ctx, x, y - ctx.size, logo_p=0.5, ref=False)
    title = rng.choice(["ពាក្យស្នើសុំ", "ពាក្យសុំចូលបម្រើការងារ", "ពាក្យស្នើសុំបើកគណនី", "ពាក្យសុំចុះឈ្មោះចូលរៀន", "ទម្រង់ចុះឈ្មោះ", "ពាក្យសុំលិខិតឆ្លងដែន", "ប្រវត្តិរូបសង្ខេប"])
    photo = rng.random() < 0.6
    pw = int(ctx.size * 5)
    ph = int(pw * 1.33)
    if photo:
        px, py = x + w - pw, y
        if rng.random() < 0.85:
            B.b_photo(pg, ctx, px, py, pw, ph)
            pg.d.rectangle([px - 3, py - 3, px + pw + 2, py + ph + 2], outline=(80, 80, 80), width=1)
        else:  # empty photo frame "4x6"
            pg.d.rectangle([px, py, px + pw, py + ph], outline=(80, 80, 80), width=1)
            q = pg.text(px + pw / 2, py + ph / 2 - ctx.size, "រូបថត ៤x៦", ctx.font("body", "រូបថត ៤x៦"), int(ctx.size * 0.8), (100, 100, 100), align="c")
            pg.add("text", [q])
    y = B.b_heading(pg, ctx, x, y + ctx.size * 0.5, w - (pw + 20 if photo else 0), cls="title", text=title, size=ctx.size * 1.5, align="c", role="moul")
    if photo:
        y = max(y, py + ph + ctx.size)
    fw = w
    ymax = H * rng.uniform(0.7, 0.78)
    labels = list(B.FIELD_LABELS["km"])
    rng.shuffle(labels)
    sec = 0
    while y < ymax - ctx.size * 3 and labels:
        r = rng.random()
        if r < 0.1:
            sec += 1
            y = B.b_heading(pg, ctx, x, y, w, ymax, number=f"{ctx.d(sec)}." if rng.random() < 0.5 else rng.choice(["ក.", "ខ.", "គ."]),
                            text=rng.choice(["ព័ត៌មានផ្ទាល់ខ្លួន", "ប្រវត្តិការសិក្សា", "ព័ត៌មានទំនាក់ទំនង", "បទពិសោធន៍ការងារ", "ព័ត៌មានគ្រួសារ"]), size=ctx.size * 1.1)
        elif r < 0.22:
            y = B.b_check_group(pg, ctx, x, y, w, ymax)
        elif r < 0.3 and y < ymax - 300:
            hdr = rng.choice([["ល.រ", "ឆ្នាំសិក្សា", "គ្រឹះស្ថាន", "សញ្ញាបត្រ"], ["ល.រ", "ឈ្មោះ", "ត្រូវជា", "មុខរបរ"], ["ពីឆ្នាំ", "ដល់ឆ្នាំ", "ស្ថាប័ន", "តួនាទី"]])
            n = rng.randint(2, 4)
            y0 = y
            y = B.b_table(pg, ctx, x, y, w, ymax, hdr=hdr, rows=[[ctx.d(i + 1) if hdr[0] == "ល.រ" else "", "", "", ""] for i in range(n)], al=["c", "l", "l", "l"], style="grid")
            # handwritten cell values
            if y > y0:
                tb = pg.objs[-1][1]
                tx0, ty0 = tb.min(0)
                tx1, ty1 = tb.max(0)
                rh = (ty1 - ty0) / (n + 1)
                cws = (tx1 - tx0) / 4
                for rr in range(n):
                    for cc in range(1, 4):
                        if rng.random() < 0.75:
                            v = B.hw_value(ctx)
                            B.draw_hw(pg, ctx, tx0 + cc * cws + 6, ty0 + (rr + 1) * rh + rh * 0.05, v, ctx.size * 0.95,
                                      bound=(tx0 + cc * cws + 2, ty0 + (rr + 1) * rh - 2, tx0 + (cc + 1) * cws - 2, ty0 + (rr + 2) * rh + 2))
        else:
            cols = 2 if rng.random() < 0.3 else 1
            for c in range(cols):
                if not labels:
                    break
                y_new = B.b_field(pg, ctx, x + c * (fw / cols), y, fw / cols - 20, label=labels.pop(), ymax=ymax)
            y = y_new
    y = B.b_para(pg, ctx, x, y + ctx.size * 0.4, w, n=rng.randint(2, 3), text="ខ្ញុំសូមធានាអះអាងថា ព័ត៌មានខាងលើពិតជាត្រឹមត្រូវប្រាកដមែន។ " + ctx.para(120))
    y = date_line(pg, ctx, x + w * 0.45, y, w * 0.55, "c", lunar=False)
    if rng.random() < 0.3:  # thumbprint box
        pg.d.rectangle([x + 20, y, x + 160, y + 170], outline=(90, 90, 90))
        q = pg.text(x + 90, y + 175, "ស្នាមមេដៃស្តាំ", ctx.font("body", "ស្នាមមេដៃស្តាំ"), int(ctx.size * 0.75), ctx.fg, align="c")
        pg.add("text", [q])
    B.b_sign(pg, ctx, x + w * 0.55, y, w * 0.4, role=rng.choice(["ហត្ថលេខាអ្នកស្នើសុំ", "អ្នកស្នើសុំ", "ហត្ថលេខា និងឈ្មោះ"]), stamp_p=0.1)
    return pg, ctx


# =========================================================================== 8. newspaper
def newspaper(rng, fake, lang):
    pg = new_page(rng, "a4", 0.15)
    ctx = Ctx(rng, lang, fake, style=rng.choice(["serif", "serif", "sans", "cond"]) if lang == "en" else None)
    ctx.size = rng.choice([18, 19, 20, 21, 22]) if lang == "km" else rng.choice([16, 17, 18, 19])
    ctx.justify = rng.random() < 0.7
    W, H = pg.W, pg.H
    m = int(W * 0.045)
    x, w = m, W - 2 * m
    y = m * 0.6
    if rng.random() < 0.7:  # top line: date / issue / price
        B.b_header(pg, ctx, x, y, w, text=f"{ctx.date()}  |  {ctx.label('លេខ', 'No.', 0.0)} {ctx.d(rng.randint(100, 9999))}", rule=False)
        y += ctx.size * 1.6
    mast = rng.choice(["កាសែតពេលព្រឹក", "ព័ត៌មានថ្មី", "សារព័ត៌មានកម្ពុជា", "នគរធំ", "Phnom Penh Daily", "ទស្សនាវដ្តី អង្គរ"]) if lang == "km" else \
        rng.choice(["The Daily Ledger", "Morning Courier", "City Herald", "The Evening Post", "Valley Tribune", "Harbor Gazette"])
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=mast, size=rng.randint(60, 95), align="c", role="display" if rng.random() < 0.6 else "title")
    pg.d.line([x, y, x + w, y], fill=ctx.fg, width=3)
    pg.d.line([x, y + 6, x + w, y + 6], fill=ctx.fg, width=1)
    y += 20
    # lead headline across columns
    if rng.random() < 0.8:
        y = B.b_heading(pg, ctx, x, y, w, cls="section_header", text=ctx.phrase(25, 70), size=ctx.size * rng.uniform(1.8, 2.6), align="l", role="bold")
    n = rng.choice([2, 3, 3, 4]) if lang == "en" else rng.choice([2, 3, 3])
    gap = int(W * 0.022)
    cw = (w - gap * (n - 1)) // n
    ybot = H - m * 1.3
    for c in range(n):
        cx = x + c * (cw + gap)
        if c > 0 and rng.random() < 0.5:
            pg.d.line([cx - gap / 2, y, cx - gap / 2, ybot], fill=(150, 150, 150), width=1)
        yy = y
        if c == 0 and rng.random() < 0.6:
            yy = B.b_figure(pg, ctx, cx, yy, cw, ybot, kind=rng.choice(["scene", "scene", "chart"]), fw=cw)
        B.fill_flow(pg, ctx, cx, yy, cw, ybot, [("para", 6), ("heading", 2), ("figure", 1.0)], max_blocks=14)
    B.b_footer(pg, ctx, x, H - m * 0.85, w, page_no=True, text=rng.random() < 0.4)
    return pg, ctx


def km_newspaper(rng, fake):
    return newspaper(rng, fake, "km")


# =========================================================================== 9. textbook / report
def textbook(rng, fake, lang):
    pg = new_page(rng, "a4", 0.06)
    ctx = Ctx(rng, lang, fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.08, 0.12))
    x, w = m, W - 2 * m
    if rng.random() < 0.75:
        B.b_header(pg, ctx, x, m * 0.45, w)
    y = m
    if rng.random() < 0.35:  # chapter opening
        ch = rng.randint(1, 12)
        y = B.b_heading(pg, ctx, x, y, w, cls="title", text=(f"ជំពូកទី{ctx.d(ch)}៖ " if lang == "km" else f"Chapter {ch}: ") + ctx.heading(),
                        size=ctx.size * rng.uniform(1.6, 2.2), align=rng.choice(["l", "c"]))
    fn_top = H - m * 1.05
    if rng.random() < 0.45:
        fn_top = B.b_footnotes(pg, ctx, x, H - m * 1.1, w)
    math_p = rng.random() < 0.45
    cols = 2 if rng.random() < 0.2 else 1
    kinds = [("para", 5), ("heading", 2), ("list", 1.2), ("table", 1.0), ("figure", 1.0)] + ([("formula", 2.0)] if math_p else [("formula", 0.2)])
    if cols == 1:
        B.fill_flow(pg, ctx, x, y, w, fn_top, kinds)
    else:
        gap = int(W * 0.03)
        cw = (w - gap) // 2
        for c in range(2):
            B.fill_flow(pg, ctx, x + c * (cw + gap), y, cw, fn_top, kinds)
    B.b_footer(pg, ctx, x, H - m * 0.65, w, page_no=rng.random() < 0.92, text=rng.random() < 0.3)
    return pg, ctx


# =========================================================================== 1b. press release / statement (letterhead row)
PR_TITLES = ["សេចក្តីប្រកាសព័ត៌មាន", "សេចក្តីថ្លែងការណ៍", "សេចក្តីថ្លែងការណ៍របស់អ្នកនាំពាក្យ", "សេចក្តីជូនដំណឹង", "សេចក្តីប្រកាស",
             "សេចក្តីណែនាំ", "សេចក្តីសម្រេច", "លិខិតអញ្ជើញ", "សារាចរ"]
PR_ORGS = [("រដ្ឋលេខាធិការដ្ឋានកិច្ចការព្រំដែន", "STATE SECRETARIAT OF BORDER AFFAIRS"), ("ក្រសួងការបរទេស និងសហប្រតិបត្តិការអន្តរជាតិ", "MINISTRY OF FOREIGN AFFAIRS"),
           ("ក្រសួងអប់រំ យុវជន និងកីឡា", "MINISTRY OF EDUCATION"), ("ទីស្តីការគណៈរដ្ឋមន្ត្រី", "OFFICE OF THE COUNCIL OF MINISTERS"),
           ("ក្រសួងព័ត៌មាន", "MINISTRY OF INFORMATION"), ("សាលារាជធានីភ្នំពេញ", "PHNOM PENH CAPITAL HALL"), ("ក្រសួងសុខាភិបាល", "MINISTRY OF HEALTH"),
           ("អគ្គស្នងការដ្ឋាននគរបាលជាតិ", "GENERAL COMMISSARIAT OF NATIONAL POLICE")]
PR_SECTIONS = ["លក្ខខណ្ឌនៃការដាក់ពាក្យ", "នីតិវិធីនៃការជ្រើសរើស", "កាលបរិច្ឆេទ និងនីតិវិធីនៃការដាក់ពាក្យ", "គោលបំណង", "អត្ថប្រយោជន៍", "ឯកសារតម្រូវ",
               "ទំនាក់ទំនង", "កម្មវិធីសិក្សា", "លក្ខខណ្ឌបេក្ខជន", "ការអនុវត្ត"]
PR_ADDR = ["អគារ Connexion (ជាន់ទី៣) វិថីកោះពេជ្រ សង្កាត់ទន្លេបាសាក់", "លេខ ៣ វិថី សម្តេច ហ៊ុន សែន សង្កាត់ទន្លេបាសាក់", "មហាវិថីព្រះនរោត្តម សង្កាត់ជ័យជំនះ ខណ្ឌដូនពេញ",
           "ផ្លូវលេខ ១៦៩ សង្កាត់វាលវង់ ខណ្ឌ៧មករា", "អគារលេខ ៨០ មហាវិថីព្រះនរោត្តម"]


def letterhead_row(pg, ctx, x, y, w, org):
    """Emblem (+ ministry name, reference) on the left, motto (+ ornament) on the right -> y below the row."""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(1.0, 1.15))
    left_w = w * rng.uniform(0.42, 0.5)
    emb = int(sz * rng.uniform(3.2, 4.6))
    emb_x = x + (left_w / 2 - emb / 2 if rng.random() < 0.6 else sz * 0.6)
    B.b_logo(pg, ctx, emb_x, y, emb, kind="emblem")
    y_emb = y + emb
    motto_cx = x + w - (w - left_w) / 2
    ym = motto(pg, ctx, motto_cx, y + sz * rng.uniform(0.0, 0.6), size=sz * rng.uniform(1.0, 1.2))
    yy = y_emb + sz * rng.uniform(0.3, 0.7)
    p = ctx.font("moul", org[0])
    name_cx = emb_x + emb / 2 if rng.random() < 0.5 else x
    al = "c" if name_cx != x else "l"
    pg.begin("header")
    q = B.fit_text(pg, name_cx, yy, org[0], p, sz, left_w * 1.15, ctx.fg, align=al)
    pg.end()
    qs = [q] if q else []
    if q:
        yy = bbox([q])[3] + sz * 0.5
    ref = f"លេខ{rng.choice(['ៈ', '៖', ':'])} {ctx.d(rng.randint(1, 999)).zfill(3) if rng.random() < 0.5 else ctx.d(rng.randint(1, 999))} {rng.choice(['កបទ.សអ', 'សជណ', 'រលក', 'អយក', 'ស.ណ.ន', 'ក.ប'])}"
    if rng.random() < 0.85:
        pg.begin("header")
        q2 = pg.text(name_cx, yy, ref, ctx.font("body", ref), int(sz * 0.95), ctx.fg, align=al)
        pg.end()
        if q2:
            qs.append(q2)
            yy = bbox([q2])[3]
    if qs:
        pg.add("header", qs)
    return max(yy, ym) + sz * rng.uniform(0.8, 1.6)


def footer_row(pg, ctx, x, y, w, org):
    """Address lines left, phone / fax / e-mail right (two footer blocks on one row) + page number above."""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.62, 0.75))
    if rng.random() < 0.6:
        pg.d.line([x, y - sz * 0.9, x + w, y - sz * 0.9], fill=(160, 160, 160), width=1)
    left = [rng.choice(PR_ADDR), rng.choice(["រាជធានីភ្នំពេញ ព្រះរាជាណាចក្រកម្ពុជា", "Phnom Penh, Kingdom of Cambodia", "ខណ្ឌចំការមន រាជធានីភ្នំពេញ"])]
    right = [f"ទូរស័ព្ទ{ctx.colon()} (+៨៥៥) {ctx.d(str(rng.randint(10, 99)))} {ctx.d(str(rng.randint(100, 999)))} {ctx.d(str(rng.randint(100, 999)))}",
             rng.choice([f"Tel: (+855) {rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}", f"E-mail: info@{rng.choice(['mfaic', 'moeys', 'ssba', 'moh', 'pressocm'])}.gov.kh",
                         f"ទូរសារ{ctx.colon()} (+៨៥៥) ២៣ {ctx.d(str(rng.randint(100, 999)))} {ctx.d(str(rng.randint(100, 999)))}"])]
    col = (90, 90, 90) if rng.random() < 0.6 else ctx.fg
    yy = y
    lq, rq = [], []
    two_cols = rng.random() < 0.8
    for i in range(2):
        pg.begin("footer")
        q = pg.text(x, yy, left[i], ctx.font("body", left[i]), sz, col)
        pg.end()
        if q:
            lq.append(q)
        if two_cols:
            pg.begin("footer")
            q2 = pg.text(x + w, yy, right[i], ctx.font("body", right[i]), sz, col, align="r")
            pg.end()
            if q2:
                rq.append(q2)
        yy += sz * 1.7
    if lq:
        pg.add("footer", lq)
    if rq:
        pg.add("footer", rq)
    return yy


def km_press_release(rng, fake):
    pg = new_page(rng, "a4", 0.05)
    ctx = Ctx(rng, "km", fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.07, 0.1))
    x, w = m, W - 2 * m
    org = rng.choice(PR_ORGS)
    y = letterhead_row(pg, ctx, x, int(H * rng.uniform(0.03, 0.05)), w, org)
    title = rng.choice(PR_TITLES)
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=title, size=ctx.size * rng.uniform(1.15, 1.5), align="c", role="moul", max_lines=1)
    if rng.random() < 0.4:  # second title line (the organisation)
        y = B.b_heading(pg, ctx, x, y - ctx.size * 0.3, w, cls="title", text=org[0], size=ctx.size * rng.uniform(1.0, 1.25), align="c", role="moul", max_lines=1)
    if rng.random() < 0.2:
        o = B.b_ornament(pg, ctx, x + w / 2, y + ctx.size * 0.2, ctx.size * rng.uniform(3.5, 7))
        y = (o[3] if o else y) + ctx.size * 0.5
    y += ctx.size * rng.uniform(0.3, 0.8)
    ymax = H * rng.uniform(0.74, 0.8)
    notice = rng.random() < 0.4  # scholarship-notice style: sections + lists; else statement: indented paragraphs
    ind = int(ctx.size * rng.uniform(1.8, 2.6))
    if notice:
        sec = 0
        letters = ["ក", "ខ", "គ", "ឃ", "ង", "ច"]
        while y < ymax - ctx.size * 4 and sec < 5:
            y = B.b_heading(pg, ctx, x, y, w, ymax, cls="section_header", text=f"{letters[sec]}) {rng.choice(PR_SECTIONS)}", size=ctx.size * rng.uniform(1.0, 1.1), role="bold", max_lines=1)
            r = rng.random()
            if r < 0.5:
                y = B.b_list(pg, ctx, x, y, w, ymax, n=rng.randint(2, 4), style=rng.choice(["dash", "dash", "bullet", "num"]))
            else:
                y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(2, 5), indent=ind if rng.random() < 0.7 else 0)
            sec += 1
    else:
        n_par = rng.randint(2, 4)
        for i in range(n_par):
            if y > ymax - ctx.size * 3:
                break
            org_lead = rng.random() < 0.5
            text = (org[0] + " " if org_lead else "") + ctx.para(rng.randint(260, 620))
            y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(3, 8), text=text, indent=ind)
            if rng.random() < 0.25 and y < ymax - ctx.size * 5:
                y = B.b_list(pg, ctx, x, y, w, ymax, n=rng.randint(2, 4), style="dash")
    # closing: date (lunar + civil) at the right, stamp over it
    y += ctx.size * rng.uniform(0.3, 0.9)
    if y < H * 0.9:
        dx = x + w * rng.uniform(0.38, 0.5)
        y_date = y
        y = date_line(pg, ctx, dx, y, x + w - dx, "r", lunar=rng.random() < 0.7)
        if rng.random() < 0.75:
            B.b_stamp_at(pg, ctx, dx + (x + w - dx) * rng.uniform(0.3, 0.6), (y_date + y) / 2 + ctx.size * rng.uniform(0.5, 2.0), d=ctx.size * rng.uniform(4.5, 6.5), org=org[0])
    pn_y = H - m * 0.95 - ctx.size * 1.6
    if rng.random() < 0.7:
        B.b_page_number(pg, ctx, x + w / 2, pn_y, "c", int(ctx.size * 0.8))
    footer_row(pg, ctx, x, H - m * 0.95 + ctx.size * 0.2, w, org)
    return pg, ctx


# =========================================================================== 1c. royal decree
ROYAL_LINES = ["ព្រះករុណាព្រះបាទសម្តេចព្រះបរមនាថ នរោត្តម សីហមុនី", "សមានភូមិជាតិសាសនា រក្ខតខត្តិយា ខេមរារដ្ឋរាស្ត្រ ពុទ្ធិន្ទ្រាធរាមហាក្សត្រ",
               "ខេមរាជនា សមូហោភាស កម្ពុជឯករាជរដ្ឋបូរណសន្តិ សុភមង្គលា សិរីវិបុលា", "ខេមរាស្រីពិរាស្ត្រ ព្រះចៅក្រុងកម្ពុជាធិបតី"]
DECREE_REFS = ["បានទ្រង់យល់រដ្ឋធម្មនុញ្ញ នៃព្រះរាជាណាចក្រកម្ពុជា", "បានទ្រង់យល់ព្រះរាជក្រឹត្យលេខ នស/រកត/{a}/{b} ចុះថ្ងៃទី{d} ខែ{m} ឆ្នាំ{y} ស្តីពីការតែងតាំងរាជរដ្ឋាភិបាល",
               "បានទ្រង់យល់ព្រះរាជក្រមលេខ នស/រកម/{a}/{b} ចុះថ្ងៃទី{d} ខែ{m} ឆ្នាំ{y} ដែលប្រកាសឱ្យប្រើច្បាប់ស្តីពីការរៀបចំ និងការប្រព្រឹត្តទៅនៃគណៈរដ្ឋមន្ត្រី",
               "បានទ្រង់យល់ព្រះរាជក្រឹត្យលេខ នស/រកត/{a}/{b} ចុះថ្ងៃទី{d} ខែ{m} ឆ្នាំ{y} ស្តីពីការកែសម្រួល និងតែងតាំងសមាសភាពរាជរដ្ឋាភិបាល",
               "បានទ្រង់យល់សេចក្តីក្រាបបង្គំទូលថ្វាយរបស់សម្តេចមហាបវរធិបតី នាយករដ្ឋមន្ត្រី នៃព្រះរាជាណាចក្រកម្ពុជា",
               "បានទ្រង់យល់ព្រះរាជក្រមលេខ នស/រកម/{a}/{b} ចុះថ្ងៃទី{d} ខែ{m} ឆ្នាំ{y} ដែលប្រកាសឱ្យប្រើច្បាប់ស្តីពីពន្ធនាគារ"]
KM_MONTHS = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា", "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]
DECREE_ARTICLES = ["លើកលែងទោសឱ្យទណ្ឌិតឈ្មោះ {n} ភេទប្រុស កើតឆ្នាំ{y} ដែលត្រូវបានតុលាការសម្រេចផ្តន្ទាទោសដាក់ពន្ធនាគារកំណត់ {k}ឆ្នាំ",
                   "ការលើកលែងទោសនេះអនុវត្តចំពោះតែទោសដាក់ពន្ធនាគារដែលជាមូលទោសប៉ុណ្ណោះ ដោយមិនអនុវត្តចំពោះទោសបន្ថែមឡើយ។",
                   "សម្តេចមហាបវរធិបតី នាយករដ្ឋមន្ត្រី នៃព្រះរាជាណាចក្រកម្ពុជា ត្រូវទទួលបន្ទុកអនុវត្តព្រះរាជក្រឹត្យនេះ ចាប់ពីថ្ងៃឡាយព្រះហស្តលេខាតទៅ។",
                   "តែងតាំង {n} ជា {r} នៃក្រសួង {min} ចាប់ពីថ្ងៃចុះហត្ថលេខាតទៅ។", "រដ្ឋមន្ត្រីក្រសួងពាក់ព័ន្ធ ត្រូវទទួលបន្ទុកអនុវត្តព្រះរាជក្រឹត្យនេះ តាមភារកិច្ចរៀងៗខ្លួន។"]


def _fill(ctx, t):
    rng = ctx.rng
    return t.format(a=ctx.d(f"{rng.randint(1, 12):02d}{rng.randint(18, 26)}"), b=ctx.d(str(rng.randint(100, 1999))), d=ctx.d(str(rng.randint(1, 28))),
                    m=rng.choice(KM_MONTHS), y=ctx.d(str(rng.randint(1993, 2026))), n=ctx.name(), k=ctx.d(str(rng.randint(2, 30))), r=rng.choice(["រដ្ឋលេខាធិការ", "អនុរដ្ឋលេខាធិការ", "អគ្គនាយក", "ទីប្រឹក្សា"]),
                    min=rng.choice(MINISTRIES)[0])


def km_royal_decree(rng, fake):
    pg = new_page(rng, "a4", 0.04)
    ctx = Ctx(rng, "km", fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.09, 0.12))
    x, w = m, W - 2 * m
    y = int(H * rng.uniform(0.025, 0.045))
    emb = int(ctx.size * rng.uniform(3.6, 5.0))
    B.b_logo(pg, ctx, x + w / 2 - emb / 2, y, emb, kind="emblem")
    y += emb + ctx.size * rng.uniform(0.5, 1.0)
    # reference number at the left, title centred (same row)
    ref = f"នស/រកត/{ctx.d(f'{rng.randint(1, 12):02d}{rng.randint(20, 27)}')}/{ctx.d(str(rng.randint(100, 999)))}"
    y_ref = y + ctx.size * rng.uniform(0.6, 1.6)
    pg.begin("header")
    q = pg.text(x, y_ref, ref, ctx.font("body", ref), int(ctx.size * 0.95), ctx.fg)
    pg.end()
    if q:
        pg.add("header", [q])
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=rng.choice(["ព្រះរាជក្រឹត្យ", "ព្រះរាជក្រម", "ព្រះរាជក្រឹត្យ"]), size=ctx.size * rng.uniform(1.4, 1.8), align="c", role="moul", max_lines=1)
    y = B.b_heading(pg, ctx, x, y - ctx.size * 0.2, w, cls="title", text="យើង", size=ctx.size * rng.uniform(1.2, 1.5), align="c", role="moul", max_lines=1)
    y += ctx.size * 0.3
    for t in ROYAL_LINES:  # centred bold royal titles
        y = B.b_para(pg, ctx, x, y, w, n=1, text=t, align="c", role="bold", indent=0, size=ctx.size * rng.uniform(1.0, 1.12))
        y -= ctx.size * 0.25
    y += ctx.size * rng.uniform(0.4, 0.8)
    refs = rng.sample(DECREE_REFS, rng.randint(3, 6))
    y = B.b_list(pg, ctx, x, y, w, H * 0.72, items=[_fill(ctx, t) for t in refs], style="dash")
    y = B.b_heading(pg, ctx, x, y + ctx.size * 0.3, w, cls="section_header", text="ត្រាស់បង្គាប់", size=ctx.size * rng.uniform(1.15, 1.4), align="c", role="moul" if rng.random() < 0.6 else "bold", max_lines=1)
    arts = rng.sample(DECREE_ARTICLES, rng.randint(2, 3))
    ymax = H * 0.9
    for i, t in enumerate(arts):
        if y > ymax - ctx.size * 3:
            break
        label = f"មាត្រា{ctx.d(str(i + 1))}{rng.choice([' : ', '៖ ', ' ៖ '])}"
        bp = ctx.font("bold", label)
        pg.begin("text", indent=0)
        lq = pg.text(x, y, label, bp, ctx.size, ctx.fg)
        lw = bbox([lq])[2] - x + ctx.size * 0.3 if lq else 0
        ye, qs = B.text_block(pg, ctx, x, y, w, _fill(ctx, t), "body", ctx.size, indent=int(lw), max_lines=6, ymax=ymax)
        pg.end()
        if qs or lq:
            pg.add("text", qs + ([lq] if lq else []))
        y = ye + ctx.size * rng.uniform(0.4, 0.8)
    # closing: place/date right, then two centred lines and the signature / stamp
    if y < H * 0.86:
        dx = x + w * 0.45
        y = date_line(pg, ctx, dx, y, x + w - dx, "r", lunar=False)
        for t in ["ក្នុងព្រះបរមនាម និងតាមព្រះរាជត្រាស់បង្គាប់", rng.choice(["ប្រមុខរដ្ឋស្តីទី", "នាយករដ្ឋមន្ត្រី"])]:
            y = B.b_para(pg, ctx, dx, y, x + w - dx, n=1, text=t, align="c", role="bold", indent=0)
            y -= ctx.size * 0.2
        if rng.random() < 0.7:
            B.b_stamp_at(pg, ctx, dx + (x + w - dx) * rng.uniform(0.25, 0.5), y + ctx.size * rng.uniform(0.5, 2.5), d=ctx.size * rng.uniform(4.5, 6.5))
        y += ctx.size * 3.5
        name = rng.choice(["សម្តេចអគ្គមហាសេនាបតីតេជោ ហ៊ុន សែន", "សម្តេចមហាបវរធិបតី ហ៊ុន ម៉ាណែត", ctx.name()])
        if y < H - m:
            B.b_para(pg, ctx, dx, y, x + w - dx, n=1, text=name, align="c", role="bold", indent=0)
    if rng.random() < 0.5:
        B.b_page_number(pg, ctx, x + w / 2, H - m * 0.8, "c", int(ctx.size * 0.8))
    return pg, ctx


def km_textbook(rng, fake):
    return textbook(rng, fake, "km")


from .templates_km_v4 import km_gov_notice, km_letter_v4  # noqa: E402  (v4 official notices / letters)

KM_TEMPLATES = {
    "km_gov_notice": (km_gov_notice, 2.6),
    "km_letter_v4": (km_letter_v4, 2.0),
    "km_official_letter": (km_official_letter, 1.4),
    "km_press_release": (km_press_release, 0.8),
    "km_royal_decree": (km_royal_decree, 0.6),
    "km_id_card": (km_id_card, 0.8),
    "km_passport": (km_passport, 0.5),
    "km_certificate": (km_certificate, 0.7),
    "km_invoice": (km_invoice, 1.2),
    "km_bank_statement": (km_bank_statement, 0.6),
    "km_application_form": (km_application_form, 1.1),
    "km_newspaper": (km_newspaper, 0.8),
    "km_textbook": (km_textbook, 1.3),
}
