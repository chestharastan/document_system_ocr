"""Contracts / agreements / company letters (Khmer + English) for the structure model (struct_v2).

What real phone photos of Cambodian business paperwork look like: a letterhead (logo + Khmer company name + Latin
name + address + Tel, often a rule line under it), a right-aligned date line with dotted blanks, a centred bold
title, numbered clauses with hanging indent (Arabic or Khmer numerals, sometimes "ប្រការ១៖" article headings),
"-" sub-items glued to their text, multi-line paragraphs, a closing paragraph, two signature columns and dotted
form fields. Every printed line gets a structure entry (canvas.Page.begin/_rec): role, paragraph, level, marker.

All names / companies / addresses are synthetic (random combinations)."""

from __future__ import annotations

from . import blocks as B
from .canvas import bbox, rect_quad
from .ctx import KH_COMMUNE, KH_DISTRICT, KH_VILLAGE, Ctx
from .res import font
from .templates_km import KINGDOM, motto, new_page

KM_BIZ = ["សន្តិសុខ", "សំណង់", "ដឹកជញ្ជូន", "អចលនទ្រព្យ", "ពាណិជ្ជកម្ម", "បច្ចេកវិទ្យា", "ធានារ៉ាប់រង", "ទេសចរណ៍", "អនាម័យ", "ផលិតកម្ម", "សេវាកម្ម", "កសិកម្ម"]
EN_BIZ = ["SECURITY", "CONSTRUCTION", "LOGISTICS", "REAL ESTATE", "TRADING", "TECHNOLOGY", "INSURANCE", "TRAVEL", "CLEANING SERVICES", "MANUFACTURING", "SERVICES", "AGRO"]
KM_SUFFIX = ["ឯ.ក", "ខូ អិលធីឌី", "", ""]
EN_SUFFIX = ["CO., LTD", "CO., LTD.", "LTD", "PLC", "GROUP", "INC."]
KM_TITLES = ["កិច្ចសន្យាជួលសេវាកម្ម", "កិច្ចព្រមព្រៀងសេវាសន្តិសុខ", "កិច្ចសន្យាជួលផ្ទះ", "កិច្ចសន្យាការងារ", "កិច្ចព្រមព្រៀងរវាងក្រុមហ៊ុន និងអតិថិជន",
             "កិច្ចសន្យាទិញលក់", "លិខិតព្រមព្រៀង", "កិច្ចសន្យាផ្គត់ផ្គង់ទំនិញ", "លក្ខខណ្ឌនៃការផ្តល់សេវា", "កិច្ចសន្យាខ្ចីប្រាក់", "លិខិតបញ្ជាក់ការងារ"]
EN_TITLES = ["SERVICE AGREEMENT", "SECURITY SERVICES CONTRACT", "LEASE AGREEMENT", "EMPLOYMENT CONTRACT", "TERMS AND CONDITIONS",
             "SUPPLY AGREEMENT", "MEMORANDUM OF UNDERSTANDING", "LOAN AGREEMENT", "CONFIRMATION LETTER", "Agreement between the Company and the Client"]
KM_FIELDS = ["ឈ្មោះអតិថិជន", "លេខផ្ទះ", "លេខផ្លូវ", "លេខទូរស័ព្ទ", "អាសយដ្ឋាន", "លេខអត្តសញ្ញាណប័ណ្ណ", "ថ្ងៃខែឆ្នាំកំណើត", "ហត្ថលេខា", "កាលបរិច្ឆេទ", "ឈ្មោះ"]
EN_FIELDS = ["Client name", "House No.", "Street", "Phone", "Address", "ID card No.", "Date of birth", "Signature", "Date", "Name", "Email"]
KM_SIGN = [("ហត្ថលេខា និងត្រា", "នាយកក្រុមហ៊ុន ឬ តំណាងក្រុមហ៊ុន"), ("តំណាងភាគី ក", "ហត្ថលេខា និងឈ្មោះ"), ("អ្នកជួល", "ស្នាមមេដៃ"),
           ("ស្នាមមេដៃអតិថិជន", None), ("តំណាងក្រុមហ៊ុន", None), ("ភាគី ខ", "ស្នាមមេដៃ និងឈ្មោះ"), ("សាក្សី", None), ("អ្នកទទួល", "ហត្ថលេខា")]
EN_SIGN = [("For and on behalf of the Company", "Director"), ("The Client", "Signature"), ("Witness", None), ("Party A", "Name and signature"),
           ("Party B", "Name and signature"), ("Authorized Signatory", None), ("Accepted by", "Date")]
KM_CLOSING = ["ភាគីទាំងពីរបានឯកភាពគ្នាទៅតាមកិច្ចព្រមព្រៀងខាងលើដោយគ្មានការបង្ខិតបង្ខំ។", "កិច្ចសន្យានេះមានសុពលភាពចាប់ពីថ្ងៃចុះហត្ថលេខាតទៅ។",
              "ភាគីណាមួយមិនគោរពតាមកិច្ចព្រមព្រៀងខាងលើ នឹងត្រូវទទួលខុសត្រូវចំពោះមុខច្បាប់ជាធរមាន។"]


def _company(ctx):
    rng = ctx.rng
    fam, _ = ctx.kh_name()
    k_biz = rng.randrange(len(KM_BIZ))
    km = f"ក្រុមហ៊ុន{KM_BIZ[k_biz]} {fam} {rng.choice(KM_SUFFIX)}".strip() if rng.random() < 0.6 else f"ក្រុមហ៊ុន {fam} {KM_BIZ[k_biz]}"
    lat = ctx.kh_name(latin=True)[0]
    en = f"{lat} {EN_BIZ[k_biz]} {rng.choice(EN_SUFFIX)}"
    return km, en


def _addr(ctx):
    rng = ctx.rng
    if ctx.lang == "km":
        return (f"{rng.choice(['ផ្ទះលេខ', 'អគារលេខ', 'លេខ'])}{ctx.d(rng.randint(1, 999))}{rng.choice(['', 'A', 'Eo'])} ផ្លូវលេខ{ctx.d(rng.randint(1, 600))} "
                f"{rng.choice(KH_VILLAGE)} {rng.choice(KH_COMMUNE)} {rng.choice(KH_DISTRICT)} {rng.choice(['រាជធានីភ្នំពេញ', 'ខេត្តកណ្ដាល', 'ខេត្តសៀមរាប'])}")
    f = ctx.fake
    return f"No. {rng.randint(1, 999)}, St. {rng.randint(1, 600)}, {f.city()}, Phnom Penh" if rng.random() < 0.5 else f.address().replace("\n", ", ")


def _tel(ctx):
    rng = ctx.rng
    n = lambda: f"0{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}"  # noqa: E731
    t = rng.choice(["Tel: ", "Tel : ", "ទូរស័ព្ទ៖ ", "Phone: ", "H/P: "]) + n() + (f" / {n()}" if rng.random() < 0.6 else "")
    if rng.random() < 0.3:
        t += f"  Email: info@{ctx.fake.domain_word()}.com"
    return t


def _line(pg, ctx, x, y, text, role, size, srole, align="l", fill=None, level=0, item=None, path=None):
    """One recorded line as its own paragraph."""
    p = path or ctx.font(role, text) or ctx.font("body", text)
    pg.begin(srole, level, item)
    q = pg.text(x, y, text, p, int(size), fill or ctx.fg, align=align)
    pg.end()
    return q


def letterhead(pg, ctx, x, y, w):
    """Logo left + stacked company block right of it (or centred) -> y below (rule line included)."""
    rng = ctx.rng
    sz = ctx.size
    km, en = _company(ctx)
    style = rng.choices(["logo_left", "centered", "text_only", "kingdom"], [0.55, 0.2, 0.12, 0.13])[0]
    qs = []
    if style == "kingdom":  # official style: kingdom motto right, company block left
        ym = motto(pg, ctx, x + w * rng.uniform(0.7, 0.78), y, size=sz * rng.uniform(1.0, 1.15))
        yy = ym + sz * 0.2
        for t, r, s in ((km, "moul", sz * 1.05), (en if rng.random() < 0.7 else None, "latin_bold", sz * 0.9)):
            if t:
                q = _line(pg, ctx, x, yy, t, r, s, "header")
                if q:
                    qs.append(q); yy = bbox([q])[3] + sz * 0.5
        if qs:
            pg.add("header", qs)
        return yy + sz * 0.8
    if style == "logo_left":
        ls = int(sz * rng.uniform(3.6, 5.2))
        B.b_logo(pg, ctx, x + rng.uniform(0, w * 0.06), y, ls, kind=rng.choice(["corp", "emblem"]))
        tx, al, yy = x + ls + sz * rng.uniform(0.4, 1.0) + w * rng.uniform(0, 0.04), "l", y + sz * rng.uniform(-0.2, 0.4)
        cx = tx
    else:
        tx, al, yy = x + w / 2, "c", y
        cx = tx
        if style == "centered" and rng.random() < 0.6:
            ls = int(sz * rng.uniform(2.6, 3.6))
            B.b_logo(pg, ctx, x + w / 2 - ls / 2, y, ls, kind="corp")
            yy = y + ls + sz * 0.3
    big = sz * rng.uniform(1.35, 1.9)
    q = _line(pg, ctx, cx, yy, km, "moul" if rng.random() < 0.7 else "bold", big, "header", align=al)
    if q:
        qs.append(q); yy = bbox([q])[3] + sz * rng.uniform(0.3, 0.6)
    if rng.random() < 0.85:
        q = _line(pg, ctx, cx, yy, en if rng.random() < 0.8 else en.title(), "latin_bold", sz * rng.uniform(1.05, 1.35), "header", align=al)
        if q:
            qs.append(q); yy = bbox([q])[3] + sz * rng.uniform(0.3, 0.6)
    for t in ([_addr(ctx)] if rng.random() < 0.9 else []) + ([_tel(ctx)] if rng.random() < 0.85 else []):
        q = _line(pg, ctx, cx, yy, t, "body" if not t.startswith(("Tel", "Phone", "H/P")) else "latin_bold", sz * rng.uniform(0.8, 0.95), "header", align=al)
        if q:
            qs.append(q); yy = bbox([q])[3] + sz * rng.uniform(0.25, 0.45)
    if qs:
        pg.add("header", qs)
    yy += sz * 0.3
    if rng.random() < 0.75:
        thick = rng.choice([1, 2, 3, 4])
        pg.d.line([x - w * 0.03, yy, x + w * 1.03, yy], fill=ctx.fg, width=thick)
        if rng.random() < 0.3:
            pg.d.line([x - w * 0.03, yy + thick + 3, x + w * 1.03, yy + thick + 3], fill=ctx.fg, width=1)
        yy += sz * 0.5
    return yy


def date_line(pg, ctx, x, y, w):
    rng = ctx.rng
    sz = ctx.size * rng.uniform(0.9, 1.0)
    dots = lambda a, b: "." * rng.randint(a, b)  # noqa: E731
    if ctx.lang == "km":
        city = rng.choice(["រាជធានីភ្នំពេញ", "ភ្នំពេញ", "ខេត្តសៀមរាប", "ខេត្តកណ្ដាល", "ខេត្តបាត់ដំបង"])
        if rng.random() < 0.6:
            t = f"{city} ថ្ងៃទី{dots(6, 14)} ខែ{dots(6, 14)} ឆ្នាំ{ctx.d(rng.choice([202, 2025, 2026, 20]))}{dots(3, 8)}"
        else:
            t = f"{city} {ctx.date()}" if rng.random() < 0.5 else f"{city} ថ្ងៃទី{ctx.d(rng.randint(1, 28))} ខែ{rng.choice(['មករា', 'មីនា', 'ឧសភា', 'កក្កដា', 'កញ្ញា', 'ធ្នូ'])} ឆ្នាំ{ctx.d(rng.randint(2019, 2027))}"
    else:
        t = rng.choice([f"Phnom Penh, {dots(8, 16)} 20{dots(2, 4)}", f"Date: {dots(10, 20)}", f"Phnom Penh, {ctx.date()}", ctx.date()])
    al = rng.choices(["r", "c", "l"], [0.8, 0.1, 0.1])[0]
    ax = x + w if al == "r" else (x + w * 0.5 if al == "c" else x)
    q = _line(pg, ctx, ax, y, t, "body", sz, "text", align=al)
    if q:
        pg.add("text", [q])
        return bbox([q])[3] + sz * rng.uniform(0.3, 0.8)
    return y


def title(pg, ctx, x, y, w):
    rng = ctx.rng
    sz = ctx.size * rng.uniform(1.1, 1.5)
    t = rng.choice(KM_TITLES if ctx.lang == "km" else EN_TITLES)
    if ctx.lang == "km" and rng.random() < 0.4:
        t += f" នៅក្នុង{rng.choice(['បុរី', 'ភូមិ', 'គម្រោង'])} {rng.choice(['LC', 'ABC', 'Borey Peng Huoth', 'ពិភពថ្មី'])}" + (f" ({rng.choice(['អង្គតាម៉ិញ', 'ចំការដូង', 'សែនសុខ'])})" if rng.random() < 0.5 else "")
    ye, qs = B.text_block(pg, ctx, x + w * 0.04, y, w * 0.92, t, rng.choice(["moul", "bold", "title"]), int(sz), align="c", max_lines=2, lh=1.4, srole="title")
    if qs:
        pg.add("title", qs)
        if rng.random() < 0.25:
            b = bbox(qs)
            pg.d.line([b[0], b[3] + 4, b[2], b[3] + 4], fill=ctx.fg, width=2)
        return ye + sz * rng.uniform(0.1, 0.5)
    return y


def parties(pg, ctx, x, y, w, ymax):
    """"ភាគី ក៖ ..." / "Party A: ..." one-to-two-line paragraphs."""
    rng = ctx.rng
    sz = ctx.size
    labels = ["ភាគី ក", "ភាគី ខ"] if ctx.lang == "km" else ["Party A", "Party B"]
    for lab in labels:
        t = f"{lab}{ctx.colon()} " + (ctx.name() + " " + ctx.para(rng.randint(40, 160)) if ctx.lang == "km" else f"{ctx.fake.company()}, {ctx.para(rng.randint(40, 140))}")
        y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(1, 3), text=t, indent=0)
    return y


def marker(ctx, i, style):
    if style == "num":
        return f"{i + 1}."
    if style == "kmnum":
        return f"{ctx.d(i + 1)}."
    if style == "paren":
        return f"{ctx.d(i + 1)})"
    if style == "article":
        return (f"ប្រការ{ctx.d(i + 1)}{ctx.colon()}" if ctx.lang == "km" else f"Article {i + 1}:")
    if style == "alpha":
        return ("abcdefgh"[i % 8] + ")") if ctx.lang == "en" else ("កខគឃងចឆជ"[i % 9] + ".")
    return "-"


def clause(pg, ctx, x, y, w, ymax, mark, text, hang: bool, level: int, item: str, size=None, glued=False, gap=None):
    """A list item: marker at x, text at x + gap, continuation lines under the text (hang) or back at x.
    glued: the marker is part of the text line ("-អតិថិជន...") instead of a separate drawn marker."""
    rng = ctx.rng
    sz = int(size or ctx.size)
    p = ctx.font("body", text)
    mp = ctx.font("body", mark) or ctx.font("latin", mark)
    if not p or not mp:
        return y
    if glued:
        pg.begin("list_item", level, item)
        try:
            ye, qs = B._text_block(pg, ctx, x, y, w, mark + text, "body", sz, None, "l", ctx.justify, None, 99, 0, ymax, "print", 1.0, p) if not hang \
                else _hanging(pg, ctx, x, y, w, ymax, mark + text, p, sz, font(p, sz).getlength(mark))
        finally:
            pg.end()
        return ye if qs else y
    mw = font(mp, sz).getlength(mark)
    gap = gap if gap is not None else max(mw + sz * rng.uniform(0.3, 0.8), sz * rng.uniform(1.2, 2.0))
    pg.begin("list_item", level, item)
    try:
        if hang:
            ye, qs = B._text_block(pg, ctx, x + gap, y, w - gap, text, "body", sz, None, "l", ctx.justify, None, 99, 0, ymax, "print", 1.0, p)
        else:
            ye, qs = B._text_block(pg, ctx, x, y, w, text, "body", sz, None, "l", ctx.justify, None, 99, gap, ymax, "print", 1.0, p)
        if qs:
            pg.text(x, y, mark, mp, sz, ctx.fg, marker=True)
    finally:
        pg.end()
    return ye if qs else y


def _hanging(pg, ctx, x, y, w, ymax, text, p, sz, off):
    """Glued marker + hanging continuation lines (first line at x, the rest at x + off)."""
    f = font(p, sz)
    from .canvas import wrap
    first = wrap(text, f, w)
    if not first:
        return y, []
    head = first[0]
    rest = text[len(head):].strip()
    step = sz * ctx.lh
    qs = []
    q = pg.text(x, y, head, p, sz, ctx.fg)
    if q:
        qs.append(q)
    yy = y + step
    for ln in (wrap(rest, f, w - off) if rest else []):
        if yy + sz * 1.45 > ymax:
            break
        q = pg.text(x + off, yy, ln, p, sz, ctx.fg)
        if q:
            qs.append(q)
        yy += step
    if not qs:
        return y, []
    return max(yy, max(max(q[1::2]) for q in qs) + sz * 0.25), qs


def clauses(pg, ctx, x, y, w, ymax):
    rng = ctx.rng
    sz = ctx.size
    style = rng.choices(["num", "kmnum", "paren", "article", "alpha"], [0.4, 0.25 if ctx.lang == "km" else 0.0, 0.1, 0.15, 0.1])[0]
    hang = rng.random() < 0.7
    indent = sz * rng.uniform(0.0, 1.6)  # the numbers often sit right of the margin
    n = rng.randint(3, 8)
    gap = None if rng.random() < 0.5 else sz * rng.uniform(1.3, 2.2)
    sub_style = rng.choice(["dash", "dash", "dash", "alpha", "bullet"])
    allq_start = len(pg.lines)
    for i in range(n):
        if y > ymax - sz * 3:
            break
        t = ctx.para(rng.choice([rng.randint(30, 90), rng.randint(90, 420)]))
        if style == "article" and rng.random() < 0.5:  # article heading on its own line, body paragraph below
            q = _line(pg, ctx, x + indent, y, marker(ctx, i, style) + " " + ctx.heading(), "bold", sz, "section_header", level=0)
            if q:
                y = bbox([q])[3] + sz * 0.4
            y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(1, 4), text=t, indent=int(sz * rng.choice([0, 0, 1.6])))
            continue
        y = clause(pg, ctx, x + indent, y, w - indent, ymax, marker(ctx, i, style), t, hang, 1, style, gap=gap)
        y += sz * rng.choice([0.0, 0.0, 0.15, 0.5])
        if rng.random() < 0.3:  # dash sub-items
            sub_x = x + indent + (sz * rng.uniform(1.0, 2.2) if rng.random() < 0.8 else 0)
            glued = rng.random() < 0.7
            for j in range(rng.randint(2, 5)):
                if y > ymax - sz * 2:
                    break
                st = ctx.para(rng.choice([rng.randint(20, 70), rng.randint(70, 260)]))
                mk = "-" if sub_style == "dash" else ("•" if sub_style == "bullet" else marker(ctx, j, "alpha"))
                y = clause(pg, ctx, sub_x, y, x + w - sub_x, ymax, mk, st, rng.random() < 0.5, 2, sub_style, glued=glued and mk == "-",
                           gap=sz * rng.uniform(0.9, 1.4))
            y += sz * rng.choice([0.0, 0.2])
    qs = [q for q, s in zip(pg.lines[allq_start:], pg.struct[allq_start:])]
    if qs:
        pg.add("list", qs)
    return y


def closing(pg, ctx, x, y, w, ymax):
    rng = ctx.rng
    t = (" ".join(rng.sample(KM_CLOSING, rng.randint(1, 2))) + (" " + ctx.para(rng.randint(20, 120)) if rng.random() < 0.5 else "")) if ctx.lang == "km" \
        else ctx.para(rng.randint(80, 260))
    return B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(1, 3), text=t, indent=int(ctx.size * rng.choice([0, 0, 1.8])))


def signatures(pg, ctx, x, y, w, ymax):
    """Two signature columns (labels + optional signature ink / stamp + name), sometimes three."""
    rng = ctx.rng
    sz = ctx.size * rng.uniform(0.9, 1.0)
    pool = KM_SIGN if ctx.lang == "km" else EN_SIGN
    ncol = rng.choices([2, 1, 3], [0.75, 0.15, 0.1])[0]
    picks = rng.sample(pool, ncol)
    yb = y
    for c, (a, b) in enumerate(picks):
        if ncol == 1:
            cx = x + w * rng.choice([0.2, 0.75])
        else:
            cx = x + w * (0.12 + 0.76 * c / (ncol - 1)) + rng.uniform(-0.05, 0.05) * w
        al = "c"
        yy = y + (rng.uniform(-0.3, 0.3) * sz if c else 0)
        q = _line(pg, ctx, cx, yy, a, rng.choice(["bold", "body", "moul"]), sz, "text", align=al)
        if q:
            if rng.random() < 0.3:
                bb = bbox([q])
                pg.d.line([bb[0], bb[3] + 3, bb[2], bb[3] + 3], fill=ctx.fg, width=2)
            yy = bbox([q])[3] + sz * 0.4
        if b:
            q = _line(pg, ctx, cx, yy, b, "body", sz, "text", align=al)
            if q:
                yy = bbox([q])[3] + sz * 0.4
        if rng.random() < 0.4:
            try:
                B.b_sign(pg, ctx, cx - w * 0.15, yy + sz * 0.3, w * 0.3, ymax, role=None, name=None, stamp_p=0.3)
            except Exception:
                pass
        yb = max(yb, yy + sz * rng.uniform(1.5, 5.0))
    return yb


def form_fields(pg, ctx, x, y, w, ymax):
    rng = ctx.rng
    sz = ctx.size * rng.uniform(0.9, 1.05)
    labels = rng.sample(KM_FIELDS if ctx.lang == "km" else EN_FIELDS, rng.randint(2, 5))
    fx = x + (w * rng.uniform(0.45, 0.6) if rng.random() < 0.6 else rng.uniform(0, 0.1) * w)
    dash = rng.random() < 0.6
    qs = []
    for lab in labels:
        if y > ymax:
            break
        t = ("-" if dash else "") + lab + ctx.colon() + "." * rng.randint(15, 45)
        q = _line(pg, ctx, fx, y, t, "bold" if rng.random() < 0.5 else "body", sz, "form_field")
        if q:
            qs.append(q)
            y = bbox([q])[3] + sz * rng.uniform(0.6, 1.2)
    if qs:
        pg.add("form_field", qs)
    return y


def contract(rng, fake, lang):
    pg = new_page(rng, rng.choice(["a4", "a4", "letter"]), 0.08)
    ctx = Ctx(rng, lang, fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.06, 0.11))
    x, w = m, W - 2 * m
    y = int(H * rng.uniform(0.025, 0.05))
    sz = ctx.size
    pg.meta["degrade_p"] = 1.0
    if rng.random() < 0.92:
        y = letterhead(pg, ctx, x, y, w)
    if rng.random() < 0.75:
        y = date_line(pg, ctx, x, y, w)
    ymax = H - m * rng.uniform(0.6, 1.0)
    if rng.random() < 0.9:
        y = title(pg, ctx, x, y, w)
    if rng.random() < 0.3:
        y = parties(pg, ctx, x, y, w, ymax - sz * 10)
    if rng.random() < 0.35:
        y = B.b_para(pg, ctx, x, y, w, ymax - sz * 10, n=rng.randint(1, 3), text=ctx.para(rng.randint(80, 300)), indent=int(sz * rng.choice([0, 1.8, 2.4])))
    y = clauses(pg, ctx, x, y, w, ymax - sz * 6)
    if rng.random() < 0.7 and y < ymax - sz * 6:
        y = closing(pg, ctx, x, y, w, ymax - sz * 5)
    if y < ymax - sz * 3:
        y = signatures(pg, ctx, x, y + sz * rng.uniform(0.2, 1.0), w, ymax)
    if rng.random() < 0.55 and y < ymax - sz * 3:
        y = form_fields(pg, ctx, x, y + sz * rng.uniform(0.5, 4), w, ymax)
    if rng.random() < 0.3:
        t = _addr(ctx) if rng.random() < 0.6 else f"{ctx.d(1)}/{ctx.d(rng.randint(1, 3))}"
        q = _line(pg, ctx, x + w / 2, H - m * 0.7, t, "body", sz * 0.75, "footer" if len(t) > 6 else "page_number", align="c")
        if q:
            pg.add("footer" if len(t) > 6 else "page_number", [q])
    return pg, ctx


def company_letter(rng, fake, lang):
    """Business letter: letterhead, date, addressee, subject, body paragraphs with an optional list, sign-off."""
    pg = new_page(rng, rng.choice(["a4", "letter"]), 0.08)
    ctx = Ctx(rng, lang, fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.07, 0.11))
    x, w = m, W - 2 * m
    sz = ctx.size
    pg.meta["degrade_p"] = 1.0
    y = letterhead(pg, ctx, x, int(H * rng.uniform(0.025, 0.05)), w)
    y = date_line(pg, ctx, x, y, w)
    ymax = H - m
    if rng.random() < 0.6:
        to = ["សូមគោរពជូន", ctx.name(), _addr(ctx)] if lang == "km" else ["To:", ctx.fake.name(), ctx.fake.company()]
        for t in to[: rng.randint(2, 3)]:
            q = _line(pg, ctx, x, y, t, "body", sz, "text")
            if q:
                y = bbox([q])[3] + sz * 0.5
    sub = (f"កម្មវត្ថុ{ctx.colon()} " if lang == "km" else "Subject: ") + ctx.para(rng.randint(30, 120))
    y = B.b_para(pg, ctx, x, y + sz * 0.3, w, ymax, n=2, text=sub, indent=0, role="bold" if rng.random() < 0.5 else "body")
    for _ in range(rng.randint(1, 3)):
        y = B.b_para(pg, ctx, x, y, w, ymax - sz * 8, n=rng.randint(2, 6), text=ctx.para(rng.randint(150, 500)), indent=int(sz * rng.choice([0, 1.8, 2.4])))
    if rng.random() < 0.6:
        y = clauses(pg, ctx, x, y, w, ymax - sz * 7)
    if rng.random() < 0.6:
        y = closing(pg, ctx, x, y, w, ymax - sz * 5)
    signatures(pg, ctx, x, y + sz * 0.5, w, ymax)
    return pg, ctx


def km_contract(rng, fake):
    return contract(rng, fake, "km")


def en_contract(rng, fake):
    return contract(rng, fake, "en")


def km_company_letter(rng, fake):
    return company_letter(rng, fake, "km")


def en_company_letter(rng, fake):
    return company_letter(rng, fake, "en")


CONTRACT_KM = {"km_contract": (km_contract, 3.0), "km_company_letter": (km_company_letter, 1.0)}
CONTRACT_EN = {"en_contract": (en_contract, 2.0), "en_company_letter": (en_company_letter, 1.0)}
