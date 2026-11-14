"""US / English document families. Names, addresses, companies from Faker (synthetic). Forms are generic (no copy of
any real government form)."""

from __future__ import annotations

from . import blocks as B
from .canvas import Page, bbox
from .ctx import Ctx, US_STATES
from .res import font
from .templates_km import _bank_statement, new_page, newspaper, pay_qr_block, textbook, thermal_receipt, totals, KH_PRODUCTS_EN


def letterhead(pg, ctx, x, y, w):
    rng, fk = ctx.rng, ctx.fake
    ls = int(ctx.size * rng.uniform(2.5, 4))
    comp = fk.company()
    side = rng.choice(["l", "c", "r"])
    lx = {"l": x, "c": x + w / 2 - ls / 2, "r": x + w - ls}[side]
    if rng.random() < 0.85:
        B.b_logo(pg, ctx, lx, y, ls, kind="corp", wordmark=comp if (side == "l" and rng.random() < 0.5) else None)
        yy = y + ls + 10
    else:
        yy = y
    lines = [comp if side != "l" or rng.random() < 0.5 else None, fk.street_address(), f"{fk.city()}, {rng.choice(US_STATES)} {fk.zipcode()}",
             rng.choice([f"Tel {fk.phone_number()}", f"{fk.phone_number()} | {fk.company_email()}", "www." + fk.domain_name()])]
    qs = []
    al = {"l": "l", "c": "c", "r": "r"}[side]
    ax = {"l": x, "c": x + w / 2, "r": x + w}[side]
    for i, t in enumerate([t for t in lines if t]):
        role = "bold" if (i == 0 and t == comp) else "body"
        q = pg.text(ax, yy, t, ctx.font(role, t), int(ctx.size * (1.05 if role == "bold" else 0.8)), ctx.accent if role == "bold" else (80, 80, 80), align=al)
        if q:
            qs.append(q)
            yy = bbox([q])[3] + ctx.size * 0.35
    if qs:
        pg.add("header", qs)
    if rng.random() < 0.6:
        pg.d.line([x, yy + 10, x + w, yy + 10], fill=ctx.accent, width=rng.choice([1, 2, 3]))
    return yy + ctx.size * 1.5


def addr_block(pg, ctx, x, y, lines, size=None, cls="text", bound=None):
    qs = []
    sz = int(size or ctx.size)
    for t in lines:
        q = pg.text(x, y, t, ctx.font("body", t), sz, ctx.fg, bound=bound)
        if q:
            qs.append(q)
            y = bbox([q])[3] + sz * 0.4
    if qs:
        pg.add(cls, qs)
    return y + sz * 0.6


# =========================================================================== US letter
def us_letter(rng, fake):
    pg = new_page(rng, "letter", 0.06)
    ctx = Ctx(rng, "en", fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.08, 0.12))
    x, w = m, W - 2 * m
    if rng.random() < 0.05:
        B.watermark(pg, ctx)
    y = letterhead(pg, ctx, x, int(H * 0.04), w)
    y = addr_block(pg, ctx, x, y, [ctx.date()])
    y = addr_block(pg, ctx, x, y, [fake.name(), fake.job()[:40], fake.company(), fake.street_address(), f"{fake.city()}, {rng.choice(US_STATES)} {fake.zipcode()}"][: rng.randint(3, 5)])
    if rng.random() < 0.6:
        t = rng.choice(["Re: ", "RE: ", "Subject: "]) + ctx.phrase(15, 60)
        y = B.b_heading(pg, ctx, x, y, w, cls="section_header", text=t, size=ctx.size, role="bold")
    y = addr_block(pg, ctx, x, y, [f"Dear {rng.choice(['Mr.', 'Ms.', 'Dr.', 'Mx.'])} {fake.last_name()}," if rng.random() < 0.8 else "To whom it may concern:"])
    fn_top = H - m
    if rng.random() < 0.15:
        fn_top = B.b_footnotes(pg, ctx, x, H - m * 1.0, w, n=1)
    y = B.fill_flow(pg, ctx, x, y, w, min(fn_top, H * 0.78), [("para", 8), ("list", 1)], max_blocks=6)
    y = B.b_sign(pg, ctx, x, y, w * 0.4, fn_top, role=rng.choice(["Sincerely,", "Best regards,", "Respectfully,", "Yours truly,"]), align="l",
                 stamp_p=0.08)
    if rng.random() < 0.5 and y < fn_top - 60:
        addr_block(pg, ctx, x, y, [rng.choice(["Enclosure", "Enclosures (2)", f"cc: {fake.name()}", "Attachment: " + ctx.phrase(8, 30)])], size=ctx.size * 0.85)
    if rng.random() < 0.4:
        B.b_footer(pg, ctx, x, H - m * 0.55, w, page_no=rng.random() < 0.6, text=rng.random() < 0.6)
    return pg, ctx


# =========================================================================== US invoice
def us_invoice(rng, fake):
    ctx = Ctx(rng, "en", fake)
    if rng.random() < 0.35:
        return thermal_receipt(rng, fake, ctx)
    pg = new_page(rng, "letter", 0.08)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.06, 0.09))
    x, w = m, W - 2 * m
    y = letterhead(pg, ctx, x, int(H * 0.035), w * 0.6)
    title = rng.choice(["INVOICE", "Invoice", "TAX INVOICE", "SALES RECEIPT", "BILL", "STATEMENT OF ACCOUNT", "PURCHASE ORDER", "QUOTE"])
    q = pg.text(x + w, int(H * 0.04), title, ctx.font("title", title), int(ctx.size * rng.uniform(1.8, 2.6)), ctx.accent, align="r")
    if q:
        pg.add("title", [q])
        yy = bbox([q])[3] + ctx.size
        meta = [f"Invoice #: {rng.randint(1000, 99999)}", f"Date: {ctx.date()}", f"Due: {ctx.date()}", f"PO #: {rng.randint(10000, 999999)}"][: rng.randint(2, 4)]
        qs = []
        for t in meta:
            qq = pg.text(x + w, yy, t, ctx.font("body", t), int(ctx.size * 0.9), ctx.fg, align="r")
            if qq:
                qs.append(qq)
                yy = bbox([qq])[3] + ctx.size * 0.4
        pg.add("text", qs)
        y = max(y, yy + ctx.size)
    # bill to / ship to
    for i, lab in enumerate(rng.sample(["BILL TO", "SHIP TO", "Bill To:", "Customer"], rng.choice([1, 2]))):
        bx = x + i * w * 0.5
        q = pg.text(bx, y, lab, ctx.font("bold", lab), int(ctx.size * 0.95), ctx.accent)
        ye = addr_block(pg, ctx, bx, (bbox([q])[3] + 8) if q else y, [fake.name(), fake.company(), fake.street_address(), f"{fake.city()}, {rng.choice(US_STATES)} {fake.zipcode()}"][: rng.randint(2, 4)],
                        bound=(0, 0, bx + w * 0.48, H))
        if q:
            pg.add("section_header", [q])
    y = ye + ctx.size * 0.3
    y = B.b_table(pg, ctx, x, y, w, H * 0.74, kind="items", nrow=rng.randint(2, 14), style=rng.choice(["grid", "zebra", "hlines", "none"]))
    y = totals(pg, ctx, x + w * 0.45, y, w * 0.55)
    if rng.random() < 0.35:
        y = B.b_check_group(pg, ctx, x, y, w * 0.7, q="Payment method", opts=["Check", "Credit card", "Wire transfer", "ACH"])
    if rng.random() < 0.6:
        y2 = B.b_heading(pg, ctx, x, y, w * 0.6, text=rng.choice(["Notes", "Terms & Conditions", "Payment Terms"]), size=ctx.size, role="bold")
        y = B.b_para(pg, ctx, x, y2, w * 0.6, n=rng.randint(2, 4), size=ctx.size * 0.85)
    if rng.random() < 0.35:
        st = B.make_stamp(ctx, d=rng.uniform(170, 240))
        pg.ink(st, (x + w * rng.uniform(0.3, 0.6), y - rng.uniform(0, 120)), "stamp")
    if rng.random() < 0.3 and y < H - 300:
        pay_qr_block(pg, ctx, x + w - 260, y, 200)
    elif rng.random() < 0.3 and y < H - 200:
        B.b_barcode(pg, ctx, x, y + 10, rng.randint(240, 360), rng.randint(60, 90))
    if rng.random() < 0.3 and y < H - 300:
        B.b_sign(pg, ctx, x, y + 20, w * 0.35, role="Authorized Signature", align="l", stamp_p=0.0)
    if rng.random() < 0.6:
        t = rng.choice(["Thank you for your business!", f"Questions? Contact {fake.company_email()}", f"Make all checks payable to {fake.company()}"])
        q = pg.text(W / 2, H - m * 0.9, t, ctx.font("italic", t), int(ctx.size * 0.85), ctx.fg, align="c")
        pg.add("footer", [q])
        if rng.random() < 0.5:
            B.b_page_number(pg, ctx, x + w, H - m * 0.9, "r", ctx.size * 0.8)
    return pg, ctx


# =========================================================================== generic government / tax-style form
AGENCIES = ["Department of Revenue Services", "Office of Licensing and Permits", "Bureau of Motor Vehicles", "Department of Labor and Workforce",
            "County Assessor's Office", "Office of Vital Records", "Department of Housing", "State Benefits Administration"]


def us_tax_form(rng, fake):
    pg = new_page(rng, "letter", 0.12)
    ctx = Ctx(rng, "en", fake, style=rng.choice(["sans", "sans", "cond", "serif"]))
    ctx.size = rng.choice([18, 19, 20, 21])
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.05, 0.07))
    x, w = m, W - 2 * m
    y = int(H * 0.03)
    d = pg.d
    # top band: form id | title | year
    fid = f"Form {rng.choice(['DR', 'ST', 'BL', 'UC', 'HS', 'TX'])}-{rng.randint(10, 999)}{rng.choice(['', 'A', 'EZ', 'S'])}"
    q1 = pg.text(x, y, fid, ctx.font("bold", fid), int(ctx.size * 1.3), ctx.fg)
    yr = str(rng.randint(2019, 2027))
    q3 = pg.text(x + w, y, yr, ctx.font("bold", yr), int(ctx.size * 2.2), ctx.fg, align="r")
    if q1:
        pg.add("header", [q1])
    if q3:
        pg.add("header", [q3])
    ag = rng.choice(AGENCIES)
    title = rng.choice(["Application for ", "Annual Return of ", "Statement of ", "Claim for ", "Request for "]) + rng.choice(["Income", "Business License", "Refund", "Unemployment Benefits", "Property Tax Exemption", "Withholding Allowance", "Change of Address"])
    yt = y
    qa = pg.text(W / 2, yt, ag, ctx.font("body", ag), int(ctx.size * 0.85), ctx.fg, align="c", bound=(x + w * 0.2, 0, x + w * 0.8, H))
    if qa:
        pg.add("header", [qa])
        yt = bbox([qa])[3] + 6
    yt = B.b_heading(pg, ctx, x + w * 0.2, yt, w * 0.6, cls="title", text=title, size=ctx.size * 1.35, align="c", role="bold")
    y = max(yt, y + ctx.size * 3) + 6
    d.line([x, y, x + w, y], fill=(0, 0, 0), width=3)
    y += 14
    if rng.random() < 0.6:
        y = B.b_para(pg, ctx, x, y, w, n=2, size=ctx.size * 0.85, text=rng.choice(["Instructions: ", "Note: ", "▶ "]) + ctx.para(160), indent=0)
    ymax = H * 0.8
    line_no = 0
    two_col = rng.random() < 0.4
    amount_col = rng.random() < 0.6
    while y < ymax - ctx.size * 3:
        r = rng.random()
        if r < 0.15:
            y = B.b_check_group(pg, ctx, x, y, w, ymax, q=rng.choice(["Filing status", "Check one", "Type of entity", "Residency"]),
                                opts=rng.choice(B.CHECK_OPTS["en"]))
        elif r < 0.22:
            t = rng.choice(["Part I", "Part II", "Section A", "Section B", "Part III"]) + "  " + ctx.heading()
            bh = int(ctx.size * 1.6)
            if y + bh > ymax:
                break
            d.rectangle([x, y, x + w, y + bh], fill=rng.choice([(30, 30, 30), (225, 225, 225), (40, 60, 110)]))
            dark = pg.img.getpixel((int(x + 2), int(y + 2)))[0] < 128
            q = pg.text(x + 8, y + (bh - ctx.size * 1.2) / 2, t, ctx.font("bold", t), ctx.size, (255, 255, 255) if dark else (0, 0, 0), bound=(x, y - 4, x + w, y + bh + 4))
            if q:
                pg.add("section_header", [q])
            y += bh + 10
        elif amount_col and r < 0.6:  # numbered line with amount box at the right
            line_no += 1
            lab = f"{line_no}{rng.choice(['', 'a', 'b']) if rng.random() < 0.2 else ''}  " + rng.choice(
                ["Wages, salaries, and tips", "Taxable interest", "Total income", "Adjustments to income", "Standard deduction", "Taxable amount",
                 "Credits", "Total payments", "Amount you owe", "Refund", "Estimated tax penalty", "Other income", "Gross receipts", "Total deductions"])
            bw = w * 0.22
            p = ctx.font("body", lab)
            ql = pg.text(x, y, lab, p, ctx.size, ctx.fg, bound=(0, 0, x + w - bw - 20, H))
            if not ql:
                break
            # dot leader
            lb = bbox([ql])
            for dx in range(int(lb[2] + 10), int(x + w - bw - 30), 9):
                d.point((dx, lb[3] - 2), fill=(90, 90, 90))
            bx0 = x + w - bw
            d.rectangle([bx0, y - ctx.size * 0.3, x + w, y + ctx.size * 1.35], outline=(0, 0, 0), width=1)
            nq = pg.text(bx0 + 4, y - ctx.size * 0.25, str(line_no), ctx.font("body", str(line_no)), int(ctx.size * 0.6), ctx.fg)
            parts = [ql, (bx0, y - ctx.size * 0.3, x + w + 1, y + ctx.size * 1.35 + 1)] + ([nq] if nq else [])
            v = rng.random()
            if v < 0.45:
                amt = f"{rng.uniform(0, 99999):,.2f}" if rng.random() < 0.6 else f"{rng.randint(0, 99999):,}"
                hq = B.draw_hw(pg, ctx, bx0 + bw * 0.3, y - ctx.size * 0.25, amt, ctx.size * 1.15, bound=(bx0 + 2, y - ctx.size * 0.9, x + w - 2, y + ctx.size * 1.6))
                if hq:
                    parts.append(hq)
            elif v < 0.75:
                amt = f"{rng.uniform(0, 99999):,.2f}"
                tq = pg.text(x + w - 8, y, amt, ctx.font("mono", amt), ctx.size, ctx.fg, align="r")
                if tq:
                    parts.append(tq)
            pg.add("form_field", parts)
            y += ctx.size * 2.0
        else:
            if two_col:
                y1 = B.b_field(pg, ctx, x, y, w * 0.48, style=rng.choice(["box", "line", "comb"]), ymax=ymax)
                y = max(y1, B.b_field(pg, ctx, x + w * 0.52, y, w * 0.48, style=rng.choice(["box", "line"]), ymax=ymax))
            else:
                y = B.b_field(pg, ctx, x, y, w, style=rng.choice(["box", "line", "comb"]), ymax=ymax)
    # sign here
    y = max(y, H * 0.8)
    t = rng.choice(["Sign Here", "Signature", "Declaration"])
    q = pg.text(x, y, t, ctx.font("bold", t), int(ctx.size * 1.1), ctx.fg)
    if q:
        pg.add("section_header", [q])
        y = bbox([q])[3] + 6
    y = B.b_para(pg, ctx, x, y, w, n=2, size=ctx.size * 0.8, text="Under penalties of perjury, I declare that I have examined this form and, to the best of my knowledge and belief, it is true, correct, and complete. " + ctx.para(60), indent=0)
    sy = y + ctx.size * 2.6
    if sy < H - 100:
        d.line([x, sy, x + w * 0.5, sy], fill=(0, 0, 0), width=1)
        d.line([x + w * 0.6, sy, x + w, sy], fill=(0, 0, 0), width=1)
        lq = pg.text(x, sy + 4, "Your signature", ctx.font("body", "Your signature"), int(ctx.size * 0.7), ctx.fg)
        dq = pg.text(x + w * 0.6, sy + 4, "Date", ctx.font("body", "Date"), int(ctx.size * 0.7), ctx.fg)
        sig = B.make_signature(ctx)
        k = min(w * 0.4 / sig.width, ctx.size * 2.6 / sig.height, 1.0)
        sig = sig.resize((max(6, int(sig.width * k)), max(6, int(sig.height * k))))
        if rng.random() < 0.85:
            pg.ink(sig, (x + 20, sy - sig.height + 6), "signature")
        pg.add("form_field", [lq, (x, sy - ctx.size * 2.4, x + w * 0.5, sy + 1)])
        parts = [dq, (x + w * 0.6, sy - ctx.size * 2, x + w, sy + 1)]
        if rng.random() < 0.8:
            hq = B.draw_hw(pg, ctx, x + w * 0.62, sy - ctx.size * 1.6, fake.date(pattern="%m/%d/%Y"), ctx.size * 1.2, bound=(x + w * 0.6, sy - ctx.size * 3, x + w, sy + 8))
            if hq:
                parts.append(hq)
        pg.add("form_field", parts)
    ft = f"{fid} ({yr})   Cat. No. {rng.randint(10000, 99999)}{rng.choice('ABCDEFGHJK')}"
    B.b_footer(pg, ctx, x, H - m * 0.9, w, page_no=rng.random() < 0.5, text=False)
    q = pg.text(x, H - m * 0.9 - ctx.size * 1.6, ft, ctx.font("body", ft), int(ctx.size * 0.75), ctx.fg)
    if q:
        pg.add("footer", [q])
    return pg, ctx


# =========================================================================== resume
def resume(rng, fake):
    pg = new_page(rng, "letter", 0.05)
    ctx = Ctx(rng, "en", fake, style=rng.choice(["sans", "serif", "cond"]))
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.06, 0.09))
    side = rng.random() < 0.4
    d = pg.d
    sx, sw = m, 0
    if side:
        sw = int(W * rng.uniform(0.28, 0.34))
        d.rectangle([0, 0, sw, H], fill=rng.choice([(35, 50, 75), (230, 236, 242), (240, 235, 225), (45, 45, 45)]))
        dark = pg.img.getpixel((5, 5))[0] < 128
    x, w = (sw + m * 0.8, W - sw - m * 1.8) if side else (m, W - 2 * m)
    y = m
    photo = rng.random() < 0.35
    name = fake.name()
    if side:
        fg_side = (240, 240, 240) if dark else ctx.fg
        sy = m
        if photo:
            pw = int(sw * 0.6)
            B.b_photo(pg, ctx, (sw - pw) / 2, sy, pw, int(pw * 1.2), frame=False)
            sy += int(pw * 1.2) + 30
        for head, items in [("Contact", [fake.phone_number(), fake.free_email(), fake.city() + ", " + rng.choice(US_STATES)]),
                            ("Skills", [fake.job()[:22] for _ in range(rng.randint(3, 5))]), ("Languages", rng.sample(["English", "Spanish", "French", "Khmer", "German", "Mandarin"], 2))]:
            q = pg.text(m * 0.7, sy, head.upper(), ctx.font("bold", head.upper()), int(ctx.size * 1.05), fg_side)
            if not q:
                break
            pg.add("section_header", [q])
            sy = bbox([q])[3] + 12
            qs = []
            for it in items:
                qq = pg.text(m * 0.7, sy, it, ctx.font("body", it), int(ctx.size * 0.85), fg_side, bound=(0, 0, sw - 8, H - 20))
                if qq:
                    qs.append(qq)
                    sy = bbox([qq])[3] + 10
            if qs:
                pg.add("list" if head != "Contact" else "text", qs)
            sy += 24
    else:
        if photo:
            pw = int(ctx.size * 5)
            B.b_photo(pg, ctx, x + w - pw, y, pw, int(pw * 1.25))
    y = B.b_heading(pg, ctx, x, y, w - (ctx.size * 6 if photo and not side else 0), cls="title", text=name, size=ctx.size * rng.uniform(2, 2.8), role="title", align="l")
    t = fake.job()
    y = addr_block(pg, ctx, x, y - 6, [t] + ([f"{fake.phone_number()} · {fake.free_email()} · {fake.city()}, {rng.choice(US_STATES)}"] if not side else []), size=ctx.size * 0.9)
    if photo and not side:
        y = max(y, m + ctx.size * 6.5)
    d.line([x, y, x + w, y], fill=ctx.accent, width=2)
    y += ctx.size
    for sec in rng.sample(["Summary", "Experience", "Education", "Projects", "Certifications", "Volunteer Work"], rng.randint(3, 5)):
        if y > H - m - ctx.size * 4:
            break
        y = B.b_heading(pg, ctx, x, y, w, cls="section_header", text=sec.upper() if rng.random() < 0.5 else sec, size=ctx.size * 1.2, fill=ctx.accent)
        if sec == "Summary":
            y = B.b_para(pg, ctx, x, y, w, H - m, n=rng.randint(2, 4))
            continue
        for _ in range(rng.randint(1, 3)):
            if y > H - m - ctx.size * 4:
                break
            role = f"{fake.job()[:34]}, {fake.company()[:26]}" if sec != "Education" else f"{rng.choice(['B.S.', 'B.A.', 'M.S.', 'MBA', 'Ph.D.'])} {rng.choice(['Computer Science', 'Economics', 'Biology', 'Accounting', 'Nursing', 'History'])}, {fake.city()} {rng.choice(['University', 'College', 'Institute of Technology'])}"
            dates = f"{rng.randint(2005, 2020)} – {rng.choice(['Present', str(rng.randint(2021, 2026))])}"
            q1 = pg.text(x, y, role, ctx.font("bold", role), ctx.size, ctx.fg, bound=(0, 0, x + w * 0.76, H))
            q2 = pg.text(x + w, y, dates, ctx.font("italic", dates), int(ctx.size * 0.9), ctx.fg, align="r")
            qs = [q for q in (q1, q2) if q]
            if qs:
                pg.add("text", qs)
                y = bbox(qs)[3] + ctx.size * 0.4
            y = B.b_list(pg, ctx, x + ctx.size * 0.5, y, w - ctx.size * 0.5, H - m, n=rng.randint(2, 4), style=rng.choice(["bullet", "dash", "square"]),
                         items=[R_sent(ctx) for _ in range(4)])
    if rng.random() < 0.3:
        B.b_footer(pg, ctx, x, H - m * 0.6, w, page_no=True, text=False)
    return pg, ctx


def R_sent(ctx):
    fk, rng = ctx.fake, ctx.rng
    return rng.choice([f"{rng.choice(['Led', 'Built', 'Managed', 'Designed', 'Improved', 'Reduced', 'Coordinated'])} {fk.bs()} for {rng.randint(3, 400)} clients",
                       f"{rng.choice(['Increased', 'Cut', 'Grew'])} {rng.choice(['revenue', 'costs', 'response time', 'retention'])} by {rng.randint(5, 80)}%",
                       fk.catch_phrase() + " initiative across " + rng.choice(["three regions", "the team", "two product lines"]),
                       ctx.phrase(25, 90)])


# =========================================================================== US bank statement
def us_bank_statement(rng, fake):
    pg = new_page(rng, "letter", 0.04)
    ctx = Ctx(rng, "en", fake)
    nm = rng.choice(["First", "Union", "Pacific", "Heritage", "Summit", "Liberty", "Cedar"]) + " " + rng.choice(["National Bank", "Savings Bank", "Credit Union", "Trust", "Bank & Trust"])
    return _bank_statement(pg, ctx, nm, "Member FDIC" if rng.random() < 0.5 else fake.phone_number())


# =========================================================================== academic paper
def academic_paper(rng, fake):
    pg = new_page(rng, "letter", 0.0)
    ctx = Ctx(rng, "en", fake, style=rng.choice(["serif", "serif", "sans"]))
    ctx.size = rng.choice([17, 18, 19, 20])
    ctx.justify = rng.random() < 0.8
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.07, 0.1))
    x, w = m, W - 2 * m
    first = rng.random() < 0.45
    cols = rng.choice([1, 2, 2])
    if not first and rng.random() < 0.7:
        B.b_header(pg, ctx, x, m * 0.45, w, text=rng.choice([f"{fake.last_name()} et al.", ctx.heading()[:60], f"Proceedings of {fake.company()} {rng.randint(2015, 2026)}"]))
    y = m
    if first:
        y = B.b_heading(pg, ctx, x + w * 0.05, y, w * 0.9, cls="title", text=ctx.heading() + (": " + ctx.phrase(10, 40) if rng.random() < 0.5 else ""),
                        size=ctx.size * rng.uniform(1.6, 2.1), align="c", max_lines=3)
        authors = ", ".join(fake.name() + (str(i + 1) if rng.random() < 0.0 else "") for i in range(rng.randint(2, 5)))
        y = B.b_para(pg, ctx, x, y, w, n=2, text=authors, align="c", indent=0)
        y = B.b_para(pg, ctx, x, y - 6, w, n=2, size=ctx.size * 0.85, text=f"{fake.city()} {rng.choice(['University', 'Institute of Technology'])}, {fake.city()}, {rng.choice(US_STATES)}  {fake.free_email()}", align="c", indent=0)
        y = B.b_heading(pg, ctx, x + w * 0.1, y, w * 0.8, text="Abstract", size=ctx.size * 1.05, align="c")
        y = B.b_para(pg, ctx, x + w * 0.08, y, w * 0.84, n=rng.randint(5, 9), size=ctx.size * 0.92, indent=0)
    fn_top = H - m * 1.05
    if rng.random() < 0.4:
        fn_top = B.b_footnotes(pg, ctx, x, H - m * 1.05, w if cols == 1 else (w - 30) / 2)
    kinds = [("para", 5), ("heading", 1.5), ("formula", 1.5), ("table", 0.8), ("figure", 1.0), ("list", 0.5)]
    if cols == 1:
        B.fill_flow(pg, ctx, x, y, w, fn_top, kinds)
    else:
        gap = int(W * 0.03)
        cw = (w - gap) // 2
        for c in range(2):
            B.fill_flow(pg, ctx, x + c * (cw + gap), y, cw, fn_top if c == 0 else H - m * 1.05, kinds)
    B.b_footer(pg, ctx, x, H - m * 0.6, w, page_no=rng.random() < 0.9, text=rng.random() < 0.2)
    return pg, ctx


# =========================================================================== menu / flyer
def menu_flyer(rng, fake):
    if rng.random() < 0.55:
        return _menu(rng, fake)
    return _flyer(rng, fake)


def _menu(rng, fake):
    bil = rng.random() < 0.35
    ctx = Ctx(rng, "km" if bil else "en", fake, style=rng.choice(["serif", "sans", "cond"]))
    pg = new_page(rng, "letter", 0.0)
    W, H = pg.W, pg.H
    bg = rng.choice([(250, 246, 235), (30, 30, 35), (245, 240, 230), (255, 255, 255), (235, 245, 240), (60, 20, 25)])
    pg.d.rectangle([0, 0, W, H], fill=bg)
    dark = sum(bg) < 300
    ctx.fg = (240, 235, 220) if dark else ctx.fg
    ctx.accent = rng.choice([(230, 180, 60), (240, 120, 90)]) if dark else ctx.accent
    m = int(W * rng.uniform(0.07, 0.1))
    x, w = m, W - 2 * m
    y = m * 0.6
    ls = int(ctx.size * rng.uniform(3, 4.5))
    B.b_logo(pg, ctx, W / 2 - ls / 2, y, ls, kind="corp")
    y += ls + 12
    nm = rng.choice(["The " + fake.last_name() + " Kitchen", fake.first_name() + "'s Bistro", "Cafe " + fake.city(), "Mekong Noodle House", "Angkor Grill"])
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=nm, size=ctx.size * rng.uniform(2.2, 3.2), align="c", role="display", fill=ctx.accent)
    cols = rng.choice([1, 2])
    gap = int(W * 0.05)
    cw = (w - gap * (cols - 1)) // cols
    sections = rng.sample(["Starters", "Soups", "Salads", "Mains", "Noodles & Rice", "Desserts", "Drinks", "Coffee & Tea", "Specials"], rng.randint(3, 6))
    col_y = [y] * cols
    for i, s in enumerate(sections):
        c = min(range(cols), key=lambda k: col_y[k])
        cx = x + c * (cw + gap)
        yy = col_y[c]
        if yy > H - m - ctx.size * 6:
            continue
        head = s if not bil else rng.choice(["មុខម្ហូប", "ភេសជ្ជៈ", "បង្អែម", "អាហារពេលព្រឹក", "ម្ហូបពិសេស"]) + " / " + s
        yy = B.b_heading(pg, ctx, cx, yy, cw, text=head, size=ctx.size * 1.3, fill=ctx.accent, align="l")
        qs = []
        for _ in range(rng.randint(3, 6)):
            if yy > H - m - ctx.size * 2:
                break
            item = (rng.choice(KH_PRODUCTS_EN) if rng.random() < 0.5 else fake.word().title() + " " + rng.choice(["Bowl", "Plate", "Wrap", "Curry", "Latte", "Tart", "Soup"]))
            if bil:
                from .ctx import KH_PRODUCTS
                item = rng.choice(KH_PRODUCTS) + " " + item
            price = f"${rng.uniform(2, 28):.2f}" if rng.random() < 0.7 else f"{rng.uniform(2, 28):.1f}"
            q1 = pg.text(cx, yy, item, ctx.font("bold" if rng.random() < 0.5 else "body", item), ctx.size, ctx.fg, bound=(cx, 0, cx + cw * 0.78, H))
            q2 = pg.text(cx + cw, yy, price, ctx.font("body", price), ctx.size, ctx.fg, align="r")
            qs += [q for q in (q1, q2) if q]
            yy += ctx.size * 1.6
            if rng.random() < 0.4:
                ds = ctx.phrase(15, 50) if not bil else fake.sentence(nb_words=6)[:50]
                q3 = pg.text(cx, yy, ds, ctx.font("italic", ds), int(ctx.size * 0.8), ctx.fg, bound=(cx, 0, cx + cw, H))
                if q3:
                    qs.append(q3)
                yy += ctx.size * 1.4
        if qs:
            pg.add("list", qs)
        col_y[c] = yy + ctx.size
    if rng.random() < 0.5:
        s = rng.randint(110, 170)
        B.b_qr(pg, ctx, x + w - s, H - m * 0.5 - s, s)
    t = f"{fake.street_address()} · {fake.phone_number()}"
    q = pg.text(x, H - m * 0.6, t, ctx.font("body", t), int(ctx.size * 0.8), ctx.fg)
    if q:
        pg.add("footer", [q])
    return pg, ctx


def _flyer(rng, fake):
    ctx = Ctx(rng, "en" if rng.random() < 0.7 else "km", fake)
    pg = new_page(rng, rng.choice(["letter", "a4"]), 0.0)
    W, H = pg.W, pg.H
    bg = rng.choice([(240, 240, 235), (20, 40, 90), (180, 30, 30), (250, 220, 60), (30, 110, 70), (255, 255, 255), (240, 120, 40)])
    pg.d.rectangle([0, 0, W, H], fill=bg)
    dark = sum(bg) < 400
    ctx.fg = (255, 255, 255) if dark else (20, 20, 20)
    m = int(W * 0.07)
    x, w = m, W - 2 * m
    y = m
    if rng.random() < 0.5:
        fh = int(H * rng.uniform(0.25, 0.4))
        from . import assets2 as X
        pg.paste(X.make_scene(rng, W, fh), (0, 0), "figure")
        y = fh + m * 0.5
    t = ctx.phrase(10, 32).upper() if ctx.lang == "en" else ctx.phrase(8, 26)
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=t, size=rng.randint(70, 120), align="c", role="display", fill=ctx.fg)
    y = B.b_heading(pg, ctx, x, y, w, text=ctx.phrase(15, 60), size=rng.randint(34, 50), align="c", fill=ctx.fg)
    y = B.b_para(pg, ctx, x + w * 0.08, y, w * 0.84, n=rng.randint(2, 5), size=ctx.size * 1.2, fill=ctx.fg, align="c", indent=0)
    if rng.random() < 0.6:
        items = [ctx.phrase(10, 40) for _ in range(rng.randint(3, 5))]
        y = B.b_list(pg, ctx, x + w * 0.15, y, w * 0.7, H - m * 3, items=items, size=ctx.size * 1.15)
    if rng.random() < 0.5:
        y = B.b_figure(pg, ctx, x, y, w, H - m * 3, kind=rng.choice(["scene", "chart"]), caption=False)
    yb = H - m - 180
    s = rng.randint(130, 190)
    if rng.random() < 0.75:
        B.b_qr(pg, ctx, x + w - s, yb, s)
    B.b_logo(pg, ctx, x, yb + 20, rng.randint(80, 130), kind="corp")
    t = f"{fake.street_address()}, {fake.city()} · {fake.phone_number()}"
    q = pg.text(x, H - m * 0.7, t, ctx.font("body", t), int(ctx.size * 0.9), ctx.fg, bound=(0, 0, W - 4, H))
    if q:
        pg.add("footer", [q])
    return pg, ctx


# =========================================================================== business card
def business_card(rng, fake):
    km = rng.random() < 0.3
    ctx = Ctx(rng, "km" if km else "en", fake, style=rng.choice(["sans", "serif", "cond"]))
    cw, ch = 1050, 600
    many = rng.random() < 0.3
    if many:  # several cards scanned on one sheet
        pg = new_page(rng, "a4", 0.0)
        s = 0.55
        pos = [(int(pg.W * 0.08 + i % 2 * pg.W * 0.45), int(pg.H * 0.06 + i // 2 * (ch * s + 60))) for i in range(rng.randint(2, 6))]
        for (px, py) in pos:
            _bcard(pg, ctx, px, py, int(cw * s), int(ch * s))
            ctx.fake.seed_instance(rng.getrandbits(32))
    else:
        mx, my = rng.randint(30, 140), rng.randint(30, 140)
        pg = Page(cw + 2 * mx, ch + 2 * my, rng.choice([(70, 70, 75), (150, 120, 90), (230, 230, 230), (40, 50, 60), (200, 190, 170), (250, 250, 250)]))
        _bcard(pg, ctx, mx, my, cw, ch)
    return pg, ctx


def _bcard(pg, ctx, x, y, cw, ch):
    rng, fk = ctx.rng, ctx.fake
    d = pg.d
    bg = rng.choice([(255, 255, 255), (250, 248, 242), (30, 35, 45), (235, 240, 245), (20, 60, 100)])
    d.rectangle([x, y, x + cw, y + ch], fill=bg, outline=(180, 180, 180))
    dark = sum(bg) < 300
    fg = (245, 245, 245) if dark else (25, 25, 25)
    acc = rng.choice([(200, 150, 40), (220, 60, 60), (40, 140, 200)]) if dark else ctx.accent
    if rng.random() < 0.5:
        d.rectangle([x, y + ch * 0.86, x + cw, y + ch], fill=acc)
    sz = int(ch * 0.05)
    bnd = (x + 4, y + 4, x + cw - 4, y + ch - 4)
    ls = int(ch * rng.uniform(0.18, 0.28))
    B.b_logo(pg, ctx, x + ch * 0.08, y + ch * 0.08, ls, kind="corp", wordmark=fk.company().split(",")[0][:22] if rng.random() < 0.5 else None)
    if ctx.lang == "km":
        nm, (lat, _) = ctx.kh_name()
        names = [nm, lat.title()]
    else:
        names = [fk.name()]
    yy = y + ch * 0.42
    nq = []
    for i, t in enumerate(names):
        q = pg.text(x + ch * 0.08, yy, t, ctx.font("bold" if i == 0 else "latin_bold", t), int(sz * (1.6 if i == 0 else 1.2)), fg, bound=bnd)
        if q:
            nq.append(q)
            yy = bbox([q])[3] + sz * 0.4
    if nq:
        pg.add("title", nq)
    jt = fk.job()[:36]
    q = pg.text(x + ch * 0.08, yy, jt, ctx.font("italic" if ctx.lang == "en" else "body", jt), sz, acc, bound=bnd)
    if q:
        pg.add("text", [q])
    rx = x + cw * 0.55
    ry = y + ch * 0.42
    cq = []
    for t in [ctx.phone(), fk.company_email(), "www." + fk.domain_name(), (ctx.address()[-1] if ctx.lang == "km" else fk.city() + ", " + rng.choice(US_STATES))]:
        q = pg.text(rx, ry, t, ctx.font("body", t), int(sz * 0.85), fg, bound=bnd)
        if q:
            cq.append(q)
            ry = bbox([q])[3] + sz * 0.45
    if cq:
        pg.add("text", cq)
    if rng.random() < 0.4:
        s = int(ch * 0.26)
        B.b_qr(pg, ctx, x + cw - s - ch * 0.06, y + ch * 0.06, s)


# =========================================================================== shipping label
def shipping_label(rng, fake):
    ctx = Ctx(rng, "en", fake, style=rng.choice(["sans", "cond", "mono_doc"]))
    W, H = 1200, 1800
    pg = Page(W, H, rng.choice([(255, 255, 255), (252, 252, 248)]))
    d = pg.d
    m = 30
    x, w = m, W - 2 * m
    d.rectangle([m - 10, m - 10, W - m + 10, H - m + 10], outline=(0, 0, 0), width=4)
    sz = rng.choice([26, 28, 30])
    ctx.size = sz
    y = m
    carrier = rng.choice(["PRIORITY", "GROUND", "EXPRESS", "STANDARD POST", "2-DAY AIR", "FREIGHT"])
    big = rng.choice(["P", "G", "E", "1", "2"])
    qb = pg.text(x + 10, y, big, ctx.font("title", big), 160, (0, 0, 0))
    if qb:
        pg.add("text", [qb])
    q = pg.text(x + 200, y + 30, carrier, ctx.font("bold", carrier), int(sz * 2), (0, 0, 0), bound=(0, 0, x + w - 10, H))
    if q:
        pg.add("title", [q])
    y += 190
    d.line([m - 10, y, W - m + 10, y], fill=(0, 0, 0), width=4)
    y += 16
    fq = pg.text(x, y, "FROM:", ctx.font("bold", "FROM:"), int(sz * 0.8), (0, 0, 0))
    if fq:
        pg.add("text", [fq])
    y = addr_block(pg, ctx, x + 130, y, [fake.name().upper(), fake.street_address().upper(), f"{fake.city().upper()} {rng.choice(US_STATES)} {fake.zipcode()}"], size=sz * 0.8)
    y += 10
    tq = pg.text(x, y, "SHIP TO:", ctx.font("bold", "SHIP TO:"), sz, (0, 0, 0))
    if tq:
        pg.add("text", [tq])
    y = addr_block(pg, ctx, x + 60, y + sz * 1.6, [fake.name().upper(), fake.company().upper()[:30], fake.street_address().upper(), f"{fake.city().upper()} {rng.choice(US_STATES)} {fake.zipcode()}"], size=sz * 1.25)
    d.line([m - 10, y, W - m + 10, y], fill=(0, 0, 0), width=4)
    y += 20
    s = 230
    from . import assets2 as X
    if rng.random() < 0.5:
        B.b_qr(pg, ctx, x, y, s)
    else:
        pg.paste(X.make_pdf417ish(rng, s, s), (x, y), "qr_code")
    meta = [f"WT: {rng.uniform(0.2, 40):.1f} LBS", f"SHIP DATE: {fake.date(pattern='%d%b%y').upper()}", f"REF: {rng.randint(100000, 999999)}", f"DIMS: {rng.randint(4, 30)}x{rng.randint(4, 30)}x{rng.randint(2, 20)} IN"]
    yy = y
    mq = []
    for t in meta:
        q = pg.text(x + s + 40, yy, t, ctx.font("body", t), int(sz * 0.9), (0, 0, 0))
        if q:
            mq.append(q)
            yy = bbox([q])[3] + sz * 0.5
    if mq:
        pg.add("text", mq)
    y += s + 30
    d.line([m - 10, y, W - m + 10, y], fill=(0, 0, 0), width=4)
    y += 20
    t = "TRACKING #"
    q = pg.text(W / 2, y, t, ctx.font("bold", t), sz, (0, 0, 0), align="c")
    if q:
        pg.add("text", [q])
    y += sz * 1.8
    trk = "".join(rng.choice("0123456789") for _ in range(rng.choice([12, 18, 22])))
    y = B.b_barcode(pg, ctx, x + 40, y, w - 80, rng.randint(200, 280), code=trk[:20])
    y += 30
    if y < H - 300:
        d.line([m - 10, y, W - m + 10, y], fill=(0, 0, 0), width=4)
        y += 20
        B.b_barcode(pg, ctx, x + 200, y, w - 400, 120)
    return pg, ctx


def en_textbook(rng, fake):
    return textbook(rng, fake, "en")


def en_newspaper(rng, fake):
    return newspaper(rng, fake, "en")


EN_TEMPLATES = {
    "us_letter": (us_letter, 1.1),
    "us_invoice": (us_invoice, 1.2),
    "us_tax_form": (us_tax_form, 1.1),
    "resume": (resume, 0.8),
    "us_bank_statement": (us_bank_statement, 0.7),
    "academic_paper": (academic_paper, 1.0),
    "menu_flyer": (menu_flyer, 0.8),
    "business_card": (business_card, 0.5),
    "shipping_label": (shipping_label, 0.5),
    "en_textbook": (en_textbook, 0.7),
    "en_newspaper": (en_newspaper, 0.5),
}
