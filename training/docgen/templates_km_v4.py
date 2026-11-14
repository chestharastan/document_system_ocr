"""v4 Cambodian official notices / letters, modelled on real university and ministry pages: emblem + national motto
letterhead, institution block with a HANDWRITTEN reference number, date line with handwritten day/month, Moul titles,
"យោង ៖" paragraphs, label : value rows with counts at the right, bordered tables with spanning group rows, a signature
block whose red stamp overlaps the signature AND the printed role / name / date, handwritten approval notes, footer with
address / e-mail / QR code. All content synthetic. Classes are the 22 of layout_v3; nothing new, only better coverage
of: logo vs header text vs decoration, stamp over text, handwriting inside printed lines, tables with group rows."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

from . import blocks as B
from .canvas import bbox
from .ctx import MINISTRIES, SUB_UNITS, Ctx
from .res import font
from .templates_km import KINGDOM, PR_ADDR, PR_SECTIONS, new_page, motto, date_line, footer_row

UNIS = [("សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ", "ROYAL UNIVERSITY OF PHNOM PENH"), ("សាកលវិទ្យាល័យភូមិន្ទកសិកម្ម", "ROYAL UNIVERSITY OF AGRICULTURE"),
        ("វិទ្យាស្ថានបច្ចេកវិទ្យាកម្ពុជា", "INSTITUTE OF TECHNOLOGY OF CAMBODIA"), ("សាកលវិទ្យាល័យជាតិគ្រប់គ្រង", "NATIONAL UNIVERSITY OF MANAGEMENT"),
        ("វិទ្យាស្ថានជាតិអប់រំ", "NATIONAL INSTITUTE OF EDUCATION"), ("សាកលវិទ្យាល័យបាត់ដំបង", "UNIVERSITY OF BATTAMBANG"),
        ("សាកលវិទ្យាល័យស្វាយរៀង", "SVAY RIENG UNIVERSITY"), ("សាលាជាតិរដ្ឋបាល", "ROYAL SCHOOL OF ADMINISTRATION")]
UNI_SUB = ["មហាវិទ្យាល័យវិទ្យាសាស្ត្រ", "មហាវិទ្យាល័យវិស្វកម្ម", "មហាវិទ្យាល័យសង្គមសាស្ត្រ និងមនុស្សសាស្ត្រ", "បណ្ណាល័យ", "ការិយាល័យសិក្សា", "នាយកដ្ឋានគណិតវិទ្យា",
           "វិទ្យាស្ថានភាសាបរទេស", "មហាវិទ្យាល័យអប់រំ"]
NOTICE_TITLES = ["សេចក្តីជូនដំណឹង", "សេចក្តីជូនដំណឹង", "សេចក្តីប្រកាស", "លិខិតអញ្ជើញ", "សេចក្តីណែនាំ", "សេចក្តីសម្រេច"]
NOTICE_SUBJECTS = ["ស្តីពីការប្រឡងជ្រើសរើសនិស្សិតចូលរៀនថ្នាក់បរិញ្ញាបត្រជាន់ខ្ពស់ផ្នែកគណិតវិទ្យា ជំនាន់ទី១៥ (សៅរ៍-អាទិត្យ)",
                   "ស្តីពីការប្រឡងប្រជែងជ្រើសរើសនិស្សិតឱ្យចូលរៀនថ្នាក់បរិញ្ញាបត្រ ឆ្នាំសិក្សាមូលដ្ឋាន (ឆ្នាំទី១)",
                   "ស្តីពីការជ្រើសរើសបុគ្គលិកអនាម័យនិងបរិស្ថាន សម្រាប់បម្រើការងារក្រៅម៉ោង (Part time job)",
                   "ស្តីពីការបើកវគ្គបណ្តុះបណ្តាលគ្រូបង្រៀនភាសាអង់គ្លេស ឆ្នាំសិក្សា ២០២៦-២០២៧",
                   "ស្តីពីការប្រឡងសញ្ញាបត្រមធ្យមសិក្សាទុតិយភូមិ សម័យប្រឡង ថ្ងៃទី២១ ខែសីហា ឆ្នាំ២០២៦",
                   "ស្តីពីកាលបរិច្ឆេទចុះឈ្មោះចូលរៀន និងការបង់ថ្លៃសិក្សា សម្រាប់ឆមាសទី១",
                   "ស្តីពីការផ្អាកដំណើរការបណ្ណាល័យជាបណ្តោះអាសន្ន ដើម្បីជួសជុលអគារ"]
NOTICE_PLACES = ["នៅសាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ ថ្នាក់សិក្សាចុងសប្តាហ៍សម្រាប់ឆ្នាំសិក្សា២០២៦-២០២៧", "សម្រាប់ឆ្នាំសិក្សា ២០២៦-២០២៧", "នៅមហាវិទ្យាល័យវិទ្យាសាស្ត្រ និងវិទ្យាស្ថានសិក្សាអន្តរជាតិ",
                 "នៅគ្រប់រាជធានី ខេត្ត ទូទាំងប្រទេស", "នៅបណ្ណាល័យ ហ៊ុន សែន សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ"]
ROLES = ["ច. សាកលវិទ្យាធិការ", "សាកលវិទ្យាធិការ", "សាកលវិទ្យាធិការរង", "នាយក", "ប្រធានមហាវិទ្យាល័យ", "ជ. រដ្ឋមន្ត្រី", "រដ្ឋលេខាធិការ", "អគ្គនាយក", "ប្រធាននាយកដ្ឋាន"]
LABELS = ["សំណុំលិខិត៖", "ទីកន្លែងទទួលពាក្យ៖", "សម័យប្រឡង៖", "វិញ្ញាសាប្រឡង៖", "ថ្លៃសិក្សា៖", "ទូរសព្ទទំនាក់ទំនង៖", "សារអេឡិចត្រូនិច៖", "កាលបរិច្ឆេទ៖", "លក្ខខណ្ឌ៖"]
LABEL_VALUES = ["- សញ្ញាបត្រថ្នាក់បរិញ្ញាបត្រផ្នែកគណិតវិទ្យា បញ្ជាក់ពីគ្រឹះស្ថានសាមី សាលាខណ្ឌ/ស្រុក", "- ប្រវត្តិរូបសង្ខេបមានបិទរូបថត (៤×៦) និងពាក្យសុំចុះឈ្មោះចូលរៀន",
                "- រូបថតថ្មី ថតចំពីមុខ ផ្ទៃពណ៌ស ទំហំ (៤×៦)", "- សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ (មណ្ឌល១) អគារស្តីម ជាន់ទី៥ បន្ទប់-៥០១", "- ថ្ងៃសៅរ៍ ១១រោច ខែភទ្របទ ឆ្នាំមមី អដ្ឋស័ក ព.ស. ២៥៧០ ត្រូវនឹងថ្ងៃទី៥ ខែធ្នូ ឆ្នាំ២០២៦",
                "- គណិតវិទ្យា (១៣:០០នាទី-១៦:០០នាទី) និងភាសាអង់គ្លេស (១៦:១៥នាទី-១៧:១៥នាទី)", "- បេក្ខជនត្រូវបង់សិទ្ធិប្រឡង ៤០,០០០រៀល", "- និស្សិតត្រូវបង់ថ្លៃសិក្សា ៤,០០០,០០០ រៀល/មួយឆ្នាំ (ឬអាចបង់ជាពីរដំណាក់កាល)",
                "- ០១២ ៨៧៨ ០៥៧ និង ០៩៦ ៥២២ ១២៦៨", "- secretary@rupp.edu.kh, info@rupp.edu.kh", "- មានសញ្ញាបត្រមធ្យមសិក្សាទុតិយភូមិ ឬសញ្ញាបត្រដែលមានតម្លៃស្មើ"]
COUNTS = ["ចំនួន០១ច្បាប់", "ចំនួន០២ច្បាប់", "ចំនួន០៣សន្លឹក", "ចំនួន០២សន្លឹក", "០១ច្បាប់", "ចំនួន ៤៨នាក់", "ចំនួន ១០នាក់", "ស្រី ១៤នាក់", "ស្រី ០៣នាក់"]
GROUPS = ["ក- មហាវិទ្យាល័យ សង្គមសាស្ត្រ - មនុស្សសាស្ត្រ", "ខ- មហាវិទ្យាល័យ វិទ្យាសាស្ត្រ", "គ- មហាវិទ្យាល័យ វិស្វកម្ម", "ឃ- វិទ្យាស្ថានភាសាបរទេស",
          "ក- វិទ្យាស្ថានសិក្សាអន្តរជាតិ", "ខ- មហាវិទ្យាល័យអប់រំ"]
MAJORS = ["អក្សរសាស្ត្រខ្មែរ", "ភូមិវិទ្យា", "ចិត្តវិទ្យា", "ទស្សនវិជ្ជា", "សង្គមវិទ្យា", "ប្រវត្តិវិទ្យា", "គណិតវិទ្យា", "រូបវិទ្យា", "គីមីវិទ្យា", "ជីវវិទ្យា", "វិទ្យាសាស្ត្រកុំព្យូទ័រ",
          "បរិស្ថានវិទ្យា", "វិស្វកម្មសំណង់", "វិស្វកម្មអគ្គិសនី", "ភាសាអង់គ្លេស", "ភាសាបារាំង", "ទេសចរណ៍", "ព័ត៌មានវិទ្យា"]
EXAMS = ["គណិតវិទ្យា", "អក្សរសាស្ត្រខ្មែរ", "រូបវិទ្យា", "គីមីវិទ្យា", "ជីវវិទ្យា", "ភាសាអង់គ្លេស", "ភូមិវិទ្យា", "ប្រវត្តិវិទ្យា", "សីលធម៌ពលរដ្ឋវិជ្ជា", "ចំណេះទូទៅ"]
HW_NOTES = ["បានឃើញ និងឯកភាព", "សូមអនុវត្ត", "យល់ព្រម", "សូមពិនិត្យ", "ឯកភាព", "អនុម័ត", "សូមចាត់ចែង", "ជម្រាបជូនលោកប្រធាន"]
KH_MONTHS = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា", "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]
LUNAR_M = ["ពិសាខ", "ជេស្ឋ", "អាសាឍ", "ស្រាពណ៍", "ភទ្របទ", "អស្សុជ", "កត្តិក", "មិគសិរ", "បុស្ស", "មាឃ", "ផល្គុន", "ចេត្រ"]
YEARS = ["មមី អដ្ឋស័ក", "វក នព្វស័ក", "រកា សំរឹទ្ធិស័ក", "ច ឯកស័ក", "កុរ ទោស័ក", "ជូត ត្រីស័ក"]


def _t(pg, ctx, x, y, text, role, size, fill=None, align="l", cls="text", bound=None):
    """One printed line recorded under `cls`. Returns (quad or None, y below)."""
    p = ctx.font(role, text)
    if not p:
        return None, y
    pg.begin(cls)
    q = pg.text(x, y, text, p, int(size), fill or ctx.fg, align=align, bound=bound)
    pg.end()
    if q:
        pg.add(cls, [q])
        return q, bbox([q])[3]
    return None, y


def hw_inline(pg, ctx, x, y, text, size, slot_w=None):
    """A handwritten value written on a printed line (register number, day of month)."""
    rng = ctx.rng
    q = B.draw_hw(pg, ctx, x + rng.uniform(0, 6), y - size * rng.uniform(0.05, 0.3), text, size * rng.uniform(1.0, 1.3) if slot_w is None else size * 1.05)
    return bbox([q])[2] if q else x + (slot_w or size * 3)


def hw_number(ctx, lo=1, hi=2999):
    rng = ctx.rng
    n = str(rng.randint(lo, hi))
    return ctx.d(n) if rng.random() < 0.6 else n


def letterhead(pg, ctx, x, y, w, org):
    """Emblem left (sometimes centre), national motto centred, ornament under it. -> y below"""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(1.0, 1.2))
    emb = int(sz * rng.uniform(3.4, 5.2))
    layout = rng.random()
    if layout < 0.55:   # emblem at the left, motto centred on the page (RUPP / MoEYS)
        B.b_logo(pg, ctx, x + rng.uniform(0, w * 0.06), y + rng.uniform(0, sz), emb, kind="emblem")
        ym = motto(pg, ctx, x + w * rng.uniform(0.5, 0.56), y, size=sz * rng.uniform(1.0, 1.15))
        yb = max(ym, y + emb + sz * 0.6)
    elif layout < 0.8:  # emblem centred above the motto
        B.b_logo(pg, ctx, x + w / 2 - emb / 2, y, emb, kind="emblem")
        ym = motto(pg, ctx, x + w / 2, y + emb + sz * 0.5, size=sz * rng.uniform(1.0, 1.15))
        yb = ym
    else:               # motto at the right, emblem + name at the left
        B.b_logo(pg, ctx, x + rng.uniform(0, w * 0.08), y, emb, kind="emblem")
        ym = motto(pg, ctx, x + w * rng.uniform(0.7, 0.78), y + sz * rng.uniform(0, 0.5), size=sz * rng.uniform(1.0, 1.15))
        yb = max(ym, y + emb + sz * 0.4)
    return yb


def org_lines(pg, ctx, x, y, w, org, hw_ref=True):
    """Institution name(s) in Moul at the left + "លេខ៖" with a handwritten number (+ unit code). -> y below"""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.95, 1.1))
    names = [org[0]] + ([rng.choice(UNI_SUB if org in UNIS else SUB_UNITS)] if rng.random() < 0.65 else [])
    qs, yy = [], y
    for i, t in enumerate(names):
        p = ctx.font("moul" if i == 0 or rng.random() < 0.6 else "bold", t)
        if not p:
            continue
        pg.begin("header")
        q = B.fit_text(pg, x, yy, t, p, int(sz * (1.0 if i == 0 else 0.9)), w * 0.5, ctx.fg, align="l")
        pg.end()
        if q:
            qs.append(q)
            yy = bbox([q])[3] + sz * 0.45
    if qs:
        pg.add("header", qs)
    # reference line: printed "លេខ៖" + handwritten number + printed/handwritten unit code
    lab = "លេខ" + rng.choice(["៖ ", ": ", "ៈ "])
    q, _ = _t(pg, ctx, x, yy, lab, "body", sz * 0.95)
    xx = bbox([q])[2] + sz * 0.3 if q else x + sz * 3
    if hw_ref:
        xx = hw_inline(pg, ctx, xx, yy, hw_number(ctx), sz * 0.95) + sz * 0.4
        code = rng.choice(["អយក.សអអ", "សជណ", "ស.ក.ប.ស", "រលក", "កបទ.សអ", "ភព.សអ", "ន.ស.ក"])
        if rng.random() < 0.5:
            hw_inline(pg, ctx, xx, yy, code, sz * 0.9)
        else:
            _t(pg, ctx, xx, yy, code, "body", sz * 0.9)
    else:
        _t(pg, ctx, xx, yy, f"{hw_number(ctx)} {rng.choice(['អយក.សអអ', 'សជណ', 'រលក'])}", "body", sz * 0.95)
    return yy + sz * 1.9


def hw_date_line(pg, ctx, x, y, w, align="r", lunar=None):
    """Date line(s) with the day / month written by hand into printed gaps (as on signed letters). -> y below"""
    rng = ctx.rng
    sz = int(ctx.size * 0.95)
    lunar = rng.random() < 0.55 if lunar is None else lunar
    yy = y
    parts_list = []
    if lunar:
        parts_list.append([("ថ្ងៃ", "p"), (rng.choice(["ច័ន្ទ", "អង្គារ", "ពុធ", "ព្រហស្បតិ៍", "សុក្រ", "សៅរ៍"]), "h"), (f" {ctx.d(rng.randint(1, 15))}{rng.choice(['កើត', 'រោច'])}", "h"),
                           (" ខែ", "p"), (rng.choice(LUNAR_M), "h"), (f" ឆ្នាំ{rng.choice(YEARS)} ព.ស.២៥៧០", "p")])
    parts_list.append([(rng.choice(["រាជធានីភ្នំពេញ", "ខេត្តសៀមរាប", "ខេត្តបាត់ដំបង"]) + " ថ្ងៃទី", "p"), (ctx.d(rng.randint(1, 28)), "h"), (" ខែ", "p"), (rng.choice(KH_MONTHS), "h"),
                       (f" ឆ្នាំ{ctx.d(rng.choice([2025, 2026, 2027]))}", "p")])
    for parts in parts_list:
        # printed parts are one text line; handwritten parts are placed in the gaps left for them
        pieces, cur = [], ""
        for t, k in parts:
            if k == "p":
                cur += t
            else:
                cur += " " * max(4, int(len(t) * 2.0) + 4)
                pieces.append((len(cur), t))
        p = ctx.font("body", cur)
        if not p:
            continue
        f = font(p, sz)
        total_w = f.getlength(cur)
        x0 = x + w - total_w if align == "r" else (x + w / 2 - total_w / 2 if align == "c" else x)
        pg.begin("text")
        q = pg.text(x0, yy, cur, p, sz, ctx.fg)
        pg.end()
        if q:
            pg.add("text", [q])
            for pos, t in pieces:
                gap_w = f.getlength(" " * max(4, int(len(t) * 2.0) + 4)); gap_x = x0 + f.getlength(cur[:pos]) - gap_w + sz * 0.15
                hw_inline(pg, ctx, gap_x, yy, t, sz * 0.8, slot_w=gap_w)
            yy = bbox([q])[3] + sz * 0.55
    return yy + sz * 0.3


def label_value_rows(pg, ctx, x, y, w, ymax, n=None):
    """Bold label at the left, dash items in the middle, counts at the right (RUPP notice style). -> y below"""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.9, 1.0))
    labels = rng.sample(LABELS, n or rng.randint(3, 6))
    lw = w * rng.uniform(0.17, 0.22)
    yy = y
    label_qs, item_qs, count_qs = [], [], []
    for lab in labels:
        if yy > ymax - sz * 2:
            break
        k = rng.randint(1, 3) if lab in LABELS[:2] else 1
        q, _ = _t(pg, ctx, x, yy, lab, "moul" if rng.random() < 0.5 else "bold", sz * 0.95)
        if q:
            label_qs.append(q)
        for i in range(k):
            if yy > ymax - sz * 2:
                break
            val = rng.choice(LABEL_VALUES)
            p = ctx.font("body", val)
            if not p:
                continue
            f = font(p, sz)
            with_count = rng.random() < 0.55
            maxw = w - lw - (w * 0.22 if with_count else sz)
            s = min(1.0, maxw / max(1, f.getlength(val)))
            pg.begin("text")
            q = pg.text(x + lw, yy, val, p, int(sz * max(0.75, s)), ctx.fg, bound=(x + lw - 2, yy - sz, x + lw + maxw, yy + sz * 2.2))
            pg.end()
            if q:
                item_qs.append(q)
                if with_count:
                    c = rng.choice(COUNTS)
                    pg.begin("text")
                    q2 = pg.text(x + w, yy, c, ctx.font("body", c), int(sz * 0.95), ctx.fg, align="r")
                    pg.end()
                    if q2:
                        count_qs.append(q2)
                yy = bbox([q])[3] + sz * 0.5
        yy += sz * 0.25
    if label_qs:
        pg.add("text", label_qs)
    if item_qs:
        pg.add("text", item_qs)
    if count_qs:
        pg.add("text", count_qs)
    return yy + sz * 0.3


def group_table(pg, ctx, x, y, w, ymax, groups=None):
    """Bordered table (ល.រ | ជំនាញ | counts... | exam) with spanning group-header rows. Whole box -> class table."""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.7, 0.85))
    hdr = ["ល.រ", "ជំនាញ"] + rng.sample(["វេនព្រឹក", "វេនរសៀល", "វេនយប់", "ចំនួនសរុប", "តម្លៃសិក្សា/១ឆ្នាំ", "ស្រី"], rng.randint(2, 4)) + ["វិញ្ញាសាប្រឡង"]
    ncol = len(hdr)
    ws = np.array([0.6, 3.2] + [1.1] * (ncol - 3) + [2.4], np.float32)
    ws = (ws / ws.sum() * w).astype(int)
    xs = np.concatenate([[0], np.cumsum(ws)]) + int(x)
    rh = int(sz * rng.uniform(1.9, 2.3))
    groups = groups or rng.randint(1, 3)
    lw = rng.choice([1, 2, 2])
    col = (rng.randint(10, 90),) * 3
    d = pg.d
    pg.begin("table")
    ty = int(y)
    rows = []  # (kind, cells)
    rows.append(("hdr", hdr))
    gnames = rng.sample(GROUPS, groups)
    for g in gnames:
        rows.append(("grp", [g]))
        for i in range(rng.randint(2, 5)):
            cells = [ctx.d(str(i + 1)), rng.choice(MAJORS)]
            for h in hdr[2:-1]:
                cells.append(f"{ctx.d(str(rng.choice([50, 60, 75, 100, 150, 180, 240, 300])))}នាក់" if "តម្លៃ" not in h else f"{ctx.d(str(rng.choice([1930000, 1940000, 2000000, 2200000])))}៛")
            cells.append(rng.choice(EXAMS))
            rows.append(("row", cells))
    max_rows = int((ymax - ty) / rh)
    rows = rows[:max(2, max_rows)]
    nr = len(rows)
    hf = rng.choice([(235, 235, 235), (225, 230, 240), None])
    if hf:
        d.rectangle([xs[0], ty, xs[-1], ty + rh], fill=hf)
    for r in range(nr + 1):
        yy = ty + r * rh
        d.line([xs[0], yy, xs[-1], yy], fill=col, width=lw)
    lab_cells = []  # v5 cell labels (tables.register_table); drawing and random draws are unchanged
    for r, (kind, cells) in enumerate(rows):
        yy = ty + r * rh
        if kind == "grp":
            if rng.random() < 0.5:
                d.rectangle([xs[0] + 1, yy + 1, xs[-1] - 1, yy + rh - 1], fill=(246, 246, 246))
            d.line([xs[0], yy, xs[0], yy + rh], fill=col, width=lw)
            d.line([xs[-1], yy, xs[-1], yy + rh], fill=col, width=lw)
            p = ctx.font("moul" if rng.random() < 0.6 else "bold", cells[0])
            n0 = len(pg.lines)
            if p:
                pg.text(xs[0] + 8, yy + (rh - sz * 1.25) / 2, cells[0], p, sz, ctx.fg, bound=(xs[0] + 1, yy, xs[-1] - 1, yy + rh))
            ids = list(range(n0, len(pg.lines)))
            lab_cells.append({"r": r, "c": 0, "rs": 1, "cs": ncol, "text": "".join(pg.line_texts[i] for i in ids), "header": False, "align": "l",
                              "bold": True, "line_ids": ids, "kind": "group"})
            continue
        for c in range(ncol + 1):
            d.line([xs[c], yy, xs[c], yy + rh], fill=col, width=lw)
        for c in range(ncol):
            t = cells[c] if c < len(cells) else ""
            a = "c" if kind == "hdr" or c == 0 else ("r" if c not in (1, ncol - 1) else "l")
            lc = {"r": r, "c": c, "rs": 1, "cs": 1, "text": "", "header": kind == "hdr", "align": a, "bold": kind == "hdr", "line_ids": [],
                  "kind": "hdr" if kind == "hdr" else "body"}
            lab_cells.append(lc)
            if not t:
                continue
            role = "bold" if kind == "hdr" else "body"
            p = ctx.font(role, t)
            if not p:
                continue
            f = font(p, sz)
            s = min(1.0, (ws[c] - 10) / max(1, f.getlength(t)))
            cx = xs[c] + (ws[c] / 2 if a == "c" else (ws[c] - 5 if a == "r" else 5))
            n0 = len(pg.lines)
            pg.text(cx, yy + (rh - sz * 1.25) / 2, t, p, int(sz * max(0.7, s)), ctx.fg, align=a, bound=(xs[c] + 1, yy, xs[c + 1] - 1, yy + rh))
            lc["line_ids"] = list(range(n0, len(pg.lines)))
            lc["text"] = "".join(pg.line_texts[i] for i in lc["line_ids"])
    pg.end()
    pg.add("table", (xs[0], ty, xs[-1] + 1, ty + nr * rh + 1))
    from .tables import register_table
    register_table(pg, [int(v) for v in xs], [ty + r * rh for r in range(nr + 1)], lab_cells, header_rows=1, borders="grid", style="v4_group")
    return ty + nr * rh + sz * 1.2


def sign_with_stamp(pg, ctx, x, y, w, ymax, org):
    """Right-side signature block: date (handwritten day) / role in Moul / signature / name, with a red stamp that
    overlaps signature + role + name + date (as on real letters). -> y below"""
    rng = ctx.rng
    sz = ctx.size
    cx = x + w / 2
    yy = y
    if rng.random() < 0.85:
        yy = hw_date_line(pg, ctx, x, yy, w, align=rng.choice(["c", "c", "r"]), lunar=rng.random() < 0.7)
    role = rng.choice(ROLES)
    q, yb = _t(pg, ctx, cx, yy, role, "moul" if rng.random() < 0.7 else "bold", sz * 0.95, align="c")
    role_y = yy
    yy = (yb if q else yy) + sz * 0.3
    gap = int(rng.uniform(2.2, 3.6) * sz)
    if yy + gap + sz * 2 > ymax:
        return yy
    sig = B.make_signature(ctx)
    s = min(1.0, (w * 0.8) / sig.width, gap * 1.1 / sig.height)
    if s < 1:
        sig = sig.resize((max(8, int(sig.width * s)), max(8, int(sig.height * s))), Image.LANCZOS)
    sx = cx - sig.width / 2 + rng.uniform(-0.12, 0.12) * w
    sig_y = yy + (gap - sig.height) / 2
    pg.ink(sig, (sx, sig_y), "signature")
    yy += gap
    name = ctx.name()
    q, yb = _t(pg, ctx, cx, yy, name, "moul" if rng.random() < 0.6 else "bold", sz * 0.95, align="c")
    if q:
        yy = yb
    if rng.random() < 0.85:  # the stamp: big, red, over the signature, reaching the role above and the name below
        d = sz * rng.uniform(5.0, 7.5)
        stp = B.make_stamp(ctx, d=d, org=org[0])
        scy = (role_y + yy) / 2 + rng.uniform(-0.3, 0.3) * sz
        scx = cx + rng.uniform(-0.25, 0.25) * w
        pg.ink(stp, (scx - stp.width / 2, scy - stp.height / 2), "stamp")
    return yy + sz


def approval_note(pg, ctx, x, y, w):
    """Handwritten approval note + initials somewhere in the free space (bottom-left / margin)."""
    rng = ctx.rng
    sz = ctx.size * rng.uniform(1.0, 1.4)
    txt = rng.choice(HW_NOTES)
    q = B.draw_hw(pg, ctx, x + rng.uniform(0, w * 0.3), y, txt, sz)
    if q and rng.random() < 0.7:
        B.draw_hw(pg, ctx, x + rng.uniform(0, w * 0.3), bbox([q])[3] + sz * 0.3, f"{ctx.d(rng.randint(1, 28))}/{ctx.d(rng.randint(1, 12))}/{ctx.d(rng.randint(2025, 2027))}", sz * 0.9)
    return y + sz * 2.5


def notice_footer(pg, ctx, x, y, w, org):
    """Address line + e-mail / web centred (small), optional QR at the left, page number."""
    rng = ctx.rng
    sz = int(ctx.size * rng.uniform(0.62, 0.74))
    yy = y
    if rng.random() < 0.6:
        pg.d.line([x, yy - sz * 0.8, x + w, yy - sz * 0.8], fill=(120, 120, 120), width=1)
    lines = [f"{rng.choice(PR_ADDR)} រាជធានីភ្នំពេញ ព្រះរាជាណាចក្រកម្ពុជា",
             rng.choice([f"អ៊ីមែល: secretary@{rng.choice(['rupp', 'rua', 'itc', 'num'])}.edu.kh វែបសាយ: www.{rng.choice(['rupp', 'rua', 'itc', 'num'])}.edu.kh",
                         f"info@{rng.choice(['moeys', 'mfaic', 'moh'])}.gov.kh / administration@{rng.choice(['moeys', 'mfaic', 'moh'])}.gov.kh",
                         f"ទូរស័ព្ទ៖ (៨៥៥) ២៣ {ctx.d(str(rng.randint(100, 999)))} {ctx.d(str(rng.randint(100, 999)))} · www.{rng.choice(['moeys', 'mfaic', 'moh'])}.gov.kh"])]
    qs = []
    for t in lines:
        pg.begin("footer")
        q = pg.text(x + w / 2, yy, t, ctx.font("body", t), sz, (70, 70, 70), align="c")
        pg.end()
        if q:
            qs.append(q)
            yy = bbox([q])[3] + sz * 0.5
    if qs:
        pg.add("footer", qs)
    if rng.random() < 0.35:
        qs_ = int(sz * rng.uniform(3.5, 5))
        B.b_qr(pg, ctx, x, y - qs_ - sz * 0.5 if rng.random() < 0.5 else y, qs_)
    return yy


# =========================================================================== templates
def km_gov_notice(rng, fake):
    """University / ministry announcement: letterhead, org block with handwritten ref, Moul title + subject lines,
    "យោង ៖" paragraphs, section heading, label : value rows or dash list, optional table with group rows,
    signature block with a stamp over it, handwritten approval, footer."""
    pg = new_page(rng, "a4", 0.05)
    ctx = Ctx(rng, "km", fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.06, 0.1))
    x, w = m, W - 2 * m
    org = rng.choice(UNIS) if rng.random() < 0.6 else rng.choice(MINISTRIES)
    pg.meta["degrade_p"] = 0.85
    y = letterhead(pg, ctx, x, int(H * rng.uniform(0.025, 0.045)), w, org)
    y_org = org_lines(pg, ctx, x, y - ctx.size * rng.uniform(0.0, 0.8), w, org, hw_ref=rng.random() < 0.8)
    y = y_org
    # title block (Moul, centred): "សេចក្តីជូនដំណឹង" + 1-2 subject lines, sometimes on the right half only
    title = rng.choice(NOTICE_TITLES)
    tx, tw = (x, w) if rng.random() < 0.6 else (x + w * 0.3, w * 0.7)
    y = B.b_heading(pg, ctx, tx, y, tw, cls="title", text=title, size=ctx.size * rng.uniform(1.25, 1.6), align="c", role="moul", max_lines=1)
    for k in range(rng.randint(1, 2)):
        sub = rng.choice(NOTICE_SUBJECTS) if k == 0 else rng.choice(NOTICE_PLACES)
        y = B.b_heading(pg, ctx, tx, y - ctx.size * 0.2, tw, cls="title", text=sub, size=ctx.size * rng.uniform(0.95, 1.1), align="c", role="moul", max_lines=2)
    y += ctx.size * rng.uniform(0.2, 0.6)
    ymax = H * rng.uniform(0.70, 0.78)
    ind = int(ctx.size * rng.uniform(1.8, 2.6))
    if rng.random() < 0.75:  # "យោង ៖" / "កម្មវត្ថុ ៖" lead-ins
        for pre in rng.sample(["យោង ៖ ", "កម្មវត្ថុ ៖ ", "សូមគោរពជូន "], rng.randint(1, 2)):
            y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(1, 3), text=pre + ctx.para(rng.randint(120, 260)), indent=0)
    n_par = rng.randint(1, 3)
    for _ in range(n_par):
        if y > ymax - ctx.size * 4:
            break
        y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(2, 6), text=ctx.para(rng.randint(220, 520)), indent=ind)
    body_kind = rng.random()
    if y < ymax - ctx.size * 6:
        sec = rng.choice(PR_SECTIONS)
        y = B.b_heading(pg, ctx, x, y, w, ymax, cls="section_header", text=rng.choice(["I- ", "II- ", "ក- ", "១- ", ""]) + sec, size=ctx.size * rng.uniform(0.95, 1.1), role="moul" if rng.random() < 0.5 else "bold", max_lines=1)
        if body_kind < 0.4:
            y = label_value_rows(pg, ctx, x, y, w, ymax)
        elif body_kind < 0.7:
            y = B.b_list(pg, ctx, x + ind * 0.5, y, w - ind * 0.5, ymax, n=rng.randint(2, 5), style=rng.choice(["dash", "dash", "arrow", "bullet", "paren", "num"]))
        else:
            y = group_table(pg, ctx, x, y + ctx.size * 0.3, w, ymax)
    if rng.random() < 0.3 and y < ymax - ctx.size * 5:
        y = B.b_list(pg, ctx, x + ind * 0.5, y, w - ind * 0.5, ymax, n=rng.randint(2, 4), style=rng.choice(["dash", "arrow", "num"]))
    # signature block at the right, stamp over it
    sw = w * rng.uniform(0.38, 0.5)
    ys = sign_with_stamp(pg, ctx, x + w - sw, y + ctx.size * rng.uniform(0.2, 0.8), sw, H - m * 1.6, org)
    if rng.random() < 0.6:  # handwritten approval / initials at the left
        approval_note(pg, ctx, x, y + ctx.size * rng.uniform(1.0, 3.0), w * 0.45)
    if rng.random() < 0.4 and ys < H - m * 2.2:  # a second handwritten name / "ហត្ថលេខា" under the block
        B.draw_hw(pg, ctx, x + w - sw * rng.uniform(0.6, 0.9), ys, ctx.name(), ctx.size * 1.3)
    if rng.random() < 0.6:
        B.b_page_number(pg, ctx, x + w / 2, H - m * 0.95 - ctx.size * 1.7, "c", int(ctx.size * 0.8))
    notice_footer(pg, ctx, x, H - m * 0.95, w, org)
    return pg, ctx


def km_letter_v4(rng, fake):
    """Ministry letter: two-part letterhead (org block left with handwritten ref, motto right), handwritten date
    gaps, "ជម្រាបជូន" + addressee lines, paragraphs, a numbered/lettered list, closing with stamp over the date
    and signature block, copy-to list."""
    pg = new_page(rng, "a4", 0.06)
    ctx = Ctx(rng, "km", fake)
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.07, 0.1))
    x, w = m, W - 2 * m
    org = rng.choice(MINISTRIES) if rng.random() < 0.7 else rng.choice(UNIS)
    pg.meta["degrade_p"] = 0.8
    y0 = int(H * rng.uniform(0.03, 0.05))
    sz = ctx.size
    # letterhead: emblem + motto (centre or right), org lines left
    if rng.random() < 0.5:
        emb = int(sz * rng.uniform(3.2, 4.4))
        B.b_logo(pg, ctx, x + rng.uniform(0, w * 0.05), y0, emb, kind="emblem")
        ym = motto(pg, ctx, x + w * rng.uniform(0.62, 0.72), y0 + sz * rng.uniform(0, 0.5), size=sz * rng.uniform(1.0, 1.15))
        y = org_lines(pg, ctx, x, y0 + emb + sz * 0.3, w, org, hw_ref=True)
        y = max(y, ym)
    else:
        y = letterhead(pg, ctx, x, y0, w, org)
        y = org_lines(pg, ctx, x, y - sz * 0.5, w, org, hw_ref=True)
    if rng.random() < 0.7:
        y = hw_date_line(pg, ctx, x + w * 0.45, y - sz * 2.2, w * 0.55, align="r", lunar=rng.random() < 0.6)
    # addressee / subject
    y = B.b_heading(pg, ctx, x, y + sz * 0.3, w, cls="title", text=rng.choice(["ឧបនាយករដ្ឋមន្ត្រី រដ្ឋមន្ត្រីក្រសួងអប់រំ យុវជន និងកីឡា", "សូមគោរពជូន", "ជម្រាបជូន", rng.choice(NOTICE_TITLES)]),
                     size=sz * rng.uniform(1.1, 1.4), align="c", role="moul", max_lines=2)
    if rng.random() < 0.7:
        tos = rng.sample(["ឯកឧត្តម លោកជំទាវ លោក លោកស្រី ប្រធានអង្គភាពក្រោមឱវាទក្រសួង", "លោក លោកស្រី ប្រធានមន្ទីរអប់រំ យុវជន និងកីឡា រាជធានី-ខេត្ត", "លោកសាកលវិទ្យាធិការ នៃសាកលវិទ្យាល័យទាំងអស់",
                          "ឯកឧត្តមប្រធានមហាវិទ្យាល័យ", "លោកនាយកវិទ្យាល័យទូទាំងប្រទេស"], rng.randint(1, 2))
        y = B.b_list(pg, ctx, x + sz, y, w - sz, items=tos, style="dash", size=sz * 0.95)
    ymax = H * rng.uniform(0.66, 0.74)
    ind = int(sz * rng.uniform(1.8, 2.6))
    for pre in rng.sample(["កម្មវត្ថុ៖ ", "យោង៖ "], rng.randint(1, 2)):
        y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(1, 3), text=pre + ctx.para(rng.randint(100, 240)), indent=0)
    for _ in range(rng.randint(1, 3)):
        if y > ymax - sz * 4:
            break
        y = B.b_para(pg, ctx, x, y, w, ymax, n=rng.randint(2, 6), text=ctx.para(rng.randint(200, 520)), indent=ind)
    if rng.random() < 0.7 and y < ymax - sz * 5:
        if rng.random() < 0.5:
            y = B.b_heading(pg, ctx, x, y, w, ymax, cls="section_header", text=rng.choice(["ក) វគ្គសិក្សា", "ខ) លក្ខខណ្ឌ", "១. ការចុះឈ្មោះ", "គ) កាលបរិច្ឆេទ"]), size=sz, role="bold", max_lines=1)
        items = [f"C{rng.randint(100, 499)}: {rng.choice(['Advanced Specialist Certificate in Teaching Listening and Speaking', 'Postgraduate Diploma in TESOL', 'Master of Teaching English to Speakers of Other Languages', 'Specialist Certificate in Language Assessment'])} ({ctx.d(rng.randint(1, 28))} {rng.choice(KH_MONTHS)} {ctx.d(2027)} - {ctx.d(rng.randint(1, 28))} {rng.choice(KH_MONTHS)} {ctx.d(2027)}) ({rng.choice(['In-Person', 'Blended', 'Online'])})" for _ in range(rng.randint(2, 5))] if rng.random() < 0.5 else None
        y = B.b_list(pg, ctx, x + sz, y, w - sz, ymax, n=rng.randint(2, 5), items=items, style=rng.choice(["paren", "num", "dash", "alpha"]))
    # closing: date at the right with a stamp over it + signature block
    sw = w * rng.uniform(0.4, 0.5)
    ys = sign_with_stamp(pg, ctx, x + w - sw, y + sz * rng.uniform(0.3, 1.0), sw, H - m * 1.8, org)
    if rng.random() < 0.5:
        approval_note(pg, ctx, x, y + sz * rng.uniform(0.5, 2.5), w * 0.4)
    if rng.random() < 0.5 and ys < H - m * 2.5:  # copy-to list bottom-left
        q, yb = _t(pg, ctx, x, ys - sz * 2, "ចម្លងជូន៖", "bold", sz * 0.85)
        if q:
            B.b_list(pg, ctx, x, yb + sz * 0.3, w * 0.5, H - m * 1.2, items=[rng.choice(["ខុទ្ទកាល័យ", "ឯកសារ-កាលប្បវត្តិ", "ដូចប្រការ៣", "នាយកដ្ឋានពាក់ព័ន្ធ"]) + (" «ដើម្បីជ្រាប»" if rng.random() < 0.5 else "") for _ in range(rng.randint(2, 4))], style="dash", size=sz * 0.85)
    if rng.random() < 0.5:
        notice_footer(pg, ctx, x, H - m * 0.95, w, org)
    elif rng.random() < 0.5:
        footer_row(pg, ctx, x, H - m * 0.95 + sz * 0.2, w, org)
    return pg, ctx
