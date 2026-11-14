"""Sheet doc types (docgen v5): full-page tables as users print or photograph them -> doc_kind "sheet".

Spreadsheet printouts (Excel look: thin grey gridlines, sometimes column letters A-H + row numbers like a printed
worksheet with headings on, title + date above, "Page 1 of N" footer), attendance / name lists, price lists,
budget / expense statements, class timetables (merged cells across periods), score sheets, inventory lists, ledgers and
payrolls; Khmer, English and mixed; portrait and landscape; signature lines or a stamp under the table sometimes.
Every table goes through blocks.b_table (tables.py engine) with ``sheet=True`` so the record carries cell-level labels.
Registered in generate.py only with --with-sheets / --table-only (or when named in --doc-types) so that v3/v4 runs
keep their doc-type stream.
"""

from __future__ import annotations

import random

from khmerocr.textgen import MONTHS

from . import blocks as B
from . import tables as T
from .canvas import bbox
from .ctx import KH_COMPANIES, KH_PRODUCTS, KH_PROV, KH_PROV_EN, MINISTRIES, SUB_UNITS, Ctx
from .templates_km import KH_PRODUCTS_EN, motto, new_page
from .templates_km_v4 import UNIS

SEX_KM = ["ប្រុស", "ស្រី"]
POS_KM = ["គ្រូបង្រៀន", "បុគ្គលិក", "មន្ត្រី", "ប្រធានផ្នែក", "គណនេយ្យករ", "អ្នកបើកបរ", "អនុប្រធាន", "លេខាធិការ", "និស្សិត", "កម្មករ", "ជំនួយការ", "អ្នកលក់"]
POS_EN = ["Teacher", "Staff", "Officer", "Manager", "Accountant", "Driver", "Assistant", "Engineer", "Clerk", "Intern", "Supervisor", "Analyst", "Cashier"]
DEPT_KM = ["ផ្នែកលក់", "ផ្នែកគណនេយ្យ", "ផ្នែកធនធានមនុស្ស", "ផ្នែកបច្ចេកទេស", "ផ្នែករដ្ឋបាល", "ផ្នែកទីផ្សារ", "ផ្នែកឃ្លាំង"] + SUB_UNITS[:4]
DEPT_EN = ["Sales", "Accounting", "Human Resources", "Engineering", "Operations", "Marketing", "IT", "Administration", "Logistics", "Finance"]
SUBJ_KM = ["គណិតវិទ្យា", "អក្សរសាស្ត្រខ្មែរ", "រូបវិទ្យា", "គីមីវិទ្យា", "ជីវវិទ្យា", "ភាសាអង់គ្លេស", "ភូមិវិទ្យា", "ប្រវត្តិវិទ្យា", "សីលធម៌", "ផែនដីវិទ្យា", "កីឡា", "ព័ត៌មានវិទ្យា"]
SUBJ_EN = ["Math", "English", "Science", "History", "Geography", "Physics", "Chemistry", "Biology", "Art", "Music", "PE", "Computer Sci.", "Khmer", "Economics"]
DAYS_KM = ["ច័ន្ទ", "អង្គារ", "ពុធ", "ព្រហស្បតិ៍", "សុក្រ", "សៅរ៍"]
DAYS_EN = [["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]]
CAT_KM = ["គ្រឿងទេស", "ភេសជ្ជៈ", "សម្ភារការិយាល័យ", "សម្ភារសំណង់", "គ្រឿងអេឡិចត្រូនិច", "អាហារ", "សម្លៀកបំពាក់", "ឱសថ", "គ្រឿងសង្ហារឹម"]
CAT_EN = ["Beverages", "Office Supplies", "Hardware", "Electronics", "Food", "Clothing", "Stationery", "Cleaning", "Furniture", "Tools"]
UNITS_KM = ["គីឡូ", "ដប", "កញ្ចប់", "ដុំ", "ប្រអប់", "លីត្រ", "ឈុត", "កេស", "គ្រឿង"]
UNITS_EN = ["kg", "pcs", "box", "set", "L", "pack", "ea", "ctn", "roll", "pair"]
PROD_EN = ["Copy paper A4", "Ballpoint pen", "Stapler", "Printer toner", "USB cable", "Desk lamp", "Water 500ml", "Coffee 1kg", "Sugar 1kg",
           "Cement 50kg", "Steel bar 12mm", "Paint 5L", "Keyboard", "Mouse", "Monitor 24in", "Notebook", "Envelope", "Hand soap", "Detergent",
           "Rice 25kg", "Fish sauce 1L", "Cooking oil 1L", "T-shirt (M)", "Sandals", "Phone charger", "Power strip", "Chair", "Folding table"]
BUDGET_KM = ["ប្រាក់បៀវត្សបុគ្គលិក", "ថ្លៃឈ្នួលការិយាល័យ", "អគ្គិសនី និងទឹក", "សម្ភារការិយាល័យ", "ការធ្វើដំណើរ", "ថ្លៃទំនាក់ទំនង", "ការជួសជុល", "ការបណ្តុះបណ្តាល",
             "ការផ្សព្វផ្សាយ", "ឧបករណ៍", "ថ្លៃសេវាកម្ម", "ផ្សេងៗ", "ប្រាក់ឧបត្ថម្ភ", "ថ្លៃប្រេងឥន្ធនៈ", "ការធានារ៉ាប់រង", "សិក្ខាសាលា", "ថ្លៃជួលរថយន្ត"]
BUDGET_EN = ["Salaries & wages", "Office rent", "Utilities", "Office supplies", "Travel", "Communications", "Repairs & maintenance", "Training",
             "Advertising", "Equipment", "Professional services", "Insurance", "Miscellaneous", "Fuel", "Workshops", "Vehicle rental", "Software licences"]
BGRP_KM = ["ជំពូកទី១៖ ចំណាយបុគ្គលិក", "ជំពូកទី២៖ ចំណាយប្រតិបត្តិការ", "ជំពូកទី៣៖ ចំណាយមូលធន", "ក. ចំណូល", "ខ. ចំណាយ", "គ. ការវិនិយោគ"]
BGRP_EN = ["Personnel", "Operating expenses", "Capital expenditure", "A. Revenue", "B. Expenses", "Programme costs", "Administration"]
LEDG_KM = ["លក់ទំនិញ", "ទិញសម្ភារ", "បង់ប្រាក់ខែ", "ទទួលប្រាក់ពីអតិថិជន", "បង់ថ្លៃអគ្គិសនី", "ផ្ទេរប្រាក់ចូលធនាគារ", "ដកសាច់ប្រាក់", "បង់ថ្លៃឈ្នួល", "ចំណូលសេវាកម្ម", "ការប្រាក់"]
LEDG_EN = ["Sales invoice", "Purchase of supplies", "Salary payment", "Customer receipt", "Electricity bill", "Bank deposit", "Cash withdrawal", "Rent payment",
           "Service income", "Interest received", "Loan repayment", "Petty cash"]
STATUS_EN = ["Paid", "Pending", "Open", "Closed", "Done", "Overdue", "Active", "Shipped", "Cancelled"]
STATUS_KM = ["បានបង់", "មិនទាន់", "កំពុងដំណើរការ", "រួចរាល់", "បានបញ្ជូន", "បោះបង់", "សកម្ម"]


# --------------------------------------------------------------------------- small helpers
def _name(ctx, latin_p=0.15):
    if ctx.lang == "km":
        return ctx.kh_name(True)[0].title() if ctx.rng.random() < latin_p else ctx.kh_name()[0]
    return ctx.fake.name() if ctx.rng.random() > latin_p else ctx.kh_name(True)[0].title()


def _sex(ctx):
    rng = ctx.rng
    if ctx.lang == "km":
        return rng.choice(SEX_KM)
    return rng.choice([("M", "F"), ("Male", "Female")])[rng.randint(0, 1)]


def _amount(ctx, lo, hi, cents=None, cur=""):
    rng = ctx.rng
    v = rng.uniform(lo, hi)
    cents = (rng.random() < 0.5) if cents is None else cents
    s = f"{v:,.2f}" if cents else f"{int(v):,}"
    return ctx.d(cur + s) if cur in ("$", "") else ctx.d(s) + cur


def _date(ctx, year=None):
    rng = ctx.rng
    yy = year or rng.randint(2019, 2027)
    dd, mm = rng.randint(1, 28), rng.randint(1, 12)
    if ctx.lang == "km":
        return ctx.d(rng.choice([f"{dd:02d}/{mm:02d}/{yy}", f"{dd:02d}-{mm:02d}-{yy}", f"{dd:02d}/{mm:02d}/{str(yy)[2:]}"]))
    return rng.choice([f"{mm:02d}/{dd:02d}/{yy}", f"{yy}-{mm:02d}-{dd:02d}", f"{dd:02d}-{mm:02d}-{yy}"])


def _period(ctx):
    rng = ctx.rng
    yy, mm = rng.randint(2019, 2027), rng.randint(1, 12)
    if ctx.lang == "km":
        return rng.choice([f"ប្រចាំខែ{MONTHS[mm - 1]} ឆ្នាំ{ctx.d(yy)}", f"ឆ្នាំ{ctx.d(yy)}", f"ត្រីមាសទី{ctx.d(rng.randint(1, 4))} ឆ្នាំ{ctx.d(yy)}",
                           f"ឆ្នាំសិក្សា {ctx.d(yy)}-{ctx.d(yy + 1)}", f"ឆមាសទី{ctx.d(rng.randint(1, 2))} ឆ្នាំសិក្សា {ctx.d(yy)}-{ctx.d(yy + 1)}"])
    mon = ctx.fake.month_name()
    return rng.choice([f"{mon} {yy}", f"Q{rng.randint(1, 4)} {yy}", f"FY{yy}", f"{yy}-{yy + 1}", f"Semester {rng.randint(1, 2)}, {yy}"])


def _bil(ctx, km, en, p=None):
    """Column header: Khmer, 'Khmer / English' or English depending on the page language and a per-table choice."""
    mode = ctx._sheet_hdr_mode
    if ctx.lang != "km":
        return en
    if mode == "bil":
        return f"{km} / {en}" if ctx.rng.random() < 0.85 else km
    if mode == "en":
        return en
    return km


def _mark(ctx):
    return ctx._sheet_mark


# --------------------------------------------------------------------------- data builders -> spec
def data_attendance(ctx, n, wide):
    rng = ctx.rng
    km = ctx.lang == "km"
    daily = wide and rng.random() < 0.7
    if daily:  # one column per day with marks
        nd = rng.randint(8, 16) if wide else rng.randint(5, 8)
        start = rng.randint(1, 28 - nd + 1)
        days = [ctx.d(str(start + i)) for i in range(nd)]
        hdr = [_bil(ctx, "ល.រ", "No."), _bil(ctx, "ឈ្មោះ", "Name"), _bil(ctx, "ភេទ", "Sex")] + days + [_bil(ctx, "សរុប", "Total")]
        al = ["c", "l", "c"] + ["c"] * nd + ["c"]
        marks = [_mark(ctx), _mark(ctx), _mark(ctx), rng.choice(["A", "x", "អ", "-", ""]) if km else rng.choice(["A", "x", "-", ""])]
        rows = []
        for i in range(n):
            ms = [rng.choice(marks[:3]) if rng.random() < 0.85 else marks[3] for _ in range(nd)]
            rows.append([ctx.d(i + 1), _name(ctx), _sex(ctx)] + ms + [ctx.d(sum(1 for m in ms if m in marks[:3]))])
        spec = T.spec_from_data(hdr, rows, al)
        if rng.random() < 0.5:
            lab = (f"ខែ{MONTHS[rng.randint(0, 11)]}" if km else ctx.fake.month_name())
            T.two_row_header(ctx, spec, labels=[lab], groups=[(3, nd)])
        spec["row_header"] = rng.random() < 0.3
        return spec, {"title": "attendance", "sig_col": None}
    extra = rng.choice([("តួនាទី", "Position", POS_KM, POS_EN), ("ថ្នាក់", "Class", None, None), ("ផ្នែក", "Department", DEPT_KM, DEPT_EN)])
    cols = [("ល.រ", "No.", "c"), ("ឈ្មោះ", "Name", "l"), ("ភេទ", "Sex", "c"), (extra[0], extra[1], "l")]
    if rng.random() < 0.5:
        cols.append(("ទូរស័ព្ទ", "Phone", "l"))
    if rng.random() < 0.3:
        cols.append(("ម៉ោងចូល", "Time in", "c"))
        cols.append(("ម៉ោងចេញ", "Time out", "c"))
    sig = rng.random() < 0.8
    cols.append(("ហត្ថលេខា", "Signature", "c") if sig else ("ផ្សេងៗ", "Remarks", "l"))
    hdr = [_bil(ctx, k, e) for k, e, _ in cols]
    al = [a for _, _, a in cols]
    rows = []
    for i in range(n):
        r = [ctx.d(i + 1), _name(ctx), _sex(ctx)]
        if extra[2] is None:
            r.append((f"ថ្នាក់ទី{ctx.d(rng.randint(7, 12))}{rng.choice('កខគឃ')}" if km else f"Grade {rng.randint(7, 12)}{rng.choice('ABCD')}"))
        else:
            r.append(rng.choice(extra[2] if km else extra[3]))
        for k, e, _ in cols[4:]:
            if k == "ទូរស័ព្ទ":
                r.append(ctx.phone())
            elif k in ("ម៉ោងចូល", "ម៉ោងចេញ"):
                r.append(ctx.d(f"{rng.randint(7, 17)}:{rng.choice(['00', '15', '30', '45'])}") if rng.random() < 0.9 else "")
            else:
                r.append("" if sig else (ctx.phrase(4, 18) if rng.random() < 0.25 else ""))
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al)
    if rng.random() < 0.25:
        T.add_group_rows(ctx, spec, n=rng.randint(1, 3), texts=rng.sample(DEPT_KM if km else DEPT_EN, 3))
    spec["row_header"] = False
    return spec, {"title": "attendance", "sig_col": len(hdr) - 1 if sig else None}


def data_price_list(ctx, n, wide):
    rng = ctx.rng
    km = ctx.lang == "km"
    menu = rng.random() < 0.3
    if menu:  # 2-3 columns, dotted leaders look
        hdr = [_bil(ctx, "ឈ្មោះមុខម្ហូប", "Item"), _bil(ctx, "តម្លៃ", "Price")] if rng.random() < 0.6 else [_bil(ctx, "ល.រ", "No."), _bil(ctx, "ឈ្មោះទំនិញ", "Item"), _bil(ctx, "តម្លៃ", "Price")]
        rows = []
        for i in range(n):
            item = rng.choice(KH_PRODUCTS + KH_PRODUCTS_EN) if km else rng.choice(KH_PRODUCTS_EN + PROD_EN)
            price = _amount(ctx, 1, 60, cents=True, cur="$") if rng.random() < 0.6 else _amount(ctx, 2000, 80000, cents=False, cur=" ៛")
            rows.append(([ctx.d(i + 1)] if len(hdr) == 3 else []) + [item, price])
        spec = T.spec_from_data(hdr, rows, ["c", "l", "r"][-len(hdr):])
        if rng.random() < 0.6:
            T.add_group_rows(ctx, spec, n=rng.randint(1, 3), texts=rng.sample(CAT_KM if km else CAT_EN, 3), align=rng.choice(["l", "c"]))
        return spec, {"title": "price_list", "style": "dotted" if rng.random() < 0.6 else None}
    cols = [("ល.រ", "No.", "c"), ("កូដ", "Code", "c"), ("ឈ្មោះទំនិញ", "Description", "l"), ("ឯកតា", "Unit", "c")]
    if rng.random() < 0.3:
        cols.pop(1)
    cur = rng.choice(["៛", "$", "both"]) if km else "$"
    if cur == "both":
        cols += [("តម្លៃ (៛)", "Price (KHR)", "r"), ("តម្លៃ ($)", "Price (USD)", "r")]
    else:
        cols += [(f"តម្លៃលក់រាយ ({cur})", "Retail price", "r"), (f"តម្លៃលក់ដុំ ({cur})", "Wholesale", "r")] if rng.random() < 0.5 else [(f"តម្លៃ ({cur})", "Unit price", "r")]
    if rng.random() < 0.4:
        cols.append(("ផ្សេងៗ", "Remarks", "l"))
    hdr = [_bil(ctx, k, e) for k, e, _ in cols]
    al = [a for _, _, a in cols]
    rows = []
    for i in range(n):
        r = []
        for k, e, a in cols:
            if k == "ល.រ":
                r.append(ctx.d(i + 1))
            elif k == "កូដ":
                r.append(f"{rng.choice('ABCDEFGHKMP')}{rng.randint(100, 9999):04d}")
            elif k == "ឈ្មោះទំនិញ":
                r.append(rng.choice(KH_PRODUCTS) if km and rng.random() < 0.8 else rng.choice(PROD_EN))
            elif k == "ឯកតា":
                r.append(rng.choice(UNITS_KM if km and rng.random() < 0.8 else UNITS_EN))
            elif "៛" in k or "KHR" in e:
                r.append(_amount(ctx, 1000, 400000, cents=False))
            elif k.startswith("តម្លៃ"):
                r.append(_amount(ctx, 0.5, 900, cents=True))
            else:
                r.append(ctx.phrase(4, 16) if rng.random() < 0.2 else "")
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al)
    if rng.random() < 0.6:
        T.add_group_rows(ctx, spec, n=rng.randint(1, 4), texts=rng.sample(CAT_KM if km else CAT_EN, 4))
    if rng.random() < 0.3:
        T.add_row_spans(ctx, spec, col=next((i for i, (k, _, _) in enumerate(cols) if k == "ឯកតា"), None))
    return spec, {"title": "price_list"}


def data_budget(ctx, n, wide):
    rng = ctx.rng
    km = ctx.lang == "km"
    cur = rng.choice(["៛", "$"]) if km else "$"
    cols = [("កូដ", "Code", "c"), ("បរិយាយ", "Description", "l"), ("ថវិកាអនុម័ត", "Budget", "r"), ("ចំណាយជាក់ស្តែង", "Actual", "r"), ("សមតុល្យ", "Variance", "r")]
    if rng.random() < 0.5:
        cols.append(("%", "% Used", "r"))
    if rng.random() < 0.3:
        cols.append(("ផ្សេងៗ", "Notes", "l"))
    if rng.random() < 0.3:
        cols.pop(0)
    hdr = [_bil(ctx, k, e) for k, e, _ in cols]
    al = [a for _, _, a in cols]
    items = BUDGET_KM if km else BUDGET_EN
    rows = []
    for i in range(n):
        b = rng.uniform(500, 50000) * (4000 if cur == "៛" else 1)
        a = b * rng.uniform(0.3, 1.2)
        r = []
        for k, e, _ in cols:
            if k == "កូដ":
                r.append(ctx.d(f"{rng.randint(60, 69)}{rng.randint(0, 9)}{rng.randint(0, 9)}") if rng.random() < 0.7 else f"{rng.randint(1, 9)}.{rng.randint(1, 9)}")
            elif k == "បរិយាយ":
                r.append(rng.choice(items))
            elif k == "ថវិកាអនុម័ត":
                r.append(ctx.d(f"{b:,.0f}" if cur == "៛" else f"{b:,.2f}"))
            elif k == "ចំណាយជាក់ស្តែង":
                r.append(ctx.d(f"{a:,.0f}" if cur == "៛" else f"{a:,.2f}") if rng.random() < 0.92 else "")
            elif k == "សមតុល្យ":
                v = b - a
                r.append(ctx.d((f"{v:,.0f}" if cur == "៛" else f"{v:,.2f}") if v >= 0 else (f"({-v:,.0f})" if rng.random() < 0.4 else f"-{-v:,.2f}")))
            elif k == "%":
                r.append(ctx.d(f"{100 * a / b:.1f}%"))
            else:
                r.append(ctx.phrase(5, 20) if rng.random() < 0.2 else "")
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al)
    if rng.random() < 0.7:
        T.add_group_rows(ctx, spec, n=rng.randint(1, 4), texts=rng.sample(BGRP_KM if km else BGRP_EN, 4))
    if rng.random() < 0.4:
        labs = [f"ឆ្នាំ{ctx.d(rng.randint(2022, 2027))}" if km else f"FY{rng.randint(2022, 2027)}", "ទឹកប្រាក់ (៛)" if km else f"Amount ({'KHR' if cur == '៛' else 'USD'})"]
        c0 = hdr.index(_bil(ctx, "ថវិកាអនុម័ត", "Budget"))
        T.two_row_header(ctx, spec, labels=[rng.choice(labs)], groups=[(c0, 3)])
    T.add_totals(ctx, spec, label=("សរុបរួម" if km else rng.choice(["TOTAL", "Grand total"])))
    spec["row_header"] = False
    return spec, {"title": "budget"}


def data_timetable(ctx, n, wide):
    rng = ctx.rng
    km = ctx.lang == "km"
    nd = rng.choice([5, 5, 6])
    days = DAYS_KM[:nd] if km else rng.choice(DAYS_EN)[:nd]
    np_ = rng.randint(6, 10)
    t0 = rng.choice([7, 8, 13])
    slots = []
    for p in range(np_):
        h0 = t0 + p
        slots.append(ctx.d(f"{h0}:00-{h0}:{rng.choice(['45', '50', '55'])}") if rng.random() < 0.5 else ctx.d(f"{h0:02d}:00 - {h0 + 1:02d}:00"))
    subj = SUBJ_KM if km else SUBJ_EN
    corner = _bil(ctx, "ម៉ោង / ថ្ងៃ", "Time / Day") if rng.random() < 0.6 else (_bil(ctx, "ម៉ោង", "Period"))
    hdr = [corner] + list(days)
    al = ["c"] * (nd + 1)
    rows = []
    brk = rng.randint(2, np_ - 2) if rng.random() < 0.7 else -1
    for p in range(np_):
        if p == brk:
            rows.append([slots[p]] + [("សម្រាក" if rng.random() < 0.6 else "អាហារថ្ងៃត្រង់") if km else rng.choice(["Break", "Lunch", "Recess"])] + [""] * (nd - 1))
            continue
        r = [slots[p]]
        for d_ in range(nd):
            if rng.random() < 0.12:
                r.append("")
            else:
                s = rng.choice(subj)
                if rng.random() < 0.35:
                    s += (f" ({'បន្ទប់' if km else 'Rm'} {ctx.d(rng.randint(101, 312))})" if rng.random() < 0.5 else f" - {ctx.kh_name(True)[0].title() if not km else ctx.kh_name()[0]}")
                r.append(s)
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al, max_lines=3)
    for i, r in enumerate(spec["rows"]):
        if i == 0:
            continue
        if i - 1 == brk:
            r["cells"][1]["cs"] = nd
            r["cells"][1]["bold"] = True
            r["cells"][1]["fill"] = rng.choice([None, (240, 240, 240), (250, 240, 225)])
            for d_ in range(2, nd + 1):
                r["cells"][d_] = None
        r["cells"][0]["bold"] = True
    # double periods: a subject spanning two consecutive periods
    for d_ in range(1, nd + 1):
        if rng.random() < 0.5:
            T.add_row_spans(ctx, spec, col=d_, n=rng.randint(1, 2), max_rs=2)
    if rng.random() < 0.3:  # second header row with dates
        dates = [None] + [ctx.d(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}") for _ in range(nd)]
        top = [dict(spec["rows"][0]["cells"][0], rs=2)] + [spec["rows"][0]["cells"][c] for c in range(1, nd + 1)]
        sub = [None] + [T.cell(dates[c], align="c", bold=False, header=True) for c in range(1, nd + 1)]
        spec["rows"][0] = {"kind": "hdr", "cells": top}
        spec["rows"].insert(1, {"kind": "hdr", "cells": sub})
        spec["header_rows"] = 2
    spec["row_header"] = True
    cls = (f"ថ្នាក់ទី{ctx.d(rng.randint(7, 12))}{rng.choice('កខគ')}" if km else f"Grade {rng.randint(6, 12)}{rng.choice('ABC')}")
    return spec, {"title": "timetable", "subtitle": cls}


def data_scores(ctx, n, wide):
    rng = ctx.rng
    km = ctx.lang == "km"
    ns = rng.randint(4, 7) if wide else rng.randint(3, 5)
    subs = rng.sample(SUBJ_KM if km else SUBJ_EN, ns)
    tail = [("សរុប", "Total", "r"), ("មធ្យមភាគ", "Average", "r")]
    if rng.random() < 0.7:
        tail.append(("ចំណាត់ថ្នាក់", "Rank", "c"))
    if rng.random() < 0.5:
        tail.append(("និទ្ទេស", "Grade", "c"))
    if rng.random() < 0.3:
        tail.append(("ផ្សេងៗ", "Remarks", "l"))
    head = [("ល.រ", "No.", "c"), ("ឈ្មោះសិស្ស", "Student name", "l"), ("ភេទ", "Sex", "c")]
    hdr = [_bil(ctx, k, e) for k, e, _ in head] + subs + [_bil(ctx, k, e) for k, e, _ in tail]
    al = [a for _, _, a in head] + ["r"] * ns + [a for _, _, a in tail]
    top = rng.choice([10, 20, 50, 100])
    rows = []
    for i in range(n):
        sc = [rng.uniform(0.2, 1.0) * top for _ in range(ns)]
        absent = rng.random() < 0.08
        cells = ["" if absent else (ctx.d(f"{s:.1f}") if top <= 20 or rng.random() < 0.3 else ctx.d(f"{s:.0f}")) for s in sc]
        tot, avg = sum(sc), sum(sc) / ns
        r = [ctx.d(i + 1), _name(ctx), _sex(ctx)] + cells
        for k, e, _ in tail:
            if k == "សរុប":
                r.append("" if absent else ctx.d(f"{tot:.1f}"))
            elif k == "មធ្យមភាគ":
                r.append("" if absent else ctx.d(f"{avg:.2f}"))
            elif k == "ចំណាត់ថ្នាក់":
                r.append("" if absent else ctx.d(rng.randint(1, n)))
            elif k == "និទ្ទេស":
                r.append("" if absent else rng.choice(["A", "B", "C", "D", "E", "F"] if not km or rng.random() < 0.5 else ["ល្អប្រសើរ", "ល្អ", "ល្អបង្គួរ", "មធ្យម", "ខ្សោយ"]))
            else:
                r.append(("អវត្តមាន" if km else "Absent") if absent else (ctx.phrase(4, 14) if rng.random() < 0.1 else ""))
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al)
    if rng.random() < 0.55:
        g1 = (f"ឆមាសទី{ctx.d(rng.randint(1, 2))}" if km else f"Semester {rng.randint(1, 2)}") if rng.random() < 0.6 else (("ពិន្ទុតាមមុខវិជ្ជា" if km else "Subject scores"))
        g2 = "លទ្ធផល" if km else "Results"
        groups = [(3, ns)]
        labels = [g1]
        if rng.random() < 0.6:
            groups.append((3 + ns, 2))
            labels.append(g2)
        T.two_row_header(ctx, spec, labels=labels, groups=groups)
    if rng.random() < 0.5:
        T.add_totals(ctx, spec, label=("មធ្យមភាគថ្នាក់" if km else "Class average"), numeric_cols=list(range(3, 3 + ns + 2)), span_label=True)
    spec["row_header"] = rng.random() < 0.3
    return spec, {"title": "scores", "subtitle": (f"ថ្នាក់ទី{ctx.d(rng.randint(7, 12))}{rng.choice('កខគឃ')}" if km else f"Grade {rng.randint(6, 12)}{rng.choice('ABC')}")}


def data_inventory(ctx, n, wide):
    rng = ctx.rng
    km = ctx.lang == "km"
    cols = [("ល.រ", "#", "c"), ("កូដ", "SKU", "c"), ("ឈ្មោះទំនិញ", "Item", "l"), ("ឯកតា", "Unit", "c"), ("ដើមគ្រា", "Opening", "r"), ("ចូល", "In", "r"),
            ("ចេញ", "Out", "r"), ("នៅសល់", "Closing", "r")]
    if rng.random() < 0.5:
        cols.append(("តម្លៃឯកតា", "Unit cost", "r"))
    if rng.random() < 0.4:
        cols.append(("ផ្សេងៗ", "Remarks", "l"))
    if rng.random() < 0.3:
        cols.pop(1)
    hdr = [_bil(ctx, k, e) for k, e, _ in cols]
    al = [a for _, _, a in cols]
    rows = []
    for i in range(n):
        o, i_, u = rng.randint(0, 500), rng.randint(0, 200), rng.randint(0, 300)
        r = []
        for k, e, _ in cols:
            if k == "ល.រ":
                r.append(ctx.d(i + 1))
            elif k == "កូដ":
                r.append(f"{rng.choice(['SKU', 'IT', 'P', 'M'])}-{rng.randint(1000, 99999)}")
            elif k == "ឈ្មោះទំនិញ":
                r.append(rng.choice(KH_PRODUCTS) if km and rng.random() < 0.8 else rng.choice(PROD_EN))
            elif k == "ឯកតា":
                r.append(rng.choice(UNITS_KM if km and rng.random() < 0.8 else UNITS_EN))
            elif k == "ដើមគ្រា":
                r.append(ctx.d(o))
            elif k == "ចូល":
                r.append(ctx.d(i_) if i_ or rng.random() < 0.5 else "")
            elif k == "ចេញ":
                r.append(ctx.d(u) if u or rng.random() < 0.5 else "")
            elif k == "នៅសល់":
                r.append(ctx.d(max(0, o + i_ - u)))
            elif k == "តម្លៃឯកតា":
                r.append(_amount(ctx, 0.5, 400, cents=True))
            else:
                r.append(ctx.phrase(4, 16) if rng.random() < 0.15 else "")
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al)
    if rng.random() < 0.5:
        T.add_group_rows(ctx, spec, n=rng.randint(1, 4), texts=rng.sample(CAT_KM if km else CAT_EN, 4))
    if rng.random() < 0.35:
        c0 = hdr.index(_bil(ctx, "ដើមគ្រា", "Opening"))
        T.two_row_header(ctx, spec, labels=[("បរិមាណ" if km else "Quantity")], groups=[(c0, 4)])
    if rng.random() < 0.5:
        T.add_totals(ctx, spec)
    return spec, {"title": "inventory"}


def data_ledger(ctx, n, wide):
    rng = ctx.rng
    km = ctx.lang == "km"
    if rng.random() < 0.4:  # payroll
        cols = [("ល.រ", "No.", "c"), ("ឈ្មោះ", "Name", "l"), ("តួនាទី", "Position", "l"), ("ប្រាក់ខែគោល", "Basic salary", "r"), ("ប្រាក់បន្ថែម", "Allowance", "r"),
                ("ការកាត់", "Deductions", "r"), ("ប្រាក់សុទ្ធ", "Net pay", "r")]
        if rng.random() < 0.6:
            cols.append(("ហត្ថលេខា", "Signature", "c"))
        hdr = [_bil(ctx, k, e) for k, e, _ in cols]
        al = [a for _, _, a in cols]
        rows = []
        for i in range(n):
            b = rng.uniform(200, 2500)
            a, d_ = b * rng.uniform(0, 0.3), b * rng.uniform(0, 0.15)
            r = [ctx.d(i + 1), _name(ctx), rng.choice(POS_KM if km else POS_EN), ctx.d(f"{b:,.2f}"), ctx.d(f"{a:,.2f}") if a > 20 else "",
                 ctx.d(f"{d_:,.2f}") if d_ > 10 else "", ctx.d(f"{b + a - d_:,.2f}")]
            if len(cols) == 8:
                r.append("")
            rows.append(r)
        spec = T.spec_from_data(hdr, rows, al)
        if rng.random() < 0.4:
            c0 = 3
            T.two_row_header(ctx, spec, labels=[("ទឹកប្រាក់ (ដុល្លារ)" if km else "Amount (USD)")], groups=[(c0, 4)])
        T.add_totals(ctx, spec, numeric_cols=[3, 4, 5, 6])
        return spec, {"title": "payroll", "sig_col": 7 if len(cols) == 8 else None}
    cash = rng.random() < 0.4
    if cash:
        cols = [("កាលបរិច្ឆេទ", "Date", "c"), ("បរិយាយ", "Particulars", "l"), ("ចំណូល", "Receipts", "r"), ("ចំណាយ", "Payments", "r"), ("សមតុល្យ", "Balance", "r")]
    else:
        cols = [("កាលបរិច្ឆេទ", "Date", "c"), ("លេខយោង", "Ref", "c"), ("បរិយាយ", "Description", "l"), ("ដេប៊ី", "Debit", "r"), ("ក្រេឌីត", "Credit", "r"), ("សមតុល្យ", "Balance", "r")]
    hdr = [_bil(ctx, k, e) for k, e, _ in cols]
    al = [a for _, _, a in cols]
    year = rng.randint(2019, 2027)
    bal = rng.uniform(500, 20000)
    rows = []
    mm, dd = rng.randint(1, 12), 1
    desc = LEDG_KM if km else LEDG_EN
    for i in range(n):
        dd = min(28, dd + rng.randint(0, 3))
        amt = rng.uniform(10, 3000)
        deb = rng.random() < 0.5
        bal = bal + amt if deb == cash else bal - amt
        r = [ctx.d(f"{dd:02d}/{mm:02d}/{year}") if km else f"{mm:02d}/{dd:02d}/{year}"]
        if not cash:
            r.append(f"{rng.choice(['INV', 'RC', 'JV', 'PV'])}-{rng.randint(100, 9999)}")
        r.append(rng.choice(desc) + (f" {ctx.fake.company()[:16]}" if not km and rng.random() < 0.3 else ""))
        r += [ctx.d(f"{amt:,.2f}") if deb else "", "" if deb else ctx.d(f"{amt:,.2f}"), ctx.d(f"{bal:,.2f}")]
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al)
    if rng.random() < 0.3:
        c0 = len(cols) - 3
        T.two_row_header(ctx, spec, labels=[("ទឹកប្រាក់ (៛)" if km and rng.random() < 0.5 else ("ទឹកប្រាក់ ($)" if km else "Amount (USD)"))], groups=[(c0, 3)])
    if rng.random() < 0.6:
        T.add_totals(ctx, spec, numeric_cols=[len(cols) - 3, len(cols) - 2])
    if rng.random() < 0.2:
        T.add_group_rows(ctx, spec, n=rng.randint(1, 2), texts=[f"ខែ{MONTHS[rng.randint(0, 11)]}" if km else ctx.fake.month_name() for _ in range(2)])
    return spec, {"title": "ledger"}


def data_excel(ctx, n, wide):
    """Generic worksheet export: random mix of typed columns."""
    rng = ctx.rng
    km = ctx.lang == "km"
    pool = [("ID", "ID", "c"), ("ឈ្មោះ", "Name", "l"), ("កាលបរិច្ឆេទ", "Date", "c"), ("ចំនួន", "Qty", "r"), ("តម្លៃឯកតា", "Unit price", "r"), ("ទឹកប្រាក់", "Amount", "r"),
            ("ស្ថានភាព", "Status", "c"), ("ប្រភេទ", "Category", "l"), ("ទូរស័ព្ទ", "Phone", "l"), ("អ៊ីមែល", "Email", "l"), ("ចំណាំ", "Notes", "l"), ("ទីតាំង", "Location", "l"),
            ("%", "%", "r"), ("កូដ", "Code", "c"), ("អ្នកទទួលខុសត្រូវ", "Owner", "l")]
    k = rng.randint(4, 8 if wide else 6)
    cols = [("ល.រ", "No.", "c")] + rng.sample(pool, k) if rng.random() < 0.7 else rng.sample(pool, k)
    hdr = [_bil(ctx, kk, e) for kk, e, _ in cols]
    al = [a for _, _, a in cols]
    rows = []
    for i in range(n):
        r = []
        for kk, e, _ in cols:
            if kk == "ល.រ":
                r.append(ctx.d(i + 1))
            elif kk == "ID":
                r.append(f"{rng.choice(['C', 'P', 'O', 'E'])}{rng.randint(1000, 99999)}")
            elif kk == "ឈ្មោះ":
                r.append(_name(ctx, 0.3))
            elif kk == "កាលបរិច្ឆេទ":
                r.append(_date(ctx))
            elif kk == "ចំនួន":
                r.append(ctx.d(rng.randint(1, 500)))
            elif kk in ("តម្លៃឯកតា", "ទឹកប្រាក់"):
                r.append(_amount(ctx, 1, 20000))
            elif kk == "ស្ថានភាព":
                r.append(rng.choice(STATUS_KM if km and rng.random() < 0.7 else STATUS_EN))
            elif kk == "ប្រភេទ":
                r.append(rng.choice(CAT_KM if km else CAT_EN))
            elif kk == "ទូរស័ព្ទ":
                r.append(ctx.phone())
            elif kk == "អ៊ីមែល":
                r.append(ctx.fake.free_email())
            elif kk == "ទីតាំង":
                r.append(rng.choice(KH_PROV if km else KH_PROV_EN + [ctx.fake.city()]))
            elif kk == "%":
                r.append(ctx.d(f"{rng.uniform(0, 100):.1f}%"))
            elif kk == "កូដ":
                r.append(f"{rng.randint(100, 999)}-{rng.randint(10, 99)}")
            elif kk == "អ្នកទទួលខុសត្រូវ":
                r.append(_name(ctx, 0.4))
            else:
                r.append(ctx.phrase(5, 30) if rng.random() < 0.4 else "")
        rows.append(r)
    spec = T.spec_from_data(hdr, rows, al)
    if rng.random() < 0.3:
        T.add_totals(ctx, spec)
    if rng.random() < 0.25:
        T.add_group_rows(ctx, spec, n=rng.randint(1, 3))
    if rng.random() < 0.3:
        T.add_empties(ctx, spec, rng.uniform(0.05, 0.2))
    return spec, {"title": "excel"}


BUILDERS = {"attendance": data_attendance, "price_list": data_price_list, "budget": data_budget, "timetable": data_timetable,
            "scores": data_scores, "inventory": data_inventory, "ledger": data_ledger, "excel": data_excel}


def add_excel_chrome(spec, max_letters=26):
    """Printed worksheet with headings on: a row of column letters above the table and a row-number column at the left."""
    C = spec["ncol"]
    letters = [chr(ord("A") + i) if i < max_letters else f"A{chr(ord('A') + i - max_letters)}" for i in range(C)]
    for i, r in enumerate(spec["rows"]):
        r["cells"] = [T.cell(str(i + 1), align="c", bold=False, max_lines=1)] + r["cells"]
    spec["rows"].insert(0, {"kind": "hdr", "cells": [T.cell("", align="c", header=True, max_lines=1)] + [T.cell(L, align="c", bold=False, header=True, max_lines=1) for L in letters]})
    spec["ncol"] = C + 1
    spec["header_rows"] += 1
    spec["align"] = ["c"] + list(spec["align"])
    if spec.get("col_frac"):
        spec["col_frac"] = [0.6] + list(spec["col_frac"])
    spec["excel_letters"] = True
    return spec


# --------------------------------------------------------------------------- titles
def sheet_title(ctx, kind, info):
    rng = ctx.rng
    km = ctx.lang == "km"
    per = _period(ctx)
    if km:
        t = {"attendance": [f"បញ្ជីវត្តមាន{rng.choice(['បុគ្គលិក', 'សិស្ស', 'មន្ត្រី', 'អ្នកចូលរួម', 'និស្សិត'])}", "បញ្ជីឈ្មោះអ្នកចូលរួម", "បញ្ជីរាយនាមបុគ្គលិក", "បញ្ជីឈ្មោះសិស្ស"],
             "price_list": ["បញ្ជីតម្លៃទំនិញ", "តារាងតម្លៃ", "បញ្ជីតម្លៃលក់", "បញ្ជីមុខម្ហូប"],
             "budget": ["ផែនការថវិកា", "របាយការណ៍ចំណាយ", "តារាងថវិកា", "របាយការណ៍ចំណូល-ចំណាយ"],
             "timetable": ["កាលវិភាគសិក្សា", "កាលវិភាគបង្រៀន", "តារាងម៉ោងសិក្សា"],
             "scores": ["តារាងពិន្ទុសិស្ស", "បញ្ជីពិន្ទុប្រឡង", "លទ្ធផលប្រឡង", "តារាងពិន្ទុប្រចាំឆមាស"],
             "inventory": ["បញ្ជីសារពើភ័ណ្ឌ", "របាយការណ៍ស្តុកទំនិញ", "បញ្ជីទំនិញក្នុងឃ្លាំង"],
             "ledger": ["សៀវភៅបញ្ជី", "បញ្ជីប្រតិបត្តិការ", "សៀវភៅសាច់ប្រាក់"], "payroll": ["តារាងបើកប្រាក់ខែ", "បញ្ជីបើកប្រាក់បៀវត្ស"],
             "excel": ["របាយការណ៍លក់", "បញ្ជីអតិថិជន", "បញ្ជីការបញ្ជាទិញ", "ទិន្នន័យប្រចាំខែ", "តារាងតាមដានគម្រោង", "បញ្ជីការងារ"]}
        base = rng.choice(t.get(info.get("title", kind), t[kind]))
        return f"{base} {per}" if rng.random() < 0.7 else base
    t = {"attendance": [f"{rng.choice(['Staff', 'Student', 'Employee', 'Participant', 'Training'])} Attendance {rng.choice(['Sheet', 'List', 'Register', 'Record'])}", "Sign-in Sheet", "Class Roster"],
         "price_list": ["Price List", "Product Price List", "Wholesale Price List", "Menu", "Rate Card"],
         "budget": ["Budget Summary", "Expense Report", "Budget vs Actual", "Operating Budget", "Income and Expense Statement"],
         "timetable": ["Class Timetable", "Weekly Schedule", "Course Schedule", "Teaching Timetable"],
         "scores": ["Score Sheet", "Grade Report", "Exam Results", "Student Marks"],
         "inventory": ["Inventory List", "Stock Report", "Stock Count Sheet", "Warehouse Inventory"],
         "ledger": ["General Ledger", "Cash Book", "Transaction Register", "Account Statement"], "payroll": ["Payroll Register", "Monthly Payroll", "Salary Sheet"],
         "excel": ["Sales Report", "Customer List", "Order Export", "Monthly Data", "Project Tracker", "Task List", "Vendor List", "Sales by Region"]}
    base = rng.choice(t.get(info.get("title", kind), t[kind]))
    return rng.choice([f"{base} - {per}", f"{base} ({per})", f"{base} {per}", base, base.upper()])


# --------------------------------------------------------------------------- page composer
def sheet_page(rng: random.Random, fake, lang: str, kind: str):
    ctx = Ctx(rng, lang, fake)
    ctx._sheet_hdr_mode = rng.choices(["km", "bil", "en"], [0.6, 0.3, 0.1])[0] if lang == "km" else "en"
    ctx._sheet_mark = rng.choice(["✓", "√", "x", "P", "/", "X"])
    km = lang == "km"
    wide = rng.random() < ({"timetable": 0.55, "scores": 0.6, "attendance": 0.4, "excel": 0.35}.get(kind, 0.2))
    pg = new_page(rng, "a4l" if wide else rng.choice(["a4", "a4", "a4", "letter"]), 0.06)
    pg.meta["doc_kind"] = "sheet"
    pg.meta["degrade_p"] = rng.choice([0.6, 0.65, 0.7])
    W, H = pg.W, pg.H
    m = int(W * rng.uniform(0.045, 0.08))
    x, w = m, W - 2 * m
    excel = kind == "excel" or rng.random() < 0.18
    official = km and not excel and rng.random() < 0.45
    y = int(H * rng.uniform(0.03, 0.06))
    sz_t = ctx.size
    # ---- running header (Excel header / file name / org name)
    if excel and rng.random() < 0.4:
        t = rng.choice([f"{fake.file_name(extension='xlsx')}", f"Sheet{rng.randint(1, 3)}", f"{fake.file_name(extension='xlsx')} - Sheet1", _date(ctx)])
        p = ctx.font("latin", t)
        side = rng.choice(["l", "c", "r"])
        pg.begin("header")
        q = pg.text({"l": x, "c": x + w / 2, "r": x + w}[side], y, t, p, int(sz_t * 0.6), (110, 110, 110), align=side)
        pg.end()
        if q:
            pg.add("header", [q])
            y = bbox([q])[3] + sz_t * 0.8
    if official:
        y = motto(pg, ctx, W / 2, y, size=sz_t * rng.uniform(1.0, 1.2))
        org = rng.choice(UNIS)[0] if rng.random() < 0.4 else rng.choice([mi[0] for mi in MINISTRIES])
        lines = [org] + ([rng.choice(SUB_UNITS)] if rng.random() < 0.6 else [])
        if rng.random() < 0.5:  # org block at the left under the motto, like letters
            qs = []
            yy = y
            for t in lines:
                q = pg.text(x, yy, t, ctx.font("bold" if rng.random() < 0.5 else "moul", t), int(sz_t * 0.95), ctx.fg, bound=(0, 0, x + w * 0.6, H))
                if q:
                    qs.append(q)
                    yy = bbox([q])[3] + sz_t * 0.5
            if qs:
                pg.add("header", qs)
            y = max(y, yy) + sz_t * 0.3
        else:
            for t in lines:
                q = pg.text(W / 2, y, t, ctx.font("moul" if rng.random() < 0.5 else "bold", t), int(sz_t * 1.0), ctx.fg, align="c")
                if q:
                    pg.add("header", [q])
                    y = bbox([q])[3] + sz_t * 0.5
            y += sz_t * 0.4
    elif rng.random() < 0.55:
        org = (rng.choice(KH_COMPANIES) if km else fake.company()) if rng.random() < 0.7 else (rng.choice(UNIS)[0] if km else f"{fake.city()} {rng.choice(['High School', 'University', 'College', 'Academy'])}")
        al = rng.choice(["l", "c"])
        q = pg.text(x if al == "l" else W / 2, y, org, ctx.font("bold", org), int(sz_t * rng.uniform(0.95, 1.2)), ctx.accent if rng.random() < 0.4 else ctx.fg, align=al)
        if q:
            pg.add("header", [q])
            y = bbox([q])[3] + sz_t * 0.6
            if rng.random() < 0.3:
                pg.d.line([x, y, x + w, y], fill=ctx.accent, width=rng.choice([1, 2]))
                y += sz_t * 0.5
    # ---- data
    nrow = rng.randint(6, 10) if kind == "timetable" else rng.randint(15, 45)
    spec, info = BUILDERS[kind](ctx, nrow, wide)
    if excel:
        add_excel_chrome(spec)
    # ---- title + subtitle lines
    title = sheet_title(ctx, kind, info)
    tal = rng.choice(["c", "c", "l"]) if not excel else rng.choice(["l", "l", "c"])
    y = B.b_heading(pg, ctx, x, y, w, cls="title", text=title, size=sz_t * rng.uniform(1.25, 1.8), align=tal, role="moul" if km and rng.random() < 0.5 else "title", max_lines=2)
    subs = []
    if info.get("subtitle"):
        subs.append(info["subtitle"])
    if rng.random() < 0.5:
        subs.append((f"ផ្នែក៖ {rng.choice(DEPT_KM)}" if km else f"Department: {rng.choice(DEPT_EN)}") if rng.random() < 0.5 else
                    (f"កាលបរិច្ឆេទ៖ {ctx.date()}" if km else f"Date: {ctx.date()}"))
    if rng.random() < 0.3:
        subs.append(f"រៀបចំដោយ៖ {ctx.name()}" if km else f"Prepared by: {fake.name()}")
    for t in subs[:3]:
        q = pg.text(x if tal == "l" else W / 2, y, t, ctx.font("body", t), int(sz_t * 0.9), ctx.fg, align=tal)
        if q:
            pg.add("text", [q])
            y = bbox([q])[3] + sz_t * 0.45
    y += sz_t * rng.uniform(0.3, 0.8)
    # ---- table: size the font to the rows that should fit
    foot_h = sz_t * 1.6
    sig_plan = rng.random() < 0.45 and kind != "timetable"
    reserve = sz_t * (6.5 if sig_plan else 0.5)
    ymax = H - int(H * rng.uniform(0.035, 0.06)) - foot_h
    R = len(spec["rows"])
    avail = ymax - reserve - y
    per_row = avail / (R + 0.5)
    sz = int(min(ctx.size * rng.uniform(0.75, 0.95), max(15 if km else 13, per_row / (2.0 if km else 1.7))))
    name = rng.choice(["grid"] * 5 + ["hlines"] * 2 + ["zebra", "thick_hdr", "dark_hdr", "outer", "none"]) if not info.get("style") else info["style"]
    st = T.choose_style(rng, name, sheet=True, excel=excel, ncol=spec["ncol"])
    if kind in ("budget", "ledger", "scores") and st["rules"] != "grid":
        st["double_total"] = rng.random() < 0.6
    n_tab0 = len(pg.meta.get("tables", []))
    y_tab = y
    y = B.b_table(pg, ctx, x, y, w, ymax - (reserve if sig_plan else 0), spec=spec, sheet=True, tstyle=st, size=sz, min_body=3)
    lab = pg.meta["tables"][-1] if len(pg.meta.get("tables", [])) > n_tab0 else None
    if lab is None:
        return pg, ctx
    # ---- signatures inside the signature column (attendance / payroll)
    sc = info.get("sig_col")
    if sc is not None and excel:
        sc += 1
    if sc is not None and rng.random() < 0.7:
        for c in lab["cells"]:
            if c["c"] == sc and c["kind"] == "body" and rng.random() < 0.65:
                bx = c["box"]
                hh, ww = bx[3] - bx[1], bx[2] - bx[0]
                if hh < 14 or ww < 30:
                    continue
                sig = B.make_signature(ctx)
                s = min((hh * rng.uniform(0.9, 1.4)) / sig.height, (ww * 0.85) / sig.width)
                sig = sig.resize((max(6, int(sig.width * s)), max(6, int(sig.height * s))))
                pg.ink(sig, (bx[0] + rng.uniform(0.05, 0.4) * (ww - sig.width), bx[1] + (hh - sig.height) / 2 + rng.uniform(-0.15, 0.15) * hh), "signature")
    # ---- below the table: signature lines / stamp / note
    if sig_plan and y < ymax - sz_t * 5:
        nsig = rng.choice([1, 2, 2, 3])
        cw = w / nsig
        yy = y + sz_t * rng.uniform(0.3, 0.9)
        roles_km = ["រៀបចំដោយ", "ពិនិត្យដោយ", "អនុម័តដោយ", "បានឃើញ និងឯកភាព", "គ្រូប្រចាំថ្នាក់", "នាយកសាលា", "ប្រធានផ្នែក", "អ្នកទទួល"]
        roles_en = ["Prepared by", "Checked by", "Approved by", "Received by", "Class teacher", "Principal", "Department head", "Verified by"]
        rs = rng.sample(roles_km if km else roles_en, nsig)
        for i in range(nsig):
            B.b_sign(pg, ctx, x + i * cw + cw * 0.1, yy, cw * 0.8, ymax, role=rs[i], date=(i == nsig - 1 and rng.random() < 0.5), stamp_p=0.25 if i == nsig - 1 else 0.05)
    elif rng.random() < 0.25 and y < ymax - sz_t * 2:
        t = rng.choice(["សម្គាល់៖ " + ctx.phrase(10, 60), "ចំណាំ៖ " + ctx.phrase(10, 50)] if km else ["Note: " + ctx.phrase(10, 60), "* " + ctx.phrase(10, 50), "Figures in USD unless stated otherwise."])
        B.b_para(pg, ctx, x, y + sz_t * 0.3, w, ymax, n=2, text=t, size=sz_t * 0.8, indent=0)
    elif rng.random() < 0.2:
        B.b_stamp_at(pg, ctx, x + w * rng.uniform(0.6, 0.85), y + sz_t * rng.uniform(1.0, 3.0), d=sz_t * rng.uniform(5, 7))
    # ---- footer: page number / print date / sheet name
    fy = H - int(H * rng.uniform(0.025, 0.045))
    if rng.random() < 0.85:
        n_, tot = rng.randint(1, 4), rng.randint(1, 6)
        tot = max(tot, n_)
        t = rng.choice([f"Page {n_} of {tot}", f"Page {n_}", f"{n_}/{tot}", f"- {n_} -"] if not km or rng.random() < 0.5 else [f"ទំព័រ {ctx.d(n_)}/{ctx.d(tot)}", f"ទំព័រទី{ctx.d(n_)}", f"{ctx.d(n_)}"])
        side = rng.choice(["c", "r", "r", "l"])
        pg.begin("page_number")
        q = pg.text({"c": x + w / 2, "r": x + w, "l": x}[side], fy, t, ctx.font("body", t), int(sz_t * 0.72), (80, 80, 80), align=side)
        pg.end()
        if q:
            pg.add("page_number", [q])
        if rng.random() < 0.5:
            t2 = rng.choice([f"{fake.file_name(extension='xlsx')}", f"Printed {ctx.date()}", _date(ctx), f"Sheet{rng.randint(1, 3)}", (rng.choice(KH_COMPANIES) if km else fake.company())])
            side2 = "l" if side != "l" else "r"
            pg.begin("footer")
            q2 = pg.text(x if side2 == "l" else x + w, fy, t2, ctx.font("body", t2), int(sz_t * 0.68), (110, 110, 110), align=side2)
            pg.end()
            if q2 and q and not B.overlaps(q, q2):
                pg.add("footer", [q2])
    return pg, ctx


def _mk(lang, kind):
    def f(rng, fake):
        return sheet_page(rng, fake, lang, kind)
    f.__name__ = f"{lang}_sheet_{kind}"
    return f


SHEET_KM = {
    "km_sheet_excel": (_mk("km", "excel"), 1.5),
    "km_attendance": (_mk("km", "attendance"), 1.3),
    "km_price_list": (_mk("km", "price_list"), 0.8),
    "km_budget": (_mk("km", "budget"), 1.0),
    "km_timetable": (_mk("km", "timetable"), 0.8),
    "km_score_sheet": (_mk("km", "scores"), 1.0),
    "km_inventory": (_mk("km", "inventory"), 0.8),
    "km_ledger": (_mk("km", "ledger"), 0.8),
}
SHEET_EN = {
    "en_sheet_excel": (_mk("en", "excel"), 1.5),
    "en_attendance": (_mk("en", "attendance"), 0.8),
    "en_price_list": (_mk("en", "price_list"), 0.7),
    "en_budget": (_mk("en", "budget"), 0.8),
    "en_timetable": (_mk("en", "timetable"), 0.6),
    "en_score_sheet": (_mk("en", "scores"), 0.6),
    "en_inventory": (_mk("en", "inventory"), 0.7),
    "en_ledger": (_mk("en", "ledger"), 0.7),
}
