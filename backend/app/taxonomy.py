"""Layout taxonomy v2, vendored from training/taxonomy.py so the backend never imports training/.

Keep CLASSES / TEXT_CLASSES / COLORS identical to training/taxonomy.py (tests/test_taxonomy.py checks it when
training/ is present). Order = class id. Never reorder; only append.

Besides the v2 list this module maps the class names of older models onto v2, so the 12-class v1 layout model
(det/common.py: header_footer, qr_barcode) and DocLayout-YOLO DocLayNet/DocStructBench names keep working.
"""

from __future__ import annotations

CLASSES = [
    "text",            # 0  body paragraph / text block
    "title",           # 1  document title
    "section_header",  # 2  section / sub-section heading
    "list",            # 3  bulleted or numbered list
    "table",           # 4  table (whole grid incl. header row)
    "figure",          # 5  picture, chart, diagram, map, illustration
    "caption",         # 6  figure / table caption
    "header",          # 7  running page header, letterhead text, document reference line at the top
    "footer",          # 8  running page footer
    "page_number",     # 9  standalone page number
    "footnote",        # 10 footnote / endnote
    "formula",         # 11 math formula
    "signature",       # 12 handwritten signature or initials
    "stamp",           # 13 seal / official ink stamp
    "logo",            # 14 logo, emblem, coat of arms
    "qr_code",         # 15 QR / 2D code
    "barcode",         # 16 1D barcode
    "photo",           # 17 portrait / ID photo of a person
    "handwriting",     # 18 handwritten text (filled form values, notes, dates)
    "checkbox",        # 19 checkbox or radio button (ticked or empty)
    "form_field",      # 20 fill-in field: label + its input box/line/value ("Name: ______")
    "decoration",      # 21 ornament / divider / rule drawn between parts of the page (not text; pipeline heuristic, not predicted yet)
]
CLS = {c: i for i, c in enumerate(CLASSES)}

# Which classes carry text that the recogniser should read (printed, and handwriting since v2).
TEXT_CLASSES = {"text", "title", "section_header", "list", "caption", "header", "footer",
                "page_number", "footnote", "table", "form_field",
                "handwriting"}  # v2: the recogniser is trained on handwritten lines (exam sheets, filled forms)
# Everything else is returned as an image crop (formula included: the Khmer CRNN cannot read maths).
CROP_CLASSES = set(CLASSES) - TEXT_CLASSES

COLORS = {  # RGB, used by previews and the frontend
    "text": (60, 120, 216), "title": (230, 120, 20), "section_header": (240, 170, 40), "list": (110, 80, 200),
    "table": (30, 160, 110), "figure": (200, 60, 160), "caption": (150, 110, 60), "header": (120, 140, 160),
    "footer": (100, 120, 140), "page_number": (80, 80, 120), "footnote": (140, 140, 90), "formula": (130, 60, 60),
    "signature": (220, 30, 30), "stamp": (180, 20, 90), "logo": (20, 150, 190), "qr_code": (90, 90, 90),
    "barcode": (50, 50, 50), "photo": (255, 100, 100), "handwriting": (200, 150, 0), "checkbox": (0, 180, 200),
    "form_field": (160, 200, 60), "decoration": (150, 150, 150),
}

LABELS = {c: c.replace("_", " ").capitalize() for c in CLASSES}
LABELS.update(qr_code="QR code", page_number="Page number", form_field="Form field", section_header="Section header")

# --------------------------------------------------------------------------- legacy names -> v2
# Names whose v2 class depends on geometry are resolved in resolve_class().
_ALIASES = {
    # v1 12-class model (Khm-Dataset-Creation det/common.py)
    "header_footer": "@header_footer", "qr_barcode": "@qr_barcode",
    # DocLayout-YOLO (DocStructBench) names
    "plain text": "text", "plain_text": "text", "abandon": "@header_footer", "figure_caption": "caption",
    "table_caption": "caption", "table_footnote": "footnote", "isolate_formula": "formula", "formula_caption": "caption",
    # DocLayNet names
    "picture": "figure", "list-item": "list", "list_item": "list", "page-header": "header", "page_header": "header",
    "page-footer": "footer", "page_footer": "footer", "section-header": "section_header",
}

# Legacy frontend "type" values (the v1 UI knows title/paragraph/header/caption/image).
LEGACY_TYPE = {
    "title": "title", "section_header": "title",
    "header": "header", "footer": "header", "page_number": "header",
    "caption": "caption",
}
LEGACY_TYPE_TO_CLASS = {"paragraph": "text", "title": "title", "header": "header", "caption": "caption", "image": "figure"}


def resolve_class(name: str, box=None, page_w: float = 0, page_h: float = 0) -> str:
    """Model class name (any taxonomy) -> v2 class name. ``box`` = [x0,y0,x1,y1] for the geometry-dependent aliases."""
    n = str(name).strip().lower()
    if n in CLS:
        return n
    alias = _ALIASES.get(n, _ALIASES.get(n.replace(" ", "_"), "text"))
    if not alias.startswith("@"):
        return alias
    if box is None:
        return {"@header_footer": "header", "@qr_barcode": "qr_code"}[alias]
    x0, y0, x1, y1 = box
    if alias == "@header_footer":
        return "header" if page_h <= 0 or (y0 + y1) / 2 < page_h / 2 else "footer"
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    return "barcode" if (w / h > 1.8 or h / w > 1.8) else "qr_code"


def legacy_type(cls: str) -> str:
    """v2 class -> the v1 frontend block type."""
    if cls in CROP_CLASSES:
        return "image"
    return LEGACY_TYPE.get(cls, "paragraph")


def class_from_request(t: str) -> str:
    """Block type sent by a client: a v2 class name or a legacy v1 UI type."""
    t = str(t or "").strip().lower()
    if t in CLS:
        return t
    return LEGACY_TYPE_TO_CLASS.get(t, resolve_class(t))
