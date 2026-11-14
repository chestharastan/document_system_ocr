"""Layout taxonomy v2 - the single source of truth for class ids.

Shared by the dataset builders (training/), the trainers, the ONNX exports and the backend (backend/app/).
Order = class id. Never reorder; only append.
"""

CLASSES = [
    "text",            # 0  body paragraph / text block
    "title",           # 1  document title
    "section_header",  # 2  section / sub-section heading
    "list",            # 3  bulleted or numbered list (whole list, or one list item from DocLayNet)
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

# DocLayNet (CDLA-Permissive-1.0) category -> v2 class
DOCLAYNET_MAP = {
    "Caption": "caption", "Footnote": "footnote", "Formula": "formula", "List-item": "list",
    "Page-footer": "footer", "Page-header": "header", "Picture": "figure",
    "Section-header": "section_header", "Table": "table", "Text": "text", "Title": "title",
}

COLORS = {  # RGB, used by previews and the frontend
    "text": (60, 120, 216), "title": (230, 120, 20), "section_header": (240, 170, 40), "list": (110, 80, 200),
    "table": (30, 160, 110), "figure": (200, 60, 160), "caption": (150, 110, 60), "header": (120, 140, 160),
    "footer": (100, 120, 140), "page_number": (80, 80, 120), "footnote": (140, 140, 90), "formula": (130, 60, 60),
    "signature": (220, 30, 30), "stamp": (180, 20, 90), "logo": (20, 150, 190), "qr_code": (90, 90, 90),
    "barcode": (50, 50, 50), "photo": (255, 100, 100), "handwriting": (200, 150, 0), "checkbox": (0, 180, 200),
    "form_field": (160, 200, 60), "decoration": (150, 150, 150),
}
