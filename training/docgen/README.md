# docgen v2 — synthetic Cambodian + US/English document pages

```
PY=/home/hello/ocr/khm_repo_clone/det_venv/bin/python   # run from the repo root, CPU only
$PY -m training.docgen.generate --out DIR --n N --seed S --workers 8 [--lang-mix P] [--degrade-p 0.6]
$PY -m training.docgen.preview --manifest DIR/manifest.jsonl --out PREV --n 40 [--doc-type km_invoice]
$PY -m training.docgen.stats --manifest DIR/manifest.jsonl
$PY -m training.docgen.check_fonts --manifest DIR/manifest.jsonl     # fontTools cmap + HarfBuzz check of every line
```
`--lang-mix P` = share of pages from the Cambodian families (default 0.5). `--doc-types A,B` restricts generation to the
named doc types (then `--lang-mix` splits the Khmer / English ones). One-time setup (already done): fonts in
work/data/docgen_fonts, English sentences built by `prep_text.py` (see training/DATA_SOURCES.md).

## manifest.jsonl record
`image, w, h, source="docgen", doc_type, lang (km|en|mixed), lines (printed-text quads, tight ink), line_texts (same order),
line_fonts (font file per line), hw_lines + hw_texts (handwriting), formula_lines (rendered maths, no text), objs
[[cls, x0,y0,x1,y1]] (taxonomy v2 ids), lines_ok, objs_ok, seed, aug {degrade, mode, effects, H (3x3 page->image)}, [watermark]`.
Watermarks are drawn very light and are intentionally not labelled.

## labelling rules
text = paragraph block; title = document title; section_header = headings; list = whole list block; table = whole grid
(or text extent for rule-less tables, thermal-receipt item rows); caption; header/footer = running header/footer / letterhead /
kingdom-motto block / reference line; page_number = number alone; footnote = one box per footnote; formula = equation
(+ its number); form_field = label + input region (+ value), incl. ID-card label/value pairs and checkbox groups;
checkbox = box glyph + tick only; handwriting = each handwritten value; signature/stamp = ink hull; logo = emblem
(+ wordmark if beside it); photo = synthetic portrait; qr_code also covers PDF417-like 2D codes; barcode = bars only
(digits below are a separate text box). Object boxes come from warped ink point sets, so they stay tight after warps.

## exam answer sheets (templates_exam.py)
`km_exam_sheet` (Khmer national-exam answer sheet: printed candidate form + "ចម្លើយ", or continuation page with an X-ed box)
and `en_exam_worksheet` (US school worksheet / notebook paper), both fully handwritten on rules with red teacher marks.
They set `meta.degrade_profile = "exam_phone"` (degrade2.apply_exam_phone: close-up 3:4 phone photo, mild perspective,
show-through, grey-blue paper, shadows, defocus, sharpening, JPEG; the framing never half-cuts a labelled line), always
degraded, final long side 1400-1800. Labelling decisions are in the module docstring (handwritten answer block = one
`handwriting` box, handwritten lists are `handwriting`, red ticks/crosses unlabelled, red scores/comments = handwriting,
X-ed box unlabelled). Built set: work/data/docgen_exam/{train,val,test}
(`--n 3000/150/150 --seed 101/202/303 --lang-mix 0.85 --doc-types km_exam_sheet,en_exam_worksheet`).
Note: handwritten lines are in `hw_lines`/`hw_texts` (as everywhere in docgen); `training/data/line_crops.py` only reads
`lines`, so recogniser crops of handwriting need an opt-in there.
