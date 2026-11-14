# Data sources and licences

## docgen (synthetic page generator, training/docgen) — added by the synthetic-data engineer
Generated pages are fully synthetic. External inputs:
- **English text**: 19 public-domain Project Gutenberg books (ids 1342 1404 3300 1661 20203 205 147 2701 1228 84 98 4300 74 1232 345 2600 3207 15399 46),
  downloaded to work/data/docgen_text/raw (16 MB), Gutenberg header/footer licence boilerplate stripped -> en_sentences.txt.gz (3.4 MB).
  Public domain in the US; "Project Gutenberg" trademark not used in outputs.
- **Faker** (MIT) en_US: synthetic names, addresses, companies, phones, emails, dates. Khmer names = random combinations of
  common family/given name parts (ctx.py); ID/passport/account numbers random digits. No real person data.
- **Khmer text**: old repo corpus Khm-Dataset-Creation/data/corpus (train split only), same source as the v1 generator.
- **Fonts (Latin, system)**: Noto Sans/Serif/Serif Display/Sans Mono/Sans Math (OFL 1.1), Liberation Sans/Serif/Mono (OFL 1.1),
  Adwaita Sans/Mono (OFL 1.1), TikTok Sans variable (OFL 1.1; static instances Condensed/SemiCondensed/Regular/Bold/
  CondensedBold/Expanded made with fontTools.varLib.instancer -> work/data/docgen_fonts/inst).
  Coverage: serif, sans, mono, condensed, bold, italic, bold-italic, display.
- **Fonts (Latin handwriting, Google Fonts repo)**: Caveat, Kalam, Patrick Hand, Indie Flower, Shadows Into Light, Dancing Script,
  Reenie Beanie, Nothing You Could Do, Gochi Hand, Architects Daughter, Sacramento (OFL 1.1); Homemade Apple (Apache-2.0).
  -> work/data/docgen_fonts/hand.
- **Fonts (Khmer)**: repo fonts/print + fonts/display + fonts/handwriting (OFL 1.1), Noto Sans/Serif Khmer (OFL 1.1),
  Khmer OS / Khmer OS Battambang/Content/Siemreap/System/Fasthand/Freehand (LGPL 2.1). NOT used: Khmer UI (Microsoft EULA),
  Khmer Savuth Pen (no licence).
- **Formula fonts**: matplotlib mathtext built-ins (DejaVu: Bitstream Vera licence; STIX: OFL; Computer Modern/BaKoMa: free licence).
- **Signatures/stamps**: det_data/assets/{signatures,stamps} (SignverOD Apache-2.0 / mapo80 stamps CC-BY-4.0, as in v1) + procedural.
- **Figures**: procedural charts/diagrams/maps + 34 scene pictures from det_data/assets/photos hand-picked to contain NO people,
  logos or document scans (list in docgen/assets2.py SCENES). Those come from the v1 Khmer PDF crawl (government/NGO PDFs);
  licence of the individual pictures not verified -> treat as research use.
- **Photo class (portraits)**: only procedurally drawn silhouettes (assets2.make_portrait); no real faces.
- **Emblems/logos**: procedural shapes only (no real coat of arms or company logo).

## External real datasets (training/data/ext_*.py) — added by the data-acquisition engineer, 2026-11-01
Canonical page manifests under work/data/ext/<name>/<split>/ (training-allowed) and work/data/eval_nc/<name>/<split>/
(NON-COMMERCIAL, **evaluation only**, every record carries `"nc": true`). JPEG q88, long side <= 1600. Raw downloads deleted.
Converter docstrings hold the full label-mapping tables. Previews: work/reviews/ext_previews/<name>/.

| Name | URL | Licence (quoted) | NC | Used for | Counts | Disk |
|---|---|---|---|---|---|---|
| **D4LA** (form, invoice, letter, memo only) | https://modelscope.cn/datasets/damo/D4LA ; files read from https://huggingface.co/datasets/tuandunghcmut/D4LA (verified identical to https://huggingface.co/datasets/juliozhao/D4LA: images, splits, boxes, ids) | ModelScope API `"License":"Apache License 2.0"`; parquet mirror tag `cdla-permissive-2.0` (mirror claim). Page images are IIT-CDIP / Legacy Tobacco Documents (UCSF), third-party copyright on content | no | layout training/eval (real US scans): text, title, section_header, list, table, form_field, header, footer, page_number, caption, logo (top-of-page Figures only) | train 2,834 / test 711 pages; 37,066 + 8,459 boxes; 158 + 49 `ignore_boxes` | 838 MB |
| **CommonForms** shard `data/valid-00000-of-00012.parquet` | https://huggingface.co/datasets/jbarrow/CommonForms | card: `license: apache-2.0` (pages are Common Crawl PDFs, third-party content) | no | **checkbox** only (partial labels, `partial_classes`/`eval_classes` [19]); input/signature widgets kept as `input_boxes`/`sig_fields` | 1,517 pages, 5,374 checkboxes (411 pages have >= 1), 14,689 input boxes, 90 sig fields | 390 MB |
| **CORD v2** | https://huggingface.co/datasets/naver-clova-ix/cord-v2 ; https://github.com/clovaai/cord | "This work is licensed under a Creative Commons Attribution 4.0 International License." (GitHub); HF tag `cc-by-4.0` | no | receipt text lines + transcriptions (recognition crops, line detection with ignore), form_field = key+value line | train 800 / validation 100 / test 100; 17,777 lines; 4,274 form_field | 148 MB |
| **nutrientdocs form-field-v1-benchmark**, row groups 0,30,60,90,118 | https://huggingface.co/datasets/nutrientdocs/form-field-v1-benchmark | "CC-BY-4.0. Attribute to Nutrient. Empty pages are real blank forms; filled and handwritten values are synthetic. No PII." (all 214 rows: `license_class` = CC-BY-4.0) | no | checkbox eval (partial, [19]); `input_boxes` for containment-style form_field eval | 214 pages (150 empty / 50 filled / 14 handwritten), 1,347 checkboxes, 2,695 input boxes, 210 sig fields | 54 MB |
| **FUNSD** test split | https://guillaumejaume.github.io/FUNSD/ (dataset.zip) | Terms of Use: "Use of the FUNSD Dataset is solely for non-commercial, research and educational purposes." | **YES, eval only** | English scanned forms: exhaustive text lines + transcriptions, form_field (question + linked answers) | 50 pages, 2,649 lines, 600 form_field, 477 unlinked questions (listed, not boxed) | 6 MB |
| **KhmerEduDoc** test_setA + test_setB | https://huggingface.co/datasets/CADT-IDRI/KhmerEduDoc | card: `license: cc-by-nc-4.0` | **YES, eval only** | real Khmer textbook pages: end-to-end page CER against human-verified `page_text` (no boxes exist) | setA 60 + setB 100 pages | 33 MB |

Notes:
- D4LA class ids follow DocLayout-YOLO's d4la.yaml names; the tuandunghcmut card's name list is wrong for its ids (verified visually).
  D4LA "LetterHead" = inside address (mapped to text), "RegionList" = mostly borderless tables (mapped to table), "Number" = vertical
  Bates id (mapped to footer). Unlabelled on D4LA pages: signature, stamp, handwriting, checkbox, most logos, barcode.
- CommonForms/nutrientdocs annotate only fillable PDF widgets: printed checkboxes that are not widgets are unboxed. Some widget rects
  are looser than the printed square, and an invisible widget can have no visible box.

## user_docs (the owner's own documents, camera card)
- Source: photos the owner copies to the `Image/` folder of the SD card labelled `EOS_DIGITAL`; imported read-only by
  `training/data/import_flash.py` into `work/data/user_docs/` (dedup by SHA-1; fixed 25% `test` split, rest `pool`).
- Content (2026-11-01): 46 iPhone photos of handwritten Khmer school exam answer sheets with teacher marks.
- Private: contains real people's names/handwriting. Never publish, never commit, never use as synthetic identities.

## Real Khmer handwriting lines (added 2026-11-08, recogniser v2 fine-tune)
| Source | Where | Licence | Used for | Local |
|---|---|---|---|---|
| **Thareah/khmer-handwriting** (2,891 lines from 147 scanned pages, split by page) | https://huggingface.co/datasets/Thareah/khmer-handwriting | MIT (dataset card) | train (x8 oversampled) + val/test held out | work/data/ext/khm_hand_thareah -> work/data/rec_hand_real_thareah |
| **SoyVitou/khmer-handwritten-dataset-4.2k** (4,202 word/short-line snippets, Khmer + some English) | https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k | MIT (dataset card) | train (x4) + val/test held out | work/data/ext/khm_hand_soyvitou -> work/data/rec_hand_real_soyvitou |
