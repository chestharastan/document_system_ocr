# Khmer Document OCR

Document AI for Cambodian (Khmer) and English documents: a Next.js frontend, a FastAPI backend and small ONNX models
that run on CPU (CUDA if available) and are meant to run on phones later.

```text
image ─► 1 layout (YOLO, 21-class taxonomy) ─► 2 text lines (DB) ─► 3 recognition (CRNN-CTC) ─► 4 post-correction (hook) ─► reading order
```

## Project structure

```text
document_system_ocr/
+-- .model/                     # local model files, NOT in git (see "Models")
|   +-- v2/models.json          # manifest: one entry per model file
|   +-- v2/{layout,textdet,rec}/
+-- training/taxonomy.py        # layout classes - single source of truth (vendored into backend/app/taxonomy.py)
+-- backend/
|   +-- requirements.txt        # ONNX-only runtime (no torch)
|   +-- requirements-torch.txt  # optional legacy PyTorch models + YOLO->ONNX export
|   +-- models.example.json     # copy of the manifest used during development
|   +-- scripts/register_model.py
|   +-- tests/                  # pytest smoke + unit tests
|   +-- app/
|       +-- main.py, config.py, taxonomy.py
|       +-- routers/            # models.py (model + pipeline info), pipeline.py (/process ...)
|       +-- services/
|           +-- model_registry.py   # manifest -> ModelSpec -> stage implementation
|           +-- model_pool.py       # active model per stage, lazy loading, switching
|           +-- pipeline_service.py # the pipeline logic
|           +-- reading_order.py    # XY-cut
|           +-- stages/             # base.py (interfaces), *_onnx.py, legacy_torch.py, postcorrect.py
+-- frontend/                   # Next.js 14 (components/OcrApp.js, lib/api.js)
```

## Pipeline

| stage | interface (`services/stages/base.py`) | default implementation | notes |
|---|---|---|---|
| layout | `LayoutDetector.detect(image) -> [Region(cls, label, score, box)]` | `yolo_onnx`: ultralytics YOLO exported to ONNX, letterbox + class-aware NMS in numpy; default `layout_yolo11n_v3_22cls` (YOLO11n, 10.9 MB, 22 classes incl. `decoration`) | class names read from the ONNX metadata `names`, mapped onto the v2 taxonomy (v1 `header_footer` -> header/footer by position, `qr_barcode` -> qr_code/barcode by aspect) |
| textdet | `TextLineDetector.detect(image, **kw) -> [(quad, score)]` | `db_onnx`: DB / PP-OCR det, ImageNet mean/std, long side 1600..2200 (multiple of 32), threshold 0.3, box_thresh 0.5, unclip 1.6 (pyclipper) | page pass, then re-detection inside text regions the page pass missed (det/cascade.py) |
| rec | `LineRecognizer.recognize(crops) -> [(text, confidence)]` | `crnn_onnx`: CRNN-CTC, height 48, width-bucketed batches, greedy CTC, confidence = mean max-prob | |
| font | `FontClassifier.classify(crops) -> [{family, weight, slant, confidence, em_px}]` | `font_cnn_onnx`: 0.48 M-param CNN on 32x256 gray windows (training/font), 29 families: Khmer OS + Google Khmer + Noto Khmer + Latin serif/sans | optional (skipped when no font model is registered or `DETECT_FONTS=false`); printed lines only; one typeface per block by width-weighted vote |
| structure | `StructureClassifier.predict(lines, W, H) -> [{role, first, level, bullet, order}]` | `structure_onnx`: 0.98 M-param Transformer over the page's text lines (training/structure), 4 MB ONNX | optional; sets block roles and reading order, adds paragraph starts the rules missed; rules remain the fallback |
| postcorrect | `PostCorrector.correct(lines, context) -> lines` | `IdentityPostCorrector` (`POSTCORRECT=none`) | future stage; contract in `base.py` |

Pipeline rules (`pipeline_service.py`): a `figure` covering > 50 % of the page is ignored (layout mistake on dense
tables); near-identical boxes of different classes are de-duplicated; every line goes to the smallest region containing
its centre; lines outside every region are grouped into extra `text` blocks (`source: "orphan"`); only `TEXT_CLASSES`
blocks are recognised (tables: the lines inside them), all other classes (figure, photo, signature, stamp, logo,
qr_code, barcode, checkbox, handwriting, formula) get an image crop; blocks are ordered with a column-aware XY-cut.

Document structure (`services/doc_structure.py`, `services/page_structure.py`, all heuristics on the page's ink,
each switchable in `config.py`): list bullets the line detector drops ("-", "o") are found left of each line and
attached to it (`DETECT_BULLETS`); lines are grouped into paragraphs and the paragraph starts of a text column give
its tab stops (`indent_level`, 0 = margin); figure / logo / stamp boxes are grown to the connected ink they cut and
clamped against neighbouring text (`SNAP_CROPS`); an ornament / divider read as text (mirror-symmetric ink, low
confidence or tiny) becomes a `decoration` crop (`DETECT_DECORATIONS`); layout `text` blocks above the body become
`header`, a centred short block right above it `title`, blocks below the body `footer` / `page_number`, a small
square figure at the top `logo` (`RELABEL_BY_POSITION`, the model's class is kept in `relabel`); every block gets one
font size (`font_px`: the font model v2 predicts each line's em size from its crop, the block takes the median, snapped to the page's size levels; `font_src` says "model" or "ink" for the consonant-body fallback) and lines a `baseline`.
Red ink stamps are found by colour (`services/stamps.py`, `DETECT_STAMPS`): a round saturated-red blob becomes a
`stamp` block (its crop is the red ink alone), lines under it stay with their paragraph, are detected again and
recognised on a copy of the page where the stamp is faded out (per stamp, the red channel on the red mask is re-stretched with an Otsu split so stamp-only pixels turn white and the text under it stays dark; the plain red channel left photo-grade stamps as a grey ring). A `figure` or `logo` that holds
detected text lines (an emblem with the title under it, a footer, a loose box around two logos and the letterhead
between them) is split (`SPLIT_FIGURES`): the pictures are the ink clusters that touch no text line (an emblem whose
ribbon touches the title is clipped at the title), the box shrinks to the largest one, further clusters become blocks
of their own, duplicates of other picture blocks are dropped and the text lines leave the block; a box that is
mostly non-paper (a photo, a coloured seal) is left alone. Text around every picture is detected again at the
picture's own scale (a display-font title under an emblem is missed at page scale). Horizontal rule lines outside
every block become `decoration` blocks (pages with more than 3 such lines - ruled paper, tables - are skipped). Text
line boxes are grown to the glyph ink they cut through (`SNAP_LINES`: at most one line height sideways, 0.35 up /
down; underlines and ink shared by two lines are ignored) so a cut first / last glyph is still recognised.
Bold words (`services/bold.py`, `DETECT_BOLD`; with the font stage present and `BOLD_MODEL` on, each word crop is also
judged by the font model's regular/bold head and is bold when the model is sure or when it leans bold and the strokes
are thicker and darker than the page's typical word): the CTC recogniser reports where each character was emitted, so a
line splits into words with pixel extents; a word whose ink survives a small erosion clearly better than the page's
typical word (and is darker) is bold. Lines carry `spans[{text, x0, x1, bold}]` and `bold`, blocks `bold`.
Ink colour (`DETECT_COLORS`): the median of the darkest ink pixels of each line / word / block as `text_color`
(hex; neutral dark greys are reported as black ink). The demo draws every line in its own detected typeface
(block typeface as fallback), in its colour, bold words in bold - except for display faces (Muol, Koulen ...),
which are heavy by design.

## Structured document (`document` in the response)

`services/document.py` turns the blocks into what a word processor holds: elements in reading order - heading (1
title, 2 section header, 3 short bold line), paragraph (`first_line_indent` for the Khmer indented first line,
`level`), list (bulleted paragraphs merged, one bullet per level), header, footer (`footer_row` with columns when
footer blocks share a row), page_number, figure (crop + width fraction), decoration - each with alignment, typeface,
size, colour and bold runs. No photo coordinates: `static*/docview.js` reflows it with the page margins (the default
right pane in both demos), and `/export/docx` writes the same structure to Word. The overlay at photo positions is
still available as a view.

## Demo pages

`/demo/` serves the folder in `DEMO_DIR` (default `backend/app/static`, the classic one-page demo). The KhmerOCR app
lives in `backend/app/static_new`; run a second backend with `DEMO_DIR=…/app/static_new PORT=8012`
(`work/run_app_8012.sh`) to serve it on its own port while it is being improved.

### KhmerOCR app (`static_new`)

`backend/app/static_new/` is a single-page app following the Figma design "Khmer Doc Extractor" (KhmerOCR): Dashboard
(drop zone, defaults, stats, recent documents, queue), New OCR job (file cards, settings), Process (live boxes +
pipeline timings), Review & export (original with boxes / editable rebuilt page in the detected fonts, sizes,
colours and bold / text in reading order, "Needs review" list of low-confidence lines with in-place editing,
document details), Documents, Batch jobs, Settings (model switch per stage), Help. Documents are kept in the
browser's localStorage; exports: Word (.docx, `POST /export/docx`, keeps paragraphs, alignment, indents, bullets,
fonts, sizes, colours, bold words, logo / stamp pictures and page margins), plain text, JSON, page image. The
previous one-page demo is `legacy.html`.

## API

The v1 contract is unchanged; v2 only adds fields and endpoints.

```text
GET  /models              models per stage + active ids (v1 shape: stage = Layout | Detection | Recognition)
POST /models              {model_id}          switch recognition model (v1)
POST /det-models          {model_id}          switch text-line detection model (v1)
POST /layout-models       {model_id}          switch layout model
POST /pipeline/models     {stage, model_id}   switch any stage (layout | textdet | rec)
GET  /pipeline/info       manifest, providers, active/available models, runtime details per stage
GET  /health
POST /process             multipart files[] -> {pages: [{page_id, image_url, width, height, blocks, timings, models, ignored_regions}]}
POST /reprocess           {page_id}
POST /regions             {page_id, regions: [{type, box}]}          type = v1 type or v2 class name
POST /lines               {page_id, blocks: [{type, box, lines: [{x, y, width, height}]}]}
POST /export/docx         {pages: [<page result, possibly edited>], title} -> .docx
GET  /storage/...         page images and crops
```

Block: `id, type (v1: title/paragraph/header/caption/image), label (= cls), cls, cls_id, display, color, score,
source, reading_order, relabel, box{x,y,width,height}, align (left|center|right), font_px,
font{family, weight, slant, confidence}|null, text_color,
lines[{box, quad, text, confidence, score, reading_order, indent_level, indent, tab, bullet, bullet_box, baseline,
core_h, font, bold, spans, text_color}], paragraphs[{indent_level, bullet, text, lines}], crop{url,file}|null, text (one tab per indent level,
bullet glyph before list items), recognized`. Page: `layout{margins{left,right,top,bottom}, zones{header,body,footer}}`.
`timings`: `layout_ms, textdet_ms, bullets_ms, rec_ms, font_ms, postcorrect_ms, order_ms, total_ms` (+ `lines_detected`,
`lines_recognized`, `bullets_found`, `crops_snapped`, `decorations`). `models` names the models that actually served
the request (`font` is null when the stage is off).

Input rules: `page_id` must be the 32-hex id from `/process` (else 422); uploads > `MAX_UPLOAD_MB` (40) -> 413;
undecodable files -> 422 and nothing is stored; transparent images are composited on white; pages longer than
`MAX_IMAGE_SIDE` (3600 px) are downscaled on upload and stored that way (all coordinates refer to `image_url`);
user boxes must be finite and are clipped to the page. A model switch loads the model first; if loading fails the
previous model stays active (422 + `errors` in `/models`).

## Models

Model files are not in git (`.model/` is ignored). Layout:

```text
.model/v2/
+-- models.json
+-- layout/layout_v1_12cls.onnx      # YOLO11s v1, 12 classes, imgsz 1024, opset 17 (exported from layout_best.pt)
+-- layout/layout_v1_12cls.pt        # -> symlink, legacy torch kind
+-- textdet/db_v1.onnx               # -> symlink, DB PP-OCRv5 mobile fine-tuned on Khmer
+-- textdet/db_ppocrv5_orig.onnx     # -> symlink, original PP-OCRv5 mobile det
+-- rec/crnn_m_v1.onnx, rec/crnn_m_v1_int8.onnx, rec/crnn_m_v1.vocab.json   # -> symlinks
```

`models.json` (example in `backend/models.example.json`):

```json
{"schema": 2,
 "defaults": {"layout": "layout_yolo11s_v1_12cls", "textdet": "db_ppocrv5_khmer_v1", "rec": "crnn_m_v1"},
 "models": [{"id": "crnn_m_v1", "stage": "rec", "kind": "crnn_onnx", "file": "rec/crnn_m_v1.onnx",
             "vocab": "rec/crnn_m_v1.vocab.json", "version": "v1", "sha256": "…", "input_size": [48, null]}]}
```

`file`/`vocab` are relative to the manifest folder. `kind`: `yolo_onnx`, `db_onnx`, `crnn_onnx` (default, onnxruntime)
or the legacy torch kinds `ultralytics_pt`, `doclayout_yolo_pt`, `ultralytics_lines`, `torch_resnet_bilstm` (need
`requirements-torch.txt`; imported only when selected). Optional `params` override stage parameters (e.g.
`{"unclip_ratio": 1.5}` for a DB model, `{"output_layout": "NTC"}` for a batch-major recogniser). The sha256 is checked
on load (warning only). The old `.model/v1/...` files are still listed automatically if present.

### Add a new model version (no code changes)

```bash
# 1. export (layout example; CPU is fine)
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=1024, opset=17, simplify=False, device='cpu')"
mv best.onnx .model/v2/layout/layout_v2_21cls.onnx
# 2. register it (sha256, ONNX class names and input size are filled in); --default makes it the stage default
cd backend && python scripts/register_model.py --stage layout --kind yolo_onnx --id layout_v2_21cls \
    --file layout/layout_v2_21cls.onnx --version v2 --label "YOLO11s layout v2 (21 classes)" --default
# 3. restart the backend, or switch at runtime: POST /pipeline/models {"stage":"layout","model_id":"layout_v2_21cls"}
```

Same for `--stage textdet --kind db_onnx` and `--stage rec --kind crnn_onnx --vocab rec/vocab_v2.json`. A model with
a different contract (new architecture) needs a new class in `services/stages/` plus one line in
`model_registry.KINDS`. Layout class ids must follow `training/taxonomy.py` (copy any change to
`backend/app/taxonomy.py`; `tests/test_units.py` checks they match).

## Setup

### Backend

```bash
cd backend
cp .env.example .env                    # optional; defaults work
python -m venv env && source env/bin/activate      # or: uv venv env && source env/bin/activate
pip install -r requirements.txt         # onnxruntime path, no torch
# optional: pip install -r requirements-torch.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload      # or ./run.sh (PORT=8000)
```

Settings (see `.env.example`): `MODEL_MANIFEST`, `LAYOUT_MODEL` / `TEXTDET_MODEL` / `REC_MODEL`, `POSTCORRECT`,
`ORT_DEVICE` (auto|cpu|cuda), `ORT_THREADS`, `LAYOUT_CONF`, `DET_BOX_THRESH`, `DET_UNCLIP_RATIO`,
`DET_LONG_MIN/MAX`, `DET_CASCADE`, `REC_BATCH`, `CORS_ORIGINS`. For CUDA install `onnxruntime-gpu` instead of
`onnxruntime`.

Tests (need the models in `.model/v2`; model tests are skipped otherwise):

```bash
cd backend && python -m pytest -q tests
```

### Frontend

```bash
cd frontend
cp .env.example .env.local              # NEXT_PUBLIC_API_BASE=http://localhost:8000
npm install
npm run dev                             # http://localhost:3000   (npm run build for production)
```

The UI shows all 21 classes with their taxonomy colours, image crops for non-text classes, per-stage timings, and a
model picker per stage (header "Models" menu). The detection editor offers all 21 classes.

## Phone / browser model set (all ONNX, CPU)

Every default stage is small enough for on-device use (ONNX Runtime Web / Mobile); sizes on disk and CPU latency on
the dev machine (4 threads):

| stage | model id | file | size | latency |
|---|---|---|---|---|
| layout | `layout_yolo11n_v3_22cls` | `layout/layout_v3_n_22cls.onnx` | 10.9 MB | ~130 ms / page at 1024 |
| textdet | `db_ppocrv5_khmer_v2` | `textdet/db_khmer_v2.onnx` | 4.8 MB | ~300-500 ms / page (long side 1600-2200) |
| rec | `crnn_m_v2_stamp2_int8` (int8) or `crnn_m_v2_stamp2` | `rec/…` | 4.8 MB / 19 MB | ~12 ms / line |
| font | `font_cnn_v2` | `font/font_cnn_v2.onnx` | 1.9 MB | ~1 ms / line (family, weight, slant + em size) |
| structure | `struct_v1` | `structure/struct_v1.onnx` | 4.0 MB | ~3 ms / page |

Layout v3 (YOLO11n) vs the 38 MB YOLO11s round-2 model on the 1,100-page layout_v3 validation set: mAP50 0.973 vs
0.971, mAP50-95 0.843 vs 0.847 (the s model cannot predict `decoration`); on the real test pages the two are
equivalent with the pipeline rules above, and only v3 finds the ornaments under emblems. The heuristics in
`services/` (bullets, stamps, split figures, snapping, bold, colours) are numpy / OpenCV and would be ported to the
client for a fully on-device pipeline.

## Mobile notes

- All three default models are plain ONNX (opset 17, fp32; int8 recogniser available) and run with onnxruntime
  Mobile (Android: NNAPI/XNNPACK EP, iOS: CoreML EP) or can be converted to TFLite / CoreML.
- Shapes: layout is static `[1,3,1024,1024]`; DB is dynamic `[1,3,H,W]` (H, W multiples of 32 - use a smaller
  long side such as 960..1280 on phones); recogniser is `[N,1,48,W]` with dynamic width.
- Everything outside the networks is small numpy/OpenCV code meant to be ported 1:1: letterbox + NMS
  (`layout_yolo_onnx.py`), DB box extraction + unclip (`textdet_db_onnx.py`), width bucketing + greedy CTC
  (`rec_crnn_onnx.py`), line assignment and XY-cut (`pipeline_service.py`, `reading_order.py`).
- Class names travel inside the layout ONNX (`names` metadata) and the vocab is a JSON next to the recogniser, so an
  app can ship `models.json` + files exactly as the backend does.
- On CPU most time goes to recognition (it scales with the number of lines) and the 1600-2200 px DB pass.

## Notes

- `.model/`, `*.pt`, `*.pth`, `backend/storage/`, `node_modules/`, `.next/` are git-ignored.
- `backend/server.py` and `backend/inspect_ckpt.py` are pre-v2 leftovers and are not used by `app.main`.
