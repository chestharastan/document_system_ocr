# Khmer Document OCR

An end-to-end OCR system for Khmer documents. Upload one or many page images and
the system runs the full pipeline — **layout analysis → text-line detection →
text recognition** — animating each stage over the page, then rebuilds an
**editable document** with the recognized text in place and figures cropped
back in at their exact coordinates. The interface is deliberately monochrome
(white to black), modern and minimal.

## Pipeline

1. **Layout analysis** — `doclayout_yolo` (YOLOv10, DocStructBench) classifies the
   page into structural regions: title, paragraph, header, caption, figure.
2. **Text detection** — a trained Khmer YOLO line detector localises the
   individual text lines and each line is assigned to the region it falls inside.
3. **Recognition** — non-header line crops are read by a fine-tuned CRNN
   (CNN-BiLSTM-CTC) Khmer recognizer. Headers are detected but not recognized.
4. **Reconstruction** — text blocks become editable text positioned at their
   original coordinates; figures are cropped from the page and placed at their
   exact coordinates.

All coordinates are in original page-pixel space, so crops line up with the page
exactly.

## Architecture

A **FastAPI** backend (models + API) and a **Next.js** frontend (App Router).

```
backend/                 FastAPI API service
  pipeline.py      Loads the 3 models once; process_image() -> staged JSON
  server.py        FastAPI: POST /api/process (1..N images), /api/health, /storage
  requirements.txt Web deps to add to the existing `thaocr` conda env
  run.sh           Launch helper (installs web deps on first run)
  storage/         Saved page images + figure crops (created at runtime)
frontend/                Next.js app
  app/layout.js          Root layout
  app/page.js            Renders <OcrApp/>
  app/globals.css        Monochrome design system
  components/OcrApp.js    UI shell + staged-animation controller + editable rebuild
  lib/api.js             API base URL + processFiles() upload helper
  .env.local             NEXT_PUBLIC_API_BASE (backend origin)
```

The backend reuses the validated model stack from the detection project:
- Layout weights: `doclayout_yolo_docstructbench_imgsz1024.pt`
- Text-line detector: the `archive/ocr_pipeline.py` `KhmerOCR` detector.
- Recognizer: selectable standalone CRNN runs, loaded in-process via the
  reah_ocr `Predictor` with charset `text_img/original/khmer_100k/char.json`:
  `previous` (`/home/thareah/Desktop/Khmer-OCR/Model/final2`) and
  `resnet_bilstm_ctc_100k`
  (`/run/media/thareah/231d5e3e-9869-4244-9694-7d02d106153b/home/thareah/model_final_fail/fail/origial/resnet_bilstm_ctc_100k`).

Override any path via environment variables (`LAYOUT_WEIGHTS`, `RECOG_DIR`,
`RECOG_CHARSET`, `DET_MODEL`). Set `RECOG_MODEL=previous` or
`RECOG_MODEL=resnet_bilstm_ctc_100k` to choose the startup recognizer.

### Document reconstruction rules

The rebuilt page keeps each block at its **original coordinates**. Paragraph
lines also keep first-line tabs: when a detected text-line box starts
noticeably inside the paragraph layout box's left edge, that line is marked as a
tabbed line and copy/download output prefixes it with a tab character. In the
analysis overlay, paragraphs are drawn with dotted boxes, titles solid, headers
dashed, and figures as a hatched box.

## Run

Two processes: the FastAPI backend and the Next.js frontend.

**1. Backend** — the models live in the `thaocr` conda env (torch + doclayout_yolo
+ the OCR stack). `run.sh` installs the small web deps (FastAPI/uvicorn) into it
on first launch:

```bash
cd backend
./run.sh                # API on http://127.0.0.1:8200
# or a custom port:  PORT=9000 ./run.sh
```

**2. Frontend** — Next.js (needs Node 18+):

```bash
cd frontend
npm install
npm run dev             # app on http://localhost:3000
```

Open <http://localhost:3000>, drop in one or more document images, and click
**Analyze document**. If the backend is not on `http://localhost:8200`, set
`NEXT_PUBLIC_API_BASE` in `frontend/.env.local` and add the frontend origin to
the backend's `CORS_ORIGINS` env var (defaults already allow ports 3000/3001).

### API

```
POST /api/process     multipart form field `files` (one or many images)
  -> { "pages": [ { page_id, image_url, width, height, blocks: [...] } ] }
```

Each block has a `type` (`title|paragraph|header|caption|image`), a `box`
(`x,y,width,height` in page pixels), detected `lines` (each with its own box +
recognized `text`, blank for headers), and, for picture blocks, a `crop` `{url}`
pointing at the saved PNG.

## Notes

- The first request loads the models (a few seconds); afterwards each page is
  processed in well under a second of model time and the models stay warm.
- Recognition uses the GPU if available and falls back to CPU automatically.
- Rebuilt text blocks are `contenteditable` — edit in place, then **Copy** or
  **Download** the page text.
