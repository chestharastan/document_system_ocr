# Khmer Document OCR

End-to-end Khmer document recognition system with a Next.js frontend, FastAPI backend, and local OCR model files.

## Project Structure

```text
document_system_ocr/
+-- README.md
+-- .gitignore
+-- .model/
|   +-- v1/
|       +-- detection/
|       |   +-- best.pt
|       +-- layout/
|       |   +-- doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt
|       +-- recognition/
|           +-- best_model.pth
|           +-- char.json
+-- backend/
|   +-- .env.example
|   +-- requirements.txt
|   +-- run.sh
|   +-- app/
|   |   +-- main.py
|   |   +-- config.py
|   |   +-- dependencies.py
|   |   +-- routers/
|   |   +-- schemas/
|   |   +-- services/
|   +-- storage/
|       +-- pages/
|       +-- crops/
+-- frontend/
    +-- .env.example
    +-- package.json
    +-- next.config.mjs
    +-- app/
    |   +-- layout.js
    |   +-- page.js
    |   +-- globals.css
    +-- components/
    |   +-- OcrApp.js
    +-- lib/
        +-- api.js
```

## What Each Part Does

### Frontend

The frontend is a Next.js app.

- Location: `frontend/`
- Framework: Next.js `14`
- Main page: `frontend/app/page.js`
- Main OCR UI component: `frontend/components/OcrApp.js`
- API helper functions: `frontend/lib/api.js`
- Backend URL config: `frontend/.env.local`

The frontend sends uploaded document images to the backend API and displays OCR results.

### Backend

The backend is a FastAPI app.

- Location: `backend/`
- API entry point: `backend/app/main.py`
- Environment config: `backend/app/config.py`
- API routes: `backend/app/routers/`
- OCR/model logic: `backend/app/services/`
- Request/response schemas: `backend/app/schemas/`
- Uploaded pages and crops: `backend/storage/`

The backend loads the OCR models, processes uploaded images, saves generated page/crop files, and returns OCR blocks/lines/text to the frontend.

Main API endpoints:

```text
GET  /models
POST /models
POST /det-models
POST /process
POST /reprocess
POST /regions
POST /lines
```

### Models

The `.model/` folder stores local model weights and charset files.

Expected structure:

```text
.model/
+-- v1/
    +-- detection/
    |   +-- best.pt
    +-- layout/
    |   +-- doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt
    +-- recognition/
        +-- best_model.pth
        +-- char.json
```

Important: `.model/`, `*.pt`, and `*.pth` are ignored by Git because model files are usually large. Keep the model files locally or share them separately.

## Setup

### 1. Backend Setup

Go to the backend folder:

```bash
cd backend
```

Create the backend environment file:

```bash
cp .env.example .env
```

Create and activate a Python virtual environment:

```bash
python -m venv env
source env/bin/activate
```

For Windows:

```bash
python -m venv env
env\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Run the backend server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the script:

```bash
chmod +x run.sh
./run.sh
```

Backend URL:

```text
http://localhost:8000
```

### 2. Frontend Setup

Open a new terminal and go to the frontend folder:

```bash
cd frontend
```

Create the frontend environment file:

```bash
cp .env.example .env.local
```

Install frontend dependencies:

```bash
npm install
```

Run the frontend server:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

## Environment Files

### Backend `.env`

Default values come from `backend/.env.example`:

```env
MODEL_ROOT=.model

LAYOUT_WEIGHTS=.model/v1/layout/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt
LAYOUT_IMGSZ=1120
LAYOUT_CONF=0.25

DET_MODEL=line_level
DET_WEIGHTS=.model/v1/detection/best.pt
DET_CONF=0.25

RECOG_MODEL=resnet_bilstm_ctc
RECOG_DIR=.model/v1/recognition
RECOG_CHARSET=.model/v1/recognition/char.json

CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001
```

These paths are relative to the project root.

### Frontend `.env.local`

Default value comes from `frontend/.env.example`:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

This tells the Next.js frontend where the FastAPI backend is running.

## Run Full Project

Terminal 1:

```bash
cd backend
source env/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Then open:

```text
http://localhost:3000
```

## Notes

- Start the backend before using the frontend OCR upload flow.
- Keep `.model/` at the project root, not inside `backend/` or `frontend/`.
- Generated upload files are stored in `backend/storage/`.
- The frontend uses `.js` files, but those files can still contain React JSX because Next.js supports JSX inside `.js`.
