#!/bin/bash
# run.sh - start the API (development). Settings are read from backend/.env by the app itself.
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload

# Production (remove --reload):
# uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 2
