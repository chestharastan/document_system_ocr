#!/bin/bash
# run.sh

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Run with uvicorn (development)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production (remove --reload):
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4