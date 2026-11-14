#!/bin/bash
# keeps the camera-card importer alive: bash tools/monitor/flash_watch.sh
cd "$(dirname "$0")/../.."
while true; do
  /home/hello/ocr/khm_repo_clone/det_venv/bin/python -m training.data.import_flash --watch 20
  sleep 5
done
