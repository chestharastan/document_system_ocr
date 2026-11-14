#!/bin/bash
# keeps the monitor alive: bash tools/monitor/run.sh [port]
PORT=${1:-8005}
cd "$(dirname "$0")"
while true; do
  /usr/bin/python3 server.py --port "$PORT" --work /home/hello/ocr/docsys/work
  sleep 3
done
