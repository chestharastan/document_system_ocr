#!/usr/bin/env python3
"""Training monitor for the document OCR v2 project (stdlib only).

    python tools/monitor/server.py --work /home/hello/ocr/docsys/work --port 8005

GET /            dashboard (index.html, refreshes itself)
GET /api/status  JSON: GPU, disk, CPU/RAM, running jobs, every run in <work>/runs (YOLO results.csv or recogniser log.jsonl),
                 live progress parsed from <work>/logs/<run>.log, data-job log tails, and <work>/status.json (stages/team/results).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def tail(path: Path, n_bytes=6000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n_bytes))
            return ANSI.sub("", f.read().decode("utf-8", "replace"))
    except OSError:
        return ""


def last_segment(text: str) -> str:
    segs = [s.strip() for s in re.split(r"[\r\n]", text) if s.strip()]
    return segs[-1] if segs else ""


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        name, util, used, total, temp, power = [x.strip() for x in out.strip().split("\n")[0].split(",")]
        return dict(name=name, util=float(util), mem_used=float(used), mem_total=float(total), temp=float(temp),
                    power=float(power) if power.replace(".", "").isdigit() else None)
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e))


def sysinfo(work: Path):
    du = shutil.disk_usage(work)
    mem = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":")
            mem[k] = int(v.split()[0]) / 2**20
    except OSError:
        pass
    load = os.getloadavg()
    return dict(disk_free_gb=du.free / 1e9, disk_total_gb=du.total / 1e9, ram_total_gb=mem.get("MemTotal"),
                ram_avail_gb=mem.get("MemAvailable"), load=load, cpus=os.cpu_count())


def jobs():
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,etimes,pcpu,rss,args", "--no-headers"], text=True, timeout=5)
    except Exception:  # noqa: BLE001
        return []
    res = []
    for ln in out.splitlines():
        parts = ln.split(None, 4)
        if len(parts) < 5:
            continue
        pid, et, cpu, rss, args = parts
        m = re.search(r"-m (training\.[\w.]+)", args)
        if not m or "multiprocessing" in args:
            continue
        name = m.group(1)
        nm = re.search(r"--name (\S+)", args) or re.search(r"--out (\S+)", args)
        res.append(dict(pid=int(pid), elapsed_s=int(et), cpu=float(cpu), rss_gb=int(rss) / 2**20, module=name,
                        target=Path(nm.group(1)).name if nm else ""))
    # keep only the parent process of each module+target (dataloader workers share the command line)
    seen, out_ = set(), []
    for j in sorted(res, key=lambda j: -j["elapsed_s"]):
        k = (j["module"], j["target"])
        if k not in seen:
            seen.add(k)
            out_.append(j)
    return out_


def yolo_run(d: Path, logs: Path):
    rows = list(csv.DictReader(open(d / "results.csv")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    ep = [dict(epoch=int(float(r["epoch"])), minutes=float(r["time"]) / 60, mAP50=float(r["metrics/mAP50(B)"]),
               mAP=float(r["metrics/mAP50-95(B)"]), P=float(r["metrics/precision(B)"]), R=float(r["metrics/recall(B)"]),
               box_loss=float(r["train/box_loss"]), cls_loss=float(r["train/cls_loss"])) for r in rows]
    live, prog = last_segment(tail(logs / f"{d.name}.log")), None
    m = re.search(r"(\d+)/(\d+)\s+[\d.]+G.*?(\d+)/(\d+)\s+([\d.]+)it/s\s+([\d:]+)<([\d:]+)", live)
    if m:
        prog = dict(epoch=int(m.group(1)), epochs=int(m.group(2)), step=int(m.group(3)), steps=int(m.group(4)),
                    speed=f"{m.group(5)} it/s", elapsed=m.group(6), eta_epoch=m.group(7))
    elif "Box(P" in live or "mAP50" in live:
        prog = dict(phase="validating")
    best = max(ep, key=lambda e: e["mAP"]) if ep else None
    return dict(name=d.name, kind="layout (YOLO)", epochs=ep, progress=prog, best=best,
                weights=sorted(p.name for p in (d / "weights").glob("*.pt")) if (d / "weights").exists() else [],
                updated=os.path.getmtime(d / "results.csv"))


def rec_run(d: Path, logs: Path):
    ep = [json.loads(l) for l in open(d / "log.jsonl") if l.strip()] if (d / "log.jsonl").exists() else []
    st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
    prog = None
    for ln in ([] if st.get("state") == "done" else reversed(tail(logs / f"{d.name}.log").splitlines())):
        m = re.search(r"ep (\d+) step (\d+)/(\d+) loss~([\d.]+) (\d+) img/s", ln)
        if m:
            prog = dict(epoch=int(m.group(1)), epochs=st.get("epochs"), step=int(m.group(2)), steps=int(m.group(3)),
                        loss=float(m.group(4)), speed=f"{m.group(5)} img/s")
            break
    return dict(name=d.name, kind="recogniser (CRNN-CTC)", epochs=ep, progress=prog, state=st.get("state"),
                params_m=st.get("params_m"), best_val_cer=st.get("best_val_cer"),
                updated=os.path.getmtime(d / "state.json") if (d / "state.json").exists() else 0)


def textdet_run(d: Path, logs: Path):
    hist = []
    for f in ("history.json", "hist.json"):
        if (d / f).exists():
            hist = json.loads((d / f).read_text())
            break
    live = last_segment(tail(logs / f"{d.name}.log"))
    return dict(name=d.name, kind="text-line detector (DB)", history=hist, live=live[-220:],
                updated=max((os.path.getmtime(p) for p in d.iterdir()), default=0))


def runs(work: Path):
    out = []
    logs = work / "logs"
    for d in sorted((work / "runs").glob("*")):
        if not d.is_dir():
            continue
        try:
            if (d / "results.csv").exists():
                out.append(yolo_run(d, logs))
            elif (d / "log.jsonl").exists() or (d / "state.json").exists():
                out.append(rec_run(d, logs))
            elif d.name.startswith("textdet"):
                out.append(textdet_run(d, logs))
        except Exception as e:  # noqa: BLE001
            out.append(dict(name=d.name, kind="?", error=repr(e)))
    out.sort(key=lambda r: -r.get("updated", 0))
    return out


def status(work: Path):
    st = {}
    try:
        st = json.loads((work / "status.json").read_text())
    except Exception as e:  # noqa: BLE001
        st = dict(error=f"status.json: {e!r}")
    data_logs = []
    for name in st.get("data_logs", []):
        p = work / "logs" / name
        if p.exists():
            data_logs.append(dict(name=name, last=last_segment(tail(p, 3000))[-240:], updated=os.path.getmtime(p)))
    return dict(now=time.time(), gpu=gpu(), sys=sysinfo(work), jobs=jobs(), runs=runs(work), board=st, data_logs=data_logs,
                user_docs=user_docs(work))


def user_docs(work: Path):
    """Photos imported from the user's camera card by training/data/import_flash.py."""
    d = work / "data" / "user_docs"
    try:
        st = json.loads((d / "import_status.json").read_text())
    except Exception:  # noqa: BLE001
        return None
    recs = []
    if (d / "index.jsonl").exists():
        recs = [json.loads(l) for l in open(d / "index.jsonl") if l.strip()]
    recs.sort(key=lambda r: (r.get("src_mtime", 0), r["source"]))
    # watcher writes import_status.json every scan; stale file => watcher not running
    st["watcher_alive"] = time.time() - os.path.getmtime(d / "import_status.json") < 120
    st["photos"] = [dict(thumb=r["thumb"], image=r["image"], source=r["source"], split=r["split"]) for r in recs]
    return st


def make_handler(work: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/status":
                try:
                    body = json.dumps(status(work)).encode()
                    self._send(200, body, "application/json")
                except Exception as e:  # noqa: BLE001
                    self._send(500, json.dumps(dict(error=repr(e))).encode(), "application/json")
            elif re.fullmatch(r"/userdocs/(thumbs|images)/[\w.-]+\.jpg", self.path.split("?")[0]):
                f = work / "data" / "user_docs" / self.path.split("?")[0][len("/userdocs/"):]
                if f.is_file():
                    self._send(200, f.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            elif self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/home/hello/ocr/docsys/work"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8005)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(a.work))
    print(f"monitor on http://{a.host}:{a.port}/  (work={a.work})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
