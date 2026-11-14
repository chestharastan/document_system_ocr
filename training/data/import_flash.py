"""Import the user's document photos from a camera card / USB flash drive into the project.

    python -m training.data.import_flash                    # one scan
    python -m training.data.import_flash --watch 20         # keep watching (re-scan every 20 s)

What it does on every scan:
  * finds the card by filesystem label (default EOS_DIGITAL). If it is plugged in but not mounted it is mounted
    READ-ONLY with udisksctl. Nothing on the card is ever written, moved or deleted.
  * reads only the folder(s) given by --subdir (default "Image": the folder the user copies document photos to;
    DCIM camera photos are NOT touched). macOS "._*" sidecar files are skipped.
  * converts HEIC/HEIF/JPG/PNG to an upright (EXIF-rotated) RGB JPEG q92 at full resolution in
    <out>/images/, plus a 360 px thumbnail in <out>/thumbs/ for the monitor.
  * de-duplicates by SHA-1 of the source bytes, so re-plugging the card or renaming files never double-imports.
  * assigns a fixed split per photo (hash-based): 25% "test" (held out, never trained on), 75% "pool".
  * appends to <out>/index.jsonl and rewrites <out>/import_status.json (read by tools/monitor).

These are the user's own real documents (they can contain real names), so they stay local: never published,
never used as identities in synthetic data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # HEIC files will then be reported as failed
    pass

EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
USER = os.environ.get("USER", "hello")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_card(label: str):
    """-> (device, mountpoint or None) for the partition with this filesystem label."""
    try:
        out = subprocess.check_output(["lsblk", "-J", "-o", "NAME,LABEL,MOUNTPOINTS,PATH"], text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return None, None
    stack = json.loads(out)["blockdevices"]
    while stack:
        d = stack.pop()
        stack.extend(d.get("children") or [])
        if d.get("label") == label:
            mps = [m for m in (d.get("mountpoints") or []) if m]
            return d["path"], (mps[0] if mps else None)
    return None, None


def ensure_mounted(label: str):
    dev, mp = find_card(label)
    if dev and not mp:
        r = subprocess.run(["udisksctl", "mount", "-b", dev, "-o", "ro", "--no-user-interaction"],
                           capture_output=True, text=True, timeout=60)
        log("mount", dev, r.stdout.strip() or r.stderr.strip())
        dev, mp = find_card(label)
    return dev, mp


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index(out: Path):
    idx = {}
    if (out / "index.jsonl").exists():
        for ln in open(out / "index.jsonl"):
            if ln.strip():
                r = json.loads(ln)
                idx[r["sha1"]] = r
    return idx


def scan(a, idx):
    dev, mp = ensure_mounted(a.label)
    st = dict(label=a.label, device=dev, mount=mp, connected=bool(mp), last_scan=time.time(), new=0, failed=[])
    if mp:
        for sub in a.subdir:
            src = Path(mp) / sub
            if not src.is_dir():
                continue
            for p in sorted(src.rglob("*")):
                if not p.is_file() or p.name.startswith("._") or p.suffix.lower() not in EXTS:
                    continue
                try:
                    h = sha1(p)
                    if h in idx:
                        continue
                    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                    name = f"{h[:10]}_{p.stem}.jpg"
                    im.save(a.out / "images" / name, quality=92)
                    th = im.copy()
                    th.thumbnail((360, 360))
                    th.save(a.out / "thumbs" / name, quality=80)
                    rec = dict(sha1=h, image=f"images/{name}", thumb=f"thumbs/{name}", source=str(p.relative_to(mp)),
                               card=a.label, w=im.width, h=im.height, src_mtime=p.stat().st_mtime,
                               imported=time.time(), split="test" if int(h[:8], 16) % 4 == 0 else "pool")
                    with open(a.out / "index.jsonl", "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    idx[h] = rec
                    st["new"] += 1
                    log("imported", rec["source"], "->", name, rec["split"])
                except Exception as e:  # noqa: BLE001
                    st["failed"].append(f"{p.name}: {e!r}"[:200])
                    log("FAILED", p, repr(e))
    recs = sorted(idx.values(), key=lambda r: r["imported"])
    st.update(total=len(recs), test=sum(r["split"] == "test" for r in recs),
              pool=sum(r["split"] == "pool" for r in recs),
              last_import=recs[-1]["imported"] if recs else None)
    tmp = a.out / "import_status.json.tmp"
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(a.out / "import_status.json")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="EOS_DIGITAL", help="filesystem label of the card / flash drive")
    ap.add_argument("--subdir", nargs="+", default=["Image"], help="folder(s) on the card holding document photos")
    ap.add_argument("--out", type=Path, default=Path("/home/hello/ocr/docsys/work/data/user_docs"))
    ap.add_argument("--watch", type=float, default=0, help="re-scan every N seconds (0 = scan once)")
    a = ap.parse_args()
    for d in ("images", "thumbs"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    idx = load_index(a.out)
    was = None
    while True:
        st = scan(a, idx)
        if st["connected"] != was or st["new"]:
            log(f"card {'connected at ' + st['mount'] if st['connected'] else 'not connected'}; "
                f"new {st['new']}, total {st['total']} (test {st['test']}, pool {st['pool']}), failed {len(st['failed'])}")
            was = st["connected"]
        if not a.watch:
            return 0 if not st["failed"] else 1
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
