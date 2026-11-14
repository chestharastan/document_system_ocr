#!/usr/bin/env python3
"""TabRecSet (CC BY 4.0 on figshare / CC BY-SA 4.0 in its README; camera photos + scans of tables, English + Chinese)
-> work/data/tsr_real_tabrecset.

    python -m training.data.ext_tables_tabrecset fetch   --raw work/data/ext/tables_tabrecset/raw [--per-lang 1500] [--ann-only]
    python -m training.data.ext_tables_tabrecset convert --raw work/data/ext/tables_tabrecset/raw --out work/data/tsr_real_tabrecset

Source: https://doi.org/10.6084/m9.figshare.20647788 file "TabRecSet (CurveTabSet).zip" (3.4 GB, figshare file id
40332520). Nothing is extracted: `fetch` reads the zip's central directory with HTTP range requests and then fetches
only the wanted members (contiguous byte ranges per folder; members are inflated with zlib, size-checked) into
--raw/ann/{TD,TSR}/<lang>/*.json and --raw/image/<lang>_<variant>/*.jpg.
Layout of the zip: image/{chinese,english}_all-line (the 10,846 original photos / scans), image/*_no-line and
*_three-line (the same images with the rule lines erased by the authors, same annotations), TD annotation/<lang>/*.json,
TSR_TCR_annotation/<lang>/*.json (LabelMe JSON: shapes with `points` polygon, `group_id` = table instance, `label` =
"<Row>-<Column>-<Rowspan>-<Colspan>-<Text>").
Selection: per language the first --per-lang `all-line` images in zip (= name) order (ids are random strings, so this is
a pseudo-random sample); split by sha1 of the source-image group (`group_key`: the
"<k>_<md5>_segR" table segments of one photo share a group; 60 / 10 / 30 train / val / test); the line-erased variants
(--variants, default no-line only, to stay inside the download budget) are fetched for the train images only (same split
as their original). Conversion details: see `convert`.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import subprocess
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

from shapely.geometry import Polygon, box as sbox

from training.data.ext_tables_common import TsrWriter, check_grid, fill_missing, split_for, union, finish_table

ZIP_URL = "https://ndownloader.figshare.com/files/40332520"
ROOT = "TabRecSet (CurveTabSet)/"
VARIANTS = ("all-line", "no-line", "three-line")
LANGS = ("english", "chinese")


# --------------------------------------------------------------------------- remote zip reading
def _curl_range(url: str, start: int, end: int, out: Path | None = None, attempts: int = 4) -> bytes:
    """One HTTP range request (figshare redirects to a short-lived signed URL, so every request restarts at the redirect)."""
    cmd = ["curl", "-sL", "--retry", "3", "-r", f"{start}-{end}", url]
    for k in range(attempts):
        try:
            if out is not None:
                subprocess.run(cmd + ["-o", str(out)], check=True)
                if out.stat().st_size == end - start + 1:
                    return b""
                raise RuntimeError(f"short read {out.stat().st_size} of {end - start + 1}")
            data = subprocess.run(cmd, check=True, capture_output=True).stdout
            if len(data) == end - start + 1:
                return data
            raise RuntimeError(f"short read {len(data)} of {end - start + 1}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"  range {start}-{end} attempt {k + 1} failed: {e}", flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"range {start}-{end} failed after {attempts} attempts")


def _total_size(url: str) -> int:
    out = subprocess.run(["curl", "-sL", "-r", "0-0", "-D", "-", "-o", "/dev/null", url], capture_output=True, text=True).stdout
    return int(re.findall(r"[Cc]ontent-[Rr]ange: bytes 0-0/(\d+)", out)[-1])


def central_directory(url: str):
    """-> list of (name, method, csize, usize, local_header_offset) from the zip's central directory."""
    size = _total_size(url)
    tail = _curl_range(url, max(0, size - 66000), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, nrec, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if nrec == 0xFFFF or cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        (_, _, _, _, _, _, _, nrec, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", tail[j:j + 56])
    cd = _curl_range(url, cd_off, cd_off + cd_size - 1)
    pos, entries = 0, []
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, _, csize, usize, nlen, elen, clen, _, _, _, lhoff) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + elen]
        if 0xFFFFFFFF in (usize, csize, lhoff):
            p = 0
            while p + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[p:p + 4])
                if hid == 1:
                    q = p + 4
                    vals = []
                    for cur in (usize, csize, lhoff):
                        if cur == 0xFFFFFFFF:
                            vals.append(struct.unpack("<Q", extra[q:q + 8])[0])
                            q += 8
                        else:
                            vals.append(cur)
                    usize, csize, lhoff = vals
                    break
                p += 4 + hsz
        entries.append((name, method, csize, usize, lhoff))
        pos += 46 + nlen + elen + clen
    return size, entries


def fetch_members(url: str, entries: list, wanted: list, dest_of, tmp: Path, max_chunk: int = 120 << 20) -> int:
    """Fetch `wanted` entries (subset of `entries`, same tuples) as few contiguous ranges, inflate, write via dest_of(name).
    A range spans from one wanted member to the next as long as the chunk stays under max_chunk. Members whose
    destination file already exists are skipped (resumable)."""
    by_off = sorted(entries, key=lambda e: e[4])
    next_off = {e[4]: (by_off[i + 1][4] if i + 1 < len(by_off) else None) for i, e in enumerate(by_off)}
    wanted = sorted((e for e in wanted if not dest_of(e[0]).exists()), key=lambda e: e[4])
    n_done = 0
    i = 0
    while i < len(wanted):
        start = wanted[i][4]
        j = i
        while j + 1 < len(wanted) and wanted[j + 1][4] + wanted[j + 1][2] + 100 - start < max_chunk:
            j += 1
        end_entry = wanted[j]
        end = (next_off[end_entry[4]] or (end_entry[4] + 30 + 1024 + end_entry[2] + 64)) - 1
        tmp.parent.mkdir(parents=True, exist_ok=True)
        _curl_range(url, start, end, tmp)
        buf = tmp.read_bytes()
        for e in wanted[i:j + 1]:
            name, method, csize, usize, lhoff = e
            p = lhoff - start
            if buf[p:p + 4] != b"PK\x03\x04":
                raise RuntimeError(f"bad local header for {name}")
            nlen, elen = struct.unpack("<HH", buf[p + 26:p + 30])
            data = buf[p + 30 + nlen + elen:p + 30 + nlen + elen + csize]
            raw = zlib.decompress(data, -15) if method == 8 else data
            if len(raw) != usize:
                raise RuntimeError(f"size mismatch for {name}")
            dest = dest_of(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            n_done += 1
        print(f"  fetched {n_done}/{len(wanted)} members ({(end - start) / 1e6:.0f} MB chunk)", flush=True)
        i = j + 1
    tmp.unlink(missing_ok=True)
    return n_done


def fetch(a):
    t = time.time()
    size, entries = central_directory(ZIP_URL)
    print(f"zip {size / 1e9:.2f} GB, {len(entries)} members, central directory read in {time.time() - t:.0f}s")
    json.dump(entries, open(a.raw / "zip_entries.json", "w"))
    by_dir = collections.defaultdict(list)
    for e in entries:
        if e[0].startswith(ROOT) and not e[0].endswith("/"):
            by_dir[e[0][len(ROOT):].rsplit("/", 1)[0]].append(e)
    for d in sorted(by_dir):
        print(f"  {len(by_dir[d]):6d} members {sum(e[2] for e in by_dir[d]) / 1e6:8.1f} MB  {d}")

    def dest_of(name):
        rel = name[len(ROOT):]
        if rel.startswith("TD annotation/"):
            return a.raw / "ann" / "TD" / rel[len("TD annotation/"):]
        if rel.startswith("TSR_TCR_annotation/"):
            return a.raw / "ann" / "TSR" / rel[len("TSR_TCR_annotation/"):]
        return a.raw / rel  # image/<lang>_<variant>/<name>.jpg

    wanted = [e for d, es in by_dir.items() for e in es if d.startswith("TD annotation/") or d.startswith("TSR_TCR_annotation/")]
    fetch_members(ZIP_URL, entries, wanted, dest_of, a.raw / "_chunk.bin")
    if a.ann_only:
        return
    sel = {}
    for lang in LANGS:
        originals = sorted(by_dir[f"image/{lang}_all-line"], key=lambda e: e[0])[: a.per_lang]
        names = {Path(e[0]).name for e in originals}
        train_names = {n for n in names if tsr_split(Path(n).stem) == "train"}
        sel[lang] = {"all-line": originals}
        for v in [v for v in a.variants.split(",") if v and v != "all-line"]:
            sel[lang][v] = [e for e in by_dir[f"image/{lang}_{v}"] if Path(e[0]).name in train_names]
        print(lang, {v: len(es) for v, es in sel[lang].items()}, "train originals", len(train_names))
    wanted = [e for lang in sel for v in sel[lang] for e in sel[lang][v]]
    print("image members to fetch:", len(wanted), f"{sum(e[2] for e in wanted) / 1e6:.0f} MB")
    fetch_members(ZIP_URL, entries, wanted, dest_of, a.raw / "_chunk.bin")
    json.dump({lang: {v: [Path(e[0]).name for e in es] for v, es in sel[lang].items()} for lang in sel},
              open(a.raw / "selection.json", "w"))
    print(f"done in {time.time() - t:.0f}s")


# --------------------------------------------------------------------------- convert
# Photographed tables are often rotated / perspective-warped / curved, while the TSR label is axis-aligned bands. Every
# table gets a *fit* score = mean over cells of the fraction of each annotated cell polygon that falls inside its axis-aligned
# grid box (row band x column band, bands derived like every other set: union of the rs==1 / cs==1 cells of an index,
# boundary at the mid-gap). fit >= --min-fit: kept as photographed (bbox crop of the TD polygon + cells, 2 % pad).
# fit < --min-fit: rectified with the homography that maps the four outer corners of the corner cells (r0c0, r0cN,
# rNcN, rNc0; the polygon point of each corner cell farthest from the table centroid) to a rectangle (a 6 % margin of
# real surrounding pixels is warped along); kept if the rectified fit >= --min-fit, else dropped (strong curvature).
LABEL_RE = re.compile(r"^(\d+)-(\d+)-(\d+)-(\d+)-(.*)$", re.S)
RECT_MARGIN = 0.06


SEG_RE = re.compile(r"^\d+_([0-9a-f]{32})_segR")


def group_key(stem: str) -> str:
    """Source-image group for the split: "<k>_<md5>_segR__tR_[_ann]" are k-th table segments of one source image (up to
    50 per source), "<id>_<n>" is the n-th table photo of <id>, "<id>_<annotator>" carries the annotator initials."""
    m = SEG_RE.match(stem)
    if m:
        return m.group(1)
    return re.sub(r"_\d+$", "", re.sub(r"_[a-z]{2,3}$", "", stem))


def tsr_split(stem: str) -> str:
    return split_for(group_key(stem), 0.6, 0.1, "tabrecset")


def poly_box(pts):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())], p


def skew_deg(p: np.ndarray) -> float:
    """Orientation of the longest edge of the table polygon, folded into [-45, 45) degrees (0 = axis aligned)."""
    best, ang = 0.0, 0.0
    for i in range(len(p)):
        d = p[(i + 1) % len(p)] - p[i]
        L = float(np.hypot(*d))
        if L > best:
            best, ang = L, float(np.degrees(np.arctan2(d[1], d[0])))
    return round(((ang + 45) % 90) - 45, 2)


def tables_from_json(tsr: dict, td: dict | None):
    """LabelMe TSR/TCR annotation (+ TD polygons) -> list of table records (source pixels; cells keep their `poly`) and
    problems."""
    groups = collections.defaultdict(list)
    for sh in tsr.get("shapes", []):
        m = LABEL_RE.match(sh.get("label", ""))
        if not m or len(sh.get("points", [])) < 3:
            continue
        r, c, rs, cs = (int(v) for v in m.groups()[:4])
        box, p = poly_box(sh["points"])
        groups[sh.get("group_id")].append({"r": r, "c": c, "rs": max(1, rs), "cs": max(1, cs), "box": box, "poly": p,
                                           "text": m.group(5)})
    td_polys = {}
    for sh in (td or {}).get("shapes", []):
        td_polys[sh.get("group_id")] = poly_box(sh["points"])
    out, problems = [], []
    for gid, cells in groups.items():
        base_r = min(c["r"] for c in cells)
        base_c = min(c["c"] for c in cells)
        for c in cells:  # labels are 1-based in the released files (checked), make them 0-based
            c["r"] -= base_r
            c["c"] -= base_c
        R = max(c["r"] + c["rs"] for c in cells)
        C = max(c["c"] + c["cs"] for c in cells)
        bad = check_grid(cells, R, C)
        if any(b.startswith("overlap") or b.startswith("cell out") for b in bad):
            problems.append(f"table {gid}: {bad[0]}")
            continue
        filled = fill_missing(cells, R, C)
        ub = union([c["box"] for c in cells])
        tb, tp = td_polys.get(gid, (None, None))
        if tb is None:
            tb, tp = ub, np.array([[ub[0], ub[1]], [ub[2], ub[1]], [ub[2], ub[3]], [ub[0], ub[3]]])
        tb = [min(tb[0], ub[0]), min(tb[1], ub[1]), max(tb[2], ub[2]), max(tb[3], ub[3])]
        out.append({"table_box": tb, "R": R, "C": C, "cells": sorted(cells, key=lambda c: (c["r"], c["c"])),
                    "header_rows": None, "prh_rows": [], "skew_deg": skew_deg(tp), "gid": gid, "filled": filled,
                    "one_based": base_r == 1, "td_poly": tp})
    return out, problems


def table_fit(rec: dict) -> float:
    """Mean over annotated cells of the fraction of the cell polygon inside its axis-aligned grid box (1 = bands exact).
    Unweighted, so that a few big spanning cells cannot hide many misaligned small ones."""
    lab = finish_table(rec)
    grid = {(c["r"], c["c"]): c["grid_box"] for c in lab["cells"]}
    fr = []
    for c in rec["cells"]:
        p = c.get("poly")
        if p is None:
            continue
        P = Polygon(p).buffer(0)
        if P.area <= 0:
            continue
        fr.append(P.intersection(sbox(*grid[(c["r"], c["c"])])).area / P.area)
    return float(np.mean(fr)) if fr else 0.0


def corner_quad(rec: dict):
    """Outer corners TL, TR, BR, BL of the table from the polygons of the cells covering the four corner positions."""
    R, C = rec["R"], rec["C"]
    pts = np.concatenate([c["poly"] for c in rec["cells"] if c.get("poly") is not None])
    ctr = pts.mean(0)
    quad = []
    for (a, b) in ((0, 0), (0, C - 1), (R - 1, C - 1), (R - 1, 0)):
        cell = next((c for c in rec["cells"] if c["r"] <= a < c["r"] + c["rs"] and c["c"] <= b < c["c"] + c["cs"]), None)
        if cell is None or cell.get("poly") is None:
            return None
        p = cell["poly"]
        quad.append(p[np.argmax(((p - ctr) ** 2).sum(1))])
    quad = np.asarray(quad, np.float32)
    if cv2.contourArea(quad) < 0.3 * cv2.contourArea(cv2.convexHull(pts.astype(np.float32))):
        return None  # degenerate (corner cells share points / tiny grids)
    return quad


def rectify(img: np.ndarray, rec: dict):
    """-> (warped image, rectified record) or None. The table maps to the rectangle [m, m+w] x [m, m+h]."""
    q = corner_quad(rec)
    if q is None:
        return None
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if w < 16 or h < 16:
        return None
    mx, my = RECT_MARGIN * w, RECT_MARGIN * h
    dst = np.float32([[mx, my], [mx + w, my], [mx + w, my + h], [mx, my + h]])
    M = cv2.getPerspectiveTransform(q, dst)
    W, H = int(round(w + 2 * mx)), int(round(h + 2 * my))
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cells = []
    for c in rec["cells"]:
        cc = dict(c)
        if c.get("poly") is not None:
            p = cv2.perspectiveTransform(c["poly"].reshape(-1, 1, 2).astype(np.float64), M.astype(np.float64)).reshape(-1, 2)
            cc["poly"] = p
            cc["box"] = [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
        cells.append(cc)
    ub = union([c["box"] for c in cells if c.get("box")])
    tb = [max(0.0, min(mx, ub[0])), max(0.0, min(my, ub[1])), min(W, max(mx + w, ub[2])), min(H, max(my + h, ub[3]))]
    out = dict(rec, cells=cells, table_box=tb, skew_deg=0.0)
    return warped, out


def _strip(rec: dict) -> dict:
    rec = dict(rec)
    rec["cells"] = [{k: v for k, v in c.items() if k != "poly"} for c in rec["cells"]]
    for k in ("td_poly", "gid", "filled", "one_based"):
        rec.pop(k, None)
    return rec


def convert(a):
    w = TsrWriter(a.out, "tabrecset")
    stats = collections.Counter()
    fits = {"photo": [], "rectified_before": [], "rectified_after": [], "dropped": []}
    for lang in LANGS:
        for v in VARIANTS:
            d = a.raw / "image" / f"{lang}_{v}"
            if not d.exists():
                continue
            for ip in sorted(d.glob("*.jpg")):
                stem = ip.stem
                split = tsr_split(stem)
                if v != "all-line" and split != "train":
                    stats["variant outside train (skipped)"] += 1
                    continue
                tsr_p = a.raw / "ann" / "TSR" / lang / f"{stem}.json"
                td_p = a.raw / "ann" / "TD" / lang / f"{stem}.json"
                if not tsr_p.exists():
                    stats["no annotation"] += 1
                    continue
                img = cv2.imread(str(ip))
                if img is None:
                    stats["undecodable"] += 1
                    continue
                tsr = json.loads(tsr_p.read_text(encoding="utf-8"))
                td = json.loads(td_p.read_text(encoding="utf-8")) if td_p.exists() else None
                recs, problems = tables_from_json(tsr, td)
                for p in problems:
                    w.errors.append(f"{lang}/{v}/{stem}: {p}")
                    stats["inconsistent grid (skipped)"] += 1
                for k, rec in enumerate(recs):
                    stats["filled_empty_cells"] += rec["filled"]
                    stats["tables"] += 1
                    if rec["R"] * rec["C"] < 2:
                        stats["dropped: 1x1 table"] += 1
                        continue
                    try:
                        fit = table_fit(rec)
                    except ValueError as e:
                        w.errors.append(f"{lang}/{v}/{stem}/t{k}: {e}")
                        stats["label error"] += 1
                        continue
                    src, mode, fit_after = img, "photo", fit
                    if fit < a.min_fit:
                        r2 = rectify(img, rec)
                        fit_after = table_fit(r2[1]) if r2 else 0.0
                        fits["rectified_before"].append(fit)
                        if r2 is None or fit_after < a.min_fit:
                            stats["dropped: low fit after rectification" if r2 else "dropped: no corner quad"] += 1
                            fits["dropped"].append(fit)
                            continue
                        src, rec, mode = r2[0], r2[1], "rectified"
                        fits["rectified_after"].append(fit_after)
                    else:
                        fits["photo"].append(fit)
                    out = _strip(rec)
                    out.update(source="tabrecset", doc_id=group_key(stem), src_split="-", src_name=f"{lang}/{v}/{stem}", nc=False,
                               lang="zh" if lang == "chinese" else "en", variant=v,
                               borders="grid" if v == "all-line" else ("hlines" if v == "three-line" else "none"))
                    e = w.add(split, f"{lang[:2]}_{stem}_{v}_t{k}", src, out)
                    if e is None:
                        stats["label error"] += 1
                        continue
                    e.update(geom=mode, fit=round(fit, 3), fit_after=round(fit_after, 3))
                    stats[f"kept {mode}"] += 1
                    stats[f"kept {mode} {split}"] += 1
            print(lang, v, dict(stats), flush=True)
    s = w.close()
    s["conversion"] = dict(stats)
    s["fit"] = {k: {"n": len(v), "p10_50_90": [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])] if v else []}
                for k, v in fits.items()}
    s["min_fit"] = a.min_fit
    s["column_header"] = "not labelled in TabRecSet (header_rows=None): no column_header boxes in this set"
    json.dump(s, open(a.out / "summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "convert"])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--per-lang", type=int, default=1500)
    ap.add_argument("--ann-only", action="store_true")
    ap.add_argument("--variants", default="no-line", help="line-erased variants to fetch for the train images (no-line,three-line)")
    ap.add_argument("--min-fit", type=float, default=0.90, help="min cell-polygon-in-grid-box fit to keep a table (else rectify / drop)")
    a = ap.parse_args()
    a.raw.mkdir(parents=True, exist_ok=True)
    fetch(a) if a.cmd == "fetch" else convert(a)


if __name__ == "__main__":
    main()
