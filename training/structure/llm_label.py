"""The LLM teacher: label real pages for the structure model with the LLM structure engine (contract §1 goal 2).

    python -m training.structure.llm_label --inputs work/data/struct_v2/train/pipe_*.jsonl --out labels.jsonl [--limit N]
                                           [--workers 1] [--max-bad 0.3] [--doc-types km_contract,...] [--progress p.json]
    python -m training.structure.llm_label --tabrecset work/data/ext/tables_tabrecset/raw --tabrecset-split train ...
    python -m training.structure.llm_label --dump-tabrecset work/data/ext/tables_tabrecset/raw --out pages.jsonl --limit N

Never point it at work/data/user_docs or work/data/eval_nc (evaluation only): those paths are refused.

Input records are pipe_extract pages ({"image", "W", "H", "lines": [{"bbox", "text", "bold", "bullet"}], ...}; the
existing label lists are ignored) or full process results ({"width", "height", "blocks": [...]}, lines taken like the
backend does). One LLM call per page (sequential by default; ``--workers`` > 1 only for endpoints that allow it).

Output: one record per page in the structure-training format, loadable by training.structure.data.page_from_pipe:
    {"image", "W", "H", "doc_type", "lines": [{"bbox", "text", "bold", "bullet"}],
     "role": [ROLE id per line], "first": [1 = first line of its element / cell], "level": [list depth, 0 for
     headings / header / footer / page number, -100 = unknown], "bullet": [1 = first line of a list item],
     "rank": [reading position], "tid" / "row" / "col" / "cell": [per line table id, row, column, cell id; -1 outside
     tables] (the v3 table heads, like data.table_labels_from_record), "kind": KIND_ID of the page kind,
     "tables": [contract §5 tables, page pixels, line_ids into "lines"], "doc_kind", "model", "provider", "latency_ms",
     "llm_tokens", "repairs", "source"; TabRecSet pages also keep their human labels as "gt_tables" (for agreement)}
Resumable (pages already in --out are skipped); pages whose answer failed, or where the model left more than
``--max-bad`` of the lines unplaced / invalid, are skipped and listed in <out>.skipped.jsonl."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.llm_providers import LLMError, get_provider, llm_status  # noqa: E402
from app.services import llm_structure as LS  # noqa: E402
from app.services.llm_structure import page_lines, structure_lines, table_structure  # noqa: E402
from training.structure.data import KIND_ID, MAX_LEVEL, ROLE_ID  # noqa: E402

FORBIDDEN = ("user_docs", "eval_nc")  # evaluation-only data: never labelled / trained on

UNKNOWN = -100
TYPE_ROLE = {"para": "text", "list": "list_item", "header": "header", "footer": "footer", "pagenum": "page_number",
             "caption": "caption", "footnote": "footnote", "field": "form_field", "table": "table"}


def lines_from_record(rec: dict) -> tuple[list[dict], int, int, list | None]:
    """-> (neutral lines for the engine, W, H, process-result blocks or None)."""
    if "blocks" in rec:
        lines = page_lines(rec)
        return lines, int(rec["width"]), int(rec["height"]), rec.get("blocks")
    out = []
    for ln in rec["lines"]:
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = [float(v) for v in ln["bbox"]]
        out.append({"id": len(out) + 1, "bi": 0, "li": len(out), "block": 1, "cls": "text", "box": [x0, y0, x1, y1],
                    "font_px": float(ln.get("font_px") or (y1 - y0)), "bold": bool(ln.get("bold")), "bullet": ln.get("bullet") or "",
                    "text": text, "confidence": float(ln.get("confidence", 1.0) or 1.0)})
    blocks = None
    if rec.get("layout"):  # layout hints [{"cls", "box": [x0, y0, x1, y1]}] (what the app's layout model would give)
        blocks = [{"cls": b["cls"], "reading_order": k + 1,
                   "box": {"x": b["box"][0], "y": b["box"][1], "width": b["box"][2] - b["box"][0], "height": b["box"][3] - b["box"][1]}}
                  for k, b in enumerate(rec["layout"])]
        n = len(blocks)
        for ln in out:
            cx, cy = (ln["box"][0] + ln["box"][2]) / 2, (ln["box"][1] + ln["box"][3]) / 2
            for k, b in enumerate(rec["layout"]):
                if b["box"][0] <= cx <= b["box"][2] and b["box"][1] <= cy <= b["box"][3]:
                    ln["block"], ln["cls"] = k + 1, b["cls"]
                    break
            else:
                ln["block"] = n + 1
    return out, int(rec["W"]), int(rec["H"]), blocks


def answer_to_labels(answer: dict, lines: list[dict], kind: str) -> dict:
    """Validated answer -> role / first / level / bullet / rank lists aligned with ``lines`` + contract §5 tables."""
    n = len(lines)
    pos = {l["id"]: k for k, l in enumerate(lines)}
    role, first, level, bullet, rank = ([UNKNOWN] * n for _ in range(5))
    tid, row, col, cid = ([-1] * n for _ in range(4))
    tables = []
    r = 0
    for el in answer["els"]:
        if el["t"] == "table":
            cells = []
            for cell in sorted(el["c"], key=lambda c: (c["r"], c["c"])):
                idx = [pos[i] for i in cell["l"]]
                for j, k in enumerate(idx):
                    role[k], first[k], level[k], bullet[k], rank[k] = ROLE_ID["table"], 1 if j == 0 else 0, 0, 0, r
                    r += 1
                cells.append({**cell, "idx": idx})
                for k in idx:
                    tid[k], row[k], col[k], cid[k] = len(tables), cell["r"], cell["c"], len(tables) * 10000 + len(cells) - 1
            t = table_structure(cells, [l["box"] for l in lines], [l["text"] for l in lines], [l["bold"] for l in lines],
                                [l["confidence"] for l in lines])
            tables.append({"box": t["box"], "rows": t["rows"], "cols": t["cols"], "header_rows": t["header_rows"],
                           "borders": t["borders"], "sheet": kind == "sheet",
                           "cells": [{"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "box": c["box"], "header": c["header"],
                                      "text": c["text"], "align": c["align"], "bold": c["bold"], "line_ids": c["lines"]} for c in t["cells"]]})
            continue
        if el["t"] == "heading":
            rname = "title" if el["lv"] <= 1 else "section_header"
        else:
            rname = TYPE_ROLE.get(el["t"], "text")
        is_list = el["t"] == "list"
        for j, lid in enumerate(el["l"]):
            k = pos[lid]
            role[k], first[k], rank[k] = ROLE_ID[rname], 1 if j == 0 else 0, r
            bullet[k] = 1 if (is_list and j == 0) else 0
            level[k] = min(MAX_LEVEL, el["lv"]) if is_list else (0 if el["t"] in ("heading", "header", "footer", "pagenum") else UNKNOWN)
            r += 1
    return {"role": role, "first": first, "level": level, "bullet": bullet, "rank": rank, "tid": tid, "row": row, "col": col,
            "cell": cid, "kind": KIND_ID[kind], "tables": tables}


def wait_idle(provider, max_wait_s: float = 3600.0, poll_s: float = 15.0) -> float:
    """Wait until the server has a free slot (someone else's request finished): -> seconds waited. The labeller is a
    background job and must not hold the owner's interactive requests behind a queue of its own pages."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s and getattr(provider, "busy", lambda: None)():
        time.sleep(poll_s)
    return time.time() - t0


def label_record(rec: dict, provider, max_bad: float, wait_s: float = 0.0) -> tuple[dict | None, str | None]:
    """-> (output record, None) or (None, reason) when the page is skipped. ``wait_s`` > 0: first wait up to that long
    for an idle server (otherwise the request simply queues on the server; llama.cpp serves requests in order)."""
    lines, W, H, blocks = lines_from_record(rec)
    if len(lines) < 2:
        return None, "fewer than 2 lines"
    if wait_s > 0:
        waited = wait_idle(provider, wait_s)
        if waited > 1:
            print(f"  waited {waited:.0f}s for a free slot", flush=True)
    t0 = time.perf_counter()
    try:
        answer, meta = structure_lines(lines, W, H, provider, blocks)
    except LLMError as exc:
        return None, f"llm error: {exc}"
    bad = meta["repairs"]["unused"] + meta["repairs"]["bad_ids"] + meta["repairs"]["dup_ids"]
    if bad > max_bad * len(lines):
        return None, f"validation: {bad} of {len(lines)} lines unplaced / invalid"
    labels = answer_to_labels(answer, lines, answer["kind"])
    out = {"image": rec.get("image"), "W": W, "H": H, "doc_type": rec.get("doc_type"), "source": rec.get("source"),
           "layout": rec.get("layout"),
           "lines": [{"bbox": [round(v, 1) for v in l["box"]], "text": l["text"], "bold": l["bold"], "bullet": l["bullet"] or None} for l in lines],
           **labels, "doc_kind": answer["kind"], "model": meta["model"], "provider": meta["provider"],
           "latency_ms": round((time.perf_counter() - t0) * 1000), "llm_tokens": meta["tokens"], "llm_calls": meta["calls"],
           "server_ms": round(sum(v for k, v in (meta.get("server_timings") or {}).items() if k.endswith("_ms"))),
           "repairs": meta["repairs"], "llm_raw": meta.get("raw")}   # raw answers: labels can be re-derived later
    if rec.get("gt_tables"):
        out["gt_tables"] = rec["gt_tables"]
    return out, None


def relabel_record(out: dict) -> dict:
    """A labelled record with ``llm_raw`` -> the same record with its labels re-derived by the CURRENT validation /
    repair code (no LLM call): improvements to the repairs reach labels made earlier."""
    lines, W, H, _ = lines_from_record(out)
    raws = out.get("llm_raw") or []
    if not raws:
        return out
    merged = LS._merge_bands(raws, LS._bands(lines)) if len(raws) > 1 else raws[0]
    answer, rep = LS.validate_and_repair(merged, [l["id"] for l in lines], {l["id"]: l["box"] for l in lines})
    labels = answer_to_labels(answer, lines, answer["kind"])
    return {**out, **labels, "doc_kind": answer["kind"], "repairs": rep, "relabelled": True}


# ------------------------------------------------------------------------------------------------ TabRecSet photos
def tabrecset_records(raw: str, lang: str = "english", split: str | None = "train", limit: int | None = None,
                      skip: int = 0, max_cells: int = 160):
    """TabRecSet (CC BY 4.0) photos -> pipe-format records whose lines are the human-transcribed cell texts (one line
    per printed line of a cell: multi-line cells are cut into equal-height strips of the cell box, shrunk a little
    toward the text). The image name decides the split exactly like training.data.ext_tables_tabrecset (60/10/30), so
    only ``split`` pages are produced. The human cell labels go along as ``gt_tables`` (line ids into ``lines``)."""
    import cv2  # noqa: F401  (ext_tables_tabrecset needs it)
    from training.data.ext_tables_common import split_for
    from training.data.ext_tables_tabrecset import tables_from_json
    root = Path(raw)
    ann = sorted((root / "ann" / "TSR" / lang).glob("*.json"))
    n = 0
    for k, p in enumerate(ann):
        if k < skip:
            continue
        if split and split_for(p.stem, 0.6, 0.1, "tabrecset") != split:
            continue
        img = root / "image" / f"{lang}_all-line" / (p.stem + ".jpg")
        if not img.exists():
            continue
        try:
            tsr = json.loads(p.read_text(encoding="utf-8"))
            tables, problems = tables_from_json(tsr, None)
        except Exception:  # noqa: BLE001 - a broken annotation is skipped
            continue
        if not tables or problems or sum(len(t["cells"]) for t in tables) > max_cells:
            continue
        W, H = int(tsr.get("imageWidth") or 0), int(tsr.get("imageHeight") or 0)
        if W <= 0 or H <= 0:
            continue
        lines, gt, layout = [], [], []   # layout = table boxes, the hint the app's layout model gives
        for t in tables:
            cells = []
            for c in t["cells"]:
                parts = [s.strip() for s in (c.get("text") or "").split("\n") if s.strip()] if c.get("box") else []
                x0, y0, x1, y1 = c.get("box") or (0, 0, 0, 0)
                ids = []
                for j, txt in enumerate(parts):
                    h = (y1 - y0) / len(parts)
                    a, b = y0 + j * h, y0 + (j + 1) * h
                    ids.append(len(lines))
                    lines.append({"bbox": [round(x0 + 0.06 * (x1 - x0), 1), round(a + 0.15 * h, 1), round(x1 - 0.06 * (x1 - x0), 1), round(b - 0.15 * h, 1)],
                                  "text": txt, "bold": False, "bullet": None})
                cells.append({"r": c["r"], "c": c["c"], "rs": c["rs"], "cs": c["cs"], "line_ids": ids})
            gt.append({"rows": t["R"], "cols": t["C"], "cells": cells})
            boxes_t = [c["box"] for c in t["cells"] if c.get("box")]
            if boxes_t:
                layout.append({"cls": "table", "box": [round(min(b[0] for b in boxes_t), 1), round(min(b[1] for b in boxes_t), 1),
                                                       round(max(b[2] for b in boxes_t), 1), round(max(b[3] for b in boxes_t), 1)]})
        if len(lines) < 4:
            continue
        yield {"image": str(img), "W": W, "H": H, "doc_type": f"tabrecset_{lang}", "source": "tabrecset", "lines": lines,
               "gt_tables": gt, "gt_kind": "sheet", "layout": layout}
        n += 1
        if limit and n >= limit:
            return


def iter_records(paths: list[str], image_root: str | None = None):
    """Records of the input jsonl files; a relative "image" becomes absolute (against ``image_root`` or the file's
    directory) so labels from several splits / sets can share one manifest."""
    for pattern in paths:
        for p in sorted(glob.glob(pattern)) or [pattern]:
            base = Path(image_root) if image_root else Path(p).resolve().parent
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    img = rec.get("image")
                    if isinstance(img, str) and img and not Path(img).is_absolute():
                        rec["image"] = str(base / img)
                    rec.setdefault("source", Path(p).parent.name + "/" + Path(p).name)
                    yield rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=[], help="pipe_extract jsonl files (globs ok) or process-result jsonl")
    ap.add_argument("--tabrecset", default=None, help="TabRecSet raw dir (ann/TSR/<lang>, image/<lang>_all-line): label its photos")
    ap.add_argument("--tabrecset-lang", default="english")
    ap.add_argument("--tabrecset-split", default="train")
    ap.add_argument("--dump-tabrecset", default=None, help="only write TabRecSet pages as pipe-format records to --out (no LLM)")
    ap.add_argument("--wait-idle", type=float, default=0.0, help="wait up to N s for an idle server before each page "
                                                                     "(0 = just queue on the server)")
    ap.add_argument("--max-hours", type=float, default=None, help="stop starting new pages after this much wall time")
    ap.add_argument("--relabel", default=None, help="re-derive the labels of this labelled jsonl from its llm_raw "
                                                     "answers with the current repair code -> --out (no LLM calls)")
    ap.add_argument("--progress", default=None, help="write a JSON progress file after every page")
    ap.add_argument("--image-root", default=None, help="make relative 'image' paths absolute against this directory "
                                                         "(default: the input file's directory)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="label at most N new pages")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests (keep 1 for a shared llama-server)")
    ap.add_argument("--max-bad", type=float, default=0.3, help="skip a page when more than this share of its lines was unplaced / invalid")
    ap.add_argument("--doc-types", default=None, help="comma-separated doc_type filter")
    a = ap.parse_args(argv)
    for p in list(a.inputs) + [a.tabrecset or "", a.dump_tabrecset or "", a.out]:
        if any(f in Path(p).parts or f"/{f}/" in p for f in FORBIDDEN):
            print(f"refusing {p}: user_docs / eval_nc are evaluation data, never labelled", file=sys.stderr)
            return 2
    if a.relabel:
        n = 0
        with open(a.relabel, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip():
                    fout.write(json.dumps(relabel_record(json.loads(line)), ensure_ascii=False, separators=(",", ":")) + "\n")
                    n += 1
        print(f"{n} records relabelled -> {a.out}")
        return 0
    if a.dump_tabrecset:
        with open(a.out, "w", encoding="utf-8") as fh:
            k = 0
            for k, rec in enumerate(tabrecset_records(a.dump_tabrecset, a.tabrecset_lang, a.tabrecset_split, a.limit), start=1):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{k} TabRecSet pages -> {a.out}")
        return 0
    provider = get_provider()
    st = llm_status()
    if provider is None or not st["reachable"]:
        print(f"LLM not available: {st}", file=sys.stderr)
        return 2
    print(f"provider {st}", flush=True)
    out_path = Path(a.out)
    skipped_path = out_path.with_suffix(out_path.suffix + ".skipped.jsonl")
    done: set = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["image"])
                except (json.JSONDecodeError, KeyError):
                    pass
    types = set(a.doc_types.split(",")) if a.doc_types else None
    todo = []
    def records():
        yield from iter_records(a.inputs, a.image_root)
        if a.tabrecset:
            yield from tabrecset_records(a.tabrecset, a.tabrecset_lang, a.tabrecset_split)

    for i, rec in enumerate(records()):
        img = rec.get("image") or rec.get("page_id") or rec.get("filename") or f"record#{i}"
        rec["image"] = img
        if img in done or (types and rec.get("doc_type") not in types):
            continue
        todo.append(rec)
        done.add(img)
        if a.limit and len(todo) >= a.limit:
            break
    print(f"{len(todo)} pages to label ({len(done) - len(todo)} already done)", flush=True)
    lock = threading.Lock()
    lat, ok, skipped, tokens = [], 0, 0, {"prompt_tokens": 0, "completion_tokens": 0}
    t_start = time.time()
    kinds: dict = {}

    def progress(state="running", last=None):
        if not a.progress:
            return
        el = time.time() - t_start
        Path(a.progress).write_text(json.dumps({
            "state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "out": str(out_path), "todo": len(todo),
            "labelled": ok, "skipped": skipped, "elapsed_s": round(el), "kinds": kinds, "tokens": tokens,
            "median_s": round(statistics.median(lat) / 1000, 1) if lat else None,
            "eta_s": round(el / max(1, ok + skipped) * (len(todo) - ok - skipped)) if ok + skipped else None,
            "last": last, "provider": st}, ensure_ascii=False, indent=1))
    progress()

    def work(rec):
        nonlocal ok, skipped
        res, reason = label_record(rec, provider, a.max_bad, a.wait_idle)
        with lock:
            if res is None:
                skipped += 1
                with open(skipped_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"image": rec.get("image"), "reason": reason}, ensure_ascii=False) + "\n")
                print(f"[skip] {rec.get('image')}: {reason}", flush=True)
                progress(last={"image": rec.get("image"), "skipped": reason})
                return
            ok += 1
            lat.append(res["latency_ms"])
            for k in tokens:
                tokens[k] += res["llm_tokens"].get(k, 0)
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(res, ensure_ascii=False, separators=(",", ":")) + "\n")
            n_tab = len(res["tables"])
            kinds[res["doc_kind"]] = kinds.get(res["doc_kind"], 0) + 1
            progress(last={"image": rec.get("image"), "lines": len(res["lines"]), "kind": res["doc_kind"], "tables": n_tab,
                           "s": round(res["latency_ms"] / 1000, 1)})
            print(f"[ok] {rec.get('image')}: {len(res['lines'])} lines kind={res['doc_kind']} tables={n_tab} "
                  f"{res['latency_ms'] / 1000:.1f}s tokens={res['llm_tokens']} repairs={res['repairs']}", flush=True)

    if a.workers > 1:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, todo))
    else:
        for rec in todo:
            if a.max_hours and time.time() - t_start > a.max_hours * 3600:
                print(f"time budget of {a.max_hours} h used up: stopping", flush=True)
                break
            work(rec)
    stats = getattr(provider, "stats", {})
    calls = max(1, stats.get("calls", 0))
    print(f"done: {ok} labelled, {skipped} skipped in {time.time() - t_start:.0f}s; "
          f"latency median {statistics.median(lat) / 1000:.1f}s mean {statistics.mean(lat) / 1000:.1f}s max {max(lat) / 1000:.1f}s; "
          f"tokens {tokens}; provider calls {stats.get('calls')} repairs {stats.get('repairs')} truncated {stats.get('truncated')} "
          f"failed {stats.get('failed')} -> first-try JSON validity {(calls - stats.get('repairs', 0) - stats.get('failed', 0)) / calls:.0%}"
          if lat else f"done: nothing labelled, {skipped} skipped", flush=True)
    progress("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
