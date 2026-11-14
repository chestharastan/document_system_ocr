"""Run the REAL backend pipeline (layout -> text lines -> recognition -> bullets / markers -> rules) on generated pages and
record exactly what the structure stage gets as input, matched to the generator's structure labels.

    python -m training.structure.pipe_extract --manifest DIR/manifest.jsonl --out OUT.jsonl [--shard i/n] [--limit N]

Output record per page: {"image", "W", "H", "doc_type", "lines": [{"bbox", "text", "bold", "bullet"}], "role", "first",
"level", "bullet", "rank"} - label lists aligned with lines; -100 = unknown (a detected line that matches no printed
line: stamp text, handwriting, noise). rank = reading position (lines sharing a printed line are ordered left to
right). The structure model trains on these (training.structure.data.load_pipe_pages) so its features match inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "backend"


def _box(q):
    xs, ys = q[0::2], q[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


class Capture:
    """Stand-in for the structure stage: records its input, changes nothing (role_p 0, rule paragraphs kept)."""
    stage = "structure"

    def __init__(self):
        self.got = None

    def info(self):
        return {"stage": "structure", "id": "capture"}

    def predict(self, lines, W, H):
        self.got = [{"bbox": [round(float(v), 1) for v in l["bbox"]], "text": l.get("text", ""), "bold": bool(l.get("bold")),
                     "bullet": l.get("bullet") or None} for l in lines]
        return [{"role": "text", "role_p": 0.0, "first": False, "first_p": 0.0, "level": 0, "bullet": bool(l.get("bullet")), "order": i}
                for i, l in enumerate(lines)]


def label_page(rec: dict, det: list[dict]) -> dict:
    from training.structure.data import ROLE_ID, _resolve_role, MAX_LEVEL
    gt = [(_box(q), t, s) for q, t, s in zip(rec["lines"], rec["line_texts"], rec["struct"])]
    objs = rec.get("objs") or []
    text_idx = [i for i, (_, _, s) in enumerate(gt) if not s["marker"]]
    rank_of = {i: r for r, i in enumerate(text_idx)}
    para_first = {}
    for i in text_idx:
        para_first.setdefault(gt[i][2]["para"], i)
    hw = [_box(q) for q in rec.get("hw_lines") or []]
    n = len(det)
    match = [None] * n          # gt index
    is_marker = [False] * n
    for k, d in enumerate(det):
        b = d["bbox"]
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        best, bv = None, 0.0
        for i, (g, _, s) in enumerate(gt):
            v = _inter(b, g) / area
            if v > bv:
                best, bv = i, v
        if any(_inter(b, g) / area > max(bv, 0.3) for g in hw):
            continue
        if best is None or bv < 0.3:
            continue
        if gt[best][2]["marker"]:
            p = gt[best][2]["para"]
            if p not in para_first:
                continue
            is_marker[k] = True
            match[k] = para_first[p]
        else:
            match[k] = best
    # order: by the matched printed line, a marker before its line, pieces of one line left to right
    keyed = sorted([k for k in range(n) if match[k] is not None],
                   key=lambda k: (rank_of[match[k]], 0 if is_marker[k] else 1, det[k]["bbox"][0]))
    rank = [-100] * n
    for r, k in enumerate(keyed):
        rank[k] = r
    role, first, level, bullet = [-100] * n, [-100] * n, [-100] * n, [-100] * n
    seen_para = set()
    for k in keyed:
        i = match[k]
        g, _, s = gt[i]
        r = _resolve_role(s["role"], g, objs)
        role[k] = ROLE_ID.get(r, ROLE_ID["text"])
        level[k] = min(MAX_LEVEL, int(s["level"]))
        is_first = s["para"] not in seen_para
        seen_para.add(s["para"])
        first[k] = 1 if is_first else 0
        bullet[k] = 1 if (is_first and r == "list_item") else 0
    # v3: table cell labels (docgen v5 ``tables``) and the page kind, carried over to the matched detected lines
    from training.structure.data import kind_label_from_record, table_labels_from_record
    gt_roles = [ROLE_ID.get(_resolve_role(s["role"], g, objs), ROLE_ID["text"]) for g, _, s in gt]
    tl = table_labels_from_record(rec, {i: i for i in range(len(gt))}, gt_roles)
    tab = {key: [-100] * n for key in ("tid", "row", "col", "cell")}
    for k in keyed:
        if not is_marker[k]:
            for key in tab:
                tab[key][k] = int(tl[key][match[k]])
    return {"role": role, "first": first, "level": level, "bullet": bullet, "rank": rank, **tab, "kind": kind_label_from_record(rec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.environ["ORT_DEVICE"] = a.device
    os.environ.setdefault("ORT_THREADS", "4")
    sys.path.insert(0, str(BACKEND))
    os.chdir(BACKEND)
    from app.services.model_pool import ModelPool
    from app.services.pipeline_service import PipelineService
    cap = Capture()
    ModelPool.structure = property(lambda self: cap)  # the pipeline asks the pool for its structure stage
    pool = ModelPool()
    pipe = PipelineService(pool)
    si, sn = (int(v) for v in a.shard.split("/"))
    root = Path(a.manifest).parent
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["image"] for l in open(a.out)}
    tmp = Path(tempfile.mkdtemp(prefix="pipe_extract_"))
    t0, k = time.time(), 0
    with open(a.manifest) as fh, open(a.out, "a") as out:
        for i, line in enumerate(fh):
            if i % sn != si:
                continue
            if a.limit and k >= a.limit:
                break
            rec = json.loads(line)
            if not rec.get("struct") or rec["image"] in done:
                continue
            cap.got = None
            try:
                res = pipe.process_image(str(root / rec["image"]), crop_dir=str(tmp), crop_url_prefix="", recognize=True)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {rec['image']}: {e!r}", flush=True)
                continue
            finally:
                for f in tmp.iterdir():
                    f.unlink()
            det = cap.got or []
            if len(det) < 2:
                continue
            lab = label_page(rec, det)
            out.write(json.dumps({"image": rec["image"], "W": res["width"], "H": res["height"], "doc_type": rec.get("doc_type"),
                                  "lines": det, **lab}, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()
            k += 1
            if k % 50 == 0:
                print(f"{k} pages {time.time() - t0:.0f}s ({k / (time.time() - t0):.2f} p/s)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"done {k} pages in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
