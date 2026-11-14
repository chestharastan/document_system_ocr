"""Cell-level table-structure evaluation (docs/TABLES_CONTRACT.md §6): adjacency-relation F1 and TEDS-struct.

    python -m training.table.eval_tsr --gt GT.json --pred PRED.json [--iou 0.5] [--match iou|text] [--blank skip|keep]
                                      [--per-table] [--json OUT.json]

Inputs are JSON lists of tables, aligned by index (or dicts with the same keys), each with
``"cells": [{"r", "c", "rs", "cs", "box": [x0,y0,x1,y1], "text"?, "header"?}]`` (the docgen v5 / tsr_v1 ``tables.jsonl``
format and the pipeline's ``blocks[i].table`` format). A file may also hold ``{"tables": [...]}`` or a list of pipeline
results (``blocks`` with ``cls == "table"``); ``tables.jsonl`` (one table per line) is accepted too. Missing predictions
(``null`` / no cells) count as empty tables.

Metrics (micro over all tables + macro mean per table):
- **adjacency F1** (ICDAR-2013 style): every cell is linked to its nearest neighbour to the right and below (blank
  cells skipped with --blank skip when texts are available); predicted cells are matched to ground-truth cells by box
  IoU >= --iou (or by exact text with --match text); a relation is correct when both ends map to matched cells with the
  same relation in the ground truth.
- **TEDS-struct**: tree edit distance (Zhang-Shasha, unit costs) between the HTML structure trees
  ``table > thead/tbody > tr > td[colspan,rowspan]`` ignoring the text, normalised by the larger tree:
  ``1 - TED / max(|T_gt|, |T_pred|)``.
- extras: exact structure match rate, row / column count accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path


# --------------------------------------------------------------------------- loading
def _as_tables(obj):
    """Flatten the accepted input shapes into a list of tables ({"cells": [...]} or None)."""
    if obj is None:
        return [None]
    if isinstance(obj, dict):
        if "cells" in obj:
            return [obj]
        if "tables" in obj:
            return [t for t in obj["tables"]] if obj["tables"] else []
        if "table" in obj and isinstance(obj["table"], dict):
            return [obj["table"]]
        if "blocks" in obj:
            return [b["table"] for b in obj["blocks"] if b.get("cls") == "table" and isinstance(b.get("table"), dict)]
        return [None]
    if isinstance(obj, list):
        out = []
        for it in obj:
            out += _as_tables(it)
        return out
    return [None]


def load_tables(path: Path):
    txt = Path(path).read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or (txt.lstrip()[:1] == "{" and "\n{" in txt):
        return [t for ln in txt.splitlines() if ln.strip() for t in _as_tables(json.loads(ln))]
    return _as_tables(json.loads(txt))


def norm_cells(t):
    if not t or not t.get("cells"):
        return []
    out = []
    for c in t["cells"]:
        out.append({"r": int(c["r"]), "c": int(c["c"]), "rs": max(1, int(c.get("rs", 1))), "cs": max(1, int(c.get("cs", 1))),
                    "box": [float(v) for v in c["box"]] if c.get("box") else None, "text": (c.get("text") or "").strip(), "header": bool(c.get("header"))})
    return out


# --------------------------------------------------------------------------- matching
def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_cells(gt, pred, thr=0.5, mode="iou"):
    """Greedy one-to-one matching; returns pred index -> gt index."""
    m = {}
    if mode == "text":
        pool = {}
        for j, g in enumerate(gt):
            if g["text"]:
                pool.setdefault(g["text"], []).append(j)
        for i, p in enumerate(pred):
            lst = pool.get(p["text"])
            if lst:
                m[i] = lst.pop(0)
        return m
    pairs = []
    for i, p in enumerate(pred):
        if p["box"] is None:
            continue
        for j, g in enumerate(gt):
            if g["box"] is None:
                continue
            v = iou(p["box"], g["box"])
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used = set()
    for v, i, j in pairs:
        if i in m or j in used:
            continue
        m[i] = j
        used.add(j)
    return m


# --------------------------------------------------------------------------- adjacency relations
def relations(cells, skip_blank=True):
    """Set of (i, j, 'h'|'v'): cell i's nearest non-blank neighbour j to the right / below (ICDAR-2013 style)."""
    R = max((c["r"] + c["rs"] for c in cells), default=0)
    C = max((c["c"] + c["cs"] for c in cells), default=0)
    grid = [[None] * C for _ in range(R)]
    for i, c in enumerate(cells):
        for a in range(c["r"], c["r"] + c["rs"]):
            for b in range(c["c"], c["c"] + c["cs"]):
                if a < R and b < C:
                    grid[a][b] = i
    blank = [skip_blank and not c["text"] for c in cells]
    rel = set()
    for i, c in enumerate(cells):
        if blank[i]:
            continue
        # right
        for a in range(c["r"], c["r"] + c["rs"]):
            b = c["c"] + c["cs"]
            while b < C:
                j = grid[a][b]
                if j is not None and j != i and not blank[j]:
                    rel.add((i, j, "h"))
                    break
                b += 1 if j is None else cells[j]["cs"]
        # below
        for b in range(c["c"], c["c"] + c["cs"]):
            a = c["r"] + c["rs"]
            while a < R:
                j = grid[a][b]
                if j is not None and j != i and not blank[j]:
                    rel.add((i, j, "v"))
                    break
                a += 1 if j is None else cells[j]["rs"]
    return rel


def adjacency_counts(gt, pred, m, skip_blank=True):
    """(tp, n_pred, n_gt) of adjacency relations after mapping predicted cells to ground-truth ids."""
    rg = relations(gt, skip_blank)
    rp = relations(pred, skip_blank)
    mapped = set()
    for (i, j, d) in rp:
        mapped.add((m.get(i, -1 - i), m.get(j, -1 - j), d))
    tp = len(mapped & rg)
    return tp, len(mapped), len(rg)


# --------------------------------------------------------------------------- TEDS-struct
class Node:
    __slots__ = ("label", "children")

    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []


def html_tree(cells, header_rows=None):
    """table > (thead > tr*)? > tbody > tr* > td[colspan,rowspan]; rows are the anchor rows of the cells."""
    R = max((c["r"] + c["rs"] for c in cells), default=0)
    rows = [[] for _ in range(R)]
    for c in cells:
        if c["r"] < R:
            rows[c["r"]].append(c)
    if header_rows is None:
        header_rows = 0
        for r in range(R):
            if rows[r] and all(c["header"] for c in rows[r]):
                header_rows = r + 1
            else:
                break
    def tr(r):
        return Node("tr", [Node(("td", c["cs"], c["rs"])) for c in sorted(rows[r], key=lambda c: c["c"])])
    kids = []
    if header_rows > 0:
        kids.append(Node("thead", [tr(r) for r in range(header_rows)]))
    kids.append(Node("tbody", [tr(r) for r in range(header_rows, R)]))
    return Node("table", kids)


def tree_size(n):
    return 1 + sum(tree_size(c) for c in n.children)


def _postorder(root):
    nodes, lmd = [], []

    def walk(n):
        first = None
        for c in n.children:
            k = walk(c)
            if first is None:
                first = k
        nodes.append(n)
        idx = len(nodes) - 1
        lmd.append(first if first is not None else idx)
        return lmd[-1]
    walk(root)
    return nodes, lmd


def tree_edit_distance(a: Node, b: Node) -> int:
    """Zhang-Shasha ordered tree edit distance with unit insert / delete / relabel costs."""
    A, la = _postorder(a)
    B, lb = _postorder(b)
    na, nb = len(A), len(B)

    def keyroots(lm):  # nodes with no later (= ancestor, in post-order) node sharing their leftmost leaf
        seen, out = set(), []
        for i in range(len(lm) - 1, -1, -1):
            if lm[i] not in seen:
                seen.add(lm[i])
                out.append(i)
        return sorted(out)

    ka, kb = keyroots(la), keyroots(lb)
    td = [[0] * nb for _ in range(na)]

    def fd(i, j):
        li, lj = la[i], lb[j]
        m, n = i - li + 2, j - lj + 2
        D = [[0] * n for _ in range(m)]
        for x in range(1, m):
            D[x][0] = D[x - 1][0] + 1
        for y in range(1, n):
            D[0][y] = D[0][y - 1] + 1
        for x in range(1, m):
            for y in range(1, n):
                ii, jj = li + x - 1, lj + y - 1
                if la[ii] == li and lb[jj] == lj:
                    cost = 0 if A[ii].label == B[jj].label else 1
                    D[x][y] = min(D[x - 1][y] + 1, D[x][y - 1] + 1, D[x - 1][y - 1] + cost)
                    td[ii][jj] = D[x][y]
                else:
                    D[x][y] = min(D[x - 1][y] + 1, D[x][y - 1] + 1, D[la[ii] - li][lb[jj] - lj] + td[ii][jj])
    for i in ka:
        for j in kb:
            fd(i, j)
    return td[na - 1][nb - 1]


def teds_struct(gt_cells, pred_cells, gt_header_rows=None, pred_header_rows=None) -> float:
    tg = html_tree(gt_cells, gt_header_rows)
    tp = html_tree(pred_cells, pred_header_rows)
    n = max(tree_size(tg), tree_size(tp))
    if n == 0:
        return 1.0
    return 1.0 - tree_edit_distance(tg, tp) / n


# --------------------------------------------------------------------------- driver
def evaluate(gt_tables, pred_tables, iou_thr=0.5, match="iou", skip_blank=True):
    n = len(gt_tables)
    if len(pred_tables) < n:
        pred_tables = list(pred_tables) + [None] * (n - len(pred_tables))
    tp_all = np_all = ng_all = 0
    per = []
    for k in range(n):
        gt = norm_cells(gt_tables[k])
        pr = norm_cells(pred_tables[k])
        if not gt:
            continue
        has_text = any(c["text"] for c in gt)
        sb = skip_blank and has_text
        m = match_cells(gt, pr, iou_thr, match)
        tp, npred, ngt = adjacency_counts(gt, pr, m, sb)
        tp_all += tp
        np_all += npred
        ng_all += ngt
        p = tp / npred if npred else 0.0
        r = tp / ngt if ngt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        ghr = gt_tables[k].get("header_rows") if isinstance(gt_tables[k], dict) else None
        phr = pred_tables[k].get("header_rows") if isinstance(pred_tables[k], dict) else None
        teds = teds_struct(gt, pr, ghr, phr)
        Rg = max(c["r"] + c["rs"] for c in gt)
        Cg = max(c["c"] + c["cs"] for c in gt)
        Rp = max((c["r"] + c["rs"] for c in pr), default=0)
        Cp = max((c["c"] + c["cs"] for c in pr), default=0)
        exact = sorted((c["r"], c["c"], c["rs"], c["cs"]) for c in gt) == sorted((c["r"], c["c"], c["rs"], c["cs"]) for c in pr)
        per.append({"index": k, "adj_p": round(p, 4), "adj_r": round(r, 4), "adj_f1": round(f1, 4), "teds_struct": round(teds, 4),
                    "rows_gt": Rg, "rows_pred": Rp, "cols_gt": Cg, "cols_pred": Cp, "exact": exact, "cells_gt": len(gt), "cells_pred": len(pr), "matched": len(m)})
    P = tp_all / np_all if np_all else 0.0
    Rr = tp_all / ng_all if ng_all else 0.0
    F = 2 * P * Rr / (P + Rr) if P + Rr else 0.0
    nt = len(per)
    summary = {"tables": nt, "adjacency": {"precision": round(P, 4), "recall": round(Rr, 4), "f1": round(F, 4), "tp": tp_all, "pred": np_all, "gt": ng_all,
                                           "macro_f1": round(sum(t["adj_f1"] for t in per) / nt, 4) if nt else 0.0},
               "teds_struct": round(sum(t["teds_struct"] for t in per) / nt, 4) if nt else 0.0,
               "exact_match": round(sum(t["exact"] for t in per) / nt, 4) if nt else 0.0,
               "rows_acc": round(sum(t["rows_gt"] == t["rows_pred"] for t in per) / nt, 4) if nt else 0.0,
               "cols_acc": round(sum(t["cols_gt"] == t["cols_pred"] for t in per) / nt, 4) if nt else 0.0}
    return summary, per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--match", choices=["iou", "text"], default="iou")
    ap.add_argument("--blank", choices=["skip", "keep"], default="skip")
    ap.add_argument("--per-table", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    gt = load_tables(a.gt)
    pred = load_tables(a.pred)
    summary, per = evaluate(gt, pred, a.iou, a.match, a.blank == "skip")
    out = {"summary": summary, "per_table": per if a.per_table else None}
    print(json.dumps(summary, indent=1))
    if a.per_table:
        for t in per:
            print(json.dumps(t))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
