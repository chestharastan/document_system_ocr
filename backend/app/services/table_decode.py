# Vendored copy of training/structure/table_decode.py (the backend never imports training/). Keep both in sync.
"""Decode the structure model's v3 table heads into cell grids (numpy only, shared by training eval and the backend).

Inputs per page: ``intab`` [N] probability that a text line sits inside a table, ``pair`` [N,N,3] probabilities that
two lines share a row / a column / a cell, ``boxes`` [N,4] page pixels. Output: tables, each with its lines, R x C
and logical cells {r, c, rs, cs, lines} (docs/TABLES_CONTRACT.md §2.2 without text / boxes, which the caller adds).

Steps: lines -> cells (connected components of the same-cell relation), cells -> tables (components of row | column
links), rows and columns by greedy assignment in y / x order: a cell joins an existing row when it is linked to most of
that row's members; a cell linked to two or more rows is a spanning cell (rs = the rows it covers) and does not vote.
"""

from __future__ import annotations

import numpy as np


def _components(adj: np.ndarray) -> list[list[int]]:
    n = adj.shape[0]
    seen = np.zeros(n, bool)
    out = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            i = stack.pop(); comp.append(i)
            for j in np.where(adj[i] & ~seen)[0]:
                seen[j] = True; stack.append(int(j))
        out.append(sorted(comp))
    return out


def _assign_bands(cells: list[list[int]], link: np.ndarray, centres: np.ndarray, min_share: float = 0.5):
    """Greedy row (or column) assignment. ``link`` [n_cells, n_cells] bool, ``centres`` per cell (y for rows, x for
    columns). -> (index per cell, span per cell, number of bands)."""
    order = sorted(range(len(cells)), key=lambda c: centres[c])
    bands: list[list[int]] = []          # member cells per band (spanning cells excluded)
    idx = np.zeros(len(cells), int); span = np.ones(len(cells), int)

    def share(c, mem):
        mem = [m for m in mem if m != c]
        return (sum(link[c, m] for m in mem) / len(mem)) if mem else -1.0

    for c in order:  # pass 1: greedy bands in centre order
        hits = [b for b, mem in enumerate(bands) if share(c, mem) >= min_share]
        if not hits:
            bands.append([c]); idx[c] = len(bands) - 1
        elif len(hits) == 1:
            bands[hits[0]].append(c); idx[c] = hits[0]
        else:
            idx[c] = min(hits); span[c] = max(hits) - min(hits) + 1
    # pass 2: a cell linked to two or more OTHER bands is a spanning cell (it may have founded a band of its own
    # in pass 1 because it came first in centre order); it leaves its band and covers the hit bands
    spanning = {}
    for c in range(len(cells)):
        hits = [b for b, mem in enumerate(bands) if share(c, mem) >= min_share]
        if len(hits) >= 2:
            spanning[c] = hits
    for c, hits in spanning.items():
        for mem in bands:
            if c in mem:
                mem.remove(c)
    keep = [b for b, mem in enumerate(bands) if mem]
    renum = {b: k for k, b in enumerate(keep)}
    bands = [bands[b] for b in keep]
    for b, mem in enumerate(bands):
        for c in mem:
            idx[c] = b; span[c] = 1
    for c, hits in spanning.items():
        hs = sorted(renum[h] for h in hits if h in renum)
        if hs:
            idx[c] = hs[0]; span[c] = hs[-1] - hs[0] + 1
        else:
            bands.append([c]); idx[c] = len(bands) - 1; span[c] = 1
    # bands were created in centre order, so their indices are sorted top-to-bottom / left-to-right
    return idx, span, len(bands)


def decode_tables(intab: np.ndarray, pair: np.ndarray, boxes: np.ndarray, thr: float = 0.5, min_lines: int = 2) -> list[dict]:
    n = len(intab)
    lines = np.where(intab > thr)[0]
    if len(lines) < min_lines:
        return []
    P = pair[np.ix_(lines, lines)]
    same_cell = (P[..., 2] > thr) & ~np.eye(len(lines), dtype=bool)
    cells = [[int(lines[i]) for i in comp] for comp in _components(same_cell)]
    nc = len(cells)
    cell_of = {ln: ci for ci, cl in enumerate(cells) for ln in cl}
    # cell-level link strengths = mean pair probability between the member lines
    def cell_link(k):
        M = np.zeros((nc, nc))
        for a in range(nc):
            for b in range(a + 1, nc):
                ia = [int(np.where(lines == l)[0][0]) for l in cells[a]]; ib = [int(np.where(lines == l)[0][0]) for l in cells[b]]
                M[a, b] = M[b, a] = float(P[np.ix_(ia, ib)][..., k].mean())
        return M > thr
    row_l, col_l = cell_link(0), cell_link(1)
    cb = np.array([[boxes[cl, 0].min(), boxes[cl, 1].min(), boxes[cl, 2].max(), boxes[cl, 3].max()] for cl in cells], np.float32)
    cy, cx = (cb[:, 1] + cb[:, 3]) / 2, (cb[:, 0] + cb[:, 2]) / 2
    tables = []
    for comp in _components(row_l | col_l):
        if sum(len(cells[c]) for c in comp) < min_lines:
            continue
        sub = np.array(comp)
        r_idx, r_span, R = _assign_bands([cells[c] for c in comp], row_l[np.ix_(sub, sub)], cy[sub])
        c_idx, c_span, C = _assign_bands([cells[c] for c in comp], col_l[np.ix_(sub, sub)], cx[sub])
        out_cells, taken = [], set()
        for k, c in enumerate(comp):
            key = (int(r_idx[k]), int(c_idx[k]))
            if key in taken:  # two cells landed on one grid position: keep both lines in the first one
                for oc in out_cells:
                    if (oc["r"], oc["c"]) == key:
                        oc["lines"] = sorted(oc["lines"] + cells[c]); break
                continue
            taken.add(key)
            out_cells.append({"r": key[0], "c": key[1], "rs": int(r_span[k]), "cs": int(c_span[k]), "lines": cells[c]})
        out_cells.sort(key=lambda d: (d["r"], d["c"]))
        tables.append({"rows": R, "cols": C, "cells": out_cells, "lines": sorted(l for c in comp for l in cells[c]),
                       "box": [float(cb[sub, 0].min()), float(cb[sub, 1].min()), float(cb[sub, 2].max()), float(cb[sub, 3].max())]})
    return tables


def fill_empty(table: dict) -> dict:
    """Add the grid positions no cell covers as empty cells (text-less), so exports get a full R x C grid."""
    covered = set()
    for c in table["cells"]:
        for r in range(c["r"], c["r"] + c["rs"]):
            for k in range(c["c"], c["c"] + c["cs"]):
                covered.add((r, k))
    for r in range(table["rows"]):
        for k in range(table["cols"]):
            if (r, k) not in covered:
                table["cells"].append({"r": r, "c": k, "rs": 1, "cs": 1, "lines": []})
    table["cells"].sort(key=lambda d: (d["r"], d["c"]))
    return table
