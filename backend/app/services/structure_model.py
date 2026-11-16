"""Apply the structure model's per-line predictions on top of the rule-based structure.

The rules (doc_structure.structure_blocks, page_structure.relabel_by_position) run first and fill every field. The
model then decides what it is best at - block roles (text -> title / section_header / list / header / footer /
page_number) and the reading order of lines and blocks - and adds paragraph starts the rules missed when the
geometry supports them. Indent levels stay with the rules' tab stops (consistent across the page)."""

from __future__ import annotations

from app.services.reading_order import sort_lines

ROLE_TO_CLS = {"title": "title", "section_header": "section_header", "list_item": "list", "header": "header", "footer": "footer",
               "page_number": "page_number", "caption": "caption", "footnote": "footnote", "text": "text"}
ROLE_MIN_P = 0.85       # a block takes the model's role when the width-weighted majority of its lines agree this strongly
NO_OVERRIDE = {"table", "form_field", "handwriting"}
HEADING = {"section_header", "title"}
SPLIT_MIN_P = 0.95      # a heading line inside a body block (or body lines inside a heading block) leaves it at this confidence


def split_by_role(blocks: list, preds: dict, text_classes: set) -> int:
    """The layout model's boxes sometimes glue a section heading to the paragraph above or below it; the block then
    renders in the paragraph's typeface and size. Where the structure model is sure (>= SPLIT_MIN_P) that some lines
    of a block are headings and others are not, the block is cut into runs of equal kind (in the model's reading
    order); each new piece is appended with the heading's class and gets its paragraphs from the parent."""
    n = 0
    for b in list(blocks):
        if b["cls"] not in text_classes or b["cls"] in NO_OVERRIDE or len(b["lines"]) < 2:
            continue
        pl = [(ln, preds.get(id(ln))) for ln in b["lines"]]
        if any(p is None for _, p in pl):
            continue
        pl.sort(key=lambda x: x[1]["order"])
        kinds = []
        for ln, p in pl:
            if p["role_p"] < SPLIT_MIN_P:
                kinds.append(None)
            else:
                kinds.append("heading" if p["role"] in HEADING else "body")
        if "heading" not in kinds or "body" not in kinds:
            continue
        # runs of equal kind; unsure lines join the run they sit in
        runs: list[list] = []
        for (ln, p), k in zip(pl, kinds):
            if runs and (k is None or runs[-1][0] in (None, k)):
                runs[-1][1].append((ln, p))
                if runs[-1][0] is None:
                    runs[-1][0] = k
            else:
                runs.append([k, [(ln, p)]])
        if len(runs) < 2:
            continue
        parent_paras = b.get("paras", [])
        pieces = []
        for k, items in runs:
            lines = [ln for ln, _ in items]
            ids = {id(ln) for ln in lines}
            xs0, ys0, xs1, ys1 = zip(*(ln["bbox"] for ln in lines))
            role = max(items, key=lambda it: it[1]["role_p"])[1]["role"] if k == "heading" else None
            cls = ROLE_TO_CLS.get(role, "section_header") if k == "heading" else ("text" if b["cls"] in HEADING else b["cls"])
            paras = [{**par, "lines": [ln for ln in par["lines"] if id(ln) in ids]} for par in parent_paras]
            paras = [par for par in paras if par["lines"]] or [{"lines": lines, "bullet": lines[0].get("bullet"), "anchor": lines[0].get("text_x0", lines[0]["bbox"][0]), "indent_level": lines[0].get("indent_level", 0)}]
            pieces.append({"cls": cls, "lines": lines, "paras": paras, "box": [float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1))]})
        first, rest = pieces[0], pieces[1:]
        b["relabel"] = b.get("relabel") or b["cls"]
        b["cls"], b["lines"], b["paras"], b["box"] = first["cls"], first["lines"], first["paras"], first["box"]
        b["split_role"] = True
        for pc in rest:
            blocks.append({**{k: v for k, v in b.items() if k not in ("font", "font_px")}, **pc, "source": "split_role", "split_role": True})
        n += len(rest)
    return n


def _keeps_rows(lines: list) -> bool:
    """True when ``lines`` (in some order) never go back up to an earlier row: rows = vertical centres within half a
    line height of each other, as in reading_order.sort_lines."""
    rows: list[list[float]] = []  # [centre, height] per row, in geometric order
    row_of = {}
    for ln in sorted(lines, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2):
        cy, h = (ln["bbox"][1] + ln["bbox"][3]) / 2, ln["bbox"][3] - ln["bbox"][1]
        if rows and abs(cy - rows[-1][0]) < 0.5 * min(rows[-1][1], h):
            row_of[id(ln)] = len(rows) - 1
        else:
            rows.append([cy, h]); row_of[id(ln)] = len(rows) - 1
    seq = [row_of[id(ln)] for ln in lines]
    return all(a <= b for a, b in zip(seq, seq[1:]))


def apply_structure(blocks: list, preds: dict, text_classes: set) -> dict:
    """preds: id(line) -> prediction dict (see StructureOnnx.predict). Mutates blocks; returns counters."""
    stats = {"roles_changed": 0, "levels_changed": 0, "paras_changed": 0, "blocks_reordered": 0}
    stats["blocks_split"] = split_by_role(blocks, preds, text_classes)
    for b in blocks:
        if b["cls"] not in text_classes or b["cls"] in NO_OVERRIDE or not b["lines"]:
            continue
        pl = [(ln, preds.get(id(ln))) for ln in b["lines"]]
        if any(p is None for _, p in pl):
            continue
        # order of lines inside the block: the model's reading order, unless it jumps over a row - a block's lines
        # are a vertical stack, so the model may only reorder lines that sit side by side on the same row (footer
        # lines next to the page rule confused it: top line read last)
        by_model = sorted(pl, key=lambda x: x[1]["order"])
        pl = by_model if _keeps_rows([ln for ln, _ in by_model]) else [(ln, preds[id(ln)]) for ln in sort_lines([ln for ln, _ in pl])]
        if pl is not by_model:
            stats["line_order_kept"] = stats.get("line_order_kept", 0) + 1
        b["lines"] = [ln for ln, _ in pl]
        b["order_key"] = min(p["order"] for _, p in pl)
        # levels: the rules' tab stops are consistent within a page - keep them; the model only lifts a bulleted line
        # the rules left at the margin
        for ln, p in pl:
            if ln.get("bullet") and ln.get("indent_level", 0) == 0 and p["level"] > 0:
                ln["indent_level"], ln["tab"] = p["level"], True; stats["levels_changed"] += 1
        # paragraph starts: the rules' starts, plus confident model starts that the geometry supports (the line above
        # ended early, or this line starts at a different x) - the model alone over-splits long paragraphs a little
        rule_starts = {id(par["lines"][0]) for par in b.get("paras", []) if par.get("lines")}
        right = max(ln["bbox"][2] for ln, _ in pl)
        paras, cur = [], None
        for i, (ln, p) in enumerate(pl):
            start = i == 0 or ln.get("bullet") or id(ln) in rule_starts
            if not start and p["first_p"] >= 0.9:
                prev = pl[i - 1][0]
                h = max(1.0, ln["bbox"][3] - ln["bbox"][1])
                start = prev["bbox"][2] < right - 2.0 * h or abs(ln["bbox"][0] - prev["bbox"][0]) > 0.5 * h
            if start:
                cur = {"lines": [ln], "bullet": ln.get("bullet"), "anchor": ln.get("text_x0", ln["bbox"][0]), "indent_level": ln.get("indent_level", 0)}
                paras.append(cur)
            else:
                cur["lines"].append(ln)
        if [len(p["lines"]) for p in paras] != [len(p["lines"]) for p in b.get("paras", [])]:
            stats["paras_changed"] += 1
        b["paras"] = paras
        # role: width-weighted vote of confident lines
        votes: dict[str, float] = {}
        tot = 0.0
        for ln, p in pl:
            w = max(1.0, ln["bbox"][2] - ln["bbox"][0])
            tot += w
            if p["role_p"] >= 0.5:
                votes[p["role"]] = votes.get(p["role"], 0.0) + w * p["role_p"]
        if votes:
            role, score = max(votes.items(), key=lambda kv: kv[1])
            cls = ROLE_TO_CLS.get(role)
            if cls and score / tot >= ROLE_MIN_P and cls != b["cls"] and b["cls"] in ("text", "list", "title", "section_header", "header", "footer", "page_number", "caption"):
                b["relabel"] = b.get("relabel") or b["cls"]
                b["cls"] = cls; stats["roles_changed"] += 1
    # block order: by the model's first-line order when every text block got one
    keyed = [b for b in blocks if "order_key" in b]
    if keyed:
        stats["blocks_reordered"] = len(keyed)
    return stats


def order_blocks(blocks: list, fallback) -> list:
    """Blocks in reading order: model order for the blocks that have it, the geometric fallback (xy_cut) decides where
    the others (figures, stamps ...) go: each is placed before the first text block that lies below it."""
    if not any("order_key" in b for b in blocks):
        return fallback(blocks)
    text = sorted([b for b in blocks if "order_key" in b], key=lambda b: b["order_key"])
    others = [b for b in blocks if "order_key" not in b]
    out: list = []
    for b in text:
        for o in sorted(others, key=lambda o: o["box"][1]):
            # above the block, or beside it on its left (an emblem left of the company name: the letterhead row)
            if (o["box"][1] + o["box"][3] <= b["box"][1] + b["box"][3] or _left_beside(o["box"], b["box"])) and o not in out:
                out.append(o)
        out.append(b)
    out += [o for o in sorted(others, key=lambda o: o["box"][1]) if o not in out]
    return _repair_order(out)


def _left_beside(o, b) -> bool:
    """box o ([x0, y0, x1, y1]) sits left of box b and starts within the upper half of its own height of b's top."""
    ox0, oy0, ox1, oy1 = o
    bx0, by0, bx1, by1 = b
    return ox1 <= bx0 + 0.02 * max(1.0, bx1 - bx0) and oy0 < by1 and oy1 > by0 and oy0 <= by0 + 0.5 * max(1.0, oy1 - oy0)


def _repair_order(order: list) -> list:
    """A block may not come before a block that lies wholly above it in the same column (horizontal overlap): the
    structure model sometimes reads the form fields at the bottom of a page before the closing paragraph above them.
    Stable topological sort: every block waits for the blocks above it in its column, ties keep the given order."""
    def must_follow(a, b) -> bool:  # a lies below b in b's column
        ax0, ay0, ax1, ay1 = a["box"]
        bx0, by0, bx1, by1 = b["box"]
        hov = min(ax1, bx1) - max(ax0, bx0)
        return hov > 0.1 * min(ax1 - ax0, bx1 - bx0) and ay0 >= by1 - 0.1 * min(ay1 - ay0, by1 - by0)

    n = len(order)
    preds = [{j for j in range(n) if j != i and must_follow(order[i], order[j])} for i in range(n)]
    done: set = set()
    out = []
    while len(out) < n:
        ready = [i for i in range(n) if i not in done and preds[i] <= done]
        i = ready[0] if ready else min(i for i in range(n) if i not in done)  # a cycle (overlapping boxes): given order
        done.add(i)
        out.append(order[i])
    return out


# ------------------------------------------------------------------------------------------------ regroup (v2)
REGROUP_SKIP = {"table", "form_field", "handwriting"}
FIRST_P = 0.5           # the model's paragraph start probability that opens a new paragraph
ROLE_SWITCH_P = 0.7     # both lines this sure of different roles -> paragraph boundary


def _vov(a, b) -> float:
    return min(a[3], b[3]) - max(a[1], b[1])


def _same_row(a, b) -> bool:
    return _vov(a, b) > 0.5 * min(a[3] - a[1], b[3] - b[1])


def tangled_blocks(blocks: list, min_cover: float = 0.25) -> list:
    """Text blocks whose layout boxes overlap another one's (>= min_cover of the smaller box, or a line of one lies in
    the other's box): the layout model cut those lines wrongly (a list box that swallowed the next clause, a row box
    inside a paragraph box) and only they are rebuilt; well-separated blocks keep the layout model's grouping."""
    def area(b):
        x0, y0, x1, y1 = b
        return max(1.0, (x1 - x0) * (y1 - y0))

    def inter(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    def line_in(a, b):  # a line of block a has its centre inside block b's box
        bx0, by0, bx1, by1 = b["box"]
        return any(bx0 <= (l["bbox"][0] + l["bbox"][2]) / 2 <= bx1 and by0 <= (l["bbox"][1] + l["bbox"][3]) / 2 <= by1
                   for l in a["lines"])

    out = set()
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            if inter(a["box"], b["box"]) >= min_cover * min(area(a["box"]), area(b["box"])) or line_in(a, b) or line_in(b, a):
                out.add(id(a)); out.add(id(b))
    return [b for b in blocks if id(b) in out]


def _centred_next(prev: dict, ln: dict, mh: float) -> bool:
    """``ln`` is the next centred line under ``prev``: both indented from each other on both sides (not left-aligned),
    sharing their centre, without an extra gap."""
    a, b = prev["bbox"], ln["bbox"]
    return (abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) <= 1.5 * mh and abs(a[0] - b[0]) > 1.0 * mh
            and -0.3 * mh <= b[1] - a[3] <= 0.7 * mh)


def _continues(para: list, ln: dict, mh: float) -> bool:
    """``ln`` continues the paragraph ``para`` (lines in reading order) although the model saw a paragraph start: the
    paragraph's last row reaches the paragraph's right edge (a justified line; a short last line ends a paragraph),
    ``ln`` starts at the left edge of the paragraph's continuation lines, and follows without an extra gap (the
    model splits a long justified clause after a fold or a wide-gap row: user's folded contract photo)."""
    last = para[-1]["bbox"]
    row = [l["bbox"] for l in para if _same_row(l["bbox"], last)]
    right = max(l["bbox"][2] for l in para)
    body = [l["bbox"][0] for l in para if not _same_row(l["bbox"], para[0]["bbox"])] or [para[0]["bbox"][0]]
    b = ln["bbox"]
    return (max(r[2] for r in row) >= right - 1.0 * mh and b[2] <= right + 1.0 * mh
            and abs(b[0] - min(body)) <= 0.8 * mh and 0 <= b[1] - max(r[3] for r in row) + 0.3 * mh <= 1.0 * mh)


def _join_row(lines: list) -> dict:
    """Pieces of ONE printed line (the detector cut it at a wide justified gap) -> one line dict."""
    import numpy as np
    import cv2
    lines = sorted(lines, key=lambda l: l["bbox"][0])
    base = dict(lines[0])
    xs0, ys0, xs1, ys1 = zip(*(l["bbox"] for l in lines))
    base["bbox"] = [float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1))]
    pts = np.concatenate([np.asarray(l["quad"], np.float32).reshape(-1, 2) for l in lines])
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    c = box.mean(0)
    base["quad"] = box[np.argsort(np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0]))].astype(np.float32)
    base["text"] = " ".join(l.get("text", "") for l in lines if l.get("text"))
    base["confidence"] = min(l.get("confidence", 0.0) for l in lines)
    base["score"] = max(l.get("score", 0.0) for l in lines)
    spans = []
    for l in lines:
        spans += l.get("spans") or [{"text": l.get("text", ""), "bold": bool(l.get("bold")),
                                     "x0": float(l["bbox"][0]), "x1": float(l["bbox"][2])}]  # text_colors reads x0/x1
    base["spans"] = [sp for sp in spans if sp.get("text")]
    base["bold"] = all(l.get("bold") for l in lines)
    base.pop("char_pos", None)
    base["joined"] = len(lines)
    return base


def regroup(blocks: list, preds: dict, text_classes: set, block_factory, only_tangled: bool = False) -> dict:
    """Rebuild the text blocks of the page from the structure model: the layout model cuts one paragraph into
    several boxes (a clause split at a page fold, a heading glued to the clause below it, a justified line cut in
    two), and the rules never join paragraphs across boxes. Here all text lines (tables / forms / handwriting kept as
    they are) are walked in the model's reading order and cut into paragraphs where the model sees a paragraph start
    (or a bullet, a confident role change, a jump to another column / back up the page); every paragraph becomes one
    block with the role of its width-weighted line vote. Pieces of one printed line on the same row are joined.
    -> counters. Blocks without predictions are left untouched."""
    from app.services.reading_order import sort_lines  # noqa: F401  (kept for callers' symmetry)
    stats = {"regroup_in": 0, "regroup_out": 0, "rows_joined": 0}
    src = [b for b in blocks if b["cls"] in text_classes and b["cls"] not in REGROUP_SKIP and b["lines"]
           and all(id(ln) in preds for ln in b["lines"])]
    if only_tangled:
        src = tangled_blocks(src)
    if not src:
        return stats
    owner = {id(ln): b for b in src for ln in b["lines"]}
    seq = sorted((ln for b in src for ln in b["lines"]), key=lambda ln: preds[id(ln)]["order"])
    hs = sorted(max(1.0, ln["bbox"][3] - ln["bbox"][1]) for ln in seq)
    mh = hs[len(hs) // 2]
    paras: list[list] = []
    for ln in seq:
        p = preds[id(ln)]
        if not paras:
            paras.append([ln]); continue
        prev = paras[-1][-1]
        pp = preds[id(prev)]
        a, b_ = prev["bbox"], ln["bbox"]
        if _same_row(a, b_) and b_[0] >= a[2] - 0.5 * mh and b_[0] - a[2] < 4.0 * mh and not ln.get("bullet") \
                and owner[id(ln)]["cls"] == owner[id(prev)]["cls"] or (_same_row(a, b_) and b_[0] >= a[2] - 0.5 * mh and b_[0] - a[2] < 2.5 * mh and not ln.get("bullet")):
            paras[-1].append(ln); continue  # the rest of the same printed row
        start = bool(ln.get("bullet")) or p["first_p"] >= FIRST_P
        if start and not ln.get("bullet") and len(paras[-1]) >= 2 and _continues(paras[-1], ln, mh):
            start = False  # the line above ran to the paragraph's right edge and this one sits at its left edge
        if start and not ln.get("bullet") and owner[id(ln)] is owner[id(prev)] and _centred_next(prev, ln, mh):
            start = False  # the next line of a centred block (a royal title over several lines) from one layout box
        if not start and p["role"] != pp["role"] and p["role_p"] >= ROLE_SWITCH_P and pp["role_p"] >= ROLE_SWITCH_P:
            start = True
        if not start:  # geometry: going back up, a big gap, or another column
            gap = b_[1] - a[3]
            hov = min(a[2], b_[2]) - max(a[0], b_[0])
            if b_[1] < a[1] - 0.5 * mh or gap > 2.2 * mh or hov < -0.5 * mh:
                start = True
        if start:
            paras.append([ln])
        else:
            paras[-1].append(ln)
    # role per paragraph (width-weighted vote of confident lines; the first line's block class otherwise)
    new_blocks = []
    for pl in paras:
        votes: dict[str, float] = {}
        for ln in pl:
            p = preds[id(ln)]
            if p["role_p"] >= 0.4:
                votes[p["role"]] = votes.get(p["role"], 0.0) + max(1.0, ln["bbox"][2] - ln["bbox"][0]) * p["role_p"]
        b0 = owner[id(pl[0])]
        role = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
        cls = ROLE_TO_CLS.get(role, b0["cls"]) if role else b0["cls"]
        if b0["cls"] in ("header", "footer", "page_number") and cls == "text":
            cls = b0["cls"]  # the position rules' letterhead / footer stays unless the model names another role
        # join pieces of one printed row
        rows: list[list] = []
        for ln in pl:
            if rows and _same_row(rows[-1][-1]["bbox"], ln["bbox"]) and not ln.get("bullet"):
                rows[-1].append(ln)
            else:
                rows.append([ln])
        lines = [r[0] if len(r) == 1 else _join_row(r) for r in rows]
        stats["rows_joined"] += sum(len(r) - 1 for r in rows)
        for ln, r in zip(lines, rows):
            if len(r) > 1:
                preds[id(ln)] = preds[id(r[0])]
        xs0, ys0, xs1, ys1 = zip(*(ln["bbox"] for ln in lines))
        nb = block_factory(cls, [min(xs0), min(ys0), max(xs1), max(ys1)], b0.get("score", 1.0), b0.get("label", cls), source="structure")
        nb["lines"] = lines
        nb["paras"] = [{"lines": lines, "bullet": lines[0].get("bullet"), "anchor": lines[0].get("text_x0", lines[0]["bbox"][0]), "indent_level": 0}]
        nb["order_key"] = min(preds[id(ln)]["order"] for ln in pl)
        nb["split_role"] = True  # its own typeface vote
        if cls != b0["cls"]:
            nb["relabel"] = b0["cls"]
        for k in ("ink", "text_color"):
            if k in b0:
                nb[k] = b0[k]
        new_blocks.append(nb)
    ids = {id(b) for b in src}
    blocks[:] = [b for b in blocks if id(b) not in ids] + new_blocks
    stats["regroup_in"], stats["regroup_out"] = len(src), len(new_blocks)
    return stats
