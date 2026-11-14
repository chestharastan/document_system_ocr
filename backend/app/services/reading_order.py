"""Column-aware reading order (recursive XY-cut, from det/web/server.py) and line-row grouping."""

from __future__ import annotations


def _find_cut(items, axis, min_gap, key):
    """First gap >= min_gap in the projection of the boxes on ``axis`` (1 = y, horizontal cut; 0 = x, vertical cut)
    -> (first, rest) split at that gap, or None."""
    iv = sorted((key(i)[axis], key(i)[axis + 2]) for i in items)
    end, cut = iv[0][1], None
    for a, b in iv[1:]:
        if a - end >= min_gap:
            cut = (end + a) / 2
            break
        end = max(end, b)
    if cut is None:
        return None
    first = [i for i in items if (key(i)[axis] + key(i)[axis + 2]) / 2 < cut]
    rest = [i for i in items if (key(i)[axis] + key(i)[axis + 2]) / 2 >= cut]
    return (first, rest) if first and rest else None


def _span(items, axis, key):
    return max(key(i)[axis + 2] for i in items) - min(key(i)[axis] for i in items)


def xy_cut(items: list, page_w: float, page_h: float, key=lambda it: it["box"], depth: int = 0) -> list:
    """items with key(item) = [x0, y0, x1, y1] -> same items in reading order.

    Recursive XY-cut. Horizontal cuts (stacked blocks) normally come first, but when the current set also has a
    clear column gutter that runs its full height (a vertical cut exists over the whole set and both columns span
    most of its height) the vertical cut wins - otherwise paragraph breaks that happen to line up across two columns
    would interleave them (L1, R1, L2, R2 instead of L1, L2, R1, R2)."""
    if len(items) <= 1 or depth > 12:
        return sorted(items, key=lambda i: (key(i)[1], key(i)[0]))
    h_cut = _find_cut(items, 1, 0.008 * page_h, key)
    v_cut = _find_cut(items, 0, 0.02 * page_w, key)
    if v_cut is not None and h_cut is not None:
        height = max(1e-6, _span(items, 1, key))
        if min(_span(v_cut[0], 1, key), _span(v_cut[1], 1, key)) < 0.6 * height:
            v_cut = None  # the "columns" are short side-by-side pieces: keep the horizontal cut
    cut = v_cut or h_cut
    if cut is not None:
        first, rest = cut
        return xy_cut(first, page_w, page_h, key, depth + 1) + xy_cut(rest, page_w, page_h, key, depth + 1)
    # no clean cut (overlapping boxes): top-to-bottom, left-to-right
    return sorted(items, key=lambda i: (key(i)[1], key(i)[0]))


def sort_lines(lines: list, key=lambda ln: ln["bbox"]) -> list:
    """Lines of one block -> rows top-to-bottom, each row left-to-right (a row = vertical centres within half a line)."""
    if not lines:
        return []
    items = sorted(lines, key=lambda ln: (key(ln)[1] + key(ln)[3]) / 2)
    rows = [[items[0]]]
    for ln in items[1:]:
        b = key(ln)
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        row = rows[-1]
        rcy = sum((key(r)[1] + key(r)[3]) / 2 for r in row) / len(row)
        rh = sorted(key(r)[3] - key(r)[1] for r in row)[len(row) // 2]
        if abs(cy - rcy) < 0.5 * min(rh, h):
            row.append(ln)
        else:
            rows.append([ln])
    return [ln for row in rows for ln in sorted(row, key=lambda r: key(r)[0])]
