"""Page canvas: draws shaped text lines and assets and records exact labels.

Every printed line is rendered into its own coverage mask, its ink bbox is measured from that mask, so the quad is
tight to the visible ink. Layout objects keep a POINT SET (line-quad corners, ink hull, rectangle corners), not just
a box, so that after a rotation/perspective warp the box is recomputed from the warped ink points and stays tight.
"""

from __future__ import annotations

import math
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..taxonomy import CLS
from .res import font, normalize_khmer

KHMER = re.compile(r"[ក-៿᧠-᧿]")


class Page:
    def __init__(self, W: int, H: int, paper=(255, 255, 255)):
        self.W, self.H = W, H
        self.img = Image.new("RGB", (W, H), paper)
        self.d = ImageDraw.Draw(self.img)
        self.lines: list[list[float]] = []
        self.line_texts: list[str] = []
        self.line_fonts: list[str] = []
        self.hw_lines: list[list[float]] = []
        self.hw_texts: list[str] = []
        self.formula_lines: list[list[float]] = []
        self.objs: list[tuple[int, np.ndarray]] = []
        self.meta: dict = {}
        # structure labels, one entry per printed line (aligned with self.lines): paragraph id, role, list level /
        # marker style, whether the line IS the marker, first line of its paragraph, first-line indent (px)
        self.struct: list[dict] = []
        self._para = 0
        self.cur: dict | None = None

    # ------------------------------------------------------------------ structure
    def begin(self, role: str, level: int = 0, item=None, indent: float = 0.0):
        """Start a paragraph (list item, heading, footer line ...): every printed line until end() belongs to it."""
        self._para += 1
        self.cur = {"para": self._para, "role": role, "level": int(level), "item": item, "indent": float(indent), "n": 0}
        return self.cur

    def end(self):
        self.cur = None

    def _rec(self, q, text, path, marker=False):
        """Record one printed line + its structure entry."""
        self.lines.append(q)
        self.line_texts.append(text)
        self.line_fonts.append(path.rsplit("/", 1)[-1])
        cur = self.cur
        if cur is None:  # a template drew text directly: its own one-line paragraph, role resolved from the layout box later
            self._para += 1
            cur = {"para": self._para, "role": "other", "level": 0, "item": None, "indent": 0.0, "n": 0}
        first = cur["n"] == 0 and not marker
        if not marker:
            cur["n"] += 1
        self.struct.append({"para": cur["para"], "role": cur["role"], "level": cur["level"], "item": cur["item"],
                            "marker": bool(marker), "first": first, "indent": cur["indent"] if first else 0.0})

    # ------------------------------------------------------------------ labels
    def add(self, cls: str, pts, min_size: float = 3):
        """pts: box (x0,y0,x1,y1) | list of quads/boxes | Nx2 array."""
        if pts is None:
            return None
        P = pts.reshape(-1, 2).astype(np.float32) if isinstance(pts, np.ndarray) else to_pts(pts)
        if P is None or not len(P):
            return None
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        if x1 - x0 < min_size or y1 - y0 < min_size:
            return None
        self.objs.append((CLS[cls], P))
        return (float(x0), float(y0), float(x1), float(y1))

    # ------------------------------------------------------------------ text
    def text(self, x, y, text, path, size, fill=(0, 0, 0), align="l", angle=0.0, kind="print", squeeze=1.0,
             stroke=0, bound=None, marker=False):
        """Draw one line; (x, y) = left/centre/right of the pen at the TOP of the line box.

        Returns the 4-point quad [x1,y1,...,x4,y4] of the ink, or None if nothing was drawn (empty, or the line
        would leave the page / the given bound (x0,y0,x1,y1)).
        kind: "print" -> lines/line_texts, "hw" -> hw_lines, "formula" -> formula_lines, None -> not recorded.
        """
        if not text or not path or not text.strip():
            return None
        if KHMER.search(text):
            text = normalize_khmer(text)
        size = max(6, int(size))
        f = font(path, size)
        wl = f.getlength(text)
        pad = int(size * 0.9) + 2 + stroke
        mw, mh = int(wl) + 2 * pad, int(size * 2.0) + 2 * pad
        m = Image.new("L", (mw, mh), 0)
        ImageDraw.Draw(m).text((pad, pad), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        a = np.asarray(m)
        if squeeze != 1.0:
            a = cv2.resize(a, (max(4, int(mw * squeeze)), mh), interpolation=cv2.INTER_AREA)
            pad_x = pad * squeeze
            wl = wl * squeeze
        else:
            pad_x = pad
        ys, xs = np.nonzero(a > 8)
        if not len(xs):
            return None
        if align == "c":
            x = x - wl / 2
        elif align == "r":
            x = x - wl
        ox, oy = int(round(x - pad_x)), int(round(y - pad))
        if angle:
            h0, w0 = a.shape
            M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), angle, 1.0)
            c, s = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h0 * s + w0 * c) + 2, int(h0 * c + w0 * s) + 2
            M[0, 2] += nw / 2 - w0 / 2
            M[1, 2] += nh / 2 - h0 / 2
            a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR)
            ox -= (nw - w0) // 2
            oy -= (nh - h0) // 2
            ox_f, oy_f = ox, oy
            # rotate the tight unrotated ink rectangle -> quad
            bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            cr = np.float32([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])
            cr = cr @ M[:, :2].T + M[:, 2]
            quad = (cr + [ox_f, oy_f]).reshape(-1)
        else:
            bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            quad = np.float32([bx0, by0, bx1, by0, bx1, by1, bx0, by1]) + np.float32([ox, oy] * 4)
        qx, qy = quad[0::2], quad[1::2]
        B = bound or (0, 0, self.W, self.H)
        if qx.min() < B[0] - 2 or qy.min() < B[1] - 2 or qx.max() > B[2] + 2 or qy.max() > B[3] + 2 \
                or qx.min() < 0 or qy.min() < 0 or qx.max() > self.W or qy.max() > self.H:
            return None
        self.img.paste(fill, (ox, oy), Image.fromarray(a))
        q = [round(float(v), 1) for v in quad]
        if kind == "print":
            self._rec(q, text, path, marker=marker)
        elif kind == "hw":
            self.hw_lines.append(q)
            self.hw_texts.append(text)
        elif kind == "formula":
            self.formula_lines.append(q)
        return q

    # ------------------------------------------------------------------ images
    def ink(self, rgba: Image.Image, xy, cls=None, alpha_thr=40):
        """Multiply-blend an ink layer (signature/stamp/handwriting) - text below shows through like real ink."""
        x, y = int(xy[0]), int(xy[1])
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.W, x + rgba.width), min(self.H, y + rgba.height)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        sub = np.asarray(rgba.crop((x0 - x, y0 - y, x1 - x, y1 - y)), np.float32)
        reg = np.asarray(self.img.crop((x0, y0, x1, y1)), np.float32)
        a = sub[..., 3:] / 255.0
        out = reg * (1 - a * (1 - sub[..., :3] / 255.0))
        self.img.paste(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), (x0, y0))
        if cls:
            m = sub[..., 3] > alpha_thr
            # safety net: never label ink that is not visibly there
            if not m.any() or np.abs(out - reg).max(2)[m].mean() < 12:
                return None
            return self.add(cls, hull_pts(m, x0, y0))
        return (x0, y0, x1, y1)

    def paste(self, im: Image.Image, xy, cls=None):
        x, y = int(xy[0]), int(xy[1])
        if x < 0 or y < 0 or x + im.width > self.W or y + im.height > self.H:
            return None
        if im.mode == "RGBA":
            self.img.paste(im.convert("RGB"), (x, y), im)
            if cls:
                return self.add(cls, hull_pts(np.asarray(im)[..., 3] > 40, x, y))
            return (x, y, x + im.width, y + im.height)
        self.img.paste(im.convert("RGB"), (x, y))
        if cls:
            return self.add(cls, (x, y, x + im.width, y + im.height))
        return (x, y, x + im.width, y + im.height)


def hull_pts(mask: np.ndarray, ox, oy):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    pts = np.stack([xs, ys], 1).astype(np.int32)
    h = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    # convex hull of pixel centres -> extend to pixel edges
    cx, cy = h[:, 0].mean(), h[:, 1].mean()
    h[:, 0] += np.where(h[:, 0] >= cx, 1.0, 0.0)
    h[:, 1] += np.where(h[:, 1] >= cy, 1.0, 0.0)
    return h + np.float32([ox, oy])


def to_pts(items) -> np.ndarray | None:
    """box / quad / list of boxes+quads -> Nx2 points."""
    if items is None:
        return None
    if len(items) and np.isscalar(items[0]):
        items = [items]
    out = []
    for it in items:
        if it is None:
            continue
        it = list(it)
        if len(it) == 4:
            x0, y0, x1, y1 = it
            out += [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        else:
            out += list(zip(it[0::2], it[1::2]))
    return np.float32(out) if out else None


def bbox(items):
    P = to_pts([i for i in items if i is not None]) if items else None
    if P is None or not len(P):
        return None
    return (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))


def rect_quad(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def wrap(text: str, f, max_w: float) -> list[str]:
    from khmerocr.normalize import split_clusters
    lines, cur = [], ""
    for t in text.split(" "):
        if not t:
            continue
        cand = f"{cur} {t}" if cur else t
        if f.getlength(cand) <= max_w:
            cur = cand
            continue
        if cur and KHMER.search(t) and max_w - f.getlength(cur + " ") > 0.25 * max_w:
            # Khmer is set without word spaces: fill the line, break at a cluster boundary
            acc = ""
            for c in split_clusters(t):
                if f.getlength(f"{cur} {acc}{c}") > max_w:
                    break
                acc += c
            if acc and len(acc) < len(t):
                cur = f"{cur} {acc}"
                t = t[len(acc):]
        if cur:
            lines.append(cur)
        cur = ""
        while f.getlength(t) > max_w:
            acc = ""
            for c in (split_clusters(t) if KHMER.search(t) else list(t)):
                if f.getlength(acc + c) > max_w:
                    break
                acc += c
            if not acc:
                break
            lines.append(acc)
            t = t[len(acc):]
        cur = t
    if cur:
        lines.append(cur)
    return lines


def rot_pts(P: np.ndarray, angle_deg: float, cx: float, cy: float) -> np.ndarray:
    th = math.radians(-angle_deg)
    c, s = math.cos(th), math.sin(th)
    d = P - [cx, cy]
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1) + [cx, cy]
