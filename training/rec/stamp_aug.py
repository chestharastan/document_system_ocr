"""Stamp-over-text augmentation for recogniser line crops (grayscale uint8, height 48).

Official Cambodian documents carry a round ink stamp over the date / signature lines. In the pipeline such a line is
read from the page's red channel (the stamp fades to light grey) or, when the stamp was not detected, as is (the
stamp is mid grey). This overlays ring arcs and short "stamp text" strokes in light-to-mid grey on a crop so the
recogniser learns to read through them. Enabled in training.rec.train with STAMP_AUG=<probability>."""

from __future__ import annotations

import math

import cv2
import numpy as np


def overlay_stamp(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    layer = np.full_like(out, 255)
    # intensity: mostly light (after red-channel fading), sometimes mid grey (raw photo)
    v = int(rng.uniform(150, 225)) if rng.random() < 0.7 else int(rng.uniform(90, 150))
    t = max(1, int(round(h * rng.uniform(0.05, 0.12))))
    # one or two ring arcs of a big circle whose centre may lie outside the crop
    cx, cy = int(rng.uniform(-0.2 * w, 1.2 * w)), int(rng.uniform(-2.5 * h, 3.5 * h))
    r = int(rng.uniform(1.5 * h, 4.0 * h))
    for k in range(rng.integers(1, 3)):
        rr = r - k * int(h * rng.uniform(0.35, 0.6))
        if rr > 4:
            a0 = rng.uniform(0, 360)
            cv2.ellipse(layer, (cx, cy), (rr, rr), 0, a0, a0 + rng.uniform(60, 360), v, t)
    # stamp text: short wavy strokes following the ring
    for _ in range(rng.integers(3, 10)):
        ang = rng.uniform(0, 2 * math.pi)
        px, py = cx + r * math.cos(ang), cy + r * math.sin(ang)
        if not (-10 <= px <= w + 10 and -10 <= py <= h + 10):
            continue
        pts = []
        for s in range(rng.integers(2, 5)):
            pts.append([int(px + rng.uniform(-0.25 * h, 0.25 * h)), int(py + rng.uniform(-0.3 * h, 0.3 * h))])
        cv2.polylines(layer, [np.array(pts, np.int32)], False, v, max(1, t - 1))
    if rng.random() < 0.5:
        layer = cv2.GaussianBlur(layer, (0, 0), rng.uniform(0.4, 1.0))
    # ink multiplies: the stamp darkens the paper, the text under it stays dark
    return np.minimum(out, layer)


def patch_mixed_lines(prob: float) -> None:
    """Monkey-patch khmerocr.mixed_dataset so the TRAIN mix overlays a stamp on a fraction of the crops."""
    import khmerocr.mixed_dataset as md
    orig_image = md.MixedLines.image
    orig_build = md.build_mix

    def image(self, i):
        im = orig_image(self, i)
        p = getattr(self, "stamp_p", 0.0)
        if p > 0:
            rng = np.random.default_rng((i * 1_000_003 + int(getattr(self, "stamp_seed", 0))) & 0xFFFFFFFF)
            if rng.random() < p:
                im = overlay_stamp(im, rng)
        return im

    def build_mix(*a, **k):
        train, val = orig_build(*a, **k)
        train.stamp_p, train.stamp_seed = prob, np.random.randint(1 << 30)
        return train, val

    md.MixedLines.image = image
    md.build_mix = build_mix
