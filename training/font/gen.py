"""Synthetic text-line crops for the font classifier, rendered on the fly (one sample per call, fork-safe).

A sample imitates what the pipeline hands the recogniser: a detector box around one printed line with a little
margin, taken from a scan or phone photo. Rendering: PIL + HarfBuzz (raqm) for Khmer shaping; text from the Khmer
news corpus (+ Latin tokens / English sentences); random size, colour, synthetic bold (dilation) and italic (shear),
neighbouring lines intruding at the top/bottom, underlines; degradation: resolution loss, blur, noise, JPEG, paper
tint and lighting gradient. Output: grayscale H x W uint8 (H = 32), labels (family, weight, slant)."""

from __future__ import annotations

import gzip
import io
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.font.fonts import KHMER_WITH_LATIN, LATIN_ONLY, SLANTS, WEIGHTS, families

CORPUS = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation/data/corpus")
EN_SENTENCES = Path("/home/hello/ocr/docsys/work/data/docgen_text/en_sentences.txt.gz")
H = 32
W = 256


# --------------------------------------------------------------------------------------------- text sources
class Texts:
    def __init__(self, split: str = "train", limit: int = 120_000):
        self.km: list[str] = []
        with gzip.open(CORPUS / f"{split}_news.tsv.gz", "rt", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if i >= limit:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and len(parts[1]) >= 4:
                    self.km.append(parts[1])
        self.latin = [l.split("\t")[0] for l in (CORPUS / "latin_tokens_train.tsv").read_text(encoding="utf-8").splitlines()[1:2000]]
        self.en: list[str] = []
        if EN_SENTENCES.exists():
            with gzip.open(EN_SENTENCES, "rt", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 20_000:
                        break
                    s = line.strip()
                    if 8 <= len(s) <= 120:
                        self.en.append(s)
        if not self.en:
            self.en = ["Kingdom of Cambodia Nation Religion King", "Ministry of Education Youth and Sport"]

    def khmer(self, rng: random.Random) -> str:
        s = rng.choice(self.km)
        toks = s.split(" ")
        n = rng.randint(1, min(7, len(toks)))
        i = rng.randint(0, len(toks) - n)
        t = " ".join(toks[i:i + n])
        r = rng.random()
        if r < 0.15:  # a number / English word in the Khmer line (dates, acronyms, names)
            ins = rng.choice([str(rng.randint(1, 2030)), rng.choice(self.latin), "".join(rng.choice("០១២៣៤៥៦៧៨៩") for _ in range(rng.randint(1, 4)))])
            t = f"{t} {ins}" if rng.random() < 0.5 else f"{ins} {t}"
        elif r < 0.22:
            t = f"{rng.randint(1, 20)}. {t}" if rng.random() < 0.5 else f"{rng.choice('កខគឃងចឆជ')}) {t}"
        return t[:60]

    def english(self, rng: random.Random) -> str:
        s = rng.choice(self.en)
        toks = s.split(" ")
        n = rng.randint(1, min(8, len(toks)))
        i = rng.randint(0, len(toks) - n)
        return " ".join(toks[i:i + n])[:60]


# --------------------------------------------------------------------------------------------- fonts
@lru_cache(maxsize=4096)
def _font(path: str, size: int, wght: int | None):
    f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    if wght is not None:
        try:
            f.set_variation_by_axes([wght])
        except Exception:
            pass
    return f


class Catalog:
    def __init__(self):
        self.families = families()
        self.names = [n for n, _ in self.families]
        self.index = {n: i for i, n in enumerate(self.names)}

    def pick(self, rng: random.Random):
        """-> (family index, file path, weight label, variable-font wght or None)"""
        fi = rng.randrange(len(self.families))
        name, files = self.families[fi]
        path, weight = rng.choice(files)
        wght = None
        if "[wght]" in path.name:
            wght = rng.choice([300, 400, 400, 500, 700, 900])
            weight = "bold" if wght >= 700 else "regular"
        return fi, str(path), weight, wght


# --------------------------------------------------------------------------------------------- rendering
def _render_text(text: str, font, fill, bg, pad: int) -> np.ndarray:
    """RGB uint8 image of the text with ``pad`` px around (PIL bbox, HarfBuzz shaping)."""
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        l, t, r, b = d.textbbox((0, 0), text, font=font, language="km" if any("ក" <= c <= "៿" for c in text) else "en")
    except Exception:
        l, t, r, b = d.textbbox((0, 0), text, font=font)
    w, h = max(1, r - l), max(1, b - t)
    img = Image.new("RGB", (w + 2 * pad, h + 2 * pad), bg)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=fill)
    return np.array(img)


def _paper(h: int, w: int, rng: random.Random, base) -> np.ndarray:
    """Paper: flat tint, or low-frequency lighting gradient / mottling."""
    bg = np.full((h, w, 3), base, np.float32)
    if rng.random() < 0.6:
        g = np.linspace(rng.uniform(-25, 25), rng.uniform(-25, 25), w, dtype=np.float32)[None, :, None]
        bg += g
    if rng.random() < 0.4:
        n = cv2.resize(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 14, (max(2, h // 8), max(2, w // 8))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        bg += n[:, :, None]
    return bg.clip(0, 255).astype(np.uint8)


def make_sample(texts: Texts, cat: Catalog, rng: random.Random, train: bool = True):
    """-> (gray H x W uint8, family idx, weight idx, slant idx, em ratio = font size px / crop height before resizing).
    The em ratio is what turns a line crop's height back into a CSS font size: it depends on the family (Khmer OS
    bodies sit at ~70 % of the em, Noto's at ~60 %), on the glyphs present (subscripts / superscripts) and on the
    detector's margin - so it is learned per line rather than taken as one constant."""
    fi, path, weight, wght = cat.pick(rng)
    name = cat.names[fi]
    if name in LATIN_ONLY:
        text = texts.english(rng) if rng.random() < 0.85 else str(rng.randint(1, 99999))
    elif name in KHMER_WITH_LATIN and rng.random() < 0.15:
        text = texts.english(rng)
    else:
        text = texts.khmer(rng)
    size = rng.randint(30, 80)
    font = _font(path, size, wght)
    dark = rng.random() < 0.97
    ink = tuple(int(v) for v in (rng.randint(0, 70),) * 3) if rng.random() < 0.8 else (rng.randint(0, 120), rng.randint(0, 120), rng.randint(0, 200))
    paper = rng.randint(180, 255)
    if not dark:
        ink, paper = (rng.randint(200, 255),) * 3, rng.randint(0, 70)
    pad = int(size * rng.uniform(0.25, 0.6))
    img = _render_text(text, font, ink, (paper,) * 3, pad)
    h, w = img.shape[:2]
    # synthetic bold: dilate the ink (what Word does to fonts without a bold face)
    slant = 0
    if weight == "regular" and rng.random() < 0.18:
        k = max(2, int(round(size / 22)))
        img = cv2.erode(img, np.ones((k, k), np.uint8)) if dark else cv2.dilate(img, np.ones((k, k), np.uint8))
        weight = "bold"
    # synthetic italic: shear
    if rng.random() < 0.12:
        sh = rng.uniform(0.18, 0.32)
        M = np.float32([[1, sh, -sh * h / 2], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        slant = 1
    # underline
    if rng.random() < 0.06:
        y = int(h - pad * rng.uniform(0.3, 0.8))
        cv2.line(img, (pad // 2, y), (w - pad // 2, y), ink, max(1, size // 25))
    # neighbouring lines intruding at the top / bottom (tight Khmer line spacing)
    if rng.random() < 0.35:
        other = _render_text(texts.khmer(rng) if name not in LATIN_ONLY else texts.english(rng), font, ink, (paper,) * 3, 0)
        oh, ow = other.shape[:2]
        for top in ([True, False] if rng.random() < 0.3 else [rng.random() < 0.5]):
            off = int(rng.uniform(0.55, 0.85) * oh)  # how much of the other line is pushed outside
            x = rng.randint(-ow // 2, max(0, w - ow // 2))
            y = -off if top else h - (oh - off)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + ow), min(h, y + oh)
            if x1 > x0 and y1 > y0:
                patch = other[y0 - y:y1 - y, x0 - x:x1 - x]
                roi = img[y0:y1, x0:x1]
                img[y0:y1, x0:x1] = np.minimum(roi, patch) if dark else np.maximum(roi, patch)
    # paper tint + lighting replaces the flat background
    bg = _paper(h, w, rng, paper)
    alpha = (img.astype(np.float32) - paper) if dark else (img.astype(np.float32) - paper)
    img = (bg.astype(np.float32) + alpha).clip(0, 255).astype(np.uint8)
    # small rotation
    if rng.random() < 0.5:
        ang = rng.uniform(-1.5, 1.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # crop like the pipeline: the ink box grown by a random margin (detector unclip + rec padding)
    ys, xs = np.where((gray < paper - 40) if dark else (gray > paper + 40))
    if len(ys) > 10:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        th = max(1, y1 - y0)
        my0, my1 = int(th * rng.uniform(0.02, 0.22)), int(th * rng.uniform(0.02, 0.22))
        mx = int(th * rng.uniform(0.05, 0.4))
        gray = gray[max(0, y0 - my0):min(h, y1 + my1 + 1), max(0, x0 - mx):min(w, x1 + mx + 1)]
    h, w = gray.shape[:2]
    em_ratio = size / max(1, h)
    # resolution loss (small print, phone photos): go through a smaller height first
    hh = rng.choice([H, H, H, 28, 24, 20, 16, 14])
    if rng.random() < 0.5:
        gray = cv2.GaussianBlur(gray, (0, 0), rng.uniform(0.3, 1.0) * h / 40)
    nw = max(8, int(round(w * hh / h)))
    gray = cv2.resize(gray, (nw, hh), interpolation=cv2.INTER_AREA)
    if hh != H:
        gray = cv2.resize(gray, (max(8, int(round(w * H / h))), H), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    # photometric
    g = gray.astype(np.float32)
    if rng.random() < 0.7:
        g = g * rng.uniform(0.75, 1.15) + rng.uniform(-30, 30)
    if rng.random() < 0.6:
        g += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 10), g.shape)
    gray = g.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        q = rng.randint(25, 90)
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    # width: pad with the border colour or random-crop to W
    if gray.shape[1] >= W:
        x = rng.randint(0, gray.shape[1] - W) if train else (gray.shape[1] - W) // 2
        gray = gray[:, x:x + W]
    else:
        fill = int(np.median(np.concatenate([gray[:, :2].ravel(), gray[:, -2:].ravel()])))
        pad_w = W - gray.shape[1]
        left = rng.randint(0, pad_w) if train else pad_w // 2
        gray = cv2.copyMakeBorder(gray, 0, 0, left, pad_w - left, cv2.BORDER_CONSTANT, value=fill)
    return gray, fi, WEIGHTS.index(weight), slant, em_ratio


def line_crops_to_batch(crops: list[np.ndarray]) -> tuple[np.ndarray, list[list[int]]]:
    """Inference preprocessing shared with the backend stage: gray crop -> one or more H x W windows (a long line
    gives up to 3 windows whose logits are averaged). -> (float32 [N,1,H,W], window indices per crop)."""
    xs, owners = [], []
    for i, c in enumerate(crops):
        g = c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        h, w = g.shape[:2]
        nw = max(8, int(round(w * H / max(1, h))))
        g = cv2.resize(g, (nw, H), interpolation=cv2.INTER_AREA if h > H else cv2.INTER_CUBIC)
        wins = []
        if nw <= W:
            fill = int(np.median(np.concatenate([g[:, :2].ravel(), g[:, -2:].ravel()])))
            pad_w = W - nw
            wins.append(cv2.copyMakeBorder(g, 0, 0, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=fill))
        else:
            n = min(3, int(np.ceil(nw / W)))
            for k in range(n):
                x = int(round(k * (nw - W) / max(1, n - 1))) if n > 1 else 0
                wins.append(g[:, x:x + W])
        owners.append(list(range(len(xs), len(xs) + len(wins))))
        xs += wins
    arr = (np.stack(xs).astype(np.float32) / 255.0 - 0.5) / 0.5 if xs else np.zeros((0, H, W), np.float32)
    return arr[:, None], owners


if __name__ == "__main__":  # preview sheet
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/font_preview.png")
    rng = random.Random(0)
    texts, cat = Texts(limit=5000), Catalog()
    rows = []
    for _ in range(40):
        g, fi, wi, si, r = make_sample(texts, cat, rng)
        tile = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        label = np.full((H, 260, 3), 255, np.uint8)
        cv2.putText(label, f"{cat.names[fi]} {WEIGHTS[wi][0]}{SLANTS[si][0]} em{r:.2f}", (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(np.hstack([tile, label]))
    cv2.imwrite(str(out), np.vstack(rows))
    print(out, len(cat.names), "families")
