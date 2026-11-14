"""Handwritten school exam answer sheets (Khmer national-exam style + a smaller English school-worksheet variant).

Modelled on phone photos of handwritten Khmer exam answer sheets (style only; every name, date, subject value and
answer text here is synthetic: Khmer corpus sentences / words, random name parts, random dates).

Page kinds
  * km first page  : printed candidate form (exam session / centre / name / date of birth / signature fields with
                     dotted or solid fill-in lines and handwritten values, "អក្សរសម្គាល់" score boxes), a printed
                     2-line notice, a dashed separator, "វិញ្ញាសា៖" subject field, printed title "ចម្លើយ", then the ruled
                     answer area (thick left margin rule).
  * km continuation: a printed box crossed by an X with the one-line notice in the lower triangle, then ruled area.
  * en worksheet   : printed Name / Class / Date / Score header + title, blue rules with a red margin line.
Body: ballpoint handwriting (blue / black) sitting on the rules: Roman-numeral section markers, numbered answers,
"-" bullet items (vocabulary "word (ន)៖ definition។"), continuation lines, occasional struck-out words, then red
teacher marks (ticks, crosses, circles, underlines, margin brackets, long strokes through unused lines, margin scores,
short comments).

Handwriting renderer: the existing Khmer hand fonts (res km_hand) / Latin hand fonts (en_hand) are drawn cluster by
cluster (Khmer) or word by word (Latin) with a per-page "writer" (horizontal stretch, slant, spacing) and per-unit
jitter (size, baseline wobble, rotation), then elastic distortion, stroke-width (pressure) change and pressure-
modulated ink opacity. The quad of each line is the tight box of the final ink mask.

Labelling decisions (taxonomy v2)
  * handwriting   : one box per handwritten ANSWER BLOCK = a run of consecutive written rules, split at blank rules,
                    at every Roman-numeral section marker (the marker line is its own block) and at every new top-level
                    numbered answer. Handwritten "-" bullet items and numbered sub-items are NOT `list`: `list` is a
                    printed-text class in this taxonomy (TEXT_CLASSES) and the downstream reader must route these
                    regions to handwriting recognition, so handwritten lists are `handwriting` (they stay inside the
                    block of the question they answer). Each handwritten header value (name, date, subject, ...) is
                    also its own `handwriting` box, as in the other docgen form templates.
  * red teacher marks: ticks, crosses, circles, underlines, brackets and strokes are NOT objects (no box, no line).
                    Red marks that are legible text - margin / score-box scores ("3", "2.5", "៤", "18/20") and short red
                    comment words - ARE handwriting (box + hw line + text), because they are handwritten text that the
                    model must find.
  * struck-out student words keep their text in the line transcription (the word stays legible under 1-2 strokes).
  * printed form: field label + fill-in line (+ handwritten value) -> form_field (README rule); score box
                    ("អក្សរសម្គាល់" + its rectangle, + red score if any) -> form_field; printed 2-line notice -> text;
                    "ចម្លើយ" / worksheet title -> title; continuation-page notice line -> text.
  * the X-ed box of continuation pages is NOT labelled: `figure` means picture / chart / diagram / illustration and
                    the existing templates never label form rules, frames or ornaments as figure (ornaments are only
                    included in the header block they decorate); labelling a ruled frame as figure would teach the
                    detector to fire on form graphics. Its notice line is `text`.
  * invigilator / student signatures -> signature (ink hull).
  * rules, margin lines, dashed separators: unlabelled. Show-through of the reverse side (added in degrade2's
                    exam_phone profile) is unlabelled, like watermarks.
Every printed line is in lines/line_texts, every handwritten line (incl. red scores/comments) in hw_lines/hw_texts,
so lines_ok / objs_ok are true (the generator's default).
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import blocks as B
from .canvas import KHMER, Page, bbox
from .ctx import KH_DISTRICT, KH_PROV, Ctx
from .res import R, font, font_ok, normalize_khmer

# ----------------------------------------------------------------------------------------------- content
SUBJECTS_KM = ["គណិតវិទ្យា", "ភាសាខ្មែរ", "ប្រវត្តិវិទ្យា", "ភូមិវិទ្យា", "សីលធម៌-ពលរដ្ឋវិជ្ជា", "រូបវិទ្យា", "គីមីវិទ្យា", "ជីវវិទ្យា",
               "ផែនដីវិទ្យា", "ភាសាអង់គ្លេស", "អក្សរសាស្ត្រខ្មែរ", "សេដ្ឋកិច្ចវិទ្យា"]
SECTIONS_KM = ["ពន្យល់ពាក្យ", "សំណួរ", "ចម្លើយ", "តែងសេចក្តី", "វេយ្យាករណ៍", "អំណាន", "លំហាត់", "ប្រយោគ", "អត្ថន័យ", "សេចក្តីសន្និដ្ឋាន",
               "សំណួរទី១", "សំណួរទី២", "ការយល់ដឹង", "ធ្វើលំហាត់", "សេចក្តីផ្តើម", "តួសេចក្តី"]
POS_KM = ["ន", "កិ", "គុ", "កិ.វិ", "ស", "ន", "កិ"]
COMMENTS_KM = ["ល្អ", "ខុស", "ត្រូវ", "មិនគ្រប់", "ល្អណាស់", "ខ្វះ", "ពិនិត្យ", "មើលឡើងវិញ", "អក្សរ", "ត្រូវកែ"]
COMMENTS_EN = ["good", "Good!", "see me", "why?", "no", "spelling", "incomplete", "nice", "check", "?"]
NOTICES_KM = [
    "បេក្ខជនមិនត្រូវធ្វើសញ្ញាសម្គាល់អ្វីមួយនៅលើក្រដាសប្រឡងឡើយ។ ក្រដាសប្រឡងណាដែលមានសញ្ញាសម្គាល់នឹងត្រូវបានពិន័យ",
    "បេក្ខជនមិនត្រូវចម្លងប្រធានវិញ្ញាសាដាក់លើក្រដាសប្រឡងជាដាច់ខាត។",
    "បេក្ខជនត្រូវសរសេរចម្លើយដោយប៊ិចពណ៌ខៀវ ឬពណ៌ខ្មៅតែប៉ុណ្ណោះ។",
    "ហាមសរសេរឈ្មោះ ឬធ្វើសញ្ញាសម្គាល់ផ្សេងៗក្នុងផ្នែកចម្លើយ។",
    "សន្លឹកកិច្ចការនេះត្រូវប្រគល់ជូនគណៈកម្មការវិញ ក្រោយពេលប្រឡងចប់។",
]
NOTICE_CONT_KM = ["បេក្ខជនមិនត្រូវសរសេរលើបន្ទាត់ខ្វែងនេះទេ។", "សូមកុំសរសេរក្នុងប្រអប់នេះ។", "មិនត្រូវសរសេរលើផ្នែកនេះទេ។",
                  "ផ្នែកនេះសម្រាប់គណៈកម្មការ។"]
LEFT_LABELS_KM = ["ហត្ថលេខាអនុរក្ស", "លេខសម្គាល់អនុរក្ស", "ហត្ថលេខាគណៈកម្មការ", "លេខបន្ទប់", "លេខតុ"]
SCORE_LABELS_KM = ["អក្សរសម្គាល់", "ពិន្ទុ", "លេខសម្គាល់"]
CENTRE_KM = ["វិទ្យាល័យ", "អនុវិទ្យាល័យ", "សាលាបឋមសិក្សា", "មណ្ឌលប្រឡង"]

EN_SUBJECTS = ["Science", "English", "History", "Geography", "Mathematics", "Biology", "Reading", "Social Studies", "Civics",
               "Chemistry", "Physics", "Language Arts"]
EN_TITLES = ["Answer Sheet", "ANSWER SHEET", "Unit {n} Test", "Quiz {n}", "Worksheet {n}", "Chapter {n} Review", "Homework",
             "Midterm Exam - Answers", "Final Exam", "Reading Comprehension"]
EN_SECTIONS = ["Part A", "Part B", "Part C", "Section I", "Section II", "Vocabulary", "Short Answer", "Questions", "Essay"]

BLUE_INKS = [(25, 40, 150), (30, 45, 125), (20, 30, 110), (40, 55, 160), (30, 40, 95), (15, 30, 120), (25, 35, 105)]
BLACK_INKS = [(25, 25, 35), (40, 40, 50), (15, 15, 25)]
RED_INKS = [(205, 30, 40), (185, 20, 35), (220, 50, 50), (170, 25, 45), (210, 40, 70)]
ROMAN = ["I", "II", "III", "IV", "V", "VI"]


def _km_words(rng, n=1):
    ts = R["km"].ts
    from khmerocr.textgen import _pick
    out = []
    for _ in range(80):
        w = _pick(ts.words, rng)
        if len(w) >= 3 and all("ក" <= c <= "៿" for c in w) and "។" not in w and not any("០" <= c <= "៩" for c in w):
            out.append(w)
            if len(out) >= n:
                break
    return out or ["ការសិក្សា"]


def _km_sentence(rng, lo, hi):
    """A sentence-like chunk of corpus text ending with ។ (answers are written as sentences)."""
    t = R["km"].paragraph(rng, hi + 40)
    st = 0
    if rng.random() < 0.7:  # start after a sentence end / a space to look like a sentence start
        cands = [i + 1 for i, c in enumerate(t[: len(t) // 2]) if c in "។ "]
        if cands:
            st = rng.choice(cands)
    t = t[st:].strip()
    t = " ".join(w for w in t.replace("«", "").replace("»", "").replace('"', "").split(" ")
                 if w and not any("A" <= c <= "z" for c in w))
    n = rng.randint(lo, hi)
    if len(t) > n:
        cut = t.rfind(" ", int(n * 0.6), n)
        t = t[: cut if cut > 0 else n]
    t = t.strip(" ,។")
    from khmerocr.textgen import trunc
    t = trunc(t, hi) or "ការសិក្សា"
    return t + rng.choice(["។", "។", "។", " ។", "", "៕"])


# ----------------------------------------------------------------------------------------------- writer / renderer
class Writer:
    """Per-page handwriting persona."""

    def __init__(self, rng: random.Random, lang: str, ctx: Ctx):
        self.rng = rng
        self.lang = lang
        self.fonts = [p for p in (ctx.F["km_hand"] if lang == "km" else ctx.F["en_hand"])]
        if lang == "en":  # signature-style scripts are too hard to read as body handwriting -> rarer
            plain = [p for p in self.fonts if p not in ctx.F["en_hand_sig"]]
            self.fonts = plain * 3 + [p for p in self.fonts if p in ctx.F["en_hand_sig"] and "Dancing" in p]
        self.latin_fonts = [p for p in ctx.F["en_hand"] if p not in ctx.F["en_hand_sig"]] or ctx.F["en_hand"]
        self.font = rng.choice(self.fonts)
        self.latin_path = rng.choice([p for p in self.latin_fonts if not any(k in p for k in ("HomemadeApple", "Reenie"))] or self.latin_fonts)
        self.latin_scale = rng.uniform(1.0, 1.25)
        self.stretch = rng.uniform(0.95, 1.25) if lang == "km" else rng.uniform(0.92, 1.12)
        self.slant = rng.choice([rng.uniform(-0.28, -0.05), rng.uniform(-0.1, 0.12), rng.uniform(0.05, 0.25)])
        if lang == "km":
            self.slant -= 0.12  # the Khmer hand fonts are italic already; real writers are mostly upright
        self.gap = rng.uniform(0.0, 0.14)  # extra inter-unit spacing (x size)
        self.word_gap = rng.uniform(0.35, 0.9)
        self.size_j = rng.uniform(0.03, 0.09)
        self.y_j = rng.uniform(0.015, 0.06)
        self.rot_j = rng.uniform(0.5, 4.0)
        self.wobble = rng.uniform(0.02, 0.09)
        self.elastic = rng.uniform(0.01, 0.045)
        self.weight = rng.choice([0, 1, 1]) if lang == "km" else rng.choice([-1, 0, 0, 1])
        self.ink = rng.choice(BLUE_INKS) if rng.random() < 0.8 else rng.choice(BLACK_INKS)
        self.pressure = rng.uniform(0.85, 1.0)
        self.size_scale = rng.uniform(0.85, 1.12)

    def path_for(self, text: str, latin_only=False):
        if latin_only or not KHMER.search(text):
            if self.lang == "en" and font_ok(self.font, text):
                return self.font
            c = [p for p in self.latin_fonts if font_ok(p, text)]
            return self.rng.choice(c) if c else None
        if font_ok(self.font, text):
            return self.font
        c = [p for p in self.fonts if font_ok(p, text)]
        return self.rng.choice(c) if c else None


_PROBE: dict = {}


def path_probe(path: str) -> str:
    """'ក' if the font is a Khmer font (has Khmer glyphs), else ''."""
    v = _PROBE.get(path)
    if v is None:
        v = _PROBE[path] = "ក" if font_ok(path, "ក") else ""
    return v


def units_of(text: str, per_char_latin=False):
    """Text -> render units: Khmer grapheme clusters, Latin words (or chars), and spaces (' ')."""
    from khmerocr.normalize import split_clusters
    out = []
    for i, tok in enumerate(text.split(" ")):
        if i:
            out.append(" ")
        if not tok:
            continue
        if KHMER.search(tok):
            out += split_clusters(tok)
        elif per_char_latin:
            out += list(tok)
        else:
            out.append(tok)
    return out


def layout_units(units, start, max_w, size, wr: Writer, path, rng, allow_cluster_break=True):
    """Greedy placement of units from `start` until max_w. Returns (placements, end_index).
    placement = (unit, x, size_u, path, rot, stretch). Latin / digit units inside a Khmer line are written with the
    page's Latin hand font (the Khmer hand fonts' Latin glyphs look printed)."""
    lat = wr.latin_path if KHMER.search(path_probe(path)) else None
    pl = []
    x = 0.0
    i = start
    last_space = None
    n = len(units)
    while i < n and units[i] == " " and not pl:
        i += 1
    while i < n:
        u = units[i]
        if u == " ":
            last_space = (len(pl), i)
            x += size * wr.word_gap * wr.rng.uniform(0.7, 1.3) * wr.stretch
            i += 1
            continue
        pu = path
        su = size * (1 + rng.gauss(0, wr.size_j))
        if lat and not KHMER.search(u) and any(c.isalnum() for c in u) and font_ok(lat, u):
            pu = lat
            su *= wr.latin_scale
        f = font(pu, max(8, int(round(su))))
        st = wr.stretch * rng.uniform(0.95, 1.05)
        adv = f.getlength(u) * st
        if x + adv > max_w and pl:
            # prefer breaking at the last space if it is not too far back (or for Latin text always)
            if last_space and (not allow_cluster_break or last_space[0] >= len(pl) * 0.6):
                pl = pl[: last_space[0]]
                i = last_space[1] + 1
            break
        pl.append((u, x, su, pu, rng.gauss(0, wr.rot_j * 0.5), st))
        x += adv + size * wr.gap * rng.uniform(0.3, 1.7)
        i += 1
    while i < n and units[i] == " ":
        i += 1
    return pl, i


def render_units(pl, path, size, wr: Writer, rng):
    """Placements -> (ink mask uint8, baseline y in mask, x origin offset). Applies wobble, slant, elastic, weight."""
    if not pl:
        return None
    last = pl[-1]
    tot = last[1] + font(last[3], max(8, int(round(last[2])))).getlength(last[0]) * last[5] + size
    pad = int(size * 1.4)
    Wm = int(tot + 2 * pad + abs(wr.slant) * size * 2.5)
    Hm = int(size * 3.4)
    base = int(size * 1.9)
    m = np.zeros((Hm, Wm), np.uint8)
    # baseline wobble: low-frequency random curve + slight drift
    ph, fr = rng.uniform(0, 6.28), rng.uniform(0.6, 2.2) / max(tot, 1) * 6.28
    drift = rng.gauss(0, wr.wobble * 0.6) * size
    amp = wr.wobble * size
    for (u, x, su, pu, rot, st) in pl:
        f = font(pu, max(8, int(round(su))))
        l, t, r, b = f.getbbox(u, anchor="ls")
        p = int(su * 0.5) + 2
        w0, h0 = int(r - l) + 2 * p, int(b - t) + 2 * p
        if w0 < 3 or h0 < 3:
            continue
        cm = Image.new("L", (w0, h0), 0)
        ImageDraw.Draw(cm).text((p - l, p - t), u, font=f, fill=255, anchor="ls")
        a = np.asarray(cm)
        if abs(st - 1) > 0.01:
            a = cv2.resize(a, (max(3, int(w0 * st)), h0), interpolation=cv2.INTER_LINEAR)
        if abs(rot) > 0.2:
            M = cv2.getRotationMatrix2D((p - l * st, p - t), rot, 1.0)
            a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
        xx = x + pad
        dy = amp * math.sin(ph + fr * x) + drift * (x / max(tot, 1)) + rng.gauss(0, wr.y_j) * size
        ox = int(round(xx + l * st - p * st))
        oy = int(round(base + dy + t - p))
        y0, x0 = max(0, oy), max(0, ox)
        y1, x1 = min(Hm, oy + a.shape[0]), min(Wm, ox + a.shape[1])
        if y1 <= y0 or x1 <= x0:
            continue
        sub = a[y0 - oy: y1 - oy, x0 - ox: x1 - ox]
        np.maximum(m[y0:y1, x0:x1], sub, out=m[y0:y1, x0:x1])
    # slant (shear around the baseline)
    k = wr.slant + rng.gauss(0, 0.03)
    M = np.float32([[1, k, -k * base], [0, 1, 0]])
    m = cv2.warpAffine(m, M, (Wm, Hm), flags=cv2.INTER_LINEAR)
    # elastic distortion
    if wr.elastic > 0.012:
        rs = np.random.default_rng(rng.getrandbits(32))
        sg = size * rng.uniform(0.35, 0.7)
        dx = cv2.GaussianBlur(rs.standard_normal((Hm, Wm)).astype(np.float32), (0, 0), sg)
        dy = cv2.GaussianBlur(rs.standard_normal((Hm, Wm)).astype(np.float32), (0, 0), sg)
        s = wr.elastic * size / (np.abs(dx).max() + 1e-6)
        gx, gy = np.meshgrid(np.arange(Wm, dtype=np.float32), np.arange(Hm, dtype=np.float32))
        m = cv2.remap(m, gx + dx * s, gy + dy * s * (np.abs(dx).max() / (np.abs(dy).max() + 1e-6)), cv2.INTER_LINEAR)
    # pen weight
    if wr.weight < 0 and size > 26:
        m = cv2.erode(m, np.ones((2, 2), np.uint8))
    elif wr.weight > 0:
        m = cv2.dilate(m, np.ones((2, 2), np.uint8))
    return m, base, pad


def put_ink(pg: Page, mask: np.ndarray, ox: int, oy: int, color, rng, pressure=1.0, rs=None):
    """Multiply-blend an ink mask (0..255) onto the page with pressure-modulated opacity. Returns tight ink box."""
    H, W = pg.H, pg.W
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(W, ox + mask.shape[1]), min(H, oy + mask.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    a = mask[y0 - oy: y1 - oy, x0 - ox: x1 - ox].astype(np.float32) / 255.0
    if not (a > 0.2).any():
        return None
    rs = rs or np.random.default_rng(rng.getrandbits(32))
    hh, ww = a.shape
    # ballpoint pressure: smooth variation along the stroke + fine grain
    pv = cv2.resize(rs.uniform(0.0, 1.0, (1, max(2, ww // 40))).astype(np.float32), (ww, 1), interpolation=cv2.INTER_CUBIC)
    pv = np.clip(pressure * (0.8 + 0.2 * pv), 0.45, 1.0)
    a = a * pv * (0.88 + 0.12 * rs.random((hh, ww), dtype=np.float32))
    reg = np.asarray(pg.img.crop((x0, y0, x1, y1)), np.float32)
    col = np.float32(color) / 255.0
    out = reg * (1 - a[..., None] * (1 - col))
    pg.img.paste(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), (x0, y0))
    ys, xs = np.nonzero(a > 0.12)
    if not len(xs):
        return None
    return (float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1))


def hw_place(pg: Page, wr: Writer, pl, path, size, x, base_y, rng, color=None, record=True, pressure=None, bound=None):
    """Render placements with the baseline at base_y starting at x. Records hw line + text. Returns (quad, unit boxes)."""
    r = render_units(pl, path, size, wr, rng)
    if r is None:
        return None, []
    m, base, pad = r
    ox, oy = int(round(x - pad)), int(round(base_y - base))
    ys, xs = np.nonzero(m > 30)
    if not len(xs):
        return None, []
    bx0, by0, bx1, by1 = ox + xs.min(), oy + ys.min(), ox + xs.max() + 1, oy + ys.max() + 1
    B_ = bound or (0, 0, pg.W, pg.H)
    if bx0 < max(0, B_[0]) or by0 < max(0, B_[1]) or bx1 > min(pg.W, B_[2]) or by1 > min(pg.H, B_[3]):
        return None, []
    box = put_ink(pg, m, ox, oy, color or wr.ink, rng, pressure or wr.pressure)
    if box is None:
        return None, []
    q = [round(float(v), 1) for v in (box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3])]
    text = "".join(u for u, *_ in pl).strip()
    text = " ".join(text.split())
    if KHMER.search(text):
        text = normalize_khmer(text)
    if record:
        pg.hw_lines.append(q)
        pg.hw_texts.append(text)
    # approximate unit boxes (for teacher marks / strike-through)
    k = wr.slant
    ub = []
    for (u, ux, su, pu, _, st) in pl:
        uw = font(pu, max(8, int(round(su)))).getlength(u) * st
        ub.append((u, x + ux - k * size * 0.3, base_y - su * 0.75, x + ux + uw - k * size * 0.3, base_y + su * 0.15))
    return q, ub


def hw_text_line(pg, wr, text, size, x, base_y, rng, max_w=None, color=None, latin_only=False, bound=None, pressure=None):
    """One whole short handwritten string (header values, scores, comments); shrinks (down to 0.7x) to fit.
    Returns quad or None."""
    path = wr.path_for(text, latin_only)
    if not path:
        return None
    units = units_of(text, per_char_latin=True)
    for k in range(4):
        sz = size * (0.89 ** k)
        pl, end = layout_units(units, 0, max_w or 1e9, sz, wr, path, rng, allow_cluster_break=False)
        if end < len(units):
            continue
        q, _ = hw_place(pg, wr, pl, path, sz, x, base_y, rng, color=color, bound=bound, pressure=pressure)
        if q:
            return q
    return None


# ----------------------------------------------------------------------------------------------- strokes (marks)
def strokes(pg: Page, polylines, color, width, rng, alpha=0.9):
    """Draw pen strokes (list of point lists) as ink (multiply blend), antialiased. Not labelled."""
    if not polylines:
        return
    P = np.concatenate([np.float32(p) for p in polylines])
    x0, y0 = int(max(0, P[:, 0].min() - width * 3)), int(max(0, P[:, 1].min() - width * 3))
    x1, y1 = int(min(pg.W, P[:, 0].max() + width * 3)), int(min(pg.H, P[:, 1].max() + width * 3))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    S = 2  # supersample
    m = Image.new("L", ((x1 - x0) * S, (y1 - y0) * S), 0)
    d = ImageDraw.Draw(m)
    for p in polylines:
        pts = [((px - x0) * S, (py - y0) * S) for px, py in p]
        if len(pts) >= 2:
            d.line(pts, fill=255, width=max(1, int(round(width * S))), joint="curve")
    a = np.asarray(m.resize((x1 - x0, y1 - y0), Image.BILINEAR))
    put_ink(pg, a, x0, y0, color, rng, alpha)


def _curve(p0, p1, bend, n=12, rng=None, jit=0.0):
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = -(y1 - y0), (x1 - x0)
    L = math.hypot(nx, ny) + 1e-6
    cx, cy = mx + nx / L * bend, my + ny / L * bend
    out = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1
        if rng and jit:
            x += rng.gauss(0, jit)
            y += rng.gauss(0, jit)
        out.append((x, y))
    return out


def tick(x, y, s, rng):
    a = (x, y - s * 0.1)
    b = (x + s * rng.uniform(0.25, 0.4), y + s * rng.uniform(0.25, 0.4))
    c = (x + s * rng.uniform(0.9, 1.5), y - s * rng.uniform(0.5, 0.9))
    return [_curve(a, b, rng.uniform(-2, 2), 5) + _curve(b, c, rng.uniform(-s * 0.15, s * 0.1), 8)[1:]]


def cross(x, y, s, rng):
    return [_curve((x, y - s / 2), (x + s, y + s / 2), rng.uniform(-3, 3), 6),
            _curve((x + s * rng.uniform(0.85, 1.1), y - s / 2), (x + rng.uniform(-0.1, 0.15) * s, y + s / 2), rng.uniform(-3, 3), 6)]


def ellipse(x0, y0, x1, y1, rng):
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + 6, (y1 - y0) / 2 + 5
    a0 = rng.uniform(0, 6.28)
    n = 36
    pts = []
    for i in range(n + rng.randint(2, 6)):
        t = a0 + i / n * 6.28
        k = 1 + 0.08 * math.sin(t * 2 + a0) + rng.gauss(0, 0.01)
        pts.append((cx + rx * k * math.cos(t), cy + ry * k * math.sin(t)))
    return [pts]


# ----------------------------------------------------------------------------------------------- page parts
def ruled_area(pg, rng, x0, x1, y_top, y_bot, sp, col, width, margin_x=None, margin_col=None, margin_w=2, double=False):
    """Draw horizontal rules (+ left margin line). Returns rule y list."""
    d = pg.d
    ys = []
    y = y_top
    while y <= y_bot:
        yj = y + rng.uniform(-0.3, 0.3)
        d.line([x0, yj, x1, yj], fill=col, width=width)
        ys.append(yj)
        y += sp
    if margin_x is not None:
        d.line([margin_x, y_top - sp * rng.choice([0, 1]), margin_x, ys[-1] + rng.choice([0, sp * 0.5])], fill=margin_col, width=margin_w)
        if double:
            d.line([margin_x - 5, y_top - sp, margin_x - 5, ys[-1]], fill=margin_col, width=1)
    return ys


def km_header(pg, ctx, wr, rng, x0, x1, y):
    """Printed candidate form block with handwritten values. Returns (y_end, safe_cut_ys)."""
    W = pg.W
    sz = int(rng.uniform(21, 26))
    pf = ctx.font("body", "នាមត្រកូល និងនាមខ្លួន៖ ថ្ងៃខែឆ្នាំកំណើត")
    pb = ctx.font("bold", "ហត្ថលេខាអនុរក្ស") or pf
    lc = rng.choice([(30, 30, 30), (60, 60, 60), (20, 20, 20)])
    cuts = []
    row_h = sz * rng.uniform(2.0, 2.4)
    mid = x0 + (x1 - x0) * rng.uniform(0.36, 0.44)
    box_w = (x1 - x0) * rng.uniform(0.2, 0.26)
    dotted = rng.random() < 0.5

    def fill_line(xa, xb, yb, style):
        if style == "dots":
            for dx in range(int(xa), int(xb), 6):
                pg.d.ellipse([dx, yb - 1, dx + 2, yb + 1], fill=lc)
        else:
            pg.d.line([xa, yb, xb, yb], fill=lc, width=rng.choice([1, 2]))
        return (xa, yb - 2, xb, yb + 2)

    def field(xl, yl, label, xr, value, latin=False, vsize=None):
        lq = pg.text(xl, yl, label, pf, sz, ctx.fg)
        if not lq:
            return None
        lb = bbox([lq])
        yb = yl + sz * 1.3
        parts = [lq, fill_line(lb[2] + 4, xr, yb, "dots" if dotted else "line")]
        if value:
            vs = vsize or sz * rng.uniform(1.0, 1.35)
            xs = lb[2] + rng.uniform(10, 45)
            q = hw_text_line(pg, wr, value, vs, xs, yb - rng.uniform(1, 5), rng, max_w=xr - xs - 10,
                             latin_only=latin, bound=(lb[2], yl - sz * 1.5, xr + 5, yb + sz * 1.2))
            if q:
                pg.add("handwriting", [q])
                parts.append(q)
        pg.add("form_field", parts)
        return yb

    def date_val(long=False):
        dd, mm, yy = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1985, 2009)
        sep = rng.choice([".", ".", "/", "-", " . "])
        y_s = f"{yy}" if long else f"{yy % 100:02d}"
        return f"{dd:02d}{sep}{mm:02d}{sep}{y_s}"

    yy = y
    rows = []
    # row 1: session date | centre
    if rng.random() < 0.85:
        rows.append(("r1",))
    rows.append(("r2",))
    rows.append(("r3",))
    if rng.random() < 0.8:
        rows.append(("r4",))
    box_rows = []
    left_parts = []
    left_label_done = False
    for k, (r,) in enumerate(rows):
        cuts.append(yy - sz * 0.5)
        if r == "r1":
            field(x0, yy, "សម័យប្រឡង៖", mid - 20, date_val(True), latin=rng.random() < 0.7)
            cv = rng.choice(CENTRE_KM) + rng.choice(["", " "]) + rng.choice(B.X_KH_PROV() + KH_DISTRICT).replace("ខេត្ត", "").replace("រាជធានី", "")
            field(mid + 20, yy, "មណ្ឌលប្រឡង៖", x1, cv if rng.random() < 0.9 else None)
        elif r == "r2":
            if rng.random() < 0.8:
                lab = rng.choice(LEFT_LABELS_KM)
                q = pg.text(x0, yy, lab + "៖", pb, sz, ctx.fg)
                if q:
                    left_parts.append(q)
                    left_label_done = True
            field(mid + 20, yy, "នាមត្រកូល និងនាមខ្លួន៖", x1 - box_w - 20 if rng.random() < 0.3 else x1, ctx.kh_name()[0])
        elif r == "r3":
            if left_label_done:
                left_parts.append(fill_line(x0, mid - 40, yy + sz * 1.3, "dots"))
                if rng.random() < 0.75:
                    sig = B.make_signature(ctx)
                    s = min(1.0, (mid - 40 - x0) * 0.6 / sig.width, sz * 2.2 / sig.height)
                    sig = sig.resize((max(8, int(sig.width * s)), max(8, int(sig.height * s))), Image.LANCZOS)
                    sb = pg.ink(sig, (x0 + rng.uniform(10, 80), yy + sz * 1.3 - sig.height * rng.uniform(0.6, 0.9)), "signature")
                    if sb:
                        left_parts.append(sb)
            box_rows.append(yy)
            field(mid + 20, yy, "ថ្ងៃខែឆ្នាំកំណើត៖", x1 - box_w - 20, date_val(), latin=rng.random() < 0.8)
        elif r == "r4":
            if left_label_done and rng.random() < 0.7:
                left_parts.append(fill_line(x0, mid - 40, yy + sz * 1.3, "dots"))
            box_rows.append(yy)
            yb = field(mid + 20, yy, "ហត្ថលេខា៖", x1 - box_w - 20, None)
            if yb and rng.random() < 0.6:
                sig = B.make_signature(ctx)
                s = min(1.0, 220 / sig.width, sz * 2.0 / sig.height)
                sig = sig.resize((max(8, int(sig.width * s)), max(8, int(sig.height * s))), Image.LANCZOS)
                pg.ink(sig, (mid + 20 + sz * 5, yb - sig.height * 0.8), "signature")
        yy += row_h
    if len(left_parts) > 1:  # label + dotted input lines (+ signature) = one form field
        pg.add("form_field", left_parts)
    elif left_parts:
        pg.add("text", left_parts)
    if box_rows:
        score_box(pg, ctx, wr, rng, x1 - box_w, box_rows[0] - sz * 0.5, x1, box_rows[0] + sz * 2.3, sz, lc, score_p=0.15)
    # printed notice
    cuts.append(yy - sz * 0.4)
    yy += sz * 0.2
    nsz = int(sz * rng.uniform(0.85, 1.0))
    texts = rng.sample(NOTICES_KM, rng.choice([1, 2, 2]))
    qs = []
    for t in texts:
        ye, q = B.text_block(pg, ctx, x0, yy, x1 - x0, t, "body", nsz, max_lines=2, lh=1.55)
        qs += q
        yy = ye
    if qs:
        pg.add("text", qs)
    # dashed separator
    yy += sz * rng.uniform(0.5, 1.0)
    dash, gap = rng.choice([(10, 6), (8, 5), (14, 7)])
    xx = x0 - rng.uniform(0, 40)
    while xx < x1 + 30:
        pg.d.line([xx, yy, min(xx + dash, W - 1), yy], fill=lc, width=2)
        xx += dash + gap
    cuts.append(yy + sz * 0.6)
    yy += sz * rng.uniform(1.8, 2.8)
    # subject field + second score box
    subj = rng.choice(SUBJECTS_KM)
    field(x0 + rng.uniform(40, 120), yy, "វិញ្ញាសា៖", x1 - box_w - 30, subj)
    score_box(pg, ctx, wr, rng, x1 - box_w, yy - sz * 0.5, x1, yy + sz * 1.8, sz, lc, score_p=0.35)
    yy += sz * 2.4
    # title
    tp = ctx.font("bold", "ចម្លើយ")
    q = pg.text((x0 + x1) / 2 + rng.uniform(-30, 30), yy, "ចម្លើយ", tp, int(sz * rng.uniform(1.05, 1.3)), ctx.fg, align="c")
    if q:
        pg.add("title", [q])
        yy = bbox([q])[3] + sz * 0.3
    return yy, cuts


def score_box(pg, ctx, wr, rng, bx0, by0, bx1, by1, sz, lc, score_p=0.3):
    """Rectangle with a printed label ('អក្សរសម្គាល់') [+ red score] -> form_field."""
    pg.d.rectangle([bx0, by0, bx1, by1], outline=lc, width=rng.choice([1, 2, 2]))
    lab = rng.choice(SCORE_LABELS_KM) if rng.random() < 0.3 else "អក្សរសម្គាល់"
    p = ctx.font("bold", lab)
    q = pg.text(bx0 + 8, by0 + (by1 - by0) / 2 - sz * 0.75, lab, p, sz, ctx.fg, bound=(bx0, by0, bx1, by1))
    parts = [(bx0, by0, bx1 + 1, by1 + 1)] + ([q] if q else [])
    if q and rng.random() < score_p:
        lb = bbox([q])
        v = score_text(rng, "km")
        red = Writer(rng, "km" if KHMER.search(v) else "en", ctx)
        red.ink = rng.choice(RED_INKS)
        sq = hw_text_line(pg, red, v, sz * 1.5, lb[2] + 12, by1 - 6, rng, max_w=bx1 - lb[2] - 14, latin_only=not KHMER.search(v),
                          bound=(lb[2], by0 - sz, bx1 + sz, by1 + sz * 0.5))
        if sq:
            pg.add("handwriting", [sq])
            parts.append(sq)
    pg.add("form_field", parts)


def score_text(rng, lang):
    if lang == "km" and rng.random() < 0.3:
        from khmerocr.textgen import to_khmer_digits
        return to_khmer_digits(str(rng.randint(1, 10)))
    r = rng.random()
    if r < 0.5:
        return str(rng.randint(0, 10))
    if r < 0.7:
        return f"{rng.randint(0, 9)}.5" if rng.random() < 0.7 else f"{rng.randint(0, 9)},5"
    if r < 0.85:
        return str(rng.randint(10, 20))
    return f"{rng.randint(5, 20)}/20" if rng.random() < 0.5 else f"{rng.randint(3, 10)}/10"


def x_box(pg, ctx, rng, x0, x1, y0, h, lc):
    """Continuation page: box crossed by an X, notice in the lower triangle (box unlabelled, notice = text)."""
    y1 = y0 + h
    wdt = rng.choice([1, 2, 2, 3])
    top_inset = rng.choice([0, 0, rng.uniform(0, 0.05) * (x1 - x0)])
    pg.d.polygon([(x0 + top_inset, y0), (x1 - top_inset, y0), (x1, y1), (x0, y1)], outline=lc, width=wdt)
    pg.d.line([(x0 + top_inset, y0), (x1, y1)], fill=lc, width=wdt)
    pg.d.line([(x1 - top_inset, y0), (x0, y1)], fill=lc, width=wdt)
    t = rng.choice(NOTICE_CONT_KM) if ctx.lang == "km" else rng.choice(["Do not write in this box.", "Do not write on the crossed area.", "For office use only."])
    sz = int(rng.uniform(20, 26))
    p = ctx.font("bold" if rng.random() < 0.4 else "body", t)
    q = pg.text((x0 + x1) / 2, y1 - h * rng.uniform(0.3, 0.38) - sz * 0.6, t, p, sz, ctx.fg, align="c")
    if q:
        pg.add("text", [q])


# ----------------------------------------------------------------------------------------------- body content
def km_items(rng, ctx, wr, cont):
    """Stream of answer items: (kind, text). kinds: sec, num, sub, dash, cont, blank."""
    items = []
    nstyle = rng.choice(["{}.", "{}.", "{}.", "{}-", "{})"])
    kh_num = rng.random() < 0.25
    from khmerocr.textgen import to_khmer_digits

    def num(i):
        s = nstyle.format(i)
        return to_khmer_digits(s) if kh_num else s
    sec_i = rng.randint(0, 2) if cont else 0
    q_i = rng.randint(1, 4) if cont else 1
    if cont and rng.random() < 0.6:
        items.append(("cont", _km_sentence(rng, 30, 220)))
        if rng.random() < 0.5:
            items.append(("blank", ""))
    for _ in range(40):
        r = rng.random()
        if r < 0.16 or (not items and not cont):
            mk = ROMAN[min(sec_i, 5)] + rng.choice(["", ".", " ", "-"])
            sec_i += 1
            q_i = 1
            t = mk + " " + rng.choice(SECTIONS_KM) if rng.random() < 0.8 else mk
            items.append(("sec", t))
            if rng.random() < 0.55:
                items.append(("blank", ""))
            if rng.random() < 0.5:  # vocabulary list
                for _ in range(rng.randint(2, 6)):
                    w = _km_words(rng)[0]
                    items.append(("dash", f"- {w} ({rng.choice(POS_KM)})៖ {_km_sentence(rng, 12, 70)}"))
                    if rng.random() < 0.15:
                        items.append(("blank", ""))
                items.append(("blank", ""))
                continue
        elif r < 0.75:
            items.append(("num", f"{num(q_i)} {_km_sentence(rng, 20, 320)}"))
            q_i += 1
            if rng.random() < 0.35:  # sub-items
                sub_dash = rng.random() < 0.6
                for j in range(rng.randint(2, 6)):
                    mk = "-" if sub_dash else num(j + 1)
                    items.append(("sub", f"{mk} {_km_sentence(rng, 8, 60)}"))
                    if rng.random() < (0.5 if not sub_dash else 0.1):
                        items.append(("blank", ""))
        else:
            items.append(("cont", _km_sentence(rng, 20, 200)))
        items.append(("blank", "") if rng.random() < 0.6 else ("noblank", ""))
    return items


def en_items(rng, ctx, cont):
    items = []
    q_i = rng.randint(3, 9) if cont else 1
    sec_i = 0
    nstyle = rng.choice(["{}.", "{})", "{}.", "Q{}."])
    E = R["en"]
    for _ in range(40):
        r = rng.random()
        if r < 0.1:
            t = rng.choice(EN_SECTIONS) if rng.random() < 0.7 else f"{ROMAN[min(sec_i, 5)]}."
            sec_i += 1
            items.append(("sec", t))
        elif r < 0.75:
            s = E.sentence(rng, 20, 140)
            if rng.random() < 0.5:
                s += " " + E.sentence(rng, 15, 120)
            items.append(("num", f"{nstyle.format(q_i)} {s}"))
            q_i += 1
            if rng.random() < 0.25:
                for j in range(rng.randint(2, 4)):
                    items.append(("sub", f"{'abcdef'[j]}) {E.sentence(rng, 10, 70)}"))
        else:
            items.append(("cont", E.sentence(rng, 20, 160)))
        items.append(("blank", "") if rng.random() < 0.5 else ("noblank", ""))
    return items


def write_body(pg, ctx, wr, rng, items, rules, x_text, x_right, sp, size, fill_frac):
    """Write items onto rules. Returns list of blocks: dict(quads, rule_idx list, kind, unit boxes per line)."""
    blocks = []
    cur = None
    ri = 0
    n_rules = len(rules)
    stop = max(2, int(n_rules * fill_frac))
    if rng.random() < 0.5:
        ri = rng.choice([0, 0, 1])  # first rule sometimes left empty
    lift = rng.uniform(0.04, 0.22) * size

    def close():
        nonlocal cur
        if cur and cur["quads"]:
            blocks.append(cur)
        cur = None

    for kind, text in items:
        if ri >= stop:
            break
        if kind == "blank":
            close()
            ri += 1
            continue
        if kind == "noblank":
            continue
        if kind in ("sec", "num"):
            close()
        if kind == "cont" and cur is None:
            pass
        if cur is None:
            cur = {"quads": [], "lines": [], "kind": kind}
        path = wr.path_for(text)
        if not path:
            continue
        units = units_of(text, per_char_latin=(wr.lang == "en" and False))
        i = 0
        first = True
        indent0 = size * rng.uniform(0.0, 0.6) if kind in ("sub", "dash") else rng.uniform(0, size * 0.3)
        if kind == "sub" and rng.random() < 0.5:
            indent0 += size * rng.uniform(0.6, 1.5)
        while i < len(units) and ri < stop:
            xs = x_text + (indent0 if first else rng.uniform(-3, size * 0.25))
            mw = x_right - xs - size * rng.uniform(0.2, 1.5)
            pl, j = layout_units(units, i, mw, size * rng.uniform(0.97, 1.03), wr, path, rng, allow_cluster_break=(wr.lang == "km"))
            if not pl:
                break
            q, ub = hw_place(pg, wr, pl, path, size, xs, rules[ri] - lift + rng.uniform(-0.05, 0.05) * size, rng,
                             bound=(x_text - size, rules[ri] - sp * 1.6, x_right + 2, rules[ri] + sp * 0.9))
            if q:
                cur["quads"].append(q)
                cur["lines"].append((ri, q, ub))
            i = j
            ri += 1
            first = False
        if kind in ("sec",):
            close()
    close()
    return blocks, ri


def strike_words(pg, wr, rng, blocks, p=0.04):
    for b in blocks:
        for (ri, q, ub) in b["lines"]:
            if rng.random() < p and len(ub) > 4:
                # contiguous run of 1-3 units (a short word part)
                k = rng.randrange(1, len(ub) - 1)
                run = [u for u in ub[k: k + rng.randint(1, 4)] if u[0] != " "]
                if not run:
                    continue
                x0, x1 = run[0][1], run[-1][3]
                y0, y1 = min(u[2] for u in run), max(u[4] for u in run)
                ym = (y0 + y1) / 2 + rng.uniform(-3, 3)
                pl = [_curve((x0 - 3, ym + rng.uniform(-3, 3)), (x1 + 3, ym + rng.uniform(-4, 4)), rng.uniform(-3, 3), 8)]
                if rng.random() < 0.5:
                    pl.append(_curve((x0 - 2, ym + 4), (x1 + 2, ym + 2 + rng.uniform(-3, 3)), rng.uniform(-2, 2), 8))
                strokes(pg, pl, wr.ink, rng.uniform(1.6, 2.6), rng)


def teacher_marks(pg, ctx, rng, blocks, rules, x_margin, x_text, x_right, sp, size, used_rules, lang):
    """Red marks. Scores / comments are handwriting; ticks, crosses, circles, lines are unlabelled."""
    red = Writer(rng, "en" if lang == "en" else "km", ctx)
    red.ink = rng.choice(RED_INKS)
    red.pressure = rng.uniform(0.8, 1.0)
    w = rng.uniform(1.8, 3.2)
    intensity = rng.random()
    if intensity < 0.12:
        return
    p_line = 0.15 + 0.6 * intensity
    for b in blocks:
        lines = b["lines"]
        for (ri, q, ub) in lines:
            if rng.random() > p_line:
                continue
            r = rng.random()
            ex = q[2] + rng.uniform(5, 40)
            ey = q[5] - size * rng.uniform(0.1, 0.6)
            if r < 0.5 and ex + size * 1.4 < pg.W - 4:
                strokes(pg, tick(ex, ey, size * rng.uniform(0.6, 1.2), rng), red.ink, w, rng)
            elif r < 0.65 and ex + size < pg.W - 4:
                strokes(pg, cross(ex, ey - size * 0.3, size * rng.uniform(0.5, 0.9), rng), red.ink, w, rng)
            elif r < 0.75 and ub:
                u = [v for v in ub if v[0] != " "]
                if len(u) > 3:
                    k = rng.randrange(len(u) - 2)
                    run = u[k: k + rng.randint(1, 3)]
                    strokes(pg, ellipse(run[0][1], min(v[2] for v in run), run[-1][3], max(v[4] for v in run), rng), red.ink, w * 0.8, rng)
            elif r < 0.85:
                # underline part of the line
                xa = rng.uniform(q[0], (q[0] + q[2]) / 2)
                xb = rng.uniform(xa + 30, q[2])
                yb = q[5] + rng.uniform(1, 6)
                strokes(pg, [_curve((xa, yb), (xb, yb + rng.uniform(-4, 4)), rng.uniform(-4, 4), 10, rng, 0.6)], red.ink, w * 0.8, rng)
            elif r < 0.92:
                # small tick above a word
                u = [v for v in ub if v[0] != " "]
                if u:
                    v = rng.choice(u)
                    strokes(pg, tick(v[1], v[2] - 2, size * 0.5, rng), red.ink, w * 0.8, rng)
            else:
                # short red comment after the line end (labelled handwriting)
                t = rng.choice(COMMENTS_KM if lang == "km" else COMMENTS_EN)
                cs = size * rng.uniform(0.7, 0.9)
                if q[2] + cs * 4 < pg.W - 10:
                    cq = hw_text_line(pg, red, t, cs, q[2] + rng.uniform(10, 30), q[5] - rng.uniform(0, 0.3) * size, rng,
                                      max_w=pg.W - 10 - q[2], bound=(q[2], q[1] - sp * 0.8, pg.W - 4, q[5] + sp * 0.5))
                    if cq:
                        pg.add("handwriting", [cq])
        # margin score next to the block (labelled handwriting) + bracket
        if lines and rng.random() < 0.25 + 0.5 * intensity:
            ys = [l[1][1] for l in lines] + [l[1][5] for l in lines]
            yc = (min(ys) + max(ys)) / 2 + rng.uniform(-0.3, 0.3) * sp
            if rng.random() < 0.4 and max(ys) - min(ys) > sp * 1.2:
                xb = x_margin - rng.uniform(4, 20)
                strokes(pg, [_curve((xb, min(ys)), (xb, max(ys)), rng.uniform(6, 18), 14, rng, 0.5)], red.ink, w, rng)
            v = score_text(rng, lang)
            sw = size * rng.uniform(0.9, 1.4)
            sx = rng.uniform(4, max(5, x_margin - sw * len(v) * 0.45 - 8))
            if x_margin > sw * 0.9:
                sq = hw_text_line(pg, red, v, sw, sx, yc + sw * 0.4, rng, max_w=x_margin + 10 - sx, latin_only=not KHMER.search(v),
                                  bound=(0, yc - sw * 2, x_margin + 14, yc + sw * 1.6))
                if sq:
                    pg.add("handwriting", [sq])
            elif rng.random() < 0.6:  # no margin room: score after the last line
                lq = lines[-1][1]
                sq = hw_text_line(pg, red, v, sw, min(lq[2] + 20, x_right - sw * 2), lq[5], rng, latin_only=not KHMER.search(v),
                                  bound=(x_text, lq[1] - sp, pg.W - 2, lq[5] + sp))
                if sq:
                    pg.add("handwriting", [sq])
    # long strokes through unused rules at the bottom (teacher or student closes the page)
    free = len(rules) - used_rules
    if free >= 4 and rng.random() < 0.5:
        y0, y1 = rules[used_rules], rules[-1]
        col = red.ink if rng.random() < 0.7 else rng.choice(BLUE_INKS)
        xa = rng.uniform(x_text + 50, (x_text + x_right) / 2)
        if rng.random() < 0.5:
            strokes(pg, [_curve((xa, y0), (xa + rng.uniform(-60, 60), y1), rng.uniform(-30, 30), 16, rng, 0.8)], col, w, rng)
        else:
            strokes(pg, [_curve((x_text + 20, y0), (x_right - 20, y1), rng.uniform(-40, 40), 16, rng, 0.8)], col, w, rng)
    # stray vertical red line in the margin
    if rng.random() < 0.3 * intensity:
        xv = rng.uniform(4, max(6, x_margin - 6))
        ya = rng.uniform(rules[0], rules[len(rules) // 2])
        strokes(pg, [_curve((xv, ya), (xv + rng.uniform(-15, 15), ya + rng.uniform(3, 8) * sp), rng.uniform(-10, 10), 12, rng, 0.6)],
                red.ink, w * 0.8, rng)


# ----------------------------------------------------------------------------------------------- templates
def _page(rng):
    W, H = 1240, 1754
    paper = rng.choice([(255, 255, 255), (250, 250, 250), (246, 248, 252), (252, 250, 245), (244, 246, 250)])
    pg = Page(W, H, paper)
    pg.meta["paper"] = paper
    pg.meta["degrade_profile"] = "exam_phone"
    pg.meta["degrade_p"] = 1.0
    pg.meta["long_side"] = (1400, 1800)
    return pg


def km_exam_sheet(rng, fake):
    pg = _page(rng)
    ctx = Ctx(rng, "km", fake)
    wr = Writer(rng, "km", ctx)
    W, H = pg.W, pg.H
    first = rng.random() < 0.45
    x0 = W * rng.uniform(0.035, 0.07)
    x1 = W * rng.uniform(0.95, 0.975)
    lc = rng.choice([(30, 30, 30), (55, 55, 55), (20, 20, 20)])
    cuts = []
    if first:
        y, cuts = km_header(pg, ctx, wr, rng, x0, x1, H * rng.uniform(0.02, 0.045))
        y_top = y + rng.uniform(4, 16)
    else:
        bh = H * rng.uniform(0.08, 0.13)
        by = H * rng.uniform(0.0, 0.03)
        x_box(pg, ctx, rng, x0 + rng.uniform(-20, 30), x1 - rng.uniform(-10, 30), by, bh, lc)
        cuts.append(by + bh + 8)
        y_top = by + bh + rng.uniform(25, 60)
    pg.meta["safe_cuts"] = cuts
    sp = rng.uniform(44, 56)
    rule_col = rng.choice([(120, 120, 130), (140, 140, 150), (100, 100, 110), (150, 155, 170), (90, 90, 95)])
    x_margin = x0 + rng.uniform(-10, 30)
    if rng.random() < 0.85:  # top rule of the answer box
        pg.d.line([x_margin, y_top, x1, y_top], fill=lc, width=rng.choice([1, 2]))
    rules = ruled_area(pg, rng, x_margin, x1 + rng.uniform(0, 40), y_top + sp, H * rng.uniform(0.955, 0.985), sp, rule_col,
                       rng.choice([1, 1, 2]), margin_x=x_margin, margin_col=lc, margin_w=rng.choice([2, 3, 3, 4]))
    size = sp * rng.uniform(0.58, 0.76) * wr.size_scale
    size = min(size, sp * 0.82)
    items = km_items(rng, ctx, wr, cont=not first)
    fill = rng.choice([1.0, 1.0, rng.uniform(0.4, 1.0)])
    x_text = x_margin + rng.uniform(6, 22)
    blocks, used = write_body(pg, ctx, wr, rng, items, rules, x_text, x1 - 4, sp, size, fill)
    strike_words(pg, wr, rng, blocks)
    for b in blocks:
        pg.add("handwriting", b["quads"])
    teacher_marks(pg, ctx, rng, blocks, rules, x_margin, x_text, x1, sp, size, used, "km")
    return pg, ctx


def en_exam_worksheet(rng, fake):
    pg = _page(rng)
    ctx = Ctx(rng, "en", fake, style=rng.choice(["serif", "sans"]))
    wr = Writer(rng, "en", ctx)
    W, H = pg.W, pg.H
    x0, x1 = W * rng.uniform(0.06, 0.1), W * rng.uniform(0.92, 0.96)
    lc = rng.choice([(30, 30, 30), (55, 55, 55)])
    sz = int(rng.uniform(22, 27))
    y = H * rng.uniform(0.03, 0.06)
    cuts = [y - 8]
    first = rng.random() < 0.7
    if first:
        # Name / Class / Date row(s)
        fields = [("Name", rng.uniform(0.45, 0.6)), ("Date", 0.0)]
        if rng.random() < 0.6:
            fields.insert(1, (rng.choice(["Class", "Period", "Grade", "Teacher"]), 0.0))
        xx = x0
        wtot = x1 - x0 - (220 if rng.random() < 0.6 else 0)
        rem = wtot
        for k, (lab, frac) in enumerate(fields):
            fw = wtot * frac if frac else rem / (len(fields) - k)
            rem -= fw
            p = ctx.font("body", lab)
            lq = pg.text(xx, y, lab + ":", p, sz, ctx.fg)
            if lq:
                lb = bbox([lq])
                yb = y + sz * 1.25
                pg.d.line([lb[2] + 6, yb, xx + fw - 20, yb], fill=lc, width=1)
                parts = [lq, (lb[2] + 6, yb - 2, xx + fw - 20, yb + 2)]
                v = {"Name": fake.name(), "Date": rng.choice([fake.date(pattern="%m/%d/%y"), fake.date(pattern="%b %d"), fake.date(pattern="%m-%d-%Y")]),
                     "Class": rng.choice(["7B", "Bio 2", "Eng 10", "8-3", "Hist"]), "Period": str(rng.randint(1, 8)),
                     "Grade": str(rng.randint(3, 12)), "Teacher": "Mr. " + fake.last_name()}[lab]
                if rng.random() < 0.92:
                    q = hw_text_line(pg, wr, v, sz * rng.uniform(1.2, 1.5), lb[2] + rng.uniform(12, 30), yb - 2, rng, max_w=xx + fw - lb[2] - 50,
                                     bound=(lb[2], y - sz * 1.6, xx + fw, yb + sz))
                    if q:
                        pg.add("handwriting", [q])
                        parts.append(q)
                pg.add("form_field", parts)
            xx += fw
        if wtot < x1 - x0:
            bx0 = x1 - 200
            pg.d.rectangle([bx0, y - sz * 0.6, x1, y + sz * 2.4], outline=lc, width=2)
            p = ctx.font("bold", "Score")
            q = pg.text(bx0 + 8, y - sz * 0.4, "Score", p, int(sz * 0.8), ctx.fg)
            parts = [(bx0, y - sz * 0.6, x1 + 1, y + sz * 2.4 + 1)] + ([q] if q else [])
            if rng.random() < 0.6:
                red = Writer(rng, "en", ctx)
                red.ink = rng.choice(RED_INKS)
                sq = hw_text_line(pg, red, score_text(rng, "en"), sz * 1.6, bx0 + 50, y + sz * 2.1, rng, max_w=140,
                                  bound=(bx0, y - sz, x1 + 30, y + sz * 3))
                if sq:
                    pg.add("handwriting", [sq])
                    parts.append(sq)
            pg.add("form_field", parts)
        y += sz * 3.0
        cuts.append(y - sz * 0.6)
        t = rng.choice(EN_TITLES).format(n=rng.randint(1, 12))
        if rng.random() < 0.6:
            t = rng.choice(EN_SUBJECTS) + (" - " if rng.random() < 0.5 else " ") + t
        p = ctx.font("bold", t)
        q = pg.text((x0 + x1) / 2, y, t, p, int(sz * rng.uniform(1.3, 1.7)), ctx.fg, align="c")
        if q:
            pg.add("title", [q])
            y = bbox([q])[3] + sz * 0.5
        if rng.random() < 0.5:
            ins = rng.choice(["Answer all questions in complete sentences.", "Write your answers on the lines below.",
                              "Show your work. Use blue or black ink only.", "Read each question carefully before answering."])
            q = pg.text(x0, y, ins, ctx.font("italic", ins), int(sz * 0.85), ctx.fg)
            if q:
                pg.add("text", [q])
                y = bbox([q])[3] + sz * 0.6
    pg.meta["safe_cuts"] = cuts
    sp = rng.uniform(42, 54)
    rule_col = rng.choice([(150, 175, 215), (130, 160, 210), (170, 185, 215), (140, 140, 150)])
    margin_col = rng.choice([(225, 120, 130), (210, 90, 100), (200, 140, 150)]) if rng.random() < 0.75 else lc
    x_margin = x0 + rng.uniform(20, 60)
    rules = ruled_area(pg, rng, 0 if rng.random() < 0.6 else x0, W, y + sp, H * rng.uniform(0.95, 0.98), sp, rule_col, 1,
                       margin_x=x_margin, margin_col=margin_col, margin_w=rng.choice([1, 2]), double=rng.random() < 0.3)
    size = sp * rng.uniform(0.7, 0.9) * wr.size_scale
    size = min(size, sp * 0.95)
    items = en_items(rng, ctx, cont=not first)
    x_text = x_margin + rng.uniform(8, 25)
    blocks, used = write_body(pg, ctx, wr, rng, items, rules, x_text, x1, sp, size, rng.choice([1.0, rng.uniform(0.4, 1.0)]))
    strike_words(pg, wr, rng, blocks, p=0.03)
    for b in blocks:
        pg.add("handwriting", b["quads"])
    teacher_marks(pg, ctx, rng, blocks, rules, x_margin, x_text, x1, sp, size, used, "en")
    return pg, ctx


EXAM_KM = {"km_exam_sheet": (km_exam_sheet, 0.8)}
EXAM_EN = {"en_exam_worksheet": (en_exam_worksheet, 0.3)}
