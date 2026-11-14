"""Resources for docgen: paths, font pools (with HarfBuzz + cmap checks), Khmer/English text sources.

Everything heavy is loaded ONCE in the parent process (load()) and inherited by the fork()ed workers.
"""

from __future__ import annotations

import glob
import gzip
import random
import sys
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

OLD_REPO = Path("/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation")
ASSETS = Path("/home/hello/ocr/khm_repo_clone/det_data/assets")
WORK = Path("/home/hello/ocr/docsys/work/data")
EXTRA_FONTS = Path("/home/hello/ocr/khm_repo_clone/extra_fonts")
EXTRA_HAND = Path("/home/hello/ocr/khm_repo_clone/extra_fonts_hand")
DOCGEN_FONTS = WORK / "docgen_fonts"
EN_SENTENCES = WORK / "docgen_text" / "en_sentences.txt.gz"

if str(OLD_REPO) not in sys.path:
    sys.path.insert(0, str(OLD_REPO))

from khmerocr.normalize import normalize_khmer  # noqa: E402
from khmerocr.shaping import FontChecker  # noqa: E402

_SYS = "/usr/share/fonts"


def _g(*pats):
    out = []
    for p in pats:
        out += sorted(glob.glob(p))
    return [p for p in out if Path(p).exists()]


def font_pools() -> dict[str, list[str]]:
    kp = str(OLD_REPO / "fonts/print")
    kd = str(OLD_REPO / "fonts/display")
    P = {
        # ---- Khmer (OFL repo fonts, Noto Khmer, Khmer OS (LGPL)); Khmer UI (MS EULA) and Savuth Pen (no licence) excluded
        "km_body": _g(f"{kp}/Battambang-Regular.ttf", f"{kp}/Content-Regular.ttf", f"{kp}/KhmerDigital-Regular.ttf",
                      f"{kp}/NotoSansKhmer-Regular.ttf", f"{kp}/Preahvihear-Regular.ttf", f"{kp}/Siemreap-Regular.ttf",
                      f"{kp}/Suwannaphum-Regular.ttf", f"{kp}/KhmerDigital-Light.ttf", f"{_SYS}/noto/NotoSerifKhmer-Regular.ttf",
                      f"{_SYS}/noto/NotoSansKhmer-Medium.ttf", f"{EXTRA_FONTS}/KhmerOS.ttf", f"{EXTRA_FONTS}/KhmerOSbattambang.ttf",
                      f"{EXTRA_FONTS}/KhmerOScontent.ttf", f"{EXTRA_FONTS}/KhmerOSsiemreap.ttf", f"{EXTRA_FONTS}/KhmerOSsys.ttf"),
        "km_bold": _g(f"{kp}/Battambang-Bold.ttf", f"{kp}/KhmerDigital-Bold.ttf", f"{kp}/NotoSansKhmer-Bold.ttf",
                      f"{_SYS}/noto/NotoSerifKhmer-Bold.ttf", f"{_SYS}/noto/NotoSansKhmer-Black.ttf", f"{kd}/Bayon-Regular.ttf",
                      f"{kd}/Koulen-Regular.ttf", f"{kd}/Dangrek-Regular.ttf"),
        "km_moul": _g(f"{kd}/Moul-Regular.ttf", f"{kd}/Moulpali-Regular.ttf"),
        "km_display": _g(f"{kd}/Angkor-Regular.ttf", f"{kd}/Bokor-Regular.ttf", f"{kd}/Metal-Regular.ttf", f"{kd}/Taprom-Regular.ttf",
                         f"{kd}/Koulen-Regular.ttf", f"{kd}/Bayon-Regular.ttf", f"{kd}/Dangrek-Regular.ttf", f"{kd}/Moul-Regular.ttf"),
        "km_hand": _g(str(OLD_REPO / "fonts/handwriting/*.ttf"), f"{EXTRA_HAND}/KhmerOSfasthand.ttf", f"{EXTRA_HAND}/KhmerOSfreehand.ttf"),
        # ---- Latin (system fonts with open licences + TikTok Sans instances, OFL)
        "en_serif": _g(f"{_SYS}/noto/NotoSerif-Regular.ttf", f"{_SYS}/noto/NotoSerif-Light.ttf", f"{_SYS}/noto/NotoSerif-Medium.ttf",
                       f"{_SYS}/liberation/LiberationSerif-Regular.ttf", f"{_SYS}/noto/NotoSerifDisplay-Regular.ttf"),
        "en_sans": _g(f"{_SYS}/noto/NotoSans-Regular.ttf", f"{_SYS}/noto/NotoSans-Light.ttf", f"{_SYS}/noto/NotoSans-Medium.ttf",
                      f"{_SYS}/liberation/LiberationSans-Regular.ttf", f"{_SYS}/Adwaita/AdwaitaSans-Regular.ttf",
                      f"{DOCGEN_FONTS}/inst/TikTokSans-Regular.ttf"),
        "en_cond": _g(f"{DOCGEN_FONTS}/inst/TikTokSans-Condensed.ttf", f"{DOCGEN_FONTS}/inst/TikTokSans-SemiCondensed.ttf"),
        "en_mono": _g(f"{_SYS}/liberation/LiberationMono-Regular.ttf", f"{_SYS}/noto/NotoSansMono-Regular.ttf",
                      f"{_SYS}/Adwaita/AdwaitaMono-Regular.ttf"),
        "en_mono_bold": _g(f"{_SYS}/liberation/LiberationMono-Bold.ttf", f"{_SYS}/noto/NotoSansMono-Bold.ttf", f"{_SYS}/Adwaita/AdwaitaMono-Bold.ttf"),
        "en_bold": _g(f"{_SYS}/noto/NotoSans-Bold.ttf", f"{_SYS}/noto/NotoSans-Black.ttf", f"{_SYS}/noto/NotoSerif-Bold.ttf",
                      f"{_SYS}/liberation/LiberationSans-Bold.ttf", f"{_SYS}/liberation/LiberationSerif-Bold.ttf",
                      f"{_SYS}/noto/NotoSerifDisplay-Bold.ttf", f"{DOCGEN_FONTS}/inst/TikTokSans-Bold.ttf",
                      f"{DOCGEN_FONTS}/inst/TikTokSans-CondensedBold.ttf"),
        "en_italic": _g(f"{_SYS}/noto/NotoSans-Italic.ttf", f"{_SYS}/noto/NotoSerif-Italic.ttf", f"{_SYS}/liberation/LiberationSans-Italic.ttf",
                        f"{_SYS}/liberation/LiberationSerif-Italic.ttf", f"{_SYS}/noto/NotoSerif-LightItalic.ttf",
                        f"{_SYS}/Adwaita/AdwaitaSans-Italic.ttf"),
        "en_bolditalic": _g(f"{_SYS}/noto/NotoSans-BoldItalic.ttf", f"{_SYS}/noto/NotoSerif-BoldItalic.ttf",
                            f"{_SYS}/liberation/LiberationSerif-BoldItalic.ttf", f"{_SYS}/liberation/LiberationSans-BoldItalic.ttf"),
        "en_display": _g(f"{_SYS}/noto/NotoSerifDisplay-Black.ttf", f"{_SYS}/noto/NotoSerifDisplay-BoldItalic.ttf",
                         f"{DOCGEN_FONTS}/inst/TikTokSans-Expanded.ttf", f"{_SYS}/noto/NotoSans-Black.ttf",
                         f"{DOCGEN_FONTS}/inst/TikTokSans-CondensedBold.ttf"),
        "en_hand": _g(f"{DOCGEN_FONTS}/hand/*.ttf"),
        "math": _g(f"{_SYS}/noto/NotoSansMath-Regular.ttf"),
    }
    P["en_hand_sig"] = [p for p in P["en_hand"] if any(k in p for k in ("Sacramento", "DancingScript", "HomemadeApple", "ReenieBeanie", "NothingYou"))]
    return P


_checkers: dict[str, FontChecker] = {}
_cmaps: dict[str, set] = {}


def checker(path: str) -> FontChecker:
    c = _checkers.get(path)
    if c is None:
        c = _checkers[path] = FontChecker(path)
    return c


def cmap(path: str) -> set:
    c = _cmaps.get(path)
    if c is None:
        from fontTools.ttLib import TTFont
        c = _cmaps[path] = set(TTFont(path, lazy=True).getBestCmap().keys())
    return c


@lru_cache(maxsize=200_000)
def font_ok(path: str, text: str) -> bool:
    """HarfBuzz check (no .notdef, no dotted circle) + fontTools cmap check (every non-space char mapped)."""
    cm = cmap(path)
    for ch in set(text):
        if ch not in " ​" and ord(ch) not in cm:
            return False
    return checker(path).problem(text) is None


@lru_cache(maxsize=4096)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, max(6, int(size)), layout_engine=ImageFont.Layout.RAQM)


# --------------------------------------------------------------------------- text
class EnglishText:
    def __init__(self, path: Path = EN_SENTENCES):
        self.pars: list[list[str]] = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for ln in f:
                self.pars.append(ln.rstrip("\n").split("\t"))
        self.sents = [s for p in self.pars for s in p]

    def paragraph(self, rng: random.Random, n_chars: int) -> str:
        p = rng.choice(self.pars)
        i = rng.randrange(len(p))
        out = []
        while sum(len(s) + 1 for s in out) < n_chars:
            if i >= len(p):
                p, i = rng.choice(self.pars), 0
            out.append(p[i])
            i += 1
        return " ".join(out)

    def sentence(self, rng: random.Random, lo=20, hi=120) -> str:
        for _ in range(20):
            s = rng.choice(self.sents)
            if lo <= len(s) <= hi:
                return s
        return rng.choice(self.sents)[:hi]


class KhmerText:
    def __init__(self):
        from khmerocr.textgen import TextSources
        self.ts = TextSources(OLD_REPO / "data/corpus", "train")
        self.news = [normalize_khmer(t) for _, t in self.ts.news]

    def paragraph(self, rng: random.Random, n_chars: int) -> str:
        out = ""
        i = rng.randrange(len(self.news))
        while len(out) < n_chars:
            out += (" " if out else "") + self.news[i % len(self.news)]
            i += 1
        return out

    def phrase(self, rng: random.Random, lo=6, hi=40) -> str:
        t = rng.choice(self.news)
        t = t[: rng.randint(lo, hi)]
        if " " in t[lo:]:
            t = t.rsplit(" ", 1)[0]
        t = t.strip("។ ,")
        from khmerocr.textgen import trunc
        return trunc(t, hi) or "ការងារ"

    def number(self, rng: random.Random) -> str:
        return self.ts.numbers(rng)


R: dict = {}


def load():
    if R:
        return R
    R["fonts"] = font_pools()
    R["en"] = EnglishText()
    R["km"] = KhmerText()
    R["photos"] = sorted(glob.glob(str(ASSETS / "photos/*")))
    R["sig_real"] = sorted(glob.glob(str(ASSETS / "signatures/*.png")))
    R["stamp_real"] = sorted(glob.glob(str(ASSETS / "stamps/*.png")))
    return R
