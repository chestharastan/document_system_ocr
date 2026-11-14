"""Font families the classifier learns. A family groups files of the same design: the Khmer OS fonts (Danh Hong,
LGPL) and their OFL re-releases by Google (Battambang, Content, Moul ...) are the same designs, so a document typed in
"Khmer OS Battambang" and one typed in "Battambang" are reported as the same family. Weight (regular / bold) is a
separate output: true bold files where they exist, synthetic (dilated) bold for the others, as Word does.

Only fonts with open licences are rendered (Khmer UI: MS EULA, Savuth Pen: no licence -> not used, as in docgen)."""

from __future__ import annotations

from pathlib import Path

WF = Path("/home/hello/ocr/work/fonts")
NOTO = Path("/usr/share/fonts/noto")
LIB = Path("/usr/share/fonts/liberation")

# family name -> [(file, weight)]  weight in {"regular", "bold"}
_FAMILIES: list[tuple[str, list[tuple[Path, str]]]] = [
    ("Khmer OS", [(WF / "KhmerOS.ttf", "regular")]),
    ("Khmer OS System", [(WF / "KhmerOSsys.ttf", "regular")]),
    ("Khmer OS Battambang", [(WF / "KhmerOSbattambang.ttf", "regular"), (WF / "Battambang-Regular.ttf", "regular"),
                             (WF / "Battambang-Light.ttf", "regular"), (WF / "Battambang-Bold.ttf", "bold"),
                             (WF / "Battambang-Black.ttf", "bold")]),
    ("Khmer OS Content", [(WF / "KhmerOScontent.ttf", "regular"), (WF / "Content-Regular.ttf", "regular"),
                          (WF / "Content-Bold.ttf", "bold")]),
    ("Khmer OS Siemreap", [(WF / "KhmerOSsiemreap.ttf", "regular"), (WF / "Siemreap.ttf", "regular")]),
    ("Khmer OS Muol", [(WF / "KhmerOSmuol.ttf", "regular"), (WF / "Moul-Regular.ttf", "regular")]),
    ("Khmer OS Muol Light", [(WF / "KhmerOSmuollight.ttf", "regular")]),
    ("Khmer OS Muol Pali", [(WF / "KhmerOSmuolpali.ttf", "regular"), (WF / "Moulpali-Regular.ttf", "regular")]),
    ("Khmer OS Bokor", [(WF / "KhmerOSbokor.ttf", "regular"), (WF / "Bokor-Regular.ttf", "regular")]),
    ("Khmer OS Metal Chrieng", [(WF / "KhmerOSmetalchrieng.ttf", "regular"), (WF / "Metal-Regular.ttf", "regular")]),
    ("Khmer OS Fasthand", [(WF / "KhmerOSfasthand.ttf", "regular"), (WF / "Fasthand-Regular.ttf", "regular")]),
    ("Khmer OS Freehand", [(WF / "KhmerOSfreehand.ttf", "regular"), (WF / "Freehand-Regular.ttf", "regular")]),
    ("Noto Sans Khmer", [(NOTO / "NotoSansKhmer-Light.ttf", "regular"), (NOTO / "NotoSansKhmer-Regular.ttf", "regular"),
                         (NOTO / "NotoSansKhmer-Medium.ttf", "regular"), (NOTO / "NotoSansKhmer-Bold.ttf", "bold"),
                         (NOTO / "NotoSansKhmer-Black.ttf", "bold")]),
    ("Noto Serif Khmer", [(NOTO / "NotoSerifKhmer-Light.ttf", "regular"), (NOTO / "NotoSerifKhmer-Regular.ttf", "regular"),
                          (NOTO / "NotoSerifKhmer-Medium.ttf", "regular"), (NOTO / "NotoSerifKhmer-Bold.ttf", "bold"),
                          (NOTO / "NotoSerifKhmer-Black.ttf", "bold")]),
    ("Khmer Digital", [(WF / "KhmerDigital-Light.ttf", "regular"), (WF / "KhmerDigital-Regular.ttf", "regular"),
                       (WF / "KhmerDigital-Bold.ttf", "bold")]),
    ("Hanuman", [(WF / "Hanuman[wght].ttf", "regular")]),
    ("Kantumruy Pro", [(WF / "KantumruyPro[wght].ttf", "regular")]),
    ("Nokora", [(WF / "Nokora[wght].ttf", "regular")]),
    ("Odor Mean Chey", [(WF / "OdorMeanChey-Regular.ttf", "regular")]),
    ("Chenla", [(WF / "Chenla.ttf", "regular")]),
    ("Preahvihear", [(WF / "Preahvihear-Regular.ttf", "regular")]),
    ("Suwannaphum", [(WF / "Suwannaphum-Light.ttf", "regular"), (WF / "Suwannaphum-Regular.ttf", "regular"),
                     (WF / "Suwannaphum-Bold.ttf", "bold"), (WF / "Suwannaphum-Black.ttf", "bold")]),
    ("Koulen", [(WF / "Koulen-Regular.ttf", "regular")]),
    ("Dangrek", [(WF / "Dangrek-Regular.ttf", "regular")]),
    ("Bayon", [(WF / "Bayon-Regular.ttf", "regular")]),
    ("Angkor", [(WF / "Angkor-Regular.ttf", "regular")]),
    ("Taprom", [(WF / "Taprom-Regular.ttf", "regular")]),
    # Latin-only families (English lines set in a Latin font: Times / Arial look-alikes)
    ("Latin serif", [(LIB / "LiberationSerif-Regular.ttf", "regular"), (LIB / "LiberationSerif-Bold.ttf", "bold"),
                     (NOTO / "NotoSerif-Regular.ttf", "regular"), (NOTO / "NotoSerif-Bold.ttf", "bold")]),
    ("Latin sans", [(LIB / "LiberationSans-Regular.ttf", "regular"), (LIB / "LiberationSans-Bold.ttf", "bold"),
                    (NOTO / "NotoSans-Regular.ttf", "regular"), (NOTO / "NotoSans-Bold.ttf", "bold")]),
]

LATIN_ONLY = {"Latin serif", "Latin sans"}
# families whose Latin glyphs are the usual way English appears inside a Khmer document
KHMER_WITH_LATIN = {"Khmer OS", "Khmer OS System", "Khmer OS Battambang", "Khmer OS Content", "Khmer OS Siemreap",
                    "Noto Sans Khmer", "Noto Serif Khmer", "Khmer Digital", "Hanuman", "Kantumruy Pro", "Nokora"}
WEIGHTS = ["regular", "bold"]
SLANTS = ["upright", "italic"]


def families() -> list[tuple[str, list[tuple[Path, str]]]]:
    """Families with the files that exist on this machine (a family with no file is dropped)."""
    out = []
    for name, files in _FAMILIES:
        have = [(p, w) for p, w in files if p.exists()]
        if have:
            out.append((name, have))
    return out


FAMILY_NAMES = [n for n, _ in families()]
