"""Per-page context: language, fonts, sizes, colours, and the text generators (Khmer corpus / Gutenberg / Faker).

All personal data is synthetic: Faker en_US for US documents, random combinations of common Khmer name parts for
Cambodian documents. ID numbers, account numbers, MRZ etc. are random digits.
"""

from __future__ import annotations

import random

from khmerocr.textgen import MONTHS, to_khmer_digits

from .res import R, font_ok

KH_FAMILY = [("សុខ", "SOK"), ("ចាន់", "CHAN"), ("លី", "LY"), ("ហេង", "HENG"), ("គឹម", "KIM"), ("ស៊ុន", "SUN"), ("ម៉ៅ", "MAO"), ("ឈឹម", "CHHIM"),
             ("ពេជ្រ", "PICH"), ("សំ", "SAM"), ("កែវ", "KEO"), ("ឡុង", "LONG"), ("ហ៊ុល", "HUL"), ("ទេព", "TEP"), ("យឹម", "YIM"), ("ណុប", "NOP"),
             ("ជា", "CHEA"), ("ឯក", "EK"), ("អ៊ុក", "OUK"), ("ប៉ែន", "PEN"), ("ហួត", "HOUT"), ("មាស", "MEAS"), ("នួន", "NUON"), ("ព្រំ", "PROM"),
             ("ខៀវ", "KHIEV"), ("វង្ស", "VONG"), ("រស់", "ROS"), ("សយ", "SOY"), ("ឈួន", "CHHUON"), ("ស៊ាង", "SEANG"), ("តាំង", "TANG"), ("អ៊ឹម", "IM")]
KH_GIVEN = [("សុភា", "SOPHEA"), ("ដារ៉ា", "DARA"), ("វណ្ណា", "VANNA"), ("ចាន់ថា", "CHANTHA"), ("សុខា", "SOKHA"), ("រតនា", "RATANA"),
            ("បុប្ផា", "BOPHA"), ("ពិសិដ្ឋ", "PISETH"), ("សុវណ្ណ", "SOVANN"), ("ស្រីមុំ", "SREYMOM"), ("វិចិត្រ", "VICHET"), ("មុន្នី", "MONY"),
            ("រិទ្ធី", "RITHY"), ("កុសល", "KOSAL"), ("សំណាង", "SAMNANG"), ("ធីតា", "THIDA"), ("សុផល", "SOPHAL"), ("លក្ខិណា", "LEAKHENA"),
            ("បញ្ញា", "PANHA"), ("វាសនា", "VEASNA"), ("ផល្លា", "PHALLA"), ("សុជាតា", "SOCHEATA"), ("ម៉ានិត", "MANIT"), ("សីហា", "SEYHA"),
            ("ចរិយា", "CHARIYA"), ("សោភា", "SOPHEA"), ("នារី", "NEARY"), ("វុទ្ធី", "VUTHY"), ("សុធា", "SOTHEA"), ("ពៅ", "POV"),
            ("រក្សា", "REAKSA"), ("តារា", "TARA"), ("បូរ៉ា", "BORA"), ("សម្បត្តិ", "SAMBATH"), ("ស្រីនិច", "SREYNICH"), ("ឧត្តម", "OUDOM")]
KH_PROV = ["រាជធានីភ្នំពេញ", "ខេត្តសៀមរាប", "ខេត្តបាត់ដំបង", "ខេត្តកំពង់ចាម", "ខេត្តកណ្ដាល", "ខេត្តតាកែវ", "ខេត្តព្រៃវែង", "ខេត្តកំពត",
           "ខេត្តព្រះសីហនុ", "ខេត្តស្វាយរៀង", "ខេត្តកំពង់ស្ពឺ", "ខេត្តពោធិ៍សាត់", "ខេត្តកំពង់ធំ", "ខេត្តក្រចេះ", "ខេត្តបន្ទាយមានជ័យ"]
KH_PROV_EN = ["Phnom Penh", "Siem Reap", "Battambang", "Kampong Cham", "Kandal", "Takeo", "Prey Veng", "Kampot", "Preah Sihanouk",
              "Svay Rieng", "Kampong Speu", "Pursat", "Kampong Thom", "Kratie", "Banteay Meanchey"]
KH_DISTRICT = ["ខណ្ឌដូនពេញ", "ខណ្ឌចំការមន", "ខណ្ឌទួលគោក", "ខណ្ឌសែនសុខ", "ខណ្ឌមានជ័យ", "ស្រុកកៀនស្វាយ", "ស្រុកពួក", "ស្រុកសង្កែ", "ក្រុងសៀមរាប", "ស្រុកអង្គស្នួល"]
KH_COMMUNE = ["សង្កាត់ផ្សារថ្មី", "សង្កាត់បឹងកេងកង", "ឃុំព្រែកឯង", "ឃុំស្វាយដង្គំ", "សង្កាត់ទឹកល្អក់", "ឃុំកំពង់ស្វាយ", "សង្កាត់ស្ទឹងមានជ័យ"]
KH_VILLAGE = ["ភូមិ១", "ភូមិ៣", "ភូមិត្រពាំងឈូក", "ភូមិព្រែកតាកុង", "ភូមិក្រាំង", "ភូមិថ្មី", "ភូមិស្វាយ", "ភូមិពោធិ៍"]
MINISTRIES = [("ក្រសួងអប់រំ យុវជន និងកីឡា", "Ministry of Education, Youth and Sport"), ("ក្រសួងសុខាភិបាល", "Ministry of Health"),
              ("ក្រសួងមហាផ្ទៃ", "Ministry of Interior"), ("ក្រសួងសេដ្ឋកិច្ចនិងហិរញ្ញវត្ថុ", "Ministry of Economy and Finance"),
              ("ក្រសួងពាណិជ្ជកម្ម", "Ministry of Commerce"), ("ក្រសួងកសិកម្ម រុក្ខាប្រមាញ់ និងនេសាទ", "Ministry of Agriculture"),
              ("ក្រសួងការងារ និងបណ្តុះបណ្តាលវិជ្ជាជីវៈ", "Ministry of Labour"), ("ក្រសួងបរិស្ថាន", "Ministry of Environment"),
              ("ក្រសួងប្រៃសណីយ៍ និងទូរគមនាគមន៍", "Ministry of Posts and Telecommunications"), ("ក្រសួងវប្បធម៌ និងវិចិត្រសិល្បៈ", "Ministry of Culture")]
SUB_UNITS = ["អគ្គនាយកដ្ឋានរដ្ឋបាល និងហិរញ្ញវត្ថុ", "នាយកដ្ឋានបុគ្គលិក", "មន្ទីរអប់រំ យុវជន និងកីឡារាជធានីភ្នំពេញ", "នាយកដ្ឋានផែនការ",
             "ខុទ្ទកាល័យ", "អគ្គលេខាធិការដ្ឋាន", "នាយកដ្ឋានព័ត៌មានវិទ្យា", "មន្ទីរសុខាភិបាលខេត្តសៀមរាប"]
KH_COMPANIES = ["ក្រុមហ៊ុន អង្គរ ត្រេដ ឯ.ក", "ហាងលក់គ្រឿងសំណង់ សុខសាន្ត", "ភោជនីយដ្ឋាន មេគង្គ", "ហាងកាហ្វេ ភ្នំ", "ក្រុមហ៊ុន បាយ័ន ឌីជីថល",
                "ផ្សារទំនើប លក្ខិណា", "ឱសថស្ថាន សុខភាពល្អ", "ក្រុមហ៊ុន ដឹកជញ្ជូន រហ័ស", "ហាងលក់ទូរស័ព្ទ ស្មាត", "ក្រុមហ៊ុន ទឹកស្អាត កម្ពុជា"]
KH_PRODUCTS = ["អង្ករ", "ស្ករស", "ទឹកត្រី", "ប្រេងឆា", "ស៊ីម៉ងត៍", "ដែកថែប", "ក្រដាស A4", "ទឹកសុទ្ធ", "កាហ្វេទឹកដោះគោ", "បាយឆា", "មីឆា",
               "ទូរស័ព្ទដៃ", "កុំព្យូទ័រយួរដៃ", "ថ្នាំលាប", "ស្បែកជើង", "អាវយឺត", "សាប៊ូ", "ថ្នាំពេទ្យ", "ប៊ិច", "សៀវភៅ"]
KH_TITLES_LETTER = ["លិខិតអញ្ជើញ", "សេចក្តីជូនដំណឹង", "លិខិតស្នើសុំ", "សេចក្តីសម្រេច", "ប្រកាស", "សារាចរណែនាំ", "លិខិតបញ្ជាក់",
                    "កម្មវត្ថុ", "លិខិតឆ្លើយតប", "ដីកាអមដំណើរ", "សេចក្តីណែនាំ"]
ROLES = ["នាយក", "ប្រធាន", "រដ្ឋមន្ត្រី", "អគ្គនាយក", "ប្រធានមន្ទីរ", "អ្នកស្នើសុំ", "អនុប្រធាន", "លេខាធិការ", "អ្នកទទួល", "ប្រធាននាយកដ្ឋាន"]
KH_SUBJECT_PREFIX = ["កម្មវត្ថុ៖ ", "យោង៖ ", "សូមគោរពជូន ", "ជូនចំពោះ "]

US_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "WA", "AZ", "MA", "CO", "OR", "MN"]


class Ctx:
    def __init__(self, rng: random.Random, lang: str, fake, style: str | None = None):
        self.rng, self.lang, self.fake = rng, lang, fake
        F = R["fonts"]
        self.F = F
        if lang == "km":
            self.pools = {"body": F["km_body"], "bold": F["km_bold"], "title": F["km_moul"] + F["km_bold"], "moul": F["km_moul"],
                          "italic": F["km_body"], "mono": F["en_mono"] + F["km_body"], "hand": F["km_hand"], "display": F["km_display"],
                          "latin": F["en_sans"] + F["en_serif"], "latin_bold": F["en_bold"], "sig": F["km_hand"] + F["en_hand_sig"]}
        else:
            style = style or rng.choice(["serif", "sans", "sans", "cond", "mono_doc"])
            body = {"serif": F["en_serif"], "sans": F["en_sans"], "cond": F["en_cond"], "mono_doc": F["en_mono"]}[style]
            self.pools = {"body": body, "bold": F["en_bold"] if style != "mono_doc" else F["en_mono_bold"] + F["en_bold"],
                          "title": F["en_bold"] + F["en_display"], "moul": F["en_display"], "italic": F["en_italic"],
                          "mono": F["en_mono"], "hand": F["en_hand"], "display": F["en_display"] + F["en_bolditalic"],
                          "latin": body, "latin_bold": F["en_bold"], "sig": F["en_hand_sig"] + F["en_hand"]}
        self.style = style
        self.pref = {k: rng.choice(v) for k, v in self.pools.items() if v}
        self.size = rng.choice([22, 24, 25, 26, 28]) if lang == "km" else rng.choice([19, 20, 21, 22, 23, 24, 26])
        self.lh = rng.uniform(1.45, 1.8) if lang == "km" else rng.uniform(1.2, 1.5)
        self.fg = rng.choice([(15, 15, 15), (30, 30, 30), (0, 0, 0), (20, 20, 50), (45, 45, 45)])
        self.accent = rng.choice([(150, 20, 30), (20, 50, 130), (10, 90, 60), (30, 30, 30), (100, 40, 120), (0, 90, 140), (180, 90, 0)])
        self.justify = rng.random() < 0.45
        self.hw_ink = rng.choice([(20, 30, 140), (10, 10, 10), (25, 45, 170), (0, 0, 90), (40, 40, 40), (15, 60, 150)])
        self.kh_digits = rng.random() < 0.6

    # ---------------------------------------------------------------- fonts
    def font(self, role: str, text: str, fallback=True) -> str | None:
        pool = self.pools.get(role) or self.pools["body"]
        p = self.pref.get(role)
        if p and font_ok(p, text):
            return p
        cand = list(pool)
        self.rng.shuffle(cand)
        for p in cand[:8]:
            if font_ok(p, text):
                return p
        if fallback:
            extra = (self.pools["body"] + self.pools["latin"] + self.F["km_body"] + self.F["en_sans"])
            for p in extra:
                if font_ok(p, text):
                    return p
        return None

    # ---------------------------------------------------------------- text
    def d(self, s: str) -> str:
        """digits in Khmer numerals on Khmer pages (per-page choice)."""
        return to_khmer_digits(str(s)) if (self.lang == "km" and self.kh_digits) else str(s)

    def para(self, n_chars: int) -> str:
        if self.lang == "km":
            return R["km"].paragraph(self.rng, n_chars)
        if self.rng.random() < 0.8:
            return R["en"].paragraph(self.rng, n_chars)
        out = ""
        while len(out) < n_chars:
            out += (" " if out else "") + self.fake.paragraph(nb_sentences=4)
        return out

    def phrase(self, lo=6, hi=40) -> str:
        if self.lang == "km":
            return R["km"].phrase(self.rng, lo, hi)
        r = self.rng.random()
        if r < 0.5:
            s = R["en"].sentence(self.rng, lo, hi * 3)
            words = s.rstrip(".!?;:,").split()
            k = self.rng.randint(2, 9)
            st = self.rng.randrange(max(1, len(words) - k))
            s = " ".join(words[st:st + k]).strip(",;:")
        elif r < 0.8:
            s = self.fake.catch_phrase()
        else:
            s = self.fake.bs()
        s = s[:hi].rsplit(" ", 1)[0] if len(s) > hi else s
        return s[:1].upper() + s[1:] if s else "Notes"

    def heading(self) -> str:
        if self.lang == "km":
            return self.phrase(8, 36)
        s = self.phrase(8, 44)
        return s.title() if self.rng.random() < 0.6 else s

    def number(self) -> str:
        if self.lang == "km":
            return R["km"].number(self.rng)[:24]
        return self.rng.choice([self.money(), str(self.rng.randint(1, 9999)), f"{self.rng.uniform(0, 100):.1f}%", self.fake.date(pattern="%m/%d/%Y")])

    def money(self, lo=1, hi=5000, cur=None) -> str:
        v = self.rng.uniform(lo, hi)
        if self.lang == "km":
            cur = cur or self.rng.choice(["$", "៛", "USD", "KHR"])
            if cur in ("៛", "KHR"):
                v = int(v * 4100 / 100) * 100
                s = f"{v:,}"
                return self.d(s) + (" ៛" if cur == "៛" else " KHR")
            return (f"${v:,.2f}" if cur == "$" else f"{v:,.2f} USD")
        return f"${v:,.2f}"

    def date(self) -> str:
        rng = self.rng
        if self.lang == "km":
            dd, mm, yy = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2015, 2027)
            return rng.choice([f"ថ្ងៃទី{self.d(dd)} ខែ{MONTHS[mm - 1]} ឆ្នាំ{self.d(yy)}", self.d(f"{dd:02d}/{mm:02d}/{yy}"), self.d(f"{dd:02d}-{mm:02d}-{yy}")])
        return rng.choice([self.fake.date(pattern="%B %d, %Y"), self.fake.date(pattern="%m/%d/%Y"), self.fake.date(pattern="%d %b %Y"),
                           self.fake.date(pattern="%Y-%m-%d")])

    def kh_name(self, latin=False):
        f, g = self.rng.choice(KH_FAMILY), self.rng.choice(KH_GIVEN)
        return (f"{f[1]} {g[1]}" if latin else f"{f[0]} {g[0]}"), (f"{f[1]} {g[1]}", f"{f[0]} {g[0]}")

    def name(self) -> str:
        if self.lang == "km":
            return self.kh_name()[0]
        return self.fake.name()

    def address(self) -> list[str]:
        rng = self.rng
        if self.lang == "km":
            parts = [f"ផ្ទះលេខ{self.d(rng.randint(1, 300))} ផ្លូវ{self.d(rng.randint(1, 600))}", rng.choice(KH_VILLAGE) + " " + rng.choice(KH_COMMUNE),
                     rng.choice(KH_DISTRICT) + " " + rng.choice(KH_PROV)]
            return parts
        return [self.fake.street_address(), f"{self.fake.city()}, {rng.choice(US_STATES)} {self.fake.zipcode()}"]

    def phone(self) -> str:
        rng = self.rng
        if self.lang == "km":
            s = rng.choice([f"0{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}", f"(+855) {rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}"])
            return self.d(s) if rng.random() < 0.4 else s
        return self.fake.phone_number()

    def company(self) -> str:
        return self.rng.choice(KH_COMPANIES) if self.lang == "km" else self.fake.company()

    def label(self, km: str, en: str, bilingual_p=0.35) -> str:
        """Field label: Khmer (optionally 'Khmer / English') on Khmer pages, English on US pages."""
        if self.lang == "km":
            return f"{km} / {en}" if self.rng.random() < bilingual_p else km
        return en

    def colon(self) -> str:
        return self.rng.choice(["៖", "៖ ", ":", " :"]) if self.lang == "km" else self.rng.choice([":", ":", ""])
