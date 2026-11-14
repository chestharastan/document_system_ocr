"""Shared fixtures. Uploaded pages/crops go to a temp folder; tests that need models skip when the manifest is missing."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

_TMP = Path(tempfile.mkdtemp(prefix="ocr_test_storage_"))
atexit.register(shutil.rmtree, _TMP, True)
os.environ.setdefault("PAGE_DIR", str(_TMP / "pages"))
os.environ.setdefault("CROP_DIR", str(_TMP / "crops"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT_CANDIDATES = [
    "/home/hello/ocr/khm_repo_clone/Khm-Dataset-Creation/fonts/print/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
]

TITLE = "របាយការណ៍ប្រចាំឆ្នាំ"
PARAGRAPH = [
    "ព្រះរាជាណាចក្រកម្ពុជា ជាតិ សាសនា ព្រះមហាក្សត្រ",
    "ក្រសួងពាណិជ្ជកម្ម បានរៀបចំកិច្ចប្រជុំប្រចាំខែ",
    "នៅថ្ងៃទី៥ ខែវិច្ឆិកា ឆ្នាំ២០២៤ រាជធានីភ្នំពេញ",
    "សមិទ្ធិផលនេះឆ្លុះបញ្ចាំងពីការយកចិត្តទុកដាក់ខ្ពស់",
]


def _font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return None


def render_page(path: Path) -> Path:
    """A synthetic A4-ish page: Khmer title, two paragraphs, a grey 'figure' box."""
    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    big, body = _font(56), _font(30)
    if big is None:
        pytest.skip("no Khmer font available to render the synthetic page")
    d.text((W // 2 - 260, 140), TITLE, font=big, fill="black")
    y = 330
    for _ in range(2):
        for line in PARAGRAPH:
            d.text((150, y), line, font=body, fill="black")
            y += 62
        y += 70
    d.rectangle([150, y + 40, 650, y + 420], fill=(150, 150, 150), outline="black", width=4)
    d.ellipse([260, y + 120, 540, y + 360], fill=(90, 90, 90))
    img.save(path)
    return path


@pytest.fixture(scope="session")
def synthetic_page(tmp_path_factory) -> Path:
    return render_page(tmp_path_factory.mktemp("pages") / "synthetic_page.png")


@pytest.fixture(scope="session")
def manifest_ok():
    from app.config import settings
    if not settings.manifest_path.exists():
        pytest.skip(f"model manifest {settings.manifest_path} missing (see README 'Models')")
    return settings.manifest_path


@pytest.fixture(scope="session")
def client(manifest_ok):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
