"""Safe image decoding for uploads and stored pages.

- the pixel count is checked from the header (Pillow, lazy) before decoding, so decompression bombs are refused
- alpha is composited onto white (transparent scans/screenshots otherwise decode as an all-black page)
- 16-bit images are reduced to 8 bit, grey to BGR
- the long side is capped at settings.max_image_side (the stored page is the capped image, so every coordinate the
  API returns refers to the image served at image_url)
- EXIF orientation is applied for images without alpha (cv2.IMREAD_COLOR does that)
"""

from __future__ import annotations

import io

import cv2
import numpy as np

from app.config import settings


class ImageDecodeError(ValueError):
    """The bytes are not a readable image (or a too large one). The message is safe to show to users."""


def _header_size(buf: bytes):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is optional; then only the post-decode checks apply
        return None
    try:
        with Image.open(io.BytesIO(buf)) as im:
            return im.size
    except Exception:
        return None  # let OpenCV decide (some formats Pillow cannot parse)


def decode_image(buf: bytes) -> tuple[np.ndarray, bool]:
    """bytes -> (BGR uint8 image, changed) where changed = the pixels differ from a plain decode of the file
    (alpha composited, depth reduced or downscaled), i.e. the caller must store the returned image."""
    if not buf:
        raise ImageDecodeError("empty file")
    size = _header_size(buf)
    if size and size[0] * size[1] > settings.max_image_pixels:
        raise ImageDecodeError(f"image too large ({size[0]}x{size[1]} px; at most "
                               f"{settings.max_image_pixels // 1_000_000} megapixels)")
    arr = np.frombuffer(buf, np.uint8)
    raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.size == 0:
        raise ImageDecodeError("could not read the file as an image")
    changed = False
    if raw.dtype != np.uint8:
        raw = (raw / (256.0 if raw.dtype == np.uint16 else max(1.0, float(raw.max()) / 255.0))).clip(0, 255)
        raw = raw.astype(np.uint8)
        changed = True
    if raw.ndim == 3 and raw.shape[2] == 4:
        a = raw[..., 3:4].astype(np.float32) / 255.0
        bgr = (raw[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)).round().astype(np.uint8)
        changed = True
    elif changed:  # 16-bit without alpha
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw[..., :3]
    else:
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # applies EXIF orientation
        if bgr is None:
            raise ImageDecodeError("could not read the file as an image")
    h, w = bgr.shape[:2]
    if max(h, w) > settings.max_image_side:
        s = settings.max_image_side / max(h, w)
        bgr = cv2.resize(bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
        changed = True
    return np.ascontiguousarray(bgr), changed


def read_image(path) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_image(f.read())[0]
