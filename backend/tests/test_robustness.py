"""Follow-ups to the QA review: model-load failures must not brick requests; decoding rules."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def test_failed_switch_keeps_previous_model(manifest_ok, monkeypatch):
    from app.services.model_pool import ModelPool
    pool = ModelPool()
    old = pool.active["rec"]
    pool.get("rec")
    other = next(s.id for s in pool.registry.for_stage("rec") if s.id != old)
    real_build = pool.registry.build

    def broken(spec, **kw):
        if spec.id == other:
            raise ImportError("no torch on this phone")
        return real_build(spec, **kw)

    monkeypatch.setattr(pool.registry, "build", broken)
    with pytest.raises(ValueError, match="previous model kept"):
        pool.switch("rec", other)
    assert pool.active["rec"] == old and pool.get("rec").spec.id == old
    assert "no torch" in pool.info()["errors"]["rec"]


def test_unloadable_active_model_falls_back(manifest_ok, monkeypatch):
    from app.services.model_pool import ModelPool
    pool = ModelPool()
    bad = pool.active["layout"]
    real_build = pool.registry.build
    monkeypatch.setattr(pool.registry, "build",
                        lambda spec, **kw: (_ for _ in ()).throw(RuntimeError("corrupt")) if spec.id == bad
                        else real_build(spec, **kw))
    impl = pool.get("layout")  # nothing loaded yet -> another available layout model
    assert impl.spec.id != bad and pool.active["layout"] == impl.spec.id
    assert "corrupt" in pool.errors["layout"]


def test_decode_rules():
    from app.services.image_io import ImageDecodeError, decode_image
    gray16 = (np.arange(200 * 300, dtype=np.uint32).reshape(200, 300) % 65535).astype(np.uint16)
    bgr, changed = decode_image(cv2.imencode(".png", gray16)[1].tobytes())
    assert bgr.dtype == np.uint8 and bgr.shape == (200, 300, 3) and changed
    rgba = np.zeros((50, 60, 4), np.uint8)  # fully transparent -> white
    bgr, changed = decode_image(cv2.imencode(".png", rgba)[1].tobytes())
    assert changed and bgr.min() == 255
    jpg = cv2.imencode(".jpg", np.full((40, 80, 3), 200, np.uint8))[1].tobytes()
    bgr, changed = decode_image(jpg)
    assert not changed and bgr.shape == (40, 80, 3)
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image")
