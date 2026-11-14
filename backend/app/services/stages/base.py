"""Stage interfaces of the document pipeline.

    image ──► LayoutDetector ──► regions [{cls, label, score, box}]
          └─► TextLineDetector ──► lines [(quad 4x2, score)]          (page pass, optionally per region crop)
                         line crops ──► LineRecognizer ──► [(text, confidence)]   (batched)
                         line results ──► PostCorrector ──► corrected line results  (future 4th stage)

Images are numpy uint8 BGR arrays [H, W, 3] (OpenCV convention), boxes are page pixels [x0, y0, x1, y1],
quads are float32 [4, 2] in the order top-left, top-right, bottom-right, bottom-left.

Every implementation is constructed from one manifest entry (``ModelSpec``) by the registry, so a new model type only
needs a class here plus a ``kind`` string in services/model_registry.py; a new *version* of an existing type needs no
code at all (drop the file in .model/v2/ and add a manifest entry).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Region:
    cls: str                 # v2 taxonomy class name (app/taxonomy.py)
    score: float
    box: list[float]         # [x0, y0, x1, y1] page pixels
    label: str = ""          # class name as the model reported it (before mapping onto v2)

    def as_dict(self) -> dict:
        return {"cls": self.cls, "label": self.label or self.cls, "score": round(float(self.score), 4),
                "box": [round(float(v), 1) for v in self.box]}


@dataclass
class LineResult:
    """One recognised text line; the unit the PostCorrector works on."""
    text: str
    confidence: float
    quad: np.ndarray | None = None       # [4,2] page pixels
    block_cls: str = ""                  # class of the layout block the line belongs to
    extra: dict = field(default_factory=dict)


class Stage(ABC):
    stage: str = ""
    spec: Any = None                     # the ModelSpec it was built from

    def info(self) -> dict:
        return {"id": getattr(self.spec, "id", None), "kind": getattr(self.spec, "kind", None)}


class LayoutDetector(Stage):
    stage = "layout"

    @abstractmethod
    def detect(self, image: np.ndarray, conf: float | None = None) -> list[Region]:
        """BGR page -> regions (all classes, already mapped onto the v2 taxonomy, NMS applied)."""


class TextLineDetector(Stage):
    stage = "textdet"

    @abstractmethod
    def detect(self, image: np.ndarray, **kw) -> list[tuple[np.ndarray, float]]:
        """BGR page or crop -> [(quad float32 [4,2] in the input image's pixels, score)].
        Accepted keywords (implementations ignore what they do not support):
        thresh, box_thresh, unclip_ratio, v_extra, long_min, long_max."""


class LineRecognizer(Stage):
    stage = "rec"

    @abstractmethod
    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Line crops (BGR or gray, any size) -> [(text, confidence in 0..1)], same order. Implementations batch."""


class FontClassifier(Stage):
    stage = "font"

    @abstractmethod
    def classify(self, crops: list[np.ndarray]) -> list[dict]:
        """Line crops -> [{family, confidence, weight, weight_confidence, slant, slant_confidence, probs}], same order."""


class PostCorrector(ABC):
    """Future 4th stage: fix recognition errors with context (language model, dictionary, field formats ...).

    ``correct`` receives every recognised line of one page in reading order plus a context dict:
        page_width, page_height  page size in pixels
        blocks                   [{cls, box, line_indices}] the layout blocks and which lines they hold
        language                 hint, e.g. "km" or "en" (may be absent)
    and returns a list of the same length (it may change ``text`` and ``confidence``, and add keys to ``extra``,
    e.g. extra["raw_text"] with the uncorrected text). It must not reorder, drop or add lines.
    Select an implementation with POSTCORRECT=<name> (see services/stages/postcorrect.py).
    """

    name: str = "none"

    @abstractmethod
    def correct(self, lines: list[LineResult], context: dict) -> list[LineResult]:
        ...

    def info(self) -> dict:
        return {"id": self.name, "kind": type(self).__name__}
