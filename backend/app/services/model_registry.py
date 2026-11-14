"""Model registry: reads the manifest (.model/v2/models.json) and builds stage implementations from its entries.

Manifest format (relative ``file`` / ``vocab`` paths are relative to the manifest's folder; symlinks are fine):

    {
      "schema": 2,
      "defaults": {"layout": "<id>", "textdet": "<id>", "rec": "<id>"},
      "models": [
        {"id": "layout_yolo11s_v1", "stage": "layout", "kind": "yolo_onnx", "file": "layout/x.onnx",
         "input_size": [1024, 1024], "classes": ["text", ...], "version": "v1", "sha256": "...", "label": "..."},
        {"id": "db_v1", "stage": "textdet", "kind": "db_onnx", "file": "textdet/db.onnx", "params": {"unclip_ratio": 1.6}},
        {"id": "crnn_m_v1", "stage": "rec", "kind": "crnn_onnx", "file": "rec/rec.onnx", "vocab": "rec/vocab.json",
         "input_size": [48, null]}
      ]
    }

stage in {layout, textdet, rec, font, structure, table}; kind selects the implementation (KINDS below). The optional
stages (font, structure, table) may have no default and no entry at all: the pipeline then skips them. A new version of a known kind needs
only a file + an entry (backend/scripts/register_model.py writes the entry incl. sha256 and ONNX class names).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

log = logging.getLogger("app.registry")

STAGES = ("layout", "textdet", "rec", "font", "structure", "table")  # v3 tables: "table" = table-structure (optional, default may be absent)
STAGE_TITLES = {"layout": "Layout", "textdet": "Detection", "rec": "Recognition", "font": "Font", "structure": "Structure",
                "table": "Table structure"}  # "stage" strings the v1 UI uses

# kind -> (module, class, required stage). Modules are imported only when a model of that kind is loaded.
KINDS = {
    "yolo_onnx": ("app.services.stages.layout_yolo_onnx", "YoloOnnxLayout", "layout"),
    "db_onnx": ("app.services.stages.textdet_db_onnx", "DBOnnxDetector", "textdet"),
    "crnn_onnx": ("app.services.stages.rec_crnn_onnx", "CrnnOnnxRecognizer", "rec"),
    "font_cnn_onnx": ("app.services.stages.font_cnn_onnx", "FontOnnxClassifier", "font"),
    "structure_onnx": ("app.services.stages.structure_onnx", "StructureOnnx", "structure"),
    "tsr_yolo_onnx": ("app.services.stages.table_yolo_onnx", "TableStructureDetector", "table"),  # v3 tables
    # legacy PyTorch stack (needs torch / ultralytics / doclayout_yolo installed)
    "ultralytics_pt": ("app.services.stages.legacy_torch", "UltralyticsPtLayout", "layout"),
    "doclayout_yolo_pt": ("app.services.stages.legacy_torch", "DocLayoutYoloPt", "layout"),
    "ultralytics_lines": ("app.services.stages.legacy_torch", "UltralyticsLineDetector", "textdet"),
    "torch_resnet_bilstm": ("app.services.stages.legacy_torch", "TorchResNetRecognizer", "rec"),
}
ONNX_KINDS = {"yolo_onnx", "db_onnx", "crnn_onnx", "tsr_yolo_onnx"}

# The pre-v2 models (.model/v1/...), listed when their files exist so old setups keep working.
LEGACY_V1 = [
    {"id": "doclayout_yolo_docstructbench", "stage": "layout", "kind": "doclayout_yolo_pt", "version": "v1",
     "label": "DocLayout-YOLO DocStructBench (torch, legacy)", "input_size": [1120, 1120],
     "file": "v1/layout/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"},
    {"id": "line_level", "stage": "textdet", "kind": "ultralytics_lines", "version": "v1",
     "label": "Line Level yolo11s (torch, legacy)", "file": "v1/detection/best.pt"},
    {"id": "word_group", "stage": "textdet", "kind": "ultralytics_lines", "version": "v1",
     "label": "Word Group yolov8 (torch, legacy)", "file": "v1/detection/word_group/best.pt"},
    {"id": "resnet_bilstm_ctc", "stage": "rec", "kind": "torch_resnet_bilstm", "version": "v1",
     "label": "ResNet BiLSTM CTC (torch, legacy)", "file": "v1/recognition/best_model.pth",
     "vocab": "v1/recognition/char.json"},
]


@dataclass
class ModelSpec:
    id: str
    stage: str
    kind: str
    path: Path
    label: str = ""
    version: str = ""
    sha256: str = ""
    vocab_path: Path | None = None
    input_size: list | None = None
    classes: list | None = None
    params: dict = field(default_factory=dict)
    source: str = "manifest"

    @property
    def available(self) -> bool:
        return self.path.exists() and (self.vocab_path is None or self.vocab_path.exists())

    @property
    def backend(self) -> str:
        return "onnxruntime" if self.kind in ONNX_KINDS else "torch"

    def as_dict(self) -> dict:
        return {"id": self.id, "stage": self.stage, "kind": self.kind, "label": self.label or self.id,
                "version": self.version, "backend": self.backend, "path": str(self.path),
                "available": self.available, "sha256": self.sha256, "input_size": self.input_size,
                "classes": self.classes, "params": self.params, "source": self.source}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _spec(entry: dict, base: Path, source: str) -> ModelSpec:
    stage, kind = entry["stage"], entry["kind"]
    if stage not in STAGES:
        raise ValueError(f"model {entry.get('id')}: unknown stage {stage!r}")
    if kind not in KINDS:
        raise ValueError(f"model {entry.get('id')}: unknown kind {kind!r} (known: {', '.join(KINDS)})")
    if KINDS[kind][2] != stage:
        raise ValueError(f"model {entry.get('id')}: kind {kind} is a {KINDS[kind][2]} model, not {stage}")

    def p(v):
        if not v:
            return None
        v = Path(v)
        return v if v.is_absolute() else base / v

    return ModelSpec(id=entry["id"], stage=stage, kind=kind, path=p(entry["file"]), label=entry.get("label", ""),
                     version=str(entry.get("version", "")), sha256=entry.get("sha256", ""),
                     vocab_path=p(entry.get("vocab")), input_size=entry.get("input_size"),
                     classes=entry.get("classes"), params=entry.get("params") or {}, source=source)


class Registry:
    def __init__(self, manifest: Path | None = None, model_root: Path | None = None):
        self.manifest_path = manifest or settings.manifest_path
        self.model_root = model_root or settings.model_root_path
        self.specs: dict[str, ModelSpec] = {}
        self.defaults: dict[str, str] = {}
        self.reload()

    def reload(self):
        self.specs, self.defaults = {}, {}
        if self.manifest_path and self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            base = self.manifest_path.parent
            for e in data.get("models", []):
                s = _spec(e, base, "manifest")
                self.specs[s.id] = s
            self.defaults = dict(data.get("defaults") or {})
        else:
            log.warning("model manifest %s not found - only legacy v1 models (if present) are available",
                        self.manifest_path)
        for e in LEGACY_V1:
            if e["id"] not in self.specs:
                s = _spec(e, self.model_root, "legacy")
                if s.available:
                    self.specs[s.id] = s

    def for_stage(self, stage: str, available_only: bool = True) -> list[ModelSpec]:
        return [s for s in self.specs.values() if s.stage == stage and (s.available or not available_only)]

    def get(self, model_id: str, stage: str | None = None) -> ModelSpec:
        s = self.specs.get(model_id)
        if s is None or (stage and s.stage != stage):
            raise ValueError(f"Unknown {stage or ''} model: {model_id}".replace("  ", " "))
        if not s.available:
            raise ValueError(f"Model {model_id} is listed but its file is missing: {s.path}")
        return s

    def default_id(self, stage: str, configured: str = "") -> str | None:
        for cand in (configured, self.defaults.get(stage)):
            if not cand:
                continue
            if cand in self.specs and self.specs[cand].available:
                return cand
            log.warning("%s model %r not available; falling back", stage, cand)
        avail = self.for_stage(stage)
        onnx = [s for s in avail if s.backend == "onnxruntime"]
        return (onnx or avail)[0].id if avail else None

    def build(self, spec: ModelSpec, **kw):
        if settings.verify_sha256 and spec.sha256:
            got = sha256_file(spec.path)
            if got != spec.sha256:
                log.warning("sha256 mismatch for %s (%s): manifest %s, file %s", spec.id, spec.path, spec.sha256, got)
        mod, cls, _ = KINDS[spec.kind]
        impl = getattr(importlib.import_module(mod), cls)
        return impl(spec, **kw)
