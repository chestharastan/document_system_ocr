"""Holds the active model of each stage (lazy-loaded, switchable at runtime) and the post-corrector."""

from __future__ import annotations

import logging
import threading

from app.config import settings
from app.services import model_registry as reg
from app.services.stages import ort_utils
from app.services.stages.postcorrect import build_postcorrector

log = logging.getLogger("app.models")


class ModelPool:
    def __init__(self, registry: reg.Registry | None = None):
        self.registry = registry or reg.Registry()
        configured = {"layout": settings.layout_model, "textdet": settings.textdet_model, "rec": settings.rec_model,
                      "font": settings.font_model, "structure": settings.structure_model,
                      "table": getattr(settings, "table_model", "")}  # v3 tables: optional TSR stage (TABLE_MODEL)
        self.active: dict[str, str | None] = {s: self.registry.default_id(s, configured.get(s, "")) for s in reg.STAGES}
        # v3 tables: the structure engine requested through POST /pipeline/settings (rules | model | llm); None = auto
        # (model when a structure model is loaded, else rules) - resolved per request by the pipeline
        self.structure_engine: str | None = None
        self._loaded: dict[str, tuple[str, object]] = {}
        self._locks = {s: threading.Lock() for s in reg.STAGES}  # one per stage: a slow load blocks only its stage
        self.errors: dict[str, str] = {}  # last load error per stage (shown in /models and /pipeline/info)
        self.postcorrector = build_postcorrector(settings.postcorrect)

    # ------------------------------------------------------------------ loading
    def _kwargs(self, spec: reg.ModelSpec) -> dict:
        common = {"device": settings.ort_device}
        if spec.backend == "onnxruntime":
            common["threads"] = settings.ort_threads
        if spec.stage == "layout":
            common.update(conf=settings.layout_conf, iou=settings.layout_iou, max_det=settings.layout_max_det)
            if spec.kind == "doclayout_yolo_pt":
                common.update(imgsz=settings.legacy_layout_imgsz, conf=min(settings.layout_conf, 0.25))
        elif spec.stage == "textdet":
            if spec.kind == "db_onnx":
                common.update(thresh=settings.det_thresh, box_thresh=settings.det_box_thresh,
                              unclip_ratio=settings.det_unclip_ratio, v_extra=settings.det_v_extra,
                              long_min=settings.det_long_min, long_max=settings.det_long_max)
            else:
                common.update(conf=settings.legacy_det_conf)
        elif spec.stage == "rec" and spec.kind == "crnn_onnx":
            common.update(max_batch=settings.rec_batch)
        elif spec.stage == "font":
            common.update(max_batch=4 * settings.rec_batch)
        return common

    def _build(self, stage: str, model_id: str):
        spec = self.registry.get(model_id, stage)
        log.info("loading %s model %s (%s) from %s", stage, spec.id, spec.kind, spec.path)
        return self.registry.build(spec, **self._kwargs(spec))

    def get(self, stage: str):
        """The loaded implementation of the active model. If it cannot be loaded, fall back to the previously loaded
        model of the stage, else to any other available model of the stage (ONNX first); the error is kept in
        self.errors. Raises only when no model of the stage can be loaded."""
        if stage not in self._locks:
            raise ValueError(f"Unknown stage {stage!r}")
        with self._locks[stage]:
            model_id = self.active.get(stage)
            cur = self._loaded.get(stage)
            if cur and cur[0] == model_id:
                return cur[1]
            if not model_id:
                raise RuntimeError(f"No {stage} model available - check {self.registry.manifest_path} (see README)")
            try:
                impl = self._build(stage, model_id)
            except Exception as exc:
                msg = f"could not load {stage} model {model_id}: {type(exc).__name__}: {exc}"
                log.error(msg)
                self.errors[stage] = msg
                if cur:  # keep serving with the model that worked
                    self.active[stage] = cur[0]
                    return cur[1]
                others = sorted((s for s in self.registry.for_stage(stage) if s.id != model_id),
                                key=lambda s: s.backend != "onnxruntime")
                for spec in others:
                    try:
                        impl = self._build(stage, spec.id)
                    except Exception as exc2:  # noqa: PERF203
                        log.error("fallback %s model %s failed too: %s", stage, spec.id, exc2)
                        continue
                    self.active[stage] = spec.id
                    self._loaded[stage] = (spec.id, impl)
                    return impl
                raise RuntimeError(msg) from exc
            self._loaded[stage] = (model_id, impl)
            return impl

    @property
    def layout(self):
        return self.get("layout")

    @property
    def textdet(self):
        return self.get("textdet")

    @property
    def recognizer(self):
        return self.get("rec")

    @property
    def font(self):
        """The font classifier, or None when no font model is registered / enabled (the stage is optional)."""
        if not settings.detect_fonts or not self.active.get("font"):
            return None
        try:
            return self.get("font")
        except Exception as exc:
            log.error("font model unavailable: %s", exc)
            return None

    @property
    def structure(self):
        """The structure model, or None when none is registered / enabled (optional stage; rules are the fallback)."""
        if not settings.use_structure_model or not self.active.get("structure"):
            return None
        try:
            return self.get("structure")
        except Exception as exc:
            log.error("structure model unavailable: %s", exc)
            return None

    @property
    def table(self):
        """v3 tables: the table-structure (TSR) model, or None when none is registered (the rules are the fallback)."""
        if not self.active.get("table"):
            return None
        try:
            return self.get("table")
        except Exception as exc:
            log.error("table-structure model unavailable: %s", exc)
            return None

    def switch(self, stage: str, model_id: str):
        """Load the new model now; only when that works does it become active (the previous one keeps serving
        otherwise and the error is raised as ValueError -> HTTP 422)."""
        if stage not in reg.STAGES:
            raise ValueError(f"Unknown stage {stage!r}; one of {', '.join(reg.STAGES)}")
        self.registry.get(model_id, stage)  # validates id + file
        with self._locks[stage]:
            cur = self._loaded.get(stage)
            if cur and cur[0] == model_id:
                self.active[stage] = model_id
                return
            try:
                impl = self._build(stage, model_id)
            except Exception as exc:
                msg = f"could not load {stage} model {model_id}: {type(exc).__name__}: {exc}"
                log.error(msg)
                self.errors[stage] = msg
                raise ValueError(msg + " (previous model kept)") from exc
            self.active[stage] = model_id
            self._loaded[stage] = (model_id, impl)
            self.errors.pop(stage, None)

    # v1 API names
    def switch_recog(self, model_id: str):
        self.switch("rec", model_id)

    def switch_det(self, model_id: str):
        self.switch("textdet", model_id)

    def warmup(self):
        for s in reg.STAGES:
            if self.active.get(s):
                try:
                    self.get(s)
                except Exception as exc:  # keep the API up; the error is reported per request
                    log.error("could not load %s model %s: %s", s, self.active[s], exc)

    # ------------------------------------------------------------------ info
    def is_loaded(self, stage: str) -> bool:
        cur = self._loaded.get(stage)
        return bool(cur and cur[0] == self.active.get(stage))

    @property
    def device(self) -> str:
        return ort_utils.device_name(settings.ort_device)

    def info(self) -> dict:
        """GET /models (v1-compatible shape + v2 extras)."""
        models = []
        for s in reg.STAGES:
            for spec in self.registry.for_stage(s):
                models.append({
                    "id": spec.id, "name": spec.id, "label": spec.label or spec.id,
                    "stage": reg.STAGE_TITLES[s], "stage_key": s, "kind": spec.kind, "backend": spec.backend,
                    "version": spec.version, "path": str(spec.path), "active": spec.id == self.active.get(s),
                })
        return {
            "device": self.device,
            "warm": all(self.is_loaded(s) for s in reg.STAGES if self.active.get(s)),
            "active_model": self.active.get("rec"),
            "active_det_model": self.active.get("textdet"),
            "active_layout_model": self.active.get("layout"),
            "active": dict(self.active),
            "postcorrect": self.postcorrector.name,
            "errors": dict(self.errors),
            "models": models,
        }

    def pipeline_info(self) -> dict:
        """GET /pipeline/info: everything about the configured pipeline."""
        stages = {}
        for s in reg.STAGES:
            mid = self.active.get(s)
            entry = {"active": mid, "loaded": self.is_loaded(s), "error": self.errors.get(s),
                     "available": [m.as_dict() for m in self.registry.for_stage(s, available_only=False)]}
            if self.is_loaded(s):
                entry["runtime"] = self._loaded[s][1].info()
            stages[s] = entry
        stages["postcorrect"] = {"active": self.postcorrector.name, **self.postcorrector.info()}
        return {"manifest": str(self.registry.manifest_path), "manifest_found": self.registry.manifest_path.exists(),
                "device": self.device, "providers": ort_utils.providers(settings.ort_device), "stages": stages}
