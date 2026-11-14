#!/usr/bin/env python3
"""Add or update one model entry in the manifest (.model/v2/models.json), computing sha256 and reading ONNX metadata.

    cd backend
    python scripts/register_model.py --stage layout  --kind yolo_onnx --id layout_v2_21cls --file layout/layout_v2.onnx \
        --version v2 --label "YOLO11s 21-class v2" --default
    python scripts/register_model.py --stage textdet --kind db_onnx   --id db_v2 --file textdet/db_v2.onnx
    python scripts/register_model.py --stage rec     --kind crnn_onnx --id crnn_v2 --file rec/rec_v2.onnx --vocab rec/vocab_v2.json

--file / --vocab are relative to the manifest folder (or absolute). For yolo_onnx the class names are copied from the
ONNX metadata ``names``; input_size from the ONNX input shape. Re-running with the same --id replaces the entry.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings  # noqa: E402
from app.services.model_registry import KINDS, sha256_file  # noqa: E402


def onnx_meta(path: Path) -> dict:
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    meta = s.get_modelmeta().custom_metadata_map
    out = {"input_size": [d if isinstance(d, int) else None for d in inp.shape[2:]]}
    if "names" in meta:
        d = ast.literal_eval(meta["names"])
        out["classes"] = [d[k] for k in sorted(d)] if isinstance(d, dict) else list(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=settings.manifest_path)
    ap.add_argument("--id", required=True)
    ap.add_argument("--stage", required=True, choices=["layout", "textdet", "rec", "font", "structure"])
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--file", required=True)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--version", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--params", default=None, help='JSON, e.g. \'{"unclip_ratio": 1.6}\'')
    ap.add_argument("--default", action="store_true", help="make it the stage default")
    a = ap.parse_args()

    base = a.manifest.parent
    base.mkdir(parents=True, exist_ok=True)
    data = json.loads(a.manifest.read_text(encoding="utf-8")) if a.manifest.exists() else {"schema": 2, "defaults": {}, "models": []}
    fpath = Path(a.file) if Path(a.file).is_absolute() else base / a.file
    if not fpath.exists():
        sys.exit(f"file not found: {fpath}")
    entry = {"id": a.id, "stage": a.stage, "kind": a.kind, "file": a.file, "version": a.version,
             "label": a.label or a.id, "sha256": sha256_file(fpath)}
    if a.vocab:
        entry["vocab"] = a.vocab
    if fpath.suffix == ".onnx":
        entry.update(onnx_meta(fpath))
    if a.params:
        entry["params"] = json.loads(a.params)
    data["models"] = [m for m in data.get("models", []) if m["id"] != a.id] + [entry]
    if a.default or a.stage not in data.setdefault("defaults", {}):
        data["defaults"][a.stage] = a.id
    a.manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{a.manifest}: {a.stage}/{a.id} sha256={entry['sha256'][:12]}... default={data['defaults'][a.stage]}")


if __name__ == "__main__":
    main()
