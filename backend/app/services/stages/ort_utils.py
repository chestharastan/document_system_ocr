"""onnxruntime session creation shared by all ONNX stages (CPU by default, CUDA when available and allowed)."""

from __future__ import annotations

import onnxruntime as ort


def providers(device: str = "auto") -> list[str]:
    avail = ort.get_available_providers()
    device = (device or "auto").lower()
    if device in ("auto", "cuda", "gpu") and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def device_name(device: str = "auto") -> str:
    return "cuda" if providers(device)[0] == "CUDAExecutionProvider" else "cpu"


def session(path, threads: int = 4, device: str = "auto") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(threads))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provs = providers(device)
    try:
        sess = ort.InferenceSession(str(path), so, providers=provs)
        if provs[0] != "CPUExecutionProvider":  # a CUDA EP without cuDNN only fails at run time: probe it
            _probe(sess)
        return sess
    except Exception:
        if provs[0] == "CPUExecutionProvider":
            raise
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])  # CUDA EP present but unusable


def _probe(sess) -> None:
    import numpy as np
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else (1 if k == 0 else 32) for k, d in enumerate(i.shape)]
        dt = np.float32 if "float" in i.type else np.int64
        feeds[i.name] = np.zeros(shape, dt)
    sess.run(None, feeds)
