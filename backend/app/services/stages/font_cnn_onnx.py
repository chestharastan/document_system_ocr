"""Font classifier (training/font): one gray H x W window per line crop -> font family, weight, slant (+ size).

Contract: input "image" [N, 1, 32, 256] (x/255 - 0.5)/0.5; outputs softmax "family" [N, F], "weight" [N, 2],
"slant" [N, 2] and, from v2, "size" [N, 1] = em ratio (CSS font px = size x crop height in page pixels); the label
lists live in the ONNX metadata (families / weights / slants, JSON). The window size is the model's static input
shape: 32 x 256 for v1 / v2 / v3_small, 40 x 320 for v3_best. A line wider than W at height H is cut into up to 3
windows whose probabilities (and em ratios) are averaged."""

from __future__ import annotations

import json

import cv2
import numpy as np

from app.services.stages.base import FontClassifier
from app.services.stages.ort_utils import session

H, W = 32, 256


def windows(crop: np.ndarray, H: int = H, W: int = W) -> list[np.ndarray]:
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    nw = max(8, int(round(w * H / max(1, h))))
    g = cv2.resize(g, (nw, H), interpolation=cv2.INTER_AREA if h > H else cv2.INTER_CUBIC)
    if nw <= W:
        fill = int(np.median(np.concatenate([g[:, :2].ravel(), g[:, -2:].ravel()])))
        pad = W - nw
        return [cv2.copyMakeBorder(g, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=fill)]
    n = min(3, int(np.ceil(nw / W)))
    return [g[:, x:x + W] for x in (int(round(k * (nw - W) / max(1, n - 1))) if n > 1 else 0 for k in range(n))]


class FontOnnxClassifier(FontClassifier):
    def __init__(self, spec, threads: int = 4, device: str = "auto", max_batch: int = 64):
        self.spec = spec
        self.sess = session(spec.path, threads, device)
        self.inp = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.families = json.loads(meta.get("families", "[]"))
        self.weights = json.loads(meta.get("weights", '["regular", "bold"]'))
        self.slants = json.loads(meta.get("slants", '["upright", "italic"]'))
        self.max_batch = max_batch
        self.has_size = any(o.name == "size" for o in self.sess.get_outputs())
        shp = self.sess.get_inputs()[0].shape  # [N, 1, H, W]
        self.H = int(shp[2]) if isinstance(shp[2], int) and shp[2] > 0 else H
        self.W = int(shp[3]) if isinstance(shp[3], int) and shp[3] > 0 else W
        if not self.families:
            raise ValueError(f"font model {spec.path} has no 'families' metadata")

    def info(self) -> dict:
        return {**super().info(), "families": len(self.families), "input": [self.H, self.W], "size_head": self.has_size}

    def _run(self, x: np.ndarray):
        outs = [[], [], [], []]
        for i in range(0, len(x), self.max_batch):
            for k, o in enumerate(self.sess.run(None, {self.inp: x[i:i + self.max_batch]})):
                outs[k].append(o)
        return [np.concatenate(o) if o else None for o in outs]

    def classify(self, crops: list[np.ndarray]) -> list[dict]:
        if not crops:
            return []
        wins, owners = [], []
        for c in crops:
            ws = windows(c, self.H, self.W)
            owners.append(list(range(len(wins), len(wins) + len(ws))))
            wins += ws
        x = (np.stack(wins).astype(np.float32) / 255.0 - 0.5) / 0.5
        pf, pw, ps, pr = self._run(x[:, None])
        out = []
        for crop, idx in zip(crops, owners):
            f, w, s = pf[idx].mean(0), pw[idx].mean(0), ps[idx].mean(0)
            fi = int(f.argmax())
            res = {"family": self.families[fi], "confidence": round(float(f[fi]), 4),
                   "weight": self.weights[int(w.argmax())], "weight_confidence": round(float(w.max()), 4),
                   "slant": self.slants[int(s.argmax())], "slant_confidence": round(float(s.max()), 4),
                   "probs": f}
            if pr is not None:  # v2: em ratio -> CSS font size of this line in page pixels
                ratio = float(pr[idx].mean())
                res["em_ratio"] = round(ratio, 4)
                res["em_px"] = round(ratio * crop.shape[0], 2)
            out.append(res)
        return out
