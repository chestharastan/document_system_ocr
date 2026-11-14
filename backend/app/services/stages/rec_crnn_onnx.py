"""Line recognition with a CRNN-CTC model in ONNX (port of Khm-Dataset-Creation det/recognizer.py).

Contract: input  image  [N, 1, H, W] gray, (x/255 - 0.5)/0.5, H fixed (48 for crnn_m), W dynamic
          output logits [T, N, C]   (time-major; log-softmax or raw logits both work), class 0 = CTC blank
Vocab:    {"itos": ["", "ក", ...]} index -> character.
Crops are height-normalised (aspect kept), sorted by width and batched in buckets whose widths differ by at most 35 %,
padded by edge replication; greedy CTC decoding, per-line confidence = mean max-probability of the emitted symbols.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from app.services.stages.base import LineRecognizer
from app.services.stages.ort_utils import session

MIN_W, MAX_W = 16, 4200


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return img


class CrnnOnnxRecognizer(LineRecognizer):
    def __init__(self, spec, threads: int = 4, device: str = "auto", max_batch: int = 16):
        self.spec = spec
        self.sess = session(spec.path, threads, device)
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        h = inp.shape[2]
        self.height = int(h) if isinstance(h, int) else int((spec.input_size or [48])[0])
        with open(spec.vocab_path, encoding="utf-8") as f:
            v = json.load(f)
        self.itos = v["itos"] if isinstance(v, dict) else list(v)
        self.max_batch = max(1, int(max_batch))
        self.time_major = (spec.params or {}).get("output_layout", "TNC").upper() == "TNC"
        # CNN width stride, measured once so any architecture works (crnn_m: 4)
        probe = 256
        t = self._logits(np.zeros((1, 1, self.height, probe), np.float32))
        self.stride = max(1, round(probe / t.shape[0]))

    def info(self) -> dict:
        return {**super().info(), "height": self.height, "stride": self.stride, "vocab_size": len(self.itos)}

    def _logits(self, x):
        out = self.sess.run(None, {self.inp: x})[0]
        return out if self.time_major else out.transpose(1, 0, 2)  # -> [T, N, C]

    def _prep(self, img):
        g = to_gray(img)
        h, w = g.shape[:2]
        nw = int(min(MAX_W, max(MIN_W, round(w * self.height / max(1, h)))))
        return cv2.resize(g, (nw, self.height), interpolation=cv2.INTER_AREA if h > self.height else cv2.INTER_CUBIC)

    def _run(self, batch):
        s = self.stride
        w = (max(b.shape[1] for b in batch) + s - 1) // s * s
        x = np.empty((len(batch), 1, self.height, w), np.float32)
        for i, b in enumerate(batch):
            if w > b.shape[1]:
                b = np.pad(b, ((0, 0), (0, w - b.shape[1])), mode="edge")
            x[i, 0] = (b.astype(np.float32) / 255.0 - 0.5) / 0.5
        logits = self._logits(x)
        e = np.exp(logits - logits.max(-1, keepdims=True))
        prob = e / e.sum(-1, keepdims=True)
        ids, conf = prob.argmax(-1), prob.max(-1)
        out = []
        for i, b in enumerate(batch):
            seq, cs, pos, prev = [], [], [], -1
            T = min(ids.shape[0], max(1, b.shape[1] // s))
            for k in range(T):  # ignore the padded tail
                c = int(ids[k, i])
                if c != prev and c != 0:
                    seq.append(self.itos[c] if c < len(self.itos) else "")
                    cs.append(conf[k, i])
                    pos.append((k + 0.5) / T)  # where in the crop (0..1 of its width) the symbol was emitted
                prev = c
            out.append(("".join(seq), float(np.mean(cs)) if cs else 0.0, pos))
        return out

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        return [(t, c) for t, c, _ in self.recognize_ex(crops)]

    def recognize_ex(self, crops: list[np.ndarray]) -> list[tuple[str, float, list[float]]]:
        """(text, confidence, positions): one 0..1 fraction of the crop width per character of ``text`` (the CTC
        time step it was emitted at), for word-level work such as bold detection."""
        if not crops:
            return []
        prepped = [self._prep(c) for c in crops]
        order = sorted(range(len(prepped)), key=lambda i: prepped[i].shape[1])
        res: list = [None] * len(prepped)
        i = 0
        while i < len(order):
            j = i + 1
            w0 = prepped[order[i]].shape[1]
            while j < len(order) and j - i < self.max_batch and prepped[order[j]].shape[1] <= 1.35 * w0 + 16:
                j += 1
            for k, r in zip(order[i:j], self._run([prepped[k] for k in order[i:j]])):
                res[k] = r
            i = j
        return res
