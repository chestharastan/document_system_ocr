"""Inline Khmer OCR recognizer — ResNet BiLSTM CTC, no external dependencies."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


IMG_H, IMG_W = 64, 512  # must match training dimensions

_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# ── Model ─────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetBiLSTMCTC(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),       # 0
            nn.BatchNorm2d(64),                                 # 1
            nn.ReLU(inplace=True),                              # 2
            nn.MaxPool2d(2, 2),                                 # 3
            nn.Sequential(ResBlock(64, 64, stride=1)),          # 4
            nn.Sequential(ResBlock(64, 128, stride=2)),         # 5
            nn.Sequential(ResBlock(128, 256, stride=1)),        # 6
        )
        self.rnn = nn.LSTM(
            input_size=256 * 16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                       # [B, 256, H', W']
        b, c, h, w = feat.shape
        seq = feat.permute(3, 0, 1, 2)          # [W', B, C, H']
        seq = seq.reshape(w, b, c * h)           # [W', B, C*H']
        out, _ = self.rnn(seq)                   # [W', B, 2*hidden]
        return self.classifier(out).log_softmax(2)  # [W', B, num_classes]



# ── Charset ───────────────────────────────────────────────────────────────────

def load_charset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {i: ch for i, ch in enumerate(data)}
    if isinstance(data, dict):
        known = {"khmer", "latin", "digits", "special"}
        if known & set(data.keys()):
            order = ["khmer", "latin", "digits", "special"]
            all_chars = "".join(data.get(cat, "") for cat in order)
            return {0: "<blank>", **{i + 1: ch for i, ch in enumerate(all_chars)}}
        try:
            return {int(k): v for k, v in data.items()}
        except ValueError:
            pass
        return {v: k for k, v in data.items()}
    raise ValueError(f"Unrecognised charset format in {path}")


# ── CTC decode ────────────────────────────────────────────────────────────────

def ctc_decode(log_probs: torch.Tensor, idx2char: dict, blank: int = 0) -> str:
    indices = log_probs.argmax(dim=1).tolist()
    chars, prev = [], None
    for idx in indices:
        if idx != prev and idx != blank:
            chars.append(idx2char.get(idx, "?"))
        prev = idx
    return "".join(chars)


# ── Recognizer class (cache-friendly, one instance per model dir) ─────────────

class KhmerRecognizer:
    """Loads and caches a ResNetBiLSTMCTC checkpoint for line recognition."""

    def __init__(self, model_dir: str, charset_path: str):
        model_dir = Path(model_dir)
        ckpt = model_dir / "best_model.pth"
        if not ckpt.exists():
            ckpt = model_dir / "last_model.pth"

        self.idx2char = load_charset(charset_path)
        num_classes = max(self.idx2char.keys()) + 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResNetBiLSTMCTC(num_classes=num_classes)
        state_dict = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def recognize(self, pil_image: Image.Image) -> str:
        """Recognize a single line crop (PIL image) -> text string."""
        img = pil_image.convert("L").resize((IMG_W, IMG_H), Image.BICUBIC)
        x = _TRANSFORM(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.model(x)[:, 0, :]   # [T, C]
        return ctc_decode(log_probs.cpu(), self.idx2char)