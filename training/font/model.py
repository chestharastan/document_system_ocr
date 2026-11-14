"""Tiny CNN font classifier (~0.45 M params): 32 x 256 gray line crop -> family, weight, slant, size.

v2 adds the size head: log(em ratio) = log(font px / crop height), a regression that gives each line its CSS
font size from the crop alone (see training/font/gen.py).

Phone-sized: 8 conv layers, no attention, static shapes; exports to ONNX (opset 17) for onnxruntime / ORT mobile."""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin, cout, pool):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class FontNet(nn.Module):
    def __init__(self, n_family: int, n_weight: int = 2, n_slant: int = 2, width: int = 24):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(1, w, 2),          # 16 x 128
            _block(w, 2 * w, 2),      # 8 x 64
            _block(2 * w, 4 * w, 2),  # 4 x 32
            _block(4 * w, 6 * w, None),
        )
        feat = 6 * w * 2  # avg + max pooled
        self.drop = nn.Dropout(0.2)
        self.family = nn.Linear(feat, n_family)
        self.weight = nn.Linear(feat, n_weight)
        self.slant = nn.Linear(feat, n_slant)
        self.size = nn.Linear(feat, 1)  # log(em ratio)

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        f = self.drop(f)
        return self.family(f), self.weight(f), self.slant(f), self.size(f).squeeze(1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------------------------- v3
class _Res(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + (x if self.sc is None else self.sc(x)))


class FontNetV3(nn.Module):
    """font_v3_best: residual CNN (40 x 320 -> 3 x 40 feature columns) + a small transformer over the width, so the
    whole visible line votes for the typeface / size; mean + max + attention pooling -> family, weight, slant,
    log(em ratio). Static shapes, ONNX opset 17."""

    def __init__(self, n_family: int, n_weight: int = 2, n_slant: int = 2, width: int = 48, depth: int = 2, in_w: int = 320):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.body = nn.Sequential(
            _Res(w, w), nn.MaxPool2d(2),                       # 20 x 160
            _Res(w, 2 * w), _Res(2 * w, 2 * w), nn.MaxPool2d(2),  # 10 x 80
            _Res(2 * w, 3 * w), _Res(3 * w, 3 * w), nn.MaxPool2d(2),  # 5 x 40
            _Res(3 * w, 4 * w), _Res(4 * w, 4 * w),
        )
        d = 4 * w
        self.hpool = nn.Conv2d(d, d, (5, 1), groups=d, bias=False)  # learned collapse of the 5 rows (keeps the vertical profile)
        self.pos = nn.Parameter(torch.zeros(1, in_w // 8, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.att = nn.Linear(d, 1)
        feat = 3 * d
        self.drop = nn.Dropout(0.2)
        self.family = nn.Linear(feat, n_family)
        self.weight = nn.Linear(feat, n_weight)
        self.slant = nn.Linear(feat, n_slant)
        self.size = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x):
        f = self.body(self.stem(x))
        g = f.amax(dim=2)                                   # [N, d, T]
        f = self.hpool(f).squeeze(2)                        # [N, d, T]
        t = (f + g).transpose(1, 2) + self.pos              # [N, T, d]
        t = self.tr(t)
        a = torch.softmax(self.att(t), dim=1)               # [N, T, 1]
        z = torch.cat([t.mean(1), t.amax(1), (a * t).sum(1)], 1)
        z = self.drop(z)
        return self.family(z), self.weight(z), self.slant(z), self.size(z).squeeze(1)


def build(arch: str, n_family: int, width: int | None = None):
    """arch: "cnn" (v1/v2/v3_small FontNet, 32 x 256) | "v3best" (FontNetV3, 40 x 320). -> (model, (H, W))"""
    if arch == "v3best":
        return FontNetV3(n_family, width=width or 48), (40, 320)
    return FontNet(n_family, width=width or 24), (32, 256)
