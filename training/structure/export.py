"""Export best.pt -> ONNX (one page per call: feats [1,N,F], chars [1,N,6]; dynamic N) with the label lists in metadata.

    python -m training.structure.export --run /home/hello/ocr/docsys/work/runs/struct_v1 --out .model/v2/structure/struct_v1.onnx

Handles every model version: feature set v1 / v2 (metadata ``feat_version`` + ``features``), the pairwise successor
bias (``pair``) and the v3 table heads (``tables``: extra outputs kind [1,3], intab [1,N,2], pair [1,N,N,3] =
probabilities of same row / column / cell). The encoder layers are re-implemented with explicit attention for the
export: PyTorch's fused attention bakes the trace length into the ONNX reshapes, so the graph would only run for pages
with exactly that many lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

from training.structure.data import FEATURES, FEATURES_V2, KINDS, MAX_LEVEL, N_CHAR_IDS, ROLES, n_feat
from training.structure.model import StructureNet, pair_geometry


class ExportLayer(nn.Module):
    """nn.TransformerEncoderLayer (norm_first, gelu, no dropout at inference) with explicit multi-head attention."""

    def __init__(self, layer: nn.TransformerEncoderLayer, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_w, self.in_proj_b = layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
        self.out_proj, self.linear1, self.linear2 = layer.self_attn.out_proj, layer.linear1, layer.linear2
        self.norm1, self.norm2 = layer.norm1, layer.norm2

    def forward(self, x):
        B, D = x.shape[0], x.shape[2]
        h = self.norm1(x)
        qkv = torch.nn.functional.linear(h, self.in_proj_w, self.in_proj_b)
        q, k, v = qkv.split(D, dim=-1)
        hd = D // self.heads
        q = q.reshape(B, -1, self.heads, hd).transpose(1, 2)
        k = k.reshape(B, -1, self.heads, hd).transpose(1, 2)
        v = v.reshape(B, -1, self.heads, hd).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, -1, D)
        x = x + self.out_proj(o)
        x = x + self.linear2(torch.nn.functional.gelu(self.linear1(self.norm2(x))))
        return x


class Exportable(nn.Module):
    def __init__(self, m: StructureNet):
        super().__init__()
        self.m = m
        heads = m.encoder.layers[0].self_attn.num_heads
        self.layers = nn.ModuleList([ExportLayer(l, heads) for l in m.encoder.layers])

    def forward(self, feats, chars):
        m = self.m
        B = feats.shape[0]
        N = feats.shape[1]
        x = m.feat_proj(feats) + m.char_proj(m.char_emb(chars).reshape(B, -1, m.char_emb.embedding_dim * chars.shape[2]))
        for layer in self.layers:
            x = layer(x)
        h = m.norm(x)
        q, k = m.q(h), m.k(h)
        k = torch.cat([k, m.eos.expand(B, 1, -1)], dim=1)
        succ = torch.matmul(q, k.transpose(-1, -2)) * m.scale
        if m.use_pair:
            pb = m.pair_mlp(pair_geometry(feats)).squeeze(-1)
            succ = succ + torch.cat([pb, torch.zeros_like(pb[:, :, :1])], dim=2)
        ar = torch.arange(N, device=feats.device)
        eye = (ar[:, None] == ar[None, :]).unsqueeze(0)
        succ = succ.masked_fill(torch.cat([eye, torch.zeros_like(eye[:, :, :1])], dim=2), -1e4)
        outs = (torch.softmax(m.role(h), -1), torch.softmax(m.first(h), -1), torch.softmax(m.level(h), -1), torch.softmax(m.bullet(h), -1), succ)
        if m.tables:
            kind, intab, pair = m.table_heads(h, feats, None)
            outs = outs + (torch.softmax(kind, -1), torch.softmax(intab, -1), torch.sigmoid(pair))
        return outs


def build_from_ckpt(ck: dict) -> StructureNet:
    m = StructureNet(d=ck["d"], layers=ck["layers"], heads=ck.get("heads", 4), ff=ck.get("ff", 2 * ck["d"]),
                     feat_version=ck.get("feat_version", 1), pair=ck.get("pair", False), tables=ck.get("tables", False))
    m.load_state_dict(ck["model"]); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    ck = torch.load(Path(a.run) / a.ckpt, map_location="cpu", weights_only=False)
    m = build_from_ckpt(ck)
    fv, F_ = m.feat_version, n_feat(m.feat_version)
    w = Exportable(m).eval()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    names = ["role", "first", "level", "bullet", "succ"] + (["kind", "intab", "pair"] if m.tables else [])
    dyn = {"feats": {1: "n"}, "chars": {1: "n"}, "role": {1: "n"}, "first": {1: "n"}, "level": {1: "n"}, "bullet": {1: "n"}, "succ": {1: "n", 2: "n1"}}
    if m.tables:
        dyn.update({"intab": {1: "n"}, "pair": {1: "n", 2: "n2"}})
    feats = torch.rand(1, 7, F_); chars = torch.zeros(1, 7, N_CHAR_IDS, dtype=torch.long)
    torch.onnx.export(w, (feats, chars), str(out), input_names=["feats", "chars"], output_names=names, dynamic_axes=dyn,
                      opset_version=17, dynamo=False)
    mo = onnx.load(str(out))
    meta = {"roles": json.dumps(ROLES), "features": json.dumps(FEATURES_V2 if fv >= 2 else FEATURES), "feat_version": str(fv),
            "max_level": str(MAX_LEVEL), "char_ids": str(N_CHAR_IDS), "pair": str(int(m.use_pair)), "tables": str(int(m.tables)),
            "kinds": json.dumps(KINDS), "val": json.dumps(ck.get("val", {}))}
    for k, v in meta.items():
        e = mo.metadata_props.add(); e.key, e.value = k, v
    onnx.save(mo, str(out))
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    errs = []
    for n in (23, 61):
        f = np.random.rand(1, n, F_).astype(np.float32); c = np.random.randint(0, 2048, (1, n, N_CHAR_IDS)).astype(np.int64)
        with torch.no_grad():
            ref = [t.detach().numpy() for t in m(torch.from_numpy(f), torch.from_numpy(c), None)]
        got = s.run(None, {"feats": f, "chars": c})
        errs.append(float(np.abs(ref[4] - got[4]).max()))  # successor logits of the trained module vs ONNX
        errs.append(float(np.abs(torch.softmax(torch.from_numpy(ref[0]), -1).numpy() - got[0]).max()))
        if m.tables:
            errs.append(float(np.abs(torch.sigmoid(torch.from_numpy(ref[7])).numpy() - got[7]).max()))
    print(f"exported {out} ({out.stat().st_size / 1e6:.2f} MB), outputs {names}, max |torch-onnx| over N=23,61 = {max(errs):.2e}")


if __name__ == "__main__":
    main()
