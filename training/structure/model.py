"""Structure model: a small Transformer over the lines of a page (~0.6 M params, ONNX-exportable, phone-sized).

Inputs  feats [B,N,F] float, chars [B,N,6] int, pad mask [B,N] bool (True = pad)
Outputs role [B,N,R], first [B,N,2], level [B,N,4], bullet [B,N,2], succ [B,N,N+1] (pointer to the next line in
reading order; index N = end of page)."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.structure.data import CHAR_BUCKETS, MAX_LEVEL, N_CHAR_IDS, ROLES, n_feat

N_PAIR = 8  # pairwise geometry for the successor pointer (v2)


def pair_geometry(feats):
    """feats [B,N,F] (x0,y0,x1,y1 normalised first) -> [B,N,N,N_PAIR]: where line j sits relative to line i
    (dy top-to-bottom, dx of left edges, horizontal overlap, same row, j right of i, j below i). Scaled by the median
    line height so it is resolution / crop invariant."""
    x0, y0, x1, y1 = feats[..., 0], feats[..., 1], feats[..., 2], feats[..., 3]
    # median line height (page-normalised) = h / h_rel for every real line; padding rows (all zero) give 0 -> max
    mh = ((y1 - y0) / feats[..., 5].clamp(min=1e-3)).amax(dim=1, keepdim=True).clamp(min=1e-4)  # [B,1]
    cy = (y0 + y1) / 2
    dy = (cy[:, None, :] - cy[:, :, None]) / mh[:, :, None]                       # j below i: > 0
    dx0 = (x0[:, None, :] - x0[:, :, None]) / mh[:, :, None]
    gap_x = (x0[:, None, :] - x1[:, :, None]) / mh[:, :, None]                    # j starts right of i's end
    ov = (torch.minimum(x1[:, :, None], x1[:, None, :]) - torch.maximum(x0[:, :, None], x0[:, None, :])) / mh[:, :, None]
    gap_y = (y0[:, None, :] - y1[:, :, None]) / mh[:, :, None]
    same_row = (dy.abs() < 0.5).float()
    return torch.stack([dy.clamp(-20, 20) / 5, dx0.clamp(-40, 40) / 10, gap_x.clamp(-40, 40) / 10, ov.clamp(-40, 40) / 10,
                        gap_y.clamp(-20, 20) / 5, same_row, (dy > 0.5).float(), (gap_x > -0.5).float() * same_row], -1)


class StructureNet(nn.Module):
    def __init__(self, d: int = 160, layers: int = 4, heads: int = 4, ff: int = 320, char_dim: int = 24, dropout: float = 0.1,
                 feat_version: int = 1, pair: bool = False, tables: bool = False, pair_dim: int = 48):
        super().__init__()
        self.feat_version, self.use_pair, self.tables = feat_version, pair, tables
        if tables:  # v3 heads: page kind, line-in-table, pairwise same row / column / cell (docs/TABLES_CONTRACT.md)
            self.kind_head = nn.Linear(d, 3)
            self.intab = nn.Linear(d, 2)
            self.pq = nn.Linear(d, 3 * pair_dim)
            self.pk = nn.Linear(d, 3 * pair_dim)
            self.pair_tab = nn.Sequential(nn.Linear(N_PAIR, 32), nn.GELU(), nn.Linear(32, 3))
        self.feat_proj = nn.Sequential(nn.Linear(n_feat(feat_version), d), nn.GELU(), nn.Linear(d, d))
        self.char_emb = nn.Embedding(CHAR_BUCKETS, char_dim, padding_idx=0)
        self.char_proj = nn.Linear(char_dim * N_CHAR_IDS, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.role = nn.Linear(d, len(ROLES))
        self.first = nn.Linear(d, 2)
        self.level = nn.Linear(d, MAX_LEVEL + 1)
        self.bullet = nn.Linear(d, 2)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.eos = nn.Parameter(torch.zeros(d))
        self.scale = d ** -0.5
        if pair:
            self.pair_mlp = nn.Sequential(nn.Linear(N_PAIR, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, feats, chars, pad_mask=None):
        B, N, _ = feats.shape
        x = self.feat_proj(feats) + self.char_proj(self.char_emb(chars).reshape(B, N, -1))
        h = self.norm(self.encoder(x, src_key_padding_mask=pad_mask))
        q, k = self.q(h), self.k(h)
        k = torch.cat([k, self.eos.expand(B, 1, -1)], dim=1)            # [B,N+1,d]
        succ = torch.einsum("bnd,bmd->bnm", q, k) * self.scale            # [B,N,N+1]
        if self.use_pair:
            pb = self.pair_mlp(pair_geometry(feats)).squeeze(-1)            # [B,N,N]
            succ = succ + torch.cat([pb, torch.zeros(B, N, 1, device=feats.device, dtype=pb.dtype)], dim=2)
        ar = torch.arange(N, device=feats.device)
        eye = (ar[:, None] == ar[None, :]).unsqueeze(0)  # (Equal exports to ONNX; EyeLike does not run in onnxruntime)
        succ = succ.masked_fill(torch.cat([eye, torch.zeros(1, N, 1, dtype=torch.bool, device=feats.device)], dim=2), -1e4)  # not its own successor
        if pad_mask is not None:
            succ = succ.masked_fill(torch.cat([pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=feats.device)], dim=1).unsqueeze(1), -1e4)
        if not self.tables:
            return self.role(h), self.first(h), self.level(h), self.bullet(h), succ
        kind, intab, pair = self.table_heads(h, feats, pad_mask)
        return self.role(h), self.first(h), self.level(h), self.bullet(h), succ, kind, intab, pair

    def table_heads(self, h, feats, pad_mask=None):
        """-> kind [B,3] (masked mean pooling), intab [B,N,2], pair [B,N,N,3] symmetric logits (same row / col / cell)."""
        B, N, _ = h.shape
        m = (~pad_mask).to(h.dtype).unsqueeze(-1) if pad_mask is not None else torch.ones(B, N, 1, device=h.device, dtype=h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        pq = self.pq(h).reshape(B, N, 3, -1)
        pk = self.pk(h).reshape(B, N, 3, -1)
        pair = torch.einsum("bnrd,bmrd->bnmr", pq, pk) * (pq.shape[-1] ** -0.5) + self.pair_tab(pair_geometry(feats))
        pair = 0.5 * (pair + pair.transpose(1, 2))
        return self.kind_head(pooled), self.intab(h), pair


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def decode_order(succ_logits, boxes=None) -> list[int]:
    """Greedy path decoding of the successor matrix [N,N+1] (numpy) -> reading order (list of indices).
    Starts at the line that is least often chosen as a successor (top-most on ties); follows argmax among unused
    lines; on end-of-page or exhaustion restarts from the unused line that is least pointed to."""
    import numpy as np
    N = succ_logits.shape[0]
    P = succ_logits - succ_logits.max(1, keepdims=True)
    P = np.exp(P); P /= P.sum(1, keepdims=True)
    incoming = P[:, :N].sum(0)
    used = np.zeros(N, bool)
    order: list[int] = []

    def pick_start():
        cand = np.where(~used)[0]
        key = incoming[cand] + (boxes[cand, 1] / 1e6 if boxes is not None else cand / 1e6)
        return int(cand[np.argmin(key)])

    while len(order) < N:
        cur = pick_start()
        used[cur] = True; order.append(cur)
        while True:
            row = P[cur].copy(); row[:N][used] = -1
            nxt = int(np.argmax(row))
            if nxt == N or row[nxt] <= 0:
                break
            used[nxt] = True; order.append(nxt); cur = nxt
            if len(order) == N:
                break
    return order
