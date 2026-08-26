"""Patch-based single-channel Transformer with co-attention fusion.

Faithful re-implementation of the architecture in
"Integrative stock price trend prediction via hierarchical LLM text processing
and patch-based transformer with co-attention" (Zhang et al., ESWA 2026),
Sections 3.3-3.5, parameterised for the Indian (NSE) market.

Tensor shapes follow the paper:
  B  batch, L  sequence length (30 trading days), d_p price-feature channels,
  N  patches, P patch length, D = d_model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def n_patches(L: int, P: int, S: int) -> int:
    """Eq. (4): N = floor((L - P) / S) + 2  (the +2 accounts for the replicated tail patch)."""
    return (L - P) // S + 2


class MHA(nn.Module):
    """Multi-head attention over (B, N, D) with separate q / kv sources."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dh).transpose(1, 2)  # B,h,N,dh

    def forward(self, q_src, kv_src):
        q, k, v = self._split(self.wq(q_src)), self._split(self.wk(kv_src)), self._split(self.wv(kv_src))
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)
        out = (self.drop(att) @ v).transpose(1, 2).flatten(2)
        return self.wo(out)


class EncoderLayer(nn.Module):
    """Standard pre-norm Transformer block used for the modality-specific encoders."""

    def __init__(self, d_model, d_ff, n_heads, dropout, activation="gelu"):
        super().__init__()
        self.attn = MHA(d_model, n_heads, dropout)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), act, nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h))
        return x + self.drop(self.ff(self.n2(x)))


class EmbeddingAlignment(nn.Module):
    """Section 3.3 - text/price dimension alignment, cross-attention alignment,
    patch partitioning + positional projection, and self-attention enhancement."""

    def __init__(self, cfg):
        super().__init__()
        L, P, S = cfg["seq_len"], cfg["patch_len"], cfg["stride"]
        D, dp = cfg["d_model"], cfg["n_price_feats"]
        # Alignment width. The paper hard-codes d_a = d_p = 17, forcing the 768-d text
        # embedding through a 45x linear squeeze. Decoupling it lets us widen the text
        # pathway (price is projected up to meet it) and measure what that squeeze costs.
        da = int(cfg.get("d_align", dp))
        self.L, self.P, self.S, self.dp, self.da = L, P, S, dp, da
        self.N = n_patches(L, P, S)

        # Eq. (1): map LLM embedding dim d -> alignment width d_a
        self.text_proj = nn.Sequential(nn.Linear(cfg["text_dim"], da), nn.Dropout(cfg["dropout"]))
        # d_a == d_p reproduces the paper exactly (no extra price parameters)
        self.price_proj = nn.Identity() if da == dp else nn.Linear(dp, da)
        # Eqs. (2)-(3): price queries attend over text keys/values
        self.wq, self.wk, self.wv = (nn.Linear(da, da) for _ in range(3))

        # Eq. (5): shared patch projection + learnable position embedding, applied
        # identically to both modalities so neither is privileged.
        self.patch_proj = nn.Linear(P, D)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.N, D))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc = lambda: nn.ModuleList(
            [EncoderLayer(D, cfg["d_ff"], cfg["n_heads"], cfg["dropout"], cfg["activation"])
             for _ in range(cfg["n_layers"])])
        self.enc_text, self.enc_price = enc(), enc()          # single-channel: shared across d_p
        self.drop = nn.Dropout(cfg["dropout"])

    def _patchify(self, x):
        """(B, L, d_a) -> (B, d_a, N, P) with overlapping windows and a replicated tail."""
        x = x.permute(0, 2, 1)                                # B, d_p, L
        x = F.pad(x, (0, self.S), mode="replicate")           # keeps the final partial window
        return x.unfold(dimension=-1, size=self.P, step=self.S)[:, :, : self.N, :]

    def _encode(self, patches, layers):
        B, dp, N, P = patches.shape
        x = self.patch_proj(patches) + self.pos               # B, d_p, N, D
        x = self.drop(x).reshape(B * dp, N, -1)               # Eq. (6): flatten d_p into batch
        for layer in layers:
            x = layer(x)                                      # Eq. (7)
        return x                                              # (B*d_p, N, D)

    def forward(self, x_price, x_text):
        """x_price: (B, L, d_p) z-scored price features. x_text: (B, L, d) daily LLM embeddings."""
        x_t = self.text_proj(x_text)                                  # Eq. (1)
        x_price = self.price_proj(x_price)                            # (B, L, d_a)
        q, k, v = self.wq(x_price), self.wk(x_t), self.wv(x_t)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)), dim=-1)
        x_tp = att @ v                                                # Eq. (3): aligned text stream
        return self._encode(self._patchify(x_tp), self.enc_text), \
               self._encode(self._patchify(x_price), self.enc_price)


class CoAttentionFusion(nn.Module):
    """Section 3.4 - symmetric dual-stream (text->price, price->text) fusion."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.s1 = MHA(d_model, n_heads, dropout)   # text queries price
        self.s2 = MHA(d_model, n_heads, dropout)   # price queries text
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, xt, xp):
        o1 = self.s1(xt, xp)                                   # Eqs. (8)-(9)
        o2 = self.s2(xp, xt)                                   # Eqs. (10)-(11)
        fused = self.proj(torch.cat([o1, o2], dim=-1))
        return self.norm(self.drop(fused) + 0.5 * (xt + xp))   # residual + LayerNorm


class ConcatFusion(nn.Module):
    """Ablation: replace co-attention with plain concatenation + a linear projection.
    Same parameter budget in the projection, but the two streams never interact
    element-wise. This is the control for the paper's central architectural claim."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, xt, xp):
        return self.norm(self.drop(self.proj(torch.cat([xt, xp], dim=-1))) + 0.5 * (xt + xp))


class OneWayFusion(nn.Module):
    """Ablation: keep cross-attention but drop the symmetry - price queries text only
    (Eqs. 10-11 without 8-9). Isolates whether bidirectionality is what earns the gain."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.s2 = MHA(d_model, n_heads, dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, xt, xp):
        return self.norm(self.drop(self.proj(self.s2(xp, xt))) + 0.5 * (xt + xp))


FUSIONS = {"coattention": CoAttentionFusion, "concat": ConcatFusion, "oneway": OneWayFusion}


class StockTrendModel(nn.Module):
    """Full framework: alignment/patching -> co-attention -> classification (Section 3.5)."""

    def __init__(self, cfg):
        super().__init__()
        self.align = EmbeddingAlignment(cfg)
        fusion = cfg.get("fusion", "coattention")            # spec 05 ablation axis
        self.fusion = FUSIONS[fusion](cfg["d_model"], cfg["n_heads"], cfg["dropout"])
        self.da, self.N, self.D = self.align.da, self.align.N, cfg["d_model"]
        # Eq. (15): flatten (patch x channel) and score one logit per sample
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(cfg["dropout"]),
                                  nn.Linear(self.da * self.N * self.D, 1))

    def forward(self, x_price, x_text):
        xt, xp = self.align(x_price, x_text)
        z = self.fusion(xt, xp)                                 # (B*d_a, N, D)
        z = z.view(-1, self.da, self.N, self.D)
        return self.head(z).squeeze(-1)                         # logits; BCEWithLogits in train.py
