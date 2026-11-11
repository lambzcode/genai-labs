"""Working versions of every stage of the Session 1 lab.

Use them to unblock yourself, never as a first step. In the notebook:

    from reference import CausalAttentionHead
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device):
    """Boolean (T, T) matrix: True where position i may look at position j (j <= i)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


class CausalAttentionHead(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.D = D
        self.query = nn.Linear(C, D, bias=False)
        self.key = nn.Linear(C, D, bias=False)
        self.value = nn.Linear(C, D, bias=False)

    def forward(self, x):
        # x: (B, T, C) -> (B, T, D)
        T = x.size(1)
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.D)  # (B, T, T)
        scores = scores.masked_fill(~causal_mask(T, x.device), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        assert C % H == 0
        self.heads = nn.ModuleList([CausalAttentionHead(C, C // H) for _ in range(H)])
        self.out_proj = nn.Linear(C, C, bias=True)

    def forward(self, x):
        joined = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, C)
        return self.out_proj(joined)


class MLP(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = MultiHeadCausalAttention(C, H)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP(C)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, V, C, H, max_t):
        super().__init__()
        self.max_t = max_t
        self.token_embedding = nn.Embedding(V, C)
        self.position_embedding = nn.Embedding(max_t, C)
        self.block = DecoderBlock(C, H)
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V, bias=True)

    def forward(self, token_ids, targets=None):
        # token_ids: (B, T), T <= max_t
        T = token_ids.size(1)
        assert T <= self.max_t, f"Sequence length {T} exceeds max_t={self.max_t}"
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        logits = self.lm_head(self.ln_f(self.block(x)))  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
