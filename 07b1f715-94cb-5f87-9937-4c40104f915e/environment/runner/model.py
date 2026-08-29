"""BIA S02 frozen architecture. Harness-owned. Not a free axis.

Decoder-only transformer with RMSNorm, GEGLU-free GELU MLP, learned position
embeddings, and an untied zero-initialized output head. Every shape, every
initialization distribution and the parameter-group partition below are frozen:
the verifier reads the recorded architecture signature out of the run telemetry
and compares it against the frozen constants, so a modified copy of this file
scores zero rather than scoring differently.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 50304


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, 1e-6)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc_out = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        h = self.norm2(x)
        x = x + self.fc_out(F.gelu(self.fc(h)))
        return x


class FrozenGPT(nn.Module):
    def __init__(self, n_layer: int, n_head: int, d_model: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.wte = nn.Embedding(VOCAB_SIZE, d_model)
        self.wpe = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.head(x).float()
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.reshape(-1))


def frozen_init(model: nn.Module, seed: int) -> None:
    """Frozen initialization. Seed selects the draw and nothing else."""
    g = torch.Generator(device="cpu").manual_seed(1000003 * seed + 17)
    for name, p in model.named_parameters():
        with torch.no_grad():
            if name.endswith("head.weight"):
                p.zero_()
            elif p.ndim == 2:
                fan_in = p.shape[1]
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / math.sqrt(fan_in)))
            elif p.ndim == 1:
                p.fill_(1.0)


def group_of(name: str) -> str:
    """Frozen parameter-group partition. The schedule multiplies by this name."""
    if name.startswith("wte") or name.startswith("wpe"):
        return "embed"
    if name.startswith("head"):
        return "head"
    if name.endswith("weight") and ("qkv" in name or "proj" in name or "fc" in name):
        return "hidden_matrix"
    return "vector"


def build_param_groups(model: nn.Module):
    """Return the frozen group list the runner hands to build_update_rule."""
    buckets = {"embed": [], "hidden_matrix": [], "head": [], "vector": []}
    for name, p in model.named_parameters():
        buckets[group_of(name)].append(p)
    return [{"name": k, "params": v, "lr": 0.0} for k, v in buckets.items() if v]


def architecture_signature(n_layer: int, n_head: int, d_model: int, seq_len: int) -> dict:
    return {
        "n_layer": n_layer,
        "n_head": n_head,
        "d_model": d_model,
        "seq_len": seq_len,
        "vocab_size": VOCAB_SIZE,
        "norm": "rmsnorm",
        "mlp": "gelu_4x",
        "pos": "learned",
        "head_tied": False,
    }
