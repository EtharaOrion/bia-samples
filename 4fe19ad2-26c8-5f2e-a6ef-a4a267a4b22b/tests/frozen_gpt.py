"""The frozen architecture.

Layer count, model dimension, head dimension, sequence length and vocabulary are
read from shape.json and are frozen axes. Nothing in a recipe reaches them. What
a recipe does reach is the initialization, which is why `apply_init` takes the
scheme and the scale from the recipe and nothing else does.


VERIFIER-OWNED COPY. This is the verifier's own copy of the substrate module of
the same name under environment/. The two carry identical arithmetic on purpose,
so the agent explores the optimizer and the architecture the verifier actually
runs. They are separate FILES on purpose too: a verifier that imported the
agent-visible tree would be grading bytes the agent can reach.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        assert dim % head_dim == 0, "model_dim must be a whole number of heads"
        self.n_head = dim // head_dim
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = (item.transpose(1, 2) for item in qkv.unbind(dim=2))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(batch, length, dim))


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, head_dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.out = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class FrozenGPT(nn.Module):
    def __init__(self, shape: dict):
        super().__init__()
        dim = int(shape["model_dim"])
        self.seq_len = int(shape["seq_len"])
        self.wte = nn.Embedding(int(shape["vocab_size"]), dim)
        self.wpe = nn.Embedding(self.seq_len, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, int(shape["head_dim"])) for _ in range(int(shape["num_layers"]))]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, int(shape["vocab_size"]), bias=False)

    def forward(self, idx, targets):
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def apply_init(model: nn.Module, scheme: str, scale: float, seed: int) -> None:
    """Initialization is a free axis, so it is applied from the recipe and seed alone."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weight = module.weight
            fan_in = weight.shape[-1]
            base = 1.0 / math.sqrt(fan_in)
            if scheme == "orthogonal":
                flat = torch.empty(weight.shape, device="cpu")
                nn.init.orthogonal_(flat, gain=float(scale), generator=generator)
                with torch.no_grad():
                    weight.copy_(flat.to(weight.device))
                continue
            std = base if scheme == "scaled_normal" else 0.02
            draw = torch.empty(weight.shape, device="cpu").normal_(
                0.0, std * float(scale), generator=generator
            )
            with torch.no_grad():
                weight.copy_(draw.to(weight.device))
