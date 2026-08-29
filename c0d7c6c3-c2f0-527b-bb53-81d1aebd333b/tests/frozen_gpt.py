"""The frozen architecture, and the frozen forward-backward accounting.

One definition, shared by the training image and the verifier, so the two
cannot disagree about what "frozen" means. The verifier loads every submitted
checkpoint into its own copy of this class with strict=True, which makes depth,
width, head dimension and vocabulary structurally enforced rather than trusted.

forward() returns the SUM of token-level cross entropy, never the mean, because
the verifier divides by its own token count. Returning a mean here would make
the graded loss depend on how the verifier chose to batch its evaluation.

The forward counter exists because the task freezes the rule of exactly one
forward-backward pass per optimizer step, and a rule nothing counts is a rule
nothing enforces. The count is written by the loader into the run ledger and
reconciled by the verifier against the step count. It is written inside the
submission's own process and is therefore forgeable by a submission willing to
reimplement the loader; that residue is named in solution/rubrics.json rather
than papered over. What the counter does buy is that the ordinary route to a
larger effective batch, running several forward-backward passes per step, is
caught deterministically.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_device() -> torch.device:
    want = os.environ.get("BIA_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.gain * (x.float() * scale).to(x.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        assert dim % head_dim == 0, "model_dim must be divisible by head_dim"
        self.n_head = dim // head_dim
        self.head_dim = head_dim
        self.norm_attn = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm_mlp = RMSNorm(dim)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        b, t, c = x.shape
        h = self.norm_attn(x)
        q, k, v = self.qkv(h).split(c, dim=2)
        shape = (b, t, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(a.transpose(1, 2).contiguous().view(b, t, c))
        return x + self.down(F.gelu(self.up(self.norm_mlp(x))))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, head_dim) for _ in range(num_layers)])
        self.norm_out = RMSNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02 / math.sqrt(2 * num_layers))
        self.forward_calls = 0

    def forward(self, inputs, targets):
        self.forward_calls += 1
        x = self.embed(inputs)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm_out(x)).float()
        return F.cross_entropy(
            logits.view(-1, self.vocab_size), targets.reshape(-1), reduction="sum"
        )
