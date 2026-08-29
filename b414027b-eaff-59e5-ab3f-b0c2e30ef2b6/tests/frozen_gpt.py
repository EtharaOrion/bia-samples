"""The frozen architecture. One definition, shared by the training script and the verifier.

It lives in the image rather than in the submission, so the agent cannot redefine
it, and the verifier loads submitted checkpoints into this same class with
strict=True. A submission that altered depth, width, head dimension or vocabulary
therefore fails to load rather than being trusted to have left them alone.

forward() returns the SUM of the token-level cross entropy, not the mean. The
verifier divides by its own token count, so returning a mean here would make the
graded loss depend on how the verifier happened to batch its evaluation.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_device() -> torch.device:
    """cuda when present, otherwise the best available. Declared, never silent.

    The graded environment is a single H100, so cuda is the real target. The
    fallback exists so the bundle can be exercised end to end off the target
    hardware, and every run records which device it used.
    """
    want = os.environ.get("BIA_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.gain * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        assert dim % head_dim == 0, "model_dim must be divisible by head_dim"
        self.n_head = dim // head_dim
        self.head_dim = head_dim
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        B, T, C = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(B, T, C))
        h = self.norm2(x)
        return x + self.fc2(F.relu(self.fc(h)).square())


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, head_dim) for _ in range(num_layers)])
        self.norm = RMSNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02 / math.sqrt(2 * num_layers))

    def forward(self, inputs, targets):
        x = self.embed(inputs)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x)).float()
        return F.cross_entropy(
            logits.view(-1, self.vocab_size), targets.reshape(-1), reduction="sum"
        )
