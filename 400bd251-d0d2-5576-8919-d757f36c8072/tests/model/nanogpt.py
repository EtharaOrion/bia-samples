#!/usr/bin/env python3
"""The frozen nanoGPT decoder, with its execution plan exposed.

modded-nanoGPT lineage, 6 layers / model_dim 384 / head_dim 64 / seq_len 512.

WHAT IS FROZEN AND WHAT IS NOT
    The MATHEMATICS is frozen. Embedding lookup, six pre-norm blocks of rotary
    causal self attention and a squared-ReLU MLP, a final RMSNorm, the untied
    output projection, the upstream logit softcap and the mean token-level
    cross-entropy. The parameters are frozen. The batch shape is frozen.

    What a submission chooses is HOW those frozen operations are executed: which
    attention backend evaluates the same softmax-weighted average, in how many
    chunks the same cross-entropy is summed, and in what precision the rotary,
    the qk normalisation and the logit softcap are carried.

    Every option in this file is intended to compute the same function. Whether a
    given option actually does, to the tolerance the verifier holds, is not taken
    on trust: the verifier checks it against the reference plan on fixed weights
    and a fixed batch before it times anything, and a plan that fails is refused.

    `logit_dtype` is the interesting one, and it is here on purpose. Carrying the
    softcap and the cross-entropy in bfloat16 over 50304 classes is markedly
    faster and is NOT the same function to the tolerance this task holds. It is
    reachable, it is fast, and it is refused.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

LOGIT_SOFTCAP = 15.0
ATTENTION_SCALE = 0.12

BACKENDS = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
}

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq",
                             torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor, work: torch.dtype) -> Tensor:
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos().to(work), theta.sin().to(work)
        x1, x2 = x_BTHD.to(dtype=work).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor, plan: dict) -> Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        norm_work = DTYPES[plan["qk_norm_dtype"]]
        q = F.rms_norm(q.to(norm_work), (q.size(-1),)).type_as(x)
        k = F.rms_norm(k.to(norm_work), (k.size(-1),)).type_as(x)
        rot_work = DTYPES[plan["rotary_dtype"]]
        q, k = self.rotary(q, rot_work), self.rotary(k, rot_work)
        with sdpa_kernel(BACKENDS[plan["attention"]]):
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=ATTENTION_SCALE, is_causal=True,
            ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor, plan: dict) -> Tensor:
        x = x + self.attn(self.norm1(x), plan)
        x = x + self.mlp(self.norm2(x))
        return x


def softcap(out: Tensor) -> Tensor:
    return LOGIT_SOFTCAP * out * (out.square() + LOGIT_SOFTCAP ** 2).rsqrt()


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, head_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def hidden(self, inputs: Tensor, plan: dict) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x, plan)
        return self.norm2(x)

    def forward(self, inputs: Tensor, targets: Tensor, plan: dict) -> Tensor:
        """Mean token-level cross-entropy. Chunked or not, the same sum."""
        h = self.hidden(inputs, plan)
        flat = h.reshape(-1, h.size(-1))
        flat_targets = targets.reshape(-1)
        work = DTYPES[plan["logit_dtype"]]
        chunks = int(plan["loss_chunks"])
        n = flat.size(0)
        if n % chunks:
            raise ValueError(f"loss_chunks {chunks} does not divide {n} rows")
        size = n // chunks
        total = None
        for i in range(chunks):
            piece = flat[i * size:(i + 1) * size]
            out = self.proj(piece).to(work)
            capped = softcap(out)
            part = F.cross_entropy(capped.float() if work is torch.float32 else capped,
                                   flat_targets[i * size:(i + 1) * size], reduction="sum")
            total = part if total is None else total + part
        return total / n
