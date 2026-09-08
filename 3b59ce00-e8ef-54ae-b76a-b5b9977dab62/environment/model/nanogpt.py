#!/usr/bin/env python3
"""The frozen nanoGPT decoder. modded-nanoGPT lineage, sized for this compute envelope.

Transcribed from the vendored decoder at
samples/3161f7d5-0cec-507c-aae1-ef7e60bdaf99/environment/model/nanogpt.py and reduced
to 6 layers / model_dim 384 / head_dim 64 / seq_len 512, which is the size that lets a
verifier train THREE independent runs -- the shipped default, a held-out reference and
the submission -- inside one grading pass on a single H100.

The forward pass is a real forward pass: embedding lookup, six pre-norm blocks of rotary
causal self attention and a squared-ReLU MLP, a final RMSNorm, the untied output
projection and a logit softcap. The SHAPE of this network is FROZEN -- nothing a
submission says may move vocab_size, num_layers, model_dim, head_dim or seq_len. What a
submission moves is the OUTPUT-DISTRIBUTION AND REGULARISATION POLICY applied on top of
it: the cap on the logits, the label smoothing and the log-partition penalty in the
training objective, and the per-role decoupled weight decay. The verifier reads its own
copy of this file.
"""


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Upstream attention scale, train_gpt_simple.py. The upstream logit softcap is kept
# here as the DEFAULT_LOGIT_SOFTCAP only: in this slot the softcap is one of the
# output-distribution controls a submission sets, so it arrives as a run-time TENSOR
# argument rather than as a module constant. Passing it as a tensor is deliberate --
# a Python float would be burned into the compiled graph and every distinct value a
# run used would pay a fresh compilation, which would make grading cost a function of
# what was submitted.
DEFAULT_LOGIT_SOFTCAP = 15.0
ATTENTION_SCALE = 0.12


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
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor) -> Tensor:
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
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

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
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
        x = self.fc(x)
        x = x.relu().square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, head_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def logits(self, inputs: Tensor, softcap: Tensor) -> Tensor:
        """Capped logits. `softcap` is a 0-dim tensor, never a Python float."""
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        out = self.proj(self.norm2(x)).float()
        return softcap * out * (out.square() + softcap.square()).rsqrt()

    def forward(self, inputs: Tensor, targets: Tensor, softcap: Tensor,
                smoothing: Tensor, z_weight: Tensor) -> Tensor:
        """The TRAINING objective: smoothed cross entropy plus a log-partition penalty.

        Every control arrives as a 0-dim tensor so that one compiled graph serves every
        policy. Setting `smoothing` and `z_weight` to zero recovers plain cross entropy
        exactly, which is what `evaluate` calls this with.

        The smoothing term is written out rather than delegated to
        F.cross_entropy(label_smoothing=...), because that argument is a Python float
        and would specialise the graph to one value.
        """
        out = self.logits(inputs, softcap)
        flat = out.reshape(targets.numel(), -1)
        target = targets.reshape(-1)
        logprobs = F.log_softmax(flat, dim=-1)
        nll = -logprobs.gather(1, target.unsqueeze(1)).squeeze(1).mean()
        uniform = -logprobs.mean()
        loss = (1.0 - smoothing) * nll + smoothing * uniform
        partition = torch.logsumexp(flat, dim=-1)
        return loss + z_weight * partition.square().mean()

    def cross_entropy(self, inputs: Tensor, targets: Tensor, softcap: Tensor) -> Tensor:
        """The GRADED reading: plain mean cross entropy, no smoothing, no penalty.

        The softcap is part of the trained model rather than a scoring choice, so the
        reading is taken through the same cap the run trained under.
        """
        out = self.logits(inputs, softcap)
        return F.cross_entropy(out.reshape(targets.numel(), -1), targets.reshape(-1),
                               reduction="mean")
