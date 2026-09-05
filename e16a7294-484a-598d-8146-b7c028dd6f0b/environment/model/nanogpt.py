#!/usr/bin/env python3
"""The canonical nanoGPT decoder for slot OER-21, and the parameter shard table over it.

This module is the ONLY definition of the architecture in this bundle. It is transcribed
from the vendored record set at harness/records/track_3_optimization/train_gpt_simple.py
and it is shape-bound at construction time to environment/nanogpt_substrate.json, so a
declaration and a module cannot drift apart without `build_model` refusing to return.

The architecture is the frozen one: vocab_size 50304, num_layers 12, model_dim 768,
head_dim 128, num_heads 6, seq_len 1024. Nothing in this task may move it. What a
submission allocates bits over is the parameter shard table this module derives from that
architecture, and the numbers in that table are a function of the architecture alone.

The forward pass here is a real forward pass: embedding lookup, twelve pre-norm blocks of
rotary causal self attention and a squared-relu MLP, a final RMSNorm, the untied output
projection and the upstream logit softcap. Delete it and no perplexity exists to grade.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

HERE = Path(__file__).resolve().parent
SUBSTRATE = HERE.parent / "nanogpt_substrate.json"

# The upstream logit softcap constant, train_gpt_simple.py GPT.forward.
LOGIT_SOFTCAP = 15.0
# The upstream attention scale, train_gpt_simple.py CausalSelfAttention.forward.
ATTENTION_SCALE = 0.12


class ArchitectureError(RuntimeError):
    """The parameters on disk are not the frozen architecture. Refused, never coerced."""


def load_substrate(path: Path = SUBSTRATE) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    def __init__(self, dim: int, head_dim: int = 128):
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
    """The frozen decoder. Constructed only from the substrate declaration."""

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

    def logits(self, inputs: Tensor) -> Tensor:
        """One real forward pass. Returns softcapped float32 logits, shape B x T x vocab."""
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        out = self.proj(self.norm2(x)).float()
        return LOGIT_SOFTCAP * out * (out.square() + LOGIT_SOFTCAP ** 2).rsqrt()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        out = self.logits(inputs)
        return F.cross_entropy(out.reshape(targets.numel(), -1), targets.reshape(-1), reduction="mean")


def build_model(substrate: dict | None = None) -> GPT:
    """Instantiate the frozen decoder from the substrate declaration and nothing else."""
    declaration = substrate if substrate is not None else load_substrate()
    architecture = declaration["architecture"]
    model = GPT(
        vocab_size=int(architecture["vocab_size"]),
        num_layers=int(architecture["num_layers"]),
        model_dim=int(architecture["model_dim"]),
        head_dim=int(architecture["head_dim"]),
    )
    expected_heads = int(architecture["num_heads"])
    if model.model_dim // model.head_dim != expected_heads:
        raise ArchitectureError(
            "the substrate declares num_heads "
            + str(expected_heads)
            + " but model_dim // head_dim is "
            + str(model.model_dim // model.head_dim)
        )
    return model


def shard_table(substrate: dict | None = None) -> list:
    """Every quantizable parameter shard, in a fixed order, derived from the architecture.

    A shard is one two-dimensional weight matrix of the frozen decoder. Biases and RMSNorm
    gains are not shards: they are 0.09 percent of the parameters, they are held at the
    checkpoint's own bfloat16 width, and they sit outside the bit budget, which is stated
    in environment/model/checkpoint.json rather than left for a reader to infer.
    """
    declaration = substrate if substrate is not None else load_substrate()
    architecture = declaration["architecture"]
    vocab = int(architecture["vocab_size"])
    layers = int(architecture["num_layers"])
    dim = int(architecture["model_dim"])
    hidden = 4 * dim
    rows = [{"name": "embed.weight", "shape": [vocab, dim], "role": "token-embedding"}]
    for index in range(layers):
        stem = "blocks." + str(index) + "."
        rows.append({"name": stem + "attn.q.weight", "shape": [dim, dim], "role": "attention-query"})
        rows.append({"name": stem + "attn.k.weight", "shape": [dim, dim], "role": "attention-key"})
        rows.append({"name": stem + "attn.v.weight", "shape": [dim, dim], "role": "attention-value"})
        rows.append({"name": stem + "attn.proj.weight", "shape": [dim, dim], "role": "attention-output"})
        rows.append({"name": stem + "mlp.fc.weight", "shape": [hidden, dim], "role": "mlp-input"})
        rows.append({"name": stem + "mlp.proj.weight", "shape": [dim, hidden], "role": "mlp-output"})
    rows.append({"name": "proj.weight", "shape": [vocab, dim], "role": "output-projection"})
    for row in rows:
        row["numel"] = int(row["shape"][0]) * int(row["shape"][1])
    return rows


def residue_parameters(substrate: dict | None = None) -> int:
    """Parameters that are not shards: every bias and every RMSNorm gain vector."""
    declaration = substrate if substrate is not None else load_substrate()
    architecture = declaration["architecture"]
    vocab = int(architecture["vocab_size"])
    layers = int(architecture["num_layers"])
    dim = int(architecture["model_dim"])
    hidden = 4 * dim
    per_block = 4 * dim + hidden + dim + 2 * dim
    return layers * per_block + vocab + 2 * dim


def assert_checkpoint_matches(state: dict, substrate: dict | None = None) -> dict:
    """Refuse a parameter snapshot that is not the frozen architecture.

    This is the first simulator condition made executable: the graded artifact has to be a
    real parameter snapshot of the declared decoder, shape-bound to vocab_size, num_layers,
    model_dim and head_dim. A table of the wrong shape is refused here rather than quietly
    graded.
    """
    declaration = substrate if substrate is not None else load_substrate()
    table = shard_table(declaration)
    missing = [row["name"] for row in table if row["name"] not in state]
    if missing:
        raise ArchitectureError("the snapshot is missing " + str(len(missing)) + " declared shard(s): " + ", ".join(missing[:4]))
    wrong = []
    for row in table:
        observed = list(state[row["name"]].shape)
        if observed != list(row["shape"]):
            wrong.append(row["name"] + " is " + str(observed) + " and the architecture declares " + str(row["shape"]))
    if wrong:
        raise ArchitectureError("; ".join(wrong[:4]))
    return {
        "shards": len(table),
        "quantizable_parameters": sum(row["numel"] for row in table),
        "residue_parameters": residue_parameters(declaration),
        "vocab_size": int(declaration["architecture"]["vocab_size"]),
        "num_layers": int(declaration["architecture"]["num_layers"]),
        "model_dim": int(declaration["architecture"]["model_dim"]),
        "head_dim": int(declaration["architecture"]["head_dim"]),
    }


def load_checkpoint(path: Path, substrate: dict | None = None) -> dict:
    """Read a parameter snapshot from disk and shape-check it before returning it."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ArchitectureError("the checkpoint does not carry a parameter dictionary")
    assert_checkpoint_matches(state, substrate)
    return state
