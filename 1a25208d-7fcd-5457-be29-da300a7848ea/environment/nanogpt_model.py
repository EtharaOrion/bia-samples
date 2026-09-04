#!/usr/bin/env python3
"""The frozen decoder. Agent-visible, read-only, and shape-bound to the substrate.

This is the real model. Its architecture is read from
`environment/nanogpt_substrate.json` at construction and asserted rather than
assumed, so a checkpoint whose shapes disagree with the declared vocab_size,
num_layers, model_dim or head_dim fails to load instead of loading as something
else. Normalization is parameter-free root mean square normalization and the
positional encoding is rotary, so every parameter the checkpoint carries is one
of the fifty weight matrices `environment/model_stats.json` lists, and every one
of them is inside the bit budget.

The verifier carries its own definition of this same architecture in
`tests/decoder.py` and does not import this file. The two are bound together by
the checkpoint: they agree because they load the same state dict, with the same
keys, at the same shapes, and either one refuses a checkpoint that does not
match. That is deliberate. A verifier that imported the agent surface would be
grading a model the agent surface described.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SUBSTRATE = Path(__file__).resolve().parent / "nanogpt_substrate.json"


def architecture(path: Path = SUBSTRATE) -> dict:
    """Read the frozen architecture. Never hardcode what the substrate declares."""
    block = json.loads(Path(path).read_text(encoding="utf-8"))["architecture"]
    fields = ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len")
    values = {name: int(block[name]) for name in fields}
    if values["model_dim"] // values["head_dim"] != values["num_heads"]:
        raise ValueError("substrate architecture is inconsistent: model_dim // head_dim is not num_heads")
    return values


def rotary(seq_len: int, head_dim: int, device, dtype):
    half = head_dim // 2
    frequency = (1.0 / 1024.0) ** (torch.arange(half, device=device, dtype=torch.float32) / half)
    angle = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None] * frequency[None, :]
    return angle.cos().to(dtype), angle.sin().to(dtype)


def apply_rotary(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, model_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.head_dim, self.num_heads = head_dim, num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x, cos, sin):
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = apply_rotary(F.rms_norm(query, (self.head_dim,)), cos, sin)
        key = apply_rotary(F.rms_norm(key, (self.head_dim,)), cos, sin)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=True
        )
        return self.proj(out.transpose(1, 2).reshape(batch, seq, -1))


class MLP(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.fc = nn.Linear(model_dim, 4 * model_dim, bias=False)
        self.proj = nn.Linear(4 * model_dim, model_dim, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, model_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.attn = Attention(model_dim, head_dim, num_heads)
        self.mlp = MLP(model_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), cos, sin)
        return x + self.mlp(F.rms_norm(x, (x.size(-1),)))


class Decoder(nn.Module):
    """The canonical 12-layer 768-dim decoder. Fifty weight matrices, no others."""

    def __init__(self, arch: dict | None = None):
        super().__init__()
        self.arch = arch or architecture()
        dim, heads, head_dim = self.arch["model_dim"], self.arch["num_heads"], self.arch["head_dim"]
        self.wte = nn.Embedding(self.arch["vocab_size"], dim)
        self.blocks = nn.ModuleList(
            [Block(dim, head_dim, heads) for _ in range(self.arch["num_layers"])]
        )
        self.lm_head = nn.Linear(dim, self.arch["vocab_size"], bias=False)

    def forward(self, tokens, targets=None):
        cos, sin = rotary(tokens.size(1), self.arch["head_dim"], tokens.device, torch.float32)
        x = F.rms_norm(self.wte(tokens).float(), (self.arch["model_dim"],))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(F.rms_norm(x, (self.arch["model_dim"],))).float()
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


def quantizable_names(arch: dict) -> list:
    """The fifty tensor names the bit budget is spent over, in manifest order."""
    names = ["wte.weight"]
    for index in range(arch["num_layers"]):
        stem = "blocks." + str(index)
        names.extend(
            [
                stem + ".attn.qkv.weight",
                stem + ".attn.proj.weight",
                stem + ".mlp.fc.weight",
                stem + ".mlp.proj.weight",
            ]
        )
    names.append("lm_head.weight")
    return names


def load_checkpoint(model: Decoder, path) -> Decoder:
    """Load the frozen checkpoint and refuse one whose shapes are not the frozen ones."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    for name, shape in expected.items():
        if name not in state:
            raise ValueError("checkpoint is missing the frozen tensor " + name)
        if tuple(state[name].shape) != shape:
            raise ValueError("checkpoint tensor " + name + " has shape " + str(tuple(state[name].shape)) + ", not the frozen " + str(shape))
    extra = sorted(set(state) - set(expected))
    if extra:
        raise ValueError("checkpoint carries tensors the frozen architecture does not: " + ", ".join(extra))
    model.load_state_dict(state, strict=True)
    return model


def perplexity(model: Decoder, tokens, device) -> float:
    """Real forward passes. Perplexity is the exponential of the mean token loss.

    There is no closed form here and no substitute for it: the number this
    function returns exists only because the model ran over the tokens. Remove
    the forward pass and there is nothing left to return.
    """
    model.eval()
    seq = model.arch["seq_len"]
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for row in range(tokens.size(0)):
            window = tokens[row].to(device)
            loss = model(window[None, :seq], window[None, 1 : seq + 1])
            total_loss += float(loss) * seq
            total_tokens += seq
    return math.exp(total_loss / total_tokens)
