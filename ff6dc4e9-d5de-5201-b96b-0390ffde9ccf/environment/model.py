#!/usr/bin/env python3
"""The frozen nanoGPT decoder, the group-wise quantizer, and the perplexity reader.

This module is the model in the loop for OER-22. Every architectural number it uses
is read at run time from the canonical declaration at
`environment/nanogpt_substrate.json`, which is replicated into this bundle from
`.seed/nanogpt_substrate.json` and whose values come from the vendored record set at
`harness/records/track_3_optimization/train_gpt_simple.py`. Nothing here is invented
and nothing here is a stand-in.

What this file gives a solver: the real decoder, the real checkpoint loader, the real
group-wise quantizer, and a perplexity reader you may point at any token stream you
are allowed to see. What it does not give you is the graded reading. The verifier owns
its own copy of this module, its own pristine checkpoint and its own held-out FineWeb
slice, and the number that scores your submission is computed there. Anything this
module prints inside your container is telemetry.

Delete `GPT.forward` and every perplexity in this slot becomes undefined rather than
merely different. There is no count table, no cost model, no sensitivity vector and no
closed-form degradation anywhere in this bundle that could keep emitting a number
after the forward pass is removed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_SCHEMA = "oer22.checkpoint/v1"
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256

ARCHITECTURE_KEYS = ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len")


# --------------------------------------------------------------------------
# The canonical declaration. Read, never hardcoded.
# --------------------------------------------------------------------------
def load_declaration(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def architecture(declaration: dict) -> dict:
    block = declaration["architecture"]
    return {key: int(block[key]) for key in ARCHITECTURE_KEYS}


def expected_parameter_shapes(arch: dict) -> dict:
    """Every parameter shape the declaration implies, derived from it alone.

    The verifier compares a checkpoint's state dict against this map before it runs
    anything, so a parameter table that is not this architecture cannot be passed off
    as one. The map is a pure function of vocab_size, num_layers, model_dim, head_dim,
    num_heads and seq_len.
    """
    vocab = arch["vocab_size"]
    layers = arch["num_layers"]
    dim = arch["model_dim"]
    inner = arch["head_dim"] * arch["num_heads"]
    shapes = {
        "embed.weight": [vocab, dim],
        "position.weight": [arch["seq_len"], dim],
        "head.weight": [vocab, dim],
        "final_norm.weight": [dim],
    }
    for i in range(layers):
        shapes["blocks." + str(i) + ".norm_attention.weight"] = [dim]
        shapes["blocks." + str(i) + ".attention.qkv.weight"] = [3 * inner, dim]
        shapes["blocks." + str(i) + ".attention.projection.weight"] = [dim, inner]
        shapes["blocks." + str(i) + ".norm_mlp.weight"] = [dim]
        shapes["blocks." + str(i) + ".mlp.up.weight"] = [4 * dim, dim]
        shapes["blocks." + str(i) + ".mlp.down.weight"] = [dim, 4 * dim]
    return shapes


# --------------------------------------------------------------------------
# The decoder.
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, 1e-6)


class CausalSelfAttention(nn.Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.num_heads = arch["num_heads"]
        self.head_dim = arch["head_dim"]
        inner = self.num_heads * self.head_dim
        self.qkv = nn.Linear(arch["model_dim"], 3 * inner, bias=False)
        self.projection = nn.Linear(inner, arch["model_dim"], bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).split(self.num_heads * self.head_dim, dim=2)
        shape = (batch, length, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, length, self.num_heads * self.head_dim)
        return self.projection(out)


class MLP(nn.Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.up = nn.Linear(arch["model_dim"], 4 * arch["model_dim"], bias=False)
        self.down = nn.Linear(4 * arch["model_dim"], arch["model_dim"], bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.norm_attention = RMSNorm(arch["model_dim"])
        self.attention = CausalSelfAttention(arch)
        self.norm_mlp = RMSNorm(arch["model_dim"])
        self.mlp = MLP(arch)

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class GPT(nn.Module):
    """The canonical 12-layer, 768-dim decoder of the frozen substrate."""

    def __init__(self, arch: dict):
        super().__init__()
        self.arch = dict(arch)
        self.embed = nn.Embedding(arch["vocab_size"], arch["model_dim"])
        self.position = nn.Embedding(arch["seq_len"], arch["model_dim"])
        self.blocks = nn.ModuleList([Block(arch) for _ in range(arch["num_layers"])])
        self.final_norm = RMSNorm(arch["model_dim"])
        self.head = nn.Linear(arch["model_dim"], arch["vocab_size"], bias=False)

    def forward(self, tokens, targets):
        """THE forward pass. Returns mean cross entropy in nats over the batch.

        Removing this method removes every perplexity in this slot. There is no
        fallback path in this bundle that produces one without it.
        """
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.final_norm(x))
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="mean"
        )


def build(arch: dict, device: str) -> GPT:
    return GPT(arch).to(device)


# --------------------------------------------------------------------------
# Checkpoints. Real parameters, shape-bound to the declaration.
# --------------------------------------------------------------------------
def read_checkpoint(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def checkpoint_shapes(path: Path) -> dict:
    payload = read_checkpoint(Path(path))
    state = payload.get("state_dict") or {}
    return {name: list(tensor.shape) for name, tensor in state.items()}


def load_into(model: GPT, path: Path) -> dict:
    """Load a checkpoint's parameters, strictly. A shape disagreement raises."""
    payload = read_checkpoint(Path(path))
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload


def content_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Shards and slices. The FineWeb10B token stream, in the layout the pinned
# loader at data/cached_fineweb10B.py writes.
# --------------------------------------------------------------------------
def read_shard(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.int32, count=SHARD_HEADER_INTS)
    if int(raw[0]) != SHARD_MAGIC or int(raw[1]) != SHARD_VERSION:
        raise ValueError("shard " + str(path) + " does not carry the bound header")
    count = int(raw[2])
    tokens = np.fromfile(str(path), dtype=np.uint16, offset=SHARD_HEADER_INTS * 4, count=count)
    if tokens.size != count:
        raise ValueError("shard " + str(path) + " declares " + str(count) + " tokens and carries " + str(tokens.size))
    return tokens


def load_stream(paths: list) -> np.ndarray:
    pieces = [read_shard(Path(p)) for p in paths]
    if not pieces:
        return np.zeros(0, dtype=np.uint16)
    return np.concatenate(pieces)


def read_slice(root: Path, pin: dict) -> np.ndarray:
    """The token slice one pin names. Deterministic, contiguous, no random source."""
    tokens = read_shard(Path(root) / str(pin["shard"]))
    low = int(pin["token_offset"])
    high = low + int(pin["token_count"])
    window = tokens[low:high]
    if window.size != int(pin["token_count"]):
        raise ValueError("the pinned slice runs past the end of " + str(pin["shard"]))
    return window


def slice_content_sha256(root: Path, pin: dict) -> str:
    """The content digest of a pinned slice, over the bytes actually staged.

    This is the discovery value the graded path depends on. It is established only in
    built environment state, because it is a digest of the FineWeb bytes the image
    staged rather than of anything written down in this bundle. A run that never read
    that state cannot produce it, and a run that read the wrong slice produces a
    different one.
    """
    return hashlib.sha256(read_slice(Path(root), pin).tobytes()).hexdigest()


def _batches(stream: np.ndarray, seq_len: int, rows: int, offset: int, device: str):
    """Contiguous, deterministic. No shuffling, no random source, no clock."""
    span = seq_len * rows
    window = stream[offset:offset + span + 1]
    if window.size < span + 1:
        return None
    chunk = torch.from_numpy(window.astype(np.int64))
    tokens = chunk[:-1].view(rows, seq_len).to(device, non_blocking=True)
    targets = chunk[1:].view(rows, seq_len).to(device, non_blocking=True)
    return tokens, targets


# --------------------------------------------------------------------------
# The quantizer. Group-wise symmetric round to nearest, one scale per group.
# --------------------------------------------------------------------------
def quantize_dequantize(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Quantize one weight tensor to `bits` and return it dequantized in place.

    The tensor is flattened, split into contiguous groups of `group_size` elements,
    and each group carries one scale held at `scale_bits` precision. This is the
    function the verifier replays on its own pristine checkpoint, so a submission that
    declares a scheme this function cannot express has declared nothing.
    """
    flat = weight.reshape(-1).float()
    if flat.numel() % int(group_size) != 0:
        raise ValueError("group_size must divide the tensor element count exactly")
    groups = flat.view(-1, int(group_size))
    levels = float(2 ** (int(bits) - 1) - 1)
    if levels <= 0:
        raise ValueError("bits must be at least 2 for a symmetric signed grid")
    scale = groups.abs().amax(dim=1, keepdim=True) / levels
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale = scale.half().float()
    quantized = torch.clamp(torch.round(groups / scale), -levels - 1.0, levels)
    return (quantized * scale).view_as(weight).to(weight.dtype)


def apply_allocation(model: GPT, allocation: dict, group_size: int) -> dict:
    """Quantize every named tensor at its allocated width. Harness-owned arithmetic."""
    touched = {}
    state = dict(model.named_parameters())
    with torch.no_grad():
        for name in sorted(allocation):
            if name not in state:
                raise KeyError("the allocation names a tensor the architecture does not carry: " + name)
            parameter = state[name]
            parameter.copy_(quantize_dequantize(parameter.data, int(allocation[name]), group_size))
            touched[name] = int(allocation[name])
    return touched


def allocated_bits(tensors, allocation, group_size, scale_bits, include_scales) -> int:
    """The accounting in force, over the quantized tensors including their scales."""
    weight_bits = sum(int(row["numel"]) * int(allocation[row["id"]]) for row in tensors)
    if not include_scales:
        return weight_bits
    return weight_bits + sum(int(row["numel"]) // int(group_size) * int(scale_bits) for row in tensors)


# --------------------------------------------------------------------------
# Perplexity. A real forward pass over a real token stream.
# --------------------------------------------------------------------------
@torch.no_grad()
def perplexity(model: GPT, stream: np.ndarray, seq_len: int, rows: int, device: str) -> dict:
    """Mean cross entropy in nats per token, and its exponential.

    This is the only producer of a perplexity in this bundle. Point it at a stream you
    are allowed to see and it will tell you about that stream. The graded reading is
    the verifier's own call on its own held-out slice, and it never takes an argument
    a submission wrote.
    """
    model.eval()
    span = seq_len * rows
    total = 0.0
    counted = 0
    offset = 0
    while offset + span + 1 <= stream.size:
        batch = _batches(stream, seq_len, rows, offset, device)
        if batch is None:
            break
        tokens, targets = batch
        total += float(model(tokens, targets)) * span
        counted += span
        offset += span
    if counted == 0:
        raise ValueError("the stream carries no full evaluation window")
    loss = total / counted
    return {"loss": loss, "perplexity": float(np.exp(loss)), "tokens_evaluated": counted}
