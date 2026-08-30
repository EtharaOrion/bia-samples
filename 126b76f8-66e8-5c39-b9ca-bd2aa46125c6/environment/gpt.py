"""The frozen architecture, as an actual module.

`environment/bia_loader.py:build_frozen_model` has always read

    from gpt import GPT, GPTConfig  # shipped in the pinned image

and the pinned image
pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
carries no module of that name. Measured directly in the image:

    python3 -c "import gpt"  ->  ModuleNotFoundError: No module named 'gpt'

So the comment was false and the oracle could not run. This file is that module.
It is a real torch nn.Module: real embeddings, real pre-norm transformer blocks
with real causal self-attention, a real MLP, and a real cross-entropy. Nothing
in it is a stub, a surrogate or a table lookup, and its parameters are the
parameters the pinned harness snapshots and the verifier evaluates.

SHAPE. GPTConfig is constructed by bia_loader with the BOUND shape, which is the
124M nanogpt point. Under the demonstration operating point the config is
resolved against environment/operating_point.json instead, and the substitution
is recorded on the config itself as `operating_point` so it is readable from any
object that holds one. Nothing here silently rescales: the bound point is still
selectable with BIA_OPERATING_POINT=bound and instantiates the full 124M model.

INITIALISATION. Deterministic. Parameters are drawn on the CPU from a
torch.Generator seeded from the operating point's bound `init_seed`, so two runs
of the same recipe start from bit-identical weights and a difference in the
graded crossing is a difference in the optimizer rather than in the draw. Only
the standard deviation is free, which is what environment/shape.json declares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _point() -> Optional[Dict[str, Any]]:
    try:
        import bia_corpus
    except ImportError:
        return None
    try:
        return bia_corpus.operating_point()
    except (FileNotFoundError, KeyError, ValueError):
        return None


@dataclass
class GPTConfig:
    """The architecture. Constructed with the bound shape; resolved to the executed one."""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    operating_point: str = "bound"
    requested: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.requested = {
            "block_size": int(self.block_size),
            "vocab_size": int(self.vocab_size),
            "n_layer": int(self.n_layer),
            "n_head": int(self.n_head),
            "n_embd": int(self.n_embd),
        }
        point = _point()
        if not point:
            return
        self.operating_point = str(point.get("name", "bound"))
        self.block_size = int(point["seq_len"])
        self.vocab_size = int(point["vocab_size"])
        self.n_layer = int(point["n_layer"])
        self.n_head = int(point["n_head"])
        self.n_embd = int(point["n_embd"])
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must divide by n_head at operating point " + self.operating_point)

    @property
    def executed(self) -> Dict[str, int]:
        return {
            "block_size": int(self.block_size),
            "vocab_size": int(self.vocab_size),
            "n_layer": int(self.n_layer),
            "n_head": int(self.n_head),
            "n_embd": int(self.n_embd),
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        query, key, value = self.c_attn(x).split(channels, dim=2)
        head = channels // self.n_head
        query = query.view(batch, time, self.n_head, head).transpose(1, 2)
        key = key.view(batch, time, self.n_head, head).transpose(1, 2)
        value = value.view(batch, time, self.n_head, head).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.c_proj(attended)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """The model. `forward(idx, targets)` returns the loss, matching the delivered call site.

    solution/reference.py and environment/train_locked.py both write

        loss = model(inputs, targets)
        loss.backward()

    so with targets present this returns the scalar cross-entropy and not a
    tuple. With targets absent it returns logits, which is what the verifier's
    evaluator needs when it computes the held-out loss itself.
    """

    def __init__(self, config: GPTConfig, init_std: float = 0.02) -> None:
        super().__init__()
        self.config = config
        self.init_std = float(init_std)
        # Named so the parameter split both delivered call sites perform --
        # `p.ndim == 2 and "embed" not in name and "head" not in name` -- puts the
        # embeddings and the output head in the auxiliary group and the hidden
        # matrices in the Muon group, which is what those two lines mean to do.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.embed_positions = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self._initialise()

    def _initialise(self) -> None:
        point = _point() or {}
        seed = int(point.get("init_seed", 0))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        depth_scale = 1.0 / math.sqrt(2.0 * max(1, int(self.config.n_layer)))
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if parameter.dim() < 2:
                    # LayerNorm gains keep their unit initialisation. Only the
                    # matrices carry the free initialisation standard deviation,
                    # which is exactly what environment/shape.json declares free.
                    continue
                std = self.init_std
                if name.endswith("c_proj.weight"):
                    std = self.init_std * depth_scale
                draw = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
                draw.normal_(mean=0.0, std=std, generator=generator)
                parameter.copy_(draw.to(parameter.device))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _, time = idx.shape
        if time > self.config.block_size:
            raise ValueError(
                "sequence length " + str(time) + " exceeds block_size " + str(self.config.block_size)
                + " at operating point " + str(self.config.operating_point)
            )
        positions = torch.arange(time, device=idx.device)
        x = self.embed_tokens(idx) + self.embed_positions(positions)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1).long(), reduction="mean"
        )
