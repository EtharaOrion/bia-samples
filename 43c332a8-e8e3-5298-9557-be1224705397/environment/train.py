#!/usr/bin/env python3
"""The locked baseline trainer. It reaches below 3.28 validation loss and it is slow.

This is the substrate you start from. Copy it to /app/submission.py and change the
optimizer, its schedules, or the initialization. Everything else is frozen by the
upstream rule and by the checkers:

    frozen   dataset, batch size, architecture, one forward-backward pass per step
    free     optimization algorithm, hyperparameters and their schedules, init

Three structural facts you must preserve, because the verifier re-executes this file
and measures the axes it moved:

  * `step()` performs EXACTLY ONE forward and ONE backward per optimizer step. A
    second backward inside one step is a frozen-axis move, not a clever schedule.
  * `FROZEN` is echoed verbatim into the run manifest. Editing it does not change
    what the harness locked; it only makes the mismatch loud.
  * `--snapshot-dir` and `--snapshot-steps` are the harness's evaluation handle. The
    harness creates the directory and dictates the step set; this script writes
    `state_<step>.pt` at exactly those steps and at no others. Writing an
    unscheduled step, or omitting a scheduled one, is what makes a
    submission-selected checkpoint visible to the checker. You do not choose which
    state is graded.

Nothing this file prints is graded. The verifier evaluates the snapshots the harness
scheduled, in its own process, on the frozen held-out split, and reads none of this
file's output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Frozen axes. The harness locks these; a submission that moves one is rejected
# before it is scored.
# --------------------------------------------------------------------------- #
FROZEN = {
    "dataset": "fineweb10B",
    "sequence_length": 1024,
    "batch_size": 512,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50257,
    "passes_per_step": 1,
}

TARGET_VAL_LOSS = 3.28


@dataclass
class Config:
    steps: int = 3250
    lr: float = 1.8e-3
    weight_decay: float = 0.1
    warmup: int = 256
    seed: int = 0
    data_dir: str = "/data/fineweb10B"
    out_dir: str = "/app/submission/logs"


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        shape = (b, t, self.n_head, c // self.n_head)
        q, k, v = (z.view(*shape).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """The frozen architecture. Widening it is a frozen-axis move."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.wpe = nn.Embedding(cfg["sequence_length"], cfg["n_embd"])
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layer"]))
        self.lnf = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss


class ShardLoader:
    """Reads the frozen shards. The split boundary is not a tunable."""

    def __init__(self, data_dir: str, split: str, batch_size: int, seq_len: int, seed: int):
        self.path = os.path.join(data_dir, split + ".bin")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.from_numpy(
            __import__("numpy").memmap(self.path, dtype="uint16", mode="r")[:].astype("int64")
        )

    def batch(self, device: str):
        high = self.tokens.numel() - self.seq_len - 1
        starts = torch.randint(0, high, (self.batch_size,), generator=self.generator)
        x = torch.stack([self.tokens[s: s + self.seq_len] for s in starts])
        y = torch.stack([self.tokens[s + 1: s + 1 + self.seq_len] for s in starts])
        return x.to(device), y.to(device)


def lr_at(cfg: Config, step: int) -> float:
    """The baseline schedule. This is free; replace it."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model: nn.Module, cfg: Config):
    """The baseline optimizer. This is free; replace it.

    The upstream baseline this anchor was measured against is tuned Muon with an
    auxiliary AdamW on the embedding and head. Plain AdamW is shipped here so the
    starting point is honest about being slow rather than being a record already.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=True,
    )


def step(model, optimizer, loader, device, lr) -> float:
    """Exactly one forward and one backward. Do not add a second of either."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    x, y = loader.batch(device)
    _, loss = model(x, y)          # one forward
    loss.backward()                # one backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--out-dir", type=str, default=Config.out_dir)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--snapshot-dir", type=str, default="")
    parser.add_argument("--snapshot-steps", type=str, default="")
    args = parser.parse_args()

    snapshot_steps = {
        int(token) for token in args.snapshot_steps.split(",") if token.strip()
    }

    cfg = Config(steps=args.steps, seed=args.seed, data_dir=args.data_dir, out_dir=args.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(FROZEN).to(device)
    optimizer = build_optimizer(model, cfg)
    loader = ShardLoader(cfg.data_dir, "train", FROZEN["batch_size"], FROZEN["sequence_length"], cfg.seed)

    # The run manifest the harness reads back. It records the axes this process
    # actually ran under, so a moved axis is measured rather than asserted.
    manifest = {
        "frozen_axes": dict(FROZEN),
        "steps_requested": cfg.steps,
        "seed": cfg.seed,
        "target_validation_loss": TARGET_VAL_LOSS,
    }
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)

    for i in range(cfg.steps):
        train_loss = step(model, optimizer, loader, device, lr_at(cfg, i))
        completed = i + 1
        # The harness scheduled these steps and no others. Writing outside the set is
        # recorded by the ingest ledger under owner "submission" and rejected.
        if args.snapshot_dir and completed in snapshot_steps:
            torch.save(
                model.state_dict(),
                os.path.join(args.snapshot_dir, "state_" + str(completed) + ".pt"),
            )
        if i % 100 == 0:
            print("step " + str(i) + " train_loss " + format(train_loss, ".4f"), flush=True)

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "final.pt"))
    print("done " + str(cfg.steps) + " steps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
