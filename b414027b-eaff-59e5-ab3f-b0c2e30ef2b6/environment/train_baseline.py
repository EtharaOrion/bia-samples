#!/usr/bin/env python3
"""Baseline training script. This is the thing you are asked to beat.

It reaches the target on some number of training tokens. Your job is to reach the
same target on fewer. Everything in the Optimization section below is yours to
change; everything above it is frozen and enforced rather than requested.

Read the loader contract first. You do not write checkpoints and you do not count
tokens. `bia_loader.Loader` does both, because both are graded and a graded number
produced inside a process you authored is not evidence. Ask it for batches of
whatever size you like, including a size that changes over training, and it will
serve exactly that many tokens and record them.

This script refuses to run while environment/shape.json carries nulls. That is
deliberate: the scaled operating point has not been measured, and a script that
ran anyway would be training a shape nobody chose.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch

from bia_loader import Loader

SHAPE_PATH = pathlib.Path(os.environ.get("BIA_SHAPE", "/env/shape.json"))
TRAIN_SHARDS = os.environ.get("BIA_TRAIN_SHARDS", "/data/train_*.bin")


STRUCTURAL = (
    "vocab_size", "num_layers", "model_dim", "head_dim", "seq_len",
    "train_token_budget", "baseline_batch_size_tokens",
)


def load_shape() -> dict:
    """Structural keys gate the run. target_loss deliberately does not.

    The baseline has to be runnable in order to discover where it lands, and
    where it lands is what target_loss will be set to. Gating the run on
    target_loss would make the measurement that produces it impossible.
    """
    shape = json.loads(SHAPE_PATH.read_text())
    missing = sorted(k for k in STRUCTURAL if shape.get(k) is None)
    if missing:
        sys.exit(
            f"shape is structurally unbound; these are null: {missing}. "
            "See environment/shape.json for how to bind them."
        )
    return shape


def build_model(shape: dict):
    """The frozen architecture. Changing anything here fails verifier-side.

    The verifier loads your checkpoints into its own copy of this definition with
    strict shape checking, so a changed depth, width, head dimension or vocabulary
    does not produce a worse score, it produces no score at all.
    """
    from frozen_gpt import GPT, resolve_device  # frozen, ships in the image

    return GPT(
        vocab_size=shape["vocab_size"],
        num_layers=shape["num_layers"],
        model_dim=shape["model_dim"],
        head_dim=shape["head_dim"],
    ).to(resolve_device())


# ---------------------------------------------------------------------------
# Optimization. This section is yours.
# ---------------------------------------------------------------------------
def build_optimizer(model, shape: dict):
    """Reference optimizer. Replace it, retune it, or schedule it differently."""
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))


def batch_size_schedule(tokens_served: int, shape: dict) -> int:
    """Tokens per step. Constant here, and this is the obvious thing to change.

    Training tokens equal steps times batch size, so a smaller batch spends fewer
    tokens per step but needs more steps to reach the same loss. Somewhere between
    those two failure modes is a schedule that reaches the target on fewer tokens
    than a constant batch does. Finding it is the task.
    """
    return shape["baseline_batch_size_tokens"]


def lr_schedule(optimizer, tokens_served: int, shape: dict) -> None:
    """Warm up over the first 10 percent, flat, then decay over the last 30.

    The warmup is not optional at this scale. Without it the first few steps on a
    fresh initialization push the loss up faster than the schedule can pull it
    back, and the run never recovers. It is applied identically in the baseline
    and the reference solution, so it is not a differentiator.
    """
    budget = shape["train_token_budget"]
    frac = tokens_served / budget
    if frac < 0.10:
        scale = max(0.02, frac / 0.10)
    elif frac > 0.70:
        scale = max(0.0, (1.0 - frac) / 0.30)
    else:
        scale = 1.0
    for group in optimizer.param_groups:
        group["lr"] = group.get("initial_lr", 1e-3) * scale


# ---------------------------------------------------------------------------
def main() -> int:
    shape = load_shape()
    seed = int(os.environ["BIA_SEED"])
    torch.manual_seed(seed)
    torch.manual_seed(seed)

    model = build_model(shape)
    optimizer = build_optimizer(model, shape)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    loader = Loader(TRAIN_SHARDS, seq_len=shape["seq_len"])

    while True:
        bs = batch_size_schedule(loader.tokens_served, shape)
        served = False
        for inputs, targets in loader.batches(bs):
            served = True
            lr_schedule(optimizer, loader.tokens_served, shape)
            loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # The loader decides when a checkpoint is due, not this script.
            loader.maybe_checkpoint(model)
            break
        if not served:
            break
        if loader.tokens_served >= shape["train_token_budget"]:
            break

    loader.maybe_checkpoint(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
