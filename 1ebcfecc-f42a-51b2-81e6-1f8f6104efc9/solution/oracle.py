"""Reference solution. Private oracle, never agent-visible.

Structurally identical to environment/train_baseline.py. What differs is the
Optimization section, which is where the reference improvement lives.

THE REFERENCE SETTINGS ARE NOT YET AUTHORED. They cannot be, honestly: a
reference solution is defined by the training-token count it reaches the
target on, and that number does not exist until the scaled operating point is
measured. The batch-size schedule below therefore still matches the baseline,
so this file is a complete, runnable scaffold rather than a claimed oracle,
and solve.sh refuses to report success while that is true.
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


def load_shape() -> dict:
    shape = json.loads(SHAPE_PATH.read_text())
    missing = sorted(k for k, v in shape.items() if not k.startswith("_") and v is None)
    if missing:
        sys.exit(
            f"shape is unbound; these are null and must be measured first: {missing}. "
            "See environment/shape.json for how to bind them."
        )
    return shape


def build_model(shape: dict):
    """The frozen architecture. Changing anything here fails verifier-side.

    The verifier loads your checkpoints into its own copy of this definition with
    strict shape checking, so a changed depth, width, head dimension or vocabulary
    does not produce a worse score, it produces no score at all.
    """
    from frozen_gpt import GPT  # frozen, ships in the image

    return GPT(
        vocab_size=shape["vocab_size"],
        num_layers=shape["num_layers"],
        model_dim=shape["model_dim"],
        head_dim=shape["head_dim"],
    ).cuda()


# ---------------------------------------------------------------------------
# Optimization. This section is yours.
# ---------------------------------------------------------------------------
def build_optimizer(model, shape: dict):
    """Reference optimizer. Replace it, retune it, or schedule it differently."""
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))


def batch_size_schedule(tokens_served: int, shape: dict) -> int:
    """Tokens per step. Constant here, and this is the obvious thing to change.

    Training tokens equal steps times batch size, so a smaller batch spends fewer
    tokens per step but needs more steps to reach the same loss. Somewhere between
    those two failure modes is a schedule that reaches the target on fewer tokens
    than a constant batch does. Finding it is the task.
    """
    return shape["baseline_batch_size_tokens"]


def lr_schedule(optimizer, tokens_served: int, shape: dict) -> None:
    """Flat, then linear decay over the last 30 percent of the token budget."""
    budget = shape["train_token_budget"]
    frac = tokens_served / budget
    if frac > 0.7:
        scale = max(0.0, (1.0 - frac) / 0.3)
        for group in optimizer.param_groups:
            group["lr"] = group.get("initial_lr", 3e-4) * scale


# ---------------------------------------------------------------------------
def main() -> int:
    shape = load_shape()
    seed = int(os.environ["BIA_SEED"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
