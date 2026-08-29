"""Reference solution. Private oracle, never agent-visible.

Structurally identical to environment/train_baseline.py. What differs is the
Optimization section, which is where the reference improvement lives.

The reference change is batch-size warmup, in batch_size_schedule below. It is
a real, cited, mechanically motivated change rather than a placeholder, and it
targets the graded quantity directly: tokens spent in an oversized early batch
buy less progress than the same tokens spent in a right-sized one.

ITS EFFECT IS UNVERIFIED. No GPU was reachable when this was authored, so the
oracle has never been run and has never been compared against the baseline
through the live checkers. Until that comparison exists, anchors.json stays
measured false and solve.sh exits non-zero, because a reference solution is
defined by the training-token count it reaches the target on and that number
does not exist yet.
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
    """THE REFERENCE CHANGE: batch-size warmup instead of a constant batch.

    This is the one substantive difference between this file and
    environment/train_baseline.py, and it is the reason this is a reference
    solution rather than a copy of the baseline.

    Why it should reach the target on fewer training tokens. The critical batch
    size, the batch beyond which more data parallelism stops buying proportional
    progress, is not constant over a run. Merrill et al., arXiv 2505.23971,
    measure it directly on OLMo and find it sits near zero at initialization,
    rises rapidly, then plateaus. A batch chosen for the plateau is therefore far
    too large early, and every token in an oversized early batch is buying less
    progress than the same token would buy in a smaller one. Since the graded
    quantity here is tokens to target rather than steps to target, that waste is
    charged directly to the score. The same paper reports reaching slightly better
    loss with 43 percent fewer gradient steps by warming the batch size up, which
    is the effect this schedule is reaching for.

    The shape of the ramp follows their measured curve: start at an eighth of the
    baseline batch, double it at each of three thresholds spread over the first
    forty percent of the budget, then hold at the baseline batch for the rest.
    Doubling rather than ramping linearly keeps every batch a power-of-two
    multiple of the sequence length, so no step is a ragged partial batch.

    UNVERIFIED. No GPU was reachable when this was authored, so the effect has not
    been measured on this shape. The mechanism is real and cited; the magnitude on
    a 6-layer 384-wide model at 60M tokens is not known, and anchors.json stays
    measured false until it is.
    """
    full = shape["baseline_batch_size_tokens"]
    budget = shape["train_token_budget"]
    frac = tokens_served / budget
    if frac < 0.10:
        return full // 8
    if frac < 0.20:
        return full // 4
    if frac < 0.40:
        return full // 2
    return full


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
