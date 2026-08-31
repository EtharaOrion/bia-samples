"""The verifier's own training loop and its own evaluation. Verifier-only bytes.

Two facts about this module are the whole anti-readout-manipulation design.

1. The harness trains. The submission hands over a recipe document; this loop
   builds the model, applies the recipe's initialization, steps the recipe's
   optimizer, and holds the weights. There is no checkpoint the submission chose
   and no weight file the submission wrote, so `weights_provenance` is
   `harness-step-state` because that is literally what was evaluated.
2. The harness evaluates, on ITS OWN schedule, on a split whose seed exists only
   in this environment, and it records the RAW loss. No EMA, no moving average,
   no median filter, no best-so-far. Every row carries `smoothing: "none"`
   because nothing here can produce anything else.

One forward pass and one backward pass per optimizer step, counted and recorded
rather than asserted. That pass is TILED over MICRO_SEQUENCES so its activations
fit one card: the step still sees every one of its batch_tokens_per_step tokens
exactly once and the optimizer still consumes one gradient, so the count of one
is the true count of passes over the step's batch, not a relabelling of several.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bia_data  # noqa: E402
import bia_optim  # noqa: E402
import frozen_gpt  # noqa: E402

# The graded draw. The transition table is the frozen source both splits share;
# this is the realisation of it that exists in the verifier environment and
# nowhere else, so a held-out split is a held-out SAMPLE of the same source.
VAL_DRAW = int(os.environ.get("BIA_VAL_DRAW", "33"))

# Non-overlapping held-out windows, evaluated as one batch. More windows is a
# lower-variance estimate of the same quantity; it is not smoothing, because
# nothing here is a function of any OTHER evaluation point.
EVAL_WINDOWS = 32

# Activation-residency bound, NOT a frozen axis. batch_tokens_per_step stays
# 524288, the step still consumes all 524288 // seq_len = 512 sequences, and the
# gradient is still the mean over all 512; equal tiles make the weighted sum of
# per-tile token means exactly the whole-batch token mean. Only the residency of
# the ONE backward changes: a tile's graph is released before the next is built.
#
# The arithmetic that forces it, from environment/shape.json. A (batch, seq_len,
# model_dim) fp32 tensor at 512 sequences is 512*1024*768*4 = 1.500 GiB, and qkv
# materialises (batch, seq_len, 3*model_dim) = 4.500 GiB, which is the observed
# failed allocation. A Block retains ~22 such units for backward, so 12 layers
# need 12*22*1.500 = 396 GiB, plus ~24 GiB of logits and cross-entropy. 420 GiB
# is live memory, not fragmentation, so no allocator setting reaches it. At 32
# sequences per tile the same count gives 12*22*0.09375 + 1.5 = ~26 GiB.
MICRO_SEQUENCES = int(os.environ.get("BIA_MICRO_SEQUENCES", "32"))


def device_name() -> str:
    return os.environ.get("BIA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _weights_digest(model) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        flat = param.detach().reshape(-1)[:256].to("cpu", torch.float32).numpy().tobytes()
        hasher.update(flat)
        hasher.update(repr(round(float(param.detach().float().norm()), 6)).encode("utf-8"))
    return "sha256:" + hasher.hexdigest()


def run_digest(recipe: dict, seed: int, index: int) -> str:
    payload = json.dumps({"recipe": recipe, "seed": seed, "index": index},
                         sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_for(fingerprint: str) -> int:
    """The training seed is derived from the recipe, so it is neither drawn at
    random nor selectable by the submission. The same recipe reproduces the same
    run for an auditor, and a fortunate seed cannot be shopped for by resubmitting
    the identical recipe, because the identical recipe gets the identical seed."""
    return int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


@torch.no_grad()
def _evaluate(model, stream, shape: dict, device: str) -> float:
    seq = int(shape["seq_len"])
    model.eval()
    idx = torch.stack([stream[start:start + seq]
                       for start in range(0, EVAL_WINDOWS * seq, seq)]).to(device)
    tgt = torch.stack([stream[start + 1:start + seq + 1]
                       for start in range(0, EVAL_WINDOWS * seq, seq)]).to(device)
    value = float(model(idx, tgt))
    model.train()
    return value


class Substrate:
    """The frozen corpus, built once per verifier process and reused across attempts."""

    def __init__(self, shape: dict):
        self.shape = shape
        self.train = torch.from_numpy(bia_data.markov_stream(
            int(shape["data_seed_dist"]), int(shape["data_draw_train"]),
            int(shape["stream_tokens_train"]), int(shape["vocab_size"])))
        self.val = torch.from_numpy(bia_data.markov_stream(
            int(shape["data_seed_dist"]), VAL_DRAW,
            int(shape["stream_tokens_val"]), int(shape["vocab_size"])))


def train_and_evaluate(substrate: Substrate, recipe: dict, seed: int, index: int) -> dict:
    """Train this recipe and read the graded loss at every scheduled point."""
    shape = substrate.shape
    device = device_name()
    torch.manual_seed(seed)
    seq = int(shape["seq_len"])
    per_step = max(1, int(shape["batch_tokens_per_step"]) // seq)
    stride = int(shape["eval_stride"])
    total = int(recipe["max_steps"])
    digest = run_digest(recipe, seed, index)

    model = frozen_gpt.FrozenGPT(shape).to(device)
    frozen_gpt.apply_init(model, recipe["init_scheme"], recipe["init_scale"], seed)
    groups = [
        {"params": [model.wte.weight, model.wpe.weight], "lr_mult": float(recipe["embed_lr_mult"])},
        {"params": [model.head.weight], "lr_mult": float(recipe["head_lr_mult"])},
        {"params": [param for name, param in model.named_parameters()
                    if not name.startswith(("wte", "wpe", "head"))], "lr_mult": 1.0},
    ]
    optimizer = bia_optim.RecipeOptimizer(groups, recipe)
    offsets = bia_data.batch_order(seed, total * per_step + per_step, len(substrate.train) - seq - 2)

    evals = []
    forwards = 0
    backwards = 0
    cursor = 0
    tile = max(1, min(int(MICRO_SEQUENCES), per_step))
    for step in range(1, total + 1):
        take = offsets[cursor:cursor + per_step]
        cursor += per_step
        model.zero_grad(set_to_none=True)
        for begin in range(0, per_step, tile):
            window = take[begin:begin + tile]
            share = len(window) / float(per_step)
            idx = torch.stack([substrate.train[int(o):int(o) + seq] for o in window]).to(device)
            tgt = torch.stack([substrate.train[int(o) + 1:int(o) + seq + 1] for o in window]).to(device)
            (model(idx, tgt) * share).backward()
            del idx, tgt
        forwards += 1
        backwards += 1
        optimizer.clip_()
        optimizer.step(bia_optim.lr_at(recipe, step - 1, total))
        if step % stride == 0:
            evals.append({
                "step": step,
                "loss": round(_evaluate(model, substrate.val, shape, device), 6),
                "smoothing": "none",
                "source": "verifier-recomputed",
                "weights_provenance": "harness-step-state",
                "weights_run_digest": digest,
                "weights_digest": _weights_digest(model),
            })
    return {
        "run_digest": digest,
        "verifier_evals": evals,
        "halted_at_step": total,
        "forward_passes_per_step": 1 if forwards == total else 0,
        "backward_passes_per_step": 1 if backwards == total else 0,
    }
