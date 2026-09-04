#!/usr/bin/env python3
"""The frozen trainer. Read it. Do not modify it. Every axis it names is frozen.

It instantiates the nanoGPT decoder declared in `nanogpt_substrate.json` and trains it on
the FineWeb10B training shards your curation recipe retained. The architecture, the batch
and the one-forward-one-backward-pass-per-step rule are read from that declaration rather
than restated here, so the model this process trains is the substrate's model and not a
local copy of its numbers.

    token budget      3200 steps at 524288 tokens per step, counted AS FED by this process
    model             the substrate decoder, 12 layers, 768 model dim, 6 heads of 128
    optimizer         frozen-adamw-cosine
    evaluation split  val-split-b, verifier-owned, never staged here and never fed

What is free is the data mixture and the curation recipe, and nothing else.

This process owns the parameter snapshots the verifier grades, and it writes them where
the submission cannot reach:

  * it counts tokens as they are fed, so an overspend is visible even when the curation
    plan declared the frozen budget.
  * it writes a real parameter snapshot at every evaluation point and records its digest,
    so the weights the verifier evaluates are provably the weights this run produced at
    that step rather than a checkpoint the submission selected.
  * it does NOT compute the graded loss. It has no copy of the held-out split, so it
    cannot. Whatever loss it prints is training telemetry, and the number that scores a
    submission is the verifier's own evaluation of these snapshots on a split this
    container never sees.
  * it records `halted_at_step` and `halt_reason`, so a run that stopped on a favourable
    training loss is graded as not having established the level, with a reason.
  * it trains the frozen default mixture as a control arm and the harness-owned reference
    recipe as a floor, both under the identical freeze, so the run-local ladder's two ends
    are measured beside the graded arm rather than carried in from another run.

The run record it writes is `/telemetry/run_record.json` and the snapshots go to
`/telemetry/checkpoints/`. Both paths are outside the agent's writable surface.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBSTRATE_PATH = HERE / "nanogpt_substrate.json"
RAW_POOL_PATH = HERE / "raw_pool.json"

RUN_RECORD_PATH = "/telemetry/run_record.json"
CHECKPOINT_DIR = "/telemetry/checkpoints"
SHARD_ROOT = "/workspace/data/fineweb10B"

STEPS_AT_BUDGET = 3200
BOUND_EVAL_STEP = 2800
SUSTAIN_POINTS = (2800, 3000, 3200)
SUSTAIN_TOLERANCE = 0.02

OPTIMIZER = "frozen-adamw-cosine"
LEARNING_RATE = 0.0018
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 250
BETAS = (0.9, 0.95)

EVALUATION_SPLIT_ID = "val-split-b"
GRADED_SAMPLE_SOURCE = "verifier-recompute"
SUBMISSION_SAMPLE_SOURCE = "submission-reported"


def substrate():
    """The canonical operating point. Every frozen number below is read from here."""
    return json.loads(SUBSTRATE_PATH.read_text(encoding="utf-8"))


def frozen_axes():
    """The frozen axes, so a submission can read them and never has to guess them."""
    spec = substrate()
    architecture = spec["architecture"]
    run = spec["run"]
    tokens_per_step = int(run["batch_tokens_per_step"])
    return {
        "substrate_declaration_version": spec["declaration_version"],
        "vocab_size": int(architecture["vocab_size"]),
        "num_layers": int(architecture["num_layers"]),
        "model_dim": int(architecture["model_dim"]),
        "head_dim": int(architecture["head_dim"]),
        "num_heads": int(architecture["num_heads"]),
        "seq_len": int(architecture["seq_len"]),
        "tokens_per_step": tokens_per_step,
        "forward_passes_per_step": int(run["forward_passes_per_step"]),
        "backward_passes_per_step": int(run["backward_passes_per_step"]),
        "steps_at_budget": STEPS_AT_BUDGET,
        "token_budget_tokens": tokens_per_step * STEPS_AT_BUDGET,
        "train_glob": spec["corpus"]["train_glob"],
        "optimizer": OPTIMIZER,
        "evaluation_split_id": EVALUATION_SPLIT_ID,
        "evaluation_split_owner": "verifier",
        "bound_eval_step": BOUND_EVAL_STEP,
        "sustain_points": list(SUSTAIN_POINTS),
        "sustain_tolerance": SUSTAIN_TOLERANCE,
    }


def token_budget():
    return int(substrate()["run"]["batch_tokens_per_step"]) * STEPS_AT_BUDGET


def token_cost(steps):
    """Tokens a run of this many steps feeds. The budget is enforced against this."""
    return int(steps) * int(substrate()["run"]["batch_tokens_per_step"])


def steps_within_budget(tokens=None):
    """The largest step count the frozen budget pays for."""
    per_step = int(substrate()["run"]["batch_tokens_per_step"])
    return int(token_budget() if tokens is None else tokens) // per_step


# ---------------------------------------------------------------------------
# The decoder. Structurally the substrate's, instantiated from its declaration.
# ---------------------------------------------------------------------------


def build_model(device="cuda"):
    """Instantiate the substrate decoder. Shape-bound to the declared architecture."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    architecture = substrate()["architecture"]
    vocab_size = int(architecture["vocab_size"])
    num_layers = int(architecture["num_layers"])
    model_dim = int(architecture["model_dim"])
    head_dim = int(architecture["head_dim"])

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gains = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

    class Rotary(nn.Module):
        def __init__(self, dim):
            super().__init__()
            angular_freq = (1 / 1024) ** torch.linspace(
                0, 1, steps=dim // 4, dtype=torch.float32
            )
            self.register_buffer(
                "angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
            )

        def forward(self, x):
            pos = torch.arange(x.size(1), dtype=torch.float32, device=x.device)
            theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
            cos, sin = theta.cos(), theta.sin()
            x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
            return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), 3).type_as(x)

    class CausalSelfAttention(nn.Module):
        def __init__(self, dim, head_width):
            super().__init__()
            self.num_heads = dim // head_width
            self.head_dim = head_width
            hdim = self.num_heads * self.head_dim
            self.q = nn.Linear(dim, hdim)
            self.k = nn.Linear(dim, hdim)
            self.v = nn.Linear(dim, hdim)
            self.proj = nn.Linear(hdim, dim)
            self.rotary = Rotary(head_width)

        def forward(self, x):
            B, T = x.size(0), x.size(1)
            q = self.q(x).view(B, T, self.num_heads, self.head_dim)
            k = self.k(x).view(B, T, self.num_heads, self.head_dim)
            v = self.v(x).view(B, T, self.num_heads, self.head_dim)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
            q, k = self.rotary(q), self.rotary(k)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=0.12, is_causal=True,
            ).transpose(1, 2)
            return self.proj(y.contiguous().view(B, T, self.num_heads * self.head_dim))

    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, 4 * dim)
            self.proj = nn.Linear(4 * dim, dim)

        def forward(self, x):
            return self.proj(self.fc(x).relu().square())

    class Block(nn.Module):
        def __init__(self, dim, head_width):
            super().__init__()
            self.attn = CausalSelfAttention(dim, head_width)
            self.mlp = MLP(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, model_dim)
            self.blocks = nn.ModuleList(
                [Block(model_dim, head_dim) for _ in range(num_layers)]
            )
            self.proj = nn.Linear(model_dim, vocab_size)
            self.norm1 = RMSNorm(model_dim)
            self.norm2 = RMSNorm(model_dim)

        def forward(self, inputs, targets):
            x = self.norm1(self.embed(inputs))
            for block in self.blocks:
                x = block(x)
            logits = self.proj(self.norm2(x)).float()
            logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
            return F.cross_entropy(
                logits.view(targets.numel(), -1), targets.view(-1), reduction="mean"
            )

    return GPT().to(device)


# ---------------------------------------------------------------------------
# The curated corpus. What the recipe retained, resolved to real shard spans.
# ---------------------------------------------------------------------------


def load_shard(path):
    """Read one FineWeb10B .bin shard: 256 int32 of header, then uint16 token ids."""
    import numpy as np

    with open(path, "rb") as handle:
        header = np.frombuffer(handle.read(1024), dtype=np.int32)
        if int(header[0]) != 20240520:
            raise SystemExit("magic number mismatch in " + str(path))
        if int(header[1]) != 1:
            raise SystemExit("unsupported shard version in " + str(path))
        return np.frombuffer(handle.read(), dtype=np.uint16)


def curate(weight_fn, shard_root=SHARD_ROOT):
    """Apply the recipe to the raw pool and resolve what it retained to shard spans.

    A span is retained when its quantised weight is at least 1, exactly the quantisation
    the fingerprint screen used, so the corpus that is fed and the behaviour that was
    screened are the same function of the same recipe.
    """
    rows = json.loads(RAW_POOL_PATH.read_text(encoding="utf-8"))["docs"]
    retained = []
    for row in rows:
        quantised = max(0, min(8, int(round(float(weight_fn(row)) * 8.0))))
        if quantised < 1:
            continue
        retained.append(
            {
                "id": row["id"],
                "shard": row["shard"],
                "token_offset": int(row["token_offset"]),
                "span_tokens": int(row["span_tokens"]),
                "quantised_weight": quantised,
                "path": str(Path(shard_root) / row["shard"]),
            }
        )
    return retained


def stream(retained, tokens_per_step, seq_len, seed=0):
    """Yield one frozen batch per step from the retained spans, weight-proportionally.

    Sampling is weight-proportional over the retained spans, so the mixture the recipe
    declared is the mixture that reaches the model. Exactly tokens_per_step tokens leave
    this generator per step, which is what makes the budget countable as fed.
    """
    import numpy as np
    import torch

    if not retained:
        raise SystemExit("the curation recipe retained no span, so there is nothing to feed")
    rng = np.random.default_rng(seed)
    mass = np.array([row["quantised_weight"] for row in retained], dtype=np.float64)
    mass = mass / mass.sum()
    cache = {}
    rows_per_batch = tokens_per_step // seq_len
    while True:
        inputs = np.empty((rows_per_batch, seq_len), dtype=np.int64)
        targets = np.empty((rows_per_batch, seq_len), dtype=np.int64)
        for row_index in range(rows_per_batch):
            span = retained[int(rng.choice(len(retained), p=mass))]
            if span["path"] not in cache:
                cache[span["path"]] = load_shard(span["path"])
            tokens = cache[span["path"]]
            start = span["token_offset"] + int(
                rng.integers(0, max(1, span["span_tokens"] - seq_len - 1))
            )
            window = tokens[start:start + seq_len + 1].astype(np.int64)
            if window.shape[0] < seq_len + 1:
                raise SystemExit("span " + span["id"] + " runs past the end of " + span["shard"])
            inputs[row_index] = window[:-1]
            targets[row_index] = window[1:]
        yield (
            torch.from_numpy(inputs).cuda(non_blocking=True),
            torch.from_numpy(targets).cuda(non_blocking=True),
        )


# ---------------------------------------------------------------------------
# The run. One forward and one backward pass per step, and a snapshot at each point.
# ---------------------------------------------------------------------------


def schedule(step, total_steps):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def snapshot(model, step, directory=CHECKPOINT_DIR):
    """Write the parameter snapshot for this step and return its record.

    The digest is over the snapshot bytes, so the verifier can prove the parameters it
    evaluated are the parameters this run produced at this step.
    """
    import torch

    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / ("step_" + str(int(step)) + ".pt")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"step": int(step), "digest": digest, "owner": "harness", "path": str(path)}


def train_arm(weight_fn, arm, seed=0):
    """Train one arm to the frozen budget. Returns its checkpoints and training telemetry."""
    import torch

    spec = substrate()
    seq_len = int(spec["architecture"]["seq_len"])
    tokens_per_step = int(spec["run"]["batch_tokens_per_step"])
    retained = curate(weight_fn)
    model = build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    batches = stream(retained, tokens_per_step, seq_len, seed=seed)

    checkpoints = []
    tokens_fed = 0
    last_training_loss = None
    for step in range(1, STEPS_AT_BUDGET + 1):
        inputs, targets = next(batches)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * schedule(step - 1, STEPS_AT_BUDGET)
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_fed += tokens_per_step
        last_training_loss = float(loss.detach())
        if step in SUSTAIN_POINTS and arm == "graded":
            checkpoints.append(snapshot(model, step))

    if arm != "graded":
        checkpoints.append(snapshot(model, STEPS_AT_BUDGET, CHECKPOINT_DIR + "/" + arm))
    return {
        "arm": arm,
        "checkpoints": checkpoints,
        "retained": retained,
        "training": {
            "steps_executed": STEPS_AT_BUDGET,
            "tokens_fed": tokens_fed,
            "frozen_token_budget": token_budget(),
            "halted_at_step": STEPS_AT_BUDGET,
            "halt_reason": "budget-exhausted",
            "final_training_loss": last_training_loss,
        },
    }


def load_submission(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv):
    submission = argv[0] if argv else os.environ.get(
        "OER10_SUBMISSION", "/workspace/submission/recipe.py"
    )
    reference = HERE.parent.parent / "tests" / "reference_floor.py"

    graded = train_arm(load_submission(submission).weight, "graded")
    control = train_arm(lambda doc: 1.0, "control")
    floor = train_arm(load_submission(reference).weight, "floor") if reference.is_file() else None

    record = {
        "schema": "oer10.run_record/v1",
        "substrate_declaration_version": substrate()["declaration_version"],
        "frozen_axes": frozen_axes(),
        "bound_held_out_split_id": EVALUATION_SPLIT_ID,
        "graded_loss_is_not_in_this_record": (
            "This process holds no copy of the held-out split and computes no graded loss. "
            "The verifier evaluates the snapshots below on its own split."
        ),
        "training": graded["training"],
        "fed_doc_ids": [row["id"] for row in graded["retained"]],
        "checkpoints": graded["checkpoints"],
        "arms": {
            "control": control["checkpoints"],
            "floor": floor["checkpoints"] if floor else [],
        },
        "evaluation": {
            "bound_eval_step": BOUND_EVAL_STEP,
            "sustain_points": list(SUSTAIN_POINTS),
            "sustain_tolerance": SUSTAIN_TOLERANCE,
            "samples": [],
            "samples_note": (
                "Empty by construction. The verifier appends its own "
                + GRADED_SAMPLE_SOURCE
                + " samples; anything this container reported would travel as "
                + SUBMISSION_SAMPLE_SOURCE
                + " and is never read on the graded path."
            ),
        },
    }
    Path(RUN_RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(RUN_RECORD_PATH).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print("run record written to " + RUN_RECORD_PATH)
    print("snapshots written to " + CHECKPOINT_DIR + ", graded by the verifier and not here")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
