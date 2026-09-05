"""The FROZEN training recipe. Read it; do not edit it. Edits are rejected before scoring.

Four things are frozen here and each is frozen in a way the harness measures rather
than trusts:

  the token budget   measured AS FED by the harness, per shard and in total
  the model          the canonical nanoGPT decoder declared in nanogpt_substrate.json
  the optimizer      fixed algorithm, fixed schedule, fixed hyperparameters
  the eval split     PINNED to the verifier-owned held-out FineWeb split. It is never
                     resolved from the corpus snapshot, it never rotates, and its bytes
                     are not in this container.

This script does not compute the graded number. It instantiates the frozen decoder,
trains it with exactly one forward and one backward pass per optimizer step, and writes
a real parameter snapshot at every evaluation step the harness schedules. The graded
validation loss is computed by the VERIFIER, from those snapshots, on a held-out split
this container does not carry, unsmoothed. Anything this script prints is for you and
reaches no graded path.

Delete the forward pass or the backward pass below and this script writes no parameter
snapshot at all, so the graded metric becomes UNDEFINED rather than merely different.
There is no arithmetic fallback and no tick counter standing in for a trained model.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path

import corpus_api

HERE = Path(__file__).resolve().parent
HARNESS_LOGS = Path(os.environ.get("OER11_HARNESS_LOGS", "/logs/harness"))

SUBSTRATE_FILE = HERE / "nanogpt_substrate.json"
EVAL_SCHEDULE_FILE = "eval_schedule.json"
WEIGHTS_LEDGER_FILE = "weights_ledger.json"


def _substrate() -> dict:
    """The canonical nanoGPT operating point, read from this slot's own replica.

    The verifier re-reads its own copy at grading time. Editing the copy in this
    container produces a refused run with reason frozen-axis-moved rather than a
    different grade, so there is nothing to gain by moving a number here.
    """
    with SUBSTRATE_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


SUBSTRATE = _substrate()

# --- FROZEN: the model, declared entirely by the canonical substrate ----------------
# vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len
# 1024. num_heads is model_dim // head_dim, so 768 // 128 is 6. No constant below is
# authored here; every one is read from nanogpt_substrate.json.
ARCHITECTURE = dict(SUBSTRATE["architecture"])
MODEL = {
    "architecture": "decoder-only-transformer, nanoGPT",
    "vocab_size": int(ARCHITECTURE["vocab_size"]),
    "num_layers": int(ARCHITECTURE["num_layers"]),
    "model_dim": int(ARCHITECTURE["model_dim"]),
    "head_dim": int(ARCHITECTURE["head_dim"]),
    "num_heads": int(ARCHITECTURE["num_heads"]),
    "seq_len": int(ARCHITECTURE["seq_len"]),
    "init_seed": 20260820,
    "tie_embeddings": True,
    "declaration_version": str(SUBSTRATE["declaration_version"]),
}

# --- FROZEN: the run shape ----------------------------------------------------------
# The substrate's _batch_note factorises the step as 8 * 64 * 1024 tokens: eight ranks,
# sixty-four sequences of seq_len each. Upstream runs one forward and one backward per
# rank per step. This slot's envelope is one accelerator, so the same eight shards run
# sequentially on one device: still one forward and one backward per shard, still
# exactly one optimizer step per batch. Sharding is stated here rather than left silent,
# because a silent change to how the batch is realised is a frozen-axis move.
SHARD_TOKENS = 64 * int(ARCHITECTURE["seq_len"])
RUN = {
    "batch_tokens_per_step": int(SUBSTRATE["run"]["batch_tokens_per_step"]),
    "forward_passes_per_step": int(SUBSTRATE["run"]["forward_passes_per_step"]),
    "backward_passes_per_step": int(SUBSTRATE["run"]["backward_passes_per_step"]),
    "target_val_loss": float(SUBSTRATE["run"]["target_val_loss"]),
    "shard_tokens": SHARD_TOKENS,
    "shards_per_step": int(SUBSTRATE["run"]["batch_tokens_per_step"]) // SHARD_TOKENS,
    "optimizer_steps_per_batch": 1,
}

# --- FROZEN: the optimizer ---------------------------------------------------------
OPTIMIZER = {
    "algorithm": "adamw",
    "lr": 6.0e-4,
    "betas": [0.9, 0.95],
    "weight_decay": 0.1,
    "warmup_steps": 40,
    "schedule": "cosine",
    "grad_clip": 1.0,
    "one_forward_backward_per_step": True,
}

# --- FROZEN: the evaluation contract ------------------------------------------------
# The split is PINNED. It was previously resolved by calling corpus_api.eval_split_id()
# at the current snapshot, which put the graded split on the solver-reachable surface
# and let it rotate under the run. It is now the verifier's own held-out FineWeb slice,
# named by glob and absent from every agent-visible container.
EVAL = {
    "split": "the verifier-owned held-out FineWeb split, pinned",
    "split_glob": str(SUBSTRATE["corpus"]["val_glob"]),
    "split_owner": "verifier",
    "split_present_in_environment": False,
    "split_resolution": "pinned at declaration time, never read from the corpus snapshot",
    "split_rotates_with_snapshot": False,
    "loss": "mean cross-entropy per token over the whole split, unsmoothed",
    "graded_point": "the bound evaluation step recorded by the harness",
    "sustain": "every verifier-scheduled evaluation point after the bound step",
}


def frozen_manifest() -> dict:
    """Exactly what is frozen, as a record the harness digests and the checkers read."""
    return {"model": MODEL, "run": RUN, "optimizer": OPTIMIZER, "eval": EVAL}


# -----------------------------------------------------------------------------------
# The decoder. A real parameter-carrying module at the frozen shape, not a stand-in.
# -----------------------------------------------------------------------------------
def build_model():
    """Instantiate the frozen nanoGPT decoder at the declared shape.

    Shape-bound to vocab_size, num_layers, model_dim and head_dim as declared above. If
    torch is not importable this raises, because a run with no parameters has no graded
    quantity to produce and must not fall back to arithmetic.
    """
    import torch
    from torch import nn

    dim = MODEL["model_dim"]
    heads = MODEL["num_heads"]
    head_dim = MODEL["head_dim"]
    if heads * head_dim != dim:
        raise ValueError(
            "frozen-axis-moved: num_heads * head_dim is "
            + str(heads * head_dim)
            + " and model_dim is "
            + str(dim)
        )

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_attn = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)
            self.norm_mlp = nn.LayerNorm(dim)
            self.fc = nn.Linear(dim, 4 * dim, bias=False)
            self.out = nn.Linear(4 * dim, dim, bias=False)

        def forward(self, x):
            batch, length, _ = x.shape
            h = self.norm_attn(x)
            q, k, v = self.qkv(h).split(dim, dim=2)
            q = q.view(batch, length, heads, head_dim).transpose(1, 2)
            k = k.view(batch, length, heads, head_dim).transpose(1, 2)
            v = v.view(batch, length, heads, head_dim).transpose(1, 2)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(batch, length, dim)
            x = x + self.proj(a)
            h = self.norm_mlp(x)
            return x + self.out(torch.nn.functional.gelu(self.fc(h)))

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(MODEL["vocab_size"], dim)
            self.blocks = nn.ModuleList([Block() for _ in range(MODEL["num_layers"])])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, MODEL["vocab_size"], bias=False)
            if MODEL["tie_embeddings"]:
                self.head.weight = self.embed.weight

        def forward(self, idx):
            x = self.embed(idx)
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x))

    torch.manual_seed(MODEL["init_seed"])
    model = Decoder()
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    return model


def _weights_digest(model) -> str:
    """A digest over the real parameter snapshot, computed from the saved bytes."""
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _snapshot(model, step: int, root: Path) -> str:
    """Write the parameter snapshot the verifier will grade, and return its digest."""
    import torch

    root.mkdir(parents=True, exist_ok=True)
    path = root / ("step-" + str(step) + ".pt")
    torch.save(model.state_dict(), path)
    return _weights_digest(model)


def _eval_schedule() -> dict:
    """The harness-owned evaluation schedule. Absent means there is nothing to snapshot."""
    path = HARNESS_LOGS / EVAL_SCHEDULE_FILE
    if not path.is_file():
        raise FileNotFoundError(
            "the harness wrote no " + EVAL_SCHEDULE_FILE + "; without a bound evaluation "
            "step and a sustain schedule there is no graded point to snapshot for"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(pipeline_build) -> dict:
    """Resolve the corpus, run the submitted pipeline, train, and hand back a receipt.

    The receipt is a description of what the harness observed; it is not a score and it
    carries no loss the verifier will grade. The verifier evaluates the parameter
    snapshots this function wrote, on its own held-out split, afterwards.
    """
    import torch

    corpus = corpus_api.resolve_corpus()
    budget = corpus_api.token_budget()
    stream = pipeline_build(corpus)

    token_ids = list(getattr(stream, "token_ids", []) or [])
    fed = len(token_ids)

    tokens_per_step = RUN["batch_tokens_per_step"]
    steps = fed // tokens_per_step
    if steps < 1:
        raise ValueError(
            "the token stream carries " + str(fed) + " tokens, fewer than the "
            + str(tokens_per_step) + " tokens one nanoGPT optimizer step consumes; "
            "no forward pass runs, so no parameter snapshot exists and the graded "
            "metric is undefined"
        )

    schedule = _eval_schedule()
    bound_step = int(schedule["bound_evaluation_step"])
    sustain_steps = [int(value) for value in schedule.get("scheduled_sustain_steps") or []]
    snapshot_steps = sorted({bound_step, *sustain_steps})

    seq_len = MODEL["seq_len"]
    shard_tokens = RUN["shard_tokens"]
    shards_per_step = RUN["shards_per_step"]
    rows_per_shard = shard_tokens // seq_len
    model = build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=OPTIMIZER["lr"],
        betas=tuple(OPTIMIZER["betas"]),
        weight_decay=OPTIMIZER["weight_decay"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = torch.tensor(token_ids[: steps * tokens_per_step], dtype=torch.long)
    weight_records = []
    for step in range(1, steps + 1):
        warmup = OPTIMIZER["warmup_steps"]
        if step <= warmup:
            scale = step / float(warmup)
        else:
            progress = (step - warmup) / float(max(steps - warmup, 1))
            scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimizer.param_groups:
            group["lr"] = OPTIMIZER["lr"] * scale

        optimizer.zero_grad(set_to_none=True)
        base = (step - 1) * tokens_per_step
        for shard in range(shards_per_step):
            start = base + shard * shard_tokens
            batch = data[start : start + shard_tokens].view(rows_per_shard, seq_len)
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            # EXACTLY one forward and exactly one backward pass per shard, as upstream
            # runs per rank. The gradients of the eight shards sum into one optimizer
            # step, which is the single-accelerator realisation of the frozen batch.
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, MODEL["vocab_size"]), targets.reshape(-1)
            )
            (loss / shards_per_step).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), OPTIMIZER["grad_clip"])
        optimizer.step()

        if step in snapshot_steps:
            digest = _snapshot(model, step, HARNESS_LOGS / "weights")
            weight_records.append({"step": step, "digest": digest})

    if not weight_records:
        raise ValueError(
            "the run ended at step " + str(steps) + " without reaching the bound "
            "evaluation step " + str(bound_step) + "; no parameter snapshot was written "
            "and the graded metric is undefined"
        )

    HARNESS_LOGS.mkdir(parents=True, exist_ok=True)
    (HARNESS_LOGS / WEIGHTS_LEDGER_FILE).write_text(
        json.dumps({"owner": "harness", "records": weight_records}, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    receipt = {
        "snapshot_version_at_resolve": corpus.snapshot_version,
        "snapshot_version_at_feed": corpus_api.snapshot_version(),
        "budget_tokens": budget,
        "tokens_fed": fed,
        "optimizer_steps": steps,
        "forward_passes": steps,
        "backward_passes": steps,
        "weights_snapshot_steps": [row["step"] for row in weight_records],
        "per_shard_tokens": list(getattr(stream, "per_shard_tokens", []) or []),
        "built_against": int(getattr(stream, "built_against", -1)),
        "document_ids": sorted(set(getattr(stream, "document_ids", []) or [])),
        "frozen": frozen_manifest(),
    }
    (HARNESS_LOGS / "train_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    import pipeline

    print(json.dumps(run(pipeline.build), sort_keys=True, indent=2))
