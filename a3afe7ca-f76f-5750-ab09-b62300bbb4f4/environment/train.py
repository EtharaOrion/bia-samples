"""The FROZEN training recipe. Read it; do not edit it. Edits are rejected before scoring.

Four things are frozen here and each is frozen in a way the harness measures rather
than trusts:

  the token budget   measured AS FED by the harness, per shard and in total
  the model          fixed architecture, fixed initialisation seed, fixed width/depth
  the optimizer      fixed algorithm, fixed schedule, fixed hyperparameters
  the eval split     resolved from the CURRENT snapshot, never from a cached copy

This script does not compute the graded number. It trains, and it hands the harness the
weights at the bound evaluation step and at every verifier-scheduled point after it. The
graded validation loss is computed by the VERIFIER, from those weights, on the frozen
split, unsmoothed. Anything this script prints is for you and reaches no graded path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import corpus_api

HARNESS_LOGS = Path(os.environ.get("OER11_HARNESS_LOGS", "/logs/harness"))

# --- FROZEN: the model -------------------------------------------------------------
MODEL = {
    "architecture": "decoder-only-transformer",
    "layers": 12,
    "d_model": 768,
    "heads": 12,
    "context": 1024,
    "init_seed": 20260820,
    "tie_embeddings": True,
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
EVAL = {
    "split": "resolved from corpus_api.eval_split_id() at the CURRENT snapshot",
    "loss": "mean cross-entropy per token over the whole split, unsmoothed",
    "graded_point": "the bound evaluation step recorded by the harness",
    "sustain": "every verifier-scheduled evaluation point after the bound step",
}


def frozen_manifest() -> dict:
    """Exactly what is frozen, as a record the harness digests and the checkers read."""
    return {"model": MODEL, "optimizer": OPTIMIZER, "eval": EVAL}


def run(pipeline_build) -> dict:
    """Resolve the corpus, run the submitted pipeline, train, and hand back a receipt.

    The receipt is a description of what the harness observed; it is not a score and it
    carries no loss the verifier will grade. The verifier evaluates the harness's own
    weights afterwards.
    """
    corpus = corpus_api.resolve_corpus()
    budget = corpus_api.token_budget()
    stream = pipeline_build(corpus)

    fed = len(getattr(stream, "token_ids", []) or [])
    receipt = {
        "snapshot_version_at_resolve": corpus.snapshot_version,
        "snapshot_version_at_feed": corpus_api.snapshot_version(),
        "budget_tokens": budget,
        "tokens_fed": fed,
        "per_shard_tokens": list(getattr(stream, "per_shard_tokens", []) or []),
        "built_against": int(getattr(stream, "built_against", -1)),
        "document_ids": sorted(set(getattr(stream, "document_ids", []) or [])),
        "frozen": frozen_manifest(),
    }
    HARNESS_LOGS.mkdir(parents=True, exist_ok=True)
    (HARNESS_LOGS / "train_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    import pipeline

    print(json.dumps(run(pipeline.build), sort_keys=True, indent=2))
