#!/usr/bin/env python3
"""Stage 3 of 3: train. FROZEN. Editing this file changes nothing that is graded.

The harness runs its OWN copy of this file. The copy delivered under
environment/pipeline/train.py exists so the agent can read exactly what the frozen
stage does; it is never the copy that produces the graded weights.

Frozen by the task, and not free to any submission:
  - the token budget, counted in tokens and fed by the harness
  - the model, an interpolated bigram over the submitted vocabulary
  - the optimizer, a single streaming pass of count accumulation with a fixed
    add-k and a fixed interpolation weight
  - the evaluation split

The optimizer takes exactly one pass over exactly the bound number of tokens, in
the order the token stream carries them. Checkpoints are dumped at fixed fractions
of the budget. The bound evaluation point is the final checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "oer13.model/v1"

# Frozen model and optimizer constants. Not free.
ADD_K = 0.1
LAMBDA_BIGRAM = 0.7
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
CHECKPOINT_IDS = ("c1", "c2", "c3", "c4")
BOUND_EVALUATION_POINT = "final-checkpoint"
BOUND_CHECKPOINT_ID = "c4"


def _dump(path: Path, payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def train(stream, vocab_size: int, budget: int, out_dir: Path, halt_at: str = "") -> dict:
    """One streaming pass over exactly `budget` tokens. Returns harness telemetry.

    `halt_at` is honoured because composition is free: a submission may stop the
    chain at an intermediate checkpoint. Stopping is recorded, never hidden, and
    a run that never reaches the bound evaluation point has not established a
    loss.
    """
    marks = {}
    for ident, fraction in zip(CHECKPOINT_IDS, CHECKPOINT_FRACTIONS):
        marks[int(round(budget * fraction))] = ident

    unigram: dict[int, int] = {}
    bigram: dict[str, int] = {}
    previous = 0
    consumed = 0
    checkpoints = []
    halted_at_tokens = None

    fed = list(stream[:budget])
    for token in fed:
        unigram[token] = unigram.get(token, 0) + 1
        key = str(previous) + ":" + str(token)
        bigram[key] = bigram.get(key, 0) + 1
        previous = token
        consumed += 1
        ident = marks.get(consumed)
        if ident is None:
            continue
        path = out_dir / ("model_" + ident + ".json")
        digest = _dump(
            path,
            {
                "schema": SCHEMA,
                "checkpoint": ident,
                "tokens_consumed": consumed,
                "vocab_size": vocab_size,
                "add_k": ADD_K,
                "lambda_bigram": LAMBDA_BIGRAM,
                "total": consumed,
                "unigram": {str(k): v for k, v in sorted(unigram.items())},
                "bigram": dict(sorted(bigram.items())),
            },
        )
        checkpoints.append(
            {
                "id": ident,
                "tokens": consumed,
                "path": path.as_posix(),
                "model_sha256": digest,
            }
        )
        if halt_at and ident == halt_at:
            halted_at_tokens = consumed
            break

    reached = any(row["id"] == BOUND_CHECKPOINT_ID for row in checkpoints)
    return {
        "tokens_available": len(stream),
        "tokens_consumed": consumed,
        "bound_budget": budget,
        "checkpoints": checkpoints,
        "bound_evaluation_point": BOUND_EVALUATION_POINT,
        "bound_checkpoint_id": BOUND_CHECKPOINT_ID,
        "bound_point_reached": bool(reached),
        "halted_at_tokens": halted_at_tokens,
        "add_k": ADD_K,
        "lambda_bigram": LAMBDA_BIGRAM,
    }
