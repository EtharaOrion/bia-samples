#!/usr/bin/env python3
"""Stage 3 of 3: train. FROZEN. Editing this file changes nothing that is graded.

The harness runs its OWN copy of this file. The copy delivered under
environment/pipeline/train.py exists so the agent can read exactly what the
frozen stage does; it is never the copy that produces the graded parameters.

Frozen by the task, and not free to any submission:
  - the architecture, read from environment/nanogpt_substrate.json: vocab_size
    50304, 12 layers, model_dim 768, head_dim 128, 6 heads, seq_len 1024
  - the run, 65536 tokens per step with exactly one forward and one backward
    pass over each step's batch and exactly one optimizer step per step
  - the step budget, resolved by the harness from the admin plane
  - the evaluation split, which is the verifier's own held-out FineWeb
    validation shards and is absent from every agent-visible container

The stage trains the real decoder over the shards the submission's tokenize
stage produced and writes real parameter snapshots at fixed fractions of the
step budget. The bound evaluation point is the final snapshot. There is no count
table and no cost model here: with the forward and backward passes removed this
stage produces no parameters and the graded quantity has no input at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nanogpt  # noqa: E402

BOUND_EVALUATION_POINT = nanogpt.BOUND_EVALUATION_POINT
BOUND_CHECKPOINT_ID = nanogpt.BOUND_CHECKPOINT_ID
CHECKPOINT_FRACTIONS = nanogpt.CHECKPOINT_FRACTIONS
CHECKPOINT_IDS = nanogpt.CHECKPOINT_IDS


def train(shard_paths, substrate: dict, steps: int, out_dir: Path, bound: dict, halt_at: str = "") -> dict:
    """Run the frozen decoder over the submission's shards. Harness telemetry back.

    `halt_at` is honoured because composition is free: a submission may stop the
    chain at an intermediate snapshot. Stopping is recorded, never hidden, and a
    run that never reaches the bound evaluation point has not established a loss.
    """
    arch = nanogpt.architecture(substrate)
    run = nanogpt.run_declaration(substrate)
    telemetry = nanogpt.train(
        shard_paths,
        arch,
        run,
        steps,
        Path(out_dir),
        device=str(bound["train_device"]),
        micro_rows=int(bound["micro_batch_rows"]),
        peak_lr=float(bound["peak_learning_rate"]),
        halt_at=halt_at,
    )
    telemetry["architecture"] = arch
    telemetry["run_declaration"] = run
    return telemetry
