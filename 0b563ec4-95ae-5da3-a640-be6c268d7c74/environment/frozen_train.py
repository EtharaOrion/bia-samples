#!/usr/bin/env python3
"""The frozen trainer. Read it. Do not modify it. Every axis it names is frozen.

    token budget      419430400 tokens, counted AS FED by this process
    model             frozen-decoder-12L-768d
    optimizer         frozen-adamw-cosine
    evaluation split  val-split-b, 60 documents, never fed

What is free is the data mixture and the curation recipe, and nothing else.

This process owns the numbers the verifier grades, and it writes them where the submission
cannot reach:

  * it counts tokens as they are fed, so an overspend is visible even when the curation
    plan declared the frozen budget.
  * it checkpoints at every evaluation point and records the checkpoint digest, so the
    weights evaluated are provably the weights the run produced at that step rather than a
    checkpoint the submission selected.
  * it evaluates on the held-out split ITSELF, unsmoothed, at the bound evaluation point and
    at every scheduled sustain point, and marks each sample `verifier-recompute`. Anything
    the submission reports about its own loss is copied into `submission_reported` and is
    never the graded number.
  * it records `halted_at_step` and `halt_reason`, so a run that stopped on a favourable
    evaluation is graded as not having established the loss, with a reason.
  * it trains the frozen default mixture as a control arm and the harness-owned reference
    recipe as a floor, both under the identical freeze, and records both losses as the two
    ends of the run-local ladder.

The run record it writes is `/telemetry/run_record.json`. That path is outside the agent's
writable surface. tests/runner.py reads it and computes no verdict from it.
"""
from __future__ import annotations

TOKEN_BUDGET = 419430400
TOKENS_PER_STEP = 131072
STEPS_AT_BUDGET = 3200

MODEL = "frozen-decoder-12L-768d"
OPTIMIZER = "frozen-adamw-cosine"

EVALUATION_SPLIT_ID = "val-split-b"
BOUND_EVAL_STEP = 2800
SUSTAIN_POINTS = (2800, 3000, 3200)
SUSTAIN_TOLERANCE = 0.02

RUN_RECORD_PATH = "/telemetry/run_record.json"

FROZEN_AXES = {
    "token_budget_tokens": TOKEN_BUDGET,
    "tokens_per_step": TOKENS_PER_STEP,
    "steps_at_budget": STEPS_AT_BUDGET,
    "model": MODEL,
    "optimizer": OPTIMIZER,
    "evaluation_split_id": EVALUATION_SPLIT_ID,
    "bound_eval_step": BOUND_EVAL_STEP,
    "sustain_points": list(SUSTAIN_POINTS),
    "sustain_tolerance": SUSTAIN_TOLERANCE,
}


def frozen_axes():
    """The frozen axes, so a submission can read them and never has to guess them."""
    return dict(FROZEN_AXES)


def token_cost(steps):
    """Tokens a run of this many steps feeds. The budget is enforced against this."""
    return int(steps) * TOKENS_PER_STEP


def steps_within_budget(tokens=TOKEN_BUDGET):
    """The largest step count the frozen budget pays for."""
    return int(tokens) // TOKENS_PER_STEP


if __name__ == "__main__":
    import json

    print(json.dumps({"frozen_axes": frozen_axes(), "run_record": RUN_RECORD_PATH}, indent=2))
