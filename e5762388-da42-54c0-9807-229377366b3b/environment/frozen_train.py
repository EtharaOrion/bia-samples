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


GRADED_SAMPLE_SOURCE = "verifier-recompute"
SUBMISSION_SAMPLE_SOURCE = "submission-reported"


def frozen_axes():
    """The frozen axes, so a submission can read them and never has to guess them."""
    return dict(FROZEN_AXES)


def evaluation_sample(step, loss, source=GRADED_SAMPLE_SOURCE):
    """One row of the run record's `samples` series.

    The trainer evaluates the held-out split itself at every scheduled sustain point
    and appends one of these per point. `source` is what separates the graded series
    from the reported one: only `verifier-recompute` rows are graded, and whatever the
    submission says about its own loss is carried as `submission-reported` and is never
    read on the graded path.
    """
    return {"step": int(step), "loss": float(loss), "source": str(source)}


def graded_samples(samples):
    """The verifier-recomputed subset of a `samples` series, in step order."""
    return sorted(
        (row for row in samples if row.get("source") == GRADED_SAMPLE_SOURCE),
        key=lambda row: int(row["step"]),
    )


def sustained(samples, tolerance=SUSTAIN_TOLERANCE):
    """Whether the level established at the bound point held at every sustain point.

    Returns the verdict and, when it is negative, the machine-readable reason the
    score document will carry. Two distinct failures are kept apart deliberately:
    a series MISSING a scheduled point is a run that stopped early and is graded as
    not having established the loss, while a series that carries every point but
    drifts above the bound loss by more than `tolerance` reached a level it did not
    hold. Scoring the first as the second would score a dip.
    """
    graded = graded_samples(samples)
    seen = {int(row["step"]): float(row["loss"]) for row in graded}
    missing = [step for step in SUSTAIN_POINTS if step not in seen]
    if missing:
        return False, "early-stop-not-a-result"
    bound = seen[BOUND_EVAL_STEP]
    for step in SUSTAIN_POINTS:
        if seen[step] > bound + float(tolerance):
            return False, "level-not-sustained"
    return True, ""


def graded_loss(samples):
    """The one graded number: the verifier's unsmoothed loss at the bound point."""
    for row in graded_samples(samples):
        if int(row["step"]) == BOUND_EVAL_STEP:
            return float(row["loss"])
    return None


def token_cost(steps):
    """Tokens a run of this many steps feeds. The budget is enforced against this."""
    return int(steps) * TOKENS_PER_STEP


def steps_within_budget(tokens=TOKEN_BUDGET):
    """The largest step count the frozen budget pays for."""
    return int(tokens) // TOKENS_PER_STEP


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            {
                "frozen_axes": frozen_axes(),
                "run_record": RUN_RECORD_PATH,
                "run_record_evaluation": {
                    "bound_eval_step": BOUND_EVAL_STEP,
                    "sustain_points": list(SUSTAIN_POINTS),
                    "sustain_tolerance": SUSTAIN_TOLERANCE,
                    "samples": {
                        "one_row_per": "scheduled sustain point",
                        "fields": ["step", "loss", "source"],
                        "graded_source": GRADED_SAMPLE_SOURCE,
                        "ungraded_source": SUBMISSION_SAMPLE_SOURCE,
                    },
                    "submission_reported": {
                        "graded": False,
                        "note": "copied verbatim from whatever the submission reported, smoothing and all, and never read on the graded path",
                    },
                },
            },
            indent=2,
        )
    )
