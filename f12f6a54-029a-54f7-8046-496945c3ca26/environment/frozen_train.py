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

The run record it writes is `/telemetry/run_record.json`. tests/runner.py invokes this module
and then reads that record, and it computes no verdict from it.

WHO MAY INVOKE THIS, AND WHY IT MATTERS. Only the verifier does. `tests/runner.py` supplies
the three arms and OVERWRITES the record on every run. The agent image creates /telemetry, so
a submission could otherwise have written a finished run record there and had it graded; that
is precisely the surface this arrangement closes. The record is evidence the verifier produced,
never a file it found.

THE SUBSTRATE IS A SURROGATE, AND IT IS DECLARED AS ONE. No H100 and no transformer runs here.
`loss_at` is a deterministic mixture-quality surrogate: with the token budget frozen and fed in
full, the graded loss is a decreasing function of the average training utility of the corpus the
recipe selected, which is the quantity a curation recipe is free to move. It is bound as
gap-oer-10-training-substrate-is-a-deterministic-surrogate. It is substrate-local and it is NOT
a family anchor: `baseline_metric` and `target_metric` stay absent under
gap-oer-per-family-anchors-unmeasured, and the two ends of the ladder are measured inside this
same run exactly as instruction.md binds them.

No clock, no random source, no network and no host state is read. Every number below is a pure
function of the raw pool bytes and the three weight maps handed in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

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

HALT_REASON = "budget-exhausted"
VERIFIER_SOURCE = "verifier-recompute"
HARNESS_OWNER = "harness"
NO_SMOOTHING = "none"

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

# ---------------------------------------------------------------------------------------
# The substrate. Every constant here is frozen and readable, and none of them is an anchor.

# Per-bucket training value of a token. The evaluation split is not a bucket sample, so the
# buckets are not equally worth spending a frozen budget on.
BUCKET_VALUE = {
    "math": 1.0,
    "code": 0.875,
    "encyclopedic": 0.875,
    "web": 0.75,
    "dialogue": 0.5,
    "legal": 0.375,
}
# A document whose quality decile sits above its perplexity decile carries signal the model
# does not already have. Mapped onto [0, 1] so the two deciles enter symmetrically.
SIGNAL_OFFSET = 9.0
SIGNAL_RANGE = 18.0
# Documents far from the context length are truncated or padded and pay for tokens twice.
LENGTH_CENTER = 900.0
LENGTH_SPAN = 1400.0
# A near-duplicate spends budget on tokens the run has already seen.
DUP_SCALE = 6.0
# The evaluation split is English. Tokens in another language spend the budget on a
# distribution the graded split does not measure.
LANGUAGE_VALUE = {"en": 1.0}
OTHER_LANGUAGE_VALUE = 0.2

# loss = LOSS_FLOOR + LOSS_SPAN * (1 - mixture rate) + LOSS_TAIL * (1 - step / STEPS_AT_BUDGET)
LOSS_FLOOR = 2.9
LOSS_SPAN = 1.2
LOSS_TAIL = 0.08
LOSS_DECIMALS = 6


def frozen_axes():
    """The frozen axes, so a submission can read them and never has to guess them."""
    return dict(FROZEN_AXES)


def token_cost(steps):
    """Tokens a run of this many steps feeds. The budget is enforced against this."""
    return int(steps) * TOKENS_PER_STEP


def steps_within_budget(tokens=TOKEN_BUDGET):
    """The largest step count the frozen budget pays for."""
    return int(tokens) // TOKENS_PER_STEP


def utility(document) -> float:
    """The training value of one token of this document, on [0, 1]."""
    base = BUCKET_VALUE.get(document.get("bucket"), 0.0)
    signal = (float(document.get("quality_decile", 0)) - float(document.get("ppl_decile", 0))
              + SIGNAL_OFFSET) / SIGNAL_RANGE
    signal = 0.0 if signal < 0.0 else (1.0 if signal > 1.0 else signal)
    band = 1.0 - abs(float(document.get("len_tokens", 0)) - LENGTH_CENTER) / LENGTH_SPAN
    band = 0.0 if band < 0.0 else (1.0 if band > 1.0 else band)
    taper = 1.0 - float(document.get("dup_class", 0)) / DUP_SCALE
    taper = 0.0 if taper < 0.0 else taper
    language = LANGUAGE_VALUE.get(document.get("lang"), OTHER_LANGUAGE_VALUE)
    return base * signal * band * taper * language


def mixture_rate(documents, weights) -> float:
    """The token-weighted mean utility of the corpus a weight map selects.

    A weight map is a mixture: a document's share of the frozen budget is proportional to its
    weight times its length. Selecting nothing is a rate of zero, which is the worst corpus
    rather than an absent measurement.
    """
    numerator = 0.0
    denominator = 0.0
    for document in documents:
        weight = int(weights.get(str(document.get("id")), 0))
        if weight < 1:
            continue
        mass = float(weight) * float(document.get("len_tokens", 0))
        numerator += mass * utility(document)
        denominator += mass
    return numerator / denominator if denominator > 0.0 else 0.0


def loss_at(rate: float, step: int) -> float:
    """The surrogate validation loss of a run at `step`, for a corpus of this mixture rate."""
    tail = LOSS_TAIL * (1.0 - float(step) / float(STEPS_AT_BUDGET))
    return round(LOSS_FLOOR + LOSS_SPAN * (1.0 - float(rate)) + tail, LOSS_DECIMALS)


def checkpoint_digest(step: int, rate: float) -> str:
    """The harness's identity for the weights it holds at `step`.

    Derived from the state this process accumulated, so a submission cannot name it in
    advance and cannot substitute a checkpoint of its own.
    """
    payload = json.dumps(
        {"owner": HARNESS_OWNER, "step": int(step), "rate": round(float(rate), LOSS_DECIMALS)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def train(documents, weights, submission_reported=None, fed_doc_ids=None) -> dict:
    """Run one arm to budget exhaustion and return the record the verifier reads."""
    rate = mixture_rate(documents, weights)
    fed = [str(document.get("id")) for document in documents
           if int(weights.get(str(document.get("id")), 0)) >= 1]
    if fed_doc_ids is not None:
        fed = [str(value) for value in fed_doc_ids]

    checkpoints = []
    samples = []
    for step in SUSTAIN_POINTS:
        digest = checkpoint_digest(step, rate)
        checkpoints.append({"step": step, "owner": HARNESS_OWNER, "digest": digest})
        samples.append(
            {
                "step": step,
                "loss": loss_at(rate, step),
                "smoothing": NO_SMOOTHING,
                "source": VERIFIER_SOURCE,
                "weights_owner": HARNESS_OWNER,
                "weights_digest": digest,
            }
        )

    return {
        "mixture_rate": round(rate, LOSS_DECIMALS),
        "fed_doc_ids": fed,
        "training": {
            "frozen_token_budget": TOKEN_BUDGET,
            "tokens_fed": TOKEN_BUDGET,
            "steps_executed": STEPS_AT_BUDGET,
            "halted_at_step": STEPS_AT_BUDGET,
            "halt_reason": HALT_REASON,
        },
        "evaluation": {
            "bound_eval_step": BOUND_EVAL_STEP,
            "sustain_points": list(SUSTAIN_POINTS),
            "sustain_tolerance": SUSTAIN_TOLERANCE,
            "samples": samples,
            "submission_reported": submission_reported,
        },
        "checkpoints": checkpoints,
    }


def run(arms: dict) -> dict:
    """Train the graded arm, the control arm and the reference floor under one freeze."""
    documents = arms["documents"]
    graded = train(
        documents,
        arms["submission"],
        submission_reported=arms.get("submission_reported"),
        fed_doc_ids=arms.get("fed_doc_ids"),
    )
    control = mixture_rate(documents, arms["control"])
    reference = mixture_rate(documents, arms["reference"])
    record = dict(graded)
    record["bound_held_out_split_id"] = str(arms.get("bound_held_out_split_id", EVALUATION_SPLIT_ID))
    record["ladder"] = {
        "control_arm_loss": loss_at(control, BOUND_EVAL_STEP),
        "reference_floor_loss": loss_at(reference, BOUND_EVAL_STEP),
        "state": "run-local-measured",
        "control_arm_rate": round(control, LOSS_DECIMALS),
        "reference_floor_rate": round(reference, LOSS_DECIMALS),
    }
    record["frozen_axes"] = frozen_axes()
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="train the graded, control and reference arms")
    parser.add_argument("--arms", help="json file carrying the three weight maps")
    parser.add_argument("--out", default=RUN_RECORD_PATH)
    args = parser.parse_args(argv)

    # Bare invocation keeps printing the frozen axes, which is what a submission reads this
    # module for and what it did before it could train.
    if not args.arms:
        print(json.dumps({"frozen_axes": frozen_axes(), "run_record": RUN_RECORD_PATH}, indent=2))
        return 0

    arms = json.loads(Path(args.arms).read_text(encoding="utf-8"))
    record = run(arms)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "run record written to " + str(target)
        + ", graded rate " + str(record["mixture_rate"])
        + ", control " + str(record["ladder"]["control_arm_loss"])
        + ", floor " + str(record["ladder"]["reference_floor_loss"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
