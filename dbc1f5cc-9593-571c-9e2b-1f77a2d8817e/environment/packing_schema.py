#!/usr/bin/env python3
"""The batch-geometry plan schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. A plan that `validate` accepts here is a plan the
verifier will train, so a submission never has to guess what will be accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed plan inside the declared bounds -- and its answer is yes or no. It never
returns a quality, never returns a partial credit, and no number it produces reaches
the reward. A malformed plan is REFUSED with a machine-readable reason at reward 0.0;
a well-formed one is trained and scored purely on the validation loss it produces.

WHAT A PLAN IS
    A list of PHASES. The phases partition the frozen token budget in order. Each
    phase names a micro-batch geometry -- `rows` sequences of `seq_len` tokens -- and
    a `grad_accum`, the number of those micro-batches that accumulate into one
    optimizer step. `budget_frac` is the share of the frozen token budget the phase
    consumes.

WHAT A PLAN IS NOT
    It is not a schedule and it is not an optimizer. Both of those are frozen in
    frozen/task_spec.json and this schema rejects any attempt to name them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-packing/v1"

# The declared geometry grid. These MUST agree with frozen/task_spec.json; the
# verifier's build-time conformance check asserts that they do.
ALLOWED_ROWS = (8, 16, 32, 64)
ALLOWED_SEQ_LEN = (128, 256, 512, 1024)
ALLOWED_GRAD_ACCUM = (1, 2, 4)

MIN_MICROBATCH_TOKENS = 8192
MAX_MICROBATCH_TOKENS = 32768
MAX_PHASES = 6
MIN_PHASE_OPTIMIZER_STEPS = 8

# The budget shares must sum to one. Floating point is not asked to be exact.
BUDGET_FRAC_TOLERANCE = 1e-6
MIN_BUDGET_FRAC = 0.02

PHASE_KEYS = {"budget_frac", "rows", "seq_len", "grad_accum"}
PLAN_KEYS = {"schema", "phases", "notes"}


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _integer(where: str, key: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("plan-value-not-an-integer",
                      f"{where}.{key} is {value!r}, which is not an integer")
    return int(value)


def _fraction(where: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal("plan-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("plan-value-not-a-number", f"{where}.{key} is {value!r}, which is not finite")
    return value


def _exact_keys(where: str, allowed: set, block) -> None:
    if not isinstance(block, dict):
        raise Refusal("plan-block-not-an-object", f"{where} is {type(block).__name__}, not an object")
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise Refusal("plan-key-unknown", f"{where} carries unknown key(s) {', '.join(unknown)}")


def validate(plan, total_train_tokens: int) -> dict:
    """Return the plan if it is well formed and in bounds. Otherwise raise Refusal.

    `total_train_tokens` is the frozen budget, passed in rather than restated here, so
    the budget has exactly one declaration and this module cannot drift from it.
    """
    if not isinstance(plan, dict):
        raise Refusal("plan-not-an-object",
                      f"the submission parsed to {type(plan).__name__}, not an object")
    _exact_keys("plan", PLAN_KEYS, plan)
    for key in ("schema", "phases"):
        if key not in plan:
            raise Refusal("plan-key-missing", f"plan is missing {key}")
    if plan["schema"] != SCHEMA_ID:
        raise Refusal("plan-schema-unrecognised",
                      f"plan.schema is {plan['schema']!r}, expected {SCHEMA_ID!r}")

    phases = plan["phases"]
    if not isinstance(phases, list):
        raise Refusal("plan-phases-not-a-list",
                      f"plan.phases is {type(phases).__name__}, not a list")
    if not phases:
        raise Refusal("plan-phases-empty", "plan.phases carries no phase")
    if len(phases) > MAX_PHASES:
        raise Refusal("plan-too-many-phases",
                      f"plan.phases carries {len(phases)} phases, more than the {MAX_PHASES} allowed")

    total_frac = 0.0
    for index, phase in enumerate(phases):
        where = f"plan.phases[{index}]"
        _exact_keys(where, PHASE_KEYS, phase)
        missing = sorted(PHASE_KEYS - set(phase))
        if missing:
            raise Refusal("plan-key-missing", f"{where} is missing {', '.join(missing)}")

        rows = _integer(where, "rows", phase["rows"])
        seq_len = _integer(where, "seq_len", phase["seq_len"])
        accum = _integer(where, "grad_accum", phase["grad_accum"])
        frac = _fraction(where, "budget_frac", phase["budget_frac"])

        if rows not in ALLOWED_ROWS:
            raise Refusal("plan-rows-not-allowed",
                          f"{where}.rows is {rows}, not one of {list(ALLOWED_ROWS)}")
        if seq_len not in ALLOWED_SEQ_LEN:
            raise Refusal("plan-seq-len-not-allowed",
                          f"{where}.seq_len is {seq_len}, not one of {list(ALLOWED_SEQ_LEN)}")
        if accum not in ALLOWED_GRAD_ACCUM:
            raise Refusal("plan-grad-accum-not-allowed",
                          f"{where}.grad_accum is {accum}, not one of {list(ALLOWED_GRAD_ACCUM)}")

        micro = rows * seq_len
        if micro < MIN_MICROBATCH_TOKENS or micro > MAX_MICROBATCH_TOKENS:
            raise Refusal("plan-microbatch-tokens-out-of-bounds",
                          f"{where} has rows * seq_len = {micro}, outside "
                          f"[{MIN_MICROBATCH_TOKENS}, {MAX_MICROBATCH_TOKENS}]")
        if frac < MIN_BUDGET_FRAC or frac > 1.0:
            raise Refusal("plan-budget-frac-out-of-bounds",
                          f"{where}.budget_frac is {frac!r}, outside [{MIN_BUDGET_FRAC}, 1.0]")
        total_frac += frac

    if abs(total_frac - 1.0) > BUDGET_FRAC_TOLERANCE:
        raise Refusal("plan-budget-frac-does-not-sum-to-one",
                      f"the phase budget fractions sum to {total_frac!r}, not 1.0")

    # The phase schedule must be REALISABLE at the frozen budget: every phase has to
    # buy at least MIN_PHASE_OPTIMIZER_STEPS whole optimizer steps out of its share.
    # This is a bound, not a grade: a plan that cannot be run is refused by name
    # rather than silently run as something else.
    for index, phase in enumerate(phases):
        step_tokens = int(phase["rows"]) * int(phase["seq_len"]) * int(phase["grad_accum"])
        share = int(float(phase["budget_frac"]) * int(total_train_tokens))
        steps = share // step_tokens
        if steps < MIN_PHASE_OPTIMIZER_STEPS:
            raise Refusal("plan-phase-buys-too-few-steps",
                          f"plan.phases[{index}] takes {share} tokens at {step_tokens} tokens per "
                          f"optimizer step, which is {steps} step(s), fewer than the "
                          f"{MIN_PHASE_OPTIMIZER_STEPS} required")
    return plan


def load(path, total_train_tokens: int) -> dict:
    """Read and validate a plan from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no plan at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("plan-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("plan-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("plan-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed, total_train_tokens)
