#!/usr/bin/env python3
"""The evaluation-policy schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. A policy that `validate` accepts here is a
policy the verifier will replay, so a submission never has to guess what will be
accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this
a well-formed policy inside the declared budget -- and its answer is yes or no. It
never returns a quality, never returns a partial credit, and no number it produces
reaches the reward. A malformed policy is REFUSED with a machine-readable reason at
reward 0.0; a well-formed one is replayed and scored purely on the held-out loss at
the checkpoint it selects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer12-eval-policy/v1"

# The evaluation budget. These are the scarce thing in this task.
TOKENS_PER_BATCH = 8192          # one probe batch, 16 x 512
MAX_PROBES = 6                   # distinct checkpoints a policy may read
EVAL_TOKEN_BUDGET = 98304        # 12 probe batches, to be divided among the probes
MAX_TOKENS_PER_PROBE = 131072    # the harness records 16 probe batches per step

ALLOWED_RULES = ("argmin", "argmin_smoothed")
ALLOWED_WINDOWS = (1, 3, 5)

# The candidate grid, restated from frozen/task_spec.json. It is duplicated here so
# that a submission can be validated without loading the substrate, and grade.py
# asserts the two agree before it grades anything.
GRID_FIRST, GRID_LAST, GRID_STRIDE = 128, 2048, 64


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def candidate_steps() -> tuple:
    return tuple(range(GRID_FIRST, GRID_LAST + 1, GRID_STRIDE))


def _integer(where: str, key: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("policy-value-not-an-integer",
                      f"{where}.{key} is {value!r}, which is not an integer")
    return value


def validate(policy) -> dict:
    """Return the policy if it is well formed and inside budget. Otherwise raise."""
    if not isinstance(policy, dict):
        raise Refusal("policy-not-an-object",
                      f"the submission parsed to {type(policy).__name__}, not an object")

    unknown = sorted(set(policy) - {"schema", "probes", "selection", "notes"})
    if unknown:
        raise Refusal("policy-key-unknown", f"policy carries unknown key(s) {', '.join(unknown)}")
    for key in ("schema", "probes", "selection"):
        if key not in policy:
            raise Refusal("policy-key-missing", f"policy is missing {key}")
    if policy["schema"] != SCHEMA_ID:
        raise Refusal("policy-schema-unrecognised",
                      f"policy.schema is {policy['schema']!r}, expected {SCHEMA_ID!r}")

    probes = policy["probes"]
    if not isinstance(probes, list) or not probes:
        raise Refusal("policy-probes-empty",
                      "policy.probes must be a non-empty list of {step, tokens} objects")
    if len(probes) > MAX_PROBES:
        raise Refusal("policy-too-many-probes",
                      f"policy.probes holds {len(probes)} probes, at most {MAX_PROBES} are allowed")

    grid = set(candidate_steps())
    seen = set()
    spent = 0
    for i, probe in enumerate(probes):
        where = f"policy.probes[{i}]"
        if not isinstance(probe, dict):
            raise Refusal("policy-probe-not-an-object",
                          f"{where} is {type(probe).__name__}, not an object")
        extra = sorted(set(probe) - {"step", "tokens"})
        if extra:
            raise Refusal("policy-key-unknown", f"{where} carries unknown key(s) {', '.join(extra)}")
        for key in ("step", "tokens"):
            if key not in probe:
                raise Refusal("policy-key-missing", f"{where} is missing {key}")
        step = _integer(where, "step", probe["step"])
        tokens = _integer(where, "tokens", probe["tokens"])
        if step not in grid:
            raise Refusal("policy-step-off-grid",
                          f"{where}.step is {step}, which is not on the candidate grid "
                          f"{GRID_FIRST}..{GRID_LAST} stride {GRID_STRIDE}")
        if step in seen:
            raise Refusal("policy-step-repeated", f"{where}.step {step} is probed more than once")
        seen.add(step)
        if tokens <= 0 or tokens % TOKENS_PER_BATCH != 0:
            raise Refusal("policy-tokens-not-a-batch-multiple",
                          f"{where}.tokens is {tokens}, which is not a positive multiple "
                          f"of {TOKENS_PER_BATCH}")
        if tokens > MAX_TOKENS_PER_PROBE:
            raise Refusal("policy-tokens-above-per-probe-cap",
                          f"{where}.tokens is {tokens}, above the per-probe cap "
                          f"{MAX_TOKENS_PER_PROBE}")
        spent += tokens
    if spent > EVAL_TOKEN_BUDGET:
        raise Refusal("policy-evaluation-budget-overspent",
                      f"policy.probes spend {spent} probe tokens, the budget is "
                      f"{EVAL_TOKEN_BUDGET}")

    selection = policy["selection"]
    if not isinstance(selection, dict):
        raise Refusal("policy-selection-not-an-object",
                      f"policy.selection is {type(selection).__name__}, not an object")
    extra = sorted(set(selection) - {"rule", "window"})
    if extra:
        raise Refusal("policy-key-unknown",
                      f"policy.selection carries unknown key(s) {', '.join(extra)}")
    if "rule" not in selection:
        raise Refusal("policy-key-missing", "policy.selection is missing rule")
    if selection["rule"] not in ALLOWED_RULES:
        raise Refusal("policy-selection-rule-unknown",
                      f"policy.selection.rule is {selection['rule']!r}, not one of "
                      f"{list(ALLOWED_RULES)}")
    if selection["rule"] == "argmin_smoothed":
        if "window" not in selection:
            raise Refusal("policy-key-missing",
                          "policy.selection.rule is argmin_smoothed but window is missing")
        window = _integer("policy.selection", "window", selection["window"])
        if window not in ALLOWED_WINDOWS:
            raise Refusal("policy-selection-window-unknown",
                          f"policy.selection.window is {window}, not one of "
                          f"{list(ALLOWED_WINDOWS)}")
    elif "window" in selection:
        _integer("policy.selection", "window", selection["window"])

    return policy


def load(path) -> dict:
    """Read and validate a policy from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no policy at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("policy-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("policy-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("policy-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)
