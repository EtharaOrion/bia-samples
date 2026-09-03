"""Reach every checker, reduce to one float, and hand the verdict to the trap in test.sh.

This file imports `tests/checkers.py` and `solution/reference.py`. It NEVER imports the
submission; the submission is run out-of-process by `tests/runner.py` and its output is
captured as bytes that reach no graded path.

The reduction is two-stage and the two stages answer different questions.

  Stage one, admissibility. Every checker in `CHECKER_TABLE` is a required gate. The
  gates are walked in declared order and the FIRST failing one ends grading at reward
  0.0 carrying that gate's machine-readable zero_reason. This is exactly what
  `aggregation.mode: required_pass` in tests/checkers.yaml describes.

  Stage two, the reward. Only if every gate passed, the reward is the clipped
  normalisation of the graded validation loss. It is a float in the closed interval
  [0.0, 1.0], higher is better, and it is NEVER binary. Aggregation decides
  admissibility; it does not produce the number.

Nothing here reads a clock. The graded snapshot version comes from the harness-owned
snapshot ledger, which is this slot's ordering basis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "solution"))

import checkers  # noqa: E402
from checkers import (  # noqa: E402
    check_early_stop_is_not_an_established_loss,
    check_eval_split_not_trained_on,
    check_graded_loss_verifier_computed,
    check_graded_weights_are_harness_owned,
    check_loss_sustained_across_bound_points,
    check_parse_snapshot_matches_graded_eval,
    check_pipeline_ran_over_graded_corpus,
    check_readout_unsmoothed,
    check_snapshot_version_sequence_ordered,
    check_token_budget_respected_as_fed,
)

HARNESS_LOG_ROOT = "/logs/harness"

REWARD_FLOOR = 0.0
REWARD_CEILING = 1.0

# The manifest, restated where the grading process can read it without a YAML parser.
# seed/tasks/OER-11/adequacy.py asserts this table agrees with tests/checkers.yaml row
# for row, so the restatement can never drift away from the declaration.
CHECKER_TABLE = (
    ("graded_loss_verifier_computed", check_graded_loss_verifier_computed, "graded-loss-not-verifier-computed", 3, True),
    ("readout_unsmoothed", check_readout_unsmoothed, "readout-smoothing-on-graded-path", 3, True),
    ("loss_sustained_across_bound_points", check_loss_sustained_across_bound_points, "loss-not-sustained", 3, True),
    ("early_stop_is_not_an_established_loss", check_early_stop_is_not_an_established_loss, "early-stop-loss-not-established", 2, True),
    ("graded_weights_are_harness_owned", check_graded_weights_are_harness_owned, "weights-not-harness-owned", 2, True),
    ("snapshot_version_sequence_ordered", check_snapshot_version_sequence_ordered, "snapshot-sequence-out-of-order", 2, True),
    ("parse_snapshot_matches_graded_eval", check_parse_snapshot_matches_graded_eval, "parse-built-against-stale-snapshot", 4, True),
    ("token_budget_respected_as_fed", check_token_budget_respected_as_fed, "token-budget-overspent", 3, True),
    ("eval_split_not_trained_on", check_eval_split_not_trained_on, "eval-split-leaked-into-training", 4, True),
    ("pipeline_ran_over_graded_corpus", check_pipeline_ran_over_graded_corpus, "token-stream-not-produced-this-run", 2, True),
)


def clamp(value: float) -> float:
    """One float in [0, 1]. NaN orders against nothing, so it resolves to the floor."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return REWARD_FLOOR
    if number != number:
        return REWARD_FLOOR
    return max(REWARD_FLOOR, min(REWARD_CEILING, number))


def gate(handles: dict) -> list:
    """Run every declared gate in declared order and return the verdicts."""
    return [selector(handles) for _, selector, _, _, _ in CHECKER_TABLE]


def _anchors(handles: dict):
    """The two numbers this lane does not author, read from the harness anchor record.

    F13 anchors are unmeasured, recorded under gap-oer-per-family-anchors-unmeasured.
    An absent, null or degenerate record resolves to the floor with a reason, never to a
    vacuous pass and never to a number invented here.
    """
    record = handles.get("anchor_record")
    if not isinstance(record, dict):
        return None, None
    if str(record.get("anchors_state", "")) != "measured":
        return None, None
    try:
        baseline = float(record.get("baseline_metric"))
        target = float(record.get("target_metric"))
    except (TypeError, ValueError):
        return None, None
    if baseline != baseline or target != target or baseline <= target:
        return None, None
    return baseline, target


def _graded_loss(handles: dict):
    point = (handles.get("eval_ledger") or {}).get("graded_point")
    if not isinstance(point, dict):
        return None
    try:
        return float(point.get("loss"))
    except (TypeError, ValueError):
        return None


def score(handles: dict) -> dict:
    """The whole reduction: gates first, then the continuous reward. One dict out."""
    verdicts = gate(handles)
    rows = [item.as_dict() for item in verdicts]
    ledger = handles.get("eval_ledger") or {}

    metric = {
        "graded_validation_loss": None,
        "baseline_metric": None,
        "target_metric": None,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "graded_snapshot_version": checkers.graded_snapshot_version(handles),
        "bound_evaluation_step": ledger.get("bound_evaluation_step"),
        "direction": "lower",
    }

    for (name, _selector, zero_reason, _weight, required), verdict in zip(CHECKER_TABLE, verdicts):
        if required and not verdict.passed:
            return {
                "reward": REWARD_FLOOR,
                "reason": zero_reason,
                "failed_checker": name,
                "observed": verdict.observed,
                "metric": metric,
                "checkers": rows,
                "aggregation": "required_pass",
            }

    baseline, target = _anchors(handles)
    agent = _graded_loss(handles)
    metric["graded_validation_loss"] = agent
    if agent is None:
        return {
            "reward": REWARD_FLOOR,
            "reason": "graded-loss-absent",
            "failed_checker": None,
            "observed": "every gate passed and the evaluation ledger still carries no graded loss",
            "metric": metric,
            "checkers": rows,
            "aggregation": "required_pass",
        }
    if baseline is None or target is None:
        return {
            "reward": REWARD_FLOOR,
            "reason": "anchors-unresolved",
            "failed_checker": None,
            "observed": "the harness anchor record does not resolve a measured baseline and target for family F13",
            "metric": metric,
            "checkers": rows,
            "aggregation": "required_pass",
        }

    metric["baseline_metric"] = baseline
    metric["target_metric"] = target
    metric["anchors_state"] = "measured"
    metric["anchors_gap"] = None

    raw = (baseline - agent) / (baseline - target)
    value = clamp(raw)
    if value <= REWARD_FLOOR:
        reason = "loss-not-better-than-baseline"
    elif value >= REWARD_CEILING:
        reason = "target-reached-or-beaten"
    else:
        reason = "partial-improvement-over-baseline"
    return {
        "reward": value,
        "reason": reason,
        "failed_checker": None,
        "observed": "every required gate passed; reward is the clipped normalisation of the verifier-computed validation loss",
        "metric": metric,
        "checkers": rows,
        "aggregation": "required_pass",
    }


def main(argv: list) -> int:
    """Compute the verdict and write it where the trap in tests/test.sh reads it."""
    log_root = Path(argv[0]) if argv else Path(HARNESS_LOG_ROOT)
    destination = Path(argv[1]) if len(argv) > 1 else Path("/tmp/oer11-verdict.json")
    handles = checkers.load_handles(log_root)
    verdict = score(handles)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(verdict, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reward": verdict["reward"], "reason": verdict["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
