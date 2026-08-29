#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml

The compiled mirror of the graded checker set. One test per checker, in the same order
tests/checkers.py registers them, because that order decides which machine-readable
reason a defect is attributed to.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402

DEFAULT_RUN_DIR = "/tmp/oer12-run"


def run_dir() -> pathlib.Path:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return pathlib.Path(sys.argv[1])
    return pathlib.Path(DEFAULT_RUN_DIR)


def handles() -> checkers.Handles:
    root = run_dir()
    return checkers.Handles(harness=root / "harness", submission_view=root / "submission_view")


def _assert(ident: str, zero_reason: str, verdict) -> None:
    assert verdict.passed, ident + " scored zero with reason " + (verdict.reason or zero_reason) \
        + ": " + verdict.detail


def test_graded_loss_from_verifier_evaluation() -> None:
    _assert("graded_loss_from_verifier_evaluation", "graded-loss-not-verifier-computed", checkers.check_graded_loss_from_verifier_evaluation(handles()))


def test_graded_readout_unsmoothed() -> None:
    _assert("graded_readout_unsmoothed", "graded-readout-smoothed", checkers.check_graded_readout_unsmoothed(handles()))


def test_graded_loss_sustained() -> None:
    _assert("graded_loss_sustained", "graded-loss-not-sustained", checkers.check_graded_loss_sustained(handles()))


def test_bound_evaluation_point_reached() -> None:
    _assert("bound_evaluation_point_reached", "early-stop-loss-not-established", checkers.check_bound_evaluation_point_reached(handles()))


def test_evaluated_weights_harness_owned() -> None:
    _assert("evaluated_weights_harness_owned", "evaluated-weights-not-harness-owned", checkers.check_evaluated_weights_harness_owned(handles()))


def test_parse_outcome_classes_partitioned() -> None:
    _assert("parse_outcome_classes_partitioned", "parse-outcome-classes-conflated", checkers.check_parse_outcome_classes_partitioned(handles()))


def test_parse_outcome_three_way_resolved() -> None:
    _assert("parse_outcome_three_way_resolved", "ambiguous-class-coerced-silently", checkers.check_parse_outcome_three_way_resolved(handles()))


def test_ambiguity_policy_declared_before_classification() -> None:
    _assert("ambiguity_policy_declared_before_classification", "ambiguity-policy-declared-after-classification", checkers.check_ambiguity_policy_declared_before_classification(handles()))


def test_token_budget_respected_as_fed() -> None:
    _assert("token_budget_respected_as_fed", "token-budget-overspent", checkers.check_token_budget_respected_as_fed(handles()))


def test_evaluation_split_not_trained_on() -> None:
    _assert("evaluation_split_not_trained_on", "evaluation-split-leaked-into-training", checkers.check_evaluation_split_not_trained_on(handles()))


def test_frozen_axes_unmoved_across_run() -> None:
    _assert("frozen_axes_unmoved_across_run", "frozen-axis-mutated", checkers.check_frozen_axes_unmoved_across_run(handles()))


CASES = (
    ("graded_loss_from_verifier_evaluation", test_graded_loss_from_verifier_evaluation),
    ("graded_readout_unsmoothed", test_graded_readout_unsmoothed),
    ("graded_loss_sustained", test_graded_loss_sustained),
    ("bound_evaluation_point_reached", test_bound_evaluation_point_reached),
    ("evaluated_weights_harness_owned", test_evaluated_weights_harness_owned),
    ("parse_outcome_classes_partitioned", test_parse_outcome_classes_partitioned),
    ("parse_outcome_three_way_resolved", test_parse_outcome_three_way_resolved),
    ("ambiguity_policy_declared_before_classification", test_ambiguity_policy_declared_before_classification),
    ("token_budget_respected_as_fed", test_token_budget_respected_as_fed),
    ("evaluation_split_not_trained_on", test_evaluation_split_not_trained_on),
    ("frozen_axes_unmoved_across_run", test_frozen_axes_unmoved_across_run),
)


def main() -> int:
    failures = []
    for name, case in CASES:
        try:
            case()
        except AssertionError as exc:
            failures.append(name + ": " + str(exc))
    for line in failures:
        print("FAIL " + line)
    print("checkers collected " + str(len(CASES)) + ", failed " + str(len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
