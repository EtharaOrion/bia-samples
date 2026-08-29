#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml

The compiled surface of tests/checkers.yaml: one test per graded row, each
re-asserting that row against the same live harness handles tests/runner.py
produced. It never enters the reward; tests/grade.py owns that number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import runner  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent


def _handles():
    return runner.handles(EVIDENCE)


def test_bound_evaluation_point_reached():
    outcome = checkers.check_bound_evaluation_point_reached(_handles())
    assert outcome.ok, outcome.reason


def test_token_budget_respected_as_fed():
    outcome = checkers.check_token_budget_respected_as_fed(_handles())
    assert outcome.ok, outcome.reason


def test_evaluation_split_untrained():
    outcome = checkers.check_evaluation_split_untrained(_handles())
    assert outcome.ok, outcome.reason


def test_pool_state_matches_graded_pool():
    outcome = checkers.check_pool_state_matches_graded_pool(_handles())
    assert outcome.ok, outcome.reason


def test_graded_weights_are_harness_owned():
    outcome = checkers.check_graded_weights_are_harness_owned(_handles())
    assert outcome.ok, outcome.reason


def test_graded_loss_recomputed_unsmoothed():
    outcome = checkers.check_graded_loss_recomputed_unsmoothed(_handles())
    assert outcome.ok, outcome.reason


def test_feed_precedes_every_graded_evaluation():
    outcome = checkers.check_feed_precedes_every_graded_evaluation(_handles())
    assert outcome.ok, outcome.reason


def test_loss_sustained_across_verifier_folds():
    outcome = checkers.check_loss_sustained_across_verifier_folds(_handles())
    assert outcome.ok, outcome.reason


def test_mixture_beats_default_simplex_optimum():
    outcome = checkers.check_mixture_beats_default_simplex_optimum(_handles())
    assert outcome.ok, outcome.reason


ROWS = (
    ("bound_evaluation_point_reached", test_bound_evaluation_point_reached),
    ("token_budget_respected_as_fed", test_token_budget_respected_as_fed),
    ("evaluation_split_untrained", test_evaluation_split_untrained),
    ("pool_state_matches_graded_pool", test_pool_state_matches_graded_pool),
    ("graded_weights_are_harness_owned", test_graded_weights_are_harness_owned),
    ("graded_loss_recomputed_unsmoothed", test_graded_loss_recomputed_unsmoothed),
    ("feed_precedes_every_graded_evaluation", test_feed_precedes_every_graded_evaluation),
    ("loss_sustained_across_verifier_folds", test_loss_sustained_across_verifier_folds),
    ("mixture_beats_default_simplex_optimum", test_mixture_beats_default_simplex_optimum),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    global EVIDENCE
    EVIDENCE = Path(args.evidence)
    failed = []
    for ident, row in ROWS:
        try:
            row()
        except AssertionError as problem:
            failed.append(ident + ': ' + str(problem))
    for line in failed:
        print(line)
    print(str(len(ROWS) - len(failed)) + ' of ' + str(len(ROWS)) + ' compiled rows passed')
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
