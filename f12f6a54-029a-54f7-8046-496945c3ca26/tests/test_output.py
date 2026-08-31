"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml

Compiled per-checker tests for slot OER-10. Each test drives the REAL checker in
tests/checkers.py over the frozen fixtures in tests/fixtures.json: the accepting half on
the golden trajectory, the rejecting half on the planted fixture that fires exactly that
checker's zero reason. The last two tests drive the real gate chain in tests/grade.py.

Run standalone:  python3 tests/test_output.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import grade  # noqa: E402

FIXTURES = json.loads((HERE / "fixtures.json").read_text(encoding="utf-8"))
GOLDEN = checkers.HarnessView.from_mapping(FIXTURES["golden"])


def _planted(name):
    return checkers.HarnessView.from_mapping(FIXTURES["planted"][name])


def test_recipe_not_a_replay():
    accepted = checkers.check_recipe_not_a_replay(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_recipe_not_a_replay(_planted("nc-verbatim-published-mixture"))
    assert not rejected.passed, "planted fixture nc-verbatim-published-mixture did not fire recipe_not_a_replay"
    assert rejected.reason == "recipe-matches-published-mixture", rejected.reason


def test_fingerprint_screen_precedes_training():
    accepted = checkers.check_fingerprint_screen_precedes_training(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_fingerprint_screen_precedes_training(_planted("nc-screen-after-training"))
    assert not rejected.passed, "planted fixture nc-screen-after-training did not fire fingerprint_screen_precedes_training"
    assert rejected.reason == "screen-ran-after-training", rejected.reason


def test_graded_loss_is_verifier_computed():
    accepted = checkers.check_graded_loss_is_verifier_computed(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_graded_loss_is_verifier_computed(_planted("nc-submission-reported-loss"))
    assert not rejected.passed, "planted fixture nc-submission-reported-loss did not fire graded_loss_is_verifier_computed"
    assert rejected.reason == "graded-loss-not-verifier-computed", rejected.reason


def test_graded_readout_unsmoothed():
    accepted = checkers.check_graded_readout_unsmoothed(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_graded_readout_unsmoothed(_planted("nc-smoothed-readout"))
    assert not rejected.passed, "planted fixture nc-smoothed-readout did not fire graded_readout_unsmoothed"
    assert rejected.reason == "readout-smoothed-on-graded-path", rejected.reason


def test_loss_sustained_at_scheduled_points():
    accepted = checkers.check_loss_sustained_at_scheduled_points(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_loss_sustained_at_scheduled_points(_planted("nc-unsustained-dip"))
    assert not rejected.passed, "planted fixture nc-unsustained-dip did not fire loss_sustained_at_scheduled_points"
    assert rejected.reason == "loss-not-sustained", rejected.reason


def test_early_stop_not_a_result():
    accepted = checkers.check_early_stop_not_a_result(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_early_stop_not_a_result(_planted("nc-early-stop"))
    assert not rejected.passed, "planted fixture nc-early-stop did not fire early_stop_not_a_result"
    assert rejected.reason == "early-stop-not-a-result", rejected.reason


def test_evaluated_weights_harness_owned():
    accepted = checkers.check_evaluated_weights_harness_owned(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_evaluated_weights_harness_owned(_planted("nc-submission-selected-checkpoint"))
    assert not rejected.passed, "planted fixture nc-submission-selected-checkpoint did not fire evaluated_weights_harness_owned"
    assert rejected.reason == "weights-not-harness-owned", rejected.reason


def test_token_budget_respected_as_fed():
    accepted = checkers.check_token_budget_respected_as_fed(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_token_budget_respected_as_fed(_planted("nc-budget-overspend"))
    assert not rejected.passed, "planted fixture nc-budget-overspend did not fire token_budget_respected_as_fed"
    assert rejected.reason == "token-budget-overspent", rejected.reason


def test_eval_split_not_trained_on():
    accepted = checkers.check_eval_split_not_trained_on(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_eval_split_not_trained_on(_planted("nc-split-leak"))
    assert not rejected.passed, "planted fixture nc-split-leak did not fire eval_split_not_trained_on"
    assert rejected.reason == "eval-split-leaked", rejected.reason


def test_reference_reaches_full_reward():
    document = grade.score_document(FIXTURES["golden"])
    assert document["reward"] == FIXTURES["expected"]["golden_reward"], document
    assert document["reward"] == 1.0, document
    assert document["reason"] == "graded-loss-established", document


def test_reward_is_not_binary():
    document = grade.score_document(FIXTURES["partial_credit"])
    expected = FIXTURES["expected"]["partial_reward"]
    assert document["reward"] == expected, document
    assert 0.0 < document["reward"] < 1.0, document


def test_absent_telemetry_scores_zero_with_a_reason():
    document = grade.score_document(None)
    assert document["reward"] == 0.0, document
    assert document["reason"] == "telemetry-absent", document


def main():
    failed = []
    for name, case in sorted(globals().items()):
        if not name.startswith("test_") or not callable(case):
            continue
        try:
            case()
        except AssertionError as problem:
            failed.append(name + ": " + str(problem))
    total = sum(1 for name in globals() if name.startswith("test_"))
    print("collected " + str(total) + " tests, failed " + str(len(failed)))
    for line in failed:
        print("FAIL " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
