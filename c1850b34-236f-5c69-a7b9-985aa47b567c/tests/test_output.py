#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml

The compiled tests. One per declared checker, each exercising BOTH halves: the
golden record must be accepted, and a single-defect fixture must be rejected with
exactly that checker's zero reason. The stale controls prove drift causality: the
same fixture reads clean under a pre-mutation bound value and dirty under the live
one, so a verdict carried across a silent mutation is caught.

Runs standalone (`python3 tests/test_output.py`) and under pytest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402

FIXTURES = json.loads((HERE / "fixtures" / "records.json").read_text(encoding="utf-8"))
BOUND = FIXTURES["bound_values"]
GOLDEN = FIXTURES["golden"]


def _bound(**overrides):
    merged = dict(BOUND)
    merged.update(overrides)
    return merged


def test_heldout_contamination_absent():
    accepted = checkers.selector("heldout_contamination_absent")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-paraphrased-near-duplicate-emitter"]
    rejected = checkers.selector("heldout_contamination_absent")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-paraphrased-near-duplicate-emitter was accepted"
    assert rejected.reason == "heldout-item-reproduced", rejected.reason


def test_benchmark_unread_on_generation_path():
    accepted = checkers.selector("benchmark_unread_on_generation_path")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-benchmark-read-during-generation"]
    rejected = checkers.selector("benchmark_unread_on_generation_path")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-benchmark-read-during-generation was accepted"
    assert rejected.reason == "benchmark-read-before-generation", rejected.reason


def test_grading_tree_immutable():
    accepted = checkers.selector("grading_tree_immutable")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-grading-tree-mutated"]
    rejected = checkers.selector("grading_tree_immutable")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-grading-tree-mutated was accepted"
    assert rejected.reason == "grading-tree-mutated", rejected.reason


def test_training_budget_respected():
    accepted = checkers.selector("training_budget_respected")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-budget-overspend"]
    rejected = checkers.selector("training_budget_respected")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-budget-overspend was accepted"
    assert rejected.reason == "training-budget-overspent", rejected.reason


def test_graded_score_unsmoothed():
    accepted = checkers.selector("graded_score_unsmoothed")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-smoothed-readout"]
    rejected = checkers.selector("graded_score_unsmoothed")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-smoothed-readout was accepted"
    assert rejected.reason == "graded-readout-smoothed", rejected.reason


def test_score_sustained_across_eval_points():
    accepted = checkers.selector("score_sustained_across_eval_points")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-unsustained-spike"]
    rejected = checkers.selector("score_sustained_across_eval_points")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-unsustained-spike was accepted"
    assert rejected.reason == "score-not-sustained", rejected.reason


def test_early_stop_not_a_score():
    accepted = checkers.selector("early_stop_not_a_score")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-early-stop"]
    rejected = checkers.selector("early_stop_not_a_score")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-early-stop was accepted"
    assert rejected.reason == "early-stop-without-sustained-score", rejected.reason


def test_graded_score_from_harness_state():
    accepted = checkers.selector("graded_score_from_harness_state")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-submission-reported-score"]
    rejected = checkers.selector("graded_score_from_harness_state")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-submission-reported-score was accepted"
    assert rejected.reason == "graded-score-taken-from-submission-report", rejected.reason


def test_evaluated_weights_harness_owned():
    accepted = checkers.selector("evaluated_weights_harness_owned")(GOLDEN, BOUND)
    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail
    fixture = FIXTURES["fixtures"]["neg-foreign-checkpoint"]
    rejected = checkers.selector("evaluated_weights_harness_owned")(fixture["record"], BOUND)
    assert not rejected.ok, "planted defect neg-foreign-checkpoint was accepted"
    assert rejected.reason == "evaluated-weights-not-harness-owned", rejected.reason


def test_ctl_contamination_verdict_under_loose_threshold():
    control = FIXTURES["stale_controls"]["ctl-contamination-verdict-under-loose-threshold"]
    checker = checkers.selector(control["checker"])
    stale = checker(control["record"], _bound(**control["stale"]))
    live = checker(control["record"], _bound(**control["live"]))
    assert stale.ok, "the pre-mutation verdict should read clean: " + stale.reason
    assert not live.ok, "the post-mutation verdict should read dirty"


def test_ctl_contamination_verdict_under_v1_normalization():
    control = FIXTURES["stale_controls"]["ctl-contamination-verdict-under-v1-normalization"]
    checker = checkers.selector(control["checker"])
    stale = checker(control["record"], _bound(**control["stale"]))
    live = checker(control["record"], _bound(**control["live"]))
    assert stale.ok, "the pre-mutation verdict should read clean: " + stale.reason
    assert not live.ok, "the post-mutation verdict should read dirty"


def test_ctl_sustain_verdict_under_two_points():
    control = FIXTURES["stale_controls"]["ctl-sustain-verdict-under-two-points"]
    checker = checkers.selector(control["checker"])
    stale = checker(control["record"], _bound(**control["stale"]))
    live = checker(control["record"], _bound(**control["live"]))
    assert stale.ok, "the pre-mutation verdict should read clean: " + stale.reason
    assert not live.ok, "the post-mutation verdict should read dirty"


def test_ctl_budget_verdict_under_4000_steps():
    control = FIXTURES["stale_controls"]["ctl-budget-verdict-under-4000-steps"]
    checker = checkers.selector(control["checker"])
    stale = checker(control["record"], _bound(**control["stale"]))
    live = checker(control["record"], _bound(**control["live"]))
    assert stale.ok, "the pre-mutation verdict should read clean: " + stale.reason
    assert not live.ok, "the post-mutation verdict should read dirty"


def main() -> int:
    tests = sorted(name for name in globals() if name.startswith("test_"))
    collected = len(tests)
    failed = []
    for name in tests:
        try:
            globals()[name]()
        except AssertionError as error:
            failed.append(name + ": " + str(error))
    passed = collected - len(failed)
    print(json.dumps({"collected": collected, "passed": passed, "failed": len(failed)}, sort_keys=True))
    for row in failed:
        print(row, file=sys.stderr)
    if collected == 0 or failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
