"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py

The compiled surface for slot OER-03. One accepting test and one rejecting
test for every checker in tests/checkers.yaml, plus the reference-accept test
and the negative control proving that a schedule extended past the sustained
crossing scores identically to one that is not extended.

These tests import tests/checkers.py and tests/grade.py. They never import a
submission, and they read no state a submission produced.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import grade  # noqa: E402

GOLDEN = json.loads((HERE / "golden.json").read_text(encoding="utf-8"))
ACCEPTING = GOLDEN["golden"]["telemetry"]
FIXTURES = {row["id"]: row for row in GOLDEN["fixtures"]}


def _reason(telemetry):
    return grade.grade_telemetry(telemetry)


def test_reference_accepted_end_to_end():
    result = _reason(ACCEPTING)
    assert result["reward"] == GOLDEN["golden"]["expected_reward"]
    assert result["reason"] == ""
    assert result["metric"]["graded_step"] == GOLDEN["golden"]["expected_graded_step"]


def test_schedule_extension_past_crossing_scores_identically():
    plain = _reason(ACCEPTING)
    extended = _reason(GOLDEN["extension_control"]["telemetry"])
    assert extended["metric"]["graded_step"] == plain["metric"]["graded_step"]
    assert extended["reward"] == plain["reward"]
    assert extended["metric"]["schedule_length"] != plain["metric"]["schedule_length"]


def test_fx_no_op_submission_is_rejected():
    result = _reason(FIXTURES["fx-no-op-submission"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "run-produced-no-verifier-state"


def test_fx_weights_domain_stale_is_rejected():
    result = _reason(FIXTURES["fx-weights-domain-stale"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "weights-not-harness-owned"


def test_fx_submission_selected_checkpoint_is_rejected():
    result = _reason(FIXTURES["fx-submission-selected-checkpoint"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "weights-not-harness-owned"


def test_fx_evaluation_order_broken_is_rejected():
    result = _reason(FIXTURES["fx-evaluation-order-broken"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "evaluation-order-violated"


def test_fx_readout_smoothed_is_rejected():
    result = _reason(FIXTURES["fx-readout-smoothed"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "readout-smoothed"


def test_fx_crossing_not_sustained_is_rejected():
    result = _reason(FIXTURES["fx-crossing-not-sustained"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "crossing-not-sustained"


def test_fx_early_stop_is_rejected():
    result = _reason(FIXTURES["fx-early-stop"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "early-stop-without-sustained-crossing"


def test_fx_non_convergence_ungraded_is_rejected():
    result = _reason(FIXTURES["fx-non-convergence-ungraded"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "non-convergence-ungraded"


def test_fx_graded_step_is_schedule_length_is_rejected():
    result = _reason(FIXTURES["fx-graded-step-is-schedule-length"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "graded-step-not-the-crossing"


def test_fx_crossing_bar_stale_is_rejected():
    result = _reason(FIXTURES["fx-crossing-bar-stale"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "graded-step-not-the-crossing"


def test_fx_reported_crossing_diverges_is_rejected():
    result = _reason(FIXTURES["fx-reported-crossing-diverges"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "reported-crossing-diverges"


def test_fx_frozen_batch_stale_is_rejected():
    result = _reason(FIXTURES["fx-frozen-batch-stale"]["telemetry"])
    assert result["reward"] == 0.0
    assert result["reason"] == "frozen-axis-moved"


def test_non_convergence_graded_as_failure_with_a_reason():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["non_convergence_graded_as_failure_with_a_reason"].passed


def test_run_produced_verifier_owned_state():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["run_produced_verifier_owned_state"].passed


def test_frozen_axes_held():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["frozen_axes_held"].passed


def test_evaluated_weights_owned_by_the_harness():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["evaluated_weights_owned_by_the_harness"].passed


def test_evaluation_points_ascend_on_the_verifier_schedule():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["evaluation_points_ascend_on_the_verifier_schedule"].passed


def test_graded_readout_unsmoothed():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["graded_readout_unsmoothed"].passed


def test_early_stop_is_not_a_crossing():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["early_stop_is_not_a_crossing"].passed


def test_crossing_sustained_across_verifier_evaluations():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["crossing_sustained_across_verifier_evaluations"].passed


def test_graded_step_is_the_crossing_not_the_schedule():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["graded_step_is_the_crossing_not_the_schedule"].passed


def test_reported_crossing_matches_recomputed_crossing():
    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}
    assert verdicts["reported_crossing_matches_recomputed_crossing"].passed


def _outcome(telemetry):
    filled = dict(telemetry)
    filled["outcome"] = checkers.derive_outcome(filled)
    return filled
