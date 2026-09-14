"""GENERATED SECTION. DO NOT HAND-EDIT.

Derived from solution/grounding.yaml by solution/recompute.py.

Both halves of every checker, over the compiled fixtures in tests/fixtures/.
The accepting half is the clean fixture, on which every checker passes and the
reward is exactly 1.0. The rejecting half is that checker's planted fixture, on
which that checker and only that checker fails, with its declared reason.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import reward  # noqa: E402

FIXTURES = HERE / "fixtures"


def load(name):
    return json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))


def first_failure(session):
    for name, verdict in checkers.run_chain(session):
        if not verdict.ok:
            return name, verdict.reason
    return None, ""


def test_accepting_half_scores_full_reward():
    session = load("accept_reference_session")
    assert first_failure(session) == (None, "")
    metric, _ = checkers.consolidation(session)
    anchors = session["anchors"]
    value = reward.anchored(metric, anchors["baseline_metric"], anchors["target_metric"])
    assert value == 1.0


def test_accepting_half_survives_a_non_empty_report():
    session = load("accept_reported_within_tolerance")
    claims = [row for row in session["attempts"]
              if (row.get("reported") or {}).get("crossing_step") is not None]
    assert claims
    assert first_failure(session) == (None, "")


def test_graded_crossing_recomputed_unsmoothed():
    clean = load("accept_reference_session")
    assert checkers.check_graded_crossing_recomputed_unsmoothed(clean).ok
    planted = load("reject_readout_smoothed")
    verdict = checkers.check_graded_crossing_recomputed_unsmoothed(planted)
    assert not verdict.ok
    assert verdict.reason == "readout-smoothed-on-the-graded-path"
    assert first_failure(planted) == ("graded_crossing_recomputed_unsmoothed", "readout-smoothed-on-the-graded-path")


def test_crossing_sustained_across_scheduled_evaluations():
    clean = load("accept_reference_session")
    assert checkers.check_crossing_sustained_across_scheduled_evaluations(clean).ok
    planted = load("reject_crossing_not_sustained")
    verdict = checkers.check_crossing_sustained_across_scheduled_evaluations(planted)
    assert not verdict.ok
    assert verdict.reason == "crossing-not-sustained"
    assert first_failure(planted) == ("crossing_sustained_across_scheduled_evaluations", "crossing-not-sustained")


def test_early_halt_yields_no_crossing():
    clean = load("accept_reference_session")
    assert checkers.check_early_halt_yields_no_crossing(clean).ok
    planted = load("reject_early_stop_claimed")
    verdict = checkers.check_early_halt_yields_no_crossing(planted)
    assert not verdict.ok
    assert verdict.reason == "early-stop-claimed-as-crossing"
    assert first_failure(planted) == ("early_halt_yields_no_crossing", "early-stop-claimed-as-crossing")


def test_evaluated_weights_owned_by_the_harness():
    clean = load("accept_reference_session")
    assert checkers.check_evaluated_weights_owned_by_the_harness(clean).ok
    planted = load("reject_weights_not_harness_owned")
    verdict = checkers.check_evaluated_weights_owned_by_the_harness(planted)
    assert not verdict.ok
    assert verdict.reason == "weights-not-harness-owned"
    assert first_failure(planted) == ("evaluated_weights_owned_by_the_harness", "weights-not-harness-owned")


def test_graded_metric_consolidated_over_distinct_recipes():
    clean = load("accept_reference_session")
    assert checkers.check_graded_metric_consolidated_over_distinct_recipes(clean).ok
    planted = load("reject_unconsolidated")
    verdict = checkers.check_graded_metric_consolidated_over_distinct_recipes(planted)
    assert not verdict.ok
    assert verdict.reason == "single-attempt-metric-unconsolidated"
    assert first_failure(planted) == ("graded_metric_consolidated_over_distinct_recipes", "single-attempt-metric-unconsolidated")


def test_refinement_frontier_carried_across_iterations():
    clean = load("accept_reference_session")
    assert checkers.check_refinement_frontier_carried_across_iterations(clean).ok
    planted = load("reject_not_carried")
    verdict = checkers.check_refinement_frontier_carried_across_iterations(planted)
    assert not verdict.ok
    assert verdict.reason == "graded-outcome-not-carried"
    assert first_failure(planted) == ("refinement_frontier_carried_across_iterations", "graded-outcome-not-carried")


def test_reported_crossing_reconciles_with_verifier_measurement():
    clean = load("accept_reference_session")
    assert checkers.check_reported_crossing_reconciles_with_verifier_measurement(clean).ok
    planted = load("reject_report_diverges")
    verdict = checkers.check_reported_crossing_reconciles_with_verifier_measurement(planted)
    assert not verdict.ok
    assert verdict.reason == "reported-crossing-diverges-from-measurement"
    assert first_failure(planted) == ("reported_crossing_reconciles_with_verifier_measurement", "reported-crossing-diverges-from-measurement")


def test_bound_envelope_respected():
    clean = load("accept_reference_session")
    assert checkers.check_bound_envelope_respected(clean).ok
    planted = load("reject_attempt_budget_exceeded")
    verdict = checkers.check_bound_envelope_respected(planted)
    assert not verdict.ok
    assert verdict.reason == "attempt-budget-exceeded"
    assert first_failure(planted) == ("bound_envelope_respected", "attempt-budget-exceeded")


def test_frozen_axes_untouched():
    clean = load("accept_reference_session")
    assert checkers.check_frozen_axes_untouched(clean).ok
    planted = load("reject_frozen_axis_moved")
    verdict = checkers.check_frozen_axes_untouched(planted)
    assert not verdict.ok
    assert verdict.reason == "frozen-axis-moved"
    assert first_failure(planted) == ("frozen_axes_untouched", "frozen-axis-moved")


def test_carried_summary_is_current():
    clean = load("accept_reference_session")
    assert checkers.check_carried_summary_is_current(clean).ok
    planted = load("reject_stale_summary")
    verdict = checkers.check_carried_summary_is_current(planted)
    assert not verdict.ok
    assert verdict.reason == "stale-summary-carried"
    assert first_failure(planted) == ("carried_summary_is_current", "stale-summary-carried")
