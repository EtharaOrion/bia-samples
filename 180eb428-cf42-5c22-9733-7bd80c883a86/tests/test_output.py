"""GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with solution/recompute.py.

One assertion per checker declared in tests/checkers.yaml. These read the outcome
carrier that tests/grade.py wrote; they never recompute a verdict, so a disagreement
between this file and the carrier is impossible by construction rather than by luck.
"""

import json
import os
import pathlib

import pytest

CARRIER = pathlib.Path(os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json"))


def _outcomes():
    if not CARRIER.is_file():
        pytest.skip("no outcome carrier at %s" % CARRIER)
    return json.loads(CARRIER.read_text())


def test_every_declared_checker_reported():
    got = set(_outcomes())
    want = set(DECLARED)
    assert got == want, "carrier reported %s, checkers.yaml declares %s" % (sorted(got), sorted(want))


DECLARED = ['agent_improves_on_standard_control', 'agent_phase_covered_every_declared_step', 'agent_report_reproduces_under_verifier_execution', 'all_phases_started_from_the_same_initialization', 'anchor_span_is_positive_and_finite', 'attempt_wallclock_within_declared_budget', 'baseline_anchor_ran_the_shipped_narrow_recipe', 'control_cache_agrees_with_graded_telemetry', 'format_in_force_at_every_reduction', 'fp32_control_ran_outside_the_frozen_format', 'frozen_format_digest_matches', 'frozen_harness_bytes_unmodified', 'graded_record_authored_by_the_verifier', 'master_weights_updated_in_declared_dtype', 'no_overflow_flag_raised_in_agent_phase', 'overflow_ledger_accounting_is_consistent', 'phases_ran_in_declared_order', 'policy_ran_outside_the_measuring_process', 'reduction_agrees_with_independent_replay', 'reported_outcome_agrees_with_telemetry']

REDUCTION_KIND = {
    "agent_improves_on_standard_control": "VALUE",
    "agent_phase_covered_every_declared_step": "INVARIANT",
    "agent_report_reproduces_under_verifier_execution": "DIVERGENCE",
    "all_phases_started_from_the_same_initialization": "INVARIANT",
    "anchor_span_is_positive_and_finite": "VALUE",
    "attempt_wallclock_within_declared_budget": "VALUE",
    "baseline_anchor_ran_the_shipped_narrow_recipe": "VALUE",
    "control_cache_agrees_with_graded_telemetry": "DIVERGENCE",
    "format_in_force_at_every_reduction": "INVARIANT",
    "fp32_control_ran_outside_the_frozen_format": "ABSENCE",
    "frozen_format_digest_matches": "VALUE",
    "frozen_harness_bytes_unmodified": "VALUE",
    "graded_record_authored_by_the_verifier": "VALUE",
    "master_weights_updated_in_declared_dtype": "EFFECT",
    "no_overflow_flag_raised_in_agent_phase": "ABSENCE",
    "overflow_ledger_accounting_is_consistent": "INVARIANT",
    "phases_ran_in_declared_order": "ORDERING",
    "policy_ran_outside_the_measuring_process": "INVARIANT",
    "reduction_agrees_with_independent_replay": "DIVERGENCE",
    "reported_outcome_agrees_with_telemetry": "DIVERGENCE"
}


def test_each_checker_reduces_to_exactly_one_of_the_six_kinds():
    six = {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}
    for name, kind in REDUCTION_KIND.items():
        assert kind in six, "%s reduces to %r which is not one of the six" % (name, kind)


def test_agent_improves_on_standard_control():
    assert _outcomes().get("agent_improves_on_standard_control") is True


def test_agent_phase_covered_every_declared_step():
    assert _outcomes().get("agent_phase_covered_every_declared_step") is True


def test_agent_report_reproduces_under_verifier_execution():
    assert _outcomes().get("agent_report_reproduces_under_verifier_execution") is True


def test_all_phases_started_from_the_same_initialization():
    assert _outcomes().get("all_phases_started_from_the_same_initialization") is True


def test_anchor_span_is_positive_and_finite():
    assert _outcomes().get("anchor_span_is_positive_and_finite") is True


def test_attempt_wallclock_within_declared_budget():
    assert _outcomes().get("attempt_wallclock_within_declared_budget") is True


def test_baseline_anchor_ran_the_shipped_narrow_recipe():
    assert _outcomes().get("baseline_anchor_ran_the_shipped_narrow_recipe") is True


def test_control_cache_agrees_with_graded_telemetry():
    assert _outcomes().get("control_cache_agrees_with_graded_telemetry") is True


def test_format_in_force_at_every_reduction():
    assert _outcomes().get("format_in_force_at_every_reduction") is True


def test_fp32_control_ran_outside_the_frozen_format():
    assert _outcomes().get("fp32_control_ran_outside_the_frozen_format") is True


def test_frozen_format_digest_matches():
    assert _outcomes().get("frozen_format_digest_matches") is True


def test_frozen_harness_bytes_unmodified():
    assert _outcomes().get("frozen_harness_bytes_unmodified") is True


def test_graded_record_authored_by_the_verifier():
    assert _outcomes().get("graded_record_authored_by_the_verifier") is True


def test_master_weights_updated_in_declared_dtype():
    assert _outcomes().get("master_weights_updated_in_declared_dtype") is True


def test_no_overflow_flag_raised_in_agent_phase():
    assert _outcomes().get("no_overflow_flag_raised_in_agent_phase") is True


def test_overflow_ledger_accounting_is_consistent():
    assert _outcomes().get("overflow_ledger_accounting_is_consistent") is True


def test_phases_ran_in_declared_order():
    assert _outcomes().get("phases_ran_in_declared_order") is True


def test_policy_ran_outside_the_measuring_process():
    assert _outcomes().get("policy_ran_outside_the_measuring_process") is True


def test_reduction_agrees_with_independent_replay():
    assert _outcomes().get("reduction_agrees_with_independent_replay") is True


def test_reported_outcome_agrees_with_telemetry():
    assert _outcomes().get("reported_outcome_agrees_with_telemetry") is True
