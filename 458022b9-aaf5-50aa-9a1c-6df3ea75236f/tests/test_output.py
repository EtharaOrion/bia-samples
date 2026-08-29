"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.

Compiled tests, one per declared checker, plus the reward-schema tests. These run
against the fixtures solution/recompute.py generated, through the same live
checker functions tests/grade.py uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import grade  # noqa: E402


def _telemetry(fixture: str) -> dict:
    return grade.telemetry_for_fixture(fixture)


def test_reference_fixture_scores_full_reward():
    outcome = grade.score(_telemetry('reference'))
    assert outcome['reward'] == 1.0, outcome
    assert outcome['reason'] == 'graded', outcome


def test_reward_is_a_float_in_the_closed_interval():
    for fixture in ['early_stop', 'noop', 'overspend', 'plateau', 'reference', 'reported_metric', 'smoothed', 'unsustained']:
        outcome = grade.score(_telemetry(fixture))
        assert isinstance(outcome['reward'], float), fixture
        assert 0.0 <= outcome['reward'] <= 1.0, fixture


def test_every_zero_carries_a_machine_readable_reason():
    for fixture in ['early_stop', 'noop', 'overspend', 'plateau', 'reported_metric', 'smoothed', 'unsustained']:
        outcome = grade.score(_telemetry(fixture))
        assert outcome['reward'] == 0.0, fixture
        assert checkers.is_reason_code(outcome['reason']), (fixture, outcome)


def test_frozen_corpus_and_reference_unmodified():
    result = grade.run_one('frozen_corpus_and_reference_unmodified', _telemetry('reference'))
    assert result.passed, result.reason

def test_evaluation_follows_state_seal():
    result = grade.run_one('evaluation_follows_state_seal', _telemetry('reference'))
    assert result.passed, result.reason

def test_submitted_scheme_changes_quantized_state():
    result = grade.run_one('submitted_scheme_changes_quantized_state', _telemetry('reference'))
    assert result.passed, result.reason

def test_bit_budget_respected():
    result = grade.run_one('bit_budget_respected', _telemetry('reference'))
    assert result.passed, result.reason

def test_graded_readout_is_raw_recompute():
    result = grade.run_one('graded_readout_is_raw_recompute', _telemetry('reference'))
    assert result.passed, result.reason

def test_evaluation_not_halted_early():
    result = grade.run_one('evaluation_not_halted_early', _telemetry('reference'))
    assert result.passed, result.reason

def test_degradation_sustained_across_scheduled_points():
    result = grade.run_one('degradation_sustained_across_scheduled_points', _telemetry('reference'))
    assert result.passed, result.reason

def test_beats_default_scheme_optimum():
    result = grade.run_one('beats_default_scheme_optimum', _telemetry('reference'))
    assert result.passed, result.reason


def test_measured_gap_is_positive():
    assert 1.081420983 > 0.0
