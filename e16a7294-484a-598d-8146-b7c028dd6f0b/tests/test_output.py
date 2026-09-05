"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.

Compiled tests, one per declared checker, plus the reward-schema tests. These run
against the fixtures solution/recompute.py generated, through the same live checker
functions tests/grade.py uses.

Every test that needs a reading is guarded by `substrate_available`, because the
graded path here loads a real checkpoint and runs real forward passes. When the
verifier mount is absent these tests SKIP with a named reason rather than passing
vacuously, and tests/grade.py refuses the run outright with checkpoint-absent, so an
absent substrate can never read as a graded zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import evaluate  # noqa: E402
import grade  # noqa: E402

FIXTURES = ['early_stop', 'noop', 'overspend', 'plateau', 'reference', 'reported_metric', 'smoothed', 'unsustained']
ZERO_FIXTURES = ['early_stop', 'noop', 'overspend', 'plateau', 'reported_metric', 'smoothed', 'unsustained']

SKIP = 'the verifier substrate is not mounted, so no forward pass can be run here'


def substrate_available() -> bool:
    return evaluate.checkpoint_path().is_file() and evaluate.holdout_path().is_dir()


def _telemetry(fixture: str) -> dict:
    if not substrate_available():
        pytest.skip(SKIP)
    return grade.telemetry_for_fixture(fixture)


def test_every_fixture_is_a_well_formed_allocation():
    table = grade.anchors()['shards']
    for fixture in FIXTURES:
        scheme = evaluate.load_json(TESTS / 'fixtures' / (fixture + '.json'))
        assert len(scheme['tensors']) == table, fixture
        assert len({entry['name'] for entry in scheme['tensors']}) == table, fixture


def test_the_holdout_split_is_absent_from_the_environment():
    root = grade.environment_paths()['substrate'].parent
    for path in root.rglob('*'):
        assert 'fineweb_val' not in path.name, path


def test_reference_fixture_clears_the_gate_chain():
    outcome = grade.score(_telemetry('reference'))
    assert outcome['reason'] == 'graded', outcome
    assert outcome['reward'] > 0.0, outcome


def test_reward_is_a_float_in_the_closed_interval():
    for fixture in FIXTURES:
        outcome = grade.score(_telemetry(fixture))
        assert isinstance(outcome['reward'], float), fixture
        assert 0.0 <= outcome['reward'] <= 1.0, fixture


def test_every_zero_carries_a_machine_readable_reason():
    for fixture in ZERO_FIXTURES:
        outcome = grade.score(_telemetry(fixture))
        if outcome['reward'] == 0.0:
            assert checkers.is_reason_code(outcome['reason']), (fixture, outcome)


def test_frozen_checkpoint_and_holdout_unmodified():
    result = grade.run_one('frozen_checkpoint_and_holdout_unmodified', _telemetry('reference'))
    assert result.passed, result.reason

def test_graded_parameters_are_the_canonical_decoder():
    result = grade.run_one('graded_parameters_are_the_canonical_decoder', _telemetry('reference'))
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
