"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus the end-to-end reward test. Every test
drives the real checkers in tests/checkers.py over the golden trajectory that
solution/recompute.py derived. No test reads a clock; the telemetry is a recorded
fixture the harness produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import grade  # noqa: E402

REFERENCE = BUNDLE / 'solution' / 'fixtures' / 'reference_run'


def _outcomes():
    harness = grade.build_harness(BUNDLE, REFERENCE)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def test_every_checker_is_declared():
    declared = set(grade.CHECKER_ORDER)
    assert declared == set(_outcomes())


def test_trace_and_envelope_unmodified():
    outcome = _outcomes()['trace_and_envelope_unmodified']
    assert outcome.passed, outcome.detail


def test_iteration_sequence_ordered():
    outcome = _outcomes()['iteration_sequence_ordered']
    assert outcome.passed, outcome.detail


def test_summary_matches_durable_record():
    outcome = _outcomes()['summary_matches_durable_record']
    assert outcome.passed, outcome.detail


def test_every_request_accounted():
    outcome = _outcomes()['every_request_accounted']
    assert outcome.passed, outcome.detail


def test_no_early_stop_harvest():
    outcome = _outcomes()['no_early_stop_harvest']
    assert outcome.passed, outcome.detail


def test_unsmoothed_graded_readout():
    outcome = _outcomes()['unsmoothed_graded_readout']
    assert outcome.passed, outcome.detail


def test_slo_p99_respected():
    outcome = _outcomes()['slo_p99_respected']
    assert outcome.passed, outcome.detail


def test_measurement_windows_sustained():
    outcome = _outcomes()['measurement_windows_sustained']
    assert outcome.passed, outcome.detail


def test_reference_scores_full_reward():
    harness = grade.build_harness(BUNDLE, REFERENCE)
    bound = grade.load_bound(BUNDLE)
    verdict = grade.score(harness, bound)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded'
