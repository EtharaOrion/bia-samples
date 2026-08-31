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
PLANTED = BUNDLE / 'solution' / 'fixtures' / 'planted'


def _outcomes():
    harness = grade.build_harness(BUNDLE, REFERENCE)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def _planted_telemetry(name):
    rows = []
    for line in (PLANTED / name).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _cadence_outcome_over(name):
    harness = checkers.Harness(telemetry=_planted_telemetry(name))
    return checkers.check_arrival_cadence_ordered(harness, grade.load_bound(BUNDLE))


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
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


def test_arrival_cadence_ordered():
    outcome = _outcomes()['arrival_cadence_ordered']
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


def test_arrival_cadence_rejects_base_minus_one():
    outcome = _cadence_outcome_over('arrival_cadence_base_minus_one.jsonl')
    assert not outcome.passed, outcome.detail
    assert outcome.reason == checkers.REASON_ARRIVAL_CADENCE, outcome.reason


def test_arrival_cadence_rejects_base_plus_one():
    outcome = _cadence_outcome_over('arrival_cadence_base_plus_one.jsonl')
    assert not outcome.passed, outcome.detail
    assert outcome.reason == checkers.REASON_ARRIVAL_CADENCE, outcome.reason


def test_arrival_cadence_rejects_modulus_minus_one():
    outcome = _cadence_outcome_over('arrival_cadence_modulus_minus_one.jsonl')
    assert not outcome.passed, outcome.detail
    assert outcome.reason == checkers.REASON_ARRIVAL_CADENCE, outcome.reason


def test_arrival_cadence_rejects_modulus_plus_one():
    outcome = _cadence_outcome_over('arrival_cadence_modulus_plus_one.jsonl')
    assert not outcome.passed, outcome.detail
    assert outcome.reason == checkers.REASON_ARRIVAL_CADENCE, outcome.reason


def test_arrival_cadence_accepts_the_reference_telemetry():
    harness = grade.build_harness(BUNDLE, REFERENCE)
    bound = grade.load_bound(BUNDLE)
    outcome = checkers.check_arrival_cadence_ordered(harness, bound)
    assert outcome.passed, outcome.detail


def test_reference_scores_full_reward():
    harness = grade.build_harness(BUNDLE, REFERENCE)
    bound = grade.load_bound(BUNDLE)
    verdict = grade.score(harness, bound)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded'
