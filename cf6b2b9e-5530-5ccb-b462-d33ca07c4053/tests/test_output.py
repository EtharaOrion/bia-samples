#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. source: solution/grounding.yaml

Compiled tests over the checker fixtures. One test per declared checker, each
carrying that checker's accepting half and its rejecting half, plus the
statement-ambiguity test proving exactly one graded outcome.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import grade

FIXTURES = json.loads((TESTS / 'fixtures.json').read_text(encoding='utf-8'))
CASES = {row['id']: row for row in FIXTURES['cases']}


def _score(case_id):
    return grade.score_fixture(TESTS.parent, CASES[case_id])


def test_reference_accepted():
    row = _score('reference-accepted')
    assert row['reward'] == 1.0, row
    assert row['reason'] == 'target-scale-point-reached', row


def test_allocation_well_formed_over_every_tensor():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['allocation_well_formed_over_every_tensor'] is True, accepting
    rejecting = _score('ctl-no-op-submission')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-no-op-submission']['expect_reason'], rejecting
    assert rejecting['by_checker']['allocation_well_formed_over_every_tensor'] is False, rejecting


def test_frozen_corpus_and_reference_unmodified():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['frozen_corpus_and_reference_unmodified'] is True, accepting
    rejecting = _score('ctl-frozen-artifact-modified')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-frozen-artifact-modified']['expect_reason'], rejecting
    assert rejecting['by_checker']['frozen_corpus_and_reference_unmodified'] is False, rejecting


def test_bit_budget_respected_as_allocated():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['bit_budget_respected_as_allocated'] is True, accepting
    rejecting = _score('ctl-budget-accounted-weights-only')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-budget-accounted-weights-only']['expect_reason'], rejecting
    assert rejecting['by_checker']['bit_budget_respected_as_allocated'] is False, rejecting


def test_calibration_probe_ledger_ordered():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['calibration_probe_ledger_ordered'] is True, accepting
    rejecting = _score('ctl-probe-ledger-truncated-at-v1')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-probe-ledger-truncated-at-v1']['expect_reason'], rejecting
    assert rejecting['by_checker']['calibration_probe_ledger_ordered'] is False, rejecting


def test_allocation_matches_in_force_calibration():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['allocation_matches_in_force_calibration'] is True, accepting
    rejecting = _score('ctl-allocation-frozen-at-calibration-v1')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-allocation-frozen-at-calibration-v1']['expect_reason'], rejecting
    assert rejecting['by_checker']['allocation_matches_in_force_calibration'] is False, rejecting


def test_graded_degradation_is_verifier_recomputed():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['graded_degradation_is_verifier_recomputed'] is True, accepting
    rejecting = _score('ctl-readout-ema-blended')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-readout-ema-blended']['expect_reason'], rejecting
    assert rejecting['by_checker']['graded_degradation_is_verifier_recomputed'] is False, rejecting


def test_early_stop_does_not_establish_metric():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['early_stop_does_not_establish_metric'] is True, accepting
    rejecting = _score('ctl-early-stop-on-favourable-point')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-early-stop-on-favourable-point']['expect_reason'], rejecting
    assert rejecting['by_checker']['early_stop_does_not_establish_metric'] is False, rejecting


def test_degradation_sustained_across_scheduled_points():
    accepting = _score('reference-accepted')
    assert accepting['by_checker']['degradation_sustained_across_scheduled_points'] is True, accepting
    rejecting = _score('ctl-sustained-claim-over-two-points')
    assert rejecting['reward'] == 0.0, rejecting
    assert rejecting['reason'] == CASES['ctl-sustained-claim-over-two-points']['expect_reason'], rejecting
    assert rejecting['by_checker']['degradation_sustained_across_scheduled_points'] is False, rejecting


def test_statement_admits_exactly_one_graded_outcome():
    row = _score('reference-accepted')
    assert row['metric']['graded_quantity'] == 'mean-perplexity-degradation-over-scheduled-points', row
    assert row['metric']['reading_a'] == row['metric']['reading_b'], row


def main():
    failures = []
    for name, function in sorted(globals().items()):
        if not name.startswith('test_') or not callable(function):
            continue
        try:
            function()
        except AssertionError as problem:
            failures.append(name + ': ' + str(problem)[:400])
    for row in failures:
        print('FAIL ' + row)
    print('compiled tests: ' + str(len(failures)) + ' failed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
