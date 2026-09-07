#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. source: solution/grounding.yaml

Compiled tests over the checker fixtures. Verifier-owned.

One test per declared checker, each carrying that checker's accepting half and its
rejecting half, plus the statement-ambiguity test proving exactly one graded outcome.

Every case here is a telemetry record rather than a submission, because after the
re-base of this slot onto the nanoGPT substrate the graded numbers are forward passes
over a held-out split. Planting telemetry keeps both halves of every checker
exercisable from bundle bytes alone, with no GPU and no corpus, while leaving the
checkers themselves the same pure functions the graded path calls.
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

REJECTING = {
    'allocation_well_formed_over_every_tensor': 'ctl-no-op-submission',
    'frozen_corpus_and_reference_unmodified': 'ctl-frozen-artifact-modified',
    'graded_artifact_binds_to_declared_architecture': 'ctl-graded-artifact-is-not-the-decoder',
    'bit_budget_respected_as_allocated': 'ctl-budget-accounted-weights-only',
    'calibration_probe_ledger_ordered': 'ctl-probe-ledger-truncated-at-v1',
    'allocation_matches_in_force_calibration': 'ctl-allocation-frozen-at-calibration-v1',
    'calibration_slice_witness_matches_live_state': 'ctl-calibration-slice-witness-wrong',
    'graded_degradation_is_verifier_recomputed': 'ctl-readout-ema-blended',
    'early_stop_does_not_establish_metric': 'ctl-early-stop-on-favourable-point',
    'degradation_sustained_across_scheduled_points': 'ctl-sustained-claim-over-two-points',
}


def _score(case_id):
    return grade.score_fixture(TESTS.parent, CASES[case_id])


def test_reference_accepted():
    row = _score('reference-accepted')
    assert row['reward'] == 1.0, row
    assert row['reason'] == 'target-scale-point-reached', row


def test_every_checker_has_both_halves():
    """One accepting half and one rejecting half for each checker in the gate chain."""
    declared = [ident for ident, _ in grade.CHECKERS]
    assert sorted(declared) == sorted(REJECTING), declared
    accepting = _score('reference-accepted')
    for ident in declared:
        assert accepting['by_checker'][ident] is True, (ident, accepting)
        control = REJECTING[ident]
        rejecting = _score(control)
        assert rejecting['reward'] == 0.0, (ident, rejecting)
        assert rejecting['reason'] == CASES[control]['expect_reason'], (ident, rejecting)
        assert rejecting['by_checker'][ident] is False, (ident, rejecting)


def test_every_fixture_reaches_its_declared_outcome():
    for case_id, case in sorted(CASES.items()):
        row = _score(case_id)
        assert row['reward'] == case['expect_reward'], (case_id, row)
        assert row['reason'] == case['expect_reason'], (case_id, row)


def test_graded_artifact_must_be_the_declared_decoder():
    """The re-base control. A parameter file that is not the frozen architecture scores zero.

    This is the checker that makes the first clause of the simulator test enforceable
    from bundle bytes: a weight table, a cost model or a statistics blob cannot present
    the shape map that environment/nanogpt_substrate.json implies.
    """
    row = _score('ctl-graded-artifact-is-not-the-decoder')
    assert row['reward'] == 0.0, row
    assert row['reason'] == 'graded-artifact-not-the-declared-architecture', row
    assert row['by_checker']['graded_artifact_binds_to_declared_architecture'] is False, row
    accepted = _score('reference-accepted')
    assert accepted['metric']['architecture_bound'] is True, accepted


def test_calibration_slice_witness_is_sensitive_to_content():
    """A run that never read built environment state cannot produce the content digest."""
    for case_id in ('ctl-calibration-slice-witness-wrong', 'ctl-calibration-slice-witness-absent'):
        row = _score(case_id)
        assert row['reward'] == 0.0, row
        assert row['reason'] == 'calibration-slice-content-not-established', row
        assert row['by_checker']['calibration_slice_witness_matches_live_state'] is False, row


def test_no_graded_number_comes_from_the_submission():
    accepted = _score('reference-accepted')
    assert accepted['metric']['graded_source'] == 'verifier-recomputed-from-pristine-checkpoint', accepted
    assert accepted['metric']['reported_degradation_ignored'] is None, accepted
    lifted = _score('ctl-graded-value-lifted-from-submission')
    assert lifted['reward'] == 0.0, lifted
    assert lifted['reason'] == 'readout-smoothed', lifted


def test_reference_perplexity_is_measured_not_declared():
    accepted = _score('reference-accepted')
    assert accepted['metric']['reference_perplexity_state'] == 'measured-in-run-by-the-verifier', accepted


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
