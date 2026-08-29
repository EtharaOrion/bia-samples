#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml

Both halves of every OER-09 checker, compiled from the fixtures grounding.yaml declares.
The accepting half runs the clean fixture through the real checker. The rejecting half
runs the planted fixture and asserts the exact zero_reason that checker owns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import grade  # noqa: E402

FIXTURES = json.loads((BUNDLE / 'solution' / 'fixtures.json').read_text(encoding='utf-8'))
BASE = FIXTURES['base']
CONTROLS = {row['id']: row for row in FIXTURES['negative_controls']}


def _verdict(state, ident):
    for row in checkers.run_all(state):
        if row.ident == ident:
            return row
    raise AssertionError('no verdict for ' + ident)


def test_reference_accepted_end_to_end():
    document = grade.evaluate(FIXTURES['accepting']['state'])
    assert document['reward'] == FIXTURES['accepting']['expected_reward'], document
    assert document['reason'] == FIXTURES['accepting']['expected_reason'], document


def test_reward_is_not_binary():
    document = grade.evaluate(FIXTURES['continuity']['state'])
    assert 0.0 < document['reward'] < 1.0, document
    assert document['reward'] == FIXTURES['continuity']['expected_reward'], document


def test_filter_claim_matches_observed_pool():
    accepted = _verdict(BASE, 'filter_claim_matches_observed_pool')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-silent-filter-no-op']['state'], 'filter_claim_matches_observed_pool')
    assert not planted.ok, 'nc-silent-filter-no-op did not fire filter_claim_matches_observed_pool'
    assert planted.reason == 'silent-filter-no-op', planted.reason
    document = grade.evaluate(CONTROLS['nc-silent-filter-no-op']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'silent-filter-no-op', document
    planted = _verdict(CONTROLS['nc-no-op-submission']['state'], 'filter_claim_matches_observed_pool')
    assert not planted.ok, 'nc-no-op-submission did not fire filter_claim_matches_observed_pool'
    assert planted.reason == 'silent-filter-no-op', planted.reason
    document = grade.evaluate(CONTROLS['nc-no-op-submission']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'silent-filter-no-op', document
    planted = _verdict(CONTROLS['nc-stale-curation-verdict']['state'], 'filter_claim_matches_observed_pool')
    assert not planted.ok, 'nc-stale-curation-verdict did not fire filter_claim_matches_observed_pool'
    assert planted.reason == 'silent-filter-no-op', planted.reason
    document = grade.evaluate(CONTROLS['nc-stale-curation-verdict']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'silent-filter-no-op', document


def test_curated_pool_consumed_by_trainer():
    accepted = _verdict(BASE, 'curated_pool_consumed_by_trainer')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-pool-not-consumed']['state'], 'curated_pool_consumed_by_trainer')
    assert not planted.ok, 'nc-pool-not-consumed did not fire curated_pool_consumed_by_trainer'
    assert planted.reason == 'curated-pool-not-consumed', planted.reason
    document = grade.evaluate(CONTROLS['nc-pool-not-consumed']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'curated-pool-not-consumed', document
    planted = _verdict(CONTROLS['nc-schema-drift-unhandled']['state'], 'curated_pool_consumed_by_trainer')
    assert not planted.ok, 'nc-schema-drift-unhandled did not fire curated_pool_consumed_by_trainer'
    assert planted.reason == 'curated-pool-not-consumed', planted.reason
    document = grade.evaluate(CONTROLS['nc-schema-drift-unhandled']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'curated-pool-not-consumed', document


def test_curation_precedes_first_feed():
    accepted = _verdict(BASE, 'curation_precedes_first_feed')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-curation-after-feed']['state'], 'curation_precedes_first_feed')
    assert not planted.ok, 'nc-curation-after-feed did not fire curation_precedes_first_feed'
    assert planted.reason == 'curation-after-feed-start', planted.reason
    document = grade.evaluate(CONTROLS['nc-curation-after-feed']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'curation-after-feed-start', document


def test_token_budget_respected_as_fed():
    accepted = _verdict(BASE, 'token_budget_respected_as_fed')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-budget-overspend']['state'], 'token_budget_respected_as_fed')
    assert not planted.ok, 'nc-budget-overspend did not fire token_budget_respected_as_fed'
    assert planted.reason == 'token-budget-overspent', planted.reason
    document = grade.evaluate(CONTROLS['nc-budget-overspend']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'token-budget-overspent', document


def test_eval_split_never_trained_on():
    accepted = _verdict(BASE, 'eval_split_never_trained_on')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-split-leak']['state'], 'eval_split_never_trained_on')
    assert not planted.ok, 'nc-split-leak did not fire eval_split_never_trained_on'
    assert planted.reason == 'eval-split-leaked', planted.reason
    document = grade.evaluate(CONTROLS['nc-split-leak']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'eval-split-leaked', document
    planted = _verdict(CONTROLS['nc-stale-holdout-audit']['state'], 'eval_split_never_trained_on')
    assert not planted.ok, 'nc-stale-holdout-audit did not fire eval_split_never_trained_on'
    assert planted.reason == 'eval-split-leaked', planted.reason
    document = grade.evaluate(CONTROLS['nc-stale-holdout-audit']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'eval-split-leaked', document


def test_graded_weights_harness_owned():
    accepted = _verdict(BASE, 'graded_weights_harness_owned')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-submission-selected-weights']['state'], 'graded_weights_harness_owned')
    assert not planted.ok, 'nc-submission-selected-weights did not fire graded_weights_harness_owned'
    assert planted.reason == 'weights-not-harness-owned', planted.reason
    document = grade.evaluate(CONTROLS['nc-submission-selected-weights']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'weights-not-harness-owned', document


def test_graded_loss_unsmoothed():
    accepted = _verdict(BASE, 'graded_loss_unsmoothed')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-smoothed-readout']['state'], 'graded_loss_unsmoothed')
    assert not planted.ok, 'nc-smoothed-readout did not fire graded_loss_unsmoothed'
    assert planted.reason == 'graded-loss-smoothed', planted.reason
    document = grade.evaluate(CONTROLS['nc-smoothed-readout']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'graded-loss-smoothed', document


def test_improvement_sustained_across_points():
    accepted = _verdict(BASE, 'improvement_sustained_across_points')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-unsustained-improvement']['state'], 'improvement_sustained_across_points')
    assert not planted.ok, 'nc-unsustained-improvement did not fire improvement_sustained_across_points'
    assert planted.reason == 'improvement-not-sustained', planted.reason
    document = grade.evaluate(CONTROLS['nc-unsustained-improvement']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'improvement-not-sustained', document


def test_early_stop_does_not_establish_loss():
    accepted = _verdict(BASE, 'early_stop_does_not_establish_loss')
    assert accepted.ok, accepted.detail
    planted = _verdict(CONTROLS['nc-early-stop']['state'], 'early_stop_does_not_establish_loss')
    assert not planted.ok, 'nc-early-stop did not fire early_stop_does_not_establish_loss'
    assert planted.reason == 'early-stop-loss-not-established', planted.reason
    document = grade.evaluate(CONTROLS['nc-early-stop']['state'])
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'early-stop-loss-not-established', document


TESTS = [
    'test_reference_accepted_end_to_end',
    'test_reward_is_not_binary',
    'test_filter_claim_matches_observed_pool',
    'test_curated_pool_consumed_by_trainer',
    'test_curation_precedes_first_feed',
    'test_token_budget_respected_as_fed',
    'test_eval_split_never_trained_on',
    'test_graded_weights_harness_owned',
    'test_graded_loss_unsmoothed',
    'test_improvement_sustained_across_points',
    'test_early_stop_does_not_establish_loss',
]


def main() -> int:
    passed = 0
    for name in TESTS:
        globals()[name]()
        passed += 1
    # A positive count, never a zero one. An empty parse is ambiguous between a suite
    # that ran clean and a suite that never ran, so the count is asserted positive.
    assert passed > 0, 'no compiled test ran'
    print('compiled tests passed: ' + str(passed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
