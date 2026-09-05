"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker over the frozen reference run, one per frozen
rejecting fixture, and a byte-minimality test for every near miss. Every test drives the
real checkers in tests/checkers.py through the real grader in tests/grade.py. No test
reads a clock.
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
REJECTING = BUNDLE / 'solution' / 'fixtures' / 'rejecting'


def _verdict_over(workspace):
    harness = grade.build_harness(BUNDLE, workspace)
    bound = grade.load_bound(BUNDLE)
    return grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))


def _outcomes():
    harness = grade.build_harness(BUNDLE, REFERENCE)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes())


def test_warehouse_unmodified():
    outcome = _outcomes()['warehouse_unmodified']
    assert outcome.passed, outcome.detail


def test_plan_is_single_read_only_select():
    outcome = _outcomes()['plan_is_single_read_only_select']
    assert outcome.passed, outcome.detail


def test_quarantine_row_identified():
    outcome = _outcomes()['quarantine_row_identified']
    assert outcome.passed, outcome.detail


def test_closure_set_complete():
    outcome = _outcomes()['closure_set_complete']
    assert outcome.passed, outcome.detail


def test_closure_layers_witnessed():
    outcome = _outcomes()['closure_layers_witnessed']
    assert outcome.passed, outcome.detail


def test_closure_cardinality_attested():
    outcome = _outcomes()['closure_cardinality_attested']
    assert outcome.passed, outcome.detail


def test_excluded_rows_absent():
    outcome = _outcomes()['excluded_rows_absent']
    assert outcome.passed, outcome.detail


def test_result_set_equality():
    outcome = _outcomes()['result_set_equality']
    assert outcome.passed, outcome.detail


def test_plan_cost_readout_exact():
    outcome = _outcomes()['plan_cost_readout_exact']
    assert outcome.passed, outcome.detail


def test_plan_faster_than_baseline():
    outcome = _outcomes()['plan_faster_than_baseline']
    assert outcome.passed, outcome.detail


def test_reference_scores_full_reward():
    verdict = _verdict_over(REFERENCE)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded'


def test_rejects_guessed_neighbouring_row():
    verdict = _verdict_over(REJECTING / 'guessed_neighbouring_row')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'quarantine-row-misidentified', verdict['reason']
    assert verdict['failed_checker'] == 'quarantine_row_identified'


def test_rejects_closure_stopped_at_first_hop():
    verdict = _verdict_over(REJECTING / 'closure_stopped_at_first_hop')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'closure-incomplete', verdict['reason']
    assert verdict['failed_checker'] == 'closure_set_complete'


def test_rejects_cardinality_minus_one():
    verdict = _verdict_over(REJECTING / 'cardinality_minus_one')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'closure-cardinality-unattested', verdict['reason']
    assert verdict['failed_checker'] == 'closure_cardinality_attested'


def test_rejects_cardinality_plus_one():
    verdict = _verdict_over(REJECTING / 'cardinality_plus_one')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'closure-cardinality-unattested', verdict['reason']
    assert verdict['failed_checker'] == 'closure_cardinality_attested'


def test_rejects_layers_flattened():
    verdict = _verdict_over(REJECTING / 'layers_flattened')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'traversal-order-unwitnessed', verdict['reason']
    assert verdict['failed_checker'] == 'closure_layers_witnessed'


def test_rejects_plan_admits_every_reading():
    verdict = _verdict_over(REJECTING / 'plan_admits_every_reading')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'excluded-row-admitted', verdict['reason']
    assert verdict['failed_checker'] == 'excluded_rows_absent'


def test_rejects_plan_drops_one_admitted_row():
    verdict = _verdict_over(REJECTING / 'plan_drops_one_admitted_row')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'result-set-divergence', verdict['reason']
    assert verdict['failed_checker'] == 'result_set_equality'


def test_rejects_cost_readout_mismatch():
    verdict = _verdict_over(REJECTING / 'cost_readout_mismatch')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'cost-readout-mismatch', verdict['reason']
    assert verdict['failed_checker'] == 'plan_cost_readout_exact'


def test_rejects_baseline_plan_resubmitted():
    verdict = _verdict_over(REJECTING / 'baseline_plan_resubmitted')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'plan-not-faster-than-baseline', verdict['reason']
    assert verdict['failed_checker'] == 'plan_faster_than_baseline'


def test_rejects_plan_writes_to_the_warehouse():
    verdict = _verdict_over(REJECTING / 'plan_writes_to_the_warehouse')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'plan-not-read-only-single-statement', verdict['reason']
    assert verdict['failed_checker'] == 'plan_is_single_read_only_select'


def _submission(workspace):
    return json.loads((workspace / 'submission.json').read_text(encoding='utf-8'))


def test_near_miss_guessed_neighbouring_row_moves_one_value_only():
    reference = _submission(REFERENCE)
    fixture = _submission(REJECTING / 'guessed_neighbouring_row')
    moved = sorted(k for k in set(reference) | set(fixture) if reference.get(k) != fixture.get(k))
    assert moved == ['excluded_node_id'], moved
    reference_plan = (REFERENCE / 'plan.sql').read_bytes()
    fixture_plan = (REJECTING / 'guessed_neighbouring_row' / 'plan.sql').read_bytes()
    assert reference_plan == fixture_plan


def test_near_miss_cardinality_minus_one_moves_one_value_only():
    reference = _submission(REFERENCE)
    fixture = _submission(REJECTING / 'cardinality_minus_one')
    moved = sorted(k for k in set(reference) | set(fixture) if reference.get(k) != fixture.get(k))
    assert moved == ['closure_cardinality'], moved
    reference_plan = (REFERENCE / 'plan.sql').read_bytes()
    fixture_plan = (REJECTING / 'cardinality_minus_one' / 'plan.sql').read_bytes()
    assert reference_plan == fixture_plan


def test_near_miss_cardinality_plus_one_moves_one_value_only():
    reference = _submission(REFERENCE)
    fixture = _submission(REJECTING / 'cardinality_plus_one')
    moved = sorted(k for k in set(reference) | set(fixture) if reference.get(k) != fixture.get(k))
    assert moved == ['closure_cardinality'], moved
    reference_plan = (REFERENCE / 'plan.sql').read_bytes()
    fixture_plan = (REJECTING / 'cardinality_plus_one' / 'plan.sql').read_bytes()
    assert reference_plan == fixture_plan


def test_near_miss_cost_readout_mismatch_moves_one_value_only():
    reference = _submission(REFERENCE)
    fixture = _submission(REJECTING / 'cost_readout_mismatch')
    moved = sorted(k for k in set(reference) | set(fixture) if reference.get(k) != fixture.get(k))
    assert moved == ['reported_plan_cost'], moved
    reference_plan = (REFERENCE / 'plan.sql').read_bytes()
    fixture_plan = (REJECTING / 'cost_readout_mismatch' / 'plan.sql').read_bytes()
    assert reference_plan == fixture_plan


def test_empty_workspace_scores_zero_with_a_reason(tmp_path):
    harness = grade.build_harness(BUNDLE, tmp_path)
    bound = grade.load_bound(BUNDLE)
    verdict = grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))
    assert verdict['reward'] == 0.0
    assert verdict['reason'] == checkers.REASON_PLAN_SHAPE

