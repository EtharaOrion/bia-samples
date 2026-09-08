"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus the end-to-end reward test and one test
per frozen rejecting fixture. Every test drives the REAL checkers in tests/checkers.py
over a real index the harness rebuilt. No test reads a clock and no test asserts a
discovery value as a literal: every expectation is read back off the rebuilt index.
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

_TELEMETRY = None


def _telemetry():
    global _TELEMETRY
    if _TELEMETRY is None:
        import runner
        _TELEMETRY = runner.rebuild(BUNDLE)
    return _TELEMETRY


def _harness(workspace):
    return checkers.Harness(
        telemetry=_telemetry(),
        bundle_corpus_sha256=grade._digest(BUNDLE / 'environment' / 'corpus.jsonl'),
        bundle_index_spec_sha256=grade._digest(BUNDLE / 'environment' / 'index_spec.json'),
        workspace=workspace,
    )


def _outcomes(workspace):
    harness = _harness(workspace)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def _verdict(workspace):
    harness = _harness(workspace)
    bound = grade.load_bound(BUNDLE)
    return grade.score(harness, bound)


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes(REFERENCE))


def test_corpus_and_spec_unmodified():
    outcome = _outcomes(REFERENCE)['corpus_and_spec_unmodified']
    assert outcome.passed, outcome.detail


def test_submission_pairs_wellformed():
    outcome = _outcomes(REFERENCE)['submission_pairs_wellformed']
    assert outcome.passed, outcome.detail


def test_pair_listing_ordered():
    outcome = _outcomes(REFERENCE)['pair_listing_ordered']
    assert outcome.passed, outcome.detail


def test_adjudication_threshold_established():
    outcome = _outcomes(REFERENCE)['adjudication_threshold_established']
    assert outcome.passed, outcome.detail


def test_index_collision_census_agrees():
    outcome = _outcomes(REFERENCE)['index_collision_census_agrees']
    assert outcome.passed, outcome.detail


def test_adjudication_not_degenerate():
    outcome = _outcomes(REFERENCE)['adjudication_not_degenerate']
    assert outcome.passed, outcome.detail


def test_no_unsupported_pair():
    outcome = _outcomes(REFERENCE)['no_unsupported_pair']
    assert outcome.passed, outcome.detail


def test_collision_witness_verified():
    outcome = _outcomes(REFERENCE)['collision_witness_verified']
    assert outcome.passed, outcome.detail


def test_recall_sustained_across_shards():
    outcome = _outcomes(REFERENCE)['recall_sustained_across_shards']
    assert outcome.passed, outcome.detail


def test_reference_scores_full_reward():
    verdict = _verdict(REFERENCE)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded'


def test_the_discovery_values_are_read_off_the_rebuilt_index():
    harness = _harness(REFERENCE)
    bound = grade.load_bound(BUNDLE)
    assert harness.threshold() == bound.adjudication_threshold
    witness = harness.witness()
    assert witness is not None
    assert (witness['a'], witness['b']) == (bound.witness_a, bound.witness_b)


def test_rejects_threshold_minus_one_step():
    verdict = _verdict(REJECTING / 'threshold_minus_one_step')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'threshold-not-established', verdict['reason']


def test_rejects_threshold_plus_one_step():
    verdict = _verdict(REJECTING / 'threshold_plus_one_step')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'threshold-not-established', verdict['reason']


def test_rejects_threshold_guessed_round():
    verdict = _verdict(REJECTING / 'threshold_guessed_round')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'threshold-not-established', verdict['reason']


def test_rejects_witness_left_member_swapped():
    verdict = _verdict(REJECTING / 'witness_left_member_swapped')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'witness-not-colliding', verdict['reason']


def test_rejects_witness_right_member_swapped():
    verdict = _verdict(REJECTING / 'witness_right_member_swapped')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'witness-not-colliding', verdict['reason']


def test_rejects_witness_not_a_collision():
    verdict = _verdict(REJECTING / 'witness_not_a_collision')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'witness-not-colliding', verdict['reason']


def test_rejects_witness_runner_up_collision():
    verdict = _verdict(REJECTING / 'witness_runner_up_collision')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'witness-not-colliding', verdict['reason']


def test_rejects_every_pair_returned():
    verdict = _verdict(REJECTING / 'every_pair_returned')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'adjudication-degenerate', verdict['reason']


def test_rejects_no_pair_returned():
    verdict = _verdict(REJECTING / 'no_pair_returned')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'adjudication-degenerate', verdict['reason']


def test_rejects_collision_census_inflated():
    verdict = _verdict(REJECTING / 'collision_census_inflated')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'collision-census-diverges', verdict['reason']


def test_rejects_candidate_pairs_only():
    verdict = _verdict(REJECTING / 'candidate_pairs_only')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded', verdict['reason']

