"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, one per rejecting fixture, and the end-to-end
reward test. Every test drives the real checkers in tests/checkers.py over frozen
producer documents that solution/recompute.py derived. No test reads a clock and no
test asserts a number this file authored.
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

REFERENCE = BUNDLE / 'solution' / 'fixtures' / 'reference_run' / 'producer_output.json'
PLANTED = BUNDLE / 'solution' / 'fixtures' / 'planted'


def _document(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _outcomes(path=REFERENCE):
    harness = grade.build_harness(BUNDLE, _document(path))
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def _verdict(path):
    harness = grade.build_harness(BUNDLE, _document(path))
    bound = grade.load_bound(BUNDLE)
    return grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes())


def test_instance_and_envelope_unmodified():
    outcome = _outcomes()['instance_and_envelope_unmodified']
    assert outcome.passed, outcome.detail


def test_submission_document_wellformed():
    outcome = _outcomes()['submission_document_wellformed']
    assert outcome.passed, outcome.detail


def test_seam_offset_recovered():
    outcome = _outcomes()['seam_offset_recovered']
    assert outcome.passed, outcome.detail


def test_construction_within_ground_set():
    outcome = _outcomes()['construction_within_ground_set']
    assert outcome.passed, outcome.detail


def test_construction_canonically_ordered():
    outcome = _outcomes()['construction_canonically_ordered']
    assert outcome.passed, outcome.detail


def test_seam_not_exploited():
    outcome = _outcomes()['seam_not_exploited']
    assert outcome.passed, outcome.detail


def test_construction_is_admissible():
    outcome = _outcomes()['construction_is_admissible']
    assert outcome.passed, outcome.detail


def test_seam_witness_exhibits_the_seam():
    outcome = _outcomes()['seam_witness_exhibits_the_seam']
    assert outcome.passed, outcome.detail


def test_reported_fitness_matches_recomputation():
    outcome = _outcomes()['reported_fitness_matches_recomputation']
    assert outcome.passed, outcome.detail


def test_rejects_construction_not_admissible():
    verdict = _verdict(PLANTED / 'construction_not_admissible.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'construction_is_admissible', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_NOT_ADMISSIBLE, verdict['reason']


def test_rejects_construction_out_of_ground_set():
    verdict = _verdict(PLANTED / 'construction_out_of_ground_set.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'construction_within_ground_set', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_OUT_OF_GROUND_SET, verdict['reason']


def test_rejects_construction_unordered():
    verdict = _verdict(PLANTED / 'construction_unordered.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'construction_canonically_ordered', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_UNORDERED, verdict['reason']


def test_rejects_fitness_readout_substituted():
    verdict = _verdict(PLANTED / 'fitness_readout_substituted.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'reported_fitness_matches_recomputation', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_FITNESS_SUBSTITUTED, verdict['reason']


def test_rejects_seam_exploited_block():
    verdict = _verdict(PLANTED / 'seam_exploited_block.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'seam_not_exploited', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_SEAM_EXPLOITED, verdict['reason']


def test_rejects_seam_offset_minus_one():
    verdict = _verdict(PLANTED / 'seam_offset_minus_one.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'seam_offset_recovered', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_SEAM_OFFSET, verdict['reason']


def test_rejects_seam_offset_plus_one():
    verdict = _verdict(PLANTED / 'seam_offset_plus_one.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'seam_offset_recovered', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_SEAM_OFFSET, verdict['reason']


def test_rejects_witness_inert():
    verdict = _verdict(PLANTED / 'witness_inert.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'seam_witness_exhibits_the_seam', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_WITNESS_INERT, verdict['reason']


def test_rejects_witness_undersized():
    verdict = _verdict(PLANTED / 'witness_undersized.json')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'seam_witness_exhibits_the_seam', verdict['failed_checker']
    assert verdict['reason'] == checkers.REASON_WITNESS_INERT, verdict['reason']


def test_the_near_miss_moves_only_the_discovery_value():
    accepting = _document(REFERENCE)
    for name in ('seam_offset_plus_one', 'seam_offset_minus_one'):
        near = _document(PLANTED / (name + '.json'))
        differing = [k for k in set(accepting) | set(near) if accepting.get(k) != near.get(k)]
        assert differing == ['seam_offset'], differing
        outcomes = _outcomes(PLANTED / (name + '.json'))
        failed = [i for i, row in outcomes.items() if not row.passed]
        assert failed == ['seam_offset_recovered'], failed


def test_the_seam_is_reachable_and_refused():
    exploit = _document(PLANTED / 'seam_exploited_block.json')
    bound = grade.load_bound(BUNDLE)
    harness = grade.build_harness(BUNDLE, exploit)
    proxy = checkers.proxy_certified_size(harness, bound)
    graded = checkers.graded_certified_size(harness, bound)
    assert proxy == len(exploit['construction']), proxy
    assert graded == 0, graded
    assert proxy > graded


def test_reference_scores_full_reward():
    verdict = _verdict(REFERENCE)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded'


def test_absent_submission_scores_zero_with_a_reason():
    bound = grade.load_bound(BUNDLE)
    harness = grade.build_harness(BUNDLE, None)
    verdict = grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))
    assert verdict['reward'] == 0.0
    assert verdict['reason'] == checkers.REASON_MALFORMED
