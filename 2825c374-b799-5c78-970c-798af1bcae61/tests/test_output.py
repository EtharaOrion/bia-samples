"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus the end-to-end reward tests. Every test drives the
real checkers in tests/checkers.py over frozen fixtures that are real two-phase runs of the
bundle's own drivers. No test reads a clock; the cost model is an instrumented read counter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import grade  # noqa: E402

FIXTURES = BUNDLE / 'solution' / 'fixtures'
REFERENCE = FIXTURES / 'reference_run'
PLANTED = FIXTURES / 'planted'

TARGET_SPEEDUP = 12.012523
PARTIAL_REWARD = 0.630377

# The frozen fixtures live under solution/, which tests/Dockerfile deliberately does not copy
# into the verifier image, so a run inside that image reports these as skipped rather than as
# failures. They run wherever the whole bundle is resolvable, which is where they are evidence.
pytestmark = pytest.mark.skipif(
    not REFERENCE.is_dir(),
    reason='solution/fixtures is not resolvable from here; the verifier image omits solution/',
)


def _harness(root):
    root = Path(root)
    return grade.build_harness(
        BUNDLE, root / 'workspace', root / 'harness', root / 'app' / 'submission.py'
    )


def _verdict(root):
    bound = grade.load_bound(BUNDLE)
    return grade.score(_harness(root), bound, grade.load_bound_mapping(BUNDLE))


def _outcomes(root=REFERENCE):
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(_harness(root), bound)}


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes())


def test_frozen_inputs_unmodified():
    outcome = _outcomes()['frozen_inputs_unmodified']
    assert outcome.passed, outcome.detail


def test_agent_phase_journal_present():
    outcome = _outcomes()['agent_phase_journal_present']
    assert outcome.passed, outcome.detail


def test_journal_chain_intact():
    outcome = _outcomes()['journal_chain_intact']
    assert outcome.passed, outcome.detail


def test_splice_point_harness_established():
    outcome = _outcomes()['splice_point_harness_established']
    assert outcome.passed, outcome.detail


def test_carried_state_digest_matches():
    outcome = _outcomes()['carried_state_digest_matches']
    assert outcome.passed, outcome.detail


def test_kernel_identity_stable_across_splice():
    outcome = _outcomes()['kernel_identity_stable_across_splice']
    assert outcome.passed, outcome.detail


def test_block_coverage_ordered():
    outcome = _outcomes()['block_coverage_ordered']
    assert outcome.passed, outcome.detail


def test_resume_output_exact():
    outcome = _outcomes()['resume_output_exact']
    assert outcome.passed, outcome.detail


def test_charge_not_substituted():
    outcome = _outcomes()['charge_not_substituted']
    assert outcome.passed, outcome.detail


def test_reference_scores_full_reward():
    verdict = _verdict(REFERENCE)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded'
    assert verdict['metric']['speedup'] >= TARGET_SPEEDUP


def test_baseline_run_is_graded_at_zero_rather_than_refused():
    verdict = _verdict(FIXTURES / 'baseline_run')
    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] is None
    assert verdict['reward'] == 0.0


def test_partial_run_scores_between_the_anchors():
    verdict = _verdict(FIXTURES / 'partial_run')
    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)
    assert verdict['reward'] == PARTIAL_REWARD
    assert 0.0 < verdict['reward'] < 1.0


def test_splice_offset_plus_one_is_refused():
    verdict = _verdict(PLANTED / 'splice_offset_plus_one')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_SPLICE_NOT_ESTABLISHED
    assert verdict['failed_checker'] == 'splice_point_harness_established'


def test_carried_digest_one_nibble_is_refused():
    verdict = _verdict(PLANTED / 'carried_digest_one_nibble')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_CARRIED_DIGEST_FORGED
    assert verdict['failed_checker'] == 'carried_state_digest_matches'


def test_agent_phase_skipped_is_refused():
    verdict = _verdict(PLANTED / 'agent_phase_skipped')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_AGENT_PHASE_ABSENT
    assert verdict['failed_checker'] == 'agent_phase_journal_present'


def test_journal_row_edited_is_refused():
    verdict = _verdict(PLANTED / 'journal_row_edited')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_CHAIN_BROKEN
    assert verdict['failed_checker'] == 'journal_chain_intact'


def test_kernel_swapped_after_splice_is_refused():
    verdict = _verdict(PLANTED / 'kernel_swapped_after_splice')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_KERNEL_SWAPPED
    assert verdict['failed_checker'] == 'kernel_identity_stable_across_splice'


def test_the_two_near_misses_move_one_field_and_no_other_byte():
    accepting = json.loads((REFERENCE / 'workspace' / 'handoff.json').read_text(encoding='utf-8'))
    for name, field in (
        ('splice_offset_plus_one', 'splice_offset'),
        ('carried_digest_one_nibble', 'carried_state_sha256'),
    ):
        near = json.loads(
            (PLANTED / name / 'workspace' / 'handoff.json').read_text(encoding='utf-8')
        )
        assert set(near) == set(accepting)
        moved = [key for key in accepting if near[key] != accepting[key]]
        assert moved == [field], (name, moved)
        journal = (PLANTED / name / 'harness' / 'phase_a.jsonl').read_bytes()
        assert journal == (REFERENCE / 'harness' / 'phase_a.jsonl').read_bytes()
        kernel = (PLANTED / name / 'app' / 'submission.py').read_bytes()
        assert kernel == (REFERENCE / 'app' / 'submission.py').read_bytes()
