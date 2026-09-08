"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus the rejecting fixtures and the end-to-end reward
test. Every test drives the real checkers in tests/checkers.py over the harness record stream
the probe emits from the BUILT fence state. No test reads a clock, and no expectation below is
a transcribed literal: the accepting side reads the oracle's frozen submission and the rejecting
side reads a frozen fixture that departs from it in exactly one named way.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

import pytest  # noqa: E402

import checkers  # noqa: E402
import grade  # noqa: E402
import runner  # noqa: E402

REFERENCE = BUNDLE / 'solution' / 'fixtures' / 'reference_run'
REJECTING = BUNDLE / 'solution' / 'fixtures' / 'rejecting'

# The held-out fixtures are not present on every surface this suite can be run from. Skipping is
# the honest outcome there; the reward path in tests/test.sh does not depend on this suite.
pytestmark = pytest.mark.skipif(
    not (REFERENCE / 'submission.json').is_file(),
    reason='the solution fixtures are not mounted on this surface',
)


def _telemetry():
    return runner.observe()


def _harness_over(document):
    harness = checkers.Harness(telemetry=_telemetry())
    harness._cache['submission'] = document
    return harness


def _fixture(name):
    return json.loads((REJECTING / (name + '.json')).read_text(encoding='utf-8'))


def _reference():
    return json.loads((REFERENCE / 'submission.json').read_text(encoding='utf-8'))


def _outcomes(document):
    harness = _harness_over(document)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def _verdict_for(document):
    harness = _harness_over(document)
    bound = grade.load_bound(BUNDLE)
    return grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes(_reference()))


def test_fence_state_unmodified_accepts_the_oracle():
    outcome = _outcomes(_reference())['fence_state_unmodified']
    assert outcome.passed, outcome.detail


def test_crossing_count_accounted_accepts_the_oracle():
    outcome = _outcomes(_reference())['crossing_count_accounted']
    assert outcome.passed, outcome.detail


def test_admitted_kind_set_exact_accepts_the_oracle():
    outcome = _outcomes(_reference())['admitted_kind_set_exact']
    assert outcome.passed, outcome.detail


def test_verdict_sequence_ordered_accepts_the_oracle():
    outcome = _outcomes(_reference())['verdict_sequence_ordered']
    assert outcome.passed, outcome.detail


def test_verdict_records_match_traversal_accepts_the_oracle():
    outcome = _outcomes(_reference())['verdict_records_match_traversal']
    assert outcome.passed, outcome.detail


def test_no_unadmitted_kind_admitted_accepts_the_oracle():
    outcome = _outcomes(_reference())['no_unadmitted_kind_admitted']
    assert outcome.passed, outcome.detail


def test_quota_ledger_respected_accepts_the_oracle():
    outcome = _outcomes(_reference())['quota_ledger_respected']
    assert outcome.passed, outcome.detail


def test_admission_not_blanket_accepts_the_oracle():
    outcome = _outcomes(_reference())['admission_not_blanket']
    assert outcome.passed, outcome.detail


def test_admit_everything_is_rejected():
    verdict = _verdict_for(_fixture('admit_everything'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'fence-breached', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'no_unadmitted_kind_admitted', json.dumps(verdict, sort_keys=True)


def test_admitted_set_minus_one_kind_is_rejected():
    verdict = _verdict_for(_fixture('admitted_set_minus_one_kind'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'admitted-set-wrong', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'admitted_kind_set_exact', json.dumps(verdict, sort_keys=True)


def test_admitted_set_plus_one_kind_is_rejected():
    verdict = _verdict_for(_fixture('admitted_set_plus_one_kind'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'admitted-set-wrong', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'admitted_kind_set_exact', json.dumps(verdict, sort_keys=True)


def test_admitted_set_status_only_is_rejected():
    verdict = _verdict_for(_fixture('admitted_set_status_only'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'admitted-set-wrong', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'admitted_kind_set_exact', json.dumps(verdict, sort_keys=True)


def test_crossing_count_minus_one_is_rejected():
    verdict = _verdict_for(_fixture('crossing_count_minus_one'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'crossing-count-unaccounted', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'crossing_count_accounted', json.dumps(verdict, sort_keys=True)


def test_crossing_count_plus_one_is_rejected():
    verdict = _verdict_for(_fixture('crossing_count_plus_one'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'crossing-count-unaccounted', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'crossing_count_accounted', json.dumps(verdict, sort_keys=True)


def test_quota_ignored_is_rejected():
    verdict = _verdict_for(_fixture('quota_ignored'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'quota-ledger-exceeded', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'quota_ledger_respected', json.dumps(verdict, sort_keys=True)


def test_reasons_flattened_is_graded_at_zero():
    verdict = _verdict_for(_fixture('reasons_flattened'))
    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)


def test_reasons_precedence_inverted_is_graded_at_partial():
    verdict = _verdict_for(_fixture('reasons_precedence_inverted'))
    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)
    assert verdict['reward'] == 0.222222, json.dumps(verdict, sort_keys=True)


def test_refuse_everything_is_rejected():
    verdict = _verdict_for(_fixture('refuse_everything'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'blanket-refusal', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'admission_not_blanket', json.dumps(verdict, sort_keys=True)


def test_verdict_kinds_relabelled_is_rejected():
    verdict = _verdict_for(_fixture('verdict_kinds_relabelled'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'verdict-record-divergence', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'verdict_records_match_traversal', json.dumps(verdict, sort_keys=True)


def test_verdicts_from_disk_order_is_rejected():
    verdict = _verdict_for(_fixture('verdicts_from_disk_order'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'crossing-count-unaccounted', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'crossing_count_accounted', json.dumps(verdict, sort_keys=True)


def test_verdicts_from_wrong_chain_order_is_rejected():
    verdict = _verdict_for(_fixture('verdicts_from_wrong_chain_order'))
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'verdict-sequence-disordered', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] == 'verdict_sequence_ordered', json.dumps(verdict, sort_keys=True)


def test_the_oracle_scores_full_reward():
    verdict = _verdict_for(_reference())
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)


def test_neither_discovery_value_is_in_an_agent_visible_byte():
    harness = _harness_over(_reference())
    count = str(checkers.crossing_count(harness))
    admitted = checkers.admitted_kind_set(harness)
    offenders = []
    for surface in ('instruction.md', 'task.toml', 'environment', 'tests'):
        root = BUNDLE / surface
        if not root.exists():
            continue
        paths = [root] if root.is_file() else [p for p in root.rglob('*') if p.is_file()]
        for path in paths:
            if '__pycache__' in path.as_posix() or path.name == 'test_output.py':
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
            for kind in admitted:
                if kind in text:
                    offenders.append(path.as_posix() + ' names admitted kind ' + kind)
            if ','.join(admitted) in text:
                offenders.append(path.as_posix() + ' names the admitted set')
            if re.search(r'(?<![0-9A-Za-z_])' + count + r'(?![0-9A-Za-z_])', text):
                offenders.append(path.as_posix() + ' names the crossing count')
    assert not offenders, offenders
