"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus one per frozen rejecting fixture and the
end-to-end reward test. Every test drives the real checkers in tests/checkers.py over the
harness-minted store, never over a fixture the author planted with an answer in it.
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


def _outcomes(workspace):
    harness = grade.build_harness(BUNDLE, workspace)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def _verdict(workspace):
    harness = grade.build_harness(BUNDLE, workspace)
    bound = grade.load_bound(BUNDLE)
    return grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes(REFERENCE))


def test_frozen_inputs_unmodified():
    outcome = _outcomes(REFERENCE)['frozen_inputs_unmodified']
    assert outcome.passed, outcome.detail

def test_store_materialised():
    outcome = _outcomes(REFERENCE)['store_materialised']
    assert outcome.passed, outcome.detail

def test_attestation_order_realised():
    outcome = _outcomes(REFERENCE)['attestation_order_realised']
    assert outcome.passed, outcome.detail

def test_seal_chain_derived_from_state():
    outcome = _outcomes(REFERENCE)['seal_chain_derived_from_state']
    assert outcome.passed, outcome.detail

def test_attested_closure_complete():
    outcome = _outcomes(REFERENCE)['attested_closure_complete']
    assert outcome.passed, outcome.detail

def test_digest_recursion_holds():
    outcome = _outcomes(REFERENCE)['digest_recursion_holds']
    assert outcome.passed, outcome.detail

def test_terminal_atom_digest_matches():
    outcome = _outcomes(REFERENCE)['terminal_atom_digest_matches']
    assert outcome.passed, outcome.detail

def test_graded_cover_feasible():
    outcome = _outcomes(REFERENCE)['graded_cover_feasible']
    assert outcome.passed, outcome.detail

def test_rejects_natural_sort_order():
    verdict = _verdict(REJECTING / 'natural_sort_order')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'attestation-order-unrealised', verdict['reason']

def test_rejects_order_adjacent_transposed():
    verdict = _verdict(REJECTING / 'order_adjacent_transposed')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'attestation-order-unrealised', verdict['reason']

def test_rejects_seal_chain_fabricated():
    verdict = _verdict(REJECTING / 'seal_chain_fabricated')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'seal-chain-fabricated', verdict['reason']

def test_rejects_terminal_digest_one_hex():
    verdict = _verdict(REJECTING / 'terminal_digest_one_hex')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'terminal-digest-mismatch', verdict['reason']

def test_rejects_wrong_preimage_input_ids():
    verdict = _verdict(REJECTING / 'wrong_preimage_input_ids')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'digest-preimage-wrong', verdict['reason']

def test_reference_scores_full_reward():
    verdict = _verdict(REFERENCE)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded', verdict['reason']
