"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml via solution/recompute.py.

One test per compiled rubric item. Each reads the outcome vector the
verifier wrote for the run being graded and nothing else, so a compiled
test can never disagree with the score for the same run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkers import outcomes


def _o():
    return outcomes.load()


def test_frozen_batch_contract():
    # reduction kind INVARIANT
    assert _o()['frozen_batch_contract'] is True

def test_frozen_init_digest():
    # reduction kind VALUE
    assert _o()['frozen_init_digest'] is True

def test_frozen_architecture_signature():
    # reduction kind VALUE
    assert _o()['frozen_architecture_signature'] is True

def test_frozen_corpus_digest():
    # reduction kind VALUE
    assert _o()['frozen_corpus_digest'] is True

def test_telemetry_bound_to_submission():
    # reduction kind INVARIANT
    assert _o()['telemetry_bound_to_submission'] is True

def test_no_submission_writes():
    # reduction kind ABSENCE
    assert _o()['no_submission_writes'] is True

def test_optimizer_step_moves_parameters():
    # reduction kind EFFECT
    assert _o()['optimizer_step_moves_parameters'] is True

def test_crossing_sustained():
    # reduction kind ORDERING
    assert _o()['crossing_sustained'] is True

def test_report_matches_telemetry():
    # reduction kind DIVERGENCE
    assert _o()['report_matches_telemetry'] is True

def test_steps_to_target_within_baseline():
    # reduction kind VALUE
    assert _o()['steps_to_target_within_baseline'] is True

def test_no_verbatim_record_copy():
    # reduction kind ABSENCE
    assert _o()['no_verbatim_record_copy'] is True
