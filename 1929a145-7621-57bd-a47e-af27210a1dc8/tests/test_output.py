"""Compiled checker suite for slot OER-14.

GENERATED SECTION. DO NOT HAND-EDIT.

source: solution/grounding.yaml

Each test drives the REAL checker in tests/checkers.py over harness telemetry the
reference produced, and its planted-defect twin. Both halves of every checker.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import fixtures  # noqa: E402

CLEAN = fixtures.clean_context()


def test_suite_collected_at_least_one_case():
    """A zero-collected suite is never positive evidence that the suite ran."""
    assert len(checkers.REGISTRY) > 0


def test_graded_bpb_recomputed_unsmoothed():
    outcome = checkers.graded_bpb_recomputed_unsmoothed(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('graded_bpb_recomputed_unsmoothed')
    refused = checkers.graded_bpb_recomputed_unsmoothed(planted)
    assert not refused.passed
    assert refused.reason == 'readout-smoothing-on-graded-path'


def test_denominator_is_frozen_eval_byte_count():
    outcome = checkers.denominator_is_frozen_eval_byte_count(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('denominator_is_frozen_eval_byte_count')
    refused = checkers.denominator_is_frozen_eval_byte_count(planted)
    assert not refused.passed
    assert refused.reason == 'denominator-not-frozen-eval-bytes'


def test_compute_spend_within_frozen_budget():
    outcome = checkers.compute_spend_within_frozen_budget(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('compute_spend_within_frozen_budget')
    refused = checkers.compute_spend_within_frozen_budget(planted)
    assert not refused.passed
    assert refused.reason == 'compute-budget-overspent'


def test_evaluated_state_is_harness_owned():
    outcome = checkers.evaluated_state_is_harness_owned(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('evaluated_state_is_harness_owned')
    refused = checkers.evaluated_state_is_harness_owned(planted)
    assert not refused.passed
    assert refused.reason == 'state-not-harness-owned'


def test_early_stop_does_not_establish_reading():
    outcome = checkers.early_stop_does_not_establish_reading(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('early_stop_does_not_establish_reading')
    refused = checkers.early_stop_does_not_establish_reading(planted)
    assert not refused.passed
    assert refused.reason == 'early-stop-not-established'


def test_reading_sustained_across_scheduled_points():
    outcome = checkers.reading_sustained_across_scheduled_points(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('reading_sustained_across_scheduled_points')
    refused = checkers.reading_sustained_across_scheduled_points(planted)
    assert not refused.passed
    assert refused.reason == 'reading-not-sustained'


def test_no_submission_reported_number_on_graded_path():
    outcome = checkers.no_submission_reported_number_on_graded_path(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('no_submission_reported_number_on_graded_path')
    refused = checkers.no_submission_reported_number_on_graded_path(planted)
    assert not refused.passed
    assert refused.reason == 'submission-reported-metric-on-graded-path'


def test_reallocation_follows_flattening():
    outcome = checkers.reallocation_follows_flattening(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('reallocation_follows_flattening')
    refused = checkers.reallocation_follows_flattening(planted)
    assert not refused.passed
    assert refused.reason == 'flattened-direction-not-reallocated'


def test_carried_direction_frontier_never_collapses():
    outcome = checkers.carried_direction_frontier_never_collapses(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('carried_direction_frontier_never_collapses')
    refused = checkers.carried_direction_frontier_never_collapses(planted)
    assert not refused.passed
    assert refused.reason == 'carried-state-collapsed'


def test_graded_band_exceeds_single_direction_sweep():
    outcome = checkers.graded_band_exceeds_single_direction_sweep(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('graded_band_exceeds_single_direction_sweep')
    refused = checkers.graded_band_exceeds_single_direction_sweep(planted)
    assert not refused.passed
    assert refused.reason == 'single-direction-sweep-not-exceeded'


def test_corpus_phase_matches_frozen_offset():
    outcome = checkers.corpus_phase_matches_frozen_offset(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('corpus_phase_matches_frozen_offset')
    refused = checkers.corpus_phase_matches_frozen_offset(planted)
    assert not refused.passed
    assert refused.reason == 'frozen-corpus-phase-not-established'


def test_corpus_period_matches_frozen_stride():
    outcome = checkers.corpus_period_matches_frozen_stride(CLEAN)
    assert outcome.passed, outcome.detail
    planted = fixtures.planted_context('corpus_period_matches_frozen_stride')
    refused = checkers.corpus_period_matches_frozen_stride(planted)
    assert not refused.passed
    assert refused.reason == 'frozen-corpus-period-not-established'
