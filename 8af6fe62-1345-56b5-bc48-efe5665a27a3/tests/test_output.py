# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py
"""Compiled tests, one per checker declared in tests/checkers.yaml.

Each test drives the live checker against the bound runtime telemetry root.
There is no clock read, no random source, no network and no submission import
here; the checker does the reading and this file only asserts its verdict.
"""

import pathlib

import checkers

RUN_ROOT = pathlib.Path("/logs/verifier")


def test_graded_crossing_recomputed_unsmoothed():
    result = checkers.check_graded_crossing_recomputed_unsmoothed(RUN_ROOT)
    assert result.ok, 'readout-smoothed-on-graded-path' + ": " + result.detail


def test_crossing_sustained_across_scheduled_points():
    result = checkers.check_crossing_sustained_across_scheduled_points(RUN_ROOT)
    assert result.ok, 'crossing-not-sustained' + ": " + result.detail


def test_early_stop_is_not_a_crossing():
    result = checkers.check_early_stop_is_not_a_crossing(RUN_ROOT)
    assert result.ok, 'early-stop-without-sustained-crossing' + ": " + result.detail


def test_graded_weights_are_harness_owned():
    result = checkers.check_graded_weights_are_harness_owned(RUN_ROOT)
    assert result.ok, 'graded-weights-not-harness-owned' + ": " + result.detail


def test_multi_seed_mean_clears_margin():
    result = checkers.check_multi_seed_mean_clears_margin(RUN_ROOT)
    assert result.ok, 'multi-seed-separation-not-cleared' + ": " + result.detail


def test_iteration_sequence_ordered():
    result = checkers.check_iteration_sequence_ordered(RUN_ROOT)
    assert result.ok, 'iteration-sequence-out-of-order' + ": " + result.detail


def test_summary_agrees_with_durable_ledger():
    result = checkers.check_summary_agrees_with_durable_ledger(RUN_ROOT)
    assert result.ok, 'post-compaction-claim-contradicts-ledger' + ": " + result.detail


def test_compaction_dropped_entries_reconstructed():
    result = checkers.check_compaction_dropped_entries_reconstructed(RUN_ROOT)
    assert result.ok, 'compaction-dropped-entry-not-reconstructed' + ": " + result.detail


def test_frozen_axes_unmoved():
    result = checkers.check_frozen_axes_unmoved(RUN_ROOT)
    assert result.ok, 'frozen-axis-moved' + ": " + result.detail
