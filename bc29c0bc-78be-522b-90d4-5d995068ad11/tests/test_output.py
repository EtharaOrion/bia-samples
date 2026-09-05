#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py

One compiled test per graded checker, each proving BOTH halves: the clean
fixture accepts, and that checker's planted-defect fixture produces exactly
that checker's own zero_reason. Two of the twelve carry their rejecting half
on a planted fixture because no submission can fire them while the isolation
boundary holds; that substitution is recorded in tests/checkers.yaml and in
seed/tasks/OER-05/feasibility.yaml.

Runs under pytest and also standalone as `python3 tests/test_output.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import grade  # noqa: E402

FIXTURES = HERE / 'fixtures'


def load(name):
    return json.loads((FIXTURES / (name + '.json')).read_text(encoding='utf-8'))


def verdict(selector, ctx):
    return getattr(checkers, 'check_' + selector)(ctx)


def test_truth_computed_after_submission_exit():
    clean = verdict('truth_computed_after_submission_exit', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('truth_computed_after_submission_exit', load('defect_truth_before_exit'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'truth-computed-before-submission-exit', planted.reason


def test_grading_tree_not_writable_from_submission():
    clean = verdict('grading_tree_not_writable_from_submission', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('grading_tree_not_writable_from_submission', load('defect_grading_tree_mutated'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'grading-tree-mutated', planted.reason


def test_no_submission_number_on_graded_metric():
    clean = verdict('no_submission_number_on_graded_metric', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('no_submission_number_on_graded_metric', load('defect_submission_number_on_graded_path'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'submission-supplied-number-on-graded-path', planted.reason


def test_graded_crossing_recomputed_unsmoothed():
    clean = verdict('graded_crossing_recomputed_unsmoothed', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('graded_crossing_recomputed_unsmoothed', load('defect_readout_smoothed'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'readout-smoothing-on-graded-path', planted.reason


def test_early_stop_yields_no_crossing():
    clean = verdict('early_stop_yields_no_crossing', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('early_stop_yields_no_crossing', load('defect_early_stop'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'early-stop-without-sustained-window', planted.reason


def test_crossing_sustained_across_scheduled_evals():
    clean = verdict('crossing_sustained_across_scheduled_evals', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('crossing_sustained_across_scheduled_evals', load('defect_crossing_not_sustained'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'crossing-not-sustained', planted.reason


def test_evaluated_weights_are_harness_owned():
    clean = verdict('evaluated_weights_are_harness_owned', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('evaluated_weights_are_harness_owned', load('defect_weights_not_harness_owned'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'weights-not-harness-owned', planted.reason


def test_behavioural_probe_deterministic():
    clean = verdict('behavioural_probe_deterministic', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('behavioural_probe_deterministic', load('defect_probe_nondeterministic'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'probe-digest-nondeterministic', planted.reason


def test_probe_transcript_bound_to_training_deltas():
    clean = verdict('probe_transcript_bound_to_training_deltas', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('probe_transcript_bound_to_training_deltas', load('defect_probe_transcript_diverges'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'probe-transcript-diverges-from-harness-deltas', planted.reason


def test_corpus_pinned_before_novelty_verdict():
    clean = verdict('corpus_pinned_before_novelty_verdict', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('corpus_pinned_before_novelty_verdict', load('defect_novelty_verdict_predates_pin'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'novelty-verdict-predates-corpus-pin', planted.reason


def test_behavioural_novelty_clears_corpus():
    clean = verdict('behavioural_novelty_clears_corpus', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('behavioural_novelty_clears_corpus', load('defect_behaviour_matches_corpus'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'behaviour-matches-corpus-entry', planted.reason


def test_multi_seed_separation_clears_margin():
    clean = verdict('multi_seed_separation_clears_margin', load('clean_accepting'))
    assert clean.ok, 'accepting half failed: ' + clean.detail
    planted = verdict('multi_seed_separation_clears_margin', load('defect_separation_below_floor'))
    assert not planted.ok, 'rejecting half did not fire'
    assert planted.reason == 'separation-below-noise-floor', planted.reason


def test_clean_fixture_reaches_full_reward():
    document = grade.grade(load('clean_accepting'))
    assert document['reward'] == 1.0, document
    assert document['reason'] == 'graded', document


def test_reward_is_continuous_and_never_binary():
    rewards = [grade.grade(load(name))['reward'] for name in (
        'clean_accepting', 'partial_continuous', 'margin_ramp_continuous',
    )]
    assert rewards[0] == 1.0, rewards
    assert 0.0 < rewards[2] < rewards[1] < rewards[0], rewards


def test_absent_anchors_are_refused_rather_than_substituted():
    document = grade.grade(load('control_anchors_absent'))
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'anchor-pair-absent', document


def test_stale_control_ctl_novelty_verdict_taken_before_corpus_pin():
    document = grade.grade(load('defect_novelty_verdict_predates_pin'))
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'novelty-verdict-predates-corpus-pin', document


def test_stale_control_ctl_novelty_verdict_carried_from_probe_set_a():
    document = grade.grade(load('control_stale_probe_set'))
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'behaviour-matches-corpus-entry', document


def test_stale_control_ctl_crossing_sustained_under_two_point_window():
    document = grade.grade(load('control_stale_sustain_window'))
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'crossing-not-sustained', document


def test_stale_control_ctl_weights_accepted_under_pre_mutation_custody():
    document = grade.grade(load('control_stale_weight_custody'))
    assert document['reward'] == 0.0, document
    assert document['reason'] == 'weights-not-harness-owned', document


def main():
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
        except AssertionError as exc:
            failures.append(name + ': ' + str(exc))
    for row in failures:
        print('FAIL ' + row)
    print(('FAILED ' + str(len(failures))) if failures else 'compiled tests: all passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
