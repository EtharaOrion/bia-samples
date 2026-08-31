# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml.
"""Compiled per-checker tests. One test per declared checker, both halves.

Each test drives the LIVE gate chain in tests/grade.py over a fixture from
tests/fixtures.json and asserts the accepting half reaches full reward and the
rejecting half reaches exactly that checker's zero_reason. The fixtures are
derived, and the boundary on what that proves is recorded in the fixture file
itself and in seed/tasks/OER-01/feasibility.yaml.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grade

FIXTURES = json.loads((HERE / 'fixtures.json').read_text())
BOUND = json.loads((HERE / 'anchors.json').read_text())


def case(ident):
    for row in FIXTURES['cases']:
        if row['id'] == ident:
            return row
    raise KeyError(ident)


def score(ident):
    row = case(ident)
    bound = dict(BOUND)
    bound['admin_plane'] = dict(bound['admin_plane'])
    bound['admin_plane'].update(row.get('admin_override') or {})
    return grade.grade(row['telemetry'], row['screen'], bound)


def test_reference_reaches_full_reward():
    payload = score('golden_reference_accepted')
    assert payload['reward'] == 1.0
    assert payload['reason'] == "crossing-sustained-and-recipe-displaced"


def test_reward_is_a_graded_float_and_never_binary():
    payload = score('partial_credit_is_a_graded_float')
    assert 0.0 < payload['reward'] < 1.0


def test_every_top_level_numeric_key_is_the_single_graded_scalar():
    payload = score('golden_reference_accepted')
    numeric = sorted(k for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    assert numeric == ['reward']


def test_recipe_diverges_from_the_pinned_record_corpus():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["recipe_diverges_from_the_pinned_record_corpus"]['passed'] is True
    for ident in ["nc_jittered_replay_record_46", "nc_no_op", "nc_replay_record_46", "nc_stale_answer_pre_mutation"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "recipe-replays-published-record"


def test_fingerprint_screen_precedes_any_seed():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["fingerprint_screen_precedes_any_seed"]['passed'] is True
    for ident in ["nc_screen_after_seeds"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "screen-ran-after-accelerator-time-was-spent"


def test_graded_loss_recomputed_unsmoothed_by_verifier():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["graded_loss_recomputed_unsmoothed_by_verifier"]['passed'] is True
    for ident in ["nc_readout_smoothing"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "readout-smoothed-on-the-graded-path"


def test_crossing_sustained_across_verifier_scheduled_points():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["crossing_sustained_across_verifier_scheduled_points"]['passed'] is True
    for ident in ["nc_unsustained_crossing"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "crossing-not-sustained"


def test_no_early_stop_before_the_sustain_window_closed():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["no_early_stop_before_the_sustain_window_closed"]['passed'] is True
    for ident in ["nc_early_stop"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "run-halted-before-sustain-window-closed"


def test_graded_weights_are_harness_owned_at_the_graded_step():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["graded_weights_are_harness_owned_at_the_graded_step"]['passed'] is True
    for ident in ["nc_submission_selected_checkpoint"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "graded-weights-not-harness-owned"


def test_graded_run_produced_verifier_owned_telemetry():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["graded_run_produced_verifier_owned_telemetry"]['passed'] is True
    for ident in ["nc_no_telemetry"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "graded-run-produced-no-telemetry"


def test_frozen_axes_unmoved_by_the_graded_run():
    accepted = score('golden_reference_accepted')
    verdicts = {item['id']: item for item in accepted['checkers']}
    assert verdicts["frozen_axes_unmoved_by_the_graded_run"]['passed'] is True
    for ident in ["nc_frozen_axis_moved"]:
        refused = score(ident)
        assert refused['reward'] == 0.0
        assert refused['reason'] == "frozen-axis-moved"
