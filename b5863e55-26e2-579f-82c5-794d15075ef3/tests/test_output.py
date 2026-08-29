"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per checker, plus the reward range assertion. Each test
reads the outcome map grade.py wrote, so nothing here rests on the author's
word."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkers import outcomes

CHECKER_IDS = [
    "search_space_respected",
    "param_budget_met",
    "param_count_agreement",
    "frozen_surface_unchanged",
    "no_eval_bytes_in_training_stream",
    "no_frozen_recipe_tree_edit",
    "graded_event_order_held",
    "submission_run_mutated_telemetry",
    "advantage_held_across_run"
]


def _o():
    return outcomes.load()


def test_every_checker_reported():
    reported = _o()
    missing = [c for c in CHECKER_IDS if c not in reported]
    assert not missing, 'grade.py reported no outcome for ' + repr(missing)


def test_search_space_respected():
    assert _o()["search_space_respected"] is True


def test_param_budget_met():
    assert _o()["param_budget_met"] is True


def test_param_count_agreement():
    assert _o()["param_count_agreement"] is True


def test_frozen_surface_unchanged():
    assert _o()["frozen_surface_unchanged"] is True


def test_no_eval_bytes_in_training_stream():
    assert _o()["no_eval_bytes_in_training_stream"] is True


def test_no_frozen_recipe_tree_edit():
    assert _o()["no_frozen_recipe_tree_edit"] is True


def test_graded_event_order_held():
    assert _o()["graded_event_order_held"] is True


def test_submission_run_mutated_telemetry():
    assert _o()["submission_run_mutated_telemetry"] is True


def test_advantage_held_across_run():
    assert _o()["advantage_held_across_run"] is True


def test_reward_is_one_float_on_the_unit_interval():
    path = os.environ.get('BIA_SCORE_DETAIL', '/logs/verifier/score_detail.json')
    with open(path) as handle:
        payload = json.load(handle)
    score = payload['score']
    assert isinstance(score, float), 'reward must be a single float'
    assert 0.0 <= score <= 1.0, 'reward must sit on the closed interval zero to one'
    assert payload['reason'], 'every score must carry a machine-readable reason'
