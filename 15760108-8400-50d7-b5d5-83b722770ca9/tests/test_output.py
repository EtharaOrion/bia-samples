"""Deterministic rubric tests compiled from solution/grounding.yaml.

GENERATED SECTION. DO NOT HAND-EDIT.
Regenerate with: python3 solution/recompute.py --regen

Each test asserts one compiled rubric item against the outcomes map grade.py
wrote. These tests never re-derive an outcome, so a pytest verdict can never
disagree with the score the grader recorded.
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from checkers import outcomes


def _o():
    return outcomes.load()


def test_frontier_cardinality():
    # The submission evaluated exactly the frozen number of frontier points, indexed contiguously from zero.
    assert _o()["frontier_cardinality"] is True


def test_frozen_substrate_invariant():
    # The frozen architecture, step ceiling, one-forward-one-backward rule and per-run wall-clock cap held across every point.
    assert _o()["frozen_substrate_invariant"] is True


def test_no_unmanifested_telemetry():
    # No record the frozen runner did not write exists in the telemetry directory.
    assert _o()["no_unmanifested_telemetry"] is True


def test_runs_precede_seal():
    # Points were evaluated in sequence and every point ended before the record was sealed.
    assert _o()["runs_precede_seal"] is True


def test_training_moved_live_weights():
    # Every point's optimizer actually wrote to the live parameter tensors.
    assert _o()["training_moved_live_weights"] is True


def test_hypervolume_independently_agrees():
    # Two independent hypervolume implementations agree on the reward recomputed from the raw achieved objective values.
    assert _o()["hypervolume_independently_agrees"] is True


def test_graded_record_authored_by_the_verifier():
    # The record the reward was read off was produced by the verifier's own execution of the frozen runner, stamped with a nonce this verifier minted, over the deliverable present at grade time.
    assert _o()["graded_record_authored_by_the_verifier"] is True


def test_submission_ran_outside_the_measuring_process():
    # Every measurement in the record was authored by a process that never ran submitted bytes, with the submission hosted in a distinct worker process.
    assert _o()["submission_ran_outside_the_measuring_process"] is True


def test_frozen_harness_bytes_unmodified():
    # The three executable harness files carry the digests this bundle froze, at load and again at grading.
    assert _o()["frozen_harness_bytes_unmodified"] is True


def test_score_on_unit_interval():
    # The reward is a single float on the closed unit interval.
    s = _o()["_score"]
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_frontier_not_collapsed_diagnostic():
    # The achieved hypervolume is at least the hypervolume of its own best single point, which is the diagnostic separating a spread frontier from a collapsed one.
    score_path = pathlib.Path(os.environ.get("BIA_S09_SCORE", "/logs/verifier/score.json"))
    if not score_path.is_file():
        return
    d = json.loads(score_path.read_text())
    if "best_single_point_hypervolume" not in d:
        return
    assert d["best_single_point_hypervolume"] <= d["score"] + 1e-9
