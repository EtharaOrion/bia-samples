#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml.

Compiled per-checker tests. Each one runs BOTH halves of its checker: the
accepting half over the golden session fixture and the rejecting half over a
planted-defect fixture that moves exactly the field that checker grades.

Runs standalone under python3 and needs no test framework.
"""

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from checkers import Session  # noqa: E402
from checkers import check_attempt_sequence_contiguous  # noqa: E402
from checkers import check_session_holds_attempt_budget  # noqa: E402
from checkers import check_carried_state_monotone  # noqa: E402
from checkers import check_reallocation_follows_flattening  # noqa: E402
from checkers import check_graded_weights_are_run_weights  # noqa: E402
from checkers import check_graded_readout_unsmoothed  # noqa: E402
from checkers import check_early_stop_is_not_a_crossing  # noqa: E402
from checkers import check_crossing_sustained  # noqa: E402
from checkers import check_target_band_reached_by_session  # noqa: E402
from checkers import check_multi_seed_separation_established  # noqa: E402
from checkers import check_reference_operating_point_matches_final_state  # noqa: E402

FIXTURES = TESTS / "fixtures"


def load(name):
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    session = Session()
    for row in payload["records"]:
        kind = row.get("kind")
        if kind == "session_header":
            session.header = row
        elif kind == "attempt":
            session.attempts.append(row)
        elif kind == "session_footer":
            session.footer = row
    return session


def _both(check, reject_name, reason):
    accepted = check(load("golden_session.json"))
    assert accepted.passed, check.__name__ + " rejected the golden session: " + accepted.detail
    rejected = check(load(reject_name))
    assert not rejected.passed, check.__name__ + " accepted its planted defect"
    assert rejected.zero_reason == reason, (
        check.__name__ + " emitted " + repr(rejected.zero_reason) + ", expected " + repr(reason)
    )


def test_attempt_sequence_contiguous():
    _both(check_attempt_sequence_contiguous, "reject_attempt_sequence_contiguous.json", "attempt-sequence-out-of-order")


def test_session_holds_attempt_budget():
    _both(check_session_holds_attempt_budget, "reject_session_holds_attempt_budget.json", "session-budget-not-held")


def test_carried_state_monotone():
    _both(check_carried_state_monotone, "reject_carried_state_monotone.json", "carried-state-not-accumulated")


def test_reallocation_follows_flattening():
    _both(check_reallocation_follows_flattening, "reject_reallocation_follows_flattening.json", "flattened-direction-never-abandoned")


def test_graded_weights_are_run_weights():
    _both(check_graded_weights_are_run_weights, "reject_graded_weights_are_run_weights.json", "checkpoint-substituted")


def test_graded_readout_unsmoothed():
    _both(check_graded_readout_unsmoothed, "reject_graded_readout_unsmoothed.json", "readout-smoothing-on-graded-path")


def test_early_stop_is_not_a_crossing():
    _both(check_early_stop_is_not_a_crossing, "reject_early_stop_is_not_a_crossing.json", "early-stop-not-a-crossing")


def test_crossing_sustained():
    _both(check_crossing_sustained, "reject_crossing_sustained.json", "crossing-not-sustained")


def test_target_band_reached_by_session():
    _both(check_target_band_reached_by_session, "reject_target_band_reached_by_session.json", "no-measured-improvement")


def test_multi_seed_separation_established():
    _both(check_multi_seed_separation_established, "reject_multi_seed_separation_established.json", "separation-not-established")


def test_reference_operating_point_matches_final_state():
    _both(check_reference_operating_point_matches_final_state, "reject_reference_operating_point_matches_final_state.json", "reference-operating-point-unheld")

CASES = (
    ("test_attempt_sequence_contiguous", test_attempt_sequence_contiguous),
    ("test_session_holds_attempt_budget", test_session_holds_attempt_budget),
    ("test_carried_state_monotone", test_carried_state_monotone),
    ("test_reallocation_follows_flattening", test_reallocation_follows_flattening),
    ("test_graded_weights_are_run_weights", test_graded_weights_are_run_weights),
    ("test_graded_readout_unsmoothed", test_graded_readout_unsmoothed),
    ("test_early_stop_is_not_a_crossing", test_early_stop_is_not_a_crossing),
    ("test_crossing_sustained", test_crossing_sustained),
    ("test_target_band_reached_by_session", test_target_band_reached_by_session),
    ("test_multi_seed_separation_established", test_multi_seed_separation_established),
    ("test_reference_operating_point_matches_final_state", test_reference_operating_point_matches_final_state),
)


def main():
    failures = []
    for name, case in CASES:
        try:
            case()
        except AssertionError as exc:
            failures.append(name + ": " + str(exc))
    for line in failures:
        print("FAIL " + line)
    if failures:
        return 1
    print("PASS " + str(len(CASES)) + " compiled checker tests, both halves each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
