#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Derived from solution/grounding.yaml by solution/recompute.py.

The compiled surface. tests/test.sh runs this BEFORE tests/grade.py, so a
manifest whose bindings have drifted away from the checkers aborts the verifier
before anything is graded and the EXIT trap writes an attributed zero rather than
a number resting on a stale binding.

Every test below is either a pure text binding check over frozen bundle bytes or a
drive of one live selector over a frozen in-memory fixture. It reads no clock,
opens no socket, consults no random source, imports no submission and runs no
composition. Grading the run is tests/grade.py's job, not this file's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402

MANIFEST = (HERE / "checkers.yaml").read_text(encoding="utf-8")
CHECKER_SOURCE = (HERE / "checkers.py").read_text(encoding="utf-8")
GRADE_SOURCE = (HERE / "grade.py").read_text(encoding="utf-8")

REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_DOCUMENT_PATH = "/logs/verifier/score.json"


def block(ident: str) -> str:
    marker = "\n  - id: " + ident + "\n"
    if marker not in MANIFEST:
        raise AssertionError("tests/checkers.yaml declares no checker " + ident)
    tail = MANIFEST.split(marker, 1)[1]
    return tail.split("\n  - id: ", 1)[0]


def bind(ident: str, reduction: str, selector: str, zero_reason: str) -> None:
    body = block(ident)
    for needle in (
        "reduction: " + reduction,
        "selector: " + selector,
        "zero_reason: " + zero_reason,
        "required: true",
        "reached_by: tests/grade.py",
        'compiled_test: "tests/test_output.py::test_' + ident + '"',
    ):
        if needle not in body:
            raise AssertionError(ident + " does not bind " + repr(needle) + " in tests/checkers.yaml")
    if "def " + selector + "(" not in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py defines no " + selector)
    if selector not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py never reaches " + selector)
    if '"' + zero_reason + '"' not in CHECKER_SOURCE:
        raise AssertionError(selector + " never emits the zero reason " + zero_reason)
    if '"' + ident + '": "' + zero_reason + '"' not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py maps " + ident + " to a different zero reason")


def test_reward_contract_is_bound() -> None:
    for needle in (
        "reward_path: " + REWARD_PATH,
        "score_document_path: " + SCORE_DOCUMENT_PATH,
        "mode: required_pass",
    ):
        if needle not in MANIFEST:
            raise AssertionError("tests/checkers.yaml does not bind " + repr(needle))
    if "reward.txt" not in GRADE_SOURCE or "score.json" not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py does not write both reward carriers")


def test_anchors_declared_absent_not_invented() -> None:
    for needle in ("anchors_state: absent", "gap-oer-per-family-anchors-unmeasured"):
        if needle not in MANIFEST:
            raise AssertionError("tests/checkers.yaml does not declare " + repr(needle))
    if '"baseline_metric": None' not in GRADE_SOURCE or '"target_metric": None' not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py invents an anchor instead of declaring it absent")


def test_grading_process_never_imports_the_submission() -> None:
    for banned in ("import submission", "importlib", "exec(", "eval("):
        if banned in GRADE_SOURCE:
            raise AssertionError("tests/grade.py reaches the submission through " + banned)
    if "import random" in CHECKER_SOURCE or "import time" in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py reads a random source or a clock")


def test_end_to_end_gain_not_stage_local() -> None:
    bind("end_to_end_gain_not_stage_local", "EFFECT", "check_end_to_end_gain_not_stage_local", "stage-local-optimum-degrades-composition")


def test_eval_split_not_trained_on() -> None:
    bind("eval_split_not_trained_on", "ABSENCE", "check_eval_split_not_trained_on", "evaluation-split-leaked-into-training")


def test_evaluation_point_is_bound_not_selected() -> None:
    bind("evaluation_point_is_bound_not_selected", "VALUE", "check_evaluation_point_is_bound_not_selected", "evaluation-point-submission-selected")


def test_floor_anchor_matches_bound_reference() -> None:
    bind("floor_anchor_matches_bound_reference", "VALUE", "check_floor_anchor_matches_bound_reference", "floor-anchor-not-the-bound-reference")


def test_frozen_axes_unmodified() -> None:
    bind("frozen_axes_unmodified", "VALUE", "check_frozen_axes_unmodified", "frozen-axis-modified")


def test_full_chain_completed_no_early_stop() -> None:
    bind("full_chain_completed_no_early_stop", "ABSENCE", "check_full_chain_completed_no_early_stop", "early-stop-loss-not-established")


def test_graded_loss_from_harness_weights() -> None:
    bind("graded_loss_from_harness_weights", "VALUE", "check_graded_loss_from_harness_weights", "graded-loss-not-from-harness-weights")


def test_loss_sustained_across_bound_evals() -> None:
    bind("loss_sustained_across_bound_evals", "INVARIANT", "check_loss_sustained_across_bound_evals", "improvement-not-sustained")


def test_no_smoothing_on_graded_path() -> None:
    bind("no_smoothing_on_graded_path", "VALUE", "check_no_smoothing_on_graded_path", "readout-smoothing-on-graded-path")


def test_session_state_carried_forward() -> None:
    bind("session_state_carried_forward", "INVARIANT", "check_session_state_carried_forward", "session-state-broken")


def test_stage_chain_order_observed() -> None:
    bind("stage_chain_order_observed", "ORDERING", "check_stage_chain_order_observed", "stage-chain-order-violated")


def test_stage_claim_matches_next_stage_consumption() -> None:
    bind("stage_claim_matches_next_stage_consumption", "DIVERGENCE", "check_stage_claim_matches_next_stage_consumption", "stage-claim-contradicted-by-chain")


def test_token_budget_respected_as_fed() -> None:
    bind("token_budget_respected_as_fed", "EFFECT", "check_token_budget_respected_as_fed", "token-budget-not-respected-as-fed")


# The discovery-value fixture pair, frozen from solution/grounding.yaml `observed`.
# The accepting half feeds the selector the reference loss the live environment
# establishes and the rejecting half feeds it one the environment does not, so a
# selector that stopped reading the value would fail one half or the other.
FLOOR_ANCHOR = {
    "established": 1.054387882,
    "unestablished": 1.154387882,
    "tolerance": 1e-06,
    "budget": 3072,
    "zero_reason": "floor-anchor-not-the-bound-reference",
}


def _floor_anchor_outcome(measured: float):
    handle = checkers.Harness(
        submission=HERE,
        verifier=HERE,
        telemetry={"floor": {"loss_per_byte": measured}},
        bound={
            "reference_loss_per_byte": FLOOR_ANCHOR["established"],
            "reference_loss_tolerance": FLOOR_ANCHOR["tolerance"],
            "reference_anchor_budget_tokens": FLOOR_ANCHOR["budget"],
            "token_budget_tokens": FLOOR_ANCHOR["budget"],
        },
        eval_split=HERE / "bound.json",
        folds={},
    )
    return checkers.check_floor_anchor_matches_bound_reference(handle)


def test_floor_anchor_is_bound_on_the_admin_plane() -> None:
    bound = json.loads((HERE / "bound.json").read_text(encoding="utf-8"))
    for key, expected in (
        ("reference_loss_per_byte", FLOOR_ANCHOR["established"]),
        ("reference_loss_tolerance", FLOOR_ANCHOR["tolerance"]),
        ("reference_anchor_budget_tokens", FLOOR_ANCHOR["budget"]),
    ):
        if key not in bound:
            raise AssertionError("tests/bound.json binds no " + key)
        if bound[key] != expected:
            raise AssertionError(
                "tests/bound.json binds " + key + "=" + repr(bound[key])
                + " against the grounded " + repr(expected)
            )


def test_floor_anchor_fixture_accepts_the_established_reference() -> None:
    outcome = _floor_anchor_outcome(FLOOR_ANCHOR["established"])
    if not outcome.passed:
        raise AssertionError("the established reference loss was refused: " + outcome.detail)


def test_floor_anchor_fixture_rejects_an_unestablished_reference() -> None:
    outcome = _floor_anchor_outcome(FLOOR_ANCHOR["unestablished"])
    if outcome.passed:
        raise AssertionError("an unestablished reference loss was accepted, so nothing reads the value")
    if outcome.reason != FLOOR_ANCHOR["zero_reason"]:
        raise AssertionError("the rejecting half carries the reason " + repr(outcome.reason))


def main() -> int:
    failures = []
    for name, case in sorted(globals().items()):
        if not name.startswith("test_") or not callable(case):
            continue
        try:
            case()
        except AssertionError as exc:
            failures.append(name + ": " + str(exc))
    for row in failures:
        print("FAIL " + row, file=sys.stderr)
    total = len([n for n in globals() if n.startswith("test_")])
    print("compiled surface: " + str(total - len(failures)) + "/" + str(total) + " bindings hold")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
