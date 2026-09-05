#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Derived from solution/grounding.yaml by solution/recompute.py.

The compiled surface. tests/test.sh runs this BEFORE tests/grade.py, so a
manifest whose bindings have drifted away from the checkers aborts the verifier
before anything is graded and the EXIT trap writes an attributed zero rather than
a number resting on a stale binding.

Every test below is either a pure text binding check over frozen bundle bytes or a
drive of one live selector over a frozen in-memory fixture. It reads no clock,
opens no socket, consults no random source, imports no submission, loads no
checkpoint and runs no composition. Grading the run is tests/grade.py's job, not
this file's.

The fixture numbers below are synthetic inputs chosen to exercise a selector's two
halves. They are not measurements and no aggregate anywhere in this bundle reads
them: this family's anchors are absent, and the re-base onto the nanoGPT substrate
retired the bigram-era measurements rather than recycling them as fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402

MANIFEST = (HERE / "checkers.yaml").read_text(encoding="utf-8")
CHECKER_SOURCE = (HERE / "checkers.py").read_text(encoding="utf-8")
GRADE_SOURCE = (HERE / "grade.py").read_text(encoding="utf-8")
FROZEN_SOURCE = (HERE / "frozen" / "nanogpt.py").read_text(encoding="utf-8")
SUBSTRATE = json.loads((BUNDLE / "environment" / "nanogpt_substrate.json").read_text(encoding="utf-8"))

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
    bound = json.loads((HERE / "bound.json").read_text(encoding="utf-8"))
    if bound.get("reference_anchor_published") is not None:
        raise AssertionError("tests/bound.json publishes a reference anchor this family has not measured")
    if bound.get("reference_anchor_state") != "measured-in-run":
        raise AssertionError("tests/bound.json does not bind the floor anchor to the in-run measurement")


def test_grading_process_never_imports_the_submission() -> None:
    for banned in ("import submission", "importlib", "exec(", "eval("):
        if banned in GRADE_SOURCE:
            raise AssertionError("tests/grade.py reaches the submission through " + banned)
    if "import random" in CHECKER_SOURCE or "import time" in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py reads a random source or a clock")
    if "import torch" in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py breaks its bound import surface")


# --------------------------------------------------------------------------
# The substrate binding. The architecture the checkers derive shapes from is the
# architecture the canonical declaration carries, and nothing else.
# --------------------------------------------------------------------------
CANONICAL_ARCHITECTURE = {
    "vocab_size": 50304,
    "num_layers": 12,
    "model_dim": 768,
    "head_dim": 128,
    "num_heads": 6,
    "seq_len": 1024,
}


def test_substrate_declares_the_canonical_operating_point() -> None:
    architecture = SUBSTRATE.get("architecture") or {}
    for key, expected in CANONICAL_ARCHITECTURE.items():
        if architecture.get(key) != expected:
            raise AssertionError(
                "environment/nanogpt_substrate.json binds " + key + "=" + repr(architecture.get(key))
                + " against the canonical " + repr(expected)
            )
    run = SUBSTRATE.get("run") or {}
    for key, expected in (
        ("batch_tokens_per_step", 524288),
        ("forward_passes_per_step", 1),
        ("backward_passes_per_step", 1),
        ("target_val_loss", 3.28),
    ):
        if run.get(key) != expected:
            raise AssertionError(
                "the substrate run block binds " + key + "=" + repr(run.get(key))
                + " against the canonical " + repr(expected)
            )


def test_no_surrogate_model_survives_on_the_graded_path() -> None:
    """The re-base guard. A bigram is not a nanoGPT run and must not come back."""
    for name, source in (
        ("tests/checkers.py", CHECKER_SOURCE),
        ("tests/grade.py", GRADE_SOURCE),
    ):
        for banned in ("bigram", "add_k", "lambda_bigram"):
            if banned in source:
                raise AssertionError(name + " still carries the retired surrogate term " + repr(banned))
    for required in ("def forward(", "cross_entropy", "loss.backward()"):
        if required not in FROZEN_SOURCE:
            raise AssertionError("tests/frozen/nanogpt.py carries no " + required + ", so no pass is in the loop")


def test_no_forward_pass_means_no_score() -> None:
    """The graded scalar has exactly one producer, and it is a forward pass."""
    if FROZEN_SOURCE.count("def validation_loss(") != 1:
        raise AssertionError("the graded scalar does not have exactly one producer")
    head, _, tail = FROZEN_SOURCE.partition("def validation_loss(")
    if "model(tokens, targets)" not in tail:
        raise AssertionError("validation_loss returns a number without running the forward pass")
    if "recomputed_loss" not in CHECKER_SOURCE or "h.evaluate(" not in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py resolves the graded loss without the verifier's evaluator")
    if "val_loss" not in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py reads no validation loss at all")


def test_holdout_split_is_absent_from_the_agent_surface() -> None:
    eval_dir = BUNDLE / "environment" / "eval"
    for path in sorted(eval_dir.glob("*")):
        if path.suffix == ".bin":
            raise AssertionError("a validation shard sits on the agent surface at " + path.as_posix())
    folds = json.loads((eval_dir / "folds.json").read_text(encoding="utf-8"))
    if folds.get("bytes_present_here") is not False:
        raise AssertionError("environment/eval/folds.json does not declare the split bytes absent")
    bound = json.loads((HERE / "bound.json").read_text(encoding="utf-8"))
    root = str(bound.get("holdout_root", ""))
    if not root.startswith("/verifier/"):
        raise AssertionError("the held-out root " + repr(root) + " is not pinned to the verifier's own tree")
    if root in (BUNDLE / "environment").as_posix():
        raise AssertionError("the held-out root resolves inside the agent-visible environment tree")


# --------------------------------------------------------------------------
# The binding tests, one per graded checker.
# --------------------------------------------------------------------------
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


def test_graded_parameters_bind_frozen_architecture() -> None:
    bind("graded_parameters_bind_frozen_architecture", "VALUE", "check_graded_parameters_bind_frozen_architecture", "frozen-architecture-not-instantiated")


def test_holdout_split_is_verifier_owned() -> None:
    bind("holdout_split_is_verifier_owned", "ABSENCE", "check_holdout_split_is_verifier_owned", "holdout-split-not-verifier-owned")


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


# --------------------------------------------------------------------------
# The discovery-value fixture pair. The accepting half feeds the selector a floor
# the live environment established under the budget in force; the rejecting half
# feeds it one it did not, so a selector that stopped reading the value would fail
# one half or the other. Both halves are synthetic inputs and neither is a
# measurement of this family.
# --------------------------------------------------------------------------
STEP_BUDGET = 64
ANCHOR_SOURCE = "measured in-run by the verifier through the identical frozen protocol"


def _floor_anchor_outcome(floor_row: dict, baseline: float):
    handle = checkers.Harness(
        submission=HERE,
        verifier=HERE,
        telemetry={"floor": floor_row, "baseline": {"val_loss": baseline}},
        bound={
            "reference_anchor_state": "measured-in-run",
            "reference_anchor_published": None,
            "reference_anchor_step_budget": STEP_BUDGET,
            "reference_anchor_source": ANCHOR_SOURCE,
            "step_budget_steps": STEP_BUDGET,
        },
        holdout=[],
        folds={},
    )
    return checkers.check_floor_anchor_matches_bound_reference(handle)


def test_floor_anchor_is_bound_on_the_admin_plane() -> None:
    bound = json.loads((HERE / "bound.json").read_text(encoding="utf-8"))
    for key, expected in (
        ("reference_anchor_step_budget", STEP_BUDGET),
        ("step_budget_steps", STEP_BUDGET),
        ("reference_anchor_source", ANCHOR_SOURCE),
    ):
        if key not in bound:
            raise AssertionError("tests/bound.json binds no " + key)
        if bound[key] != expected:
            raise AssertionError(
                "tests/bound.json binds " + key + "=" + repr(bound[key])
                + " against the grounded " + repr(expected)
            )


def test_floor_anchor_fixture_accepts_an_established_reference() -> None:
    row = {"val_loss": 3.21, "step_budget": STEP_BUDGET, "source": ANCHOR_SOURCE}
    outcome = _floor_anchor_outcome(row, 3.44)
    if not outcome.passed:
        raise AssertionError("an established reference floor was refused: " + outcome.detail)


def test_floor_anchor_fixture_rejects_an_unestablished_reference() -> None:
    row = {"val_loss": 3.21, "step_budget": STEP_BUDGET - 1, "source": ANCHOR_SOURCE}
    outcome = _floor_anchor_outcome(row, 3.44)
    if outcome.passed:
        raise AssertionError("a floor from a different budget was accepted, so nothing reads the value")
    if outcome.reason != "floor-anchor-not-the-bound-reference":
        raise AssertionError("the rejecting half carries the reason " + repr(outcome.reason))


# --------------------------------------------------------------------------
# The architecture fixture pair. The accepting half is the shape map the frozen
# declaration implies; the rejecting half is a well-formed tensor set of a
# different architecture, which is exactly what a surrogate would present.
# --------------------------------------------------------------------------
def _architecture_outcome(shapes: dict):
    handle = checkers.Harness(
        submission=HERE,
        verifier=HERE,
        telemetry={"graded_model": {"path": (HERE / "bound.json").as_posix()}},
        bound={},
        holdout=[],
        folds={},
        substrate=SUBSTRATE,
        shapes_of=lambda _path: shapes,
    )
    return checkers.check_graded_parameters_bind_frozen_architecture(handle)


def test_architecture_fixture_accepts_the_declared_decoder() -> None:
    shapes = checkers.expected_shapes(CANONICAL_ARCHITECTURE)
    outcome = _architecture_outcome(shapes)
    if not outcome.passed:
        raise AssertionError("the declared architecture was refused: " + outcome.detail)


def test_architecture_fixture_rejects_a_different_architecture() -> None:
    narrower = dict(CANONICAL_ARCHITECTURE)
    narrower["num_layers"] = 6
    outcome = _architecture_outcome(checkers.expected_shapes(narrower))
    if outcome.passed:
        raise AssertionError("a six-layer tensor set was accepted as the frozen decoder")
    if outcome.reason != "frozen-architecture-not-instantiated":
        raise AssertionError("the rejecting half carries the reason " + repr(outcome.reason))


def test_architecture_fixture_rejects_a_count_table() -> None:
    outcome = _architecture_outcome({"unigram": [221], "bigram": [221, 221]})
    if outcome.passed:
        raise AssertionError("a count table was accepted as the frozen decoder")
    if outcome.reason != "frozen-architecture-not-instantiated":
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
