#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

The compiled tests tests/checkers.yaml names. Every declared checker carries both halves
here: an accepting fixture the checker must pass, and a rejecting fixture it must fail with
exactly its declared zero_reason. Two checkers are graded against planted fixtures because
an isolation boundary means no live submission can reach the window they grade; that
substitution is recorded in tests/checkers.yaml and in seed/tasks/OER-18/feasibility.yaml.

This file also enforces the checker import allowlist over the AST of tests/checkers.py, so a
checker that reached for a clock, a random source, a socket or the submission would fail the
instrument before it could grade anything.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import strata  # noqa: E402

FIXTURES = HERE / "fixtures"

ALLOWED_IMPORTS = {"__future__", "hashlib", "json", "math", "pathlib", "dataclasses", "typing"}


def load(name: str) -> checkers.Handle:
    payload = json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))
    return checkers.Handle(**payload["handle"])


def _verdict(selector: str, fixture: str):
    return getattr(checkers, selector)(load(fixture))


def test_checker_import_allowlist() -> None:
    tree = ast.parse((HERE / "checkers.py").read_text(encoding="utf-8"))
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0])
    assert seen <= ALLOWED_IMPORTS, "checkers.py imports outside the allowlist: " + repr(
        sorted(seen - ALLOWED_IMPORTS)
    )


def test_classifier_agrees_with_checkers() -> None:
    assert tuple(strata.STRATA) == tuple(checkers.STRATA)
    assert strata.UNCLASSIFIED == checkers.UNCLASSIFIED


def test_coverage_manifest_present() -> None:
    accepted = _verdict("check_coverage_manifest_present", "accept_golden")
    assert accepted.passed, "coverage_manifest_present rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_coverage_manifest_present", "reject_manifest_absent")
    assert not rejected.passed, "coverage_manifest_present accepted its rejecting fixture"
    assert rejected.zero_reason == "coverage-manifest-absent", (
        "coverage_manifest_present emitted " + rejected.zero_reason + " instead of coverage-manifest-absent"
    )


def test_coverage_divergence() -> None:
    accepted = _verdict("check_coverage_divergence", "accept_golden")
    assert accepted.passed, "coverage_divergence rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_coverage_divergence", "reject_over_declared")
    assert not rejected.passed, "coverage_divergence accepted its rejecting fixture"
    assert rejected.zero_reason == "coverage-declaration-contradicts-measurement", (
        "coverage_divergence emitted " + rejected.zero_reason + " instead of coverage-declaration-contradicts-measurement"
    )


def test_benchmark_contamination_absent() -> None:
    accepted = _verdict("check_benchmark_contamination_absent", "accept_golden")
    assert accepted.passed, "benchmark_contamination_absent rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_benchmark_contamination_absent", "reject_contaminated")
    assert not rejected.passed, "benchmark_contamination_absent accepted its rejecting fixture"
    assert rejected.zero_reason == "benchmark-near-duplicate-emitted", (
        "benchmark_contamination_absent emitted " + rejected.zero_reason + " instead of benchmark-near-duplicate-emitted"
    )


def test_training_budget_respected() -> None:
    accepted = _verdict("check_training_budget_respected", "accept_golden")
    assert accepted.passed, "training_budget_respected rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_training_budget_respected", "reject_overspend")
    assert not rejected.passed, "training_budget_respected accepted its rejecting fixture"
    assert rejected.zero_reason == "training-budget-overspent", (
        "training_budget_respected emitted " + rejected.zero_reason + " instead of training-budget-overspent"
    )


def test_corpus_frozen_before_training() -> None:
    accepted = _verdict("check_corpus_frozen_before_training", "accept_golden")
    assert accepted.passed, "corpus_frozen_before_training rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_corpus_frozen_before_training", "reject_corpus_mutated")
    assert not rejected.passed, "corpus_frozen_before_training accepted its rejecting fixture"
    assert rejected.zero_reason == "corpus-mutated-after-freeze", (
        "corpus_frozen_before_training emitted " + rejected.zero_reason + " instead of corpus-mutated-after-freeze"
    )


def test_evaluation_schedule_complete() -> None:
    accepted = _verdict("check_evaluation_schedule_complete", "accept_golden")
    assert accepted.passed, "evaluation_schedule_complete rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_evaluation_schedule_complete", "reject_early_stop")
    assert not rejected.passed, "evaluation_schedule_complete accepted its rejecting fixture"
    assert rejected.zero_reason == "early-stop-score-not-established", (
        "evaluation_schedule_complete emitted " + rejected.zero_reason + " instead of early-stop-score-not-established"
    )


def test_graded_score_from_harness_model() -> None:
    accepted = _verdict("check_graded_score_from_harness_model", "accept_golden")
    assert accepted.passed, "graded_score_from_harness_model rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_graded_score_from_harness_model", "reject_foreign_weights")
    assert not rejected.passed, "graded_score_from_harness_model accepted its rejecting fixture"
    assert rejected.zero_reason == "model-state-not-harness-owned", (
        "graded_score_from_harness_model emitted " + rejected.zero_reason + " instead of model-state-not-harness-owned"
    )


def test_graded_score_unsmoothed() -> None:
    accepted = _verdict("check_graded_score_unsmoothed", "accept_golden")
    assert accepted.passed, "graded_score_unsmoothed rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_graded_score_unsmoothed", "reject_smoothed")
    assert not rejected.passed, "graded_score_unsmoothed accepted its rejecting fixture"
    assert rejected.zero_reason == "readout-smoothed", (
        "graded_score_unsmoothed emitted " + rejected.zero_reason + " instead of readout-smoothed"
    )


def test_score_sustained_across_points() -> None:
    accepted = _verdict("check_score_sustained_across_points", "accept_golden")
    assert accepted.passed, "score_sustained_across_points rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_score_sustained_across_points", "reject_not_sustained")
    assert not rejected.passed, "score_sustained_across_points accepted its rejecting fixture"
    assert rejected.zero_reason == "score-not-sustained", (
        "score_sustained_across_points emitted " + rejected.zero_reason + " instead of score-not-sustained"
    )


def test_corpus_has_training_effect() -> None:
    accepted = _verdict("check_corpus_has_training_effect", "accept_golden")
    assert accepted.passed, "corpus_has_training_effect rejected the accepting fixture: " + accepted.zero_reason
    rejected = _verdict("check_corpus_has_training_effect", "reject_no_effect")
    assert not rejected.passed, "corpus_has_training_effect accepted its rejecting fixture"
    assert rejected.zero_reason == "corpus-had-no-training-effect", (
        "corpus_has_training_effect emitted " + rejected.zero_reason + " instead of corpus-had-no-training-effect"
    )


def main() -> int:
    failures = []
    for name in sorted(globals()):
        if not name.startswith("test_"):
            continue
        try:
            globals()[name]()
        except AssertionError as error:
            failures.append(name + ": " + str(error))
    for row in failures:
        print("FAIL " + row)
    print("compiled tests: " + str(len([n for n in globals() if n.startswith("test_")])) + " run, "
          + str(len(failures)) + " failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
