"""GENERATED SECTION. DO NOT HAND-EDIT.

source: solution/grounding.yaml

One compiled test per declared checker, each driven from the fixtures
`solution/recompute.py` derived from `solution/grounding.yaml`. Every test asserts
both halves: the accepting fixture passes the checker, and that checker's own
rejecting fixture fails it carrying exactly the declared zero reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "tests"))

import checkers  # noqa: E402

FIXTURES = BUNDLE / "solution" / "fixtures"


def _load(name):
    with (FIXTURES / (name + ".json")).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _accepting():
    return _load("accepting")


def _run(name, telemetry):
    return checkers.REGISTRY[name](telemetry)


def test_generator_collapse_absent():
    verdict = _run("generator_collapse_absent", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("generator_collapse_absent", _load("nc-silent-collapse"))
    assert not verdict.passed
    assert verdict.reason == "generator-collapsed-silently"


def test_corpus_mode_share_bounded():
    verdict = _run("corpus_mode_share_bounded", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("corpus_mode_share_bounded", _load("nc-uniform-mode"))
    assert not verdict.passed
    assert verdict.reason == "corpus-mode-share-exceeded"


def test_emission_order_monotone():
    verdict = _run("emission_order_monotone", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("emission_order_monotone", _load("nc-shuffled-collapse"))
    assert not verdict.passed
    assert verdict.reason == "emission-order-not-monotone"


def test_no_benchmark_near_duplicate():
    verdict = _run("no_benchmark_near_duplicate", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("no_benchmark_near_duplicate", _load("nc-benchmark-echo"))
    assert not verdict.passed
    assert verdict.reason == "benchmark-near-duplicate-emitted"


def test_training_budget_as_fed():
    verdict = _run("training_budget_as_fed", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("training_budget_as_fed", _load("nc-budget-overspend"))
    assert not verdict.passed
    assert verdict.reason == "training-budget-overspent"


def test_unsmoothed_graded_readout():
    verdict = _run("unsmoothed_graded_readout", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("unsmoothed_graded_readout", _load("nc-smoothed-readout"))
    assert not verdict.passed
    assert verdict.reason == "readout-smoothing-on-graded-path"


def test_early_stop_does_not_establish_score():
    verdict = _run("early_stop_does_not_establish_score", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("early_stop_does_not_establish_score", _load("nc-early-stop"))
    assert not verdict.passed
    assert verdict.reason == "early-stop-score-not-established"


def test_score_sustained_across_scheduled_points():
    verdict = _run("score_sustained_across_scheduled_points", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("score_sustained_across_scheduled_points", _load("nc-unsustained"))
    assert not verdict.passed
    assert verdict.reason == "score-not-sustained"


def test_graded_score_from_harness_model_state():
    verdict = _run("graded_score_from_harness_model_state", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("graded_score_from_harness_model_state", _load("nc-substituted-weights"))
    assert not verdict.passed
    assert verdict.reason == "score-not-from-harness-model-state"


def test_self_report_matches_recomputation():
    verdict = _run("self_report_matches_recomputation", _accepting())
    assert verdict.passed, verdict.reason
    verdict = _run("self_report_matches_recomputation", _load("nc-fabricated-self-report"))
    assert not verdict.passed
    assert verdict.reason == "self-report-diverges-from-recomputation"

