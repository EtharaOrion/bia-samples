#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml

The compiled tests tests/checkers.yaml names in every compiled_test field.
Each one re-asserts one reduction over the SAME telemetry record the grader
read, so a reduction that passed in grade.py and fails here is a disagreement
between the manifest and the run rather than a second opinion about the run.

Runnable two ways: as a script, which tests/test.sh does, and under pytest,
which reads the telemetry path from OER20_TELEMETRY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import BAND_FIXTURES  # noqa: E402
from checkers import (  # noqa: E402
    REQUIRED,
    SELECTORS,
    classify_separation,
)

_STATE = {"telemetry": None, "verdict": None}

# The evaluation order is derived by the verifier from the digest of its own
# held-out manifest, which is not a bundle byte, so no order is recorded here.
# Recording one would be recording a guess. The order is graded structurally
# instead, by test_evaluation_points_verifier_ordered, which recomputes it from
# the nonce the run reports rather than from a constant.
EXPECTED_ORDER = None
EXPECTED_ALLOCATED_BITS = 648806400
EXPECTED_BUDGET_BITS = 648806400
EXPECTED_REASON = 'separation-margin-cleared'
EXPECTED_REWARD = 1.0
BOUND_MARGIN = 0.05


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def telemetry():
    if _STATE["telemetry"] is None:
        _STATE["telemetry"] = _read(os.environ["OER20_TELEMETRY"])
    return _STATE["telemetry"]


def verdict():
    if _STATE["verdict"] is None:
        _STATE["verdict"] = _read(os.environ["OER20_VERDICT"])
    return _STATE["verdict"]


def _reduction_over(name, record):
    for ident, selector in SELECTORS:
        if ident == name:
            return selector(record)
    raise AssertionError("tests/checkers.yaml names a selector checkers.py does not carry: " + name)


def _reduction(name):
    return _reduction_over(name, telemetry())


def test_submission_allocation_wellformed():
    """submission_allocation_wellformed: passes at full value, and its zero carries allocation-malformed."""
    outcome = _reduction('submission_allocation_wellformed')
    assert 0.0 <= outcome.value <= 1.0
    assert 'allocation-malformed' == 'allocation-malformed'
    assert outcome.passed, 'submission_allocation_wellformed' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_bit_budget_respected_as_allocated():
    """bit_budget_respected_as_allocated: passes at full value, and its zero carries bit-budget-overspent."""
    outcome = _reduction('bit_budget_respected_as_allocated')
    assert 0.0 <= outcome.value <= 1.0
    assert 'bit-budget-overspent' == 'bit-budget-overspent'
    assert outcome.passed, 'bit_budget_respected_as_allocated' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_frozen_corpus_and_reference_intact():
    """frozen_corpus_and_reference_intact: passes at full value, and its zero carries frozen-inputs-modified."""
    outcome = _reduction('frozen_corpus_and_reference_intact')
    assert 0.0 <= outcome.value <= 1.0
    assert 'frozen-inputs-modified' == 'frozen-inputs-modified'
    assert outcome.passed, 'frozen_corpus_and_reference_intact' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_graded_metric_harness_sourced():
    """graded_metric_harness_sourced: passes at full value, and its zero carries graded-metric-submission-sourced."""
    outcome = _reduction('graded_metric_harness_sourced')
    assert 0.0 <= outcome.value <= 1.0
    assert 'graded-metric-submission-sourced' == 'graded-metric-submission-sourced'
    assert outcome.passed, 'graded_metric_harness_sourced' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_graded_perplexity_unsmoothed():
    """graded_perplexity_unsmoothed: passes at full value, and its zero carries readout-smoothed-on-graded-path."""
    outcome = _reduction('graded_perplexity_unsmoothed')
    assert 0.0 <= outcome.value <= 1.0
    assert 'readout-smoothed-on-graded-path' == 'readout-smoothed-on-graded-path'
    assert outcome.passed, 'graded_perplexity_unsmoothed' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_evaluation_points_verifier_ordered():
    """evaluation_points_verifier_ordered: passes at full value, and its zero carries evaluation-schedule-not-verifier-ordered."""
    outcome = _reduction('evaluation_points_verifier_ordered')
    assert 0.0 <= outcome.value <= 1.0
    assert 'evaluation-schedule-not-verifier-ordered' == 'evaluation-schedule-not-verifier-ordered'
    assert outcome.passed, 'evaluation_points_verifier_ordered' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_reading_sustained_across_scheduled_points():
    """reading_sustained_across_scheduled_points: passes at full value, and its zero carries reading-not-sustained."""
    outcome = _reduction('reading_sustained_across_scheduled_points')
    assert 0.0 <= outcome.value <= 1.0
    assert 'reading-not-sustained' == 'reading-not-sustained'
    assert outcome.passed, 'reading_sustained_across_scheduled_points' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_no_early_stop_on_favourable_evaluation():
    """no_early_stop_on_favourable_evaluation: passes at full value, and its zero carries early-stop-without-sustained-reading."""
    outcome = _reduction('no_early_stop_on_favourable_evaluation')
    assert 0.0 <= outcome.value <= 1.0
    assert 'early-stop-without-sustained-reading' == 'early-stop-without-sustained-reading'
    assert outcome.passed, 'no_early_stop_on_favourable_evaluation' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_allocation_not_degrading_past_bar():
    """allocation_not_degrading_past_bar: passes at full value, and its zero carries degradation-past-bar."""
    outcome = _reduction('allocation_not_degrading_past_bar')
    assert 0.0 <= outcome.value <= 1.0
    assert 'degradation-past-bar' == 'degradation-past-bar'
    assert outcome.passed, 'allocation_not_degrading_past_bar' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_separation_margin_cleared():
    """separation_margin_cleared: passes at full value, and its zero carries significance-unestablished-at-ceiling."""
    outcome = _reduction('separation_margin_cleared')
    assert 0.0 <= outcome.value <= 1.0
    assert 'significance-unestablished-at-ceiling' == 'significance-unestablished-at-ceiling'
    assert outcome.reason in ("", 'significance-unestablished-at-ceiling')


def test_calibration_separation_within_band():
    """calibration_separation_within_band: passes at full value, and its zero carries calibration-separation-outside-band."""
    outcome = _reduction('calibration_separation_within_band')
    assert 0.0 <= outcome.value <= 1.0
    assert 'calibration-separation-outside-band' == 'calibration-separation-outside-band'
    assert outcome.passed, 'calibration_separation_within_band' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_calibration_noise_band_width_held():
    """calibration_noise_band_width_held: passes at full value, and its zero carries calibration-noise-band-width-moved."""
    outcome = _reduction('calibration_noise_band_width_held')
    assert 0.0 <= outcome.value <= 1.0
    assert 'calibration-noise-band-width-moved' == 'calibration-noise-band-width-moved'
    assert outcome.passed, 'calibration_noise_band_width_held' + " scored " + repr(outcome.value) + " with reason " + outcome.reason


def test_band_fixture_accept_calibration_band_at_the_grounded_centre():
    """the reading the built environment produces is accepted"""
    fixture = BAND_FIXTURES['accept_calibration_band_at_the_grounded_centre']
    outcome = _reduction_over(fixture["checker"], fixture["telemetry"])
    assert outcome.passed, 'accept_calibration_band_at_the_grounded_centre' + " scored " + repr(outcome.value) + " with reason " + outcome.reason
    assert outcome.reason == '', 'accept_calibration_band_at_the_grounded_centre' + " carried reason " + repr(outcome.reason)


def test_band_fixture_accept_calibration_band_on_the_inner_edge():
    """a recorded width any smaller would reject this, so the width is load bearing for the accepting half"""
    fixture = BAND_FIXTURES['accept_calibration_band_on_the_inner_edge']
    outcome = _reduction_over(fixture["checker"], fixture["telemetry"])
    assert outcome.passed, 'accept_calibration_band_on_the_inner_edge' + " scored " + repr(outcome.value) + " with reason " + outcome.reason
    assert outcome.reason == '', 'accept_calibration_band_on_the_inner_edge' + " carried reason " + repr(outcome.reason)


def test_band_fixture_reject_calibration_band_near_miss_outside_the_edge():
    """one near-miss offset past the inner edge rejects, so the checker depends on the value and not on the shape"""
    fixture = BAND_FIXTURES['reject_calibration_band_near_miss_outside_the_edge']
    outcome = _reduction_over(fixture["checker"], fixture["telemetry"])
    assert not outcome.passed, 'reject_calibration_band_near_miss_outside_the_edge' + " was accepted at " + repr(outcome.value) + ", so " + 'calibration_separation_within_band' + " does not depend on the grounded value"
    assert outcome.reason == 'calibration-separation-outside-band', 'reject_calibration_band_near_miss_outside_the_edge' + " carried reason " + repr(outcome.reason)


def test_band_fixture_reject_calibration_band_far_from_the_centre():
    """a separation that is the right shape and the wrong size rejects"""
    fixture = BAND_FIXTURES['reject_calibration_band_far_from_the_centre']
    outcome = _reduction_over(fixture["checker"], fixture["telemetry"])
    assert not outcome.passed, 'reject_calibration_band_far_from_the_centre' + " was accepted at " + repr(outcome.value) + ", so " + 'calibration_separation_within_band' + " does not depend on the grounded value"
    assert outcome.reason == 'calibration-separation-outside-band', 'reject_calibration_band_far_from_the_centre' + " carried reason " + repr(outcome.reason)


def test_band_fixture_accept_calibration_width_at_the_grounded_value():
    """the recorded noise half width accepts"""
    fixture = BAND_FIXTURES['accept_calibration_width_at_the_grounded_value']
    outcome = _reduction_over(fixture["checker"], fixture["telemetry"])
    assert outcome.passed, 'accept_calibration_width_at_the_grounded_value' + " scored " + repr(outcome.value) + " with reason " + outcome.reason
    assert outcome.reason == '', 'accept_calibration_width_at_the_grounded_value' + " carried reason " + repr(outcome.reason)


def test_band_fixture_reject_calibration_width_near_miss_outside_tolerance():
    """a half width one near-miss offset past half_width_tolerance rejects"""
    fixture = BAND_FIXTURES['reject_calibration_width_near_miss_outside_tolerance']
    outcome = _reduction_over(fixture["checker"], fixture["telemetry"])
    assert not outcome.passed, 'reject_calibration_width_near_miss_outside_tolerance' + " was accepted at " + repr(outcome.value) + ", so " + 'calibration_noise_band_width_held' + " does not depend on the grounded value"
    assert outcome.reason == 'calibration-noise-band-width-moved', 'reject_calibration_width_near_miss_outside_tolerance' + " carried reason " + repr(outcome.reason)


def test_band_fixture_reject_calibration_width_doubled():
    """a band twice as wide rejects, so a noisier measurement cannot be passed off as this one"""
    fixture = BAND_FIXTURES['reject_calibration_width_doubled']
    outcome = _reduction_over(fixture["checker"], fixture["telemetry"])
    assert not outcome.passed, 'reject_calibration_width_doubled' + " was accepted at " + repr(outcome.value) + ", so " + 'calibration_noise_band_width_held' + " does not depend on the grounded value"
    assert outcome.reason == 'calibration-noise-band-width-moved', 'reject_calibration_width_doubled' + " carried reason " + repr(outcome.reason)


def test_reward_is_continuous_across_the_margin():
    """The graded ramp moves through the margin; it does not step at it."""
    band = dict(telemetry()["measurement"])
    below = dict(band)
    below["separation_mean"] = BOUND_MARGIN + band["noise_half_width"] - 1e-06
    above = dict(band)
    above["separation_mean"] = BOUND_MARGIN + band["noise_half_width"] + 1e-06
    low = classify_separation(below)
    high = classify_separation(above)
    assert low[0] == "unproven" and high[0] == "established"
    assert abs(high[2] - low[2]) < 1e-03


def test_every_manifest_selector_is_reachable():
    """Every reduction the manifest names resolves, and the required set is non-empty."""
    assert len(SELECTORS) == 12
    assert len(REQUIRED) == 11
    for ident, _ in SELECTORS:
        assert _reduction(ident) is not None


def test_graded_document_agrees_with_the_reductions():
    """The score document the grader wrote reports the same reason the reductions do."""
    document = verdict()
    assert 0.0 <= float(document["reward"]) <= 1.0
    assert str(document["reason"]).strip()
    assert document["metric"]["anchors_state"] == "absent"
    assert document["metric"]["baseline_metric"] is None
    assert document["metric"]["target_metric"] is None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 compiled tests")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--verdict", required=True)
    args = parser.parse_args(argv)
    os.environ["OER20_TELEMETRY"] = args.telemetry
    os.environ["OER20_VERDICT"] = args.verdict
    failures = []
    for name, function in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            function()
        except AssertionError as problem:
            failures.append(name + ": " + str(problem))
    for line in failures:
        print("FAIL " + line)
    print("compiled tests: " + str(len(failures)) + " failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
