#!/usr/bin/env python3
"""Pure reductions over one harness telemetry record. Nothing here executes a submission.

Every function below is a total function of the telemetry mapping that
`tests/runner.py` produced inside the verifier's own process. No file outside
that record is opened, no environment variable is read, no clock is consulted, no
random source exists, and the submission is never imported. The imports are the
allowlisted six and nothing else.

Each reduction returns one `Verdict` carrying a value in the closed interval and,
when that value is not full, the machine-readable reason it is not. A reason is
kebab-case because the grader downstream branches on it rather than reads it.

The separation reduction is the one that carries this slot's design. It resolves
to THREE outcomes, not two. An allocation whose measured separation clears the
bound margin with the noise band excluded is ESTABLISHED. An allocation whose
band lies wholly on the wrong side of zero has measurably degraded and is FAILED.
An allocation whose band still contains the margin at the bound evaluation
ceiling is UNPROVEN: the run completed, nothing went wrong, and the evaluation
budget was the binding limit. Collapsing that third outcome into either
neighbour would report a measurement that was never made.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

FULL = 1.0
NONE = 0.0

# The bound separation margin, transcribed from the batch binding. Never authored
# here, never recomputed from an outcome.
SEPARATION_MARGIN = 0.05

# Binary floating point cannot represent 0.05, so an edge that is exactly the
# margin in decimal lands a few ulps below it after one subtraction: 0.06 - 0.01
# evaluates to 0.049999999999999996. Without this tolerance a band whose lower
# edge IS the margin reports unproven, which would make the boundary a property
# of the representation rather than of the measurement.
MARGIN_TOLERANCE = 1e-12

# The three verdict labels the separation reduction can resolve to, and the
# machine-readable reason each one carries. They are distinct strings on purpose:
# a downstream reader must be able to tell "we measured a win", "we measured a
# loss" and "we did not measure anything" apart without parsing prose.
VERDICT_ESTABLISHED = "established"
VERDICT_FAILED = "failed"
VERDICT_UNPROVEN = "unproven"

REASON_ESTABLISHED = "separation-margin-cleared"
REASON_DEGRADED = "degradation-past-bar"
REASON_UNPROVEN = "significance-unestablished-at-ceiling"

FROZEN_INPUT_NAMES = ("eval_corpus", "model_stats", "reference_unquantized")


@dataclass(frozen=True)
class Verdict:
    """One reduction's outcome: a value in [0, 1] and the reason it is not full."""

    value: float
    reason: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.value >= FULL


def _ok(detail: str) -> Verdict:
    return Verdict(FULL, "", detail)


def _clamp(value: float) -> float:
    if value != value:
        return NONE
    return max(NONE, min(FULL, value))


def _mapping(payload: Any, key: str) -> Mapping:
    value = payload.get(key) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _points(telemetry: Mapping) -> list:
    rows = telemetry.get("points") if isinstance(telemetry, Mapping) else None
    return [row for row in (rows or []) if isinstance(row, Mapping)]


# --------------------------------------------------------------------------
# VALUE: the submission's allocation resolved against the harness tensor list.
# --------------------------------------------------------------------------
def check_allocation_wellformed(telemetry: Mapping) -> Verdict:
    allocation = _mapping(telemetry, "allocation")
    if allocation.get("wellformed") is True and allocation.get("bits"):
        return _ok("allocation resolved to " + str(len(allocation["bits"])) + " tensor widths")
    return Verdict(
        NONE,
        "allocation-malformed",
        "the harness could not resolve the submitted allocation: "
        + str(allocation.get("defect") or "no allocation was produced"),
    )


# --------------------------------------------------------------------------
# VALUE: the fixed bit budget, measured from the harness's own tensor walk.
# --------------------------------------------------------------------------
def check_bit_budget(telemetry: Mapping) -> Verdict:
    account = _mapping(telemetry, "accounting")
    if account.get("source") != "harness-tensor-walk":
        return Verdict(
            NONE,
            "bit-budget-overspent",
            "the bit accounting did not come from the harness tensor walk, so the budget is asserted rather than measured",
        )
    walked = int(account.get("tensors_walked") or 0)
    present = int(account.get("tensors_in_model") or 0)
    if walked != present or present == 0:
        return Verdict(
            NONE,
            "bit-budget-overspent",
            "the accounting walked " + str(walked) + " of " + str(present)
            + " model tensors, so bits allocated to an unwalked tensor were never counted",
        )
    spent = int(account.get("allocated_bits") or 0)
    budget = int(account.get("budget_bits") or 0)
    if budget <= 0:
        return Verdict(NONE, "bit-budget-overspent", "no bit budget was resolved")
    if spent > budget:
        return Verdict(
            NONE,
            "bit-budget-overspent",
            "the allocation spends " + str(spent) + " bits against a budget of " + str(budget)
            + ", so any perplexity it bought was bought with bits it did not have",
        )
    return _ok("allocated " + str(spent) + " bits within the fixed budget of " + str(budget))


# --------------------------------------------------------------------------
# INVARIANT: the frozen corpus and the unquantized reference never moved.
# --------------------------------------------------------------------------
def check_frozen_inputs_intact(telemetry: Mapping) -> Verdict:
    opening = _mapping(telemetry, "opening_digests")
    closing = _mapping(telemetry, "closing_digests")
    missing = [name for name in FROZEN_INPUT_NAMES if not opening.get(name)]
    if missing:
        return Verdict(
            NONE,
            "frozen-inputs-modified",
            "no opening digest was recorded for: " + ", ".join(missing),
        )
    moved = [name for name in FROZEN_INPUT_NAMES if closing.get(name) != opening.get(name)]
    if moved:
        return Verdict(
            NONE,
            "frozen-inputs-modified",
            "these frozen inputs moved across the evaluation window: " + ", ".join(moved),
        )
    return _ok("the model statistics, the evaluation corpus and the unquantized reference all held")


# --------------------------------------------------------------------------
# VALUE: the graded perplexity came from harness-owned quantized state.
# --------------------------------------------------------------------------
def check_graded_metric_harness_sourced(telemetry: Mapping) -> Verdict:
    if telemetry.get("graded_provenance") != "harness-quantized-state":
        return Verdict(
            NONE,
            "graded-metric-submission-sourced",
            "the graded perplexity was carried from "
            + str(telemetry.get("graded_provenance"))
            + " rather than recomputed from the state the harness quantized",
        )
    points = _points(telemetry)
    if not points:
        return Verdict(
            NONE,
            "graded-metric-submission-sourced",
            "no harness-computed evaluation point exists, so any metric would have to come from the submission",
        )
    # Provenance, not value equality. An honest submission that estimates the
    # right number correctly must not be punished for agreeing with the harness,
    # so what is graded is WHERE each point came from and never whether it
    # coincides with something the submission said.
    adopted = [
        str(row.get("shard_id"))
        for row in points
        if str(row.get("source") or "") != "harness-recompute"
    ]
    if adopted:
        return Verdict(
            NONE,
            "graded-metric-submission-sourced",
            "these points were not recomputed by the harness: " + ", ".join(adopted),
        )
    if not str(_mapping(telemetry, "submission").get("allocation_sha256") or ""):
        return Verdict(
            NONE,
            "graded-metric-submission-sourced",
            "the submission's own artifact was not recorded separately from the graded path, so the two cannot be told apart",
        )
    return _ok("every graded point was recomputed by the harness over its own quantized state")


# --------------------------------------------------------------------------
# VALUE: no smoothing was applied on the graded path.
# --------------------------------------------------------------------------
def check_graded_perplexity_unsmoothed(telemetry: Mapping) -> Verdict:
    window = telemetry.get("smoothing_window_applied")
    if not isinstance(window, int) or window != 0:
        return Verdict(
            NONE,
            "readout-smoothed-on-graded-path",
            "a smoothing window of " + str(window)
            + " was applied to the graded readout; a submission may smooth its own readout, the graded one is raw",
        )
    drifted = [
        str(row.get("shard_id"))
        for row in _points(telemetry)
        if float(row.get("graded_ppl_quant", 0.0)) != float(row.get("raw_ppl_quant", 1.0))
    ]
    if drifted:
        return Verdict(
            NONE,
            "readout-smoothed-on-graded-path",
            "the graded reading differs from the raw reading at: " + ", ".join(drifted),
        )
    return _ok("the graded reading is the raw reading at every scheduled point")


# --------------------------------------------------------------------------
# ORDERING: the evaluation points ran in the order the verifier derived.
# --------------------------------------------------------------------------
def check_evaluation_schedule_order(telemetry: Mapping) -> Verdict:
    schedule = _mapping(telemetry, "schedule")
    if schedule.get("source") != "verifier-derived":
        return Verdict(
            NONE,
            "evaluation-schedule-not-verifier-ordered",
            "the evaluation order was sourced from " + str(schedule.get("source")),
        )
    nonce = str(schedule.get("nonce") or "")
    order = [str(item) for item in (schedule.get("order") or [])]
    if not nonce or not order:
        return Verdict(
            NONE,
            "evaluation-schedule-not-verifier-ordered",
            "no verifier-derived order is recorded",
        )
    expected = [
        shard
        for _, shard in sorted(
            (hashlib.sha256((nonce + ":" + shard).encode()).hexdigest(), shard) for shard in order
        )
    ]
    if expected != order:
        return Verdict(
            NONE,
            "evaluation-schedule-not-verifier-ordered",
            "the recorded order " + ",".join(order) + " is not the order the corpus digest derives",
        )
    visited = [str(row.get("shard_id")) for row in _points(telemetry)]
    # A truncated run visits a PREFIX of the schedule, and a truncated run is
    # already the early-stop reduction's finding. Comparing against the whole
    # order here would take that zero away from the reduction that owns it and
    # attribute an early stop to a reordering nobody performed.
    if visited and visited != order[: len(visited)]:
        return Verdict(
            NONE,
            "evaluation-schedule-not-verifier-ordered",
            "the points were visited as " + ",".join(visited) + " rather than as scheduled",
        )
    return _ok("the evaluation ran in the order the frozen corpus digest derives")


# --------------------------------------------------------------------------
# INVARIANT: one quantized state, read at every scheduled point.
# --------------------------------------------------------------------------
def check_reading_sustained(telemetry: Mapping) -> Verdict:
    points = _points(telemetry)
    if len(points) < 2:
        return Verdict(
            NONE,
            "reading-not-sustained",
            "the reading rests on " + str(len(points))
            + " evaluation point(s); one reading has no band and is not a measurement",
        )
    seen = {str(row.get("state_digest") or "") for row in points}
    if len(seen) != 1 or "" in seen:
        return Verdict(
            NONE,
            "reading-not-sustained",
            "the graded points do not share one quantized state digest, so the state was re-made between readings",
        )
    measure = _mapping(telemetry, "measurement")
    used = measure.get("points_used")
    if isinstance(used, int) and used != len(points):
        return Verdict(
            NONE,
            "reading-not-sustained",
            "the verdict rests on " + str(used) + " point(s) while " + str(len(points))
            + " were read, so the reading was resolved over fewer points than the run produced",
        )
    unreadable = [
        str(row.get("shard_id"))
        for row in points
        if not isinstance(row.get("paired_separation"), (int, float))
        or not math.isfinite(float(row.get("paired_separation")))
    ]
    if unreadable:
        return Verdict(
            NONE,
            "reading-not-sustained",
            "no finite reading was produced at: " + ", ".join(unreadable),
        )
    return _ok("one quantized state was read at all " + str(len(points)) + " scheduled points")


# --------------------------------------------------------------------------
# ABSENCE: the evaluation ceiling was not short-circuited.
# --------------------------------------------------------------------------
def check_no_early_stop(telemetry: Mapping) -> Verdict:
    schedule = _mapping(telemetry, "schedule")
    scheduled = int(schedule.get("points_scheduled") or 0)
    completed = int(schedule.get("points_completed") or 0)
    if schedule.get("truncated") or schedule.get("truncation_record") is not None:
        return Verdict(
            NONE,
            "early-stop-without-sustained-reading",
            "the evaluation was truncated: " + str(schedule.get("truncation_record")),
        )
    floor = int(schedule.get("repeat_floor") or 0)
    if scheduled <= 0 or completed < max(scheduled, floor):
        return Verdict(
            NONE,
            "early-stop-without-sustained-reading",
            "the run completed " + str(completed) + " of " + str(max(scheduled, floor))
            + " required evaluation points, so a favourable reading was harvested rather than sustained",
        )
    return _ok("all " + str(completed) + " scheduled evaluation points were completed")


# --------------------------------------------------------------------------
# The separation reduction and its three outcomes.
# --------------------------------------------------------------------------
def score_separation(separation: float, half_width: float, margin: float = SEPARATION_MARGIN) -> float:
    """The graded ramp: continuous in the separation, never a step at the margin.

    The graded quantity is the lower edge of the measured separation band, so
    credit is earned by separation that survives the noise rather than by
    separation that happens to sit above it on one reading. The ramp is linear in
    that edge and reaches full exactly where the edge reaches the bound margin,
    which is the same point at which the verdict turns ESTABLISHED. Value and
    verdict therefore change at one place, and only the verdict changes abruptly.
    """
    if margin <= 0:
        return NONE
    edge = float(separation) - float(half_width)
    if edge >= float(margin) - MARGIN_TOLERANCE:
        return FULL
    return _clamp(edge / float(margin))


def classify_separation(measurement: Mapping) -> tuple:
    """(verdict, reason, value) for one measured band. Three outcomes, never two."""
    margin = float(measurement.get("separation_margin", SEPARATION_MARGIN))
    mean = float(measurement.get("separation_mean", 0.0))
    half = float(measurement.get("noise_half_width", 0.0))
    lower = mean - half
    upper = mean + half
    value = score_separation(mean, half, margin)
    if not measurement.get("band_resolvable"):
        return VERDICT_UNPROVEN, REASON_UNPROVEN, NONE
    if upper < 0.0:
        return VERDICT_FAILED, REASON_DEGRADED, NONE
    if lower >= margin - MARGIN_TOLERANCE:
        return VERDICT_ESTABLISHED, REASON_ESTABLISHED, FULL
    return VERDICT_UNPROVEN, REASON_UNPROVEN, value


# --------------------------------------------------------------------------
# EFFECT: the allocation did not measurably degrade past the bar.
# --------------------------------------------------------------------------
def check_not_degrading_past_bar(telemetry: Mapping) -> Verdict:
    measure = _mapping(telemetry, "measurement")
    verdict, reason, _ = classify_separation(measure)
    if verdict == VERDICT_FAILED:
        return Verdict(
            NONE,
            reason,
            "the whole measured band lies below zero: the allocation degrades perplexity by "
            + repr(-float(measure.get("separation_mean", 0.0)))
            + " against the control allocation at the same budget",
        )
    return _ok("the measured band does not lie wholly on the degrading side of the control")


# --------------------------------------------------------------------------
# DIVERGENCE: did the measurement separate from the control by the bound margin?
# --------------------------------------------------------------------------
def check_separation_margin(telemetry: Mapping) -> Verdict:
    measure = _mapping(telemetry, "measurement")
    verdict, reason, value = classify_separation(measure)
    if verdict == VERDICT_ESTABLISHED:
        return _ok(
            "the separation band's lower edge cleared the bound margin of " + repr(SEPARATION_MARGIN)
        )
    if verdict == VERDICT_FAILED:
        return Verdict(NONE, REASON_UNPROVEN, "the separation is negative past the noise band")
    return Verdict(
        value,
        REASON_UNPROVEN,
        "the separation band spans "
        + repr(float(measure.get("separation_mean", 0.0)) - float(measure.get("noise_half_width", 0.0)))
        + " to "
        + repr(float(measure.get("separation_mean", 0.0)) + float(measure.get("noise_half_width", 0.0)))
        + " and still contains the bound margin at the evaluation ceiling of "
        + str(measure.get("points_used"))
        + " points; the run completed and the evaluation budget was the limit, so nothing was established and nothing failed",
    )


# The selector table the grader reads. Order is the order the score document
# reports, so it is fixed here rather than left to a mapping's iteration.
SELECTORS = (
    ("submission_allocation_wellformed", check_allocation_wellformed),
    ("bit_budget_respected_as_allocated", check_bit_budget),
    ("frozen_corpus_and_reference_intact", check_frozen_inputs_intact),
    ("graded_metric_harness_sourced", check_graded_metric_harness_sourced),
    ("graded_perplexity_unsmoothed", check_graded_perplexity_unsmoothed),
    ("evaluation_points_verifier_ordered", check_evaluation_schedule_order),
    ("no_early_stop_on_favourable_evaluation", check_no_early_stop),
    ("reading_sustained_across_scheduled_points", check_reading_sustained),
    ("allocation_not_degrading_past_bar", check_not_degrading_past_bar),
    ("separation_margin_cleared", check_separation_margin),
)

# Which reductions gate. A gate that fails takes the reward to zero and hands its
# own reason to the score document. The separation reduction is deliberately not
# a gate: it carries the continuous value, and its unproven outcome is a third
# result rather than a failed gate.
REQUIRED = tuple(name for name, _ in SELECTORS if name != "separation_margin_cleared")
