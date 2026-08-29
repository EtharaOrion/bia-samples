"""The graded checkers for slot OER-14. Pure, deterministic, one reduction each.

Every function here is a pure function of the harness-owned context `tests/harness.py`
assembled inside the verifier's own process. Nothing in this file reads a planted file, an
environment secret, a clock, a random source or a socket, and nothing here imports the
submission. The only imports are from the allowlist: `math`, `dataclasses` and `typing`.

If a checker needs a number, that number came from a telemetry record the verifier's own
process produced. No value the submission printed or wrote reaches any comparison below.

The declared manifest is `tests/checkers.yaml`; the compiled both-halves suite is
`tests/test_output.py`; the gate chain that runs them is `tests/grade.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

HEX = frozenset("0123456789abcdef")

#: Floating-point slack. Comparisons below are over bits per byte, which is an accumulated
#: log sum, so exact equality is not a meaningful predicate on the derived quantities.
EPSILON = 1e-9


@dataclass(frozen=True)
class Outcome:
    """One checker's verdict: whether it held, why not, and what it read."""

    passed: bool
    reason: str
    detail: str


def _held(detail: str) -> Outcome:
    return Outcome(True, "", detail)


def _reached(context: dict) -> list:
    return [
        row
        for row in context.get("points") or []
        if row.get("reached") and row.get("nll_bits_total") is not None
    ]


def _readings(context: dict) -> list:
    rows = []
    for row in _reached(context):
        denominator = row.get("denominator_bytes") or 0
        if denominator > 0:
            rows.append(float(row["nll_bits_total"]) / float(denominator))
    return rows


# ---------------------------------------------------------------------------
# VALUE. The graded reading is the harness's raw evaluation, unfiltered.
# ---------------------------------------------------------------------------


def graded_bpb_recomputed_unsmoothed(context: dict) -> Outcome:
    """VALUE: no smoothing sits on the graded path, applied or requested."""
    filtered = [
        row.get("point") for row in context.get("points") or [] if str(row.get("filter")) != "none"
    ]
    requested = str((context.get("spec_requests") or {}).get("graded_readout_filter", "none"))
    if filtered:
        return Outcome(
            False,
            "readout-smoothing-on-graded-path",
            "a filter sits on the graded readout at evaluation point(s) "
            + ", ".join(str(item) for item in filtered),
        )
    if requested != "none":
        return Outcome(
            False,
            "readout-smoothing-on-graded-path",
            "the submission requested a graded readout filtered by " + repr(requested)
            + "; a filtered graded readout is refused with a reason, never silently ignored",
        )
    return _held("every scheduled point carries the raw harness evaluation and none was requested")


def denominator_is_frozen_eval_byte_count(context: dict) -> Outcome:
    """VALUE: the denominator is the frozen evaluation byte count and nothing else."""
    frozen = context.get("frozen") or {}
    expected = int(frozen.get("eval_bytes") or 0)
    expected_key = str(frozen.get("denominator_key") or "")
    if expected <= 0:
        return Outcome(
            False,
            "denominator-not-frozen-eval-bytes",
            "the frozen evaluation corpus reports no byte count, so no reading can be normalised",
        )
    for row in context.get("points") or []:
        if int(row.get("denominator_bytes") or 0) != expected:
            return Outcome(
                False,
                "denominator-not-frozen-eval-bytes",
                "evaluation point " + str(row.get("point")) + " divides by "
                + str(row.get("denominator_bytes")) + " where the frozen evaluation corpus is "
                + str(expected) + " bytes",
            )
        if str(row.get("denominator_key")) != expected_key:
            return Outcome(
                False,
                "denominator-not-frozen-eval-bytes",
                "evaluation point " + str(row.get("point")) + " names denominator key "
                + repr(row.get("denominator_key")) + ", not " + repr(expected_key),
            )
    override = int((context.get("spec_requests") or {}).get("denominator_override") or 0)
    if override:
        return Outcome(
            False,
            "denominator-not-frozen-eval-bytes",
            "the submission requested denominator " + str(override)
            + "; a vocabulary may not lower bits per byte by changing what it measures over",
        )
    return _held("every reading divides by the frozen evaluation byte count " + str(expected))


def compute_spend_within_frozen_budget(context: dict) -> Outcome:
    """VALUE: compute spend is read off the harness counter, not asserted by the loop."""
    frozen = context.get("frozen") or {}
    meter = context.get("meter") or {}
    budget = int(frozen.get("budget_units") or 0)
    if str(meter.get("source")) != str(frozen.get("meter_source")):
        return Outcome(
            False,
            "compute-budget-overspent",
            "the compute figure came from " + repr(meter.get("source"))
            + ", not from " + repr(frozen.get("meter_source"))
            + "; a training loop asserting it stayed in budget asserts something about itself",
        )
    if int(meter.get("budget_units") or 0) != budget:
        return Outcome(
            False,
            "compute-budget-overspent",
            "the meter was opened against budget " + str(meter.get("budget_units"))
            + " where the frozen budget is " + str(budget),
        )
    if int(meter.get("spent_units") or 0) > budget:
        return Outcome(
            False,
            "compute-budget-overspent",
            "the harness counter measured " + str(meter.get("spent_units"))
            + " units spent against a frozen budget of " + str(budget),
        )
    if int(meter.get("requested_units") or 0) > budget:
        return Outcome(
            False,
            "compute-budget-overspent",
            "the submission requested " + str(meter.get("requested_units"))
            + " training units against a frozen budget of " + str(budget),
        )
    return _held(
        "the harness counter measured " + str(meter.get("spent_units"))
        + " of " + str(budget) + " units"
    )


def evaluated_state_is_harness_owned(context: dict) -> Outcome:
    """VALUE: the state evaluated is the state the harness owns at that point."""
    rows = _reached(context)
    if not rows:
        return Outcome(
            False,
            "state-not-harness-owned",
            "no evaluation point carries harness-owned state, so nothing was evaluated",
        )
    for row in rows:
        if str(row.get("state_source")) != "harness":
            return Outcome(
                False,
                "state-not-harness-owned",
                "evaluation point " + str(row.get("point")) + " was read from "
                + repr(row.get("state_source"))
                + ", not from the state the harness owns at that point",
            )
        digest = str(row.get("state_digest") or "")
        if len(digest) != 64 or not set(digest) <= HEX:
            return Outcome(
                False,
                "state-not-harness-owned",
                "evaluation point " + str(row.get("point"))
                + " carries no harness state fingerprint",
            )
    return _held(str(len(rows)) + " evaluation point(s) read harness-owned state")


# ---------------------------------------------------------------------------
# EFFECT and INVARIANT. A reading is a level only if the run established it.
# ---------------------------------------------------------------------------


def early_stop_does_not_establish_reading(context: dict) -> Outcome:
    """EFFECT: halting before the scheduled points establishes no reading at all."""
    scheduled = int((context.get("frozen") or {}).get("scheduled_points") or 0)
    points = context.get("points") or []
    unreached = [row.get("point") for row in points if not row.get("reached")]
    stop_after = int((context.get("spec_requests") or {}).get("stop_after_point") or 0)
    if len(points) != scheduled:
        return Outcome(
            False,
            "early-stop-not-established",
            "the run produced " + str(len(points)) + " of " + str(scheduled)
            + " verifier-scheduled evaluation points",
        )
    if unreached:
        return Outcome(
            False,
            "early-stop-not-established",
            "the run halted before evaluation point(s) "
            + ", ".join(str(item) for item in unreached)
            + "; a halt at a favourable evaluation is graded as not having established the "
            + "metric, not as an absent result",
        )
    if stop_after:
        return Outcome(
            False,
            "early-stop-not-established",
            "the submission asked to stop after evaluation point " + str(stop_after),
        )
    return _held("all " + str(scheduled) + " verifier-scheduled evaluation points were reached")


def reading_sustained_across_scheduled_points(context: dict) -> Outcome:
    """INVARIANT: one level holds across the verifier-scheduled evaluation points."""
    tolerance = float((context.get("frozen") or {}).get("band_tolerance_bpb") or 0.0)
    readings = _readings(context)
    if len(readings) < 2:
        return Outcome(
            False,
            "reading-not-sustained",
            "a level needs at least two scheduled readings; " + str(len(readings)) + " exist",
        )
    spread = max(readings) - min(readings)
    if not math.isfinite(spread) or spread > tolerance + EPSILON:
        return Outcome(
            False,
            "reading-not-sustained",
            "the readings span " + format(spread, ".6f")
            + " bits per byte across the scheduled points, wider than the bound band "
            + format(tolerance, ".6f") + "; a favourable fluctuation at one point is not a level",
        )
    return _held(
        "the reading holds within " + format(spread, ".6f") + " across "
        + str(len(readings)) + " scheduled points"
    )


# ---------------------------------------------------------------------------
# ABSENCE. Nothing the submission reported is on the graded path.
# ---------------------------------------------------------------------------


def no_submission_reported_number_on_graded_path(context: dict) -> Outcome:
    """ABSENCE: no submission-authored number is offered to, or consumed by, grading."""
    forbidden = set(context.get("forbidden_consumed") or ())
    for key in context.get("consumed_keys") or ():
        name = str(key)
        if name.startswith("submission.") or name.split(".")[-1] in forbidden:
            return Outcome(
                False,
                "submission-reported-metric-on-graded-path",
                "the graded path consumes " + repr(name)
                + ", which is a value the submission authored",
            )
        if not name.startswith("harness."):
            return Outcome(
                False,
                "submission-reported-metric-on-graded-path",
                "the graded path consumes " + repr(name)
                + ", which names no harness-owned surface",
            )
    offered = (context.get("spec_requests") or {}).get("reported_bpb")
    if offered is not None:
        return Outcome(
            False,
            "submission-reported-metric-on-graded-path",
            "the submission reported bits per byte " + repr(offered)
            + "; a reported metric is refused with a reason rather than silently discarded, "
            + "so the agent can tell it was never read",
        )
    return _held("every consumed key names a harness-owned surface and no metric was offered")


# ---------------------------------------------------------------------------
# DIVERGENCE. The result diverges from what one greedy direction reaches.
# ---------------------------------------------------------------------------


def graded_band_exceeds_single_direction_sweep(context: dict) -> Outcome:
    """DIVERGENCE: the graded reading beats the best single-direction sweep by the margin."""
    anchors = context.get("anchors") or {}
    graded = context.get("graded") or {}
    reading = graded.get("bpb")
    baseline = anchors.get("baseline_metric")
    margin = float(anchors.get("separation_margin") or 0.0)
    if reading is None or baseline is None:
        return Outcome(
            False,
            "single-direction-sweep-not-exceeded",
            "no graded reading exists to compare against the single-direction sweep",
        )
    if float(reading) > float(baseline) - margin + EPSILON:
        return Outcome(
            False,
            "single-direction-sweep-not-exceeded",
            "the graded reading " + format(float(reading), ".6f")
            + " does not beat the best single-direction sweep "
            + format(float(baseline), ".6f") + " by the bound separation margin "
            + format(margin, ".6f"),
        )
    return _held(
        "the graded reading " + format(float(reading), ".6f") + " beats the sweep "
        + format(float(baseline), ".6f") + " by more than " + format(margin, ".6f")
    )


# ---------------------------------------------------------------------------
# ORDERING and INVARIANT over the session. What the session HELD is graded.
# ---------------------------------------------------------------------------


def _flatten_index(context: dict):
    """The attempt at which the flattened direction's marginal gain ran out.

    The first attempt carries no marginal gain by definition, so it never counts toward the
    run. Nothing in the substrate announces the flattening; this reads it off the harness's
    own re-measurement of the attempts the record carries.
    """
    flatten = context.get("flatten") or {}
    direction = str(flatten.get("direction") or "")
    epsilon = float(flatten.get("epsilon_bits") or 0.0)
    needed = int(flatten.get("run_required") or 0)
    run = 0
    for row in context.get("attempts") or []:
        if row.get("first") or str(row.get("direction")) != direction:
            run = 0
            continue
        run = run + 1 if float(row.get("marginal_gain_bits") or 0.0) < epsilon else 0
        if needed and run >= needed:
            return int(row.get("index"))
    return None


def reallocation_follows_flattening(context: dict) -> Outcome:
    """ORDERING: allocation moves off the flattened direction AFTER it stopped paying."""
    direction = str((context.get("flatten") or {}).get("direction") or "")
    attempts = context.get("attempts") or []
    if not attempts:
        return Outcome(
            False,
            "flattened-direction-not-reallocated",
            "the session carries no attempt record, so no ordering over the search exists",
        )
    flattened_at = _flatten_index(context)
    moved_at = None
    for row in attempts:
        if str(row.get("reallocated_from") or "") == direction:
            moved_at = int(row.get("index"))
            break
    if flattened_at is None:
        return Outcome(
            False,
            "flattened-direction-not-reallocated",
            "the record never drives " + repr(direction)
            + " far enough for its marginal gain to run out, so nothing was established "
            + "about where the budget stops paying",
        )
    if moved_at is None:
        return Outcome(
            False,
            "flattened-direction-not-reallocated",
            repr(direction) + " stopped paying at attempt " + str(flattened_at)
            + " and no later attempt reallocates off it; the slot budget is zero-sum, so every "
            + "slot spent past that point is a slot no other direction gets",
        )
    if moved_at <= flattened_at:
        return Outcome(
            False,
            "flattened-direction-not-reallocated",
            "allocation moved off " + repr(direction) + " at attempt " + str(moved_at)
            + ", at or before the flattening was observable at attempt " + str(flattened_at)
            + ", so the move was not a response to anything",
        )
    return _held(
        repr(direction) + " flattened at attempt " + str(flattened_at)
        + " and allocation reallocated off it at attempt " + str(moved_at)
    )


def carried_direction_frontier_never_collapses(context: dict) -> Outcome:
    """INVARIANT: the per-direction frontier the session carries never loses ground."""
    attempts = context.get("attempts") or []
    if not attempts:
        return Outcome(
            False,
            "carried-state-collapsed",
            "the session carries no attempt record, so it held no state across the horizon",
        )
    previous = {}
    for row in attempts:
        carried = row.get("carried_frontier") or {}
        if not carried:
            return Outcome(
                False,
                "carried-state-collapsed",
                "attempt " + str(row.get("index")) + " carries an empty frontier",
            )
        dropped = sorted(set(previous) - set(carried))
        if dropped:
            return Outcome(
                False,
                "carried-state-collapsed",
                "attempt " + str(row.get("index")) + " forgets direction(s) "
                + ", ".join(dropped) + " the session had already measured",
            )
        for name, value in sorted(previous.items()):
            if float(carried[name]) > float(value) + EPSILON:
                return Outcome(
                    False,
                    "carried-state-collapsed",
                    "attempt " + str(row.get("index")) + " carries " + name + " at "
                    + format(float(carried[name]), ".6f")
                    + ", worse than the " + format(float(value), ".6f")
                    + " the session had already established",
                )
        previous = {name: float(value) for name, value in carried.items()}
    return _held(
        "the carried frontier grew monotonically across " + str(len(attempts)) + " attempts"
    )


#: The gate chain, in the order `tests/grade.py` runs it and `tests/checkers.yaml` declares
#: it. Order is load-bearing for attribution only: a run that both halted early and reported
#: an unsustained level is attributed to the halt, because the halt is what caused the rest.
REGISTRY: Tuple[Tuple[str, Callable], ...] = (
    ("graded_bpb_recomputed_unsmoothed", graded_bpb_recomputed_unsmoothed),
    ("denominator_is_frozen_eval_byte_count", denominator_is_frozen_eval_byte_count),
    ("compute_spend_within_frozen_budget", compute_spend_within_frozen_budget),
    ("evaluated_state_is_harness_owned", evaluated_state_is_harness_owned),
    ("early_stop_does_not_establish_reading", early_stop_does_not_establish_reading),
    ("reading_sustained_across_scheduled_points", reading_sustained_across_scheduled_points),
    ("no_submission_reported_number_on_graded_path", no_submission_reported_number_on_graded_path),
    ("graded_band_exceeds_single_direction_sweep", graded_band_exceeds_single_direction_sweep),
    ("reallocation_follows_flattening", reallocation_follows_flattening),
    ("carried_direction_frontier_never_collapses", carried_direction_frontier_never_collapses),
)
