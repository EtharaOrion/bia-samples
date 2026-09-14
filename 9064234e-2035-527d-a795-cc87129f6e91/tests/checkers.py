"""The graded surface. Pure, deterministic, and reading live state only.

Every function below is a pure function of one session document. That document
is telemetry the VERIFIER'S OWN PROCESS produced: `tests/session_driver.py` runs
the refinement loop itself, `tests/train_eval.py` trains the frozen substrate and
evaluates the weights its own loop held, and every number a checker reads comes
off that record. Nothing here reads a value the submission printed, and the one
field the submission does author, `reported`, is graded only for DIVERGENCE
against the verifier's own measurement and never as the graded quantity.

The import list is the allowlist and nothing else: json, hashlib, dataclasses,
typing. There is no planted file read, no environment read, no `random`, no
clock, no socket, and no import of the submission anywhere in this module.

Each function returns one Verdict. A Verdict that is not `ok` carries the
machine-readable reason the manifest declares for that checker, so a zero is
never bare.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "bia.session/v1"

# Machine-readable reason codes. One per checker, and the manifest declares the
# same strings, so a reason cannot be declared in one place and emitted nowhere.
READOUT_SMOOTHED = "readout-smoothed-on-the-graded-path"
NOT_SUSTAINED = "crossing-not-sustained"
EARLY_STOP = "early-stop-claimed-as-crossing"
WEIGHTS_NOT_OWNED = "weights-not-harness-owned"
UNCONSOLIDATED = "single-attempt-metric-unconsolidated"
NOT_CARRIED = "graded-outcome-not-carried"
REPORT_DIVERGES = "reported-crossing-diverges-from-measurement"
BUDGET_EXCEEDED = "attempt-budget-exceeded"
FROZEN_AXIS_MOVED = "frozen-axis-moved"
STALE_SUMMARY = "stale-summary-carried"
SESSION_UNREADABLE = "session-record-unreadable"

REPORT_TOLERANCE_STRIDES = 4


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def summary_digest(prior: List[Dict[str, Any]]) -> str:
    """The digest of the trajectory summary handed to the next iteration.

    The summary is every prior approach and its reward, in order. This is the
    one function `session_driver.py` and this module share, so a summary the
    harness built and a summary a checker recomputes are the same arithmetic and
    a mismatch can only mean the handed summary was stale.
    """
    rows = [
        [
            int(row["index"]),
            str(row["recipe_fingerprint"]),
            row.get("sustained_crossing_step"),
            row.get("attempt_reward"),
        ]
        for row in prior
    ]
    return _digest(rows)


def sustained_crossing(evals: List[Dict[str, Any]], target: float, window: int) -> Optional[int]:
    """The first evaluation point that is at or below target AND holds for `window` more.

    This is the graded crossing, recomputed here from the raw evaluation rows.
    No smoothing, no averaging, no filtering: a dip that does not hold is not a
    crossing, which is exactly the readout manipulation this closes.
    """
    losses = [float(row["loss"]) for row in evals]
    for position in range(len(losses)):
        if position + window >= len(losses):
            return None
        span = losses[position:position + window + 1]
        if all(value <= target for value in span):
            return int(evals[position]["step"])
    return None


def _attempts(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(session.get("attempts") or [])


def _target(session: Dict[str, Any]) -> float:
    return float(session["anchors"]["target_loss"])


def _window(session: Dict[str, Any]) -> int:
    return int(session["eval_schedule"]["sustain_window"])


def _stride(session: Dict[str, Any]) -> int:
    return int(session["eval_schedule"]["stride"])


def frontier(session: Dict[str, Any]) -> List[Optional[int]]:
    """The running best crossing after each attempt. The carried state, as a series."""
    best: Optional[int] = None
    out: List[Optional[int]] = []
    for row in _attempts(session):
        value = row.get("sustained_crossing_step")
        if value is not None and (best is None or int(value) < best):
            best = int(value)
        out.append(best)
    return out


def consolidation(session: Dict[str, Any]) -> Tuple[Optional[int], List[int]]:
    """The consolidated metric and the attempt indices that establish it.

    The metric is the m-th smallest sustained crossing taken over attempts with
    PAIRWISE DISTINCT recipe fingerprints. One fortunate attempt cannot set it,
    and neither can one recipe submitted m times.
    """
    need = int(session["consolidation"]["multiplicity"])
    seen: Dict[str, Dict[str, Any]] = {}
    for row in _attempts(session):
        value = row.get("sustained_crossing_step")
        if value is None:
            continue
        key = str(row["recipe_fingerprint"])
        held = seen.get(key)
        if held is None or int(value) < int(held["sustained_crossing_step"]):
            seen[key] = row
    ranked = sorted(seen.values(), key=lambda row: (int(row["sustained_crossing_step"]), int(row["index"])))
    if len(ranked) < need:
        return None, [int(row["index"]) for row in ranked]
    chosen = ranked[:need]
    return int(chosen[-1]["sustained_crossing_step"]), sorted(int(row["index"]) for row in chosen)


# --------------------------------------------------------------------------
# The ten graded checkers. Exactly one reduction each, as tests/checkers.yaml
# declares. Every one of them reads the session document and nothing else.
# --------------------------------------------------------------------------


def check_graded_crossing_recomputed_unsmoothed(session: Dict[str, Any]) -> Verdict:
    """VALUE. Nothing on the graded path is a smoothed, filtered or derived readout.

    This checker owns the READOUT. It asserts that the schedule and every single
    evaluation row on the graded path declare `smoothing: none` and
    `source: verifier-recomputed`, so the graded quantity is the raw evaluation
    the verifier's own process computed and never an EMA blend, a moving average,
    a median filter or a best-so-far. Whether that raw series actually sustains a
    crossing is a different question, and the INVARIANT checker owns it.
    """
    if session.get("schema") != SCHEMA:
        return Verdict(False, SESSION_UNREADABLE, {"schema": session.get("schema")})
    if str(session["eval_schedule"].get("smoothing")) != "none":
        return Verdict(False, READOUT_SMOOTHED,
                       {"schedule_smoothing": session["eval_schedule"].get("smoothing")})
    if str(session["eval_schedule"].get("owner")) != "verifier":
        return Verdict(False, READOUT_SMOOTHED, {"schedule_owner": session["eval_schedule"].get("owner")})
    offenders = []
    for row in _attempts(session):
        for item in row["verifier_evals"]:
            if str(item.get("smoothing")) != "none" or str(item.get("source")) != "verifier-recomputed":
                offenders.append({"index": row["index"], "step": item.get("step"),
                                  "smoothing": item.get("smoothing"), "source": item.get("source")})
    if offenders:
        return Verdict(False, READOUT_SMOOTHED, {"offenders": offenders[:8]})
    return Verdict(True, "", {"attempts": len(_attempts(session))})


def check_crossing_sustained_across_scheduled_evaluations(session: Dict[str, Any]) -> Verdict:
    """INVARIANT. Every recorded crossing is this module's own recomputation AND holds.

    This checker owns the CROSSING. It recomputes the sustained crossing here,
    from the raw rows, and refuses any recorded value that disagrees; then it
    reads the window back out of the schedule and confirms the target holds at the
    crossing step and at each of the next `sustain_window` points the verifier
    scheduled. A single favourable evaluation therefore cannot be harvested.
    """
    target, window = _target(session), _window(session)
    offenders = []
    for row in _attempts(session):
        points = list(row["verifier_evals"])
        recomputed = sustained_crossing(points, target, window)
        step = row.get("sustained_crossing_step")
        if recomputed != step:
            offenders.append({"index": row["index"], "why": "recorded-crossing-is-not-the-recomputation",
                              "recorded": step, "recomputed": recomputed})
            continue
        if step is None:
            continue
        where = [position for position, item in enumerate(points) if int(item["step"]) == int(step)]
        if not where:
            offenders.append({"index": row["index"], "why": "crossing-step-off-the-schedule", "step": step})
            continue
        start = where[0]
        if start + window >= len(points):
            offenders.append({"index": row["index"], "why": "window-runs-past-the-schedule", "step": step})
            continue
        span = [float(points[start + offset]["loss"]) for offset in range(window + 1)]
        if any(value > target for value in span):
            offenders.append({"index": row["index"], "why": "window-broken", "step": step, "span": span})
    if offenders:
        return Verdict(False, NOT_SUSTAINED, {"offenders": offenders[:8]})
    return Verdict(True, "", {"window": window})


def check_early_halt_yields_no_crossing(session: Dict[str, Any]) -> Verdict:
    """ABSENCE. The set of attempts claiming a crossing the verifier never measured is empty.

    Halting at a favourable evaluation is the canonical way into this set: the
    run stops before the sustain window completes, so the verifier measures no
    crossing, and a claim made anyway is graded as a failure rather than as an
    absent result. The detail records how far the run got and how far it needed
    to get, so the early stop is legible in the score document.
    """
    window, stride = _window(session), _stride(session)
    claimed_but_unproven = []
    for row in _attempts(session):
        claim = (row.get("reported") or {}).get("crossing_step")
        if claim is None:
            continue
        if row.get("sustained_crossing_step") is not None:
            continue
        halted = int(row.get("halted_at_step", 0))
        claimed_but_unproven.append({"index": row["index"], "claim": claim,
                                     "halted_at_step": halted,
                                     "needed_through": int(claim) + window * stride,
                                     "measured": None})
    if claimed_but_unproven:
        return Verdict(False, EARLY_STOP, {"offenders": claimed_but_unproven[:8]})
    return Verdict(True, "", {"claims_checked": sum(
        1 for row in _attempts(session) if (row.get("reported") or {}).get("crossing_step") is not None)})


def check_evaluated_weights_owned_by_the_harness(session: Dict[str, Any]) -> Verdict:
    """VALUE. Graded weights are the ones the harness's own loop held at that step."""
    offenders = []
    for row in _attempts(session):
        run = str(row.get("run_digest") or "")
        digests = []
        for item in row["verifier_evals"]:
            if str(item.get("weights_provenance")) != "harness-step-state":
                offenders.append({"index": row["index"], "step": item.get("step"),
                                  "provenance": item.get("weights_provenance")})
            if str(item.get("weights_run_digest")) != run:
                offenders.append({"index": row["index"], "step": item.get("step"),
                                  "run": run, "weights_run": item.get("weights_run_digest")})
            digests.append(str(item.get("weights_digest") or ""))
        if len(set(digests)) != len(digests):
            offenders.append({"index": row["index"], "why": "one-checkpoint-evaluated-at-several-steps"})
    if offenders:
        return Verdict(False, WEIGHTS_NOT_OWNED, {"offenders": offenders[:8]})
    return Verdict(True, "", {"attempts": len(_attempts(session))})


def check_graded_metric_consolidated_over_distinct_recipes(session: Dict[str, Any]) -> Verdict:
    """EFFECT. Running the session produced a consolidation, not one lucky crossing."""
    need = int(session["consolidation"]["multiplicity"])
    metric, indices = consolidation(session)
    if metric is None:
        return Verdict(False, UNCONSOLIDATED,
                       {"multiplicity_required": need, "distinct_recipes_that_crossed": len(indices),
                        "indices": indices})
    return Verdict(True, "", {"metric": metric, "indices": indices})


def check_refinement_frontier_carried_across_iterations(session: Dict[str, Any]) -> Verdict:
    """ORDERING. The graded outcome rests on carried state, not on one attempt."""
    rules = session["consolidation"]
    metric, indices = consolidation(session)
    if metric is None:
        # There is nothing to attribute, and the EFFECT checker already owns that
        # case. Firing here too would attribute one failure to two reason codes.
        return Verdict(True, "", {"why": "no-consolidation-to-attribute-effect-checker-owns-it"})
    rows = {int(row["index"]): row for row in _attempts(session)}
    determining = max(indices)
    series = frontier(session)
    improvements = [
        position + 1
        for position in range(len(series))
        if series[position] is not None and (position == 0 or series[position - 1] is None
                                             or series[position] < series[position - 1])
    ]
    inherited = rows[determining].get("inherited_components") or {}
    on_frontier_sources = []
    for source, components in inherited.items():
        source_index = int(source)
        if source_index >= determining:
            continue
        if series[source_index - 1] is not None and rows[source_index].get("sustained_crossing_step") is not None:
            if int(rows[source_index]["sustained_crossing_step"]) == int(series[source_index - 1]):
                on_frontier_sources.extend(list(components))
    detail = {
        "determining_index": determining,
        "frontier_improvements": improvements,
        "inherited_from_frontier_components": sorted(set(on_frontier_sources)),
        "consolidation_indices": indices,
    }
    if determining < int(rules["min_determining_index"]):
        detail["why"] = "determining-attempt-too-early"
        return Verdict(False, NOT_CARRIED, detail)
    if len(improvements) < int(rules["min_frontier_improvements"]):
        detail["why"] = "frontier-did-not-improve-enough-times"
        return Verdict(False, NOT_CARRIED, detail)
    if len(set(on_frontier_sources)) < int(rules["min_inherited_components"]):
        detail["why"] = "determining-attempt-inherits-too-little-from-the-frontier"
        return Verdict(False, NOT_CARRIED, detail)
    return Verdict(True, "", detail)


def check_reported_crossing_reconciles_with_verifier_measurement(session: Dict[str, Any]) -> Verdict:
    """DIVERGENCE. A claimed crossing and the verifier's measurement do not disagree.

    Attempts whose claim has no measured crossing at all are excluded here,
    because the ABSENCE checker owns that case; leaving them in would give one
    failure two reason codes.
    """
    stride = _stride(session)
    tolerance = REPORT_TOLERANCE_STRIDES * stride
    offenders = []
    for row in _attempts(session):
        claim = (row.get("reported") or {}).get("crossing_step")
        measured = row.get("sustained_crossing_step")
        if claim is None or measured is None:
            continue
        if abs(int(claim) - int(measured)) > tolerance:
            offenders.append({"index": row["index"], "claim": claim, "measured": measured,
                              "tolerance": tolerance})
    if offenders:
        return Verdict(False, REPORT_DIVERGES, {"offenders": offenders[:8]})
    return Verdict(True, "", {"tolerance": tolerance})


def check_bound_envelope_respected(session: Dict[str, Any]) -> Verdict:
    """INVARIANT. Per-attempt budget_hours and across-attempt max_timeout both hold."""
    bindings = session["bindings"]
    per_attempt = float(bindings["budget_hours"]) * 3600.0
    session_bound = float(bindings["max_timeout_hours"]) * 3600.0
    rows = _attempts(session)
    offenders = []
    if len(rows) > int(bindings["max_attempts"]):
        offenders.append({"why": "attempt-count-over-max-attempts", "count": len(rows)})
    for position, row in enumerate(rows):
        if int(row["index"]) != position + 1:
            offenders.append({"why": "attempt-index-not-contiguous", "index": row["index"]})
        if float(row["attempt_seconds"]) > per_attempt:
            offenders.append({"why": "attempt-over-budget-hours", "index": row["index"],
                              "seconds": row["attempt_seconds"], "bound": per_attempt})
    if float(session.get("session_seconds", 0.0)) > session_bound:
        offenders.append({"why": "session-over-max-timeout", "seconds": session.get("session_seconds"),
                          "bound": session_bound})
    if offenders:
        return Verdict(False, BUDGET_EXCEEDED, {"offenders": offenders[:8]})
    return Verdict(True, "", {"attempts": len(rows), "per_attempt_bound_seconds": per_attempt})


def check_frozen_axes_untouched(session: Dict[str, Any]) -> Verdict:
    """ABSENCE. The set of frozen-axis writes across the whole session is empty."""
    offenders = []
    for row in _attempts(session):
        for key in row.get("frozen_axis_writes") or []:
            offenders.append({"index": row["index"], "key": key})
        if int(row.get("forward_passes_per_step", 0)) != 1:
            offenders.append({"index": row["index"], "key": "forward_passes_per_step"})
        if int(row.get("backward_passes_per_step", 0)) != 1:
            offenders.append({"index": row["index"], "key": "backward_passes_per_step"})
    if offenders:
        return Verdict(False, FROZEN_AXIS_MOVED, {"offenders": offenders[:8]})
    return Verdict(True, "", {"attempts": len(_attempts(session))})


def check_carried_summary_is_current(session: Dict[str, Any]) -> Verdict:
    """ORDERING. Each iteration was handed, and answered from, the CURRENT summary."""
    offenders = []
    rows = _attempts(session)
    for position, row in enumerate(rows):
        expected = summary_digest(rows[:position])
        if str(row.get("summary_digest_expected")) != expected:
            offenders.append({"index": row["index"], "why": "handed-summary-is-stale",
                              "handed": row.get("summary_digest_expected"), "current": expected})
        if str(row.get("summary_digest_echoed")) != str(row.get("summary_digest_expected")):
            offenders.append({"index": row["index"], "why": "echo-does-not-match-what-was-handed",
                              "echoed": row.get("summary_digest_echoed")})
        if int(row.get("carried_attempt_count", -1)) != position:
            offenders.append({"index": row["index"], "why": "carried-attempt-count-wrong",
                              "carried": row.get("carried_attempt_count")})
    if offenders:
        return Verdict(False, STALE_SUMMARY, {"offenders": offenders[:8]})
    return Verdict(True, "", {"attempts": len(rows)})


# The graded chain, in the order tests/checkers.yaml declares it. `grade.py`
# walks this and never invents an order of its own.
#
# The order is structural integrity first, then provenance of the graded number,
# then the semantics of the crossing, then the session-level claims. It is
# chosen so that each failure lands on the checker that OWNS it: a session whose
# loop was not even run correctly is reported as that, rather than as a downstream
# consolidation failure that is merely its consequence.
CHAIN = (
    ("carried_summary_is_current", check_carried_summary_is_current),
    ("bound_envelope_respected", check_bound_envelope_respected),
    ("frozen_axes_untouched", check_frozen_axes_untouched),
    ("graded_crossing_recomputed_unsmoothed", check_graded_crossing_recomputed_unsmoothed),
    ("evaluated_weights_owned_by_the_harness", check_evaluated_weights_owned_by_the_harness),
    ("crossing_sustained_across_scheduled_evaluations", check_crossing_sustained_across_scheduled_evaluations),
    ("early_halt_yields_no_crossing", check_early_halt_yields_no_crossing),
    ("reported_crossing_reconciles_with_verifier_measurement", check_reported_crossing_reconciles_with_verifier_measurement),
    ("graded_metric_consolidated_over_distinct_recipes", check_graded_metric_consolidated_over_distinct_recipes),
    ("refinement_frontier_carried_across_iterations", check_refinement_frontier_carried_across_iterations),
)


def run_chain(session: Dict[str, Any]) -> List[Tuple[str, Verdict]]:
    """Every checker, in declared order. No short circuit, so every reason is visible."""
    return [(name, function(session)) for name, function in CHAIN]
