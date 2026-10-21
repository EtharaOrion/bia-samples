from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "bia.session/v1"

READOUT_SMOOTHED = "readout-smoothed-on-the-graded-path"
NOT_SUSTAINED = "crossing-not-sustained"
EARLY_STOP = "early-stop-claimed-as-crossing"
WEIGHTS_NOT_OWNED = "weights-not-harness-owned"
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
    rows = [
        [
            int(row["index"]),
            str(row["recipe_fingerprint"]),
            row.get("sustained_crossing_step"),
            row.get("attempt_score"),
        ]
        for row in prior
    ]
    return _digest(rows)

def sustained_crossing(evals: List[Dict[str, Any]], target: float, window: int) -> Optional[int]:
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

DEFAULT_MULTIPLICITY = 1

def multiplicity(session: Dict[str, Any]) -> int:
    rules = session.get("consolidation") or {}
    value = rules.get("multiplicity", DEFAULT_MULTIPLICITY)
    if value is None:
        return DEFAULT_MULTIPLICITY
    return max(1, int(value))

def consolidation(session: Dict[str, Any]) -> Tuple[Optional[int], List[int]]:
    need = multiplicity(session)
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

def best_final_loss(session: Dict[str, Any]) -> Optional[float]:
    finals = [float(row["verifier_evals"][-1]["loss"])
              for row in _attempts(session) if row.get("verifier_evals")]
    return min(finals) if finals else None

def check_graded_crossing_recomputed_unsmoothed(session: Dict[str, Any]) -> Verdict:
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

def check_reported_crossing_reconciles_with_verifier_measurement(session: Dict[str, Any]) -> Verdict:
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

CHAIN = (
    ("carried_summary_is_current", check_carried_summary_is_current),
    ("bound_envelope_respected", check_bound_envelope_respected),
    ("frozen_axes_untouched", check_frozen_axes_untouched),
    ("graded_crossing_recomputed_unsmoothed", check_graded_crossing_recomputed_unsmoothed),
    ("evaluated_weights_owned_by_the_harness", check_evaluated_weights_owned_by_the_harness),
    ("crossing_sustained_across_scheduled_evaluations", check_crossing_sustained_across_scheduled_evaluations),
    ("early_halt_yields_no_crossing", check_early_halt_yields_no_crossing),
    ("reported_crossing_reconciles_with_verifier_measurement", check_reported_crossing_reconciles_with_verifier_measurement),
)

def run_chain(session: Dict[str, Any]) -> List[Tuple[str, Verdict]]:
    return [(name, function(session)) for name, function in CHAIN]
