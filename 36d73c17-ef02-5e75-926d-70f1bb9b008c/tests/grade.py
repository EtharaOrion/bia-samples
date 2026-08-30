"""Reduce the verifier's telemetry to one float, and attribute every zero it can emit.

This module imports the checkers and, when it needs the reference identity, the
reference metadata. It NEVER imports the submission. tests/runner.py has already
executed the submission out of process and written the one telemetry record read
here, so nothing the submission can define reaches this interpreter.

Three carriers leave this file, and the order matters:

  1. /logs/verifier/score.json   the full score document, every checker's verdict
  2. /logs/verifier/reward.json  the shaped reward document, reward + reason + metric
  3. /logs/verifier/reward.txt   the single float, written LAST

The score document carries `reward`, `reason` AND `metric`. The delivered bytes wrote
the first two and not the third, so every reader of the bound score document -- and
`seed/forge/controlrecord.py` is one -- resolved it to `carrier-missing-required-key`
and recorded every OER-04 control as NOT-RUN rather than as a measured refusal. Under
FORGE.md Phase 1 item 2b the delivery block binds every reward metric and its range, so
a score document that names a reward without naming what it measured settles nothing.
The `metric` block written here is the same block `reward_document` already wrote to
the shaped carrier: one function, two carriers, no second definition of the anchors.

The float is last because a partial write of the document carriers must never look
like a graded run. tests/test.sh registers an EXIT trap that writes the floor reward
if this file never got that far, so an aborted verifier produces a reason instead of
silence.

Reward shape:

    gates      the eight required checkers, in checkers.GATES order. The first one
               below 1.0 gives reward exactly 0.0 with that gate's zero_reason.
    passing    reward = clip((3250 - graded_step) / 560, 0, 1) * clip(sep / 0.05, 0, 1)

The second factor is continuous. It is not a threshold and it is not a gate. That is
the whole design: a submission whose improvement sits inside the seed-to-seed noise
band at the bound seed ceiling has not established anything, and that is a THIRD
outcome, recorded as `significance-unestablished-at-ceiling` and classified unproven,
distinct from every reason that means the run failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import checkers
from checkers import (
    STATE_ESTABLISHED,
    STATE_NOT_APPLICABLE,
    STATE_UNPROVEN,
    base_score,
    check_claim_reconciles_with_reexecution,
    check_crossing_is_sustained,
    check_early_stop_is_not_a_crossing,
    check_evaluation_grid_is_verifier_scheduled,
    check_frozen_axes_survive_reexecution,
    check_graded_readout_is_unsmoothed,
    check_graded_weights_are_harness_owned,
    check_significance_separation_cleared,
    check_target_reached_on_reexecuted_seeds,
    clamp,
    graded_step,
    load_telemetry,
    separation,
)

REWARD_PATH = "/logs/verifier/reward.txt"
REWARD_DOCUMENT_PATH = "/logs/verifier/reward.json"
SCORE_DOCUMENT_PATH = "/logs/verifier/score.json"
TELEMETRY_PATH = "/logs/verifier/verifier_telemetry.json"

# The bound anchors, transcribed from solution/grounding.yaml `anchors`. They are the
# fallback only: when the telemetry carries the operating point's own `bound` block,
# the metric block reports the anchors the run was actually graded against, because a
# metric block naming anchors the reward did not use would misreport the measurement.
BASELINE_METRIC = 3250
TARGET_METRIC = 2690
SEPARATION_MARGIN = 0.05

REASON_SUBSTRATE_REFUSED = "verifier-substrate-absent"
SUBSTRATE_REFUSAL_PATH = "/logs/verifier/substrate_refusal.json"

# The closed set of refusals tests/runner.py may publish through this file. A reason
# outside it is not passed through, so a malformed refusal cannot invent a reason code.
REFUSAL_REASONS = ("verifier-substrate-absent", "verifier-operating-point-unresolved")

# Reasons this file can publish, beyond the nine declared checker reasons. Every one
# is machine-readable kebab so a downstream grader branches on it rather than reads it.
REASON_NO_IMPROVEMENT = "no-improvement-over-baseline"
REASON_TELEMETRY_ABSENT = "verifier-telemetry-absent"
REASON_TELEMETRY_UNREADABLE = "verifier-telemetry-unreadable"

# Bound gate order. Selector by selector, so an auditor reading checkers.yaml can
# match every declared selector to the call that reaches it.
GATE_SELECTORS = (
    ("target_reached_on_reexecuted_seeds", check_target_reached_on_reexecuted_seeds),
    ("graded_readout_is_unsmoothed", check_graded_readout_is_unsmoothed),
    ("early_stop_is_not_a_crossing", check_early_stop_is_not_a_crossing),
    ("crossing_is_sustained", check_crossing_is_sustained),
    ("evaluation_grid_is_verifier_scheduled", check_evaluation_grid_is_verifier_scheduled),
    ("graded_weights_are_harness_owned", check_graded_weights_are_harness_owned),
    ("frozen_axes_survive_reexecution", check_frozen_axes_survive_reexecution),
    ("claim_reconciles_with_reexecution", check_claim_reconciles_with_reexecution),
)


def _verdict_row(verdict) -> dict:
    return {
        "id": verdict.ident,
        "value": verdict.value,
        "reason": verdict.reason,
        "detail": verdict.detail,
        "state": verdict.state,
        "numbers": verdict.numbers,
    }


def grade(telemetry: dict) -> dict:
    """The whole grading decision, as a plain mapping. Pure: no write, no clock."""
    bound = telemetry.get("bound") if isinstance(telemetry.get("bound"), dict) else {}
    rows = []
    for ident, selector in GATE_SELECTORS:
        verdict = selector(telemetry)
        rows.append(_verdict_row(verdict))
        if not verdict.passed:
            return {
                "bound": bound,
                "reward": 0.0,
                "reason": verdict.reason,
                "significance_state": STATE_NOT_APPLICABLE,
                "outcome_classification": "failed",
                "failed_gate": ident,
                "graded_step": graded_step(telemetry),
                "separation": separation(telemetry),
                "checkers": rows,
            }

    significance = check_significance_separation_cleared(telemetry)
    rows.append(_verdict_row(significance))

    base = base_score(telemetry)
    reward = clamp(base * significance.value)
    step = graded_step(telemetry)
    sep = separation(telemetry)

    if significance.state == STATE_UNPROVEN:
        classification = "unproven"
        reason = significance.reason if reward <= 0.0 else ""
    elif base <= 0.0:
        classification = "failed"
        reason = REASON_NO_IMPROVEMENT
    else:
        classification = "established"
        reason = ""

    if reward <= 0.0 and not reason:
        reason = REASON_NO_IMPROVEMENT

    return {
        "bound": bound,
        "reward": reward,
        "reason": reason,
        "significance_state": significance.state or STATE_ESTABLISHED,
        "outcome_classification": classification,
        "failed_gate": "",
        "graded_step": step,
        "separation": sep,
        "base_score": base,
        "significance_factor": significance.value,
        "checkers": rows,
    }


def metric_block(result: dict) -> dict:
    """What the reward measured. One definition, written into BOTH bound documents.

    The anchors come from the `bound` block the verifier's own telemetry carries, which
    tests/runner.py copies out of the operating point, and fall back to the bound
    anchors transcribed above when a run produced no telemetry at all.
    """
    bound = result.get("bound") or {}
    return {
        "graded_step": result.get("graded_step"),
        "baseline": bound.get("baseline_metric", BASELINE_METRIC),
        "target": bound.get("target_metric", TARGET_METRIC),
        "separation": result.get("separation"),
        "separation_margin": bound.get("separation_margin", SEPARATION_MARGIN),
    }


def score_document(result: dict) -> dict:
    """The full score document. Carries `reward`, `reason` and `metric` by obligation."""
    document = {key: value for key, value in result.items() if key != "bound"}
    document["metric"] = metric_block(result)
    return document


def reward_document(result: dict) -> dict:
    """The shaped carrier. Every zero carries a machine-readable reason."""
    return {
        "reward": float(result["reward"]),
        "reason": result.get("reason", ""),
        "significance_state": result.get("significance_state", ""),
        "outcome_classification": result.get("outcome_classification", ""),
        "metric": metric_block(result),
    }


def emit(result: dict, reward_path: str, document_path: str, score_path: str) -> None:
    """Write the three carriers. The single float goes LAST, and it goes alone."""
    for path in (score_path, document_path, reward_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(score_path).write_text(
        json.dumps(score_document(result), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(document_path).write_text(
        json.dumps(reward_document(result), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    Path(reward_path).write_text(format(float(result["reward"]), ".6f") + "\n", encoding="utf-8")


def floor_result(reason: str, detail: str = "") -> dict:
    return {
        "reward": 0.0,
        "reason": reason,
        "detail": detail,
        "significance_state": STATE_NOT_APPLICABLE,
        "outcome_classification": "failed",
        "failed_gate": "",
        "graded_step": None,
        "separation": None,
        "checkers": [],
    }


def substrate_refusal(path: str) -> dict:
    """The refusal tests/runner.py wrote, or None. A run nobody measured is not a zero.

    Reading this is what stops the verifier attributing its own absent substrate to the
    solver. The reason is passed through only if it is in the closed set above.
    """
    carrier = Path(path)
    if not carrier.is_file():
        return None
    try:
        payload = json.loads(carrier.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    reason = str(payload.get("reason", ""))
    if reason not in REFUSAL_REASONS:
        return None
    return {"reason": reason, "detail": str(payload.get("detail", "")),
            "operating_point": str(payload.get("operating_point", ""))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", default=TELEMETRY_PATH)
    parser.add_argument("--reward-path", default=REWARD_PATH)
    parser.add_argument("--reward-document", default=REWARD_DOCUMENT_PATH)
    parser.add_argument("--score-document", default=SCORE_DOCUMENT_PATH)
    parser.add_argument("--refusal", default=SUBSTRATE_REFUSAL_PATH)
    args = parser.parse_args()

    path = Path(args.telemetry)
    if not path.is_file():
        refused = substrate_refusal(args.refusal)
        result = (floor_result(refused["reason"], refused["detail"]) if refused
                  else floor_result(REASON_TELEMETRY_ABSENT))
        if refused:
            result["operating_point"] = refused["operating_point"]
        emit(result, args.reward_path, args.reward_document, args.score_document)
        return 0
    try:
        telemetry = load_telemetry(path)
    except (OSError, ValueError):
        emit(floor_result(REASON_TELEMETRY_UNREADABLE), args.reward_path,
             args.reward_document, args.score_document)
        return 0
    if telemetry.get("schema") != checkers.TELEMETRY_SCHEMA:
        emit(floor_result(REASON_TELEMETRY_UNREADABLE), args.reward_path,
             args.reward_document, args.score_document)
        return 0

    emit(grade(telemetry), args.reward_path, args.reward_document, args.score_document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
