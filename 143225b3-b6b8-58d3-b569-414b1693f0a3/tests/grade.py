#!/usr/bin/env python3
"""The gate chain and the reward computation for OER-12.

This module imports the checkers from tests/checkers.py and, on the reference path
only, the reference from solution/reference.py by explicit file path. IT NEVER IMPORTS
THE SUBMISSION. The submission is launched as a separate process by tests/runner.py,
which copies it alone into a fresh temporary directory, runs it as a new session leader
under an environment allowlist, and kills its whole process group in a finally block.

Aggregation is `required_pass`, and the rationale is exactly what this file does: every
required checker must pass for a score to EXIST at all. When one fails, the reward is
0.0 carrying that checker's own machine-readable zero reason. When they all pass, the
reward is the clipped continuous ratio over the graded validation loss, which spans the
interval and is never binary.

This module does not write the reward carrier. It writes one decision document and
tests/test.sh writes /logs/verifier/reward.txt and /logs/verifier/score.json from a
single trap, so the carrier and the reason can never disagree and an aborted verifier
still produces a reason rather than silence.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
from checkers import (  # noqa: E402
    Handles,
    check_ambiguity_policy_declared_before_classification,
    check_bound_evaluation_point_reached,
    check_evaluated_weights_harness_owned,
    check_evaluation_split_not_trained_on,
    check_frozen_axes_unmoved_across_run,
    check_graded_loss_from_verifier_evaluation,
    check_graded_loss_sustained,
    check_graded_readout_unsmoothed,
    check_parse_outcome_classes_partitioned,
    check_parse_outcome_three_way_resolved,
    check_token_budget_respected_as_fed,
)

# The declared surface, mirrored by tests/checkers.yaml and compiled into
# tests/test_output.py. tests/runner.py produced every document these read, and
# tests/rubrics.jsonl judges the trajectory separately and never feeds this number.
GRADED_SELECTORS = (
    check_graded_loss_from_verifier_evaluation,
    check_graded_readout_unsmoothed,
    check_graded_loss_sustained,
    check_bound_evaluation_point_reached,
    check_evaluated_weights_harness_owned,
    check_parse_outcome_classes_partitioned,
    check_parse_outcome_three_way_resolved,
    check_ambiguity_policy_declared_before_classification,
    check_token_budget_respected_as_fed,
    check_frozen_axes_unmoved_across_run,
)

AGGREGATION_MODE = "required_pass"

REASON_ANCHORS_ABSENT = "anchors-absent-metric-unanchored"
REASON_GRADED_LOSS_ABSENT = "graded-loss-absent"
REASON_VERIFIER_ABORTED = "verifier-aborted-before-grading"
REASON_SCORED = "scored"

REWARD_FLOOR = 0.0
REWARD_CEILING = 1.0


def load_reference(path):
    """Import the reference solution by explicit file path. Never the submission.

    Used only on the reference-acceptance path. The submission never travels through
    this function; tests/runner.py launches it as a subprocess instead.
    """
    spec = importlib.util.spec_from_file_location("oer12_reference", str(path))
    if spec is None or spec.loader is None:
        raise ImportError("no reference module at " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handles_for(run_dir: pathlib.Path) -> Handles:
    """The real harness handles: the two directories the verifier's process produced."""
    return Handles(harness=run_dir / "harness", submission_view=run_dir / "submission_view")


def clip(value: float) -> float:
    if value != value:
        return REWARD_FLOOR
    return max(REWARD_FLOOR, min(REWARD_CEILING, value))


def graded_loss(handles: Handles):
    """The graded metric: the verifier's own unsmoothed loss at the bound point."""
    run = checkers.harness_document(handles, "run")
    ledger = checkers.harness_document(handles, "eval_ledger")
    if run is None or ledger is None:
        return None
    for row in ledger.get("records") or []:
        if isinstance(row, dict) and row.get("step") == run.get("bound_eval_step"):
            value = row.get("loss")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def anchor_block(handles: Handles) -> dict:
    """The anchor pair, read from the harness. Never authored here, never invented."""
    payload = checkers.harness_document(handles, "anchors")
    if payload is None:
        return {"anchors_state": "absent", "baseline_metric": None, "target_metric": None}
    return {
        "anchors_state": payload.get("anchors_state", "absent"),
        "baseline_metric": payload.get("baseline_metric"),
        "target_metric": payload.get("target_metric"),
        "authority": payload.get("authority"),
    }


def decide(run_dir: pathlib.Path) -> dict:
    """Run the gate chain, then the metric. One decision document out."""
    handles = handles_for(run_dir)
    rows = []
    failed = None
    for row, verdict in checkers.run_all(handles):
        rows.append(
            {
                "id": row.ident,
                "reduction": row.reduction,
                "required": row.required,
                "weight": row.weight,
                "passed": verdict.passed,
                "zero_reason": row.zero_reason,
                "detail": verdict.detail,
            }
        )
        if row.required and not verdict.passed and failed is None:
            failed = (row, verdict)

    block = anchor_block(handles)
    metric = {
        "graded_loss": graded_loss(handles),
        "baseline_metric": block["baseline_metric"],
        "target_metric": block["target_metric"],
        "anchors_state": block["anchors_state"],
        "aggregation": AGGREGATION_MODE,
        "direction": "lower",
    }

    if failed is not None:
        return {
            "reward": REWARD_FLOOR,
            "reason": failed[1].reason or failed[0].zero_reason,
            "failed_checker": failed[0].ident,
            "metric": metric,
            "checkers": rows,
        }

    agent = metric["graded_loss"]
    if agent is None:
        return {"reward": REWARD_FLOOR, "reason": REASON_GRADED_LOSS_ABSENT,
                "failed_checker": None, "metric": metric, "checkers": rows}

    baseline = metric["baseline_metric"]
    target = metric["target_metric"]
    if (
        block["anchors_state"] != "measured"
        or not isinstance(baseline, (int, float))
        or not isinstance(target, (int, float))
        or float(baseline) == float(target)
    ):
        return {"reward": REWARD_FLOOR, "reason": REASON_ANCHORS_ABSENT,
                "failed_checker": None, "metric": metric, "checkers": rows}

    raw = (float(baseline) - float(agent)) / (float(baseline) - float(target))
    score = clip(raw)
    metric["raw"] = raw
    return {"reward": score, "reason": REASON_SCORED, "failed_checker": None,
            "metric": metric, "checkers": rows}


def main(argv) -> int:
    if len(argv) < 3:
        print("usage: grade.py <run_dir> <decision_json>", file=sys.stderr)
        return 2
    run_dir = pathlib.Path(argv[1])
    out = pathlib.Path(argv[2])
    decision = decide(run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reward": decision["reward"], "reason": decision["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
