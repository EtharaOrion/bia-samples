#!/usr/bin/env python3
"""Reach every graded checker, reduce the outcomes to one float, and name every zero.

The gate chain is `required_pass` over the ten required checkers, followed by the continuous
speedup term. That pairing is what `aggregation.mode: required_pass` in tests/checkers.yaml
describes: the reward is 0.0 unless every required checker passes, and when they all pass the
reward is the clamped normalised speedup rather than a flat 1.0. The reward is therefore never
binary, and every zero it can emit carries a machine-readable reason a downstream grader can
branch on.

The bound reward carrier is /logs/verifier/reward.txt, a bare float. The reason and the metric
block travel in /logs/verifier/score.json. This module writes the score document; the entry
point tests/test.sh owns both writes so a single trap guarantees them on every exit path.

This module imports the checkers, the bound constants and the isolated runner. It never imports
the submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import runner  # noqa: E402

# The bound reward contract path, transcribed from the delivery contract and never chosen here.
BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

# (checker id, selector in tests/checkers.py). The order is the grading order: the first failing
# required checker owns the zero and its reason.
CHECKER_ORDER = (
    ("warehouse_unmodified", "check_warehouse_unmodified"),
    ("plan_is_single_read_only_select", "check_plan_is_single_read_only_select"),
    ("quarantine_row_identified", "check_quarantine_row_identified"),
    ("closure_set_complete", "check_closure_set_complete"),
    ("closure_layers_witnessed", "check_closure_layers_witnessed"),
    ("closure_cardinality_attested", "check_closure_cardinality_attested"),
    ("excluded_rows_absent", "check_excluded_rows_absent"),
    ("result_set_equality", "check_result_set_equality"),
    ("plan_cost_readout_exact", "check_plan_cost_readout_exact"),
    ("plan_faster_than_baseline", "check_plan_faster_than_baseline"),
)

REASON_HARNESS_FAILED = "harness-execution-failed"
REASON_SUBMISSION_ABSENT = "submission-absent"
REASON_ENGINE_DRIFT = "engine-cost-model-drift"


def load_bound_mapping(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "tests" / "bound.json").read_text(encoding="utf-8"))


def load_bound(bundle: Path) -> checkers.Bound:
    return checkers.Bound.from_mapping(load_bound_mapping(bundle))


def baseline_plan_text(bundle: Path) -> str:
    return (Path(bundle) / "tests" / "baseline_plan.sql").read_text(encoding="utf-8")


def _digest(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_harness(bundle: Path, workspace: Path) -> checkers.Harness:
    """Assemble the real harness handle: frozen tables, both plan runs, and the run workspace."""
    bundle = Path(bundle)
    workspace = Path(workspace)

    plan_path = workspace / "plan.sql"
    plan_text = plan_path.read_text(encoding="utf-8") if plan_path.is_file() else ""

    tables = runner.snapshot(bundle)
    baseline_run = runner.execute(bundle, baseline_plan_text(bundle))
    plan_run = runner.execute(bundle, plan_text) if plan_text.strip() else {
        "ok": False,
        "reason": "plan-empty",
        "detail": "no plan.sql was submitted",
    }

    return checkers.Harness(
        lineage=tables["lineage"],
        derives=tables["derives"],
        reading=tables["reading"],
        warehouse_sha256=_digest(bundle / "environment" / "warehouse.db"),
        spec_sha256=_digest(bundle / "solution" / "warehouse_spec.json"),
        plan_text=plan_text,
        plan_run=plan_run,
        baseline_run=baseline_run,
        workspace=workspace,
    )


def run_all(harness: checkers.Harness, bound: checkers.Bound):
    """Every graded checker, in grading order. Pure functions over harness-owned state."""
    outcomes = []
    for ident, selector in CHECKER_ORDER:
        function = getattr(checkers, selector)
        outcomes.append(function(harness, bound))
    return outcomes


def _metric_block(harness, bound, mapping, outcomes):
    return {
        "speedup": checkers.speedup(harness),
        "plan_cost_steps": checkers.plan_cost(harness),
        "baseline_cost_steps": checkers.baseline_cost(harness),
        "bound_baseline_cost_steps": bound.baseline_plan_cost,
        "bound_reference_cost_steps": bound.reference_plan_cost,
        "instance_baseline_speedup": bound.instance_baseline_speedup,
        "instance_target_speedup": bound.instance_target_speedup,
        "instance_anchor_gap": mapping.get("instance_anchor_gap"),
        "anchors_state": mapping.get("anchors_state"),
        "anchors_gap": mapping.get("anchors_gap"),
        "family_baseline_metric": mapping.get("family_baseline_metric"),
        "family_target_metric": mapping.get("family_target_metric"),
        "cost_model": "engine virtual-machine step count, not a wall-clock time",
        "clock_owner": "harness",
        "checkers": [
            {"id": row.ident, "passed": row.passed, "reason": row.reason} for row in outcomes
        ],
    }


def score(harness: checkers.Harness, bound: checkers.Bound, mapping=None) -> dict:
    """One float in [0, 1], higher better, never binary, every zero attributed."""
    mapping = mapping or {}
    measured_baseline = checkers.baseline_cost(harness)
    if measured_baseline is not None and measured_baseline != bound.baseline_plan_cost:
        return {
            "reward": 0.0,
            "reason": REASON_ENGINE_DRIFT,
            "failed_checker": None,
            "detail": "the frozen baseline plan cost "
            + str(measured_baseline)
            + " harness steps here and "
            + str(bound.baseline_plan_cost)
            + " when the bound constants were derived, so the pinned engine is not the engine "
            "this verdict was calibrated on",
            "metric": {"anchors_state": mapping.get("anchors_state")},
        }
    outcomes = run_all(harness, bound)
    metric = _metric_block(harness, bound, mapping, outcomes)
    for row in outcomes:
        if not row.passed:
            return {
                "reward": 0.0,
                "reason": row.reason,
                "failed_checker": row.ident,
                "detail": row.detail,
                "metric": metric,
            }
    span = bound.instance_target_speedup - bound.instance_baseline_speedup
    if span <= 0.0:
        return {
            "reward": 0.0,
            "reason": "anchor-span-degenerate",
            "failed_checker": None,
            "detail": "the instance anchors do not separate, so no gradient exists to score on",
            "metric": metric,
        }
    raw = (metric["speedup"] - bound.instance_baseline_speedup) / span
    clipped = min(max(raw, 0.0), 1.0)
    return {
        "reward": round(clipped, 6),
        "reason": "graded",
        "failed_checker": None,
        "detail": "speedup normalised against the instance-local anchors, result-set gate held",
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="grade one OER-29 run")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True, help="where the score document is written")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    workspace = Path(args.workspace)
    mapping = load_bound_mapping(bundle)
    bound = load_bound(bundle)

    if not (workspace / "submission.json").is_file():
        verdict = {
            "reward": 0.0,
            "reason": REASON_SUBMISSION_ABSENT,
            "failed_checker": None,
            "detail": "no submission.json under " + workspace.as_posix(),
            "metric": {"anchors_state": mapping.get("anchors_state")},
        }
    else:
        try:
            harness = build_harness(bundle, workspace)
        except runner.RunFailed as failure:
            verdict = {
                "reward": 0.0,
                "reason": REASON_HARNESS_FAILED,
                "failed_checker": None,
                "detail": str(failure),
                "metric": {"anchors_state": mapping.get("anchors_state")},
            }
        else:
            verdict = score(harness, bound, mapping)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"reward": verdict["reward"], "reason": verdict["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
