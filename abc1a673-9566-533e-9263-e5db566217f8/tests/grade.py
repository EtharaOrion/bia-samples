#!/usr/bin/env python3
"""Reach every graded checker, reduce the outcomes to one float, and name every zero.

The gate chain is `required_pass` over the eight required checkers, followed by the continuous
agreement term. That pairing is what `aggregation.mode: required_pass` in tests/checkers.yaml
describes: the reward is 0.0 unless every required checker passes, and when they all pass the
reward is the clamped normalised verdict agreement rather than a flat 1.0. The reward is
therefore never binary, and every zero it can emit carries a machine-readable reason a
downstream grader can branch on.

The bound reward carrier is /logs/verifier/reward.txt, a bare float. The reason and the metric
block travel in /logs/verifier/score.json. This module writes the score document; the entry
point tests/test.sh owns both writes so a single trap guarantees them on every exit path.

This module imports the checkers, the bound constants and the isolated runner. It never imports
the submission.
"""

from __future__ import annotations

import argparse
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
    ("fence_state_unmodified", "check_fence_state_unmodified"),
    ("crossing_count_accounted", "check_crossing_count_accounted"),
    ("admitted_kind_set_exact", "check_admitted_kind_set_exact"),
    ("verdict_sequence_ordered", "check_verdict_sequence_ordered"),
    ("verdict_records_match_traversal", "check_verdict_records_match_traversal"),
    ("no_unadmitted_kind_admitted", "check_no_unadmitted_kind_admitted"),
    ("quota_ledger_respected", "check_quota_ledger_respected"),
    ("admission_not_blanket", "check_admission_not_blanket"),
)

REASON_PROBE_FAILED = "harness-probe-failed"
REASON_SUBMISSION_ABSENT = "submission-absent"


def load_bound_mapping(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "tests" / "bound.json").read_text(encoding="utf-8"))


def load_bound(bundle: Path) -> checkers.Bound:
    return checkers.Bound.from_mapping(load_bound_mapping(bundle))


def build_harness(workspace: Path, telemetry=None) -> checkers.Harness:
    """Assemble the real harness handle: harness-owned telemetry plus the run workspace."""
    rows = runner.observe() if telemetry is None else telemetry
    return checkers.Harness(telemetry=rows, workspace=Path(workspace))


def run_all(harness: checkers.Harness, bound: checkers.Bound):
    """Every graded checker, in grading order. Pure functions over harness-owned state."""
    outcomes = []
    for ident, selector in CHECKER_ORDER:
        function = getattr(checkers, selector)
        outcomes.append(function(harness, bound))
    return outcomes


def _metric_block(harness, bound, mapping, outcomes):
    truth = checkers.reference_verdicts(harness)
    return {
        "crossings_presented": checkers.crossing_count(harness),
        "admitted_kind_count": len(checkers.admitted_kind_set(harness)),
        "verdict_agreement": checkers.agreement(harness),
        "baseline_agreement": bound.baseline_agreement,
        "admissible_crossings": sum(1 for row in truth if row["decision"] == "admit"),
        "submitted_verdicts": len(checkers.submitted_verdicts(harness)),
        "anchors_state": mapping.get("anchors_state"),
        "anchors_gap": mapping.get("anchors_gap"),
        "family_baseline_metric": mapping.get("family_baseline_metric"),
        "family_target_metric": mapping.get("family_target_metric"),
        "instance_anchor_gap": mapping.get("instance_anchor_gap"),
        "clock_owner": "harness",
        "checkers": [
            {"id": row.ident, "passed": row.passed, "reason": row.reason} for row in outcomes
        ],
    }


def score(harness: checkers.Harness, bound: checkers.Bound, mapping=None) -> dict:
    """One float in [0, 1], higher better, never binary, every zero attributed."""
    mapping = mapping or {}
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
    span = 1.0 - bound.baseline_agreement
    if span <= 0.0:
        return {
            "reward": 0.0,
            "reason": "anchor-span-degenerate",
            "failed_checker": None,
            "detail": "the baseline agreement leaves no gradient to score on",
            "metric": metric,
        }
    raw = (metric["verdict_agreement"] - bound.baseline_agreement) / span
    clipped = min(max(raw, 0.0), 1.0)
    return {
        "reward": round(clipped, 6),
        "reason": "graded",
        "failed_checker": None,
        "detail": "verdict agreement normalised against the instance-local baseline policy",
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="grade one OER-26 run")
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
            harness = build_harness(workspace)
        except runner.RunFailed as failure:
            verdict = {
                "reward": 0.0,
                "reason": REASON_PROBE_FAILED,
                "failed_checker": None,
                "detail": str(failure),
                "metric": {"anchors_state": mapping.get("anchors_state")},
            }
        else:
            verdict = score(harness, bound, mapping)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reward": verdict["reward"], "reason": verdict["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
