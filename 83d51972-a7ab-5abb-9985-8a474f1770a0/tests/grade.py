#!/usr/bin/env python3
"""Reach every graded checker, reduce the outcomes to one float, and name every zero.

The gate chain is `required_pass` over the nine required checkers, followed by the continuous
recall term. That pairing is what `aggregation.mode: required_pass` in tests/checkers.yaml
describes: the reward is 0.0 unless every required checker passes, and when they all pass the
reward is the clamped normalised recall rather than a flat 1.0. The reward is therefore never
binary, and every zero it can emit carries a machine-readable reason a downstream grader can
branch on.

The bound reward carrier is /logs/verifier/reward.txt, a bare float. The reason and the metric
block travel in /logs/verifier/score.json. This module writes the score document; the entry
point tests/test.sh owns both writes so a single trap guarantees them on every exit path.

This module imports the checkers, the bound constants and the isolated runner. It never
imports the submission.
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

# The bound reward contract path, transcribed from the delivery contract and never chosen
# here: /logs/verifier/reward.txt carries the bare float and /logs/verifier/score.json
# carries the reason and the metric block.
BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

# (checker id, selector in tests/checkers.py). The order is the grading order: the first
# failing required checker owns the zero and its reason.
CHECKER_ORDER = (
    ("corpus_and_spec_unmodified", "check_corpus_and_spec_unmodified"),
    ("submission_pairs_wellformed", "check_submission_pairs_wellformed"),
    ("pair_listing_ordered", "check_pair_listing_ordered"),
    ("adjudication_threshold_established", "check_adjudication_threshold_established"),
    ("index_collision_census_agrees", "check_index_collision_census_agrees"),
    ("adjudication_not_degenerate", "check_adjudication_not_degenerate"),
    ("no_unsupported_pair", "check_no_unsupported_pair"),
    ("collision_witness_verified", "check_collision_witness_verified"),
    ("recall_sustained_across_shards", "check_recall_sustained_across_shards"),
)

REASON_HARNESS_BUILD_FAILED = "harness-index-build-failed"
REASON_SUBMISSION_ABSENT = "submission-absent"


def load_bound(bundle: Path) -> checkers.Bound:
    payload = json.loads((Path(bundle) / "tests" / "bound.json").read_text(encoding="utf-8"))
    return checkers.Bound.from_mapping(payload)


def load_bound_mapping(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "tests" / "bound.json").read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_harness(bundle: Path, workspace: Path) -> checkers.Harness:
    """Assemble the real harness handle: a freshly rebuilt index plus the run workspace."""
    bundle = Path(bundle)
    workspace = Path(workspace)
    telemetry = runner.rebuild(bundle)
    return checkers.Harness(
        telemetry=telemetry,
        bundle_corpus_sha256=_digest(bundle / "environment" / "corpus.jsonl"),
        bundle_index_spec_sha256=_digest(bundle / "environment" / "index_spec.json"),
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
    truth = harness.near_duplicate_pairs()
    submitted = checkers.normalised_pairs(harness)
    witness = harness.witness() or {}
    return {
        "near_duplicate_recall": checkers.recall(harness),
        "submitted_pairs": len(submitted),
        "verifier_pairs": len(truth),
        "verifier_duplicate_groups": harness.duplicate_group_count(),
        "index_collisions": len(harness.collisions()),
        "adjudication_threshold_established": harness.threshold(),
        "collision_witness": [witness.get("a"), witness.get("b")] if witness else None,
        "instance_baseline_recall": bound.instance_baseline_recall,
        "instance_target_recall": bound.instance_target_recall,
        "instance_anchor_gap": mapping.get("instance_anchor_gap"),
        "anchors_state": mapping.get("anchors_state"),
        "anchors_gap": mapping.get("anchors_gap"),
        "family_baseline_metric": mapping.get("family_baseline_metric"),
        "family_target_metric": mapping.get("family_target_metric"),
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
    span = bound.instance_target_recall - bound.instance_baseline_recall
    if span <= 0.0:
        return {
            "reward": 0.0,
            "reason": "anchor-span-degenerate",
            "failed_checker": None,
            "detail": "the instance anchors do not separate, so no gradient exists to score on",
            "metric": metric,
        }
    raw = (metric["near_duplicate_recall"] - bound.instance_baseline_recall) / span
    clipped = min(max(raw, 0.0), 1.0)
    return {
        "reward": round(clipped, 6),
        "reason": "graded",
        "failed_checker": None,
        "detail": "recall normalised against the instance-local anchors, precision gate held",
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="grade one OER-27 run")
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
                "reason": REASON_HARNESS_BUILD_FAILED,
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
