#!/usr/bin/env python3
"""Reach every graded checker, reduce the outcomes to one float, and name every zero.

The gate chain is `required_pass` over the eight required checkers, followed by the continuous
gap term. That pairing is what `aggregation.mode: required_pass` in tests/checkers.yaml
describes: the reward is 0.0 unless every required checker passes, and when they all pass the
reward is the clamped normalised closure of the gap to the exact optimum rather than a flat
1.0. The reward is therefore never binary, and every zero it can emit carries a
machine-readable reason a downstream grader can branch on.

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

BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

# (checker id, selector in tests/checkers.py). The order is the grading order: the first
# failing required checker owns the zero and its reason.
CHECKER_ORDER = (
    ("frozen_inputs_unmodified", "check_frozen_inputs_unmodified"),
    ("store_materialised", "check_store_materialised"),
    ("attestation_order_realised", "check_attestation_order_realised"),
    ("seal_chain_derived_from_state", "check_seal_chain_derived_from_state"),
    ("attested_closure_complete", "check_attested_closure_complete"),
    ("digest_recursion_holds", "check_digest_recursion_holds"),
    ("terminal_atom_digest_matches", "check_terminal_atom_digest_matches"),
    ("graded_cover_feasible", "check_graded_cover_feasible"),
)

REASON_HARNESS_MINT_FAILED = "harness-mint-failed"
REASON_SUBMISSION_ABSENT = "submission-absent"


def load_bound(bundle: Path) -> checkers.Bound:
    return checkers.Bound.from_mapping(load_bound_mapping(bundle))


def load_bound_mapping(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "tests" / "bound.json").read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_harness(bundle: Path, workspace: Path) -> checkers.Harness:
    """Assemble the real harness handle: the harness-minted store plus the run workspace."""
    bundle = Path(bundle)
    workspace = Path(workspace)
    store = runner.materialise(bundle)

    instances_path = bundle / "environment" / "instances.json"
    instances = {}
    if instances_path.is_file():
        document = json.loads(instances_path.read_text(encoding="utf-8"))
        instances = {row["instance_id"]: row for row in document["instances"]}

    return checkers.Harness(
        store=store,
        instances=instances,
        bundle_instances_sha256=_digest(instances_path),
        bundle_ledger_sha256=_digest(bundle / "environment" / "ledger_source.json"),
        bundle_minter_sha256=_digest(bundle / "environment" / "mint_store.py"),
        workspace=workspace,
    )


def run_all(harness: checkers.Harness, bound: checkers.Bound):
    """Every graded checker, in grading order. Pure functions over harness-owned state."""
    outcomes = []
    for ident, selector in CHECKER_ORDER:
        outcomes.append(getattr(checkers, selector)(harness, bound))
    return outcomes


def _metric_block(harness, bound, mapping, outcomes):
    instance = checkers.graded_instance(harness)
    return {
        "normalised_gap": checkers.normalised_gap(harness),
        "submitted_cover_weight": checkers.submitted_cover_weight(harness),
        "exact_optimum_weight": checkers.exact_optimum(instance) if instance else None,
        "graded_instance_id": instance["instance_id"] if instance else None,
        "terminal_atom_id": harness.terminal_atom_id(),
        "atoms_in_store": len(harness.atoms()),
        "attestations_in_store": len(harness.attestations()),
        "instance_baseline_gap": bound.instance_baseline_gap,
        "instance_target_gap": bound.instance_target_gap,
        "instance_anchor_gap": mapping.get("instance_anchor_gap"),
        "anchors_state": mapping.get("anchors_state"),
        "anchors_gap": mapping.get("anchors_gap"),
        "family_baseline_metric": mapping.get("family_baseline_metric"),
        "family_target_metric": mapping.get("family_target_metric"),
        "comparator": "exact bitmask dynamic program, recomputed by the verifier",
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
    span = bound.instance_baseline_gap - bound.instance_target_gap
    if span <= 0.0:
        return {
            "reward": 0.0,
            "reason": "anchor-span-degenerate",
            "failed_checker": None,
            "detail": "the instance anchors do not separate, so no gradient exists to score on",
            "metric": metric,
        }
    gap = metric["normalised_gap"]
    if gap is None:
        return {
            "reward": 0.0,
            "reason": "metric-unreadable",
            "failed_checker": None,
            "detail": "the verifier could not recompute a normalised gap for this submission",
            "metric": metric,
        }
    raw = (bound.instance_baseline_gap - gap) / span
    return {
        "reward": round(min(max(raw, 0.0), 1.0), 6),
        "reason": "graded",
        "failed_checker": None,
        "detail": "gap to the exact optimum normalised against the instance-local anchors",
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="grade one OER-30 run")
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
                "reason": REASON_HARNESS_MINT_FAILED,
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
