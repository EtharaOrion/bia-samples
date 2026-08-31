#!/usr/bin/env python3
"""Reach every graded checker, reduce the outcomes to one float, and name every zero.

The gate chain is `required_pass` over the nine required checkers, followed by the continuous
speedup term. That pairing is what `aggregation.mode: required_pass` in tests/checkers.yaml
describes: the reward is 0.0 unless every required checker passes, and when they all pass the
reward is the clamped normalised speedup rather than a flat 1.0. The reward is therefore never
binary, and every zero it can emit carries a machine-readable reason a downstream grader can
branch on.

THIS IS THE SECOND PHASE. It grades what the first phase left in a shared workspace: the harness
journal the agent-phase driver chained, and the handoff the agent wrote. It then RESUMES the
operator from the carry that handoff carries, through tests/runner.py, in an isolated
subprocess. The resume's read charge is the graded metric and this verifier counts it.

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
import os
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import runner  # noqa: E402

# The bound reward contract path, transcribed from the delivery contract and never chosen here:
# /logs/verifier/reward.txt carries the bare float and /logs/verifier/score.json carries the
# reason and the metric block.
BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

# The harness log root, which is where the AGENT phase's driver left its journal. Named here as
# a module constant as well as read from the environment so that the workspace carrier is
# discoverable from these bytes alone.
HARNESS_LOGS = Path(os.environ.get("OER28_HARNESS_LOGS", "/logs/harness"))
SUBMISSION = Path(os.environ.get("OER28_SUBMISSION", "/app/submission.py"))
JOURNAL_NAME = "phase_a.jsonl"

# (checker id, selector in tests/checkers.py). The order is the grading order: the first failing
# required checker owns the zero and its reason.
CHECKER_ORDER = (
    ("frozen_inputs_unmodified", "check_frozen_inputs_unmodified"),
    ("agent_phase_journal_present", "check_agent_phase_journal_present"),
    ("journal_chain_intact", "check_journal_chain_intact"),
    ("splice_point_harness_established", "check_splice_point_harness_established"),
    ("carried_state_digest_matches", "check_carried_state_digest_matches"),
    ("kernel_identity_stable_across_splice", "check_kernel_identity_stable_across_splice"),
    ("block_coverage_ordered", "check_block_coverage_ordered"),
    ("resume_output_exact", "check_resume_output_exact"),
    ("charge_not_substituted", "check_charge_not_substituted"),
)

REASON_HANDOFF_ABSENT = "handoff-absent"


def load_bound(bundle: Path) -> checkers.Bound:
    return checkers.Bound.from_mapping(load_bound_mapping(bundle))


def load_bound_mapping(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "tests" / "bound.json").read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError:
        return None


def derive_from_frozen_stream(bundle: Path, params: dict):
    """Re-derive the block starts and the splice point from the frozen bytes alone.

    Deliberately recomputed here rather than believed off a constant, so the splice point the
    checkers compare against is one this verifier established from the stream it is grading.
    """
    stream = _read_json(Path(bundle) / "environment" / "stream.json") or {}
    blocks = stream.get("blocks") or []
    starts = []
    produced = 0
    offset = None
    window = int(params.get("window", 0) or 0)
    budget = int(params.get("splice_charge_budget", 0) or 0)
    for block in blocks:
        starts.append(produced)
        produced += len(block)
        if offset is None and window > 0 and produced * window >= budget:
            offset = produced
    if offset is None:
        offset = produced
    return starts, offset


def build_harness(
    bundle: Path, workspace: Path, harness_logs: Path, submission: Path = SUBMISSION
) -> checkers.Harness:
    """Assemble the real handle across the splice: the agent phase's record plus the resume."""
    bundle = Path(bundle)
    workspace = Path(workspace)
    harness_logs = Path(harness_logs)

    params = _read_json(bundle / "environment" / "params.json") or {}
    starts, derived_offset = derive_from_frozen_stream(bundle, params)

    journal = checkers.read_jsonl(harness_logs / JOURNAL_NAME)
    handoff = _read_json(workspace / "handoff.json")
    handoff = handoff if isinstance(handoff, dict) else {}
    kernel_path = Path(submission)

    harness = checkers.Harness(
        journal=journal,
        resume=[],
        handoff=handoff,
        kernel_sha256=_digest(kernel_path),
        bundle_stream_sha256=_digest(bundle / "environment" / "stream.json"),
        bundle_params_sha256=_digest(bundle / "environment" / "params.json"),
        derived_splice_offset=derived_offset,
        derived_block_starts=starts,
    )

    carry = handoff.get("carried_state")
    offset = handoff.get("splice_offset")
    if not isinstance(carry, dict) or offset is None:
        harness.resume_failure = "the handoff carries no splice offset and carried state to resume from"
        return harness
    try:
        harness.resume = runner.resume(bundle, kernel_path, carry, int(offset))
    except (runner.RunFailed, TypeError, ValueError) as failure:
        harness.resume_failure = str(failure)
    return harness


def run_all(harness: checkers.Harness, bound: checkers.Bound):
    """Every graded checker, in grading order. Pure functions over phase-produced state."""
    outcomes = []
    for ident, selector in CHECKER_ORDER:
        function = getattr(checkers, selector)
        outcomes.append(function(harness, bound))
    return outcomes


def _metric_block(harness, bound, mapping, outcomes):
    charged = checkers.resume_charge(harness)
    return {
        "speedup": checkers.speedup(harness, bound),
        "resume_charge_counted_by_verifier": charged,
        "reference_resume_charge": bound.reference_resume_charge,
        "splice_offset_established_by_agent_phase": harness.halt_row().get("splice_offset"),
        "splice_offset_rederived_by_verifier": harness.derived_splice_offset,
        "carried_state_sha256_recorded": harness.halt_row().get("carried_state_sha256"),
        "agent_phase_blocks": len(harness.block_rows()),
        "resume_blocks": len(harness.resume_blocks()),
        "block_count": bound.block_count,
        "instance_baseline_speedup": bound.instance_baseline_speedup,
        "instance_target_speedup": bound.instance_target_speedup,
        "instance_anchor_gap": mapping.get("instance_anchor_gap"),
        "anchors_state": mapping.get("anchors_state"),
        "anchors_gap": mapping.get("anchors_gap"),
        "family_baseline_metric": mapping.get("family_baseline_metric"),
        "family_target_metric": mapping.get("family_target_metric"),
        "clock_owner": "harness",
        "cost_model": "instrumented Tape reads, never wall clock seconds",
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
        "detail": "resume speedup normalised against the instance-local anchors, correctness held",
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="grade one OER-28 two-phase run")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--harness-logs", default=str(HARNESS_LOGS))
    parser.add_argument("--submission", default=str(SUBMISSION))
    parser.add_argument("--out", required=True, help="where the score document is written")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    workspace = Path(args.workspace)
    mapping = load_bound_mapping(bundle)
    bound = load_bound(bundle)

    if not (workspace / "handoff.json").is_file():
        verdict = {
            "reward": 0.0,
            "reason": REASON_HANDOFF_ABSENT,
            "failed_checker": None,
            "detail": "no handoff.json under " + workspace.as_posix(),
            "metric": {"anchors_state": mapping.get("anchors_state")},
        }
    else:
        harness = build_harness(
            bundle, workspace, Path(args.harness_logs), Path(args.submission)
        )
        verdict = score(harness, bound, mapping)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reward": verdict["reward"], "reason": verdict["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
