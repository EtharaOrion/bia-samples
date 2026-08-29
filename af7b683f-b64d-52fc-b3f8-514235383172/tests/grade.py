#!/usr/bin/env python3
"""Reach every graded checker, reduce the outcomes to one float, and name every zero.

The gate chain is `required_pass` over the eight required checkers, followed by the
continuous throughput term. That pairing is what `aggregation.mode: required_pass` in
tests/checkers.yaml describes: the reward is 0.0 unless every required checker passes, and
when they all pass the reward is the clamped normalised throughput rather than a flat 1.0.
The reward is therefore never binary, and every zero it can emit carries a machine-readable
reason a downstream grader can branch on.

The bound reward carrier is /logs/verifier/reward.txt, a bare float. The reason and the
metric block travel in /logs/verifier/score.json. This module writes the score document; the
entry point tests/test.sh owns both writes so a single trap guarantees them on every exit
path.

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
    ("trace_and_envelope_unmodified", "check_trace_and_envelope_unmodified"),
    ("iteration_sequence_ordered", "check_iteration_sequence_ordered"),
    ("summary_matches_durable_record", "check_summary_matches_durable_record"),
    ("every_request_accounted", "check_every_request_accounted"),
    ("no_early_stop_harvest", "check_no_early_stop_harvest"),
    ("unsmoothed_graded_readout", "check_unsmoothed_graded_readout"),
    ("slo_p99_respected", "check_slo_p99_respected"),
    ("measurement_windows_sustained", "check_measurement_windows_sustained"),
)

REASON_HARNESS_REPLAY_FAILED = "harness-replay-failed"
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
    """Assemble the real harness handle: harness-owned telemetry plus the run workspace."""
    bundle = Path(bundle)
    workspace = Path(workspace)
    submission_path = workspace / "submission.json"
    submission = {}
    if submission_path.is_file():
        try:
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
        except ValueError:
            submission = {}
    config = submission.get("config") if isinstance(submission, dict) else None

    telemetry = []
    if isinstance(config, dict):
        telemetry = runner.replay(bundle, config)

    trace_path = bundle / "environment" / "trace.json"
    frozen_trace = {}
    if trace_path.is_file():
        frozen_trace = json.loads(trace_path.read_text(encoding="utf-8"))

    return checkers.Harness(
        telemetry=telemetry,
        frozen_trace=frozen_trace,
        bundle_trace_sha256=_digest(trace_path),
        bundle_envelope_sha256=_digest(bundle / "environment" / "envelope.json"),
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
    tps = checkers.tokens_per_second(harness)
    p99 = checkers.p99_latency_ms(harness)
    return {
        "tokens_per_second": tps,
        "p99_latency_ms": p99,
        "p99_latency_slo_ms": bound.p99_latency_slo_ms,
        "requests_in_frozen_trace": bound.request_count,
        "completions": len(harness.completions()),
        "rejections": len(harness.rejections()),
        "instance_baseline_tps": bound.instance_baseline_tps,
        "instance_target_tps": bound.instance_target_tps,
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
    span = bound.instance_target_tps - bound.instance_baseline_tps
    if span <= 0.0:
        return {
            "reward": 0.0,
            "reason": "anchor-span-degenerate",
            "failed_checker": None,
            "detail": "the instance anchors do not separate, so no gradient exists to score on",
            "metric": metric,
        }
    raw = (metric["tokens_per_second"] - bound.instance_baseline_tps) / span
    clipped = min(max(raw, 0.0), 1.0)
    return {
        "reward": round(clipped, 6),
        "reason": "graded",
        "failed_checker": None,
        "detail": "throughput normalised against the instance-local anchors, objective held",
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="grade one OER-24 run")
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
                "reason": REASON_HARNESS_REPLAY_FAILED,
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
