#!/usr/bin/env python3
"""Reach every graded checker, reduce the outcomes to one float, and name every zero.

The gate chain is `required_pass` over the nine required checkers, followed by the continuous
size term. That pairing is what `aggregation.mode: required_pass` in tests/checkers.yaml
describes: the reward is 0.0 unless every required checker passes, and when they all pass the
reward is the clamped normalised construction size rather than a flat 1.0. The reward is
therefore never binary, and every zero it can emit carries a machine-readable reason a
downstream grader can branch on.

The bound reward carrier is /logs/verifier/reward.txt, a bare float. The reason and the metric
block travel in /logs/verifier/score.json. This module writes the score document; the entry
point tests/test.sh owns both writes so a single trap guarantees them on every exit path.

This module imports the checkers, the bound constants and the isolated runner. It never
imports the submission: the submission is a program that tests/runner.py launches in its own
process group, and only the JSON document it printed crosses back.
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

# The bound reward contract path, transcribed from the delivery contract and never chosen
# here: /logs/verifier/reward.txt carries the bare float and /logs/verifier/score.json
# carries the reason and the metric block.
BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

# The graded submission path. The literal default is the one task.toml archives and the one
# instruction.md tells the agent to write, so the three agree by inspection rather than by
# convention.
SUBMISSION_PATH = os.environ.get("BIA_SUBMISSION", "/app/submission.py")

# (checker id, selector in tests/checkers.py). The order is the grading order: the first
# failing required checker owns the zero and its reason.
CHECKER_ORDER = (
    ("instance_and_envelope_unmodified", "check_instance_and_envelope_unmodified"),
    ("submission_document_wellformed", "check_submission_document_wellformed"),
    ("seam_offset_recovered", "check_seam_offset_recovered"),
    ("construction_within_ground_set", "check_construction_within_ground_set"),
    ("construction_canonically_ordered", "check_construction_canonically_ordered"),
    ("seam_not_exploited", "check_seam_not_exploited"),
    ("construction_is_admissible", "check_construction_is_admissible"),
    ("seam_witness_exhibits_the_seam", "check_seam_witness_exhibits_the_seam"),
    ("reported_fitness_matches_recomputation", "check_reported_fitness_matches_recomputation"),
)

REASON_PRODUCER_FAILED = "producer-run-failed"
REASON_SUBMISSION_ABSENT = "submission-absent"


def load_bound_mapping(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "tests" / "bound.json").read_text(encoding="utf-8"))


def load_bound(bundle: Path) -> checkers.Bound:
    return checkers.Bound.from_mapping(load_bound_mapping(bundle))


def _digest(path: Path) -> str:
    if not Path(path).is_file():
        return ""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_harness(bundle: Path, document) -> checkers.Harness:
    """Assemble the real handle: the producer's document plus state the verifier read itself."""
    bundle = Path(bundle)
    instance_path = bundle / "environment" / "instance.json"
    envelope_path = bundle / "environment" / "envelope.json"

    modulus = None
    if instance_path.is_file():
        try:
            modulus = int(json.loads(instance_path.read_text(encoding="utf-8"))["modulus"])
        except (ValueError, KeyError, TypeError):
            modulus = None

    observed = None
    state_instance = ""
    state_envelope = ""
    origin = "unreadable"
    try:
        state, origin, _root = runner.read_built_state(bundle)
        observed = int(state["seam_offset"])
        state_instance = str(state.get("instance_sha256", ""))
        state_envelope = str(state.get("envelope_sha256", ""))
    except (runner.RunFailed, OSError, ValueError, KeyError):
        observed = None

    return checkers.Harness(
        document=document if isinstance(document, dict) else None,
        bundle_instance_sha256=_digest(instance_path),
        bundle_envelope_sha256=_digest(envelope_path),
        observed_seam_offset=observed,
        state_instance_sha256=state_instance,
        state_envelope_sha256=state_envelope,
        state_origin=origin,
        modulus_on_disk=modulus,
    )


def run_all(harness: checkers.Harness, bound: checkers.Bound):
    """Every graded checker, in grading order. Pure functions over the handle."""
    outcomes = []
    for _ident, selector in CHECKER_ORDER:
        outcomes.append(getattr(checkers, selector)(harness, bound))
    return outcomes


def _metric_block(harness, bound, mapping, outcomes):
    construction = harness.construction()
    return {
        "certified_size": checkers.graded_certified_size(harness, bound),
        "proxy_certified_size": checkers.proxy_certified_size(harness, bound),
        "declared_size": harness.field_of("declared_size"),
        "construction_length": len(construction),
        "witness_length": len(harness.witness()),
        "seam_offset_declared": harness.field_of("seam_offset"),
        "seam_offset_observed": harness.observed_seam_offset,
        "seam_offset_origin": harness.state_origin,
        "seam_window_width": bound.seam_window_width,
        "modulus": bound.modulus,
        "instance_baseline_size": bound.instance_baseline_size,
        "instance_target_size": bound.instance_target_size,
        "instance_anchor_gap": mapping.get("instance_anchor_gap"),
        "anchors_state": mapping.get("anchors_state"),
        "anchors_gap": mapping.get("anchors_gap"),
        "family_baseline_metric": mapping.get("family_baseline_metric"),
        "family_target_metric": mapping.get("family_target_metric"),
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
    span = bound.instance_target_size - bound.instance_baseline_size
    if span <= 0:
        return {
            "reward": 0.0,
            "reason": "anchor-span-degenerate",
            "failed_checker": None,
            "detail": "the instance anchors do not separate, so no gradient exists to score on",
            "metric": metric,
        }
    raw = checkers.normalised(
        metric["certified_size"], bound.instance_baseline_size, bound.instance_target_size
    )
    return {
        "reward": round(raw, 6),
        "reason": "graded",
        "failed_checker": None,
        "detail": "certified construction size normalised against the instance-local anchors, seam refused",
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="grade one OER-25 run")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True, help="where the score document is written")
    parser.add_argument(
        "--submission",
        default=SUBMISSION_PATH,
        help="the producer program to launch; defaults to the bound submission path",
    )
    parser.add_argument(
        "--producer-output",
        default="",
        help="grade a frozen producer document instead of launching a producer",
    )
    args = parser.parse_args()

    bundle = Path(args.bundle)
    mapping = load_bound_mapping(bundle)
    bound = load_bound(bundle)

    document = None
    verdict = None
    if args.producer_output:
        path = Path(args.producer_output)
        if not path.is_file():
            verdict = {
                "reward": 0.0,
                "reason": REASON_SUBMISSION_ABSENT,
                "failed_checker": None,
                "detail": "no producer document at " + path.as_posix(),
                "metric": {"anchors_state": mapping.get("anchors_state")},
            }
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
    elif not Path(args.submission).is_file():
        verdict = {
            "reward": 0.0,
            "reason": REASON_SUBMISSION_ABSENT,
            "failed_checker": None,
            "detail": "no submission program at " + Path(args.submission).as_posix(),
            "metric": {"anchors_state": mapping.get("anchors_state")},
        }
    else:
        try:
            document = runner.run_producer(
                bundle, Path(args.submission), int(mapping.get("producer_timeout_seconds", 120))
            )
        except runner.RunFailed as failure:
            verdict = {
                "reward": 0.0,
                "reason": REASON_PRODUCER_FAILED,
                "failed_checker": None,
                "detail": str(failure),
                "metric": {"anchors_state": mapping.get("anchors_state")},
            }

    if verdict is None:
        verdict = score(build_harness(bundle, document), bound, mapping)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reward": verdict["reward"], "reason": verdict["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
