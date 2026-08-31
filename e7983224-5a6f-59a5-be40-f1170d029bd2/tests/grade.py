#!/usr/bin/env python3
"""The gate chain for slot OER-14: run every declared checker, then write one reward.

This file is the `reached_by` carrier every row of `tests/checkers.yaml` names. It imports
the checkers and the harness. It NEVER imports the submission: `tests/runner.py` runs that in
a fresh temporary directory as its own session leader and hands back a document.

What this does, in order:

  1. obtains the submission document through runner, executing nothing in this process;
  2. asks harness to build the harness-owned context, which is where every graded number is
     born, from the frozen substrate and the frozen evaluation corpus;
  3. runs the gate chain in `checkers.REGISTRY` order and stops at the first gate that
     refuses, because the first refusal is the one that caused the rest;
  4. writes the bare float to `/logs/verifier/reward.txt` and the machine-readable reason and
     metric block to `/logs/verifier/score.json`.

Aggregation is `required_pass` over the declared checkers: every checker is a required gate,
so one refusal resolves the reward to 0.0 with that checker's `zero_reason`. When every
required gate passes, the reward is the bound continuous formula
`raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)` clipped into the
closed interval, evaluated against anchors the harness MEASURED in this process. The reward
is therefore never binary: it takes every value in the band between the separation margin and
the target.

The selectors this carrier reaches, named here so an outside reader can recompute the
manifest's reachability claim from bytes:

    graded_bpb_recomputed_unsmoothed
    denominator_is_frozen_eval_byte_count
    compute_spend_within_frozen_budget
    evaluated_state_is_harness_owned
    early_stop_does_not_establish_reading
    reading_sustained_across_scheduled_points
    no_submission_reported_number_on_graded_path
    graded_band_exceeds_single_direction_sweep
    reallocation_follows_flattening
    carried_direction_frontier_never_collapses
    corpus_phase_matches_frozen_offset
    corpus_period_matches_frozen_stride

The trajectory rubrics in `tests/rubrics.jsonl` are judged elsewhere, by a judge reading the
trajectory. They are bucket N: they may lower an outcome and never raise one, and this file
does not read them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import harness  # noqa: E402
import runner  # noqa: E402

#: The bound reward contract. `/logs/verifier/` is a single shared host path, so an exercise
#: run may redirect the root; the bound paths inside this bundle are these two and only these.
REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_PATH = "/logs/verifier/score.json"

MANIFEST = "tests/checkers.yaml"


def _write(reward_file: Path, score_file: Path, reward: float, reason: str, metric: dict) -> None:
    """The last thing that happens. The bare float first, then the reason document."""
    value = min(max(float(reward), 0.0), 1.0)
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    score_file.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "reward": value,
        "reason": reason,
        "metric": metric,
        "manifest": MANIFEST,
        "aggregation": "required_pass",
    }
    score_file.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reward_file.write_text(repr(value) + "\n", encoding="utf-8")


def _metric_block(context) -> dict:
    if not context:
        return {
            "graded_bpb": None,
            "baseline_metric": None,
            "target_metric": None,
            "denominator_bytes": None,
            "spent_units": None,
            "anchors_state": "absent",
        }
    return {
        "graded_bpb": context["graded"]["bpb"],
        "baseline_metric": context["anchors"]["baseline_metric"],
        "target_metric": context["anchors"]["target_metric"],
        "baseline_source": context["anchors"]["baseline_source"],
        "target_source": context["anchors"]["target_source"],
        "denominator_bytes": context["frozen"]["eval_bytes"],
        "denominator_sha256": context["frozen"]["eval_sha256"],
        "spent_units": context["meter"]["spent_units"],
        "budget_units": context["frozen"]["budget_units"],
        "scheduled_points": context["frozen"]["scheduled_points"],
        "anchors_state": context["anchors"]["anchors_state"],
        "anchors_gap": context["anchors"]["anchors_gap"],
    }


def gate(context: dict) -> tuple:
    """Run every required gate in declared order. Returns (reason, detail, results)."""
    results = []
    refusal = ("", "")
    for ident, selector in checkers.REGISTRY:
        outcome = selector(context)
        results.append(
            {
                "id": ident,
                "passed": bool(outcome.passed),
                "reason": outcome.reason,
                "detail": outcome.detail,
            }
        )
        if not outcome.passed and not refusal[0]:
            refusal = (outcome.reason, outcome.detail)
    return refusal[0], refusal[1], results


def score(document) -> tuple:
    """The whole graded pass over one submission document. Returns (reward, reason, metric)."""
    context = harness.context_for(document)
    reason, detail, results = gate(context)
    metric = _metric_block(context)
    metric["checkers"] = results
    if reason:
        metric["refused_by"] = next(row["id"] for row in results if row["reason"] == reason)
        metric["detail"] = detail
        return 0.0, reason, metric
    reward = harness.reward_for(context)
    return reward, "every-required-checker-passed", metric


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="grade one OER-14 submission")
    parser.add_argument("--submission", default="", help="path to an existing submission document")
    parser.add_argument("--entry", default="", help="path to an entry point runner executes")
    parser.add_argument("--reward-file", default=REWARD_PATH)
    parser.add_argument("--score-file", default=SCORE_PATH)
    parser.add_argument("--abort", default="", help="write a refusal for an aborted verifier")
    args = parser.parse_args(argv)

    reward_file, score_file = Path(args.reward_file), Path(args.score_file)

    if args.abort:
        _write(reward_file, score_file, 0.0, args.abort, _metric_block(None))
        return 0

    if args.entry:
        run = runner.execute(Path(args.entry))
    else:
        run = runner.load(Path(args.submission or "/workspace/submission.json"))

    if not run.ok:
        _write(reward_file, score_file, 0.0, run.reason, _metric_block(None))
        return 0

    reward, reason, metric = score(run.document)
    _write(reward_file, score_file, reward, reason, metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
