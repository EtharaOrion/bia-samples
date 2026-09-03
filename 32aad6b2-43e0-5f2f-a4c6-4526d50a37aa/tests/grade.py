#!/usr/bin/env python3
"""The gate chain and the reward. Imports the checkers; never imports the submission.

`tests/runner.py` executes the submission in isolation and measures it. This file
reads that measurement, walks the bound gate chain in `tests/checkers.py`, and
resolves one float in the closed interval together with the reason that float is
what it is. It writes a verdict document; `tests/test.sh` is what puts the bound
reward carrier on disk, from a trap, so an aborted verifier still leaves a reason
rather than silence.

The graded quantity is stated once, here and in instruction.md, in the same words:
the verifier's own accuracy of the frozen model on the frozen held-out benchmark at
the bound evaluation point. It is never a number the submission printed, never a
field the submission wrote, and never a smoothed or selected reading.

Anchors are absent for family F14. This file therefore binds the reward schema in
full and refuses to invent a denominator: with either anchor unbound the reward
resolves to 0.0 carrying `anchors-absent-unnormalizable`, and the verifier-computed
`agent_metric` is recorded in the score document so the batch can normalize it once
the anchors are measured. `--baseline` and `--target` exist for the feasibility
exercise alone, which supplies a declared fixture pair and records it as fixture
material rather than as a slot binding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
MANIFEST = HERE / "checkers.yaml"
BENCHMARK = HERE / "benchmark_held_out.jsonl"

sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import runner  # noqa: E402

# The selectors this file reaches. Declared literally so the manifest's claim that
# each checker is reachable from the entry point is a fact about these bytes.
SELECTORS = (
    "generator_collapse_absent",
    "corpus_mode_share_bounded",
    "emission_order_monotone",
    "no_benchmark_near_duplicate",
    "training_budget_as_fed",
    "unsmoothed_graded_readout",
    "early_stop_does_not_establish_score",
    "score_sustained_across_scheduled_points",
    "graded_score_from_harness_model_state",
    "self_report_matches_recomputation",
)

REWARD_FLOOR = 0.0
REWARD_CEILING = 1.0


def manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def bounds() -> dict:
    return manifest().get("bounds") or {}


def anchors() -> dict:
    return manifest().get("anchors") or {}


def clip(value: float) -> float:
    if value != value:
        return REWARD_FLOOR
    return max(REWARD_FLOOR, min(REWARD_CEILING, value))


def score(telemetry, baseline=None, target=None) -> dict:
    """One float, one reason, and the metric block that explains both."""
    agent = None
    evaluation = telemetry.get("evaluation") if isinstance(telemetry, dict) else None
    if isinstance(evaluation, dict) and isinstance(evaluation.get("graded_accuracy"), (int, float)):
        agent = float(evaluation["graded_accuracy"])

    metric = {
        "agent_metric": agent,
        "baseline_metric": baseline,
        "target_metric": target,
        "anchors_state": "absent" if (baseline is None or target is None) else "fixture-supplied",
        "graded_point": (evaluation or {}).get("graded_point"),
        "direction": "higher-is-better",
        "formula": "raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)",
    }

    if not isinstance(telemetry, dict) or not telemetry.get("corpus"):
        return {"reward": REWARD_FLOOR, "reason": "telemetry-absent", "metric": metric, "gates": []}

    gates = []
    for name, verdict in checkers.run_chain(telemetry):
        gates.append({"checker": name, "passed": verdict.passed, "detail": verdict.detail})
        if not verdict.passed:
            return {
                "reward": REWARD_FLOOR,
                "reason": verdict.reason,
                "metric": metric,
                "gates": gates,
            }

    if agent is None:
        return {"reward": REWARD_FLOOR, "reason": "telemetry-absent", "metric": metric, "gates": gates}
    if baseline is None or target is None:
        return {
            "reward": REWARD_FLOOR,
            "reason": "anchors-absent-unnormalizable",
            "metric": metric,
            "gates": gates,
        }
    if float(target) == float(baseline):
        return {
            "reward": REWARD_FLOOR,
            "reason": "anchors-degenerate-interval",
            "metric": metric,
            "gates": gates,
        }
    raw = (agent - float(baseline)) / (float(target) - float(baseline))
    value = clip(raw)
    metric["raw"] = raw
    if value <= REWARD_FLOOR:
        return {
            "reward": REWARD_FLOOR,
            "reason": "score-at-or-below-baseline",
            "metric": metric,
            "gates": gates,
        }
    return {"reward": value, "reason": "graded", "metric": metric, "gates": gates}


def measure(submission: Path) -> dict:
    """Run the submission in isolation and hand back the verifier's own telemetry."""
    tools = BUNDLE / "environment" / "tools"
    outcome = runner.run_submission(submission, tools)
    held_out = runner.load_benchmark(BENCHMARK)
    return runner.produce_telemetry(
        corpus_rows=outcome["corpus"],
        held_out=held_out,
        generator_sha256=outcome["sha256"],
        generator_exit_status=outcome["exit_status"],
        self_report=outcome["self_report"],
        bounds=bounds(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade one OER-19 run.")
    parser.add_argument("--submission", help="path to the generator to execute in isolation")
    parser.add_argument("--telemetry", help="path to a telemetry document already produced")
    parser.add_argument("--emit", required=True, help="path to write the verdict document")
    parser.add_argument("--baseline", type=float, default=None, help="fixture-supplied anchor only")
    parser.add_argument("--target", type=float, default=None, help="fixture-supplied anchor only")
    args = parser.parse_args()

    if args.telemetry:
        with Path(args.telemetry).open("r", encoding="utf-8") as handle:
            telemetry = json.load(handle)
    elif args.submission:
        telemetry = measure(Path(args.submission))
    else:
        telemetry = {}

    bound = anchors()
    baseline = args.baseline if args.baseline is not None else bound.get("baseline_metric")
    target = args.target if args.target is not None else bound.get("target_metric")

    verdict = score(telemetry, baseline, target)
    verdict["telemetry_schema"] = telemetry.get("schema") if isinstance(telemetry, dict) else None
    Path(args.emit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.emit).write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"reward": verdict["reward"], "reason": verdict["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
