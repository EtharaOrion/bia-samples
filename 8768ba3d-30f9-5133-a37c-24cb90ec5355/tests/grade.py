#!/usr/bin/env python3
"""The gate chain and the reward. Imports the checkers and the reference, never the submission.

Aggregation is `required_pass` over the required criteria, exactly as
`tests/checkers.yaml` declares it: every required checker is a gate, and the
first one that fails returns the floor carrying that checker's machine-readable
zero reason. When every required criterion passes, the gate value is 1.0 and the
emitted reward is that gate multiplied by the continuous magnitude term, so the
reward is a float on [0.0, 1.0] and is never binary.

The carriers are written by `--emit-only`, which `tests/test.sh` calls from its
EXIT trap. Both `/logs/verifier/reward.txt` and `/logs/verifier/score.json` are
written from that one path, so the instrument's bound carrier and the reason
carrier can never disagree.

Selectors graded here, named so an outside reader can match them against
`tests/checkers.yaml` row by row:
  check_bound_evaluation_point_reached
  check_token_budget_respected_as_fed
  check_evaluation_split_untrained
  check_pool_state_matches_graded_pool
  check_graded_weights_are_harness_owned
  check_graded_loss_recomputed_unsmoothed
  check_feed_precedes_every_graded_evaluation
  check_loss_sustained_across_verifier_folds
  check_mixture_beats_default_simplex_optimum

The compiled surface for those rows is tests/test_output.py, generated from
solution/grounding.yaml. tests/rubrics.jsonl judges the trajectory separately and
never enters this number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402  the held-out checker set, imported by the grader alone
import runner  # noqa: E402  the verifier's own executor

REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_PATH = "/logs/verifier/score.json"

# A verdict that never got written is not an absent result. It is a verifier that
# aborted, and it is scored zero with a reason that says so.
REASON_VERIFIER_ABORTED = "verifier-aborted-before-verdict"
REASON_NO_PLAN = "submission-produced-no-plan"


def grade(evidence: Path) -> dict:
    """Walk the gate chain over live harness handles and reduce to one float."""
    live = runner.handles(evidence)
    rows = []
    reward = 0.0
    reason = ""
    for ident, selector in checkers.SELECTORS:
        outcome = selector(live)
        rows.append({"id": ident, "required": True, **outcome.as_dict()})
        if not outcome.ok and not reason:
            reason = outcome.reason
    gate = 1.0 if not reason else 0.0

    magnitude_id, magnitude_selector = checkers.MAGNITUDE
    magnitude_outcome = magnitude_selector(live)
    rows.append({"id": magnitude_id, "required": False, **magnitude_outcome.as_dict()})

    if gate >= 1.0:
        value = checkers.magnitude(live)
        if not magnitude_outcome.ok or value <= 0.0:
            reward, reason = 0.0, magnitude_outcome.reason or checkers.REASON_NO_BETTER_THAN_DEFAULT
        else:
            reward, reason = gate * value, "graded"

    report = json.loads(Path(live["eval"]).read_text(encoding="utf-8"))
    feed = json.loads(Path(live["feed"]).read_text(encoding="utf-8"))
    bound = [record for record in report.get("records") or [] if record.get("point") == "bound"]
    calibration = report.get("calibration") or {}
    metric = {
        "agent_metric": float(bound[0]["loss"]) if bound else None,
        "direction": "lower",
        "baseline_metric": None,
        "target_metric": None,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "calibration_gap": "gap-oer08-magnitude-normaliser-is-substrate-local",
        "default_simplex_optimum": (calibration.get("default_simplex_optimum") or {}).get("full"),
        "reference_optimum": (calibration.get("reference_optimum") or {}).get("full"),
        "tokens_fed": feed.get("tokens_fed"),
        "token_budget": feed.get("budget"),
        "graded_point": "budget-exhaustion",
        "readout": "unsmoothed, recomputed by the verifier from harness-owned weights",
    }
    return {"reward": round(min(max(reward, 0.0), 1.0), 12), "reason": reason, "metric": metric, "checkers": rows}


def emit(verdict_path: Path, reward_path: Path, score_path: Path, exit_code: int) -> int:
    """Write both carriers from one place. Never leaves the reward absent."""
    if verdict_path.is_file():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    else:
        verdict = {
            "reward": 0.0,
            "reason": REASON_VERIFIER_ABORTED,
            "metric": {
                "agent_metric": None,
                "baseline_metric": None,
                "target_metric": None,
                "anchors_state": "absent",
                "verifier_exit_code": exit_code,
            },
            "checkers": [],
        }
    value = min(max(float(verdict.get("reward", 0.0)), 0.0), 1.0)
    verdict["reward"] = value
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(repr(value) + "\n", encoding="utf-8")
    score_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission")
    parser.add_argument("--plan")
    parser.add_argument("--corpus", default=str(HERE.parent / "environment" / "corpus_spec.json"))
    parser.add_argument("--split", default=str(HERE / "heldout_spec.json"))
    parser.add_argument("--readout", default=str(HERE.parent / "environment" / "graded_readout.json"))
    parser.add_argument("--evidence")
    parser.add_argument("--verdict", required=True)
    parser.add_argument("--emit-only", action="store_true")
    parser.add_argument("--reward", default=REWARD_PATH)
    parser.add_argument("--score", default=SCORE_PATH)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()

    if args.emit_only:
        return emit(Path(args.verdict), Path(args.reward), Path(args.score), args.exit_code)

    evidence = Path(args.evidence)
    corpus_path = Path(args.corpus)
    if args.submission:
        outcome = runner.run_submission(Path(args.submission), corpus_path)
        plan = outcome["plan"]
    else:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan is None:
        Path(args.verdict).write_text(
            json.dumps(
                {"reward": 0.0, "reason": REASON_NO_PLAN, "metric": {"agent_metric": None}, "checkers": []},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    runner.evaluate(corpus_path, Path(args.split), Path(args.readout), plan, evidence)
    verdict = grade(evidence)
    Path(args.verdict).write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
