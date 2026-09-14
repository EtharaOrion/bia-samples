#!/usr/bin/env python3
"""Write the bound reward carrier and its companion score document. Last thing that happens.

The reward carrier is `/logs/verifier/reward.txt` and it holds one bare float.
The reason and the metric block travel in `/logs/verifier/score.json` beside it,
so the number the runtime reads and the explanation a human reads are written
from the same call and can never disagree.

This runs from an EXIT trap, which is the point: a verifier that aborted before
grading still produces a scored, attributed result. Silence would be read as an
infrastructure fault, and an infrastructure fault is not a score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REASON_ABORTED = "verifier-aborted-before-grading"
REASON_NONNUMERIC = "reward-not-a-float"


def clamp(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))


def document(verdict_path: Path, exit_code: int) -> dict:
    try:
        payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    if not isinstance(payload, dict) or "reward" not in payload:
        return {
            "reward": 0.0,
            "reason": REASON_ABORTED,
            "verdict": "unscored",
            "verifier_exit_code": exit_code,
            "metric": {
                "separation_mean": None,
                "noise_half_width": None,
                "separation_margin": 0.05,
                "points_used": 0,
                "anchors_state": "absent",
                "baseline_metric": None,
                "target_metric": None,
            },
        }
    reward = clamp(payload.get("reward"))
    if not isinstance(payload.get("reward"), (int, float)):
        payload["reason"] = REASON_NONNUMERIC
    payload["reward"] = reward
    payload["verifier_exit_code"] = exit_code
    if reward <= 0.0 and not str(payload.get("reason") or "").strip():
        payload["reason"] = REASON_ABORTED
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 reward carrier writer")
    parser.add_argument("--verdict", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--exit-code", default="0")
    args = parser.parse_args(argv)

    try:
        exit_code = int(args.exit_code)
    except (TypeError, ValueError):
        exit_code = 0

    payload = document(Path(args.verdict), exit_code)
    reward_path = Path(args.reward)
    score_path = Path(args.score)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    reward_path.write_text(repr(float(payload["reward"])) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
