"""The single writer of both reward carriers. Called only from the EXIT trap in test.sh.

Two files, one writer, one moment:

  /logs/verifier/reward.txt    one BARE FLOAT, the carrier the live seed/forge/verifier.py
                               binds. Not JSON. Not a dict. One number and a newline.
  /logs/verifier/score.json    the machine-readable reason and the metric block.

They are written here together so the instrument's binding and the reason carrier can
never disagree. If the verifier aborted before grading produced a verdict, this writes
a reward of 0.0 carrying `verifier-aborted-before-grading` and the stage it died in, so
an aborted verifier still produces a reason rather than silence.

Nothing here computes a score. It transcribes, clamps, and attributes.
"""

from __future__ import annotations

import json
import os
import sys

REWARD_FLOOR = 0.0
REWARD_CEILING = 1.0


def _empty_metric() -> dict:
    return {
        "graded_validation_loss": None,
        "baseline_metric": None,
        "target_metric": None,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "graded_snapshot_version": None,
        "bound_evaluation_step": None,
        "direction": "lower",
    }


def load_verdict(path: str, stage: str) -> dict:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = None
        if isinstance(payload, dict) and "reward" in payload:
            return payload
    return {
        "reward": REWARD_FLOOR,
        "reason": "verifier-aborted-before-grading",
        "failed_checker": None,
        "observed": "the verifier exited during stage " + str(stage) + " without producing a verdict",
        "metric": _empty_metric(),
        "checkers": [],
        "aggregation": "required_pass",
    }


def clamp(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return REWARD_FLOOR
    if number != number:
        return REWARD_FLOOR
    return max(REWARD_FLOOR, min(REWARD_CEILING, number))


def emit(verdict_path: str, stage: str, reward_path: str, score_path: str) -> float:
    payload = load_verdict(verdict_path, stage)
    reward = clamp(payload.get("reward"))
    payload["reward"] = reward
    if reward <= REWARD_FLOOR and not payload.get("reason"):
        payload["reason"] = "reward-zero-unattributed"

    for target in (reward_path, score_path):
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)

    with open(score_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")

    # The bare float, and nothing else, into /logs/verifier/reward.txt.
    with open(reward_path, "w", encoding="utf-8") as handle:
        handle.write(repr(reward) + "\n")

    sys.stderr.write(
        "OER-11 reward " + repr(reward) + " reason " + str(payload.get("reason")) + "\n"
    )
    return reward


def main(argv: list) -> int:
    verdict_path = argv[0] if argv else "/tmp/oer11-verdict.json"
    stage = argv[1] if len(argv) > 1 else "unknown"
    reward_path = argv[2] if len(argv) > 2 else "/logs/verifier/reward.txt"
    score_path = argv[3] if len(argv) > 3 else "/logs/verifier/score.json"
    emit(verdict_path, stage, reward_path, score_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
