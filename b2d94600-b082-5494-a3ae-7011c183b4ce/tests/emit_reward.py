"""Write the bound reward carrier and its companion score document, always.

Harbor ships no result parser, so these two files are the whole contract. The
bare float lands at /logs/verifier/reward.txt because that is the carrier the
live verifier binds, and the machine-readable reason and metric block land
beside it in /logs/verifier/score.json so the instrument's binding and the
reason carrier can never disagree.

This runs from an EXIT trap, so it runs on every exit path. An aborted verifier
that never staged a score still produces an attributed zero here rather than
silence, because an absent reward reads as an infrastructure fault while an
attributed zero reads as a score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_PATH = "/logs/verifier/score.json"

ABORTED = {
    "reward": 0.0,
    "reason": "verifier-aborted-before-grading",
    "reasons": ["verifier-aborted-before-grading"],
    "metric": {
        "graded_step": None,
        "multi_seed_mean_steps": None,
        "baseline_metric": None,
        "target_metric": None,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
    },
    "checkers": [],
}


def clamp(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))


def document(stage: Path, status: int) -> dict:
    if not stage.is_file():
        row = dict(ABORTED)
        row["exit_status"] = status
        return row
    try:
        row = json.loads(stage.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        row = dict(ABORTED)
        row["reason"] = "score-document-unreadable"
        row["reasons"] = ["score-document-unreadable"]
    if not isinstance(row, dict):
        row = dict(ABORTED)
    row["reward"] = clamp(row.get("reward"))
    if row["reward"] <= 0.0 and not str(row.get("reason", "")).strip():
        row["reason"] = "zero-without-attribution-repaired"
        row["reasons"] = ["zero-without-attribution-repaired"]
    row["exit_status"] = status
    return row


def main(argv) -> int:
    stage = Path(argv[1]) if len(argv) > 1 else Path("/tmp/oer06-stage.json")
    status = int(argv[2]) if len(argv) > 2 and str(argv[2]).lstrip("-").isdigit() else 0
    reward_path = Path(argv[3]) if len(argv) > 3 else Path(REWARD_PATH)
    score_path = Path(argv[4]) if len(argv) > 4 else Path(SCORE_PATH)

    row = document(stage, status)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text("{0:.6f}".format(row["reward"]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
