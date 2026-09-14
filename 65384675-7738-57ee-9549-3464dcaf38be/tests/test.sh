#!/usr/bin/env bash
# The one entry point Harbor runs for OER-12.
#
# Harbor ships no result parser of its own, so this file is the whole reward contract.
# It terminates by writing the bound carrier /logs/verifier/reward.txt, a bare float in
# the closed interval [0.0, 1.0], higher better, never binary, and the companion
# document /logs/verifier/score.json carrying the machine-readable reason and the
# metric block. BOTH are written from ONE trap, so the carrier and the reason can never
# disagree, and an aborted verifier still produces a reason rather than silence.
set -euo pipefail

REWARD_FILE="/logs/verifier/reward.txt"
SCORE_FILE="/logs/verifier/score.json"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${OER12_RUN_DIR:-/tmp/oer12-run}"
DECISION="${RUN_DIR}/decision.json"

emit_reward() {
  mkdir -p /logs/verifier
  python3 - "$DECISION" "$REWARD_FILE" "$SCORE_FILE" <<'PYEMIT' || true
import json
import pathlib
import sys

# Every mapping below is built with dict(...) rather than a brace literal. A line that
# strips to a bare closing brace would terminate this shell function as the verifier's
# trap analyser reads it, cutting the reward write out of the trap body it can see.
decision_path, reward_path, score_path = sys.argv[1], sys.argv[2], sys.argv[3]
fallback = dict(
    reward=0.0,
    reason="verifier-aborted-before-grading",
    failed_checker=None,
    metric=dict(graded_loss=None, baseline_metric=None, target_metric=None, anchors_state="absent"),
    checkers=[],
)
try:
    decision = json.loads(pathlib.Path(decision_path).read_text(encoding="utf-8"))
except Exception:
    decision = fallback
try:
    reward = float(decision.get("reward", 0.0))
except (TypeError, ValueError):
    reward = 0.0
if reward != reward:
    reward = 0.0
reward = max(0.0, min(1.0, reward))
reason = str(decision.get("reason") or "verifier-aborted-before-grading")
pathlib.Path(reward_path).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(reward_path).write_text("%.6f\n" % reward, encoding="utf-8")
document = dict(
    reward=reward,
    reason=reason,
    failed_checker=decision.get("failed_checker"),
    metric=decision.get("metric"),
    checkers=decision.get("checkers") or [],
)
pathlib.Path(score_path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PYEMIT
  if [ ! -s /logs/verifier/reward.txt ]; then
    printf '0.0\n' > /logs/verifier/reward.txt
    printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "failed_checker": null, "metric": null, "checkers": []}' > /logs/verifier/score.json
  fi
}

# Registered before anything can fail. Under errexit any failing command ends the
# script, so without this trap an unguarded script could reach its end without ever
# writing the carrier, and an absent reward reads as an infrastructure fault instead of
# as a score with a reason.
trap 'emit_reward' EXIT

mkdir -p "$RUN_DIR" /logs/verifier

# 1. The harness. tests/runner.py performs the verifier's own independent parse
#    classification, launches the submission in isolation, and writes every telemetry
#    document the checkers read. It never lets the grading process import the
#    submission.
python3 "${HERE}/runner.py"

# 2. The gate chain and the metric. tests/grade.py imports tests/checkers.py, runs
#    every checker declared in tests/checkers.yaml, and writes one decision document.
python3 "${HERE}/grade.py" "$RUN_DIR" "$DECISION"

# 3. The compiled mirror of the same checker set, generated from solution/grounding.yaml.
python3 "${HERE}/test_output.py" "$RUN_DIR"

# The trap performs the write. It is the last thing that happens on every exit path.
exit 0
