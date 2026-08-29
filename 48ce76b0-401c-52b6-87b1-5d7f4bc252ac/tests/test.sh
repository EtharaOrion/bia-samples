#!/usr/bin/env bash
set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BIA_BUNDLE="${BIA_BUNDLE:-$(dirname "$TESTS_DIR")}"
export BIA_TELEMETRY="${BIA_TELEMETRY:-/telemetry/run_record.jsonl}"
export BIA_EGRESS_AUDIT="${BIA_EGRESS_AUDIT:-$(dirname "$BIA_TELEMETRY")/egress_audit.jsonl}"
export BIA_ORDER_MODULE="${BIA_ORDER_MODULE:-/workspace/submission/order.py}"
export BIA_LOG_DIR="${BIA_LOG_DIR:-/logs/verifier}"
export BIA_SCORE_PATH="${BIA_SCORE_PATH:-$BIA_LOG_DIR/score.json}"
export BIA_OUTCOMES_PATH="${BIA_OUTCOMES_PATH:-$BIA_LOG_DIR/outcomes.json}"
export BIA_REWARD_PATH="${BIA_REWARD_PATH:-$BIA_LOG_DIR/reward.txt}"

# Fail closed on every exit path, including a crash inside tests/grade.py before
# it reaches emit(). The guard never overwrites a reward grade.py already wrote.
reward_floor() {
  REWARD_FILE="${BIA_REWARD_PATH:-/logs/verifier/reward.txt}"
  mkdir -p "$(dirname "$REWARD_FILE")"
  if [ ! -s "$REWARD_FILE" ]; then
    printf '0.0\n' > "$REWARD_FILE"
    echo "verifier-produced-no-reward"
  fi
}
trap reward_floor EXIT

mkdir -p "$BIA_LOG_DIR"

python3 "$TESTS_DIR/grade.py" 2>&1 | tee "$BIA_LOG_DIR/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

if [ ! -s "$BIA_SCORE_PATH" ]; then
  printf '%s\n' '{"score": 0.0, "reason": "verifier_produced_no_score", "detail": {}}' > "$BIA_SCORE_PATH"
fi

python3 - "$BIA_SCORE_PATH" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
raw = d.get("score", 0.0)
try:
    score = float(raw)
except (TypeError, ValueError):
    score = 0.0
    d["reason"] = "score_field_not_numeric"
score = min(max(score, 0.0), 1.0)
reason = d.get("reason") or "unattributed"
p.write_text(json.dumps({"score": score, "reason": reason}, sort_keys=True))
print(json.dumps({"score": score, "reason": reason}, sort_keys=True))
PY

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$TESTS_DIR/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$BIA_LOG_DIR/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so tests/test_output.py did not run."
    echo "This is advisory only. The reward comes from tests/grade.py and nothing was asserted here."
  } | tee "$BIA_LOG_DIR/test-stdout.md"
fi

cat "$BIA_SCORE_PATH"
exit "$GRADE_RC"
