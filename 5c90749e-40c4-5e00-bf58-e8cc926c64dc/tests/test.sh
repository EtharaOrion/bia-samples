#!/usr/bin/env bash
# BIA S01 verifier entry point. Terminates by writing the bound reward contract
# path, and every zero it writes carries a machine-readable reason.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BIA_BUNDLE="${BIA_BUNDLE:-$(dirname "$HERE")}"
export BIA_WORKSPACE="${BIA_WORKSPACE:-/workspace}"
export BIA_SUBMISSION="${BIA_SUBMISSION:-$BIA_WORKSPACE/submission}"
export BIA_TELEMETRY_DIR="${BIA_TELEMETRY_DIR:-/telemetry}"
export BIA_TELEMETRY="${BIA_TELEMETRY:-$BIA_TELEMETRY_DIR/run_record.jsonl}"
export BIA_SCORE="${BIA_SCORE:-/logs/verifier/score.json}"
export BIA_OUTCOMES="${BIA_OUTCOMES:-/logs/verifier/outcomes.json}"
export BIA_REWARD="${BIA_REWARD:-/logs/verifier/reward.txt}"
export BIA_CACHE="${BIA_CACHE:-/tmp/bia_cache}"

# Fail closed on every exit path, including a crash inside tests/grade.py before
# it reaches emit(). The guard never overwrites a reward grade.py already wrote.
reward_floor() {
  REWARD_FILE="${BIA_REWARD:-/logs/verifier/reward.txt}"
  mkdir -p "$(dirname "$REWARD_FILE")"
  if [ ! -s "$REWARD_FILE" ]; then
    printf '0.0\n' > "$REWARD_FILE"
    echo "verifier-produced-no-reward"
  fi
}
trap reward_floor EXIT

mkdir -p "$(dirname "$BIA_SCORE")" "$(dirname "$BIA_OUTCOMES")" "$(dirname "$BIA_REWARD")"
GRADE_LOG="$(dirname "$BIA_SCORE")/grade-stdout.md"

python3 "$BIA_BUNDLE/tests/grade.py" 2>&1 | tee "$GRADE_LOG"
GRADE_RC=${PIPESTATUS[0]}

if [ ! -s "$BIA_SCORE" ]; then
  printf '%s\n' '{"score": 0.0, "reason": "verifier_produced_no_score", "detail": {}}' > "$BIA_SCORE"
fi

python3 - "$BIA_SCORE" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
doc = json.loads(path.read_text())
score = float(doc.get("score", 0.0))
score = min(max(score, 0.0), 1.0)
reason = str(doc.get("reason", "unattributed"))
if score == 0.0 and not reason:
    reason = "unattributed_zero"
path.write_text(json.dumps({"score": score, "reason": reason}, sort_keys=True))
print(json.dumps({"score": score, "reason": reason}, sort_keys=True))
PY

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$BIA_BUNDLE/tests/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$(dirname "$BIA_SCORE")/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so tests/test_output.py did not run."
    echo "This is advisory only. The score comes from tests/grade.py and nothing was asserted here."
  } | tee "$(dirname "$BIA_SCORE")/test-stdout.md"
fi

exit "$GRADE_RC"
