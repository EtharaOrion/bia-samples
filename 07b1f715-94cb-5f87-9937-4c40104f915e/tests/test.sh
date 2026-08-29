#!/usr/bin/env bash
# BIA S02 verifier entry point. Terminates by writing the bound reward path
# /logs/verifier/reward.txt, alongside the fuller record at
# /logs/verifier/score.json, and attributes every zero to a machine-readable
# reason rather than to an unwritten or empty reward file.
set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIA_WORKSPACE="${BIA_WORKSPACE:-/workspace}"
LOG_DIR="${BIA_LOG_DIR:-/logs/verifier}"
mkdir -p "$LOG_DIR"

export BIA_BUNDLE="${BIA_BUNDLE:-$BIA_WORKSPACE}"
export BIA_TELEMETRY_DIR="${BIA_TELEMETRY_DIR:-/telemetry}"
export BIA_TELEMETRY="${BIA_TELEMETRY:-$BIA_TELEMETRY_DIR/run_record.jsonl}"
export BIA_SUBMISSION="${BIA_SUBMISSION:-$BIA_WORKSPACE/submission}"
export BIA_CORPUS_DIR="${BIA_CORPUS_DIR:-$TESTS_DIR/corpus}"
export BIA_SCHEDULE_PATH="${BIA_SCHEDULE_PATH:-$BIA_WORKSPACE/environment/frozen_schedule.py}"
export BIA_SCORE_PATH="${BIA_SCORE_PATH:-$LOG_DIR/score.json}"
export BIA_OUTCOMES="${BIA_OUTCOMES:-$LOG_DIR/outcomes.json}"

# The runtime mounts /logs/verifier/reward.txt and reads exactly one float from
# it. write_reward writes the floor first and only then upgrades it from the
# graded score, so a crash between the two leaves a scored zero rather than an
# absent reward, which would read as an infrastructure fault. The EXIT trap
# below is what makes that true on every exit path, including the preflight
# fault that exits before grade.py ever runs.
write_reward() {
  status=$?
  mkdir -p /logs/verifier 2>/dev/null
  printf '%s\n' '0.0' > /logs/verifier/reward.txt 2>/dev/null
  python3 - "$BIA_SCORE_PATH" '/logs/verifier/reward.txt' 2>/dev/null <<'PY'
import json
import pathlib
import sys

score_path, reward_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
value = 0.0
try:
    value = float(json.loads(score_path.read_text()).get("score", 0.0))
except Exception:
    value = 0.0
if value != value:
    value = 0.0
reward_path.write_text("%.6f\n" % min(max(value, 0.0), 1.0))
PY
  return $status
}
trap write_reward EXIT

python3 "$TESTS_DIR/grade.py" 2>&1 | tee "$LOG_DIR/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

test -s "$BIA_SCORE_PATH" || printf '%s\n' '{"score": 0.0, "reason": "verifier_produced_no_score"}' > "$BIA_SCORE_PATH"

python3 - "$BIA_SCORE_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
raw = float(data.get("score", 0.0))
payload = {"score": min(max(raw, 0.0), 1.0)}
path.write_text(json.dumps(payload))
PY

cat "$BIA_SCORE_PATH"; echo
echo "--- full verifier record ---"
cat "$LOG_DIR/grade-stdout.md"

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$TESTS_DIR/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$LOG_DIR/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so tests/test_output.py did not run."
    echo "This is advisory only and does not affect the score, which comes from grade.py."
    echo "Nothing was asserted here, so do not read this as the compiled rubric passing."
  } | tee "$LOG_DIR/test-stdout.md"
fi

exit "$GRADE_RC"
