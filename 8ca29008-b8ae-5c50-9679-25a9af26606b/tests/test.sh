#!/usr/bin/env bash
# S08 verifier entry point. Grades whatever the harness runner left behind. It never
# trains and it never writes telemetry.
set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export S08_BUNDLE="${S08_BUNDLE:-/workspace}"
export S08_TELEMETRY_DIR="${S08_TELEMETRY_DIR:-/telemetry}"
export S08_SUBMISSION="${S08_SUBMISSION:-/workspace/submission}"
export S08_FIXTURE_DIR="${S08_FIXTURE_DIR:-/opt/bia/s08/fixtures}"
export S08_PROFILE="${S08_PROFILE:-scaled}"
export LOG_DIR="${LOG_DIR:-/logs/verifier}"
export SCORE_PATH="${SCORE_PATH:-$LOG_DIR/score.json}"
export S08_OUTCOMES="${S08_OUTCOMES:-$LOG_DIR/outcomes.json}"

export REWARD_PATH="${REWARD_PATH:-/logs/verifier/reward.txt}"

mkdir -p "$LOG_DIR"

# Trapped on EXIT so no path out leaves the reward absent: an absent reward reads as
# an infrastructure fault, a zero reads as the score it is.
write_reward() {
  python3 - "$SCORE_PATH" "${REWARD_PATH:-/logs/verifier/reward.txt}" <<'PY'
import json, pathlib, sys
try:
    score = float(json.loads(pathlib.Path(sys.argv[1]).read_text()).get("score", 0.0))
except Exception:
    score = 0.0
if score != score:
    score = 0.0
out = pathlib.Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("%r\n" % min(max(score, 0.0), 1.0))
PY
}
trap write_reward EXIT

python3 "$TESTS_DIR/grade.py" 2>&1 | tee "$LOG_DIR/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

# A verifier that produces no number at all is a fault, and a fault is reported as a
# fault. It is never silently rewritten into a passing score.
if [ ! -s "$SCORE_PATH" ]; then
  printf '%s\n' '{"score":0.0,"reason":"verifier_produced_no_score"}' > "$SCORE_PATH"
fi

# Harbor reads a flat numeric object. Reduce to numeric fields only, preserving score.
python3 - "$SCORE_PATH" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
num = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
raw = d.get("score", 0.0)
try:
    raw = float(raw)
except (TypeError, ValueError):
    raw = 0.0
num["score"] = min(max(raw, 0.0), 1.0)
p.write_text(json.dumps(num, sort_keys=True))
PY

echo "--- score ---"
cat "$SCORE_PATH"
echo
echo "--- full verifier record ---"
cat "$LOG_DIR/grade-stdout.md"
echo

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$TESTS_DIR/test_output.py" "$TESTS_DIR/test_bundle_invariants.py" \
    -v --no-header -p no:cacheprovider 2>&1 | tee "$LOG_DIR/test-stdout.md"
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so tests/test_output.py did not run."
    echo "This is advisory only. The score comes from grade.py and nothing was asserted here."
  } | tee "$LOG_DIR/test-stdout.md"
fi

exit "$GRADE_RC"
