#!/usr/bin/env bash
# Verifier entry point for bia slot S07.
set -uo pipefail

BUNDLE="${BIA_BUNDLE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export BIA_BUNDLE
BIA_BUNDLE="$BUNDLE"
export BIA_ENVDIR="${BIA_ENVDIR:-$BUNDLE/environment}"
export BIA_ARTIFACTS="${BIA_ARTIFACTS:-/workspace/artifacts}"
export SCORE_PATH="${SCORE_PATH:-/logs/verifier/score.json}"
export BIA_OUTCOMES="${BIA_OUTCOMES:-/logs/verifier/outcomes.json}"
export BIA_REWARD_PATH="${BIA_REWARD_PATH:-/logs/verifier/reward.txt}"

mkdir -p "$(dirname "$SCORE_PATH")" "$(dirname "$BIA_OUTCOMES")"

# grade.py writes the reward on every path it reaches, but it cannot write one on a path that
# never reaches it: an import failure, a kill, or any exit before line 26 would leave the file
# absent, and an absent reward reads as an infrastructure fault rather than as a graded zero.
# This trap makes the floor the terminal state of every exit path, and it never overwrites a
# reward that was already written.
emit_reward_floor() {
  reward_path="${BIA_REWARD_PATH:-/logs/verifier/reward.txt}"
  mkdir -p "$(dirname "$reward_path")" 2>/dev/null
  if [ ! -s "$reward_path" ]; then
    printf '%s\n' '0.0' > "$reward_path" 2>/dev/null
  fi
}
trap emit_reward_floor EXIT

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$BIA_ARTIFACTS/telemetry.json" ]; then
  echo "=====================================================================" >&2
  echo "PREFLIGHT: no telemetry at $BIA_ARTIFACTS/telemetry.json" >&2
  echo "  The graded attempt did not run, or wrote its artifacts elsewhere." >&2
  echo "  grade.py will emit score 0.0 with reason telemetry_absent, which is" >&2
  echo "  a real graded outcome and not a harness fault." >&2
  echo "=====================================================================" >&2
fi

python3 "$TESTS_DIR/grade.py" 2>&1 | tee "$(dirname "$SCORE_PATH")/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

test -s "$SCORE_PATH" || printf '%s\n' '{"score":0.0,"reason":"verifier_produced_no_score"}' > "$SCORE_PATH"

python3 - "$SCORE_PATH" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
raw = float(d.get("score", 0.0))
p.write_text(json.dumps({"score": min(max(raw, 0.0), 1.0)}))
PY

echo "--- score ---"; cat "$SCORE_PATH"; echo

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$TESTS_DIR/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$(dirname "$SCORE_PATH")/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so tests/test_output.py did not run."
    echo "Nothing was asserted here. The score comes from grade.py and the outcome carrier at"
    echo "$BIA_OUTCOMES, both of which did run. Do not read this block as passing assertions."
  } | tee "$(dirname "$SCORE_PATH")/test-stdout.md"
fi

exit "$GRADE_RC"
