#!/usr/bin/env bash
# Verifier entry point for bia S09 multi-objective-frontier.
#
# Every path is overridable so this same script grades inside the pinned image
# and inside a scratch directory on a bare host. The defaults are the in-image
# paths.
set -uo pipefail

export BIA_S09_BUNDLE="${BIA_S09_BUNDLE:-/workspace}"
export BIA_S09_TESTS="${BIA_S09_TESTS:-/tests}"
export BIA_S09_SPEC="${BIA_S09_SPEC:-$BIA_S09_BUNDLE/harness/frontier_spec.json}"
export BIA_S09_TELEMETRY_DIR="${BIA_S09_TELEMETRY_DIR:-/telemetry}"
export BIA_S09_LOGS="${BIA_S09_LOGS:-/logs/verifier}"
export BIA_S09_SCORE="${BIA_S09_SCORE:-$BIA_S09_LOGS/score.json}"
export BIA_S09_OUTCOMES="${BIA_S09_OUTCOMES:-$BIA_S09_LOGS/outcomes.json}"
export BIA_S09_REWARD="${BIA_S09_REWARD:-$BIA_S09_LOGS/reward.txt}"

mkdir -p "$BIA_S09_LOGS"

# grade.py writes the reward on every path it reaches, but the preflight fault below exits
# before grade.py runs at all, and an import failure or a kill would do the same. An absent
# reward reads as an infrastructure fault rather than as a graded zero, so this trap makes the
# floor the terminal state of every exit path. It never overwrites a reward already written.
emit_reward_floor() {
  reward_path="${BIA_S09_REWARD:-/logs/verifier/reward.txt}"
  mkdir -p "$(dirname "$reward_path")" 2>/dev/null
  if [ ! -s "$reward_path" ]; then
    printf '%s\n' '0.0' > "$reward_path" 2>/dev/null
  fi
}
trap emit_reward_floor EXIT

if [ ! -f "$BIA_S09_SPEC" ]; then
  echo "PREFLIGHT FAULT: frozen spec not found at $BIA_S09_SPEC" >&2
  echo "  This is a CONFIG fault, not a solver failure. Do not report it as a score." >&2
  printf '%s\n' '{"score":0.0,"reason":"preflight_fault_spec_absent"}' > "$BIA_S09_SCORE"
  exit 2
fi

python3 "$BIA_S09_TESTS/grade.py" 2>&1 | tee "$BIA_S09_LOGS/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

test -s "$BIA_S09_SCORE" || printf '%s\n' '{"score":0.0,"reason":"verifier_produced_no_score"}' > "$BIA_S09_SCORE"

python3 - "$BIA_S09_SCORE" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
num = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
num.setdefault("score", float(d.get("score", 0.0)))
p.write_text(json.dumps(num))
PY

echo "--- score ---"
cat "$BIA_S09_SCORE"; echo

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$BIA_S09_TESTS/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$BIA_S09_LOGS/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so test_output.py did not run."
    echo "This is advisory only. The score comes from grade.py and nothing was asserted here."
  } | tee "$BIA_S09_LOGS/test-stdout.md"
fi

exit "$GRADE_RC"
