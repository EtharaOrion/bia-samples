#!/usr/bin/env bash
# Harbor verifier entry point for the BIA-GSN-1 kernel throughput task.
# It computes no score itself. grade.py owns the score, and this script only projects
# that score onto the bound reward path /logs/verifier/reward.txt as a single float.
# Every zero either path can produce carries a machine readable reason in report.json.
set -uo pipefail

BIA_BUNDLE="${BIA_BUNDLE:-/}"
BIA_TESTS="${BIA_TESTS:-/tests}"
BIA_ENV_DIR="${BIA_ENV_DIR:-${BIA_BUNDLE%/}/environment}"
BIA_LOGS="${BIA_LOGS:-/logs/verifier}"
BIA_ARTIFACTS="${BIA_ARTIFACTS:-/workspace/artifacts}"
BIA_SUBMISSION="${BIA_SUBMISSION:-/workspace/submission/impl.py}"
SCORE_PATH="${SCORE_PATH:-$BIA_LOGS/score.json}"
BIA_REPORT="${BIA_REPORT:-$BIA_LOGS/report.json}"
BIA_OUTCOMES="${BIA_OUTCOMES:-$BIA_LOGS/outcomes.json}"
BIA_EVENTS="${BIA_EVENTS:-$BIA_LOGS/harness_events.jsonl}"
BIA_TIMING_SUMMARY="${BIA_TIMING_SUMMARY:-$BIA_LOGS/timing_summary.json}"
export BIA_BUNDLE BIA_TESTS BIA_ENV_DIR BIA_LOGS BIA_ARTIFACTS BIA_SUBMISSION
export SCORE_PATH BIA_REPORT BIA_OUTCOMES BIA_EVENTS BIA_TIMING_SUMMARY

mkdir -p "$BIA_LOGS" "$BIA_ARTIFACTS"

# The runtime mounts /logs/verifier/reward.txt and reads exactly one float from it.
# write_reward writes the floor first and only then upgrades it from the score
# grade.py owns, so a crash between the two leaves a scored zero rather than an
# absent reward, which would read as an infrastructure fault. The EXIT trap is what
# makes that true on every path out of this script, including the preflight fault
# that exits before grade.py runs at all.
write_reward() {
  status=$?
  mkdir -p /logs/verifier 2>/dev/null
  printf '%s\n' '0.0' > /logs/verifier/reward.txt 2>/dev/null
  python3 - "$SCORE_PATH" '/logs/verifier/reward.txt' 2>/dev/null <<'PY'
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

if [ ! -f "$BIA_SUBMISSION" ]; then
  echo "=====================================================================" >&2
  echo "PREFLIGHT FAULT: no submission at $BIA_SUBMISSION" >&2
  echo "  This is a CONFIG fault, not a solver failure. Do not report it as a" >&2
  echo "  score. Fix: place the submitted implementation at that path." >&2
  echo "=====================================================================" >&2
  printf '%s\n' '{"score":0.0}' > "$SCORE_PATH"
  printf '%s\n' '{"score":0.0,"reason":"preflight_fault_submission_absent","zero_reason":"preflight-fault-submission-absent"}' > "$BIA_REPORT"
  exit 2
fi

python3 "$BIA_TESTS/grade.py" 2>&1 | tee "$BIA_LOGS/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

test -s "$SCORE_PATH" || printf '%s\n' '{"score":0.0}' > "$SCORE_PATH"
python3 - "$SCORE_PATH" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
num = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
num.setdefault("score", float(d.get("score", 0.0)))
num["score"] = min(max(float(num["score"]), 0.0), 1.0)
p.write_text(json.dumps(num))
PY
cat "$SCORE_PATH"; echo

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$BIA_TESTS/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$BIA_LOGS/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so test_output.py did not run."
    echo "This is advisory only. The score comes from grade.py and its six checkers."
    echo "Nothing was asserted here. Do not read this as the compiled rubric passing."
  } | tee "$BIA_LOGS/test-stdout.md"
fi

exit "$GRADE_RC"
