#!/usr/bin/env bash
# Harbor verifier entry point for slot OER-01.
#
# Harbor ships no result parser, so this file is the whole reward contract. Two
# bound artifacts are written, from ONE exit trap, on EVERY exit path:
#
#   /logs/verifier/reward.txt    one float in [0.0, 1.0], higher better, never binary
#   /logs/verifier/reward.json   {"reward": <float>, "reason": <kebab>, "metric": {...}}
#
# The two paths exist because two authorities name the reward contract and they
# disagree on the extension. Declared as gap-oer01-reward-path-extension-collides
# in solution/grounding.yaml; honoured here by writing both rather than by picking
# one. The trap is registered before anything can fail, so an aborted verifier
# still produces an attributed zero rather than silence.
set -euo pipefail
cd "$(dirname "$0")"

BIA_LOGS="${BIA_LOGS:-/logs/verifier}"
SCORE_PATH="${SCORE_PATH:-$BIA_LOGS/score.json}"
BIA_REPORT="${BIA_REPORT:-$BIA_LOGS/report.json}"
BIA_SUBMISSION="${BIA_SUBMISSION:-/workspace/submission/recipe.py}"
BIA_TELEMETRY="${BIA_TELEMETRY:-$BIA_LOGS/telemetry.json}"
export BIA_LOGS SCORE_PATH BIA_REPORT BIA_SUBMISSION BIA_TELEMETRY
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$BIA_LOGS"

write_reward() {
  REWARD_TXT="/logs/verifier/reward.txt"
  REWARD_DOC="/logs/verifier/reward.json"
  if [ "$BIA_LOGS" != "/logs/verifier" ]; then
    REWARD_TXT="$BIA_LOGS/reward.txt"
    REWARD_DOC="$BIA_LOGS/reward.json"
  fi
  printf '0.0\n' > "$REWARD_TXT" 2>/dev/null || true
  python3 - "$SCORE_PATH" "$REWARD_TXT" "$REWARD_DOC" <<'PY' 2>/dev/null || true
import json, sys
score_path, txt_path, doc_path = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {}
try:
    payload = json.load(open(score_path))
except Exception:
    payload = {}
try:
    value = float(payload.get("reward"))
except (TypeError, ValueError):
    value = 0.0
if value != value:
    value = 0.0
value = min(max(value, 0.0), 1.0)
reason = str(payload.get("reason") or "")
if value <= 0.0 and not reason:
    reason = "verifier-aborted-before-scoring"
metric = payload.get("metric") or {"graded_step": None, "baseline": 3250, "target": 2690}
open(txt_path, "w").write("%.6f\n" % value)
open(doc_path, "w").write(json.dumps({"reward": value, "reason": reason, "metric": metric}, sort_keys=True) + "\n")
PY
}
trap write_reward EXIT

echo "declared graded surface: checkers.yaml"

GRADE_RC=0
python3 grade.py || GRADE_RC=$?

if [ ! -s "$SCORE_PATH" ]; then
  echo "GRADER FAULT: no score document written" >&2
  printf '{"reward": 0.0, "reason": "grader-internal-error"}\n' > "$SCORE_PATH"
  GRADE_RC=1
fi

python3 -m pytest test_output.py -q --no-header -p no:cacheprovider || true

cat "$SCORE_PATH"
exit "$GRADE_RC"
