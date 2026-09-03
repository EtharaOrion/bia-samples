#!/usr/bin/env bash
# The one entry point Harbor runs for slot OER-19.
#
# Harbor ships no result parser of its own, so this file is the whole contract. It
# terminates by writing the bound reward carrier /logs/verifier/reward.txt, a bare
# float in the closed interval [0.0, 1.0] where higher is better and the reward is
# never binary, with the machine-readable reason and the metric block in the
# companion /logs/verifier/score.json.
#
# The write happens from an EXIT trap rather than from the end of the script. That
# is this slot's own archetype applied to its own grading path: a verifier that
# aborts silently and leaves no reward is a silent execution failure, and it is
# indistinguishable from a run that never started. Every exit path through this
# file therefore leaves a float and a reason behind.
#
# FORGE_REWARD_ROOT redirects both carriers to a lane-private directory for a local
# exercise, because /logs/verifier is a single shared host path that concurrent
# lanes collide on. The bound contract path is unchanged by that redirection.
set -euo pipefail

BOUND_REWARD_CONTRACT="/logs/verifier/reward.txt"
BOUND_SCORE_DOCUMENT="/logs/verifier/score.json"

REWARD_ROOT="${FORGE_REWARD_ROOT:-}"
if [ -n "$REWARD_ROOT" ]; then
  REWARD_FILE="$REWARD_ROOT/reward.txt"
  SCORE_FILE="$REWARD_ROOT/score.json"
else
  REWARD_FILE="$BOUND_REWARD_CONTRACT"
  SCORE_FILE="$BOUND_SCORE_DOCUMENT"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERDICT="$(mktemp -t oer19-verdict-XXXXXX)"

REWARD_VALUE="0.0"
REWARD_REASON="verifier-aborted-before-grading"
AGENT_METRIC="null"
GRADED_POINT="null"

emit_reward() {
  # Bound carrier: /logs/verifier/reward.txt with the bare float, and
  # /logs/verifier/score.json with the reason and the metric block. Written last,
  # written on every exit path, and never left absent.
  mkdir -p "$(dirname "$REWARD_FILE")" 2>/dev/null || true
  printf '%s' "$REWARD_VALUE" > "$REWARD_FILE" || true
  printf '{"reward": %s, "reason": "%s", "metric": {"agent_metric": %s, "baseline_metric": null, "target_metric": null, "anchors_state": "absent", "gap": "gap-oer-per-family-anchors-unmeasured", "graded_point": %s, "direction": "higher-is-better"}}\n' "$REWARD_VALUE" "$REWARD_REASON" "$AGENT_METRIC" "$GRADED_POINT" > "$SCORE_FILE" || true
  rm -f "$VERDICT" 2>/dev/null || true
}
trap emit_reward EXIT

SUBMISSION="${OER19_SUBMISSION:-${1:-}}"
if [ -z "$SUBMISSION" ]; then
  SUBMISSION="$HERE/../solution/reference.py"
fi

set +e
python3 "$HERE/grade.py" --submission "$SUBMISSION" --emit "$VERDICT" ${OER19_ANCHOR_ARGS:-}
GRADE_STATUS=$?
set -e

if [ "$GRADE_STATUS" -ne 0 ] || [ ! -s "$VERDICT" ]; then
  REWARD_VALUE="0.0"
  REWARD_REASON="grading-process-failed"
  exit 0
fi

REWARD_VALUE="$(python3 -c 'import json,sys; print(float(json.load(open(sys.argv[1]))["reward"]))' "$VERDICT")"
REWARD_REASON="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["reason"])' "$VERDICT")"
AGENT_METRIC="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["metric"]["agent_metric"]))' "$VERDICT")"
GRADED_POINT="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["metric"]["graded_point"]))' "$VERDICT")"

exit 0
