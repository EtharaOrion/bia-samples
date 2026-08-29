#!/usr/bin/env bash
# Verifier entry point. Harbor runs this.
#
# It does not train anything itself and it does not hold a shared secret. grade.py
# owns the graded run: it mints a chain key for that invocation alone, executes the
# verifier's own private copy of the runner in a directory the solving container
# never sees, and grades only the telemetry that run produced. Nothing the agent
# wrote is an input to the reward, so there is no key to publish and no preflight
# key fault to raise.
set -uo pipefail

LOGDIR="${BIA_LOGDIR:-/logs/verifier}"
mkdir -p "$LOGDIR"

# Harbor reads one float from the bound reward path. grade.py writes the graded
# value there; this floor writer runs on every exit route, including the
# preflight fault below, so no path out of this script leaves the reward absent.
write_reward_floor() {
  reward_dest="/logs/verifier/reward.txt"
  reward_dest="${BIA_REWARD_PATH:-$reward_dest}"
  mkdir -p "$(dirname "$reward_dest")" 2>/dev/null || true
  test -s "$reward_dest" || printf '%s' '0.0' > "$reward_dest"
}
trap write_reward_floor EXIT

export BIA_REWARD_PATH="${BIA_REWARD_PATH:-$LOGDIR/reward.txt}"
export SCORE_PATH="${SCORE_PATH:-$LOGDIR/score.json}"
export BIA_OUTCOMES="${BIA_OUTCOMES:-$LOGDIR/outcomes.json}"
export BIA_SUBMISSION="${BIA_SUBMISSION:-/workspace/submission/arch.json}"
export BIA_SCORE_DETAIL="$LOGDIR/score_detail.json"
export BIA_RECORD_EVIDENCE="${BIA_RECORD_EVIDENCE:-$LOGDIR/run_record.jsonl}"
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(dirname "$TESTS_DIR")"
export BIA_VERIFIER_FROZEN_DIR="${BIA_VERIFIER_FROZEN_DIR:-$TESTS_DIR/frozen}"
export BIA_AGENT_FROZEN_DIR="${BIA_AGENT_FROZEN_DIR:-${BIA_FROZEN_DIR:-$BUNDLE_DIR/environment/frozen}}"

if [ ! -d "$BIA_AGENT_FROZEN_DIR" ]; then
  echo "=====================================================================" >&2
  echo "PREFLIGHT FAULT: the agent-visible frozen tree is not reachable at"    >&2
  echo "  $BIA_AGENT_FROZEN_DIR"                                              >&2
  echo "  no_frozen_recipe_tree_edit reads that tree as live state, so the"   >&2
  echo "  verifier cannot decide it. This is a CONFIG fault, not a solver"    >&2
  echo "  failure. Do not report it as a score. Fix: mount the environment"   >&2
  echo "  tree into the verifier, or set BIA_AGENT_FROZEN_DIR."               >&2
  echo "=====================================================================" >&2
  printf '%s\n' '{"score":0.0,"reason":"preflight-fault-agent-frozen-tree-absent"}' > "$LOGDIR/grade-stdout.md"
  exit 2
fi

if [ ! -f "$TESTS_DIR/runner/run_arch.py" ]; then
  echo "PREFLIGHT FAULT: the verifier's private runner is missing" >&2
  printf '%s\n' '{"score":0.0,"reason":"preflight-fault-verifier-runner-absent"}' > "$LOGDIR/grade-stdout.md"
  exit 2
fi

python3 "$TESTS_DIR/grade.py" 2>&1 | tee "$LOGDIR/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

test -s "$SCORE_PATH" || printf '%s\n' '{"score":0.0,"reason":"verifier-produced-no-score"}' > "$SCORE_PATH"
cp "$SCORE_PATH" "$LOGDIR/score_detail.json"

python3 - "$SCORE_PATH" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
num = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
num.setdefault("score", float(d.get("score", 0.0)))
p.write_text(json.dumps(num))
PY

echo; echo "--- score ---"; cat "$SCORE_PATH"; echo
echo "--- reason and detail retained at $LOGDIR/score_detail.json ---"

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$TESTS_DIR/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$LOGDIR/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so tests/test_output.py did not run."
    echo "This is advisory only. The score comes from grade.py and nothing was asserted here."
  } | tee "$LOGDIR/test-stdout.md"
fi

exit "$GRADE_RC"
