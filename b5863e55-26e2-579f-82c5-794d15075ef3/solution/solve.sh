#!/usr/bin/env bash
# Private reference solution for BIA slot S04.
#
# The oracle does exactly what a solver does and nothing more: it writes the six
# integers to the submission path and hands control to the verifier. It holds no
# chain key, it writes no telemetry, and it does not invoke the runner. The
# graded run belongs to tests/grade.py, which mints its own key and executes its
# own private copy of the runner.
#
# Default: runs the real graded path on one H100 at the scaled operating point.
# BIA_SMOKE=1: runs the identical code path at the smoke profile, CPU only, with
# CUDA hidden, so the whole chain from submission to score can be proven end to
# end without touching an accelerator. The smoke score is not a reference score
# and grade.py refuses to mix the two: a smoke record cannot be graded as full
# and a full record cannot be graded as smoke.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$HERE")"

if [ "${BIA_SMOKE:-0}" = "1" ]; then
  PROFILE="smoke"
  WORK="${BIA_WORK:-$(mktemp -d /tmp/bia-s04-smoke.XXXXXX)}"
  export CUDA_VISIBLE_DEVICES=""
  export BIA_FORCE_CPU=1
else
  PROFILE="full"
  WORK="${BIA_WORK:-/workspace}"
fi

LOGDIR="${BIA_LOGDIR:-$WORK/logs/verifier}"
mkdir -p "$WORK/submission" "$LOGDIR"

export BIA_LOGDIR="$LOGDIR"
export BIA_SUBMISSION="$WORK/submission/arch.json"
export SCORE_PATH="$LOGDIR/score.json"
export BIA_OUTCOMES="$LOGDIR/outcomes.json"
export BIA_VERIFIER_FROZEN_DIR="$BUNDLE/tests/frozen"
export BIA_AGENT_FROZEN_DIR="$BUNDLE/environment/frozen"

# The submission is one JSON file naming how the frozen parameter budget is
# allocated. This is the whole free surface of the task.
cp "$HERE/solution_arch.json" "$BIA_SUBMISSION"
echo "submitted allocation:"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d[sys.argv[2]], sort_keys=True))" \
  "$BIA_SUBMISSION" "$PROFILE"

echo "grading: the verifier owns the paired-arm run at profile $PROFILE"
if [ "$PROFILE" = "smoke" ]; then
  BIA_SMOKE=1 bash "$BUNDLE/tests/test.sh"
else
  bash "$BUNDLE/tests/test.sh"
fi
GRADE_RC=$?

echo "score file: $SCORE_PATH"
cat "$SCORE_PATH" 2>/dev/null || true
echo
exit "$GRADE_RC"
