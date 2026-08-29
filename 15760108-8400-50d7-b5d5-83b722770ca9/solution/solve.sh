#!/usr/bin/env bash
# Private reference solution for bia S09 multi-objective-frontier.
#
# Default path: the real graded path. Loads environment/frontier_spec.json,
# uses the accelerator the frozen spec permits, runs the frozen runner over the
# reference submission, then runs the verifier and prints the score it earned.
#
# BIA_SMOKE=1: the identical code path at tiny scale, CPU only, no CUDA. It
# loads environment/frontier_spec_smoke.json instead, and it exercises the same
# engine functions, the same runner, the same grader and the same checkers. It
# is not a stub: it trains six real models, measures both real objectives on
# each, builds a real Pareto front and computes a real hypervolume from it.
#
# The smoke path exists so this reference can be proven to execute end to end
# without touching a GPU. It is a proof of executability, never a proof of the
# graded operating point, and the score it prints is a smoke-scale number that
# must never be reported as the reference score.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "$HERE/.." && pwd)"

WORK="${BIA_S09_WORK:-$(mktemp -d /tmp/bia_s09_solve.XXXXXX)}"
mkdir -p "$WORK/submission" "$WORK/telemetry" "$WORK/logs"

if [ "${BIA_SMOKE:-0}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=""
  export BIA_S09_FORCE_CPU=1
  SPEC="${BIA_S09_SPEC:-$BUNDLE/environment/frontier_spec_smoke.json}"
  echo "MODE: smoke. CPU only, CUDA hidden, tiny frozen spec. Identical code path."
else
  SPEC="${BIA_S09_SPEC:-$BUNDLE/environment/frontier_spec.json}"
  echo "MODE: reference. Graded frozen spec on the declared compute envelope."
fi

export BIA_S09_SPEC="$SPEC"
export BIA_S09_SUBMISSION="$WORK/submission/frontier.py"
export BIA_S09_TELEMETRY_DIR="$WORK/telemetry"
export BIA_S09_LOGS="$WORK/logs"
export BIA_S09_SCORE="$WORK/logs/score.json"
export BIA_S09_OUTCOMES="$WORK/logs/outcomes.json"
export BIA_S09_TESTS="$BUNDLE/tests"
export BIA_S09_BUNDLE="$BUNDLE"
# The verifier runs the frozen runner itself now, so it needs the runner location. It is a
# location and never a value: no number reaches the verifier through this export.
export BIA_S09_RUNNER="$BUNDLE/environment/harness/run_frontier.py"

cp "$HERE/reference_frontier.py" "$BIA_S09_SUBMISSION"

echo "spec        $BIA_S09_SPEC"
echo "workdir     $WORK"
echo
echo "=== frontier run ==="
python3 "$BUNDLE/environment/harness/run_frontier.py"
RUN_RC=$?
if [ "$RUN_RC" -ne 0 ]; then
  echo "frontier run failed with rc=$RUN_RC" >&2
  exit "$RUN_RC"
fi

echo
echo "=== verifier ==="
bash "$BUNDLE/tests/test.sh"
TEST_RC=$?

echo
echo "=== reward recomputation from raw achieved points ==="
# Recompute against the record the VERIFIER authored, not the one the agent-side run left
# behind, because that is the record the reward was read off.
GRADED_RECORD="$(python3 - "$BIA_S09_OUTCOMES" "$BIA_S09_TELEMETRY_DIR/frontier_record.json" <<'PY'
import json, pathlib, sys
try:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text())
    p = pathlib.Path(d.get("_graded_record", ""))
    print(str(p) if p.is_file() else sys.argv[2])
except Exception:
    print(sys.argv[2])
PY
)"
python3 "$HERE/recompute.py" --reward "$GRADED_RECORD" --score "$BIA_S09_SCORE"
RECOMP_RC=$?

echo
if [ "${BIA_SMOKE:-0}" = "1" ]; then
  echo "SMOKE RESULT: the score above is a smoke-scale number and is NOT the reference score."
  echo "The reference score at the graded operating point is unmeasured."
else
  echo "REFERENCE RESULT: the score above was earned at the graded operating point."
fi
echo "verifier rc=$TEST_RC  recompute rc=$RECOMP_RC"
exit $(( TEST_RC != 0 ? TEST_RC : RECOMP_RC ))
