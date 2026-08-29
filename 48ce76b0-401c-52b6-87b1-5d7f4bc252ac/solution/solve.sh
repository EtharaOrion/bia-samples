#!/usr/bin/env bash
# Reference solution for the S03 data-order task.
#
# Default path: the real graded reference. It installs solution/reference_order.py as the
# submission, runs the comparator ensemble and the submitted arm across every graded seed on
# one H100, and then invokes the real verifier entry point tests/test.sh to score what it
# recorded.
#
# BIA_SMOKE=1 runs the IDENTICAL code path at the CPU-only smoke scale. Same runner, same
# ordering policy, same telemetry writer, same socket hook, same checkers, same grader. Only
# the scale profile differs, and no CUDA context is created.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$HERE")"
export BIA_BUNDLE="$BUNDLE"

if [ "${BIA_SMOKE:-0}" = "1" ]; then
  export BIA_PROFILE=smoke
  export CUDA_VISIBLE_DEVICES=""
  # Bound the CPU thread pools. The smoke path exists to prove the code path executes, so it
  # takes a small fixed slice of the host rather than every core, and its wall clock stays
  # bounded on a machine that is doing other work.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export BIA_TORCH_THREADS="${BIA_TORCH_THREADS:-4}"
  echo "S03 solve.sh: SMOKE path, CPU only, no CUDA context is created."
else
  export BIA_PROFILE="${BIA_PROFILE:-graded}"
  echo "S03 solve.sh: GRADED path, one H100, scaled operating point."
fi

DEFAULT_WORK=/workspace
if [ ! -w "$DEFAULT_WORK" ] 2>/dev/null; then
  DEFAULT_WORK="${TMPDIR:-/tmp}/bia_s03_solve"
fi
export BIA_WORK="${BIA_WORK:-$DEFAULT_WORK}"
export BIA_TELEMETRY="${BIA_TELEMETRY:-$BIA_WORK/telemetry/run_record.jsonl}"
export BIA_EGRESS_AUDIT="${BIA_EGRESS_AUDIT:-$(dirname "$BIA_TELEMETRY")/egress_audit.jsonl}"
export BIA_LOG_DIR="${BIA_LOG_DIR:-$BIA_WORK/logs/verifier}"
export BIA_SCORE_PATH="${BIA_SCORE_PATH:-$BIA_LOG_DIR/score.json}"
export BIA_OUTCOMES_PATH="${BIA_OUTCOMES_PATH:-$BIA_LOG_DIR/outcomes.json}"
export BIA_CACHE_DIR="${BIA_CACHE_DIR:-$BIA_WORK/.bia_cache}"
export BIA_ORDER_MODULE="${BIA_ORDER_MODULE:-$BIA_WORK/submission/order.py}"
export BIA_REFERENCE_ORDER="$HERE/reference_order.py"

if [ -n "${BIA_RUNNER:-}" ] && [ -f "$BIA_RUNNER" ]; then
  RUNNER="$BIA_RUNNER"
elif [ -f "/opt/bia/runner/run_s03.py" ]; then
  RUNNER="/opt/bia/runner/run_s03.py"
else
  RUNNER="$BUNDLE/environment/runner/run_s03.py"
fi
export BIA_RUNNER="$RUNNER"

mkdir -p "$(dirname "$BIA_TELEMETRY")" "$BIA_LOG_DIR" "$BIA_CACHE_DIR" "$(dirname "$BIA_ORDER_MODULE")"
: > "$BIA_TELEMETRY"
: > "$BIA_EGRESS_AUDIT"

# The reference ordering policy IS the submission. It is copied rather than symlinked so the
# verifier's two independent derivations and its training children all load the same bytes.
cp "$BIA_REFERENCE_ORDER" "$BIA_ORDER_MODULE"

# The arms are no longer run here. The verifier runs the whole graded grid
# itself, in child interpreters it spawns, and measures every final validation
# loss in a process that holds no submitted byte. An oracle that produced the
# graded record would be producing the very artifact the previous revision let a
# forger produce, so this script now does exactly what an agent does: it puts an
# ordering policy on disk and hands over.

echo "S03 solve.sh: scoring through the real verifier entry point"
bash "$BUNDLE/tests/test.sh"
GRADE_RC=$?

echo "S03 solve.sh: reward contract path $BIA_SCORE_PATH"
cat "$BIA_SCORE_PATH"
echo
exit "$GRADE_RC"
