#!/usr/bin/env bash
# Private reference solution.
#
# Default path: install the reference update rule at the submission path and run
# the ordinary graded pipeline. The reference travels the exact route a
# submission travels, so a score it earns is a score the harness produced.
#
# BIA_SMOKE=1: the identical code path at tiny scale, CPU only, no CUDA
# allocation, over the frozen real-text fixture committed under
# environment/fixtures. The same three arms run, the same probe is live, the
# same ablation enforcement fires and the same seven checkers grade. It exists
# so the pipeline can be proven end to end without an accelerator, and it is
# never a substitute for the measured reference score.
set -uo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$SOLUTION_DIR")"
export BIA_BUNDLE="$BUNDLE"

if [ "${BIA_SMOKE:-0}" = "1" ]; then
  ROOT="$(mktemp -d -t bia_s10_smoke_XXXXXX)"
  echo "BIA_SMOKE=1  root=$ROOT"
  # No CUDA of any kind. The host accelerator is off limits to this path.
  export CUDA_VISIBLE_DEVICES=""
  # The host is shared, so thread counts are capped or the timings swing wildly.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
  export BIA_SMOKE_THREADS="${BIA_SMOKE_THREADS:-2}"
  export BIA_SUBSTRATE_DIR="$BUNDLE/environment/substrate"
  export BIA_CORPUS="$BUNDLE/environment/fixtures/smoke_corpus.bin"
  export BIA_SUBMISSION_DIR="$ROOT/submission"
  export BIA_WORKDIR="$ROOT/artifacts"
  export BIA_LOGDIR="$ROOT/logs"
  export BIA_ANCHOR_DIR="$ROOT/anchors"
else
  export BIA_SUBSTRATE_DIR="${BIA_SUBSTRATE_DIR:-/opt/bia/substrate}"
  export BIA_CORPUS="${BIA_CORPUS:-/opt/bia/data/enwik8}"
  export BIA_SUBMISSION_DIR="${BIA_SUBMISSION_DIR:-/workspace/submission}"
  export BIA_WORKDIR="${BIA_WORKDIR:-/workspace/artifacts}"
  export BIA_LOGDIR="${BIA_LOGDIR:-/logs/verifier}"
  export BIA_ANCHOR_DIR="${BIA_ANCHOR_DIR:-/anchors}"
fi

export SCORE_PATH="$BIA_LOGDIR/score.json"
export BIA_OUTCOMES="$BIA_LOGDIR/outcomes.json"
export BIA_RUN_RECORD="$BIA_LOGDIR/run_record.jsonl"
export BIA_EVIDENCE="$BIA_LOGDIR/evidence.json"

mkdir -p "$BIA_SUBMISSION_DIR" "$BIA_WORKDIR" "$BIA_LOGDIR" "$BIA_ANCHOR_DIR"
cp "$SOLUTION_DIR/reference_update_rule.py" "$BIA_SUBMISSION_DIR/update_rule.py"
echo "reference installed at $BIA_SUBMISSION_DIR/update_rule.py"

bash "$BUNDLE/tests/test.sh"
RC=$?

echo
echo "=== reference solution result ==="
cat "$SCORE_PATH" 2>/dev/null || echo "no score.json produced"
echo

# Both halves of every checker, over the evidence this run actually produced.
if [ -f "$BIA_EVIDENCE" ]; then
  echo "=== negative control table over this run's own evidence ==="
  python3 "$BUNDLE/tests/controls.py" --evidence "$BIA_EVIDENCE" \
    --json-out "$BIA_LOGDIR/controls.json"
  CTRL_RC=$?
  if [ "$CTRL_RC" -ne 0 ]; then
    echo "CONTROLS FAILED. A checker that cannot prove both halves is inert." >&2
    RC=$CTRL_RC
  fi
fi

exit "$RC"
