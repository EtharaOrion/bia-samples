#!/usr/bin/env bash
# BIA S02 private oracle. Runs the reference update rule through the real runner
# and grades the result with the real verifier.
#
# Two profiles, one code path:
#
#   default        the graded operating point. One H100, bfloat16, two seeds of
#                  1400 steps at 24576 tokens per step, bounded by the runner's
#                  420 second wall-clock guard so one attempt cannot exceed the
#                  per-attempt budget of 0.12 hours.
#
#   BIA_SMOKE=1    the same runner, the same reference rule, the same frozen
#                  schedule and the same grader, at a
#                  tiny scale on CPU with CUDA disabled. It exists so this
#                  oracle can be proven to execute end to end without an
#                  accelerator. Its score is a path proof and is never a
#                  difficulty measurement and never a reference score for the
#                  graded operating point.
set -uo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="full"
if [ "${BIA_SMOKE:-0}" = "1" ]; then
  PROFILE="smoke"
  export CUDA_VISIBLE_DEVICES=""
  export BIA_SMOKE=1
fi

RUN_ROOT="${BIA_RUN_ROOT:-$(mktemp -d /tmp/bia-s02-XXXXXX)}"
# The workspace defaults outside the bundle so an oracle run never writes a byte
# into the frozen bundle it is proving. A Harbor run supplies /workspace instead.
WORKSPACE="${BIA_WORKSPACE:-$RUN_ROOT/workspace}"
export BIA_TELEMETRY_DIR="${BIA_TELEMETRY_DIR:-$RUN_ROOT/telemetry}"
export BIA_LOG_DIR="${BIA_LOG_DIR:-$RUN_ROOT/logs/verifier}"
export BIA_SUBMISSION="${BIA_SUBMISSION:-$WORKSPACE/submission}"
export BIA_WORKSPACE="$WORKSPACE"
export BIA_SCHEDULE_PATH="${BIA_SCHEDULE_PATH:-$BUNDLE/environment/frozen_schedule.py}"
export BIA_DATA_DIR="${BIA_DATA_DIR:-/data/fineweb}"
mkdir -p "$BIA_TELEMETRY_DIR" "$BIA_LOG_DIR" "$BIA_SUBMISSION"

# There is no chain key any more, and no secret of any kind is shared with the
# solving side. The graded record is written by the verifier's own supervisor
# from a training process it spawns, so the record the score rests on is never
# reachable by the party being graded and needs no signature to defend it. The
# run below is the agent-side run: it produces the reported logs the
# reconciliation checker reads, and it produces nothing the reward is computed
# from.

echo "solve.sh: profile=$PROFILE bundle=$BUNDLE workspace=$WORKSPACE run_root=$RUN_ROOT"

cp "$BUNDLE/solution/reference_update_rule.py" "$BIA_SUBMISSION/update_rule.py"

rm -f "$BIA_TELEMETRY_DIR/run_record.jsonl"
rm -f "$BIA_SUBMISSION/logs/${PROFILE}_seed"*.log

STARTED=$(date +%s)
for SEED in 0 1; do
  echo "solve.sh: running seed $SEED at profile $PROFILE"
  python3 "$BUNDLE/environment/runner/run_bia_s02.py" \
    --update-rule "$BIA_SUBMISSION/update_rule.py" \
    --seed "$SEED" \
    --profile "$PROFILE" \
    --out-dir "$BIA_SUBMISSION" \
    --data-dir "$BIA_DATA_DIR" \
    --fixture-dir "$BUNDLE/environment/fixtures"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "solve.sh: runner failed for seed $SEED with exit code $RC" >&2
    exit "$RC"
  fi
done
ELAPSED=$(( $(date +%s) - STARTED ))
echo "solve.sh: both seeds finished in ${ELAPSED}s"

BIA_TELEMETRY="$BIA_TELEMETRY_DIR/run_record.jsonl" \
  bash "$BUNDLE/tests/test.sh"
GRADE_RC=$?

echo "solve.sh: verifier exit code $GRADE_RC"
echo "solve.sh: reward file $BIA_LOG_DIR/score.json"
cat "$BIA_LOG_DIR/score.json"
echo
if [ "$PROFILE" = "smoke" ]; then
  echo "solve.sh: this was the smoke path. The number above proves the oracle path"
  echo "solve.sh: executes end to end on CPU. It is not the reference score at the"
  echo "solve.sh: graded operating point, which is measured on one idle H100."
fi
exit "$GRADE_RC"
