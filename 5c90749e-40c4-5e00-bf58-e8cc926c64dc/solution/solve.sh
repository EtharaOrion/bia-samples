#!/usr/bin/env bash
# GENERATED FILE. DO NOT HAND-EDIT. Source: solution/grounding.yaml via solution/recompute.py.
#
# BIA S01 private oracle. It runs the real reference path by default and the
# identical call graph at tiny CPU-only scale when BIA_SMOKE=1.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BIA_BUNDLE="${BIA_BUNDLE:-$(dirname "$HERE")}"
export BIA_WORKSPACE="${BIA_WORKSPACE:-/workspace}"

if [ "${BIA_SMOKE:-0}" = "1" ]; then
  export BIA_SCALE=smoke
  export BIA_DEVICE=cpu
  export CUDA_VISIBLE_DEVICES=""
  ROOT="${BIA_SMOKE_ROOT:-$(mktemp -d /tmp/bia-s01-smoke.XXXXXX)}"
  export BIA_WORKSPACE="$ROOT"
  export BIA_SUBMISSION="$ROOT/submission"
  export BIA_TELEMETRY_DIR="$ROOT/telemetry"
  export BIA_CACHE="$ROOT/cache"
  export BIA_SCORE="$ROOT/logs/verifier/score.json"
  export BIA_OUTCOMES="$ROOT/logs/verifier/outcomes.json"
  echo "BIA_SMOKE=1: CPU only, no CUDA, scale=smoke, root=$ROOT"
else
  export BIA_SCALE="${BIA_SCALE:-full}"
  export BIA_DEVICE="${BIA_DEVICE:-auto}"
  export BIA_SUBMISSION="${BIA_SUBMISSION:-$BIA_WORKSPACE/submission}"
  export BIA_TELEMETRY_DIR="${BIA_TELEMETRY_DIR:-/telemetry}"
  export BIA_CACHE="${BIA_CACHE:-/tmp/bia_cache}"
  export BIA_SCORE="${BIA_SCORE:-/logs/verifier/score.json}"
  export BIA_OUTCOMES="${BIA_OUTCOMES:-/logs/verifier/outcomes.json}"
fi

mkdir -p "$BIA_SUBMISSION" "$BIA_TELEMETRY_DIR" "$(dirname "$BIA_SCORE")"
cp "$HERE/reference_optimizer.py" "$BIA_SUBMISSION/optimizer.py"

python3 "$BIA_BUNDLE/environment/runner/run_bia.py" \
  --scale "$BIA_SCALE" --device "$BIA_DEVICE" --mode full --seeds 0,1 \
  --submission "$BIA_SUBMISSION" --telemetry-dir "$BIA_TELEMETRY_DIR" \
  --cache-dir "$BIA_CACHE"
RUN_RC=$?
if [ "$RUN_RC" -ne 0 ]; then
  echo "reference run failed with exit code $RUN_RC" >&2
  exit "$RUN_RC"
fi

python3 "$BIA_BUNDLE/tests/grade.py"
GRADE_RC=$?
echo "--- score ---"
cat "$BIA_SCORE"
echo
exit "$GRADE_RC"
