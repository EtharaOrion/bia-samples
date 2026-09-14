#!/usr/bin/env bash
# The verifier entry point for OER-16. Harbor ships no result parser of its own,
# so this file is the whole reward contract.
#
# It terminates by writing the bound carrier /logs/verifier/reward.txt, one bare
# float on the closed interval [0.0, 1.0], higher is better, never binary. The
# machine-readable reason and the metric block go to /logs/verifier/score.json.
# An EXIT trap guarantees a reward on EVERY exit path, so an aborted verifier
# produces a reason rather than silence.
#
# The chain this reaches: tests/grade.py, which imports tests/checkers.py,
# tests/harness.py and tests/runner.py, grades against tests/config.json, and
# declares its surface in tests/checkers.yaml. tests/test_output.py is the
# compiled half of that manifest and runs first. tests/rubrics.jsonl is judged
# against the trajectory by a separate bucket and is never read here.
#
# MEASURED CORRECTION, wave 2: /logs/verifier/ is one shared host path and
# concurrent lanes collide on it. OER16_REWARD_ROOT redirects the ROOT for a
# lane-private local exercise only. The bound contract path that ships is
# /logs/verifier/reward.txt and it is what this file writes by default.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_ROOT="${OER16_REWARD_ROOT:-/logs/verifier}"
PYTHON="${OER16_PYTHON:-python3}"

mkdir -p "${REWARD_ROOT}"

emit_zero() {
  reason="$1"
  mkdir -p "${REWARD_ROOT}"
  printf '{"slot":"OER-16","reward":0.0,"reason":"%s","metric":{"graded_bits_per_byte":null,"claimed_bits_per_byte":null,"claim_absent":true,"denominator_bytes":null,"spent_token_updates":null,"points_completed":null,"baseline_metric":null,"target_metric":null,"anchors_state":"absent"},"gate":{"mode":"required_pass","value":0.0,"reason":"%s"}}\n' "${reason}" "${reason}" > "${REWARD_ROOT}/score.json"
  printf '0.000000\n' > "${REWARD_ROOT}/reward.txt"
}

emit_fallback() {
  code=$?
  # The bound reward contract path is /logs/verifier/reward.txt, carrying the
  # bare float, with the reason and metric block in /logs/verifier/score.json.
  # If the scoring chain never got that far, this writes a zero WITH a reason,
  # because an absent reward reads as an infrastructure fault rather than a score.
  if [ ! -f "${REWARD_ROOT}/reward.txt" ]; then
    emit_zero "verifier-aborted-before-scoring"
  fi
  exit "${code}"
}

trap 'emit_fallback' EXIT

# The compiled half of tests/checkers.yaml. Both halves of every declared
# checker, plus the four stale controls. A grading surface that cannot refuse
# its own planted defects is not a grading surface, so this runs before any
# submission is touched and a failure here is a scored zero with a reason.
if ! "${PYTHON}" "${TESTS_DIR}/test_output.py"; then
  emit_zero "compiled-checker-tests-failed"
  exit 1
fi

# Grade. grade.py runs the submission ALONE through tests/runner.py, measures in
# its own process through tests/harness.py, grades that telemetry with
# tests/checkers.py against tests/config.json, and writes both carriers LAST.
if ! "${PYTHON}" "${TESTS_DIR}/grade.py"; then
  emit_zero "grading-chain-failed"
  exit 1
fi

if [ ! -f "${REWARD_ROOT}/reward.txt" ]; then
  emit_zero "reward-carrier-unwritten"
  exit 1
fi

exit 0
