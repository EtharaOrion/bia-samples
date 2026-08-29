#!/usr/bin/env bash
# The one entry point Harbor runs. Every graded reduction is reached from here.
#
# Order of operations, and why it is this order:
#   1. the trap is registered BEFORE anything can fail, so no exit path is silent
#   2. runner.py executes the submission out of process and emits telemetry
#   3. grade.py reduces that telemetry through tests/checkers.py and tests/checkers.yaml
#   4. test_output.py re-asserts every compiled test over the same telemetry
#   5. the trap fires last and writes the bound reward carrier
#
# The reward carrier is bound to /logs/verifier/reward.txt and the reason and
# metric block to /logs/verifier/score.json. OER20_REWARD_ROOT redirects BOTH to a
# lane-private root for local exercise only, because /logs/verifier is one shared
# host path and concurrent lanes collide on it. The bound contract path is
# unchanged by that redirect.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${TEST_DIR}/.." && pwd)"
ENV_DIR="${BUNDLE_DIR}/environment"
SUBMISSION_DIR="${OER20_SUBMISSION:-${BUNDLE_DIR}/solution}"
WORK_DIR="${OER20_WORK:-$(mktemp -d -t oer20-verify-XXXXXX)}"
mkdir -p "${WORK_DIR}"

emit_reward() {
  rc=$?
  BOUND_REWARD_PATH="/logs/verifier/reward.txt"
  BOUND_SCORE_PATH="/logs/verifier/score.json"
  if [ -n "${OER20_REWARD_ROOT:-}" ]; then
    REWARD_FILE="${OER20_REWARD_ROOT}/reward.txt"
    SCORE_FILE="${OER20_REWARD_ROOT}/score.json"
  else
    REWARD_FILE="${BOUND_REWARD_PATH}"
    SCORE_FILE="${BOUND_SCORE_PATH}"
  fi
  python3 "${TEST_DIR}/emit_reward.py" \
    --verdict "${WORK_DIR}/verdict.json" \
    --reward "${REWARD_FILE}" \
    --score "${SCORE_FILE}" \
    --exit-code "${rc}" || true
  exit "${rc}"
}
trap 'emit_reward' EXIT

python3 "${TEST_DIR}/runner.py" \
  --submission "${SUBMISSION_DIR}" \
  --environment "${ENV_DIR}" \
  --workdir "${WORK_DIR}/run" \
  --out "${WORK_DIR}/telemetry.json"

# solution/ is private oracle material and is deliberately NOT mounted on the verifier
# surface, so this path is absent on the graded path BY DESIGN. Do not repair that by
# copying solution/ into the verifier image: that leaks the reference onto the grading
# surface. grade.py takes only REFERENCE_SHA256 from it, as a verdict annotation nothing
# reads back, and its --reference default is already the empty string.
REFERENCE_PATH="${BUNDLE_DIR}/solution/reference.py"
if [ ! -f "${REFERENCE_PATH}" ]; then
  REFERENCE_PATH=""
fi

python3 "${TEST_DIR}/grade.py" \
  --telemetry "${WORK_DIR}/telemetry.json" \
  --out "${WORK_DIR}/verdict.json" \
  --reference "${REFERENCE_PATH}"

python3 "${TEST_DIR}/test_output.py" \
  --telemetry "${WORK_DIR}/telemetry.json" \
  --verdict "${WORK_DIR}/verdict.json"
