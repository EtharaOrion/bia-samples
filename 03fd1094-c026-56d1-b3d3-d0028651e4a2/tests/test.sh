#!/usr/bin/env bash
# The verifier entry point Harbor runs. Harbor ships no result parser of its own, so
# this file is the whole contract: it terminates by writing the bound reward path.
#
# Bound reward contract path: /logs/verifier/reward.txt, the bare float.
# Companion score document:   /logs/verifier/score.json, reason and metric block.
#
# MEASURED CORRECTION, wave-2: /logs/verifier/ is a single shared host path and
# concurrent lanes collide on it. OER22_REWARD_ROOT redirects the root for a LOCAL
# exercise only. Unset, which is how Harbor runs this, the root is the bound one.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
REWARD_ROOT="${OER22_REWARD_ROOT:-/logs/verifier}"
WORKSPACE="${OER22_WORKSPACE:-${BUNDLE_DIR}}"

emit_reward() {
  status=$?
  REWARD_FILE="${REWARD_ROOT}/reward.txt"
  if [ -z "${OER22_REWARD_ROOT:-}" ]; then
    REWARD_FILE="/logs/verifier/reward.txt"
  fi
  mkdir -p "${REWARD_ROOT}"
  python3 "${TESTS_DIR}/grade.py" --emit --reward-root "${REWARD_ROOT}" --exit-status "${status}" || true
  if [ ! -s "${REWARD_FILE}" ]; then
    printf '0.000000\n' > "${REWARD_FILE}"
    printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "metric": {"graded_mean_degradation": null}}' > "${REWARD_ROOT}/score.json"
  fi
  exit "${status}"
}
trap emit_reward EXIT

mkdir -p "${REWARD_ROOT}"

# 1. Run the submission in isolation. runner.py never imports it: it copies the entry
#    point alone into a fresh temporary directory, launches it as a new session leader
#    under a small environment allowlist, and kills the whole process group afterwards.
python3 "${TESTS_DIR}/runner.py" \
  --workspace "${WORKSPACE}" \
  --report "${REWARD_ROOT}/runner.json" || true

# 2. Grade. Every graded number is recomputed here, inside the verifier, from its own
#    pristine substrate and the held-out evaluation payload. No number the submission
#    reported enters the graded path.
python3 "${TESTS_DIR}/grade.py" \
  --reward-root "${REWARD_ROOT}" \
  --workspace "${WORKSPACE}"

# 3. The compiled tests over the checker fixtures, both halves of every checker.
python3 "${TESTS_DIR}/test_output.py"

# 4. The reward write is the last thing that happens, and the EXIT trap above
#    guarantees it happens on every exit path including failure.
