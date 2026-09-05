#!/usr/bin/env bash
# The verifier ENTRY POINT for slot OER-14. Harbor ships no result parser of its own, so
# this file is the whole contract.
#
# It terminates by writing the bound reward carrier: the bare float to
# /logs/verifier/reward.txt and the machine-readable reason and metric block to
# /logs/verifier/score.json. The reward is one float on the closed interval [0.0, 1.0],
# higher is better, and it is never binary.
#
# An EXIT trap guarantees a reward is written on every exit path, including failure, so an
# aborted verifier still produces a reason rather than silence.
#
# MEASURED CORRECTION, wave 2: /logs/verifier/ is a single shared host path and concurrent
# lanes collide on it. OER14_REWARD_FILE and OER14_SCORE_FILE redirect a LOCAL exercise to a
# lane-private root. The bound contract paths inside this bundle are the two above and only
# those; the override exists so an exercise never writes the shared root.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${TESTS}/.." && pwd)"

DEFAULT_REWARD_FILE="/logs/verifier/reward.txt"
DEFAULT_SCORE_FILE="/logs/verifier/score.json"
REWARD_FILE="${OER14_REWARD_FILE:-${DEFAULT_REWARD_FILE}}"
SCORE_FILE="${OER14_SCORE_FILE:-${DEFAULT_SCORE_FILE}}"
SUBMISSION="${SUBMISSION_PATH:-/workspace/submission.json}"

mkdir -p "$(dirname "${REWARD_FILE}")" "$(dirname "${SCORE_FILE}")"

finish() {
  status=$?
  # Every exit path terminates by writing the bound contract path /logs/verifier/reward.txt
  # with its reason document /logs/verifier/score.json beside it. grade.py owns the write, so
  # the instrument's binding and the reason carrier can never disagree.
  if [ ! -s "${REWARD_FILE}" ]; then
    python3 "${TESTS}/grade.py" --abort verifier-aborted-before-reward \
      --reward-file "${REWARD_FILE}" --score-file "${SCORE_FILE}" || true
  fi
  if [ ! -s "${REWARD_FILE}" ]; then
    printf '0.0\n' > "${REWARD_FILE}"
  fi
  return "${status}"
}
trap finish EXIT

# Self-check. tests/test_output.py drives the real checkers in tests/checkers.py over both
# halves of every declared row of tests/checkers.yaml. It runs only when pytest and the
# generated fixture plan are both present, because a shipped verifier image may carry
# neither, and a self-check that cannot run must not be able to fail the graded pass.
if python3 -c "import pytest" >/dev/null 2>&1 && [ -f "${BUNDLE}/solution/fixtures/plan.json" ]; then
  python3 -m pytest -q "${TESTS}/test_output.py" || true
fi

# The graded pass. tests/runner.py obtains the submission document without importing it,
# tests/harness.py computes every graded number inside this process from the frozen
# substrate, and tests/grade.py runs the gate chain and writes the reward LAST.
python3 "${TESTS}/grade.py" \
  --submission "${SUBMISSION}" \
  --reward-file "${REWARD_FILE}" \
  --score-file "${SCORE_FILE}"
