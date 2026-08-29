#!/usr/bin/env bash
# Private oracle for bia slot S06, gradient estimator under injected noise.
#
# Default path: install the reference estimator as the submission and drive the real
# Harbor verifier entry point, which runs the frozen harness at the scaled operating
# point on one H100 and writes the reward record.
#
# Smoke path: BIA_SMOKE=1 runs the identical code path at the smoke operating point on
# CPU with an empty CUDA device list, so the oracle can be proven to execute end to end
# without touching an accelerator. Same solve.sh, same estimator, same harness functions,
# same verifier, same checkers, same reward formula. Only the frozen size changes.
set -uo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BUNDLE="$(cd "${SOLUTION_DIR}/.." && pwd)"
BIA_BUNDLE="${BIA_BUNDLE:-${DEFAULT_BUNDLE}}"

if [ -d "/workspace/environment/harness" ] && [ -z "${BIA_BUNDLE_EXPLICIT:-}" ]; then
  BIA_BUNDLE="/workspace"
fi

writable() { mkdir -p "$1" 2>/dev/null && [ -w "$1" ]; }

# Working state never lands in the bundle directory: the bundle's canonical content
# hash covers every file it holds, so writing into it changes the identity of the
# thing being graded. Harbor mounts the bundle at /workspace where submission/ is the
# declared artifact path; off Harbor this falls back to /submission then to a temp root.
BIA_WORK_ROOT="${BIA_WORK_ROOT:-${TMPDIR:-/tmp}/bia-s06-work}"
if [ -n "${BIA_SUBMISSION_DIR:-}" ]; then
  writable "${BIA_SUBMISSION_DIR}" || BIA_SUBMISSION_DIR=""
fi
if [ -z "${BIA_SUBMISSION_DIR:-}" ]; then
  if [ "${BIA_BUNDLE}" = "/workspace" ] && writable "/workspace/submission"; then
    BIA_SUBMISSION_DIR="/workspace/submission"
  elif writable "/submission"; then
    BIA_SUBMISSION_DIR="/submission"
  else
    BIA_SUBMISSION_DIR="${BIA_WORK_ROOT}/submission"
    mkdir -p "${BIA_SUBMISSION_DIR}"
  fi
fi

BIA_LOG_DIR="${BIA_LOG_DIR:-/logs/verifier}"
if ! writable "${BIA_LOG_DIR}"; then
  BIA_LOG_DIR="${BIA_WORK_ROOT}/logs/verifier"
  mkdir -p "${BIA_LOG_DIR}"
fi

# Byte code caches are redirected for the same reason.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${BIA_WORK_ROOT}/pycache}"

export BIA_BUNDLE BIA_LOG_DIR
export BIA_SUBMISSION="${BIA_SUBMISSION_DIR}/estimator.py"
export BIA_TESTS="${BIA_TESTS:-${BIA_BUNDLE}/tests}"
export BIA_FIXTURES="${BIA_BUNDLE}/environment/fixtures"
export BIA_SCORE_PATH="${BIA_LOG_DIR}/score.json"
export BIA_OUTCOMES_PATH="${BIA_LOG_DIR}/outcomes.json"

if [ "${BIA_SMOKE:-0}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=""
  export BIA_SMOKE=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  echo "[solve.sh] smoke path: CPU only, empty CUDA_VISIBLE_DEVICES, smoke operating point"
else
  echo "[solve.sh] reference path: scaled operating point, one H100"
fi

echo "[solve.sh] bundle=${BIA_BUNDLE}"
echo "[solve.sh] submission=${BIA_SUBMISSION}"
echo "[solve.sh] logs=${BIA_LOG_DIR}"

cp "${SOLUTION_DIR}/reference_estimator.py" "${BIA_SUBMISSION}"

bash "${BIA_TESTS}/test.sh"
RC=$?

echo "[solve.sh] verifier exit ${RC}"
if [ -s "${BIA_SCORE_PATH}" ]; then
  echo "[solve.sh] score record:"
  cat "${BIA_SCORE_PATH}"
  echo
fi
exit "${RC}"
