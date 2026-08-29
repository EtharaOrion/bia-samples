#!/usr/bin/env bash
# Harbor verifier entry point for bia slot S06.
# The verifier owns the measurement: grade.py runs the frozen harness itself against the
# submitted estimator and derives the reward only from the run record that harness wrote.
set -uo pipefail

BIA_BUNDLE="${BIA_BUNDLE:-/workspace}"
BIA_LOG_DIR="${BIA_LOG_DIR:-/logs/verifier}"
BIA_SUBMISSION="${BIA_SUBMISSION:-${BIA_BUNDLE}/submission/estimator.py}"
BIA_TESTS="${BIA_TESTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export BIA_BUNDLE BIA_LOG_DIR BIA_SUBMISSION
export BIA_SCORE_PATH="${BIA_SCORE_PATH:-${BIA_LOG_DIR}/score.json}"
export BIA_OUTCOMES_PATH="${BIA_OUTCOMES_PATH:-${BIA_LOG_DIR}/outcomes.json}"

# Byte code caches are redirected away from the bundle tree, whose canonical content
# hash covers every file under it.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/bia-s06-work/pycache}"

mkdir -p "${BIA_LOG_DIR}"

# Harbor reads one float from the bound reward path. grade.py writes the graded
# value there; this floor writer runs on every exit route, including the
# preflight return below, so no path out of this script leaves the reward absent.
write_reward_floor() {
  reward_dest="/logs/verifier/reward.txt"
  reward_dest="${BIA_REWARD_PATH:-$reward_dest}"
  mkdir -p "$(dirname "$reward_dest")" 2>/dev/null || true
  test -s "$reward_dest" || printf '%s' '0.0' > "$reward_dest"
}
trap write_reward_floor EXIT

export BIA_REWARD_PATH="${BIA_REWARD_PATH:-${BIA_LOG_DIR}/reward.txt}"

if [ ! -d "${BIA_TESTS}/private_bundle/environment/harness" ]; then
  echo "PREFLIGHT: the verifier's private harness copy is missing" >&2
  printf '%s\n' '{"score": 0.0, "reason": "preflight-fault-verifier-private-harness-absent", "detail": {}}' > "${BIA_SCORE_PATH}"
  printf '%s\n' '{}' > "${BIA_OUTCOMES_PATH}"
  cat "${BIA_SCORE_PATH}"
  exit 2
fi

if [ ! -d "${BIA_BUNDLE}/environment/harness" ] || [ ! -d "${BIA_BUNDLE}/environment/fixtures" ]; then
  echo "PREFLIGHT: the agent-visible harness and fixture trees are not reachable at ${BIA_BUNDLE}" >&2
  echo "  frozen_fixture_bytes_match_bound reads those trees as live state, so the" >&2
  echo "  verifier cannot decide it. CONFIG fault, not a solver failure." >&2
  printf '%s\n' '{"score": 0.0, "reason": "preflight-fault-agent-trees-absent", "detail": {}}' > "${BIA_SCORE_PATH}"
  printf '%s\n' '{}' > "${BIA_OUTCOMES_PATH}"
  cat "${BIA_SCORE_PATH}"
  exit 2
fi

if [ ! -f "${BIA_SUBMISSION}" ]; then
  echo "PREFLIGHT: no submission at ${BIA_SUBMISSION}" >&2
  printf '%s\n' '{"score": 0.0, "reason": "submission-absent", "detail": {}}' > "${BIA_SCORE_PATH}"
  printf '%s\n' '{}' > "${BIA_OUTCOMES_PATH}"
  cat "${BIA_SCORE_PATH}"
  exit 0
fi

python3 "${BIA_TESTS}/grade.py" 2>&1 | tee "${BIA_LOG_DIR}/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

if [ ! -s "${BIA_SCORE_PATH}" ]; then
  printf '%s\n' '{"score": 0.0, "reason": "verifier-produced-no-score", "detail": {}}' > "${BIA_SCORE_PATH}"
fi

echo "--- score ---"
cat "${BIA_SCORE_PATH}"
echo

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "${BIA_TESTS}/test_output.py" -q --no-header -p no:cacheprovider \
    2>&1 | tee "${BIA_LOG_DIR}/test-stdout.md" || true
else
  echo "SKIPPED: pytest unavailable, tests/test_output.py did not run. Advisory only; the reward comes from grade.py." \
    | tee "${BIA_LOG_DIR}/test-stdout.md"
fi

exit "${GRADE_RC}"
