#!/usr/bin/env bash
# The whole result contract. Harbor ships no result parser of its own, so this file
# terminates by writing the bound reward carrier and nothing downstream infers a score
# from an exit code, a log line, or silence.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION="${OER17_SUBMISSION:-/workspace/submission/generator.py}"

# MEASURED CORRECTION, wave-2: /logs/verifier is one shared host path and concurrent
# lanes collide on it. The bound contract path inside this bundle stays
# /logs/verifier/reward.txt; a local exercise redirects the ROOT through this variable
# and records that it did so in seed/tasks/OER-17/feasibility.yaml.
REWARD_ROOT="${OER17_REWARD_ROOT:-/logs/verifier}"

emit_reward() {
  # The bound reward contract path is /logs/verifier/reward.txt carrying one bare float
  # on the closed interval [0.0, 1.0], higher better, never binary. The companion
  # score.json carries the machine-readable reason and the metric block. This fallback
  # fires only when the grading path exited before writing them, so an aborted verifier
  # still produces an attributed zero rather than silence.
  if [ ! -f "${REWARD_ROOT}/reward.txt" ]; then
    mkdir -p "${REWARD_ROOT}"
    printf '0.000000\n' > "${REWARD_ROOT}/reward.txt"
    printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted", "metric": {"agent_metric": null, "graded_step": null, "baseline_metric": null, "target_metric": null, "anchors_state": "absent"}}' > "${REWARD_ROOT}/score.json"
  fi
}
trap 'emit_reward' EXIT

mkdir -p "${REWARD_ROOT}"

# Self-check first: the compiled tests drive every checker against frozen fixtures, so a
# checker that stopped discriminating is caught before a submission is ever graded.
if ! python3 "${TESTS_DIR}/test_output.py" > "${REWARD_ROOT}/self_check.log" 2>&1; then
  printf '0.000000\n' > "${REWARD_ROOT}/reward.txt"
  printf '%s\n' '{"reward": 0.0, "reason": "checker-self-check-failed", "metric": {"agent_metric": null, "graded_step": null, "baseline_metric": null, "target_metric": null, "anchors_state": "absent"}}' > "${REWARD_ROOT}/score.json"
  exit 0
fi

# The graded run. tests/grade.py isolates and runs the submission through tests/runner.py,
# trains the frozen model through tests/harness.py, evaluates the held-out benchmark in
# the verifier's own process, walks the required_pass gate chain from tests/checkers.yaml
# over tests/checkers.py, and writes the bare float and the score document last.
python3 "${TESTS_DIR}/grade.py" "${SUBMISSION}"

exit 0
