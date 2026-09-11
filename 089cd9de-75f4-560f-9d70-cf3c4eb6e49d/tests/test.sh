#!/usr/bin/env bash
#
# OER-15 verifier entry point. Harbor ships no result parser of its own, so this
# file is the whole contract: it terminates by writing the bound reward carrier.
#
# Chain: runner.py isolates the submission and runs the frozen harness in the
# verifier's own process; grade.py drives tests/checkers.py through the gate order
# tests/checkers.yaml declares and writes the reward. tests/rubrics.jsonl is judged
# against the trajectory by a separate bucket-N pass and never by this script.
# tests/test_output.py is the compiled per-checker suite generated from
# solution/grounding.yaml and is exercised by seed/tasks/OER-15/adequacy.py.
#
set -euo pipefail

# The bound reward contract path. One bare float on the closed interval [0.0, 1.0],
# higher is better, never binary. The companion score document at score.json in the
# same directory carries the machine-readable reason for every zero.
BOUND_REWARD_PATH="/logs/verifier/reward.txt"

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELIVERY_ROOT="${OER15_DELIVERY_ROOT:-$(cd "${TESTS_DIR}/.." && pwd)}"
SUBMISSION_DIR="${OER15_SUBMISSION:-/workspace/submission}"

# MEASURED CORRECTION, wave 2: /logs/verifier is one shared host path and concurrent
# lanes collide on it. OER15_REWARD_ROOT redirects the exercise to a lane-private
# root. Unset, it resolves to the directory of the bound contract path above, so the
# delivered behaviour is the bound one.
REWARD_ROOT="${OER15_REWARD_ROOT:-$(dirname "${BOUND_REWARD_PATH}")}"
RECORD="${OER15_RECORD:-${REWARD_ROOT}/run_record.json}"
GRADED=0

emit_reward() {
  status=$?
  # Guarantees a reward exists on EVERY exit path, including an abort before the
  # chain ran. An absent reward reads as an infrastructure fault rather than a
  # score, so silence is never allowed to stand in for a zero. Writes the same two
  # carriers grade.py writes: the bare float and the reason document.
  if [ "${GRADED}" -eq 0 ]; then
    mkdir -p "${REWARD_ROOT}"
    printf '%s\n' '0.0' > "${REWARD_ROOT}/reward.txt"
    printf '{"reward": 0.0, "reason": "verifier-aborted", "detail": "the verifier exited at status %s before the gate chain wrote a reward", "metric": {"bits_per_byte": null, "denominator_bytes": 5739, "denominator_source": "frozen-eval-corpus-bytes", "baseline_metric": null, "target_metric": null, "anchors_state": "absent", "anchors_gap": "gap-oer-per-family-anchors-unmeasured"}}\n' "${status}" > "${REWARD_ROOT}/score.json"
  fi
  exit "${status}"
}
trap emit_reward EXIT

mkdir -p "${REWARD_ROOT}"

# The submission runs here and nowhere else, as a separate program under a new
# session leader. Its non-zero exit is a graded outcome, not a verifier failure, so
# the chain continues and grade.py attributes it.
set +e
python3 "${TESTS_DIR}/runner.py" \
  --delivery-root "${DELIVERY_ROOT}" \
  --submission "${SUBMISSION_DIR}" \
  --out "${RECORD}"
set -e

# Gate chain and reward write. grade.py exits non-zero whenever it wrote a zero, so
# its status is read rather than trusted to be success.
set +e
python3 "${TESTS_DIR}/grade.py" \
  --record "${RECORD}" \
  --manifest "${TESTS_DIR}/checkers.yaml" \
  --reward-root "${REWARD_ROOT}" \
  --delivery-root "${DELIVERY_ROOT}"
set -e

if [ -f "${REWARD_ROOT}/reward.txt" ] && [ -f "${REWARD_ROOT}/score.json" ]; then
  GRADED=1
fi

exit 0
