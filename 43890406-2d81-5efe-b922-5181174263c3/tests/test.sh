#!/usr/bin/env bash
# The verifier entry point for slot OER-23. Harbor ships no result parser of its own, so
# this file is the whole contract.
#
# It terminates by writing the bound reward carrier, /logs/verifier/reward.txt, carrying one
# bare float on the closed interval [0.0, 1.0], higher better, never binary. The companion
# document /logs/verifier/score.json carries the machine-readable reason and the metric block,
# and every zero carries a reason there. An EXIT trap guarantees a reward is written on every
# exit path, so an aborted verifier still produces a reason rather than silence.
#
# LOCAL EXERCISE ONLY: /logs/verifier/ is a single shared host path and concurrent lanes
# collide on it, so OER23_REWARD_ROOT redirects the root when this script is exercised
# locally. The bound contract path inside the bundle stays /logs/verifier/reward.txt.
#
# THE CLOCK. Nothing on this path reads one. tests/harness_sim.py owns a virtual integer tick
# clock inside the serving loop and emits timing telemetry; tests/checkers.py reads those
# records. That is why a timing metric can be graded by pure, replayable checkers at all.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
REWARD_ROOT="${OER23_REWARD_ROOT:-/logs/verifier}"
REWARD_FILE="${REWARD_ROOT}/reward.txt"
SCORE_FILE="${REWARD_ROOT}/score.json"
SUBMISSION_DIR="${OER23_SUBMISSION_DIR:-/workspace/submission}"
mkdir -p "${REWARD_ROOT}"

write_score() {
  # bound contract path: /logs/verifier/reward.txt  companion: /logs/verifier/score.json
  mkdir -p "${REWARD_ROOT}"
  printf '%s\n' "$1" > "${REWARD_FILE}"
  printf '{"reward": %s, "reason": "%s", "slot": "OER-23", "reward_path": "/logs/verifier/reward.txt", "score_document_path": "/logs/verifier/score.json", "metric": {"anchors_state": "absent", "anchors_gap": "gap-oer-per-family-anchors-unmeasured", "baseline_metric": null, "target_metric": null}}\n' "$1" "$2" > "${SCORE_FILE}"
}

emit_reward() {
  # The trap body. If nothing above managed to write /logs/verifier/reward.txt, a reward is
  # still written here with a machine-readable reason, because an absent reward reads as an
  # infrastructure fault rather than as a score.
  if [ ! -f "${REWARD_FILE}" ]; then
    write_score 0.0 verifier-aborted
  fi
}
trap 'emit_reward' EXIT

# The compiled tests, tests/test_output.py, carry both halves of every graded checker over
# recorded fixtures. A verifier whose own checkers no longer refuse their planted defects has
# nothing to say about a submission.
if ! python3 "${HERE}/test_output.py" >"${REWARD_ROOT}/compiled.log" 2>&1; then
  write_score 0.0 compiled-tests-failed
  exit 0
fi

# tests/grade.py drives tests/runner.py, re-simulates every recorded configuration with
# tests/harness_sim.py, runs every checker declared in tests/checkers.yaml through
# tests/checkers.py, and writes the reward last.
if ! python3 "${HERE}/grade.py" \
      --bundle "${BUNDLE}" \
      --submission "${SUBMISSION_DIR}/search.py" \
      --reward-root "${REWARD_ROOT}" >"${REWARD_ROOT}/grade.log" 2>&1; then
  write_score 0.0 grader-failed
  exit 0
fi

exit 0
