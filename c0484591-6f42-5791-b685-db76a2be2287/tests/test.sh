#!/usr/bin/env bash
# The whole result contract for OER-05. Harbor ships no result parser of its
# own, so this file is it.
#
# The bound reward carrier is /logs/verifier/reward.txt and it holds ONE BARE
# FLOAT in the closed interval [0.0, 1.0], higher better, never binary. The
# machine-readable reason and the metric block live beside it in
# /logs/verifier/score.json, and BOTH are written from the same trap, so the
# instrument's binding and the reason carrier can never disagree.
#
# The trap is the point. Under errexit any failing command ends this script, and
# a script that ends before writing the reward leaves the run UNSCORED, which
# reads downstream as an infrastructure fault rather than as a score. The EXIT
# trap therefore reaches the reward write on every exit path, including the
# failure paths, so an aborted verifier still produces a reason rather than
# silence.
set -euo pipefail

REWARD_FILE="/logs/verifier/reward.txt"
SCORE_FILE="/logs/verifier/score.json"
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TELEMETRY="${OER05_TELEMETRY:-/logs/harness/telemetry.json}"
STAGE="verifier-not-started"

emit_reward() {
  code="$?"
  mkdir -p /logs/verifier
  if [ ! -s /logs/verifier/reward.txt ]; then
    printf '0.0\n' > /logs/verifier/reward.txt
    printf '{"reward": 0.0, "reason": "verifier-aborted", "stage": "%s", "exit_code": %s, "metric": {"graded_step": null, "anchors_state": "absent", "anchors_gap": "gap-oer-per-family-anchors-unmeasured"}}\n' "${STAGE}" "${code}" > /logs/verifier/score.json
  fi
  exit "${code}"
}

trap emit_reward EXIT

mkdir -p /logs/verifier

# The compiled tests are the same twelve checkers, compiled one test per checker
# from solution/grounding.yaml by solution/recompute.py. They run first so a
# broken grading tree is loud before anything is graded.
STAGE="compiled-tests"
python3 "${TESTS_DIR}/test_output.py"

# grade.py imports the checkers and never the submission, and terminates by
# writing the bare float to the bound carrier and the reason beside it.
STAGE="grade"
python3 "${TESTS_DIR}/grade.py" "${TELEMETRY}" "${REWARD_FILE}" "${SCORE_FILE}"

STAGE="complete"
