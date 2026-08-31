#!/usr/bin/env bash
# OER-09 verifier entry point. Harbor ships no result parser of its own, so this file is the
# whole reward contract: it terminates by writing the bound carrier, and a trap guarantees a
# reward exists on every exit path, including an abort, so a failure is a reason and never
# silence.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "${TESTS}")"
SUBMISSION="${OER09_SUBMISSION:-${BUNDLE}/solution/solve.sh}"
REWARD_FILE="/logs/verifier/reward.txt"
SCORE_FILE="/logs/verifier/score.json"

emit_reward() {
  # Fail closed. If the graded write already happened, leave it alone. If it did not, the
  # verifier died before grading and that fact is written down as a reason rather than left
  # as an absent file, which downstream cannot tell apart from a harness that never started.
  if [ -s "/logs/verifier/reward.txt" ]; then
    return 0
  fi
  mkdir -p /logs/verifier
  printf '0.000000\n' > "/logs/verifier/reward.txt"
  printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "metric": {"quantity": "validation loss of the trained model", "direction": "lower", "graded_loss": null, "control_loss": null, "reference_loss": null, "baseline_metric": null, "target_metric": null, "anchors_state": "absent", "anchors_gap": "gap-oer-per-family-anchors-unmeasured"}}' > "/logs/verifier/score.json"
}
trap 'emit_reward' EXIT

mkdir -p /logs/verifier /logs/harness

echo "OER-09 verifier: egress denied, checkers declared in checkers.yaml"

# 1. The submission runs in isolation. runner.py copies it alone into a fresh temporary
#    directory, launches it as a new session leader under a small environment allowlist and
#    kills the whole process group in a finally block. Its exit code is recorded and never
#    propagated: a submission that failed is graded with a reason, not reported as a fault.
python3 "${TESTS}/runner.py" --entry "${SUBMISSION}" --record /logs/verifier/submission_run.json \
  --artifacts /logs/verifier/submission_artifacts --bundle "${BUNDLE}"

# 2. The measurement. harness.py recomputes every graded record from the frozen register and
#    the artifacts step 1 captured out of the submission's sandbox, and OVERWRITES
#    /logs/harness. The agent phase publishes an advisory copy of the pool observation there
#    so environment/pool_probe.py has something to print; this call replaces it, so a record
#    an agent planted can never reach a checker. Nothing here computes a verdict.
python3 "${TESTS}/harness.py" --artifacts /logs/verifier/submission_artifacts --logs /logs/harness

# 3. The compiled per-checker tests, derived from solution/grounding.yaml by recompute.py.
#    Both halves of every checker are exercised here against the pinned fixtures.
python3 "${TESTS}/test_output.py"

# 4. Grading. grade.py imports checkers.py and the reference and never imports the submission.
#    It reads the harness's own telemetry under /logs/harness and writes the bound carrier
#    "/logs/verifier/reward.txt" plus the reason document "/logs/verifier/score.json".
#    Trajectory rubrics in rubrics.jsonl are judged separately and never raise this number.
python3 "${TESTS}/grade.py"

test -s "${REWARD_FILE}"
test -s "${SCORE_FILE}"
echo "OER-09 verifier: reward written to ${REWARD_FILE}, reason in ${SCORE_FILE}"
