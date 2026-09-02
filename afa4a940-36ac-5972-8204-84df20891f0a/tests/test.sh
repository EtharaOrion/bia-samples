#!/usr/bin/env bash
# The whole result contract for slot OER-10. Harbor ships no result parser of its own,
# so this file is it. It terminates by writing the bound reward carrier, and a trap
# guarantees a reward exists on every exit path including failure, so an aborted
# verifier produces a machine-readable reason rather than silence.
#
#   /logs/verifier/reward.txt   the bare float, the carrier the live verifier binds
#   /logs/verifier/score.json   the reason and the metric block, written from the same call
#
# The reward is one float on the closed interval [0.0, 1.0], higher is better, never
# binary. Every zero carries a reason field a downstream grader can branch on.
set -euo pipefail

SUBMISSION="${OER10_SUBMISSION:-/workspace/submission/recipe.py}"
TELEMETRY="${OER10_TELEMETRY:-/logs/verifier/telemetry.json}"
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Registered before anything can fail. The body writes the bound carrier literally, so a
# reader can confirm from these bytes that no exit path leaves the reward absent.
emit_reward() {
  if [ ! -s /logs/verifier/reward.txt ]; then
    mkdir -p /logs/verifier
    printf '%s\n' "0.0" > /logs/verifier/reward.txt
    printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted", "metric": {"quantity": "validation loss of the trained model", "direction": "lower", "graded_loss": null, "anchors_state": "absent", "baseline_metric": null, "target_metric": null, "anchors_gap": "gap-oer-per-family-anchors-unmeasured"}}' > /logs/verifier/score.json
  fi
}
trap emit_reward EXIT

mkdir -p /logs/verifier

# 1. Isolation and observation. tests/runner.py copies the submission alone into a fresh
#    temporary directory, launches it as a new session leader under an environment
#    allowlist, captures its output, and kills the process group in a finally block. It
#    records the stage sequence, so the fingerprint screen's position relative to training
#    is a fact this process wrote rather than a claim the submission made.
python3 "${TESTS_DIR}/runner.py" "${SUBMISSION}" "${TELEMETRY}"

# 2. The gate chain. tests/grade.py imports tests/checkers.py and never imports the
#    submission. Every checker declared in tests/checkers.yaml is reached from here.
python3 "${TESTS_DIR}/grade.py" "${TELEMETRY}" /logs/verifier/reward.txt /logs/verifier/score.json

# 3. The compiled per-checker tests, both halves of every checker over frozen fixtures.
python3 "${TESTS_DIR}/test_output.py"

cat /logs/verifier/score.json
