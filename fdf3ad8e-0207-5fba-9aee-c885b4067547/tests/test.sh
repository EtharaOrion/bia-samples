#!/usr/bin/env bash
# OER-07 verifier entry point. Harbor ships no result parser of its own, so this
# file is the whole reward contract.
#
# It terminates by writing the bound carrier /logs/verifier/reward.txt, a bare
# float in [0.0, 1.0], higher better, never binary, and the companion reason and
# metric document /logs/verifier/score.json. An EXIT trap guarantees that an
# aborted verifier still produces a reason rather than silence.
#
# Egress is denied on this surface. The solving agent's surface had it open;
# the two differ by design and the difference is bound in task.toml.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${TESTS_DIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

mkdir -p /logs/verifier

# Every exit path lands here. If tests/grade.py already wrote the carrier this
# leaves it alone; otherwise it writes the floor with a machine-readable reason,
# because a missing reward reads as an infrastructure fault rather than a score.
emit_fallback_reward() {
  if [ ! -s /logs/verifier/reward.txt ]; then
    printf '0.000000\n' > /logs/verifier/reward.txt
    printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "metric": {"graded_step": null, "baseline": null, "target": null}, "slot": "OER-07"}' > /logs/verifier/score.json
  fi
}
trap emit_fallback_reward EXIT

# 1. Run the submission in isolation. tests/runner.py copies it alone into a
#    fresh temporary directory, launches it as a new session leader under an
#    environment allowlist, and kills the whole process group in a finally block.
#    The grading process never imports the submission.
python3 "${TESTS_DIR}/runner.py" || true

# 2. The compiled tests, one per declared checker in tests/checkers.yaml. They
#    are advisory here; the reward comes from the reduction below.
python3 -m pytest -q "${TESTS_DIR}/test_output.py" > /logs/verifier/pytest.log 2>&1 || true

# 3. The reduction. tests/grade.py imports tests/checkers.py, walks the declared
#    set, applies the required_pass gate, computes the continuous reward, and
#    writes /logs/verifier/score.json and then /logs/verifier/reward.txt last of all.
#    tests/rubrics.jsonl is judged against the trajectory outside this process
#    and never raises a reward.
python3 "${TESTS_DIR}/grade.py" /logs/verifier
