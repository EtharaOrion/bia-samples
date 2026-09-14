#!/usr/bin/env bash
# The verifier entry point. Harbor ships no result parser of its own, so this file is
# the whole reward contract.
#
# It terminates by writing /logs/verifier/reward.txt: one float on the closed interval
# [0.0, 1.0], higher better, never binary. An EXIT trap guarantees a reward exists on
# every exit path, including an abort, so a failed verifier produces a machine-readable
# reason instead of silence. An absent reward reads as an infrastructure fault rather
# than as a score, and that ambiguity is what the trap closes.
#
# Files this entry point reaches, named so reachability is a fact about bytes:
#   tests/runner.py         re-executes the submission out of process
#   tests/held_out_eval.py  the verifier's own unsmoothed held-out evaluation
#   tests/checkers.py       the nine pure checkers
#   tests/checkers.yaml     their declared reductions and zero reasons
#   tests/grade.py          the gate chain and the continuous reward
#   tests/test_output.py    the compiled per-checker tests, generated from grounding
#   tests/rubrics.jsonl     the trajectory-judged rubrics, bucket N

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${OER04_WORKSPACE:-/tmp/oer04-verifier}"
SUBMISSION="${OER04_SUBMISSION:-/app/submission.py}"
CLAIM="${OER04_CLAIM:-/app/claim.json}"

emit_floor_reward() {
    mkdir -p /logs/verifier
    if [ ! -s /logs/verifier/reward.txt ]; then
        printf '0.000000\n' > /logs/verifier/reward.txt
        printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "significance_state": "significance-not-applicable", "outcome_classification": "failed", "metric": {"graded_step": null, "baseline": 3250, "target": 2690, "separation": null, "separation_margin": 0.05}}' > /logs/verifier/reward.json
    fi
}

trap emit_floor_reward EXIT

mkdir -p /logs/verifier "${WORKSPACE}"

export PYTHONPATH="${TESTS_DIR}:${PYTHONPATH:-}"

# Re-execute the submission on seeds selected after submission, and write the one
# telemetry record the checkers read. A failure here leaves the telemetry absent, and
# grade.py scores that 0.0 with reason verifier-telemetry-absent rather than crashing.
python3 "${TESTS_DIR}/runner.py" \
    --submission "${SUBMISSION}" \
    --claim "${CLAIM}" \
    --workspace "${WORKSPACE}" \
    --telemetry /logs/verifier/verifier_telemetry.json || true

# The compiled per-checker tests run against their own embedded fixtures, so a checker
# that stopped discriminating is caught before it grades anything live.
python3 "${TESTS_DIR}/test_output.py" || true

# Grade. This writes score.json, then reward.json, then the single float LAST.
python3 "${TESTS_DIR}/grade.py" \
    --telemetry /logs/verifier/verifier_telemetry.json \
    --reward-path /logs/verifier/reward.txt \
    --reward-document /logs/verifier/reward.json \
    --score-document /logs/verifier/score.json

emit_floor_reward
