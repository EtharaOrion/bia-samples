#!/usr/bin/env bash
# The one entry point Harbor runs. It terminates by writing the bound reward
# carrier, and an EXIT trap guarantees that write happens on every exit path,
# including an abort, so an interrupted verifier produces an attributed zero
# rather than silence.
#
# Reward carrier: /logs/verifier/reward.txt, one bare float in [0.0, 1.0],
# higher is better, never binary. Reason and metric block: /logs/verifier/score.json,
# mirrored to the contract primary carrier /logs/verifier/reward.json.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_PATH="${OER06_STAGE:-/tmp/oer06-stage.json}"
SUBMISSION_PATH="${OER06_SUBMISSION:-/app/submission.py}"
SESSION_LOG="${OER06_SESSION_LOG:-/logs/verifier/session.jsonl}"
REWARD_FILE="/logs/verifier/reward.txt"
SCORE_FILE="/logs/verifier/score.json"

rm -f "$STAGE_PATH"

emit_reward() {
    status=$?
    python3 "$TESTS_DIR/emit_reward.py" "$STAGE_PATH" "$status" "$REWARD_FILE" "$SCORE_FILE"
    # Phase 1 item 2b and item 8 fix the reward output at /logs/verifier/reward.json
    # first, with reward.txt as the fallback. Both are written, from this one trap, so
    # the primary and the fallback carrier can never disagree on any exit path.
    if [ -s "$SCORE_FILE" ]; then
        cp "$SCORE_FILE" /logs/verifier/reward.json
    fi
    return $status
}

trap emit_reward EXIT

python3 "$TESTS_DIR/test_output.py"

python3 "$TESTS_DIR/grade.py" \
    --submission "$SUBMISSION_PATH" \
    --stage "$STAGE_PATH" \
    --session-log "$SESSION_LOG"
