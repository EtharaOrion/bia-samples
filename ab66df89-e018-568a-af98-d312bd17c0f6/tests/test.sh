#!/bin/bash
# Verifier entry point.
#
# The ONE invariant here: a reward float and a reward document exist at
# /logs/verifier when this script exits, on EVERY path, including the paths where
# the grader crashed, timed out, was killed by the OOM reaper, or never started.
# That is why the trap is installed before anything else can fail and why this
# script deliberately does not `set -e`: an early exit must reach the trap rather
# than terminate the shell with nothing written.
#
# The trap NEVER overwrites a reward the grader already wrote. It only fills in a
# refusal when the carriers are missing or empty.
#
# /logs/verifier is a DIRECTORY created in the image, never a VOLUME. An anonymous
# volume declared at that path shadows the /logs bind mount the runner supplies, and
# every carrier this script writes would land in a throwaway that is discarded with
# the container -- an emit path that executed correctly and still reported nothing.

REWARD_DIR="${OER16_LOGDIR:-/logs/verifier}"
GRADE_TIMEOUT="${OER16_GRADE_TIMEOUT:-1500}"
mkdir -p "$REWARD_DIR"

finalise() {
    local rc=$?
    if [ -s "$REWARD_DIR/reward.txt" ] && [ -s "$REWARD_DIR/reward.json" ]; then
        echo "[test.sh] reward already written by the grader: $(cat "$REWARD_DIR/reward.txt")"
        exit 0
    fi
    local reason="grading-chain-failed"
    if [ "$rc" -eq 124 ]; then
        reason="grading-timed-out"
    elif [ "$rc" -eq 137 ]; then
        reason="grading-killed"
    fi
    echo "[test.sh] no reward carrier from the grader (rc=$rc); refusing with $reason"
    printf '0.0\n' > "$REWARD_DIR/reward.txt"
    printf '{"reward": 0.0, "reason": "%s"}\n' "$reason" > "$REWARD_DIR/reward.json"
    if [ ! -s "$REWARD_DIR/score.json" ]; then
        printf '{"reward": 0.0, "reason": "%s", "detail": "the grading chain exited %s without writing a reward"}\n' \
            "$reason" "$rc" > "$REWARD_DIR/score.json"
    fi
    exit 0
}
trap finalise EXIT

echo "[test.sh] verifier starting at $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    || echo "[test.sh] nvidia-smi unavailable"

timeout --signal=TERM --kill-after=60 "$GRADE_TIMEOUT" \
    python3 /verifier/tests/grade.py
rc=$?
echo "[test.sh] grade.py exited $rc"
exit "$rc"
