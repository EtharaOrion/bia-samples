#!/usr/bin/env bash
# OER-13 verifier entry point. Harbor runs this file and nothing else.
#
# Harbor ships no result parser of its own, so this file is the whole reward
# contract. It terminates by writing the bound reward carrier, which is the bare
# float at /logs/verifier/reward.txt, with the machine-readable reason and the
# metric block in the companion /logs/verifier/score.json.
#
# Every exit path writes a reward. The EXIT trap is registered before anything
# that can fail, so an aborted verifier produces an attributed zero rather than
# silence, and silence is never scored.
#
# OER13_REWARD_ROOT redirects the reward root for a local exercise, because
# /logs/verifier is one shared host path and concurrent lanes collide on it. The
# bound contract path is unchanged by that redirection.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"

REWARD_ROOT="${OER13_REWARD_ROOT:-/logs/verifier}"
REWARD_FILE="${REWARD_ROOT}/reward.txt"
SCORE_FILE="${REWARD_ROOT}/score.json"

emit_reward() {
    status="$?"
    DEFAULT_REWARD_PATH="/logs/verifier/reward.txt"
    if [ -z "${REWARD_FILE:-}" ]; then
        REWARD_FILE="${DEFAULT_REWARD_PATH}"
        REWARD_ROOT="$(dirname "${DEFAULT_REWARD_PATH}")"
        SCORE_FILE="${REWARD_ROOT}/score.json"
    fi
    mkdir -p "${REWARD_ROOT}" 2>/dev/null || true
    if [ ! -s "${REWARD_FILE}" ]; then
        printf '0.000000' > "${REWARD_FILE}"
        printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "metric": {"graded_validation_loss": null, "anchors_state": "absent", "anchors_gap": "gap-oer-per-family-anchors-unmeasured", "baseline_metric": null, "target_metric": null, "direction": "lower"}}' > "${SCORE_FILE}"
    fi
    exit "${status}"
}
trap emit_reward EXIT

main() {
    mkdir -p "${REWARD_ROOT}"
    # The compiled surface runs first. A manifest whose bindings have drifted away
    # from the checkers must abort before anything is graded, so the trap writes an
    # attributed zero rather than a number resting on a stale binding.
    python3 "${HERE}/test_output.py"
    # THE GRADED INPUT IS THE SUBMISSION IN THE SHARED WORKSPACE.
    # tests/grade.py:210 defaults OER13_SUBMISSION to ${BUNDLE}/environment, which
    # is the verifier's own pristine copy of the DELIVERED baseline. Left unset,
    # every run therefore measured the delivered tree against itself, scored
    # no-end-to-end-improvement, and returned 0.0 no matter what had been
    # submitted. The submission tree is what a submission is graded on, so it is
    # named here.
    #
    # It is passed only when it is actually present. An absent workspace is not
    # silently forgiven: grade.py then measures the delivered baseline as before
    # and the run scores no-end-to-end-improvement, which is the correct verdict
    # for a run that submitted nothing. Nothing here relaxes a checker, and the
    # in-run floor is still materialized by the verifier from its own copy of the
    # reference.
    SUBMISSION_DIR="${OER13_SUBMISSION:-/workspace/submission}"
    if [ ! -d "${SUBMISSION_DIR}" ]; then
        SUBMISSION_DIR="${BUNDLE}/environment"
    fi
    OER13_BUNDLE="${BUNDLE}" \
    OER13_SUBMISSION="${SUBMISSION_DIR}" \
    OER13_REWARD_ROOT="${REWARD_ROOT}" \
        python3 "${HERE}/grade.py"
}

main
