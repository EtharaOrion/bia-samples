#!/usr/bin/env bash
# The verifier entry point. Harbor ships no result parser of its own, so this file is the
# whole reward contract.
#
# It terminates by writing the bound reward carrier. A trap guarantees that a reward and a
# machine-readable reason exist on EVERY exit path, including an abort, so an aborted
# verifier produces a reason rather than silence.
#
# Reached from here: grade.py, which imports checkers.py, runner.py and bound.json, and the
# compiled per-checker suite tests/test_output.py when the private fixtures are mounted.
#
# Local exercise note: /logs/verifier/ is one shared host path and concurrent lanes collide
# on it, so FORGE_REWARD_ROOT redirects the write to a lane-private root. The BOUND contract
# path inside this bundle stays /logs/verifier/reward.txt; only a local exercise redirects.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
SCORE_SCRATCH="$(mktemp -d)"
SCORE_DOC="${SCORE_SCRATCH}/score.json"

emit_reward() {
    bound_reward="/logs/verifier/reward.txt"
    bound_score="/logs/verifier/score.json"
    out_dir="${FORGE_REWARD_ROOT:-/logs/verifier}"
    out_reward="${out_dir}/reward.txt"
    out_score="${out_dir}/score.json"
    mkdir -p "${out_dir}" || true
    if [ -s "${SCORE_DOC}" ]; then
        cp "${SCORE_DOC}" "${out_score}"
        python3 -c 'import json,sys;print(float(json.load(open(sys.argv[1]))["reward"]))' \
            "${out_score}" > "${out_reward}"
    else
        printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted", "failed_checker": null, "detail": "the verifier exited before a score document was produced", "metric": {"anchors_state": "absent"}}' > "${out_score}"
        printf '0.0\n' > "${out_reward}"
    fi
    echo "reward carrier ${out_reward} (bound path ${bound_reward}, bound score document ${bound_score})"
    rm -rf "${SCORE_SCRATCH}" || true
}

trap emit_reward EXIT

# OER25_PRODUCER_OUTPUT grades a frozen producer document instead of launching a producer.
# It exists so a rejecting fixture can be driven through this same entry point, inside this
# same image, against the same built environment state. It is never set on a graded run.
if [ -n "${OER25_PRODUCER_OUTPUT:-}" ]; then
    python3 "${TESTS_DIR}/grade.py" \
        --bundle "${BUNDLE_DIR}" \
        --producer-output "${OER25_PRODUCER_OUTPUT}" \
        --out "${SCORE_DOC}"
else
    python3 "${TESTS_DIR}/grade.py" \
        --bundle "${BUNDLE_DIR}" \
        --submission "${BIA_SUBMISSION:-/app/submission.py}" \
        --out "${SCORE_DOC}"
fi

# The compiled suite drives the checkers over the private fixtures. Those fixtures are not
# part of the delivered verifier image, so the suite runs only when they are present, and its
# absence is reported rather than silently skipped.
if [ -d "${BUNDLE_DIR}/solution/fixtures" ] && python3 -c 'import pytest' 2>/dev/null; then
    python3 -m pytest -q "${TESTS_DIR}/test_output.py" || true
else
    echo "compiled suite not run: private fixtures or pytest absent in this image"
fi
