#!/usr/bin/env bash
# OER-18 verifier entry point. Harbor ships no result parser, so this file is the whole
# contract: it terminates by writing the bound reward carrier, and an EXIT trap guarantees
# a reward with a machine-readable reason on every exit path, including an abort.
set -euo pipefail

# The bound reward contract. These two paths are the delivered contract; FORGE_REWARD_ROOT
# redirects them to a lane-private root for a local exercise only, because /logs/verifier is
# a single shared host path that concurrent lanes collide on.
BOUND_REWARD_PATH="/logs/verifier/reward.txt"
BOUND_SCORE_PATH="/logs/verifier/score.json"

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_ROOT="${FORGE_REWARD_ROOT:-$(dirname "${BOUND_REWARD_PATH}")}"
REWARD_FILE="${REWARD_ROOT}/$(basename "${BOUND_REWARD_PATH}")"
SCORE_FILE="${REWARD_ROOT}/$(basename "${BOUND_SCORE_PATH}")"
SUBMISSION_DIR="${FORGE_SUBMISSION_DIR:-/workspace/submission}"
WORK_DIR="${FORGE_WORK_DIR:-${REWARD_ROOT}/work}"

export FORGE_REWARD_ROOT="${REWARD_ROOT}"

mkdir -p "${REWARD_ROOT}"

# Every exit path lands here. If the graded write already happened this is a no-op; if the
# verifier aborted before it, this writes the floor with a reason rather than leaving the
# runtime to read silence as an infrastructure fault.
emit_fallback() {
    python3 "${TESTS_DIR}/grade.py" --fallback || true
}
trap emit_fallback EXIT

# The compiled fixture tests. They exercise both halves of every declared checker against
# the golden fixtures solution/recompute.py derived, and they enforce the checker import
# allowlist over the AST of tests/checkers.py. A failure here means the instrument is not
# trustworthy, so the run resolves through the trap rather than producing a graded number.
FORGE_FALLBACK_REASON="verifier-instrument-failed" python3 "${TESTS_DIR}/test_output.py"

# The graded run. grade.py launches the submission through tests/runner.py, kills its
# process group, measures coverage with tests/strata.py, trains and evaluates with
# tests/frozen_stack.py against tests/benchmark.json, runs tests/checkers.py, and writes
# both carriers itself as the last thing it does.
python3 "${TESTS_DIR}/grade.py" --submission "${SUBMISSION_DIR}" --workdir "${WORK_DIR}"

test -s "${REWARD_FILE}"
test -s "${SCORE_FILE}"
