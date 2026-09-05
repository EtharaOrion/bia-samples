#!/usr/bin/env bash
#
# OER-11 verifier entry point. Harbor ships no result parser of its own, so this file
# is the whole contract.
#
# It terminates by writing the bound reward carrier /logs/verifier/reward.txt, one bare
# float in the closed interval [0.0, 1.0], higher better, never binary, and the
# companion score document /logs/verifier/score.json carrying the machine-readable
# reason and the metric block. BOTH are written from the SAME trap, so the instrument's
# binding and the reason carrier can never disagree.
#
# The trap is registered before anything can fail. Under `set -e` any failing command
# ends the script, so without the trap an aborted verifier would leave the reward
# absent and the run would read as an infrastructure fault rather than as a score with
# a reason. Every exit path -- success, failure, signal -- writes a reward, and every
# zero carries a machine-readable reason.

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_LOGS="${OER11_HARNESS_LOGS:-/logs/harness}"
VERDICT="${TMPDIR:-/tmp}/oer11-verdict.json"

STAGE="startup"

emit_reward() {
    mkdir -p /logs/verifier || true
    python3 "${TESTS_DIR}/emit_reward.py" "${VERDICT}" "${STAGE}" "/logs/verifier/reward.txt" "/logs/verifier/score.json" || true
    printf 'OER-11 wrote /logs/verifier/reward.txt and /logs/verifier/score.json\n' >&2
}

trap 'emit_reward' EXIT

STAGE="isolate-submission"
# The submission is run OUT OF PROCESS by runner.py, which copies it alone into a fresh
# temporary directory, launches it as a new session leader under a small environment
# allowlist, and kills the whole process group in a finally block. The grading process
# never imports it, and nothing it prints reaches a graded path.
if [ -n "${OER11_SUBMISSION:-}" ]; then
    python3 "${TESTS_DIR}/runner.py" "${OER11_SUBMISSION}" >&2 || true
fi

STAGE="compile-checkers"
# test_output.py is the compiled per-checker suite generated from solution/grounding.yaml
# by solution/recompute.py. Compiling it here proves the carriers are syntactically
# sound before any grading decision is taken.
python3 -m py_compile "${TESTS_DIR}/checkers.py" "${TESTS_DIR}/grade.py" "${TESTS_DIR}/test_output.py"

STAGE="compiled-suite"
# Run the compiled per-checker suite. It is the both-halves witness for every checker:
# each test drives its checker over an accepting fixture and over a planted-defect
# fixture and asserts the exact zero_reason.
python3 "${TESTS_DIR}/test_output.py" >&2

STAGE="rubrics"
# rubrics.jsonl is trajectory-judged in bucket N and never raises an outcome. It is
# validated here as well-formed and is read by no grading decision below.
python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]; assert len(rows) >= 6; assert all(set(r)=={'id','rubric'} for r in rows)" "${TESTS_DIR}/rubrics.jsonl"

STAGE="grade"
# grade.py imports checkers.py and reads only harness-owned records under HARNESS_LOGS.
# It writes the verdict document; it does NOT write either reward carrier. The trap
# owns those, and the trap is the last thing that happens.
python3 "${TESTS_DIR}/grade.py" "${HARNESS_LOGS}" "${VERDICT}" >&2

STAGE="complete"
exit 0
