#!/usr/bin/env bash
# OER-08 verifier entry point. Harbor ships no result parser of its own, so this
# file is the whole contract.
#
# Surface this entry point reaches, named so reachability is readable rather than
# inferred: runner.py, checkers.py, grade.py, checkers.yaml, rubrics.jsonl and
# test_output.py all live under tests/ and are reached from here.
#
# The last thing that happens on every exit path, including failure, is that
# /logs/verifier/reward.txt receives one bare float on the closed interval
# [0.0, 1.0] and /logs/verifier/score.json receives the machine-readable reason
# and the metric block. Both are written from the one EXIT trap below, so an
# aborted verifier produces a reason rather than silence.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
WORK_DIR="${OER08_WORK_DIR:-/tmp/oer08-verifier}"
EVIDENCE_DIR="${OER08_EVIDENCE_DIR:-${WORK_DIR}/harness}"
VERDICT_PATH="${WORK_DIR}/verdict.json"
SUBMISSION_PATH="${OER08_SUBMISSION:-/workspace/submission.py}"
PLAN_PATH="${OER08_PLAN:-/workspace/plan.json}"

emit_reward() {
  rc=$?
  mkdir -p /logs/verifier
  python3 "${TESTS_DIR}/grade.py" --emit-only --verdict "${VERDICT_PATH}" --reward /logs/verifier/reward.txt --score /logs/verifier/score.json --exit-code "${rc}" || true
  return 0
}
trap emit_reward EXIT

mkdir -p "${WORK_DIR}" "${EVIDENCE_DIR}"

# grade.py owns the gate chain. It imports tests/checkers.py and tests/runner.py
# and never imports the submission; runner.py launches the submission in a fresh
# session and kills its process group in a finally block.
if [ -f "${SUBMISSION_PATH}" ]; then
  python3 "${TESTS_DIR}/grade.py" --submission "${SUBMISSION_PATH}" --evidence "${EVIDENCE_DIR}" --verdict "${VERDICT_PATH}"
else
  python3 "${TESTS_DIR}/grade.py" --plan "${PLAN_PATH}" --evidence "${EVIDENCE_DIR}" --verdict "${VERDICT_PATH}"
fi

# The compiled surface of tests/checkers.yaml. It is generated from
# solution/grounding.yaml by solution/recompute.py and re-asserts every graded
# row against the same live handles. It never enters the reward.
python3 "${TESTS_DIR}/test_output.py" --evidence "${EVIDENCE_DIR}" || true

exit 0
