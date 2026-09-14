#!/usr/bin/env bash
# Verifier entry point for slot OER-21. Harbor ships no result parser of its own,
# so this file is the whole reward contract.
#
# It terminates by writing the bound reward carrier, /logs/verifier/reward.txt, a
# bare float on the closed interval [0.0, 1.0], higher better, never binary. The
# machine-readable reason and the metric block go to the companion score document
# /logs/verifier/score.json. An EXIT trap guarantees a reward on every exit path,
# including an abort, so an aborted verifier produces a reason rather than silence.
#
# Reachable grading surface, in the order it is entered:
#   grade.py       the ordered gate chain and the reward write
#   runner.py      launches the submission as its own session leader, kills the group
#   evaluate.py    the harness-owned quantizer, forward pass and telemetry record
#   checkers.py    the eight pure checkers declared in checkers.yaml
#   anchors.json   the bound digests and the measured scaling endpoints
#   test_output.py the compiled tests, one per declared checker
#   rubrics.jsonl  the trajectory rubric, judged outside this process
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${TESTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# THE SUBMISSION THIS VERIFIER GRADES, named in this entry point's own frozen bytes.
# grade.py reads FORGE_SUBMISSION and runner.py launches whatever it names as its own
# session leader; solution/solve.sh names the SAME variable on the oracle side, so one
# name carries the join and both halves of it are readable without running either file.
# Naming it only inside grade.py left the strongest statement this bundle makes about
# its submission buried among the dict lookups of tests/checkers.py, and an instrument
# reading these bytes for an environment variable took the first hinted `.get()` it
# found, which was a telemetry key rather than a variable anything exports.
#
# The default is a DECLARATION OF SHAPE, one Python file at a named path, and not a
# promise that the file is inside this image. solution/reference.py is deliberately not
# baked into the verifier image, so a run that delivered nothing resolves this default
# to an absent path and tests/runner.py records submission-absent instead of grading the
# reference's own work and calling it the delivery's.
BUNDLE_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
FORGE_SUBMISSION="${FORGE_SUBMISSION:-${BUNDLE_DIR}/solution/reference.py}"
export FORGE_SUBMISSION

emit_reward() {
  local status=$?
  local bound_reward_path="/logs/verifier/reward.txt"
  local bound_score_path="/logs/verifier/score.json"
  local reward_root="${FORGE_REWARD_ROOT:-$(dirname "${bound_reward_path}")}"
  local reward_file="${reward_root}/$(basename "${bound_reward_path}")"
  local score_file="${reward_root}/$(basename "${bound_score_path}")"
  mkdir -p "${reward_root}"
  if [ ! -s "${reward_file}" ]; then
    printf '0.0\n' > "${reward_file}"
    printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-scoring", "failed_checker": null, "observed": "the verifier exited before the gate chain produced a score", "metric": {"graded_metric": null}}' > "${score_file}"
  fi
  return "${status}"
}
trap emit_reward EXIT

if python3 -c 'import pytest' >/dev/null 2>&1; then
  python3 -m pytest -q "${TESTS_DIR}/test_output.py" || true
fi

python3 "${TESTS_DIR}/grade.py"
