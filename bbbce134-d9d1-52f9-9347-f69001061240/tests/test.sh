#!/usr/bin/env bash
# Harbor verifier entry point for slot OER-03.
#
# This file is the whole result contract. Harbor ships no result parser of its
# own, so nothing downstream reads anything this script does not write.
#
# The reward is one float on the closed interval [0.0, 1.0], higher is better,
# never binary. Every zero carries a machine-readable reason. The write is the
# last thing that happens, and the EXIT trap below guarantees a reward exists on
# every exit path, including an abort inside the grader itself. A verifier that
# dies quietly is exactly the failure this slot grades, so it is not permitted to
# do it to its own grading path.
#
# Carriers reached from here: grade.py, which imports checkers.py and runner.py,
# reads checkers.yaml, golden.json and rubrics.jsonl, and is compiled against
# test_output.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${BIA_LOGS_DIR:-/logs/verifier}"
ABORT_DOCUMENT='{"reward": 0.0, "reason": "verifier-aborted-before-grading", "metric": {"graded_step": null, "baseline": 3250, "target": 2690, "graded_quantity": "crossing_step"}}'

on_exit() {
  rc=$?
  mkdir -p "$LOGS"
  bound_reward="$LOGS/reward.txt"
  if [ ! -s "$bound_reward" ]; then
    printf '0.0\n' > "$bound_reward"
    printf '%s\n' "$ABORT_DOCUMENT" > "$LOGS/reward.json"
    printf '%s\n' "$ABORT_DOCUMENT" > "$LOGS/score.json"
    printf '%s\n' "wrote an attributed zero to /logs/verifier/reward.txt because grading never reached its own write" >&2
  fi
  return $rc
}

trap on_exit EXIT

cd "$HERE"
mkdir -p "$LOGS"

# The compiled surface first. It is advisory: a failure here must not swallow the
# graded run, so its status is captured and reported rather than allowed to end
# the script before grade.py has written an attributed reward.
python3 -m pytest test_output.py -q --no-header || printf '%s\n' "test_output.py reported failures; the graded reward below is still authoritative" >&2

# The graded run. grade.py writes /logs/verifier/reward.txt, reward.json and
# score.json itself, and its module-level handler writes an attributed zero if it
# raises.
python3 grade.py "$LOGS"
