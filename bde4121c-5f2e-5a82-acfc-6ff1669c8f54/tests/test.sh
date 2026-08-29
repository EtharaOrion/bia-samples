#!/usr/bin/env bash
# Harbor verifier entry point. Terminates by writing the bound reward path
# /logs/verifier/reward.txt, alongside the fuller record at /logs/verifier/record.json,
# so every zero is attributed to a machine-readable reason rather than to an unwritten
# or empty reward file.
#
# grade.py carries a module-level handler that writes a reward file with reason
# grader-internal-error on any unhandled exception, so a crash is a machine-readable
# zero rather than an unwritten artifact. The prior version of this bundle made the same
# claim in this header and did not hold it: its grade.py had no handler, and its sibling
# slot crashed with no reward file at all for any submission good enough to reach a
# missing import.
set -uo pipefail
cd "$(dirname "$0")"

export BIA_REWARD_PATH="${BIA_REWARD_PATH:-/logs/verifier/reward.txt}"
export BIA_REWARD_RECORD="${BIA_REWARD_RECORD:-/logs/verifier/record.json}"

# The handler above covers every path that reaches grade.py. It cannot cover a path that
# never reaches it, so this trap makes the floor the terminal state of every exit path.
# It never overwrites a reward grade.py already wrote.
emit_reward_floor() {
  reward_path="${BIA_REWARD_PATH:-/logs/verifier/reward.txt}"
  record_path="${BIA_REWARD_RECORD:-/logs/verifier/record.json}"
  mkdir -p "$(dirname "$reward_path")" "$(dirname "$record_path")" 2>/dev/null
  if [ ! -s "$reward_path" ]; then
    printf '%s\n' '0.000000' > "$reward_path" 2>/dev/null
    test -s "$record_path" || printf '%s\n' \
      '{"reward":0.0,"pass":0,"reason":"verifier-produced-no-reward","checkers":{}}' \
      > "$record_path" 2>/dev/null
  fi
}
trap emit_reward_floor EXIT

python3 -m pytest test_static.py -q --no-header || true

python3 grade.py
GRADE_RC=$?

python3 -m pytest test_output.py -q --no-header
OUTPUT_RC=$?

if [ "$GRADE_RC" -ne 0 ]; then
  exit "$GRADE_RC"
fi
exit "$OUTPUT_RC"
