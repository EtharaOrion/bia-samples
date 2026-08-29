#!/usr/bin/env bash
# Harbor verifier entry point. Terminates by writing the bound reward path
# /logs/verifier/reward.txt, alongside the fuller record at /logs/verifier/record.json,
# so every zero is attributed to a machine-readable reason rather than to an unwritten
# or empty reward file.
set -uo pipefail
cd "$(dirname "$0")"

export BIA_REWARD_PATH="${BIA_REWARD_PATH:-/logs/verifier/reward.txt}"
export BIA_REWARD_RECORD="${BIA_REWARD_RECORD:-/logs/verifier/record.json}"

# grade.py writes the reward on every path it reaches, but it cannot write one on a path
# that never reaches it: an import failure or a kill before it starts leaves the file
# absent, and an absent reward reads as an infrastructure fault rather than as a graded
# zero. This trap makes the floor the terminal state of every exit path, and it never
# overwrites a reward grade.py already wrote.
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

python3 -m pytest test_outcome.py -q --no-header
OUTCOME_RC=$?

if [ "$GRADE_RC" -ne 0 ]; then
  exit "$GRADE_RC"
fi
exit "$OUTCOME_RC"
