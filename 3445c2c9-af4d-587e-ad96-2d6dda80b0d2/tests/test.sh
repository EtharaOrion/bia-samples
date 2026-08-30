#!/usr/bin/env bash
# Harbor verifier entry point. Terminates by writing the bound reward path.
#
# Harbor ships no result parser of its own, so this file is the whole contract:
#   /logs/verifier/reward.txt   one float on the closed interval [0.0, 1.0]
#   /logs/verifier/reward.json  the reward document, with reason and metric
#   /logs/verifier/score.json   the per-checker score document
#
# The float is higher-better and never binary. Every zero carries a
# machine-readable reason. The EXIT trap below guarantees that EVERY exit path
# writes a reward, including a crash under `set -e` and including a signal, so an
# aborted verifier produces an attributed zero rather than silence.
set -euo pipefail
cd "$(dirname "$0")"

REWARD_DIR="${BIA_REWARD_DIR:-/logs/verifier}"

emit_floor() {
    # Runs on every exit. If grade.py already wrote /logs/verifier/reward.txt
    # this leaves it alone; otherwise the abort becomes an attributed zero.
    # writes /logs/verifier/reward.txt, /logs/verifier/reward.json and /logs/verifier/score.json
    local status=$?
    if [ ! -f "${REWARD_DIR}/reward.txt" ]; then
        mkdir -p "${REWARD_DIR}"
        printf '0.000000\n' > "${REWARD_DIR}/reward.txt"
        printf '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "metric": {"graded_step": null, "baseline": 3250, "target": 2690}}\n' > "${REWARD_DIR}/reward.json"
        printf '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "checkers": []}\n' > "${REWARD_DIR}/score.json"
    fi
    exit "${status}"
}
trap emit_floor EXIT

# Static surface first. It never scores; it reports.
python3 -m pytest test_static.py -q --no-header || true

# The whole refinement session, driven by the verifier, then the gate chain.
python3 grade.py

# Both halves of every checker, compiled from solution/grounding.yaml. A zero
# collected count is a failure here, never a pass: an empty parse is ambiguous
# between a clean run and no run at all.
python3 -m pytest test_output.py -q --no-header
