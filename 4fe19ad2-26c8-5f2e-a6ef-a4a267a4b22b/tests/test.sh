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
    # Traps the fatal signals too: bash runs no EXIT trap for an UNTRAPPED
    # SIGTERM, which is when the carrier is most owed. Clears errexit first,
    # since under it a failing mkdir aborts this handler before any printf and
    # the abort becomes silence.
    #
    # The three carriers are ATOMIC, because reward.py writes all three or none.
    # So a complete set is left alone and an INCOMPLETE set is completed as a
    # whole rather than per file: patching only the absent file would leave a
    # stale float beside a floor score document, which is the same disagreement
    # in the other direction.
    #
    # The status arrives as an ARGUMENT rather than being read from $? here.
    # When a signal interrupts bash while it waits on a child, $? at handler
    # entry is the status of the last COMPLETED command, which is routinely 0.
    # Reading it here would report a killed verifier as a clean exit 0 carrying
    # "aborted", so the signal traps pass 128+signum explicitly and the exit
    # status stays truthful for the runtime that reads it.
    local status="$1"
    set +e
    trap - EXIT HUP INT TERM
    if [ -f "${REWARD_DIR}/reward.txt" ] && [ -f "${REWARD_DIR}/reward.json" ] \
       && [ -f "${REWARD_DIR}/score.json" ]; then
        exit "${status}"
    fi
    local reason="verifier-aborted-before-grading"
    if [ "${status}" -gt 128 ]; then
        reason="verifier-signalled-before-grading"
    fi
    local metric
    # baseline and target are null here and that is the point: both endpoints of
    # the reward scale are MEASURED by tests/anchors.py during grading, so a
    # verifier that aborted BEFORE grading has no endpoints to report. The
    # literals 3250 and 2690 used to sit here, which meant an aborted run
    # published a scale nothing on this substrate had ever produced.
    metric='{"graded_step": null, "baseline": null, "target": null, "anchors_state": "unmeasured-verifier-aborted", "exit_status": '"${status}"'}'
    mkdir -p "${REWARD_DIR}" 2>/dev/null
    printf '{"reward": 0.0, "reason": "%s", "metric": %s}\n' "${reason}" "${metric}" > "${REWARD_DIR}/reward.json"
    printf '{"reward": 0.0, "reason": "%s", "metric": %s, "checkers": []}\n' "${reason}" "${metric}" > "${REWARD_DIR}/score.json"
    printf '0.000000\n' > "${REWARD_DIR}/reward.txt"
    exit "${status}"
}
trap 'emit_floor "$?"' EXIT
trap 'emit_floor 129' HUP
trap 'emit_floor 130' INT
trap 'emit_floor 143' TERM

# The assertion modules are run through `run_assertions.py`, which this surface
# carries, rather than through `python3 -m pytest`, which the pinned image does
# not carry and could not install behind a denied egress. The runner collects the
# same module-level `test_` functions pytest collected and keeps the same exit
# contract: non-zero on a failing assertion, non-zero on a zero collected count.

# Static surface first. It never scores; it reports.
python3 run_assertions.py test_static.py || true

# The whole refinement session, driven by the verifier, then the gate chain.
python3 grade.py

# Both halves of every checker, compiled from solution/grounding.yaml. A zero
# collected count is a failure here, never a pass: an empty parse is ambiguous
# between a clean run and no run at all.
python3 run_assertions.py test_output.py
