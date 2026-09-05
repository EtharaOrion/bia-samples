# FORGE-CANARY-BEGIN
# 0: 4d84b6483de5473a2c6f3a2024002b897cd65fd501a15f174e3c1a877d01489a
# 1: ffca5b3fee165899971d311908e89d8ade41ea1f97d622489c3fa2125e0b2bce
# 2: 9ace70bdea1566495e2e68aaba8ad97e70e2692f96878cc9b8d89b2cb93bb0e2
# 3: afd52fe961512aa6ab4208a411248dd3501e43c0f75715d006d04b6394a838fa
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The AGENT PHASE entry point of the oracle. It installs the fused kernel at the bound
# submission path, runs the bundle's own agent-phase driver, and transcribes the splice point
# and the carried-state digest the driver established into the handoff. It never invents either
# value: both are read back off the harness journal the driver wrote.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
WORKSPACE="${OER28_WORKSPACE:-/workspace}"
HARNESS_LOGS="${OER28_HARNESS_LOGS:-/logs/harness}"
SUBMISSION="${OER28_SUBMISSION:-/app/submission.py}"

mkdir -p "${WORKSPACE}" "${HARNESS_LOGS}" "$(dirname "${SUBMISSION}")"
python3 "${HERE}/reference.py" \
    --bundle "${BUNDLE}" \
    --workspace "${WORKSPACE}" \
    --journal "${HARNESS_LOGS}/phase_a.jsonl" \
    --submission "${SUBMISSION}"

echo "kernel installed at ${SUBMISSION}"
echo "handoff written to ${WORKSPACE}/handoff.json"
echo "harness journal at ${HARNESS_LOGS}/phase_a.jsonl"
