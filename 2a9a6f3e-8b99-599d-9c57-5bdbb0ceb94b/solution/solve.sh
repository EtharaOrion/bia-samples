# FORGE-CANARY-BEGIN
# 0: 081d096f11220bf5d2546216bc6f2f56e08aab06048590c21ee4d3bd1358b24a
# 1: f551ebd78b9058129eb93c94c2434e3fa22daf3e7f3826ee3b4e5e0fb57f9d2d
# 2: afa8e735a957f4845505d3e2745659e81a2e7958bf49bf38db34b1757df7284c
# 3: f2b51e9927a3968d6e84fe58a994bdb5a68a2dd362c2fd8be36b30ab1c502af5
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
