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
