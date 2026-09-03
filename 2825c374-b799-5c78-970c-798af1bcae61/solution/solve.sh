# FORGE-CANARY-BEGIN
# 0: 30a239ceb4a726f8535a28fc22dd68d21e3f90e3c19b8e31170bb7694c41b68e
# 1: 7d254b67a9d28c7fdf9fc46831bdb57bc86f6826d9f882f00236e2161e42096d
# 2: f53faf5633908a76281ff86dd8e81456fa233300119f6b1b7af978e31f5ae012
# 3: b0c51f44a8a51bdb06ee5fabbaf902d66c1ec8a0d41ac643dd7c089e49e1cba0
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
