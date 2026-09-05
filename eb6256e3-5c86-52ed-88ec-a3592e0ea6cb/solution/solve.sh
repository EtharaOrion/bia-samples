# FORGE-CANARY-BEGIN
# 0: 3dcb1628df5aa02e52a230e10d594a14bcc6f567fa17ce8e6ee7c53a277b69e5
# 1: fc05d94b15553a0340441d8b0ff5e15fd23b47e3b12c074e30a5be3dab38b6f8
# 2: ba2745e66f3fbefb415cfe880a934458f568b917a7248513bd5e9895a6dc8801
# 3: 8a42385c298c84500d2cbf7cfd1c1bd7295db85fe5503b9aef4068301c170592
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
