# FORGE-CANARY-BEGIN
# 0: aec2c873054960f8f522fd0f1b045c97d783faccbd14848c158180de26d04167
# 1: 1a3063b71b32100a3f9ad02caa8e63ef81ddd0c45a2c8649c7cc443fc2a27015
# 2: 57c6ba302e89fe05e6bca4e1158fb808ec0dce029bccfb4ef0d7e40d77bd274a
# 3: 5531153ff8d60e2c18c0d169ca79b1e90f11f06f729beb3115a03388829893d5
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It drives the bound refinement loop over the reference
# exploration schedule, keeps its own durable ledger under notes/, detects that the
# context summary dropped iterations, reconstructs the dropped findings from the
# ledger, and writes submission.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
WORKSPACE="${OER24_WORKSPACE:-${PWD}/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --bundle "${BUNDLE}" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
