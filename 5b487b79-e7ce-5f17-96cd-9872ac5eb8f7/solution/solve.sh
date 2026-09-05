# FORGE-CANARY-BEGIN
# 0: 9965e03df76a67b0507e9256ed2c2f08719005b3dcb94c0b4ba822b7395a4df4
# 1: 2158c3653bef960d331f1b4a41518710b632fd95397d435a88792dd6050754db
# 2: 3c96ca46914d07666102039389b2f936bfce0888431aecdafb7f6ca0f0ca0984
# 3: 9fe0b4d8d6aaf3a935e4f4c3e503ef24c541f7ee12afc0b5cac6a980cd8614fd
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It reads the sealed provenance store out of BUILT ENVIRONMENT
# STATE through the read-only handle, walks the seal chain from genesis to recover the
# realised attestation order, folds every atom digest in that order, solves the graded
# instance exactly, and writes submission.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${OER30_WORKSPACE:-${PWD}/workspace}"
STORE="${OER30_STORE:-/task/state/store}"
INSTANCES="${OER30_INSTANCES:-/task/environment/instances.json}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --store "${STORE}" \
    --instances "${INSTANCES}" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
