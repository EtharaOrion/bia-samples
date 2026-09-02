# FORGE-CANARY-BEGIN
# 0: 8de261ac7817c01e0f30b661930d08a4f1e2f082965601f44049eb4b586c3e50
# 1: 98ba8b603d095119ba90540a65960842ddda5ee14a4f92f2ff480c18b865227d
# 2: 7c3b009e8ed8927f75d0f6a3581ad029f18b721a190c2c5c66abc1e4412346fc
# 3: 76bc8002e0dc9b1ab91515f921598e1c3ddadf0a2a84525804a23a88f7cfcc7c
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
