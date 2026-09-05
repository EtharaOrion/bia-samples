# FORGE-CANARY-BEGIN
# 0: 8be69ca0112976a2ce4be37dd898214b598a5721b2676706ee382a7652bb9558
# 1: c5a53d963dad4e4ad6c6c5941c5f50d66583d4d172f32b9bc340a6c89d5a191d
# 2: 1f5e1cf0e2eef97cbc6a24968e7a243505e436ee2bff90e95036d8fff5ff3f03
# 3: 5497070449a2fc76d54389246297f49642ec6595459a97ad6aba0f40c93a987f
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
