# FORGE-CANARY-BEGIN
# 0: 5da5adc16e2afe2d484194e000cd4f1ed7893c800c66aae40b16828700a62723
# 1: 9056413a58e48ff5f09df454f2c4e9818fd0d2d637bff2d6a5eb5208eef54e55
# 2: 7a72b5074e56dfb1e790fb909d95bde8a3fe4734696a3b97b8e0ee6e2028af0e
# 3: 21aec299c5654711767665a0252883456147bf57d191e58276d1cc3aee67c6bf
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
