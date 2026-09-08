# FORGE-CANARY-BEGIN
# 0: b94458bd15f72370ea97b912af962f251377752289a141d3788471da0a4c7c9d
# 1: 4aceb825f86a796dbba8d94e50d481644520be553dbf1828de48058cf53fab14
# 2: 9bd08e404169a97c2b45f7b1ad4878f2d62f67aab463f9aae5b219b20a53ef82
# 3: fa0d6e73ea6cf782b71458acdf1837bbef60fb21f05d5455a6bd5e631f3bd294
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
WORKSPACE="${OER30_WORKSPACE:-/workspace}"
STORE="${OER30_STORE:-/task/state/store}"
INSTANCES="${OER30_INSTANCES:-/task/environment/instances.json}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --store "${STORE}" \
    --instances "${INSTANCES}" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
