# FORGE-CANARY-BEGIN
# 0: caedec685e1c24c8d9e17b74df74940f1bc0a041a3f3464954c0611b04480a8e
# 1: ed781f9c579e839932627bcf2aeff84a6e584575c5241a5f34b77de7b02c7906
# 2: b4469472f016cdb25a999ca6e651f22dc1d0fd37359dcf8cc3fb7ec5ee9a746a
# 3: e38f445dfccdfc366f21eaeb01846d1edfcbc0c2f9bd5361c5f35b0e635b7522
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
