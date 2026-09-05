# FORGE-CANARY-BEGIN
# 0: 2bfa5d5ad429415d0614e8ab7ba9e8af09fa63db2edebe885ac158ef7791e341
# 1: 4a7991e7ee10002ec97a6e8a02227259829dba19ec323b6ea9c72098ed4df433
# 2: 1843f874d46b37ce66533c5412ecb18fa6e0ec0e2610cb6d547fff26410ec849
# 3: a4adabb284efc562af914419c7daea1e2ee843364e8173a3a6a315d034ab16ba
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
