# FORGE-CANARY-BEGIN
# 0: a98a2ca048bc463d15659a02a37aa232ffa56f5f5a5a2943103c8c03c294c804
# 1: cb5676e8cde35043c726e9bc9eee561cd3026b77bf3019e562b3d63989519c68
# 2: 836472f57774d1bb4b40eb47f35b4f756fbb0f00834a8941ca3532ad44953114
# 3: 507ff872b109870f12abc6f87c9ccc609365ae8bdf1265f0618bc2e761760f0e
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It reads the adjudication threshold back out of the BUILT
# index through the query handle, recomputes the exact similarity of every unordered
# pair of distinct records, closes the thresholded edges into duplicate groups, derives
# the maximal collision the grouping refused, and writes submission.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
WORKSPACE="${OER27_WORKSPACE:-${PWD}/workspace}"
INDEX="${OER27_INDEX:-/task/index}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --index "${INDEX}" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
