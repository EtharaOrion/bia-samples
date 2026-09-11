# FORGE-CANARY-BEGIN
# 0: fbdbf882b1e465854485386614e2c82838a828c55706e12d630b008c98beb60d
# 1: 4d860602457816a225c845b91ad117918896e9a2f170d5784d8fc07f840d15e1
# 2: e1bb29c81d934b907e1f3c2fdfe9d64f15e3be0796c0d368994fa06e896f0172
# 3: 2197ec877350b1871ed61f4575a25b01cc3bbbc2f5549854ede333dfd5c86929
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
