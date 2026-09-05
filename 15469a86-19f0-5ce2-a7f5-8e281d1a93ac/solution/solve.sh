# FORGE-CANARY-BEGIN
# 0: 42fa2c0179e70644cf60747c49376950a3b03e08df27477eb33719b1bc6ff95b
# 1: 486fb22732a7169491c7aa25fc7fe095018c11beccd41ce2215c2ba8087ab27a
# 2: 1fa9d456a00575ffbfed42996f3d359d0cab7462e97abb4136b5e8b6d30ef6e0
# 3: 155000a5031e29a6448ae06f35d365fc4fdd2e9a54929612f53e8328df4f4498
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
