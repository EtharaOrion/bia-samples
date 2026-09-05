# FORGE-CANARY-BEGIN
# 0: 50a9d58d06ea453535d168cbc57a88ff836c02cb670390c2568af43f2b981d25
# 1: 5fa0d1f9bd08933eb57ddffd317a9d9a9f3f15b14586f2ed478e214425083cdc
# 2: d6afc03407f82efa9d06b8a3e4e72b1a63fbac65fb626f8d135eae6d72afe110
# 3: 2aeddebf4e919e8dd5b54215ec196ff8bd2cda70788bbaae3eed109bf40ee828
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
