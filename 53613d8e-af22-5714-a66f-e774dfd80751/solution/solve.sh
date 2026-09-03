# FORGE-CANARY-BEGIN
# 0: 3ac49f1acaddc6904c573d7cb0b2ee496db56026a18de152f46486f9e7b14367
# 1: 129125c259cefba6e6bfdc5e7e93cd2fb8c1ebbede06cbb36060958604d4268d
# 2: 0bd16d4d697517ad4a4034bfae15d7d964eb1c907125c43b7d25f80f652a8146
# 3: e5d19319d5df23094317522d8c971f6ded71e16dcd498845715ac349c4a458b9
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
