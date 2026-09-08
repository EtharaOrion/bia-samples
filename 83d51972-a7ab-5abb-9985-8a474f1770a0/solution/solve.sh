# FORGE-CANARY-BEGIN
# 0: fc9d0e7805498849fa289a1921ccf89dcbd0717216eacc8dc9cd1c19298dfa84
# 1: 34463e5bce776821e60ff583948d7f1ca4e45ebf3d724064c9f3869942c8586e
# 2: 1c82434379810960b0882a51bdda643d1fd4a395572b8775fe822cfdea101f33
# 3: 0758ef008eaba9f0bfe9e36422f61d8e47a52b00df55de9a44c716e6679c262b
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
WORKSPACE="${OER27_WORKSPACE:-/workspace}"
INDEX="${OER27_INDEX:-/task/index}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --index "${INDEX}" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
