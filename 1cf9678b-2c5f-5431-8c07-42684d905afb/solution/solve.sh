# FORGE-CANARY-BEGIN
# 0: e7e8efed7e49584a3e949e540a48cbb0c909296af7193e20a84a76219d7e0926
# 1: e5cbc28137ef4c58658c356b20c717969e3eb6e97a4849f4ff6433863cde3b23
# 2: 81bd6e5592251401f5fc25d9c592f4bcfd86e8d7af57b5e89c0b3d91e8adb0cc
# 3: ce26a93bd3b98bec3644dfc9bad426976314dcdff8d7264f4f3c4b27ac91de07
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
