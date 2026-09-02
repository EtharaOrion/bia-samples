# FORGE-CANARY-BEGIN
# 0: be47125bb810af4f75f4af8125b11cca5acf2e94495ba11cc263c19331a9da34
# 1: 6bbf13b5f1facdc1c11d193c49a36263f95a5f144088ac8e0d8a7947d5b12649
# 2: 18a39d7ad37bd9cf739ebc4a50c2a0870248d19501e65a6f70efe2764d44f1bf
# 3: ff60717e5eaafd02b0ff5728384013c5dcc2176b603326eb9a257301ecc0f924
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
