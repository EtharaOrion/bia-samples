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
