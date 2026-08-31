#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It runs the oracle, which reads the built warehouse back through the
# harness handle, applies the exclusion rule, walks the closure to fixpoint, folds the closure
# into a plan and writes plan.sql and submission.json into the workspace.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
WORKSPACE="${OER29_WORKSPACE:-${PWD}/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --warehouse "${BUNDLE}/environment/warehouse.db" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
