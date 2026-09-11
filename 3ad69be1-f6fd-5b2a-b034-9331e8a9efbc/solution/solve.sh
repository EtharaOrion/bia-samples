# FORGE-CANARY-BEGIN
# 0: d542faf44de3f7c9873b085ed418a5d4b90b945f5fbf7d2a648ee2966eac8361
# 1: 1b128de0d0eca219afd6f719910af83db01495527d610e2327feb6433712bd32
# 2: 3102462e7238c2f554979fc8bde65cbf671c5f952366b1b7d0670d90ab1984b0
# 3: c4551c0756219894157c7c6db64a9e3830fe236e253d17753c7c137168b387c4
# FORGE-CANARY-END
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
WORKSPACE="${OER29_WORKSPACE:-/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --warehouse "${BUNDLE}/environment/warehouse.db" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
