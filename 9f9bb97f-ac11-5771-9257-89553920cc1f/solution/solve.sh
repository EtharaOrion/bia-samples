# FORGE-CANARY-BEGIN
# 0: 3eff5e8bf3aae73d764e34906af313189162fe6d2dab1687f456efb5b67f7a8f
# 1: 4a71b216c8fa07c359d21438be2473bdfb98eb63c7613199203f6d93ff44b8e7
# 2: c7460fa1babc388c9767beba6034401f689dd7f362dc34e6a62f10e6ca586a03
# 3: 1cd94fea9b8cada9f080a060b980ca62fabcde0e624b43a66ff6096debbe5fd6
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
WORKSPACE="${OER29_WORKSPACE:-${PWD}/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --warehouse "${BUNDLE}/environment/warehouse.db" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
