# FORGE-CANARY-BEGIN
# 0: 7ce23dfe50e21e99358c4b1814e9e9f2ff01c60a35fd37c040825bec9f4f6e54
# 1: 12846e157623bf5053df6e1e17ca3d99b2090c48adb7884203ae61e1a10659c8
# 2: 062fdf7dff0ffa8aeec42a7e56bfb001f7c5a98212b4ee8606c1f6039f79d014
# 3: 35272ff2b5ae87407f411ca306b8628c1ceccb035c799401a0ec327551ff5ea8
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
