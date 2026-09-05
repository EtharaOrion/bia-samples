# FORGE-CANARY-BEGIN
# 0: 446b22e07cb5a725afb9623881a2b91d429f6e99cffab84b5c310cf5b44ba440
# 1: 293e0bc22ff81af08a5201c3fb61c60d6f03794152284d4987307ed48e653b74
# 2: f834d8d8ad000b5a02c3b237857f8a1e78625f2a9ee7a82b51b6d6f437be5138
# 3: 0cb7767fafc11318e8ce42091c3b1affdff125b9425920edc7e221aada0ea12b
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
