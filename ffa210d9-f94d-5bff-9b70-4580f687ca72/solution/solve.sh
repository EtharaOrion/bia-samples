# FORGE-CANARY-BEGIN
# 0: e7a92abce98ba3570891bce50269fec51cc458be3517fc852506f881c03667bd
# 1: 41f20515939fd296e13c43ba8cbdc744815c14280ffcccf5be2d91792d8f47d2
# 2: 8d0e3cb5d67e5b13f99e28fc234b41e3374b4d29f93412fff49a8abf7cb0fef2
# 3: 9c3f962b14a9ef7e605bfdee9175de6cb44a0ab7b9fe0c769e08dd725f012fdf
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
