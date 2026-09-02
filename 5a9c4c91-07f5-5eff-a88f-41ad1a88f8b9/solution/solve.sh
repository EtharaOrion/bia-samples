# FORGE-CANARY-BEGIN
# 0: 653cc21d2d1377a984cb320d3badfc02ee82092d5cc644795b59e12c977e4e91
# 1: f3028397321e2e711c1838e5f09d15cd46acd56222e1f2a900ff2518f15215af
# 2: ef6229e993f8e1b636086ba8a1b813fb01fe9322be6a16c2a3a0f5a368993f99
# 3: 72dd52cb651b9e7841ec0f8118b021f258b2ea0e3b961a34a472c94b670fe214
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Derived from solution/grounding.yaml by solution/recompute.py.
#
# The reference solution is a refinement POLICY, not a configuration. The
# harness invokes it once per attempt with state.json beside it, and it
# writes proposal.json. This script only puts it where the harness looks.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${BIA_SUBMISSION:-/workspace/refine.py}"
mkdir -p "$(dirname "${TARGET}")"
cp "${HERE}/reference.py" "${TARGET}"
chmod +x "${TARGET}"
echo "reference policy installed at ${TARGET}"
