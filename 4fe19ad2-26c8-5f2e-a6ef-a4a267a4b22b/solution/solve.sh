# FORGE-CANARY-BEGIN
# 0: 6a89203d1ee2b1048949dbe6f4477c044f2ac871329b1aafb1f7379170ab04f4
# 1: 032249266941761b21c740398ae47099d8cd2274731e0c591bb4b8f78a29935f
# 2: 7c29038aed421f45dd19f4571dbb062d973e527518bb2eddbf53f9d4df00ad54
# 3: a2fad4cb549723c8be708bfeea6c8be4ca220dc09f6f3074c35f1af447ea2980
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
