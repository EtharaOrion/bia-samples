# FORGE-CANARY-BEGIN
# 0: 4dfd09094850acbf8d255d697970e6793147a032f7ddbc5dc65caecef2ac5c68
# 1: 7972c2f21da3f28099eee16fe6b2adeba15ac798e78eaf9b0099f1a7fe255d80
# 2: 2a5a40220ccedfd107c01eb48ce1f80ede0003aff3c2a3714e72717dfcd3b9e0
# 3: f70508c6ede009025e02647c96bf1be212822bd158b24bb15978dd4ef34892ee
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
