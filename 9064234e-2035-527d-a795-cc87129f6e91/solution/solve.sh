# FORGE-CANARY-BEGIN
# 0: 72d0a083b609bb0a6030bd9ce4ecf74bf5c701ec846668a633b2cee772fb5521
# 1: dc2e411da78f83cfdfe953f7d5b92eda1510c1d1f0c83ee69d5b5dfea0b0d1cb
# 2: 1df19e305caa64c82845ba59dc5a7366beeb7efe8a3483005398ff93801dd4d4
# 3: 1978373e431a71e11622f920ce3fe06efdd2be6f10bca3693170d7edf67f0271
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
