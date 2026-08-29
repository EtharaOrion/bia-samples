# FORGE-CANARY-BEGIN
# 0: 95cd8a711af133a69a5fb9411506cec4aa7ea4869c02cbdd489b1b797081b0c5
# 1: e74fb2180a82668242e44ae8537dbef244aa2f914eb80a28664779c6d8baf486
# 2: 5aaeb8b146d50e3888e26664ef88d1f03a4d8426c3fceae1037a115061bbb777
# 3: 9526d7bb1fcdce12796486d8a723b8d925c015cf2ed310599a5ea45d4fe388a7
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
